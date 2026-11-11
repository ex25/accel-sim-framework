#!/usr/bin/env python3
"""Run the three 128-token forwards concurrently and collect results on exit."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CASES = ['qwen64-t128', 'phi-t128', 'mixtral-t128']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    manifest = json.loads((run/'batch.json').read_text())
    workers = []
    opened = []
    manifest.update(state='running', controller_pid=os.getpid(),
                    started_unix=time.time(), workers=workers)
    processes = []
    for case in CASES:
        directory = run/case.split('-t')[0]
        command = [sys.executable, str(HERE/'forward.py'), '--run-dir',
                   str(directory), '--cases', case, '--sim-events']
        log = (run/(case+'.worker.log')).open('w')
        opened.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        workers.append(dict(case=case, pid=process.pid, command=command,
                            state='running', run=str(directory/case)))
        processes.append(process)
    (run/'batch.json').write_text(json.dumps(manifest, indent=2)+'\n')
    summaries = []
    for process, row, log in zip(processes, workers, opened):
        row['returncode'] = process.wait()
        directory = Path(row['run'])
        if (directory/'stage7-summary.json').exists():
            row['independent_check_returncode'] = subprocess.run(
                [sys.executable, str(HERE/'acceptance.py'), str(directory)],
                stdout=log, stderr=subprocess.STDOUT).returncode
            summaries.append(json.loads((directory/'stage7-summary.json').read_text()))
        row['state'] = ('passed' if row['returncode'] == 0 and
                        row.get('independent_check_returncode') == 0 else 'failed')
        (run/'batch.json').write_text(json.dumps(manifest, indent=2)+'\n')
        log.close()
    tables = [Path(row['run'])/'kernel-times.csv' for row in workers
              if (Path(row['run'])/'kernel-times.csv').exists()]
    with (run/'kernel-times.csv').open('w', newline='') as output:
        for index, table in enumerate(tables):
            lines = table.read_text().splitlines()
            output.write('\n'.join(lines if index == 0 else lines[1:])+'\n')
    fields = ['case', 'simulation_end_to_end_ms',
              'hardware_first_forward_cuda_event_ms', 'event_vs_hardware_first_percent',
              'hardware_warm_cuda_event_ms', 'event_vs_hardware_warm_percent',
              'simulation_event_elapsed_cycles', 'host_forward_wall_seconds']
    with (run/'forward-times.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary['case'] if key == 'case' else
                             summary['forward_timing'].get(key) for key in fields})
    manifest.update(state='completed' if all(row['state'] == 'passed'
                                            for row in workers) else 'failed',
                    finished_unix=time.time(), summaries=summaries)
    text = json.dumps(manifest, indent=2, ensure_ascii=False)+'\n'
    (run/'batch.json').write_text(text)
    (run/'final-results.json').write_text(text)


if __name__ == '__main__':
    main()
