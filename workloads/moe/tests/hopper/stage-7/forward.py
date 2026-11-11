#!/usr/bin/env python3
"""Run the frozen forwards in stage-7 order, stopping at a numerical failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MOE = ROOT / 'workloads/moe'
SIM = ROOT / 'gpu-simulator/gpgpu-sim'
ORDER = ['qwen64-t128', 'qwen64-t4096', 'phi-t128', 'mixtral-t128',
         'phi-t4096', 'mixtral-t4096', 'qwen64-t8192', 'phi-t8192',
         'mixtral-t8192']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--cases', nargs='+', choices=ORDER, default=ORDER)
    p.add_argument('--sim-events', action='store_true')
    a = p.parse_args()
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ('PTX_SIM_USE_PTX_FILE', 'PTX_SIM_KERNELFILE',
                'CUOBJDUMP_SIM_FILE', 'MOE_TRACE_LAUNCHES'):
        env.pop(key, None)
    env['PTX_SIM_MODE_FUNC'] = '0'
    libraries = subprocess.check_output(['ldd', MOE/'artifacts/moe_layer'],
                                        env=env, text=True)
    library = Path(next(line.split('=>')[1].split('(')[0].strip()
                        for line in libraries.splitlines() if 'libcudart' in line))
    assert 'gpgpu-sim' in str(library)
    results = []
    for name in (name for name in ORDER if name in a.cases):
        case, tokens = name.split('-t')
        diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=SIM)
        (run/(name+'.inner.diff')).write_bytes(diff)
        build = dict(
            outer=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            inner=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SIM, text=True).strip(),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            source_diff_sha256=hashlib.sha256(diff).hexdigest(),
            configuration_sha256=hashlib.sha256((SIM/'configs/tested-cfgs/SM90_H100/gpgpusim.config').read_bytes()).hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            libraries=libraries, ptx_mode=0)
        (run/(name+'.build.json')).write_text(json.dumps(build, indent=2)+'\n')
        commands = [
            ['/usr/bin/time', '-v', '-o', str(run/(name+'.time')),
             sys.executable, str(MOE/'run.py'), '--case', case, '--tokens', tokens,
             '--mode', 'sim', '--warmup', '0', '--iterations', '1', '--repeats', '1',
             '--run-dir', str(run/name)],
            ['/usr/bin/time', '-v', '-o', str(run/(name+'.compare.time')),
             sys.executable, str(MOE/'compare.py'), str(MOE/'artifacts/cases'/name),
             str(run/name)],
            [sys.executable, str(HERE/'summarize.py'), str(run/name)],
        ]
        if a.sim_events:
            commands[0].append('--sim-events')
        row = dict(case=name, commands=commands, state='running')
        results.append(row)
        (run/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        with (run/(name+'.driver.log')).open('w') as log:
            for step, command in enumerate(commands):
                result = subprocess.run(command, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                if result.returncode:
                    if row['state'] != 'failed':
                        row.update(state='failed', returncode=result.returncode,
                                   failed_command=command)
                    # A numerical failure still needs the completed kernel
                    # summary. A failed simulation may have an incomplete log.
                    if step == 0:
                        break
            else:
                if row['state'] != 'failed':
                    row.update(state='numerical_pass')
        (run/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(name, row['state'], flush=True)
        if row['state'] == 'failed':
            raise SystemExit(row['returncode'])


if __name__ == '__main__':
    main()
