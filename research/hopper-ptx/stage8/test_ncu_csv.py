#!/usr/bin/env python3
"""Protect counter readiness from missing values and numeric metadata columns."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_ncu_csv import REQUIRED


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in ('finite', 'empty', 'nan', 'inf', 'negative', 'missing',
                 'ncu_error', 'invalid_second_launch'):
        path = out / (case + '.csv')
        rows = [['ID', 'Metric Name', 'Metric Value', 'Metric Unit']]
        rows += [[17, metric, '123.4', 'byte'] for metric in REQUIRED]
        if case in ('empty', 'nan', 'inf', 'negative'):
            rows[1][2] = {'empty': '', 'nan': 'NaN', 'inf': 'inf', 'negative': '-1'}[case]
        elif case == 'missing':
            rows.pop()
        elif case == 'ncu_error':
            rows.append(['==ERROR== replay failure'])
        elif case == 'invalid_second_launch':
            rows.append([18, REQUIRED[0], 'N/A', 'ns'])
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('check_ncu_csv.py')),
                                 str(path), str(out / (case + '.json'))],
                                capture_output=True, text=True, timeout=10)
        if result.returncode != (0 if case == 'finite' else 3):
            raise ValueError(f'{case}: {result.stdout} {result.stderr}')
        summary.append({'case': case, 'status': 'PASS', 'validator_exit': result.returncode})
    (out / 'result.json').write_text(json.dumps({'status': 'PASS', 'cases': summary}, indent=2) + '\n')
    print('CSV_CONTRACT_RESULT status=PASS cases=8')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(Path(sys.argv[1]))
    else:
        with tempfile.TemporaryDirectory(prefix='stage8-ncu-contract-') as directory:
            run(Path(directory))
