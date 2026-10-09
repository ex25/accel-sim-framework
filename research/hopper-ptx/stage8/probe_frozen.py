#!/usr/bin/env python3
"""Record the first strict frontend failure without rewriting frozen PTX."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root, run = map(Path, sys.argv[1:])
ptx = root / 'research/hopper-ptx/stage1/artifacts/frozen_fwd.ptx'
original = ptx.read_bytes()
kernel = re.search(r'\.entry\s+(\w+)', original.decode()).group(1)
log_path = run / 'frozen-fa3-parser.log'
with log_path.open('w') as log:
    try:
        result = subprocess.run([str(root / 'research/hopper-ptx/stage3/frontend_probe'),
                                 str(ptx), kernel], cwd=run / 'detailed',
                                env={**os.environ, 'LD_LIBRARY_PATH': '/tmp/hopper-stage3-build'},
                                stdout=log, stderr=subprocess.STDOUT, timeout=45)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
text = log_path.read_text()
frontend_pass = re.search(r'^FRONTEND status=PASS target=sm_90a .*instructions=(\d+)', text, re.M)
if code == 0 and not frontend_pass:
    code = 1
match = re.search(r'^([^\n]+\.ptx:\d+ Syntax error:)\s*\n\s*([^\n]+)', text, re.M)
semantic = re.search(r'^([^\n]+\.ptx:\d+: Parse error: [^\n]+)', text, re.M)
first = (match.group(1) + ' ' + match.group(2).strip()) if match else (
    semantic.group(1) if semantic else (
        'frontend timeout' if code == 124 else 'strict frontend failure; see complete log'))
if ptx.read_bytes() != original:
    raise ValueError('frozen input changed during probe')
data = {'status': 'PASS' if code == 0 else 'BLOCKED', 'exit': code,
        'scope': 'frozen_fa3_parser', 'kernel': kernel,
        'ptx_sha256': hashlib.sha256(original).hexdigest(),
        'runtime_sha256': hashlib.sha256(Path('/tmp/hopper-stage3-build/libcudart.so').read_bytes()).hexdigest(),
        'parser_only': True,
        'instructions_including_labels': int(frontend_pass.group(1)) if frontend_pass else None,
        'first_failure': None if code == 0 else first,
        'failure_category': None if code == 0 else (
            'lexer/parser syntax' if match else 'operand/type/modifier' if semantic else 'frontend'),
        'log': str(log_path)}
(run / 'frozen-readiness.json').write_text(json.dumps(data, indent=2) + '\n')
print(f'FROZEN_FA3_RESULT status={data["status"]} exit={code} first={data["first_failure"]}')
