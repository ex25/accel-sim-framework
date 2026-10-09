#!/usr/bin/env python3
"""FP64 attention reference and one numerical contract for every backend."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ATOL = RTOL = 2e-3
SHAPE = (1, 128, 8, 128)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(inputs, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs / 'inputs.json').read_text())
    for name, info in manifest['inputs'].items():
        path = inputs / info['file']
        if tuple(info['shape']) != SHAPE or digest(path) != info['sha256']:
            raise ValueError(f'frozen input mismatch: {name}')
    for seq in (64, 128, 256):
        case = out / str(seq)
        case.mkdir(exist_ok=True)
        arrays = []
        for name, offset in zip(('q', 'k', 'v'), (17, 53, 101)):
            if seq == 128:
                raw = (inputs / f'{name}.f16le').read_bytes()
                (case / f'{name}.f16le').write_bytes(raw)
                a = np.frombuffer(raw, dtype='<f2').reshape(seq, 8, 128)
            else:
                i = np.arange(seq * 8 * 128, dtype=np.int64)
                a = (((i * 73 + offset) % 257 - 128) / 128).astype('<f2').reshape(seq, 8, 128)
                a.tofile(case / f'{name}.f16le')
            arrays.append(a.transpose(1, 0, 2).astype(np.float64))
        q, k, v = arrays
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(128.)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        ref = (weights @ v).transpose(1, 0, 2)
        if not np.isfinite(ref).all() or not np.allclose(weights.sum(-1), 1.):
            raise ValueError('invalid reference')
        ref.astype('<f8').tofile(case / 'reference.f64le')
        info = {'shape': [1, seq, 8, 128], 'input_dtype': 'float16', 'output_dtype': 'float16',
                'layout': 'BSHD', 'causal': False, 'reference_dtype': 'float64',
                'scale': 1 / np.sqrt(128.), 'atol': ATOL, 'rtol': RTOL,
                'inputs': {n: digest(case / f'{n}.f16le') for n in ('q', 'k', 'v')},
                'reference_sha256': digest(case / 'reference.f64le')}
        (case / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
        print(f'REFERENCE_RESULT status=PASS seq={seq} elements={ref.size} sha256={info["reference_sha256"]}')

def compare(case, actual, rows, heads):
    info = json.loads((case / 'manifest.json').read_text())
    shape = tuple(info['shape'][1:])
    a = np.fromfile(actual, dtype='<f2')
    if a.size != np.prod(shape):
        raise ValueError(f'output size mismatch: {a.size} vs {np.prod(shape)}')
    a = a.reshape(shape).astype(np.float64)
    ref = np.fromfile(case / 'reference.f64le', dtype='<f8').reshape(shape)
    rows = shape[0] if rows is None else rows
    heads = shape[1] if heads is None else heads
    if not (0 < rows <= shape[0] and 0 < heads <= shape[1]):
        raise ValueError('invalid comparison extent')
    a = a[:rows, :heads]; ref = ref[:rows, :heads]
    if not np.isfinite(a).all():
        raise ValueError('nonfinite output')
    err = np.abs(a - ref); budget = ATOL + RTOL * np.abs(ref)
    if not np.all(err <= budget):
        raise ValueError(f'numerical mismatch max_error={err.max()}')
    report = {'status': 'PASS', 'output': actual.name, 'seq': shape[0],
              'rows': rows, 'heads': heads, 'elements': int(a.size),
              'max_abs_error': float(err.max()), 'max_scaled_error': float((err / budget).max()),
              'rms_error': float(np.sqrt(np.mean(err ** 2))), 'sha256': digest(actual),
              'atol': ATOL, 'rtol': RTOL}
    print('NUMERICAL_RESULT ' + json.dumps(report, sort_keys=True))
    actual.with_suffix('.comparison.json').write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('inputs', type=Path); prep.add_argument('out', type=Path)
    check = sub.add_parser('compare'); check.add_argument('case', type=Path); check.add_argument('actual', type=Path)
    check.add_argument('--rows', type=int); check.add_argument('--heads', type=int)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args.inputs, args.out)
    else: compare(args.case, args.actual, args.rows, args.heads)
