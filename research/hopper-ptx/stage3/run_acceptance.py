#!/usr/bin/env python3
"""Strict stage3 gate: simulator output only; independent element scatter oracle.

Run with in-switch Python, OUT and the absolute rebuilt libcudart.so path.
--baseline records old-runtime defects without producing an acceptance PASS.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import resource
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEADER = '.version 8.7\n.target sm_90a\n.address_size 64\n'
CONFIG = ROOT / 'gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100'
SENTINEL = 0xdeadbeef


def kernel(n, trans, space, layout='compact', address='b32', fault=None):
    # Row addresses may be scattered, repeated, or at the last legal row.
    warp_stride = 512 if layout == 'compact' else 2048
    if layout == 'compact':
        row_code = 'mul.lo.u32 %r7, %r4, 16;'
    else:
        row_code = f'''shr.u32 %r7, %r4, 3;
mul.lo.u32 %r7, %r7, {256 if layout == 'padded' else 512};
and.b32 %r8, %r4, 7;
mul.lo.u32 %r8, %r8, 3;
and.b32 %r8, %r8, 7;
mul.lo.u32 %r8, %r8, {32 if layout == 'padded' else 64};
add.u32 %r7, %r7, %r8;
add.u32 %r7, %r7, {16 if layout == 'padded' else 48};'''
    addr_code = 'add.u32 %a32, %base32, %r7;'
    addr_expr = '%a32'
    if address.endswith('64'):
        addr_code += '\ncvt.u64.u32 %a64, %a32;\nsub.u64 %a64, %a64, 16;'
        addr_expr = '%a64+16'
    if address == 'symbol':
        addr_expr = 'smem+16'
    if address == 'alias':
        addr_code += '\nmov.b32 %d0, %a32;'
        addr_expr = '%d0'
    if fault == 'misaligned':
        addr_code += '\nadd.u32 %a32, %a32, 2;'
    elif fault == 'oob':
        addr_code += '\nadd.u32 %a32, %base32, 4096;'
    elif fault == 'last_provider_oob':
        addr_code += '\nsetp.eq.u32 %p, %r4, 31;\n@%p add.u32 %a32, %base32, 4096;'
    elif fault == 'short_row':
        addr_code += '\nadd.u32 %a32, %base32, 4080;'
    elif fault == 'high64':
        addr_code += '\nadd.u64 %a64, %a64, 4294967296;'
    elif fault == 'divergent':
        addr_code += '\nsetp.eq.u32 %p, %r4, 31;\n@%p bra DONE;'
    # High lanes are not address providers for x1/x2 on sm_90a.
    if n < 4 and address == 'b32' and fault is None:
        addr_code += f'\nsetp.ge.u32 %p, %r4, {n * 8};\n@%p mov.b32 %a32, 4294967295;'
    tuple_ = ', '.join(f'%d{i}' for i in range(n))
    op = f'ldmatrix.sync.aligned.m8n8.x{n}{".trans" if trans else ""}.{space}.b16'
    stores = '\n'.join(f'st.global.b32 [%o64+{4*i}], %d{i};' for i in range(4))
    return HEADER + f'''.visible .entry matrix_load(.param .u64 input, .param .u64 output) {{
.reg .pred %p;
.reg .b32 %r<16>, %d<4>, %base32;
.reg .{address if address.endswith('32') else 'b32'} %a32;
.reg .b64 %g64, %s64, %o64, %i64, %off64, %base64;
.reg .{address if address.endswith('64') else 'b64'} %a64;
.shared .align 16 .b8 smem[{4095 if fault == 'short_row' else 4096}];
mov.u32 %r0, %tid.x;
mov.u32 %r1, %ctaid.x;
mov.u64 %base64, smem;
cvt.u32.u64 %base32, %base64;
ld.param.u64 %g64, [input];
mul.wide.u32 %off64, %r1, 4096;
add.u64 %g64, %g64, %off64;
mul.lo.u32 %r2, %r0, 4;
COPY:
cvt.u64.u32 %off64, %r2;
add.u64 %i64, %g64, %off64;
ld.global.b32 %r3, [%i64];
add.u64 %s64, %base64, %off64;
st.shared.b32 [%s64], %r3;
add.u32 %r2, %r2, 256;
setp.lt.u32 %p, %r2, {4092 if fault == 'short_row' else 4096};
@%p bra COPY;
bar.sync 0;
and.b32 %r4, %r0, 31;
shr.u32 %r5, %r0, 5;
mul.lo.u32 %r6, %r5, {warp_stride};
{row_code}
add.u32 %r7, %r7, %r6;
mov.b32 %d0, 3735928559;
mov.b32 %d1, 3735928559;
mov.b32 %d2, 3735928559;
mov.b32 %d3, 3735928559;
{addr_code}
{op} {{{tuple_}}}, [{addr_expr}];
ld.param.u64 %o64, [output];
mul.lo.u32 %r9, %r1, 64;
add.u32 %r9, %r9, %r0;
mul.wide.u32 %off64, %r9, 16;
add.u64 %o64, %o64, %off64;
{stores}
DONE:
ret;
}}
'''


def reference(data, n, trans, layout, address):
    # Scatter every logical element to its destination. No simulator code or
    # lane-gather formula is reused; packing is explicit low/high halfwords.
    expected = [SENTINEL] * 512
    for cta, warp, matrix in itertools.product(range(2), range(2), range(n)):
        for row, col in itertools.product(range(8), repeat=2):
            if address == 'symbol':
                byte = 16 + col * 2
            elif layout == 'compact':
                byte = warp * 512 + matrix * 128 + row * 16 + col * 2
            else:
                byte = (warp * 2048 + matrix * (256 if layout == 'padded' else 512)
                        + ((row * 3) % 8) * (32 if layout == 'padded' else 64)
                        + (16 if layout == 'padded' else 48) + col * 2)
            dest_row, dest_col = (col, row) if trans else (row, col)
            lane = dest_row * 4 + dest_col // 2
            index = ((cta * 64 + warp * 32 + lane) * 4 + matrix)
            shift = 16 * (dest_col % 2)
            if dest_col % 2 == 0:
                expected[index] = 0
            expected[index] |= data[cta * 2048 + byte // 2] << shift
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve(strict=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = {**os.environ, 'LD_LIBRARY_PATH': str(runtime.parent)}
    results = []

    def invoke(name, command, mode=1):
        case = out / name
        case.mkdir(exist_ok=True)
        for file in ('gpgpusim.config', 'config_hopper_islip.icnt'):
            text = (CONFIG / file).read_text()
            if file == 'gpgpusim.config':
                text = text.replace('-gpgpu_ptx_sim_mode 0', f'-gpgpu_ptx_sim_mode {mode}')
            (case / file).write_text(text)
        try:
            run = subprocess.run(command, cwd=case, env=env, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            (case / 'run.log').write_bytes((error.stdout or b'') + (error.stderr or b''))
            raise RuntimeError(f'{name}: timeout, never accepted') from error
        (case / 'run.log').write_bytes(run.stdout + run.stderr)
        (case / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        return run.returncode, (run.stdout + run.stderr).decode(errors='replace')

    def execute(name, n, trans, space, layout='compact', address='b32', fault=None, mode=1, block=64):
        case = out / name
        case.mkdir(exist_ok=True)
        ptx = case / 'kernel.ptx'
        ptx.write_text(kernel(n, trans, space, layout, address, fault))
        # Nonuniform raw patterns with varied upper bits catch signedness,
        # half ordering and accidental uniformity. No float conversion occurs.
        data = [((i * 4051 + 0x8123) ^ (i >> 3)) & 0xffff for i in range(4096)]
        (case / 'input.bin').write_bytes(struct.pack('<4096H', *data))
        code, log = invoke(name, [str(HERE / 'ldmatrix_harness'), str(ptx),
                                 str(case / 'input.bin'), str(case / 'output.bin'), str(block), str(mode)], mode)
        if fault or mode == 0 or block != 64:
            pattern = ('timing is not implemented' if mode == 0 else
                       'full active warp' if block != 64 or fault == 'divergent' else
                       '16-byte aligned' if fault == 'misaligned' else 'exceeds declared shared memory')
            okay = code != 0 and pattern in log
            record = dict(case=name, kind='negative', exit=code, diagnostic=pattern, status='PASS' if okay else 'FAIL')
        else:
            expected = reference(data, n, trans, layout, address)
            (case / 'reference.bin').write_bytes(struct.pack('<512I', *expected))
            actual = list(struct.unpack('<512I', (case / 'output.bin').read_bytes())) if code == 0 else []
            failures = [i for i in range(512) if i >= len(actual) or actual[i] != expected[i]]
            errors = [abs((actual[i] >> s & 0xffff) - (expected[i] >> s & 0xffff))
                      for i in range(len(actual)) for s in (0, 16)]
            record = dict(case=name, kind='numeric', exit=code, words=len(actual),
                          loaded_halfwords=2*2*32*n*2, mismatches=len(failures),
                          first_mismatch=failures[:1], max_abs_error=max(errors, default=None),
                          rms_error=(sum(e*e for e in errors)/len(errors))**0.5 if errors else None,
                          status='PASS' if code == 0 and not failures else 'FAIL')
        results.append(record)
        print(json.dumps(record), flush=True)
        if record['status'] != 'PASS' and not args.baseline:
            raise RuntimeError(f'{name}: required case failed')

    if args.baseline:
        execute('old_high64', 4, False, 'shared', address='b64', fault='high64')
        execute('old_oob', 4, False, 'shared', fault='oob')
        (out / 'baseline.json').write_text(json.dumps(results, indent=2) + '\n')
        return

    for n, trans, space in itertools.product((1, 2, 4), (False, True), ('shared', 'shared::cta')):
        for layout, address in [('compact', 'b32'), ('padded', 'b64'), ('boundary', 'b32')]:
            execute(f'x{n}-t{int(trans)}-{space.replace("::", "-")}-{layout}', n, trans, space, layout, address)
    execute('symbol-offset', 4, False, 'shared', address='symbol')
    execute('address-destination-alias', 4, True, 'shared::cta', address='alias')
    for address in ('u32','s32','u64','s64'):
        execute('address-'+address, 2, True, 'shared::cta', layout='padded', address=address)
    for fault in ('misaligned', 'oob', 'last_provider_oob', 'short_row', 'high64', 'divergent'):
        execute(fault, 4, False, 'shared', address='b64' if fault == 'high64' else 'b32', fault=fault)
    execute('partial-warp', 4, False, 'shared', block=16)
    execute('legal-detailed-unsupported', 4, False, 'shared', mode=0)

    def frontend(name, body, checks=(), diagnostic=None):
        case = out / name
        case.mkdir(exist_ok=True)
        ptx = case / 'kernel.ptx'
        ptx.write_text(HEADER + '.visible .entry frontend() {\n' + body + '\nret;\n}\n')
        code, log = invoke(name, [str(HERE / 'frontend_probe'), str(ptx), 'frontend'])
        okay = (code != 0 and diagnostic in log) if diagnostic else (
            code == 0 and 'target=sm_90a target_number=90' in log and all(c in log for c in checks))
        record = dict(case=name, kind='frontend-negative' if diagnostic else 'frontend', exit=code, status='PASS' if okay else 'FAIL')
        results.append(record)
        if not okay:
            raise RuntimeError(f'{name}: required frontend case failed')

    # All five registered MMA forms: tokens and complete tuple identity only.
    decl = '.reg .b32 %a<4>, %b<2>; .reg .f32 %c<4>, %d<4>;\n'
    tuples = '{%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3}'
    for type_, k in [('f16',16), ('bf16',16), ('tf32',8), ('e4m3',32), ('e5m2',32)]:
        op = f'mma.sync.aligned.m16n8k{k}.row.col.f32.{type_}.{type_}.f32'
        body = decl + op + ' ' + tuples + ';'
        frontend('mma-'+type_, body, (f'opcode={op}', f'm=16 n=8 k={k}', 'tuple0=4 %d0 %d1 %d2 %d3', 'functional=0 timing=0'))
        frontend('mma-bad-mod-'+type_, body.replace('.row.col.', '.col.col.'), diagnostic='unsupported matrix opcode/modifier')
        frontend('mma-bad-op-'+type_, body.replace('{%b0,%b1}', '{%b0}'), diagnostic='matrix register tuple length mismatch')

    for n in (128,176):
        count = n // 2
        for source in ('ss','rs'):
            tuple_ = '{' + ','.join(f'%d{i}' for i in range(count)) + '}'
            decl_w = f'.reg .f32 %d<{count}>; .reg .b64 %desc<2>; .reg .b32 %a<4>;\n'
            operands = '%desc0, %desc1, 1, 1, 1, 0, 0' if source == 'ss' else '{%a0,%a1,%a2,%a3}, %desc1, 1, 1, 1, 0'
            body = decl_w + f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.f16.f16 {tuple_}, {operands};'
            frontend(f'wgmma-{n}-{source}', body, (f'tuple0={count}', f'%d{count-1}', 'functional=1 timing=1'))
            frontend(f'wgmma-bad-mod-{n}-{source}', body.replace('k16','k32'), diagnostic='unsupported matrix opcode/modifier')
            frontend(f'wgmma-bad-op-{n}-{source}', body.replace(f',%d{count-1}', ''), diagnostic='matrix register tuple length mismatch')
    for op, tail, bad in [('fence','', ' 1'), ('commit_group','', ' 1'), ('wait_group',' 7', ' 8')]:
        prefix = f'wgmma.{op}.sync.aligned'
        frontend('wgmma-'+op, prefix+tail+';', ('functional=1 timing=1',))
        frontend('wgmma-bad-op-'+op, prefix+bad+';', diagnostic='immediate must be 0..7' if op=='wait_group' else 'take no operands')
        frontend('wgmma-bad-mod-'+op, prefix.replace('aligned','unaligned')+tail+';', diagnostic='unsupported matrix opcode/modifier')

    base = '.reg .b32 %d<4>, %a;\nldmatrix.sync.aligned.m8n8.x4.shared.b16 {%d0,%d1,%d2,%d3}, [%a];'
    for name, body, diagnostic in [
        ('ld-bad-mod', base.replace('.aligned.', '.unaligned.'), 'unsupported matrix opcode/modifier'),
        ('ld-bad-count', base.replace(',%d3',''), 'matrix register tuple length mismatch'),
        ('ld-duplicate', base.replace('%d3}', '%d0}'), 'duplicate register'),
        ('ld-dest-width', base.replace('.b32 %d<4>, %a', '.b16 %d<4>; .reg .b32 %a'), 'specified bit width/type'),
        ('ld-address-float', base.replace('.b32 %d<4>, %a', '.b32 %d<4>; .reg .f32 %a'), 'ldmatrix address register'),
        ('ld-address-narrow', base.replace('.b32 %d<4>, %a', '.b32 %d<4>; .reg .b16 %a'), 'ldmatrix address register'),
        ('ld-address-global', base.replace('.reg .b32 %d<4>, %a', '.reg .b32 %d<4>; .global .b32 a').replace('[%a]','[a]'), 'ldmatrix address must'),
        ('ld-arity', base.replace('[%a];', '[%a], 0;'), 'requires a tuple and shared address'),
        ('ld-predicated', '.reg .pred %p;\n'+base.replace('ldmatrix.', '@%p ldmatrix.'), 'predicated ldmatrix'),
        ('ld-unsupported-generic', base.replace('.shared.b16', '.b16'), 'unsupported matrix opcode/modifier'),
        ('ld-unsupported-shape', base.replace('.m8n8.', '.m16n16.'), 'unsupported matrix opcode/modifier'),
    ]:
        frontend(name, body, diagnostic=diagnostic)
    # External tools gate real PTX validity; their cubins are never executed.
    for record in results:
        if record['kind'] == 'numeric':
            case = out / record['case']
            run = subprocess.run(['/usr/local/cuda/bin/ptxas', '-arch=sm_90a', str(case/'kernel.ptx'), '-o', str(case/'kernel.cubin')], capture_output=True)
            (case/'ptxas.log').write_bytes(run.stdout+run.stderr)
            if run.returncode:
                raise RuntimeError(f'{record["case"]}: ptxas rejected positive fixture')
            (case/'kernel.cubin').unlink()  # Evidence is PTX and ptxas log, not binary output.
    # Set equality guards prevent missing/skipped cases from passing the gate.
    counts = {kind: sum(r['kind']==kind for r in results) for kind in set(r['kind'] for r in results)}
    assert counts == {'numeric':42, 'negative':8, 'frontend':12, 'frontend-negative':35}, counts
    numeric_required = {f'x{n}-t{t}-{s}-{layout}' for n,t,s,layout in itertools.product(
        (1,2,4),(0,1),('shared','shared-cta'),('compact','padded','boundary'))}
    numeric_required |= {'symbol-offset','address-destination-alias','address-u32','address-s32','address-u64','address-s64'}
    assert {r['case'] for r in results if r['kind']=='numeric'} == numeric_required
    assert len({r['case'] for r in results}) == len(results), 'duplicate case cannot replace a missing case'
    manifest = {'status':'PASS', 'counts':counts, 'results':results,
                'runtime':str(runtime), 'runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest()}
    (out/'component-results.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('STAGE3_COMPONENT status=PASS '+json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
