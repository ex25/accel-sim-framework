#!/usr/bin/env python3
"""Generate textual PTX fixtures; outputs are disposable build artifacts."""
from pathlib import Path
import sys
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)

def kernel(n, registers=False, count=1, transpose=False, scale_d=0, sign=1, negative='', groups=False):
    size = (64+n)*32
    lines = ['.version 8.7', '.target sm_90a', '.address_size 64',
        '.visible .entry wgmma_kernel(.param .u64 output, .param .u64 input) {',
        '.reg .b32 %r<8>, %a<4>;', '.reg .b64 %rd<12>;', '.reg .b16 %v;',
        '.reg .pred %p, %scale;', f'.reg .f32 %d<{n//2}>;',
        f'.shared .align 16 .b8 tile[{size}];',
        'ld.param.u64 %rd0, [output];', 'ld.param.u64 %rd1, [input];',
        'mov.u32 %r0, %tid.x;', 'mov.u64 %rd2, tile;',
        'mov.u32 %r1, %r0;', 'COPY:', f'setp.ge.u32 %p, %r1, {size//2};',
        '@%p bra COPY_DONE;', 'mul.wide.u32 %rd3, %r1, 2;',
        'add.u64 %rd4, %rd1, %rd3;', 'ld.global.u16 %v, [%rd4];',
        'add.u64 %rd5, %rd2, %rd3;', 'st.shared.u16 [%rd5], %v;',
        'add.u32 %r1, %r1, 128;', 'bra COPY;', 'COPY_DONE:', 'bar.sync 0;',
        'fence.proxy.async.shared::cta;',
        # No-swizzle: K-major LBO=128/SBO=256, MN-major LBO=mn*16/SBO=128.
        f'mov.u64 %rd6, {(64*16 if transpose else 128)//16 << 16 | (128 if transpose else 256)//16 << 32};',
        f'mov.u64 %rd7, {2048//16 | (n*16 if transpose else 128)//16 << 16 | (128 if transpose else 256)//16 << 32};',
        ('setp.ne.u32 %scale, %r0, %r0;' if scale_d == 'false_predicate' else 'setp.eq.u32 %scale, %r0, %r0;')]
    if negative == 'nonuniform': lines += ['cvt.u64.u32 %rd8, %r0;', 'add.u64 %rd7, %rd7, %rd8;']
    if negative == 'swizzle': lines += ['or.b64 %rd7, %rd7, 4611686018427387904;']
    if negative == 'oob': lines += ['or.b64 %rd7, %rd7, 16383;']
    if negative == 'divergent': lines += ['setp.ge.u32 %p, %r0, 64;', '@%p bra EXIT;']
    if negative == 'partial': lines += ['setp.eq.u32 %p, %r0, 0;', '@%p bra EXIT;']
    if registers:
        lines += ['mul.wide.u32 %rd8, %r0, 16;', f'add.u64 %rd8, %rd8, {size};', 'add.u64 %rd8, %rd8, %rd1;']
        lines += [f'ld.global.b32 %a{i}, [%rd8+{4*i}];' for i in range(4)]
    lines += [f'mov.f32 %d{i}, 0f3F000000;' for i in range(n//2)]
    if negative != 'no_fence': lines += ['wgmma.fence.sync.aligned;']
    dst = '{'+', '.join(f'%d{i}' for i in range(n//2))+'}'
    source = '{%a0, %a1, %a2, %a3}' if registers else '%rd6'
    scale = '%scale' if scale_d in ('predicate', 'false_predicate') else str(scale_d)
    suffix = f'{scale}, {sign}, 1, {int(transpose)}' if registers else f'{scale}, {sign}, 1, {int(transpose)}, {int(transpose)}'
    mma = f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.f16.f16 {dst}, {source}, %rd7, {suffix};'
    for i in range(count):
        lines += [mma]
        if groups: lines += ['wgmma.commit_group.sync.aligned;']
    if not groups: lines += ['wgmma.commit_group.sync.aligned;']
    if groups: lines += ['wgmma.wait_group.sync.aligned 1;']
    if negative != 'no_wait': lines += ['wgmma.wait_group.sync.aligned 0;']
    lines += [f'mul.wide.u32 %rd9, %r0, {n//2*4};', 'add.u64 %rd9, %rd9, %rd0;']
    lines += [f'st.global.f32 [%rd9+{i*4}], %d{i};' for i in range(n//2)]
    lines += ['EXIT:', 'ret;', '}']
    return '\n'.join(lines)+'\n'

cases = {
    'ss128': (128, False, 1, False, 0, 1),
    'ss176': (176, False, 2, True, 1, 1),
    'rs128': (128, True, 2, False, 1, -1),
    'rs176': (176, True, 1, True, 'predicate', 1),
    'rs176_zero': (176, True, 1, True, 'false_predicate', 1),
    'empty': (128, False, 0, False, 1, 1),
}
for name, args in cases.items(): (out/(name+'.ptx')).write_text(kernel(*args))
(out/'ss128_hex.ptx').write_text(kernel(128).replace('68720001024', '0x1000080000').replace('68720001152', '0x1000080080'))
(out/'empty.ptx').write_text(kernel(128,count=0,scale_d=1).replace('wgmma.wait_group.sync.aligned 0;', 'wgmma.wait_group.sync.aligned 7;\nwgmma.wait_group.sync.aligned 0;'))
for n in (128,176): (out/f'timing{n}.ptx').write_text(kernel(n,count=3,scale_d=1,groups=True))
(out/'wait1_access.ptx').write_text(kernel(128, count=3, scale_d=1, groups=True, negative='no_wait'))
(out/'source_overwrite.ptx').write_text(kernel(128, registers=True).replace('wgmma.commit_group.sync.aligned;', 'wgmma.commit_group.sync.aligned;\nmov.b32 %a0, 0;'))
(out/'after_fence.ptx').write_text(kernel(128).replace('wgmma.fence.sync.aligned;', 'wgmma.fence.sync.aligned;\nmov.f32 %d0, 0f00000000;'))
base=kernel(128)
for name in ('no_fence','no_wait','swizzle','oob','nonuniform','divergent','partial'):
    (out/(name+'.ptx')).write_text(kernel(128,negative=name))
for name,text in {
    'bad_tuple':base.replace('%d0, %d1,', '%d0,', 1),
    'duplicate_tuple':base.replace('%d0, %d1,', '%d0, %d0,', 1),
    'bad_type':base.replace('.reg .f32 %d<64>', '.reg .b32 %d<64>'),
    'bad_descriptor_width':base.replace('}, %rd6, %rd7,', '}, %r6, %rd7,'),
    'literal_overflow':base.replace('mov.u64 %rd6, 68720001024;', 'mov.u64 %rd6, 18446744073709551616;'),
    'hex_overflow':base.replace('mov.u64 %rd6, 68720001024;', 'mov.u64 %rd6, 0x10000000000000000;'),
    'bad_modifier':base.replace('.f32.f16.f16', '.satfinite.f32.f16.f16'),
    'bad_scale':base.replace('%rd7, 0, 1, 1, 0, 0;', '%rd7, 0, 2, 1, 0, 0;'),
    'float_scale':base.replace('%rd7, 0, 1, 1, 0, 0;', '%rd7, 0, 0f00000000, 1, 0, 0;'),
    'bad_scale64':base.replace('%rd7, 0, 1, 1, 0, 0;', '%rd7, 0, 4294967297, 1, 0, 0;'),
    'bad_wait64':base.replace('wait_group.sync.aligned 0;', 'wait_group.sync.aligned 4294967296;'),
    'bad_wait':base.replace('wait_group.sync.aligned 0;', 'wait_group.sync.aligned 8;'),
    'predicated':base.replace('wgmma.fence.sync.aligned;', '@%scale wgmma.fence.sync.aligned;'),
}.items(): (out/(name+'.ptx')).write_text(text)
