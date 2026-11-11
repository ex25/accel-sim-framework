#!/usr/bin/env python3
"""Run frozen module 16 with independent grouped integer matrices (FC1/FC2)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SIM = ROOT / 'gpu-simulator/gpgpu-sim'
ART = ROOT / 'workloads/moe/artifacts'
sys.path.insert(0, str(HERE.parent / 'stage-2'))
from read_forward import load


def fields(raw):
    # Offsets follow frozen ld.param operands, not a host structure definition.
    names = [(16, 'Q', 'problems_mnk'), (456, 'Q', 'mainloop_workspace'),
             (464, 'Q', 'a_pointer_array'), (472, 'Q', 'a_strides_elements'),
             (480, 'Q', 'b_pointer_array'), (488, 'Q', 'b_strides_elements'),
             (512, 'f', 'beta'), (552, 'f', 'alpha'),
             (1024, 'Q', 'epilogue_workspace'), (1032, 'Q', 'c_pointer_array'),
             (1040, 'Q', 'c_strides_elements'), (1048, 'Q', 'd_pointer_array'),
             (1056, 'Q', 'd_strides_elements'), (1092, 'I', 'resident_ctas'),
             (1096, 'I', 'grid_x'), (1208, 'Q', 'scheduler_tiles'),
             (1232, 'I', 'group_count'), (1240, 'Q', 'scheduler_problems_mnk'),
             (1280, 'Q', 'scheduler_epilogue_workspace')]
    return [dict(offset=off, format=fmt, name=name,
                 value=struct.unpack_from('<' + fmt, raw, off)[0])
            for off, fmt, name in names]


def prepare(run):
    # Read the host-written problem table before each expert call. Its pointer
    # arrays are reconstructed by expert.cc, independently of simulator layouts.
    source = ART / 'cases/qwen64-t128/forward.bin'
    launches = load(source)
    tables = []
    maps = []
    writes = {}
    with source.open('rb') as f:
        def rd(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
        def blob():
            return f.read(rd('I')[0])
        assert rd('II') == (0x4d4f4550, 3)
        while f.peek(1):
            tag = rd('I')[0]
            if tag == 1:
                rd('QQ')
            elif tag == 2:
                blob()
                module = rd('Q')[0]
                rd('10I')
                args = [blob() for _ in range(rd('I')[0])]
                if module == 16:
                    address = struct.unpack_from('<Q', args[0], 16)[0]
                    tables.append(writes[address][:63 * 12])
            elif tag in (3, 10):
                address = rd('Q')[0]
                writes[address] = blob()
                if tag == 3:
                    f.read(rd('I')[0] * 4)
            elif tag in (4, 8):
                rd('QQ')
            elif tag == 5:
                original = f.read(128)
                address, dtype, rank = rd('QII')
                dims, strides = rd('Q' * rank), rd('Q' * (rank - 1))
                box, element, flags = rd('I' * rank), rd('I' * rank), rd('4I')
                maps.append(dict(original=original, address=address, dtype=dtype,
                                 dims=dims, strides=strides, box=box,
                                 element=element, flags=flags))
            elif tag == 6:
                rd('QIIQ')
            elif tag == 7:
                rd('QQIfQ')
            elif tag == 9:
                rd('QQQ')
            else:
                raise ValueError(tag)
    calls = [x for x in launches if x['module'] == 16]
    report = []
    for i, call in enumerate(calls):
        raw = bytes.fromhex(call['args'][0])
        (run / f'fc{i+1}.param.raw').write_bytes(raw)
        (run / f'fc{i+1}.shapes').write_bytes(tables[i])
        report.append(dict(consumer=f'fc{i+1}', **call,
                           fields=fields(raw),
                           problems=list(struct.iter_unpack('<iii', tables[i]))))
    (run / 'launches.json').write_text(json.dumps(report, indent=2) + '\n')
    # Descriptor offsets are established by the frozen ld.param -> shared stores.
    desc = bytearray()
    for offset in (64, 256, 640, 832):
        raw = bytes.fromhex(calls[0]['args'][0])[offset:offset+128]
        m = next(m for m in maps if m['original'] == raw)
        desc += struct.pack('<II', m['dtype'], 2)
        desc += struct.pack('<2Q', *m['dims']) + struct.pack('<Q', *m['strides'])
        desc += struct.pack('<2I', *m['box']) + struct.pack('<2I', *m['element'])
        desc += struct.pack('<4I', *m['flags'])
    (run / 'maps.raw').write_bytes(desc)
    (run / 'abi.json').write_text(json.dumps(dict(
        parameter_bytes=1344, alignment_bytes=64, physical_registers_per_thread=168,
        descriptors=[dict(offset=off, dtype=m['dtype'], dims=m['dims'],
                          strides=m['strides'], box=m['box'], element=m['element'],
                          flags=m['flags']) for off in (64, 256, 640, 832)
                     for m in maps if m['original'] == bytes.fromhex(calls[0]['args'][0])[off:off+128]],
        matrices='A[M,K] and B[N,K], BF16, element strides K; D[M,N], BF16, row stride N.',
        output='alpha=1, beta=0: C is not read. Compare all M*N outputs per group and 128 guard words after them.',
        scheduler='Tile M=128, cluster tile N=256 with two CTAs; ceil(M/128)*ceil(N/256)*2 per group.',
        reference='Independent signed integer dot product, then one nearest-even BF16 conversion.',
    ), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--test', choices=['fc1', 'fc2', 'loading', 'streams', 'normalize', 'fp32'], default='fc1')
    p.add_argument('--real-shape', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--tile-rounds', type=int, choices=[1, 2], default=1,
                   help='Small-shape N multiplier to exercise persistent tile reuse')
    p.add_argument('--active-group', type=int,
                   help='Keep only this group nonempty, preserving real N/K and launch resources')
    a = p.parse_args()
    if a.active_group is not None:
        if not a.real_shape or a.test not in ('fc1', 'fc2'):
            p.error('--active-group requires --real-shape and --test fc1 or fc2')
        if not 0 <= a.active_group < 63:
            p.error('--active-group must select one of the 63 frozen groups')
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / 'sources').mkdir()
    for s in HERE.glob('*'):
        if s.suffix in ('.py', '.cc', '.ptx'):
            shutil.copy2(s, run / 'sources')
    if a.test in ('fc1', 'fc2'):
        prepare(run)
    cuda = Path(os.environ.get('CUDA_INSTALL_PATH', '/usr/local/cuda'))
    commands = []
    def call(cmd, **kw):
        commands.append(list(map(str, cmd)))
        return subprocess.run(cmd, check=True, **kw)
    if a.test in ('loading', 'streams', 'fp32'):
        ptx = HERE / ('loading.ptx' if a.test == 'streams' else a.test + '.ptx')
    else:
        index = 75 if a.test == 'normalize' else 16
        ptx = ART / json.loads((ART / 'workloads.json').read_text())['modules'][index]['ptx']
    call([cuda / 'bin/fatbinary', '--64', '--create=' + str(run / 'expert.fatbin'),
          '--image=profile=compute_90a,file=' + str(ptx)])
    (run / 'embed.S').write_text(
        '.section .nv_fatbin,"a"\n.balign 8\n.global expert_module\n'
        'expert_module:\n.incbin "' + str(run / 'expert.fatbin') +
        '"\n.section .note.GNU-stack,"",@progbits\n')
    call(['gcc', '-c', run / 'embed.S', '-o', run / 'embed.o'])
    call(['g++', '-std=c++17', '-g', '-O2', '-I' + str(cuda / 'include'),
          HERE / 'expert.cc', run / 'embed.o', '-L' + str(cuda / 'lib64'),
          '-lcudart', '-o', run / 'expert'])
    libraries = subprocess.check_output(['ldd', run / 'expert'], text=True)
    library = Path(next(x.split('=>')[1].split('(')[0].strip()
                        for x in libraries.splitlines() if 'libcudart' in x))
    assert 'gpgpu-sim' in str(library)
    for name in ('gpgpusim.config', 'config_hopper_islip.icnt'):
        shutil.copy2(SIM / 'configs/tested-cfgs/SM90_H100' / name, run)
    if a.trace:
        with (run/'gpgpusim.config').open('a') as f:
            f.write('\n-trace_enabled 1\n-trace_components PTX_TMA\n')
    env = os.environ.copy()
    for k in ('PTX_SIM_USE_PTX_FILE', 'PTX_SIM_KERNELFILE', 'CUOBJDUMP_SIM_FILE',
              'MOE_TRACE_LAUNCHES'):
        env.pop(k, None)
    env['PTX_SIM_MODE_FUNC'] = '0'
    diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=SIM)
    (run / 'inner.diff').write_bytes(diff)
    test_diff = subprocess.check_output(
        ['git', 'diff', '--binary', '--',
         str(HERE / 'expert.py'), str(HERE / 'expert.cc')], cwd=ROOT)
    (run / 'test.diff').write_bytes(test_diff)
    metadata = dict(commands=commands, libraries=libraries,
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    outer=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    inner=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SIM,text=True).strip(),
                    ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest(),
                    real_shape=a.real_shape, test=a.test, tile_rounds=a.tile_rounds,
                    active_group=a.active_group,
                    test_diff_sha256=hashlib.sha256(test_diff).hexdigest(),
                    test_sources_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                         for name in ('expert.py', 'expert.cc')},
                    config_sha256=hashlib.sha256((run/'gpgpusim.config').read_bytes()).hexdigest())
    (run/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    cmd = ['/usr/bin/time', '-v', '-o', str(run/'expert.time'),
           str(run/'expert'), a.test, 'real' if a.real_shape else 'small', str(a.tile_rounds)]
    if a.active_group is not None:
        cmd.append(str(a.active_group))
    commands.append(cmd)
    with (run/'expert.log').open('w') as log:
        result = subprocess.run(cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(returncode=result.returncode,
                    passed=result.returncode == 0 and 'PASS expert' in (run/'expert.log').read_text())
    actual_param = run / (a.test + '.param.actual')
    if actual_param.exists():
        metadata['actual_fields'] = fields(actual_param.read_bytes())
    (run/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
