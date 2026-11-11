#!/usr/bin/env python3
"""Check Qwen64/128 routing with the frozen initialization recipes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MOE = ROOT / 'workloads/moe'
SIM = ROOT / 'gpu-simulator/gpgpu-sim'
ART = MOE / 'artifacts'
sys.path.insert(0, str(HERE.parent / 'stage-1'))
from read_forward import launches

run = Path(sys.argv[1]).resolve()
run.mkdir(parents=True, exist_ok=False)
modules = json.loads((ART / 'workloads.json').read_text())['modules']
selected = [70, 71, 3, 4, 72, 73, 74, 75, 76, 77]
specs = []
asm = []
declarations = []
for index in selected:
    module = modules[index]
    launch = next(item for item in launches if item['module'] == index)
    fatbin = run / f'{index}.fatbin'
    subprocess.run([
        os.environ.get('CUDA_INSTALL_PATH', '/usr/local/cuda') + '/bin/fatbinary',
        '--64', '--create=' + str(fatbin),
        '--image=profile=compute_90a,file=' + str(ART / module['ptx']),
    ], check=True)
    asm.append(
        f'.section .nv_fatbin,"a"\n.balign 8\n.global route_{index}\n'
        f'route_{index}:\n.incbin "{fatbin}"\n'
    )
    declarations.append(f'extern "C" const unsigned char route_{index}[];\n')
    args = '{' + ','.join(
        '{' + ','.join(map(str, bytes.fromhex(arg))) + '}'
        for arg in launch['args']
    ) + '}'
    pointers = '{' + ','.join(
        '{' + ','.join(map(str, item)) + '}'
        for item in module['parameter_words']
    ) + '}'
    shape = '{' + ','.join(map(str, launch['shape'])) + '}'
    specs.append(
        '{' + str(index) + ',{"' + module['entry'] + '",route_' +
        str(index) + ',' + shape + ',' + args + ',' + pointers + '}}'
    )

(run / 'routing_launches.inc').write_text(
    ''.join(declarations) +
    'static const std::map<unsigned, Launch> specs={' + ','.join(specs) + '};\n'
)
(run / 'embed.S').write_text(''.join(asm) +
                             '.section .note.GNU-stack,"",@progbits\n')
subprocess.run(['gcc', '-c', str(run / 'embed.S'), '-o', str(run / 'embed.o')],
               check=True)
cuda = Path(os.environ.get('CUDA_INSTALL_PATH', '/usr/local/cuda'))
subprocess.run([
    'g++', '-std=c++17', '-g', '-O0', '-I' + str(cuda / 'include'),
    '-I' + str(run), str(HERE / 'routing.cc'), str(run / 'embed.o'),
    '-L' + str(cuda / 'lib64'), '-lcudart', '-o', str(run / 'routing'),
], check=True)
libraries = subprocess.check_output(['ldd', str(run / 'routing')], text=True)
if not any('libcudart' in line and 'gpgpu-sim' in line
           for line in libraries.splitlines()):
    raise RuntimeError(libraries)
shutil.copy2(HERE / 'routing.cc', run / 'source.cc')
shutil.copy2(Path(__file__), run / 'driver.py')
source_diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=SIM)
(run / 'inner.diff').write_bytes(source_diff)
shutil.copy2(SIM / 'configs/tested-cfgs/SM90_H100/gpgpusim.config', run)
shutil.copy2(SIM / 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt', run)
env = os.environ.copy()
for key in ('PTX_SIM_USE_PTX_FILE', 'PTX_SIM_KERNELFILE', 'CUOBJDUMP_SIM_FILE'):
    env.pop(key, None)
env['PTX_SIM_MODE_FUNC'] = '0'
with (run / 'stdout.log').open('w') as log:
    result = subprocess.run([str(run / 'routing')] + sys.argv[2:], cwd=run, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
metadata = dict(
    passed=result.returncode == 0,
    returncode=result.returncode,
    binary_sha256=hashlib.sha256((run / 'routing').read_bytes()).hexdigest(),
    source_sha256={str(i): modules[i]['source_sha256'] for i in selected},
    forward_sha256=hashlib.sha256(
        (ART / 'cases/qwen64-t128/forward.bin').read_bytes()).hexdigest(),
    loaded_libraries=libraries,
    full_h100_config=True,
    library_sha256=hashlib.sha256(Path(next(
        line.split('=>')[1].split('(')[0].strip()
        for line in libraries.splitlines() if 'libcudart' in line
    )).read_bytes()).hexdigest(),
    source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
)
(run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
print((run / 'stdout.log').read_text()[-2500:])
raise SystemExit(result.returncode)
