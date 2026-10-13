#!/usr/bin/env python3
"""Build and run all mandatory stage3 checks; any exception leaves no PASS gate."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INNER = ROOT / 'gpu-simulator/gpgpu-sim'
PYTHON = Path('/home/wangjz/miniconda3/envs/in-switch/bin/python3')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--build', type=Path, default=Path('/tmp/hopper-stage3-acceptance-build'))
    args = parser.parse_args()
    out, build = args.out.resolve(), args.build.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    lock = (build/'stage3-suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'gate.json').exists():
        raise RuntimeError('use a fresh evidence directory; never reuse a prior PASS gate')
    runtime = build/'libcudart.so'
    env = {**os.environ, 'SIM_RUNTIME':str(runtime), 'PYTHON_BIN':str(PYTHON),
           'LD_LIBRARY_PATH':str(build), 'CUDA_INSTALL_PATH':'/usr/local/cuda'}
    commands = []

    def run(name, command, cwd=ROOT, timeout=600):
        commands.append(dict(name=name, argv=[str(a) for a in command], cwd=str(cwd)))
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with (out/(name+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'{name} exit={result.returncode}, see complete log')
        print(f'CHECK status=PASS name={name}', flush=True)
        return (out/(name+'.log')).read_text()

    def git(repo, *args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True)

    version = dict(outer_head=git(ROOT,'rev-parse','HEAD').strip(), inner_head=git(INNER,'rev-parse','HEAD').strip(),
                   outer_branch=git(ROOT,'branch','--show-current').strip(), inner_branch=git(INNER,'branch','--show-current').strip(),
                   outer_dirty=git(ROOT,'status','--short'), inner_dirty=git(INNER,'status','--short'),
                   python=str(PYTHON), env_overrides={k:env[k] for k in ('SIM_RUNTIME','PYTHON_BIN','LD_LIBRARY_PATH','CUDA_INSTALL_PATH')})
    assert version['outer_branch'] == version['inner_branch'] == 'research/hopper-ptx'
    (out/'versions.json').write_text(json.dumps(version, indent=2)+'\n')
    (out/'inner.patch').write_text(git(INNER,'diff','HEAD'))
    (out/'outer.patch').write_text(git(ROOT,'diff','HEAD'))
    source_paths = [INNER/p for p in ('src/abstract_hardware_model.h', 'src/abstract_hardware_model.cc',
                    'src/cuda-sim/instructions.cc', 'src/cuda-sim/ptx_parser.cc', 'src/cuda-sim/cuda-sim.cc')]
    source_paths += [p for stage in (2,3,6,8) for p in (HERE.parent/f'stage{stage}').glob('*')
                     if p.is_file() and (p.suffix in ('.cc','.py','.sh','.ptx') or p.name=='Makefile')]
    def source_hashes():
        return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    source_before = source_hashes()
    (out/'source-sha256.json').write_text(json.dumps(source_before,indent=2)+'\n')
    run('python-version', [str(PYTHON), '-c', 'import sys,importlib.metadata as m; print(sys.executable); print(sys.version); print({p: next((x.version for x in m.distributions() if x.metadata["Name"].lower()==p), "absent") for p in ("numpy","torch","triton")})'])
    for name, cmd in [('compiler',['/usr/bin/g++','--version']), ('cuda',['/usr/local/cuda/bin/nvcc','--version']),
                      ('ptxas',['/usr/local/cuda/bin/ptxas','--version']), ('cmake',['cmake','--version']),
                      ('bison',['bison','--version']), ('flex',['flex','--version'])]:
        run(name,cmd)
    run('configure', ['cmake','-S',str(INNER),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCUDAToolkit_ROOT=/usr/local/cuda'])
    run('build', ['cmake','--build',str(build),'--target','cudart','-j2'],timeout=1200)
    run('harness-build', ['make','-C',str(HERE),'-B','all',f'SIM_RUNTIME={runtime}'])
    libraries = run('loaded-libraries',['ldd',str(HERE/'ldmatrix_harness')])
    assert str(runtime) in libraries
    run('components',[str(PYTHON),str(HERE/'run_acceptance.py'),str(out/'components'),str(runtime)],timeout=900)
    component = json.loads((out/'components/component-results.json').read_text())
    assert component['status']=='PASS' and component['counts']=={'numeric':42,'negative':8,'frontend':12,'frontend-negative':35}
    ordinary = HERE.parent/'stage2'
    run('ordinary-build',['make','-C',str(ordinary),'-B','all','CXX=/usr/bin/g++',f'SIM_RUNTIME={runtime}'])
    config = INNER/'configs/tested-cfgs/SM90_H100'
    for file in ('gpgpusim.config','config_hopper_islip.icnt'):
        (out/file).write_bytes((config/file).read_bytes())
    log = run('ordinary',[str(ordinary/'stage2_harness'),str(ordinary/'stage2_kernel.ptx')],cwd=out)
    assert 'RESULT status=PASS n=128 checksum=161408 first=1007 last=1515' in log
    stage8 = HERE.parent/'stage8'
    run('param-label-build',['make','-C',str(stage8),'-B','param_layout_probe','readiness_execution_probe',f'SIM_RUNTIME={runtime}'])
    run('params',[str(PYTHON),str(stage8/'param_layout_probe.py'),str(out/'params')])
    param = json.loads((out/'params/result.json').read_text())
    assert param=={'status':'PASS','positive_cases':4,'negative_cases':5,'copy_guard':'PASS'}
    run('labels-frontend',[str(PYTHON),str(stage8/'frontend_contracts.py'),str(ROOT),str(out/'labels-frontend'),str(config)])
    labels=json.loads((out/'labels-frontend/result.json').read_text())
    assert labels['status']=='PASS' and labels['scoped_label_modes']==2 and labels['undefined_label_rejected']
    assert labels['positive_cases']==12 and labels['negative_cases']==19 and labels['strict_execution_cases']==24
    stage6 = HERE.parent/'stage6'
    run('wgmma-build',['make','-C',str(stage6),'-B','all',f'SIM_RUNTIME={runtime}'])
    log=run('wgmma',['bash',str(stage6/'run.sh'),str(out/'wgmma')],timeout=600)
    assert log.count('MATRIX_RESULT status=PASS')==16, log[-2000:]
    assert log.count('NEGATIVE_RESULT status=PASS')==36, log[-2000:]
    runtime_hash=hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert component['runtime_sha256']==runtime_hash
    assert source_hashes()==source_before, 'source changed during required checks; rerun on frozen source'
    version['runtime_sha256']=runtime_hash
    version['runtime_size']=runtime.stat().st_size
    (out/'versions.json').write_text(json.dumps(version,indent=2)+'\n')
    # Hash every durable input, full output, reference, command and log. Gate
    # is written only after all required contracts succeed on this runtime.
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    gate=dict(status='PASS', stage=3, component=component['counts'], ordinary_values=128,
              parameter_positive=4, parameter_negative=5, label_modes=2,
              frontend_regression_positive=12, frontend_regression_negative=19, strict_unsupported_checks=24,
              wgmma_numeric=16, wgmma_negative=36, runtime_sha256=runtime_hash,
              accepted_scope='frontend and static-shared m8n8.b16 functional loads; no ldmatrix detailed timing or hardware validation')
    (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('STAGE3_GATE status=PASS',flush=True)


if __name__=='__main__':
    main()
