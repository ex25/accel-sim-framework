#!/usr/bin/env python3
"""Build and run the stage 0 PTX tests after sourcing simulator setup_environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SIM = ROOT/'gpu-simulator/gpgpu-sim'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--test', choices=['launch','work','modules','parameters','shared','normal','assert','trap','all'], default='all')
p.add_argument('--concurrent', action='store_true')
a = p.parse_args()
run = a.run_dir.resolve()
run.mkdir(parents=True, exist_ok=True)
cuda = Path(os.environ.get('CUDA_INSTALL_PATH','/usr/local/cuda'))
commands=[]
def call(cmd, **kw):
    commands.append([str(x) for x in cmd])
    subprocess.run(cmd,check=True,**kw)
call([sys.executable,HERE/'generate_ptx.py',run])
for module,target in [('a','sm_90'),('b','sm_90a')]:
    call([cuda/'bin/fatbinary','--64','--create='+str(run/(module+'.fatbin')),
          '--image=profile='+target.replace('sm_','compute_')+',file='+str(run/(module+'.ptx'))])
(run/'embed.S').write_text(''.join(
    '.section .nv_fatbin,"a"\n.balign 8\n.global module_'+m+'\nmodule_'+m+':\n.incbin "'+str(run/(m+'.fatbin'))+'"\n'
    for m in ['a','b'])+'.section .note.GNU-stack,"",@progbits\n')
call(['gcc','-c',run/'embed.S','-o',run/'embed.o'])
call([cuda/'bin/nvcc','-arch=sm_90','-cudart=shared','-std=c++17','--keep',
      '--keep-dir',run,'-g','-O0',HERE/'launch.cu',run/'embed.o','-o',run/'stage0'])
libraries=subprocess.check_output(['ldd',run/'stage0'],text=True)
if not any('libcudart' in line and 'gpgpu-sim' in line for line in libraries.splitlines()):
    raise RuntimeError('Test must resolve libcudart to the simulator: '+libraries)
config=(SIM/'configs/tested-cfgs/SM90_H100/gpgpusim.config').read_text()
# A single SM isolates shared resource occupancy; all capacities/timing stay H100.
config=config.replace('-gpgpu_n_clusters 132','-gpgpu_n_clusters 1')
config=config.replace('-gpgpu_n_chiplet_partition 2','-gpgpu_n_chiplet_partition 1')
if a.concurrent:
    config+='\n-gpgpu_concurrent_kernel_sm 1\n'
(run/'gpgpusim.config').write_text(config)
shutil.copy2(SIM/'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt',run)
env=os.environ.copy()
for key in ['PTX_SIM_USE_PTX_FILE','PTX_SIM_KERNELFILE','CUOBJDUMP_SIM_FILE']:
    env.pop(key,None)
env['PTX_SIM_MODE_FUNC']='0'
tests=['launch','work','modules','parameters','shared','normal','assert','trap'] if a.test=='all' else [a.test]
records=[]
for test in tests:
    cmd=['/usr/bin/time','-v','-o',str(run/(test+'.time')),str(run/'stage0'),test]
    commands.append(cmd)
    with (run/(test+'.log')).open('w') as log:
        result=subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
    content=(run/(test+'.log')).read_text()
    expected_failure=test in ['assert','trap']
    passed=(result.returncode!=0 and ('__assertfail at' in content if test=='assert' else 'trap at' in content)) if expected_failure else (result.returncode==0 and 'PASS' in content)
    records.append(dict(test=test,returncode=result.returncode,passed=passed))
    print(records[-1],flush=True)
metadata=dict(commands=commands,libraries=libraries,tests=records,
 config_sha256=hashlib.sha256((run/'gpgpusim.config').read_bytes()).hexdigest(),
 config_changes=['gpgpu_n_clusters:132->1','gpgpu_n_chiplet_partition:2->1']+(['gpgpu_concurrent_kernel_sm:1'] if a.concurrent else []),
 outer=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 inner=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SIM,text=True).strip(),
 cuda=subprocess.check_output([cuda/'bin/nvcc','--version'],text=True))
(run/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
raise SystemExit(0 if all(r['passed'] for r in records) else 1)
