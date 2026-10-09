#!/usr/bin/env python3
"""Verify numerical, trace and trend evidence; keep missing evidence blocked."""
import argparse
import hashlib
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import subprocess
import sys
import numpy as np
from reference import compare, ATOL, RTOL


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reject_bad_outputs(case):
    size = int(np.prod(json.loads((case/'manifest.json').read_text())['shape']))
    with tempfile.TemporaryDirectory(prefix='stage8-contract-') as directory:
        path = Path(directory)/'invalid.f16le'
        for name, data in [('nan',np.full(size,np.nan,dtype='<f2')),
                           ('infinity',np.full(size,np.inf,dtype='<f2')),
                           ('wrong_size',np.zeros(size-1,dtype='<f2')),
                           ('wrong_value',np.full(size,100.,dtype='<f2'))]:
            data.tofile(path)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    compare(case,path,None,None)
            except ValueError:
                pass
            else:
                raise ValueError(f'contract accepted {name}')
    print('CONTRACT_NEGATIVE_RESULT status=PASS cases=nan,infinity,wrong_size,wrong_value')


def main(model, hardware, output):
    results = {'scope':'stage8', 'numerical':[], 'blockers':[], 'architecture_calibration':False}
    layout=json.loads((model/'parameter-layout/result.json').read_text())
    require(layout['status']=='PASS' and layout['positive_cases']==4 and layout['negative_cases']==5 and layout['copy_guard']=='PASS','parameter ABI regression failed')
    counters=json.loads((model/'counter-contracts/result.json').read_text())
    require(counters['status']=='PASS' and len(counters['cases'])==8,'counter validator contracts incomplete')
    results['parameter_abi']=layout
    results['counter_validator_contracts']=counters
    frontend=json.loads((model/'frontend-contracts/result.json').read_text())
    require(frontend['status']=='PASS' and frontend['positive_cases']==12 and
            frontend['negative_cases']==19 and frontend['strict_execution_cases']==24 and
            frontend['scoped_label_modes']==2 and frontend['undefined_label_rejected'],
            'frontend readiness/strict execution contracts incomplete')
    results['frontend_contracts']=frontend
    reject_bad_outputs(model/'inputs/128')
    runs = [(model,'ptx-functional-128',128,None,None)]
    runs += [(model,f'ptx-detailed-{s}',s,1,1) for s in (64,128,256)]
    runs += [(hardware,f'a800-{s}',s,None,None) for s in (64,128,256)]
    runs += [(hardware,'a800-traced-128',128,1,1)]
    for root, name, seq, rows, heads in runs:
        compare(root/f'inputs/{seq}',root/f'{name}.f16le',rows,heads)
        results['numerical'].append(json.loads((root/f'{name}.comparison.json').read_text()))
    a = np.fromfile(model/'ptx-functional-128.f16le',dtype='<f2').astype(np.float64)
    b = np.fromfile(hardware/'a800-128.f16le',dtype='<f2').astype(np.float64)
    require(a.size==b.size==131072,'direct comparison size mismatch')
    delta=np.abs(a-b)
    require(np.isfinite(delta).all() and np.all(delta<=ATOL+RTOL*np.abs(b)),'PTX/A800 disagreement')
    results['ptx_a800_max_abs_difference']=float(delta.max())
    results['ptx_a800_differing_elements']=int(np.count_nonzero(delta))
    times=[]; cycles=[]
    for seq in (64,128,256):
        match=re.search(r'^A800_RESULT (.+)$',(hardware/f'a800-{seq}.log').read_text(),re.M)
        require(match,'missing A800 timing')
        timing=json.loads(match.group(1))
        require(timing['rows']==seq and timing['heads']==8,'A800 timing scope mismatch')
        times.append(timing)
        match=re.search(r'PTX_ATTENTION_RESULT status=PASS mode=detailed seq=\d+ rows=1 heads=1 cycles=(\d+)',(model/f'detailed/attention{seq}.log').read_text())
        require(match,'missing detailed cycles'); cycles.append(int(match.group(1)))
    require(all(x['kernel_us_median']<y['kernel_us_median'] for x,y in zip(times,times[1:])),'non-monotonic A800 sweep')
    require(cycles[0]<cycles[1]<cycles[2],'non-monotonic PTX one-CTA sweep')
    results['a800_full_grid_timings']=times
    results['ptx_one_cta_cycles']=dict(zip((64,128,256),cycles))
    match=re.search(r'A800_TRACE_RESULT status=PASS origin=hardware_capture (.+)',(hardware/'a800-trace-audit.log').read_text())
    require(match,'missing trace audit')
    trace={k:int(v) for k,v in (part.split('=') for part in match.group(1).split())}
    replay=(hardware/'a800-sass-model.log').read_text()
    require('GPGPU-Sim: *** exit detected ***' in replay,'trace replay incomplete')
    scalar=re.search(r'^gpu_tot_sim_insn = (\d+)',replay,re.M)
    require(scalar and int(scalar.group(1))==trace['scalar_instructions'],'trace/replay instruction count mismatch')
    results['a800_trace']=trace
    results['sass_replay_cycles']=int(re.search(r'^gpu_tot_sim_cycle = (\d+)',replay,re.M).group(1))
    for n in (128,176):
        text=(model/f'sass-oracle{n}.log').read_text()
        require(text.count('SASS_ORACLE_RESULT status=PASS')==3 and text.count('SASS_WAIT_RESULT status=PASS')==3 and text.count('SASS_TMA_RESULT status=PASS')==2,'incomplete synthetic oracle')
    require((model/'tma.log').read_text().count('DETAILED_RESULT status=PASS')==4,'TMA regression incomplete')
    require((model/'wgmma-timing-check.log').read_text().count('WAIT_ORDER_RESULT status=PASS')==2,'WGMMA wait check incomplete')
    static=json.loads((model/'frozen-sass.json').read_text())
    require(static['origin']=='compiler_static_sass','incorrect SASS provenance')
    results['static_hopper_sass_instructions']=static['instructions']
    frozen=json.loads((model/'frozen-readiness.json').read_text())
    frozen_ptx=Path(__file__).resolve().parents[1]/'stage1/artifacts/frozen_fwd.ptx'
    require(frozen['ptx_sha256']==hashlib.sha256(frozen_ptx.read_bytes()).hexdigest(),'frozen PTX hash mismatch')
    if frozen['status']!='PASS':
        results['blockers'].append('frozen FA3 PTX parser: '+str(frozen.get('first_failure', 'readiness probe failed')))
    else:
        require(frozen['exit']==0 and frozen['parser_only'] and
                frozen['instructions_including_labels']>0 and
                'FRONTEND status=PASS target=sm_90a' in (model/'frozen-fa3-parser.log').read_text(),
                'missing original frozen frontend success evidence')
    results['frozen_frontend']=frozen
    profiler=json.loads((hardware/'profiler-readiness.json').read_text())
    counter_check=subprocess.run([sys.executable,str(Path(__file__).with_name('check_ncu_csv.py')),str(hardware/'ncu128.csv'),str(hardware/'profiler-csv-validation.json')],capture_output=True,text=True)
    results['hardware_counters']=json.loads((hardware/'profiler-csv-validation.json').read_text())
    if profiler['status']!='PASS' or profiler.get('exit')!=0 or counter_check.returncode:
        results['blockers'].append('A800 performance counters unavailable; actual DRAM/achieved occupancy unverified')
    results['hopper_dynamic_trace_available']=False
    results['limitations']=[
        'No Hopper hardware dynamic trace; GMMA/TMA decoder fields are explicitly synthetic. This validation does not claim hardware cycle accuracy.',
        'Frozen FA3 passed parser readiness only. elect.sync, prefetch.tensormap, init fence, stmatrix, TMA store/bulk groups, grid dependency and setmaxnreg strictly reject execution; complete FA3 functional/detailed execution is outside stage 8.']
    results['status']='BLOCKED' if results['blockers'] else 'PASS'
    output.write_text(json.dumps(results,indent=2)+'\n')
    print('STAGE8_RESULT status='+results['status']+' numerical_cases='+str(len(runs))+' blockers='+str(len(results['blockers'])))
    return 3 if results['blockers'] else 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('model',type=Path); p.add_argument('hardware',type=Path); p.add_argument('output',type=Path)
    args=p.parse_args()
    raise SystemExit(main(args.model,args.hardware,args.output))
