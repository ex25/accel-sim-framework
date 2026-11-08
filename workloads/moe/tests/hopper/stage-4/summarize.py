"""Assert saved numerical and event evidence. Does not manufacture expectations."""
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

root=Path(sys.argv[1]).resolve()
checks=0

def check(ok):
    global checks
    assert ok
    checks+=1

runs={}
for dirname in ('final-debug','delay200','regression-stage2','regression-stage3'):
    run=root/dirname
    meta=json.loads((run/'run.json').read_text())
    check(all(t['passed'] for t in meta['tests']))
    values=[]
    for test in meta['tests']:
        name=test['test']; text=(run/(name+'.log')).read_text()
        check('mode=performance simulation' in text)
        check('gpgpu-sim' in meta['libraries'] and 'debug' in meta['libraries'])
        timing=(run/(name+'.time')).read_text()
        stamps=defaultdict(list)
        for key,value in re.findall(r'HOST_TIME (\w+) ([\d.]+)',text):
            stamps[key].append(float(value))
        values.append(dict(test=name,
            device_cycles=[int(v) for v in re.findall(r'^gpu_sim_cycle\s*=\s*(\d+)',text,re.M)],
            register_wall_s=stamps['registration_done'][0]-stamps['main'][0],
            kernel_wall_s=[b-a for a,b in zip(stamps['kernel_begin'],stamps['kernel_done'])],
            comparison_wall_s=[b-a for a,b in zip(stamps['compare_begin'],stamps['compare_end'])],
            rss_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',timing)[1]),
            elapsed=re.search(r'Elapsed \(wall clock\) time .*: (.+)',timing)[1],
            polls=[int(v) for v in re.findall(r'queries=(\d+)',text)]))
    runs[dirname]=values

for name in ('local','remote'):
    base=next(x for x in runs['final-debug'] if x['test']==name)
    slow=next(x for x in runs['delay200'] if x['test']==name)
    check(all(b>a for a,b in zip(base['device_cycles'],slow['device_cycles'])))

events={}
for name in ('events-local','events-hint','events-remote','events-transport129'):
    log=(root/name/'events.log').read_text()
    check('exited normally' in log and 'PASS' in log)
    check('Python Exception' not in log and 'gdb.error' not in log)
    rows=[json.loads(line[4:]) for line in log.splitlines() if line.startswith('OBS ')]
    queries=[r for r in rows if r['kind']=='query']
    changes=[r for r in rows if r['kind']=='state']
    initial=[r for r in rows if r['kind']=='init']
    check(len(initial)==(8 if 'remote' in name or 'transport' in name else 4))
    for r in initial:
        check(r['phase']==0 and r['tx']==0)
        check(r['pending']==(1 if 'remote' in name or 'transport' in name else 2))
    check({r['predicate_encoding'] for r in queries}=={0,1})
    for q in queries:
        check(q['predicate_encoding']==int((q['phase']&1)==q['parity']))
        # The branch may issue only after the collector's writeback event.
        wb=next(r for r in rows[rows.index(q)+1:] if r['kind']=='writeback'
                and r['sm']==q['sm'] and r['warp']==q['warp'] and r['pc']==q['pc'])
        check(wb['cycle']>q['cycle'])
    if name in ('events-local','events-hint'):
        check(len(changes)==24)  # 2 arrivals * 3 rounds * 2 CTAs * 2 kernels
        groups=defaultdict(list)
        for r in changes: groups[(r['uid'],tuple(r['cta']))].append(r)
        check(len(groups)==4)
        for group in groups.values():
            for round in range(3):
                a,b=group[round*2:round*2+2]
                check(a['before']==dict(pending=2,tx=0,phase=round))
                check(a['after']==dict(pending=1,tx=0,phase=round))
                check(b['before']==dict(pending=1,tx=0,phase=round))
                check(b['after']==dict(pending=2,tx=0,phase=round+1))
        for q in queries:
            check(q['rank']==0 and q['address']==1024)
    else:
        sends=[r for r in rows if r['kind']=='remote_send']
        check(len(sends)==len(changes)==4)
        for change in changes:
            check(change['op']=='remote-arrive' and change['rank']==1)
            check(change['address']==0x01000400)
            check(change['before']==dict(pending=1,tx=0,phase=0))
            check(change['after']==dict(pending=1,tx=0,phase=1))
            send=next(r for r in sends if r['uid']==change['uid'] and
                      r['source'][0]==change['cluster'][0])
            check(change['cycle']>=send['cycle']+send['latency'])
            for q in queries:
                if q['uid']!=change['uid'] or q['cta'][0]!=change['cluster'][0]:continue
                if q['rank']==0: check(q['phase']==0 and q['predicate_encoding']==1)
                elif q['predicate_encoding']==0:check(q['cycle']>=change['cycle'])
        check(all(s['latency']==(129 if '129' in name else 29) for s in sends))
    events[name]=dict(initializations=len(initial),state_changes=len(changes),queries=len(queries),
                      true_queries=sum(q['predicate_encoding']==0 for q in queries),
                      sends=[r for r in rows if r['kind']=='remote_send'],
                      transitions=[r for r in changes if r['after']['phase']!=r['before']['phase']])

result=dict(assertions=checks,runs=runs,events=events)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS saved evidence assertions=',checks)
