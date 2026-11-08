"""Check saved PTX numerical outcomes and causal TMA events; summarize costs."""
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

root = Path(sys.argv[1]).resolve()
checks = 0

def check(value):
    global checks
    assert value
    checks += 1

def events(text):
    rows = []
    for line in text.splitlines():
        match = re.search(r'PTX_TMA (issue|group|read_done|write_done|delivered|wait|shared_reuse|service) (.*)', line)
        if not match:
            continue
        row = dict(event=match[1])
        for key, value in re.findall(r'(\w+)=([0-9a-f]+)',match[2]):
            row[key] = int(value, 16 if key in ('base','descriptor') else 10)
        if 'cycle' not in row:
            row['cycle'] = int(re.search(r'GPGPU-Sim Cycle (\d+)',line)[1])
        rows.append(row)
    return rows

runs = {}
for dirname in ('final-debug','shared129','dram1194','regression-stage4','regression-stage2'):
    run = root / dirname
    meta = json.loads((run/'run.json').read_text())
    check(all(t['passed'] for t in meta['tests']))
    check('gpgpu-sim' in meta['libraries'] and 'debug' in meta['libraries'])
    records = {}
    for test in meta['tests']:
        name = test['test']
        log = (run/(name+'.log')).read_text()
        check('mode=performance simulation' in log)
        check('PASS' in log and 'MISMATCH' not in log)
        stamps = defaultdict(list)
        for key,value in re.findall(r'HOST_TIME (\w+) ([\d.]+)',log):
            stamps[key].append(float(value))
        timing = (run/(name+'.time')).read_text()
        record = dict(
            cycles=[int(v) for v in re.findall(r'^gpu_sim_cycle\s*=\s*(\d+)',log,re.M)],
            registration_s=stamps['registration_done'][0]-stamps['main'][0],
            kernel_s=[b-a for a,b in zip(stamps['kernel_begin'],stamps['kernel_done'])],
            comparison_s=[b-a for a,b in zip(stamps['compare_begin'],stamps['compare_end'])],
            rss_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',timing)[1]),
            elapsed=re.search(r'Elapsed \(wall clock\) time .*: (.+)',timing)[1])
        rows = events(log)
        if dirname.startswith('regression'):
            records[name] = record
            continue
        grouped = defaultdict(list)
        for row in rows:
            if 'uid' in row:
                grouped[row['uid']].append(row)
        transfers = []
        for uid, group in grouped.items():
            issue = next((r for r in group if r['event']=='issue'),None)
            if issue is None:
                continue
            kind = issue['kind']
            if kind == 4:
                check(issue['requests']==4)  # one 128B warp publication
                read = next(r['cycle'] for r in group if r['event']=='read_done')
                ack = next(r['cycle'] for r in group if r['event']=='write_done')
                check(issue['cycle'] < read < ack)
            elif kind in (7,8):
                delivered = [r for r in group if r['event']=='delivered']
                check(len(delivered)==issue['targets'])
                check(all(r['bytes']==issue['bytes'] and r['phase']==1 for r in delivered))
                check(all(r['cycle']>issue['cycle'] for r in delivered))
                if kind==8:
                    mask = int(name[-1])
                    check({r['rank'] for r in delivered}=={r for r in range(2) if mask&(1<<r)})
                    check(issue['requests']==132)
                    if mask==3:
                        check(delivered[1]['cycle']>delivered[0]['cycle'])
                else:
                    wanted_oob = issue['bytes'] if name.startswith('oob') else issue['bytes']-512 if name.startswith('edge') else 0
                    check(issue['oob']==wanted_oob)
                    check(issue['requests']==4 if name.startswith('oob') else issue['requests']==20 if name.startswith('edge') else issue['requests']==issue['bytes']//32+4)
                # Each successful real query follows a data delivery at that rank.
                for r in delivered:
                    check(any(q['event']=='wait' and q['rank']==r['rank'] and q['success']==1 and q['cycle']>=r['cycle'] for q in rows))
            elif kind==9:
                read = next(r['cycle'] for r in group if r['event']=='read_done')
                ack = next(r['cycle'] for r in group if r['event']=='write_done')
                check(issue['cycle'] < read < ack)
                if name=='output':
                    check(issue['requests']==128)
            transfers.append(dict(issue, events=group))
        if name.startswith('output'):
            # wait1 excludes the newest group although its read is still pending.
            check(any(r['event']=='group' and r['kind']==11 and r['committed']==2 and r['cutoff']==1 and r['pending']==1 and r['waiting']==0 for r in rows))
            check(any(r['event']=='group' and r['kind']==11 and r['cutoff']==2 and r['pending']==1 and r['waiting']==1 for r in rows))
            check(any(r['event']=='shared_reuse' for r in rows))
        if name=='rotate':
            loads=[t for t in transfers if t['kind']==7]
            check(len(loads)==6)
            for i in (0,3):
                a,b,c=loads[i:i+3]
                check(a['base']!=b['base'] and a['base']==c['base'])
                check(a['descriptor']==b['descriptor'] and c['descriptor']!=a['descriptor'])
        record.update(transfers=transfers,
                      successful_queries=sum(r['event']=='wait' and r['success'] for r in rows),
                      queries=sum(r['event']=='wait' for r in rows))
        records[name]=record
    runs[dirname]=records
    if not dirname.startswith('regression'):
        component=(run/'state_component.log').read_text()
        check('ack0=20 ack1=40 PASS' in component and 'ack0=200 ack1=400 PASS' in component)
        check('SASS group adapter PASS (no trace execution)' in component)

for name in ('input32','mask2'):
    a=runs['final-debug'][name];b=runs['shared129'][name]
    check(all(y>x for x,y in zip(a['cycles'],b['cycles'])))
    da=[r['cycle'] for t in a['transfers'] for r in t['events'] if r['event']=='delivered']
    db=[r['cycle'] for t in b['transfers'] for r in t['events'] if r['event']=='delivered']
    check(all(y>x for x,y in zip(da,db)))
# Cold/warm memory states differ between launches: check the response-dependent
# delivery delay directly, and require some total-time increase.
a=runs['final-debug']['input32'];b=runs['dram1194']['input32']
check(any(y>x for x,y in zip(a['cycles'],b['cycles'])))

expert_dir = root / 'expert-addresses'
expert_meta = json.loads((expert_dir/'run.json').read_text())
check(all(t['passed'] for t in expert_meta['tests']))
check('gpgpu-sim' in expert_meta['libraries'] and 'debug' in expert_meta['libraries'])
source = json.loads((expert_dir/'expert-address-source.json').read_text())
ptx = Path(source['source']).read_text()
check(hashlib.sha256(ptx.encode()).hexdigest() == source['sha256'])
check(source['shared_declaration'] in ptx.splitlines())
check(sum(len(sites) for sites in source['site_classes'].values()) == 20)
check({name: len(sites) for name, sites in source['site_classes'].items()} == {
    'shared_to_global': 8, 'expert64_global_to_shared': 8,
    'expert128_multicast': 2, 'expert128_global_to_shared': 2})
check(source['expert64_starts'] == [196608, 200704, 204800, 208896] * 2)
check(all(base % 1024 == 0 for base in source['expert64_starts']))
check(all(offset % 1024 == 0 for offset in (196608, 200704, 204800, 208896)))
check(set(source['fragments']) == {
    'expert64', 'expert64_all_addresses', 'expert_output_all_addresses',
    'expert128', 'expert128_64', 'load64', 'load128', 'load128_64'})
check({name: len(source['fragments'][name]) for name in
       ('expert64_all_addresses', 'expert_output_all_addresses')} == {
    'expert64_all_addresses': 8, 'expert_output_all_addresses': 8})
check('def expert_samples(run):' in
      (Path(__file__).with_name('generate_micro.py')).read_text())
expert_expect = {}
expert_expect['expert64'] = [(0, 0, 196608, 196608, 213680, 64, 0)]
for stage in range(3):
    for rank in range(2):
        barrier = 213632 + stage * 8
        local128 = stage * 32768 + rank * 8192
        expert_expect.setdefault('expert128', []).append(
            (stage, rank, local128, stage * 32768 + 8192, barrier, 128, 0))
        local128_64 = 98304 + stage * 32768
        expert_expect.setdefault('expert128_64', []).append(
            (stage, rank, local128_64, local128_64, barrier, 128, 0))
for name, expected in expert_expect.items():
    log = (expert_dir/(name+'.log')).read_text()
    check('mode=performance simulation' in log)
    check('PASS '+name+' errors=0' in log and 'MISMATCH' not in log)
    samples = [tuple(map(int, row)) for row in re.findall(
        r'EXPERT_SAMPLE test='+name+
        r' stage=(\d+) rank=(\d+) shared_symbol=(\d+) local=(\d+) '
        r'delivered=(\d+) barrier=(\d+) swizzle=(\d+) phase=(\d+)', log)]
    normalized = [(stage, rank, local, delivered, barrier, swizzle, phase)
                  for stage, rank, symbol, local, delivered, barrier, swizzle, phase
                  in samples if symbol == 0]
    check(len(samples) == len(expected) and normalized == expected)
    rows = events(log)
    issues = [r for r in rows if r['event']=='issue' and r.get('kind') in (7,8)]
    loads = [r for r in rows if r['event']=='delivered']
    expected_kind = 7 if name in ('expert64', 'expert128_64') else 8
    expected_issue_count = 1 if name == 'expert64' else 6 if name == 'expert128' else 12
    check(len(issues) == expected_issue_count)
    check(all(r['kind'] == expected_kind and r['oob'] == 0 for r in issues))
    check(all(r['bytes'] == (4096 if name == 'expert64' else 16384)
              and r['requests'] == (132 if name == 'expert64' else 516)
              and r['targets'] == (2 if name == 'expert128' else 1)
              for r in issues))
    check(len(loads) == sum(r['targets'] for r in issues))
    for issue in issues:
        delivered = [r for r in rows if r['event']=='delivered' and r.get('uid')==issue['uid']]
        check(len(delivered) == issue['targets'])
        check(all(r['bytes']==issue['bytes'] and r['cycle']>issue['cycle']
                  for r in delivered))
    stages = 1 if name == 'expert64' else 3
    for rank in range(1 if name == 'expert64' else 2):
        rank_loads = [r for r in loads if r['rank']==rank]
        rank_waits = sorted({(r['cycle'], r['address']) for r in rows
                             if r['event']=='wait' and r.get('success')==1
                             and r.get('rank')==rank})
        check(len(rank_loads) == stages * (1 if name=='expert64' else 2))
        check(len(rank_waits) == stages)
        width = 1 if name == 'expert64' else 2
        for stage in range(stages):
            transfers = rank_loads[stage * width:(stage + 1) * width]
            check(max(r['cycle'] for r in transfers) <= rank_waits[stage][0])
    dump_bytes = 4352 if name == 'expert64' else 33024
    dump_count = 1 if name == 'expert64' else 6
    dumps = list(expert_dir.glob(name+'.stage-*.shared'))
    check(len(dumps) == dump_count and all(path.stat().st_size == dump_bytes
                                           for path in dumps))
result=dict(assertions=checks,runs=runs)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(assertions=checks, tests={k:list(v) for k,v in runs.items()}),indent=2))
