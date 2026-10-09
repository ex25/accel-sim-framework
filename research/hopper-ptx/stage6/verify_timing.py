#!/usr/bin/env python3
"""Check actual unit timing and group wait ordering, independently of GEMM."""
from pathlib import Path
import sys
for filename in sys.argv[1:]:
    events=[]
    for line in Path(filename).read_text().splitlines():
        if line.startswith('WGMMA_DETAIL '):
            events.append(dict(part.split('=',1) for part in line.split()[1:]))
    issue={}; unit={}; outstanding={i:set() for i in range(4)}
    previous={}; releases=0; completion=0; deltas=[]; intervals=[]
    for event in events:
        kind=event['event']; warp=int(event['warp']); cycle=int(event['cycle'])
        if kind=='issue':
            uid=int(event['uid']); issue[uid]=event; outstanding[warp].add(uid)
            assert int(event['latency'])==int(event['n'])//2
            assert int(event['ii'])==int(event['n'])//2
        elif kind=='unit':
            uid=int(event['uid']); unit[uid]=cycle
            if warp in previous:
                interval=cycle-previous[warp]
                assert interval >= int(issue[uid]['ii']), (filename,'II',warp,interval)
                intervals.append(interval)
            previous[warp]=cycle
        elif kind=='complete':
            uid=int(event['uid']); outstanding[warp].remove(uid); completion+=1
            elapsed=cycle-unit[uid]; deltas.append(elapsed)
            assert elapsed >= int(issue[uid]['latency']), (filename,'latency',uid,elapsed)
        elif kind=='release':
            limit=int(event['limit']); releases+=1
            assert all(len(pending)<=limit for pending in outstanding.values()), (filename,'early wait',event,outstanding)
    assert len(issue)==completion==12 and releases==8 and len(unit)==12
    assert not any(outstanding.values()) and intervals
    print(f'WAIT_ORDER_RESULT status=PASS file={Path(filename).name} completions={completion} releases={releases} unit_latency_min={min(deltas)} unit_latency_max={max(deltas)} per_warp_ii_min={min(intervals)}')
