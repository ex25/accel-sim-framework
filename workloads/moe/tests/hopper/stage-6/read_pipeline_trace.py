#!/usr/bin/env python3
"""Check resident-input GMMA traces independently of the pipeline layout."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def check(path, count, groups, latency, gmma_latency=64):
    operations, uids, service_end = {}, {}, {}
    epoch = 0
    kernels = []
    for line in path.read_text().splitlines():
        if line.startswith('SIM_KERNEL '):
            row = dict(re.findall(r'(\w+)=([^ ]+)', line))
            assert int(row['wgmma']) == count * groups
            kernels.append(row)
            epoch += 1
            continue
        match = re.match(r'PTX_GMMA (arrive|collected|read|dispatch|writeback) (.*)', line)
        if not match:
            continue
        event, rest = match.groups()
        row = {k: int(v) for k, v in re.findall(r'(cycle|sid|uid|seq|warp|wid|ready|pending)=(\d+)', rest)}
        if event == 'arrive':
            key = (epoch, row['sid'], row['wid'] // 4, row['seq'])
            uids[row['uid']] = key
            op = operations.setdefault(key, dict(arrive={}, collected={}, dispatch={}, writeback={}))
            assert row['warp'] not in op['arrive']
            op['arrive'][row['warp']] = row['cycle']
            continue
        key = uids[row['uid']]
        op = operations[key]
        if event == 'read':
            assert 'read' not in op
            assert len(op['collected']) == 4
            assert row['cycle'] >= max(op['collected'].values())
            resource = key[:2]
            # Each m64n128k16 BF16 operation reads 6144 B. No TMA occurs here.
            end = max(row['cycle'], service_end.get(resource, 0)) + 6144 // 128
            assert row['ready'] == end + latency
            service_end[resource] = end
            op['read'], op['ready'] = row['cycle'], row['ready']
        else:
            assert row['warp'] not in op[event]
            op[event][row['warp']] = row['cycle']
            if event == 'collected':
                assert row['cycle'] >= op['arrive'][row['warp']]
            elif event == 'dispatch':
                assert row['cycle'] >= op['ready']
            else:
                assert row['cycle'] >= op['dispatch'][row['warp']] + gmma_latency
    assert len(kernels) == 2
    assert len(operations) == 2 * count * groups
    gaps, overlaps = [], 0
    for key, op in operations.items():
        assert all(len(op[event]) == 4 for event in ('arrive', 'collected', 'dispatch', 'writeback'))
        if key[-1]:
            previous = operations[(*key[:-1], key[-1] - 1)]
            gaps.append(min(op['dispatch'].values()) - min(previous['dispatch'].values()))
            overlaps += op['read'] < min(previous['dispatch'].values())
    # For one warpgroup the resident inputs and free collectors make overlap
    # observable. No exact hardware issue interval is prescribed by this check.
    if groups == 1:
        assert overlaps > 0
    return dict(passed=True, operations=len(operations),
                shared_bytes=len(operations)*6144,
                early_read_reservations=overlaps,
                dispatch_gap_histogram=dict(sorted(Counter(gaps).items())),
                kernel_cycles=[int(k['end'])-int(k['start']) for k in kernels])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--smem-latency', type=int, default=29)
    a = p.parse_args()
    results = {f'readpipe{n}_wg{g}': check(a.directory/f'readpipe{n}_wg{g}.log',n,g,a.smem_latency)
               for n in (16,64) for g in (1,2)}
    (a.directory/'read-pipeline-validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
