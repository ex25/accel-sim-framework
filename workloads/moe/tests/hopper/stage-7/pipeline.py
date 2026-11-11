#!/usr/bin/env python3
"""Check the two ranks of cluster 0 in an original-module expert event log."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def fields(text):
    return dict(re.findall(r'(\w+)=([^\s]+)', text))


def bulk_wait_operands(run):
    # Only the frozen module16 is accepted. Each PTX instruction occupies eight
    # simulator-PC bytes; directives and labels do not. Read wait N from the
    # source, rather than inferring it from the reported cutoff under test.
    source = Path(__file__).resolve().parents[5] / 'workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx'
    metadata = json.loads((run/'run.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata['ptx_sha256']
    operands = {}
    pc = 0
    for line in source.read_text().splitlines():
        line = re.sub(r'^[\w$]+:\s*', '', line.split('//')[0].strip())
        instruction = re.match(r'(?:@!?[\w%.]+\s+)?([a-z][\w.:]*)\s*(.*)', line)
        if not instruction:
            continue
        if instruction[1] == 'cp.async.bulk.wait_group.read':
            operands[pc] = int(instruction[2].strip(' ;'))
        pc += 8
    return operands


def check(run, output=None):
    assert json.loads((run/'run.json').read_text())['passed']
    log = run/'expert.log'
    contexts = {}
    with log.open() as stream:
        for line in stream:
            if line.startswith('PTX_CONTEXT '):
                row = fields(line)
                if row['cluster'] == '0,0,0':
                    contexts[row['uid']] = row
    assert {row['rank'] for row in contexts.values()} == {'0', '1'}
    events = []
    previous = None
    end = None
    last_tensor_ack = 0
    with log.open() as stream:
        for line in stream:
            if line.startswith('PTX_CONTEXT '):
                previous = fields(line)
            if line.startswith('SIM_KERNEL '):
                end = int(fields(line)['end'])
            match = re.search(r'PTX_(TMA|GMMA) (issue|delivered|wait|read|dispatch|writeback|read_done|write_done|group) (.*)', line)
            if not match:
                continue
            row = fields(match[3])
            if match[2] == 'write_done':
                last_tensor_ack = max(last_tensor_ack, int(re.search(r'GPGPU-Sim Cycle (\d+)', line)[1]))
            uid = row.get('uid')
            if match[2] == 'group':
                uid = previous['uid']
            if uid not in contexts:
                continue
            row.update(uid=uid, context=contexts[uid], event=match[2], unit=match[1])
            if 'cycle' not in row:
                row['cycle'] = re.search(r'GPGPU-Sim Cycle (\d+)', line)[1]
            row['cycle'] = int(row['cycle'])
            events.append(row)
    issues = {row['uid']: row for row in events if row['event'] == 'issue'}
    deliveries = [row for row in events if row['event'] == 'delivered']
    waits = [row for row in events if row['event'] == 'wait' and row['success'] == '1']
    wait_operands = bulk_wait_operands(run)
    reads = {}
    writebacks = {}

    def matrix_key(row):
        c = row['context']
        return c['rank'], int(c['warp'])//4, row['seq']

    examples = []
    input_uses = []
    for row in events:
        if row['unit'] != 'GMMA':
            continue
        key = matrix_key(row)
        if row['event'] == 'read':
            reads[key] = row
            for operand in ('a', 'b'):
                base = (int(row[operand], 16) & 0x3fff) << 4
                # Select the latest issued tile for this buffer, including an
                # unfinished issue. Selecting only completed deliveries could
                # incorrectly accept the previous traversal of a ring buffer.
                candidates = [i for i in issues.values()
                              if i['kind'] in ('7', '8')
                              and (i['context']['rank'] == row['context']['rank']
                                   or i['targets'] == '2')
                              and int(i['shared']) <= base
                              < int(i['shared']) + int(i['bytes'])
                              and i['cycle'] <= row['cycle']]
                assert candidates, ('input tile not issued', key, operand, row)
                issued = max(candidates, key=lambda i: i['cycle'])
                completed = [d for d in deliveries if d['uid'] == issued['uid']
                             and d['rank'] == row['context']['rank']]
                assert len(completed) == 1, ('input delivery missing', key, issued)
                delivered = completed[0]
                assert delivered['cycle'] <= row['cycle'], ('input read early', key, delivered, row)
                waited = [w for w in waits if w['context']['rank'] == row['context']['rank']
                          and w['context']['warp'] == row['context']['warp']
                          and (int(w['address']) & 0xffffff) == int(delivered['barrier'])
                          and delivered['cycle'] <= w['cycle'] <= row['cycle']]
                # The trace reports the barrier's current phase, not the wait
                # operand. Match the last delivery on the same barrier/round;
                # multiple operands can complete one phase together.
                matched = []
                for wait in waited:
                    phase_deliveries = [d for d in deliveries
                                        if d['rank'] == delivered['rank']
                                        and d['barrier'] == delivered['barrier']
                                        and delivered['cycle'] <= d['cycle'] <= wait['cycle']]
                    last = max(phase_deliveries, key=lambda d: d['cycle'])
                    if wait['phase'] == last['phase']:
                        matched.append(wait)
                assert matched, ('input barrier round not observed', key, delivered, row)
                waited = matched
                input_uses.append((issued['uid'], row['context']['rank'], base,
                                   row['cycle'], int(row['ready'])))
                if len(examples) < 8:
                    examples.append(dict(rank=key[0], group=key[1], seq=key[2],
                                         operand=operand, tile_uid=issued['uid'],
                                         delivered=delivered['cycle'],
                                         wait=max(w['cycle'] for w in waited),
                                         phase=waited[-1]['phase'],
                                         read=row['cycle'], ready=int(row['ready'])))
        elif row['event'] == 'dispatch':
            assert key in reads and int(reads[key]['ready']) <= row['cycle']
        elif row['event'] == 'writeback':
            assert key in reads and int(reads[key]['ready']) <= row['cycle']
            writebacks[row['uid']] = row['cycle']
    assert reads and all('a' in r and 'b' in r for r in reads.values())
    assert {key[0] for key in reads} == {'0', '1'}
    input_reuse_checks = 0
    for delivered in deliveries:
        replacement = issues[delivered['uid']]
        for uid, rank, base, started, ready in input_uses:
            if (uid != delivered['uid'] and rank == delivered['rank']
                    and int(replacement['shared']) <= base
                    < int(replacement['shared']) + int(replacement['bytes'])
                    and started < delivered['cycle']):
                assert ready <= delivered['cycle'], ('input buffer overwritten during matrix read', uid, delivered, ready)
                input_reuse_checks += 1
    consumers = 0
    for c in contexts.values():
        if c['matrix'] != '2':  # PTX_STMATRIX consumes WGMMA register results.
            continue
        prior = [m for m in contexts.values() if m['matrix'] == '3'
                 and m['rank'] == c['rank'] and m['warp'] == c['warp']
                 and int(m['cycle']) < int(c['cycle'])]
        assert prior
        assert all(m['uid'] in writebacks and writebacks[m['uid']] <= int(c['cycle'])
                   for m in prior), ('stmatrix before matrix writeback', c)
        consumers += 1
    assert consumers
    stores = [row for row in issues.values() if row['kind'] == '9']
    read_done = {row['uid']: row['cycle'] for row in events if row['event'] == 'read_done'}
    write_done = {row['uid']: row['cycle'] for row in events if row['event'] == 'write_done'}
    reused = 0
    reuse_before_ack = 0
    for store in stores:
        uid = store['uid']
        assert uid in read_done, ('missing source read', store)
        assert store['cycle'] <= read_done[uid] <= end
        if int(store['requests']):
            assert uid in write_done, ('missing global acknowledgement', store)
            assert read_done[uid] <= write_done[uid] <= end
        else:
            assert store['oob'] == store['bytes']
        for c in contexts.values():
            if c['rank'] != store['context']['rank'] or c['shared_store'] != '1':
                continue
            if not int(store['shared']) <= int(c['address']) < int(store['shared'])+int(store['bytes']):
                continue
            if int(c['cycle']) > store['cycle']:
                assert int(c['cycle']) >= read_done[uid], ('output reused early', store, c)
                reused += 1
                reuse_before_ack += uid in write_done and int(c['cycle']) < write_done[uid]
    assert len(stores) >= 4 and reused
    assert last_tensor_ack <= end

    # Module16 elects one lane for each output operation. Its observed bulk
    # commits are lane0; keep the lane in the key so all-lane final waits cannot
    # accidentally inherit another lane's pending output groups.
    def owner(context, lane='0'):
        return context['kernel'], context['cta'], context['warp'], lane

    committed = defaultdict(int)
    cutoffs = defaultdict(int)
    pending = defaultdict(dict)
    releases = []
    waits_by_n = defaultdict(int)
    blocking_waits = 0
    for row in events:
        c = row['context']
        if row['event'] == 'issue' and row['kind'] == '9':
            k = owner(c)
            assert int(row['group']) == committed[k], ('output group ownership', row)
            pending[k][row['uid']] = int(row['group'])
        elif row['event'] == 'read_done' and row['kind'] == '9':
            k = owner(c)
            assert row['uid'] in pending[k], ('unmatched source read', row)
            del pending[k][row['uid']]
        elif row['event'] == 'group':
            k = owner(c, row['lane'])
            if row['kind'] == '10':
                # A new committed group exists even if it contains no writes.
                assert row['lane'] == '0', ('unexpected elected lane', row)
                committed[k] += 1
            else:
                assert row['kind'] == '11'
                n = wait_operands[int(c['pc'])]
                cutoffs[k] = max(0, committed[k] - n)
                waits_by_n[n] += 1
                required = [uid for uid, group in pending[k].items()
                            if group < cutoffs[k]]
                release = max([row['cycle']] + [read_done[uid] for uid in required])
                releases.append((k, row, release))
                blocking_waits += bool(required)
            active_groups = set(pending[k].values())
            waiting = any(group < cutoffs[k] for group in active_groups)
            expected = (committed[k], cutoffs[k], len(active_groups), int(waiting))
            actual = tuple(int(row[name]) for name in ('committed', 'cutoff', 'pending', 'waiting'))
            assert actual == expected, ('bulk group state', k, row, expected)
    assert not any(pending.values()), ('unread output groups at kernel completion', pending)
    assert waits_by_n[0] and waits_by_n[1]
    progress_checks = 0
    for k, wait, released in releases:
        # Only traced events are observable here. A later store/commit/fence/
        # matrix context in this warp must not progress through a blocked wait.
        following = [c for c in contexts.values()
                     if owner(c, k[-1]) == k and int(c['cycle']) > wait['cycle']]
        if following:
            progress = min(int(c['cycle']) for c in following)
            assert progress >= released, ('warp progressed before bulk read wait', k, wait, released, progress)
            progress_checks += 1
    matrix_done = defaultdict(list)
    for event in events:
        if event['unit'] == 'GMMA' and event['event'] == 'writeback':
            matrix_done[matrix_key(event)].append(event['cycle'])
    matrix_overlap = sum(any(prior[:2] == key[:2] and int(prior[2]) < int(key[2])
                             and min(cycles) > read['cycle']
                             for prior, cycles in matrix_done.items())
                         for key, read in reads.items())
    input_compute_overlap = sum(any(
        issue['cycle'] < max(cycles) and delivered['cycle'] > reads[key]['cycle']
        for key, cycles in matrix_done.items()
        if key[0] == delivered['rank'])
        for delivered in deliveries for issue in [issues[delivered['uid']]])
    rounds = defaultdict(set)
    for d in deliveries:
        rounds[d['rank'], d['barrier']].add(d['phase'])
    result = dict(passed=True, scope='Original module16, cluster0, both ranks; all selected events checked.',
                  input_deliveries=len(deliveries), successful_waits=len(waits),
                  input_buffer_reuse_checks=input_reuse_checks,
                  logical_matrix_reads=len(reads), stmatrix_consumers=consumers,
                  output_transfers=len(stores), shared_reuse_checks=reused,
                  shared_reuse_before_global_ack=reuse_before_ack,
                  matrix_reads_before_prior_writeback=matrix_overlap,
                  input_deliveries_overlapping_matrix_execution=input_compute_overlap,
                  fully_oob_outputs=sum(s['oob'] == s['bytes'] for s in stores),
                  last_output_ack=max(write_done[s['uid']] for s in stores if int(s['requests'])),
                  last_tensor_ack_all_clusters=last_tensor_ack,
                  kernel_end=end, input_order_examples=examples,
                  input_barrier_phases={f'{rank}:{barrier}': sorted(map(int, phases))
                                        for (rank, barrier), phases in rounds.items()},
                  bulk_wait_counts=dict(waits_by_n), bulk_blocking_waits=blocking_waits,
                  bulk_wait_progress_checks=progress_checks,
                  trace_limits='wait.phase is the observed current phase, not the requested parity; output lane0 is established by the original-module elected-lane trace; only cluster0 contexts are checked.')
    (output or run/'pipeline.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, help='Save separately when checking archived evidence')
    a = p.parse_args()
    check(a.run.resolve(), a.output.resolve() if a.output else None)
