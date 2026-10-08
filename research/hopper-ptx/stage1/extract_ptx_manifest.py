#!/usr/bin/env python3
"""Inventory one PTX kernel and statically classify forms against ptx.l."""
import argparse, collections, hashlib, json, re
from pathlib import Path
TYPE = re.compile(r'(?:[subf](?:8|16|32|64)(?:x2)?|pred)$')
TOKEN = re.compile(r'^(?:@[!%\w.]+\s+)?([a-z][a-zA-Z0-9_.:]*)\s*(.*)$')
def classify(form, lexer):
    base = form.split('.')[0]
    opcodes = set(re.findall(r'^([a-z][a-z0-9]*)\s+TC;[^\n]*return OPCODE;', lexer, re.M))
    known = set(re.findall(r'^\\\.([a-zA-Z0-9]+)\s+TC;', lexer, re.M))
    unknown = [x for x in form.split('.')[1:] if x not in known]
    if base not in opcodes: return 'missing', 'No baseline opcode token in src/cuda-sim/ptx.l'
    if unknown: return 'missing', 'No baseline modifier token: ' + ', '.join(unknown)
    if 'f16x2' in form: return 'parser', 'Lexer maps f16x2 to scalar F16_TYPE (TODO); functional audit pending'
    return 'parser', 'Static lexer/grammar candidate; parse and exact-form semantics unverified'
def extract(ptx, lexer):
    text = ptx.read_text(); entries = re.findall(r'^\.entry\s+(\S+)\(', text, re.M)
    if len(entries) != 1: raise ValueError('Expected exactly one FA3 kernel entry')
    records, total = {}, 0
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.split('//')[0].strip()
        for statement in line.split(';'):
            statement = statement.strip().lstrip('{').strip()
            if not statement or statement.startswith(('.', '}')) or statement.endswith(':'): continue
            match = TOKEN.match(statement)
            if not match: continue
            form, operands = match.groups()
            if form.startswith('param'): continue
            total += 1; status, reason = classify(form, lexer)
            record = records.setdefault(form, {'form': form, 'opcode': form.split('.')[0],
                'modifiers': [x for x in form.split('.')[1:] if not TYPE.fullmatch(x)],
                'types': [x for x in form.split('.')[1:] if TYPE.fullmatch(x)], 'count': 0,
                'lines': [], 'example': statement.rstrip(' }'), 'vector_tuple_lengths': set(),
                'category': status, 'evidence': reason, 'parser': 'source_candidate' if status == 'parser' else 'missing',
                'functional': 'unverified', 'timing': 'unverified', 'validated': False,
                'barrier': form.startswith(('mbarrier.', 'bar.', 'fence.', 'wgmma.wait', 'wgmma.commit')),
                'descriptor': form.startswith(('cp.async.bulk.tensor', 'cp.async.bulk.prefetch.tensor', 'prefetch.tensormap', 'wgmma.mma_async'))})
            record['count'] += 1; record['lines'].append(line_no)
            for vector in re.findall(r'\{([^{}]+)\}', operands): record['vector_tuple_lengths'].add(len(vector.split(',')))
    for item in records.values(): item['vector_tuple_lengths'] = sorted(item['vector_tuple_lengths'])
    counts = collections.Counter(x['category'] for x in records.values())
    return {'ptx_sha256': hashlib.sha256(ptx.read_bytes()).hexdigest(), 'kernel': entries[0],
            'target': re.search(r'^\.target (.+)$', text, re.M)[1], 'instruction_count': total,
            'unique_forms': len(records), 'categories': {x: counts[x] for x in ('missing','parser','functional','timing','validated')},
            'classification_method': 'Static baseline lexer/grammar audit only; no simulator or hardware execution',
            'baseline_lexer_sha256': hashlib.sha256(lexer.encode()).hexdigest(), 'instructions': [records[x] for x in sorted(records)]}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('ptx', type=Path); parser.add_argument('--lexer', type=Path, required=True); parser.add_argument('-o', '--output', type=Path, required=True); args = parser.parse_args(); args.output.write_text(json.dumps(extract(args.ptx, args.lexer.read_text()), indent=2) + '\n')
