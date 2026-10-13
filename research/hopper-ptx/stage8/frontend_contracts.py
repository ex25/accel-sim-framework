#!/usr/bin/env python3
"""Validate parser scope and strict functional/detailed failure of new syntax."""
import json
import os
from pathlib import Path
import subprocess
import sys
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

root, run, config = map(Path, sys.argv[1:])
run.mkdir(parents=True, exist_ok=True)
for name in ('gpgpusim.config', 'config_hopper_islip.icnt'):
    (run / name).write_bytes((config / name).read_bytes())
header = '''.version 8.7
.target sm_90a
.address_size 64
.visible .entry readiness_kernel(.param .u64 out) {
 .reg .b32 %r<8>;
 .reg .b64 %rd<4>;
 .reg .pred %p;
 ld.param.u64 %rd0, [out];
 mov.b64 %rd1, 0;
 mov.u32 %r0, 0;
 mov.u32 %r1, -1;
'''
tail = '\n st.global.u32 [%rd0], 1;\n ret;\n}\n'
positive = {
    'elect_register': ('elect.sync %r2|%p, %r1;', 'elect.sync'),
    'elect_immediate': ('elect.sync %r2|%p, 0xffffffff;', 'elect.sync'),
    'prefetch_generic': ('prefetch.tensormap [%rd1];', 'prefetch.tensormap'),
    'prefetch_param': ('prefetch.param.tensormap [%rd1];', 'prefetch.tensormap'),
    'prefetch_const': ('prefetch.const.tensormap [%rd1];', 'prefetch.tensormap'),
    'init_fence': ('fence.mbarrier_init.release.cluster;', 'fence.mbarrier_init'),
    'stmatrix': ('stmatrix.sync.aligned.x4.m8n8.shared.b16 [%r0], {%r2,%r3,%r4,%r5};', 'stmatrix.'),
    'tma_store': ('cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%rd1, {%r0,%r0,%r0,%r0,%r0}], [%r0];', 'cp.async.bulk.tensor.5d'),
    'bulk_commit': ('cp.async.bulk.commit_group;', 'cp.async.bulk.commit_group'),
    'bulk_wait': ('cp.async.bulk.wait_group.read 0;', 'cp.async.bulk.wait_group.read'),
    'grid_wait': ('griddepcontrol.wait;', 'griddepcontrol.wait'),
    'setmaxnreg': ('setmaxnreg.dec.sync.aligned.u32 24;', 'setmaxnreg register-pool'),
}
negative = {
    'elect_modifier': ('elect.sync.aligned %r2|%p, %r1;', 'unsupported Hopper opcode/modifier'),
    'elect_destination': ('elect.sync %r2|%r3, %r1;', 'elect.sync destination types'),
    'elect_missing_predicate': ('elect.sync %r2, %r1;', 'elect.sync requires'),
    'elect_mask_width': ('elect.sync %r2|%p, %rd1;', 'elect.sync membermask'),
    'prefetch_modifier': ('prefetch.param.tensormap.L2 [%rd1];', 'unsupported Hopper opcode/modifier'),
    'prefetch_unbracketed': ('prefetch.tensormap %rd1;', 'prefetch.tensormap requires'),
    'prefetch_width': ('prefetch.tensormap [%r0];', 'prefetch.tensormap address register'),
    'fence_modifier': ('fence.mbarrier_init.acquire.cluster;', 'unsupported Hopper opcode/modifier'),
    'fence_operand': ('fence.mbarrier_init.release.cluster %r0;', 'fence.mbarrier_init requires'),
    'stmatrix_modifier': ('stmatrix.sync.aligned.x2.m8n8.shared.b16 [%r0], {%r2,%r3};', 'unsupported Hopper opcode/modifier'),
    'stmatrix_tuple': ('stmatrix.sync.aligned.x4.m8n8.shared.b16 [%r0], {%r2,%r3};', 'stmatrix.x4 requires'),
    'stmatrix_width': ('stmatrix.sync.aligned.x4.m8n8.shared.b16 [%r0], {%r2,%r3,%r4,%rd1};', 'stmatrix sources'),
    'store_rank': ('cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%rd1, {%r0,%r0,%r0,%r0}], [%r0];', 'TMA store requires'),
    'store_coords': ('cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%rd1, {%r0,%r0,%r0,%r0,%rd1}], [%r0];', 'TMA store coordinates'),
    'bulk_commit_operand': ('cp.async.bulk.commit_group %r0;', 'commit_group takes no operands'),
    'bulk_wait_register': ('cp.async.bulk.wait_group.read %r0;', 'wait_group.read immediate zero only'),
    'bulk_wait_modifier': ('cp.async.bulk.wait_group.invalid 0;', 'unsupported TMA opcode/modifier'),
    'grid_modifier': ('griddepcontrol.invalid;', 'unsupported Hopper opcode/modifier'),
    'grid_operand': ('griddepcontrol.wait %r0;', 'griddepcontrol.wait takes no operands'),
}
env = {**os.environ, 'LD_LIBRARY_PATH': str(Path(os.environ.get(
    'SIM_RUNTIME', '/tmp/hopper-stage3-build/libcudart.so')).resolve().parent)}


def command(args, log):
    with (run / log).open('w') as output:
        result = subprocess.run(args, cwd=run, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=45)
    return result.returncode, (run / log).read_text()


results = []
for accepted, cases in ((True, positive), (False, negative)):
    for name, (instruction, diagnostic) in cases.items():
        fixture = run / (name + '.ptx')
        fixture.write_text(header + instruction + tail)
        code, text = command([str(root / 'research/hopper-ptx/stage3/frontend_probe'),
                              str(fixture), 'readiness_kernel'], name + '-parser.log')
        if accepted:
            if code or 'FRONTEND status=PASS' not in text:
                raise ValueError(f'{name}: parser failed; see complete log')
            for mode in ('functional', 'detailed'):
                code, text = command([str(root / 'research/hopper-ptx/stage8/readiness_execution_probe'),
                                      str(fixture), mode, diagnostic], name + '-' + mode + '.log')
                if code or 'STRICT_EXECUTION_RESULT status=PASS' not in text:
                    raise ValueError(f'{name}: {mode} did not reject execution before store')
        elif not code or diagnostic not in text:
            raise ValueError(f'{name}: malformed syntax was not rejected with expected diagnostic')
        results.append({'name': name, 'parser': 'accepted' if accepted else 'rejected',
                        'execution': 'strict failure in both modes; guard unchanged' if accepted else 'not launched'})

# Repeated label names in sibling asm scopes must resolve to their own symbols.
scoped = run / 'scoped_labels.ptx'
scoped.write_text(header + '''
 { bra.uni local;
   mov.u32 %r2, 99;
 local: mov.u32 %r2, 1;
 }
 { bra.uni local;
   mov.u32 %r2, 99;
 local: add.u32 %r2, %r2, 2;
 }
 mov.u32 %r6, %tid.x;
 setp.eq.u32 %p, %r6, 0;
 @%p st.global.u32 [%rd0], %r2;
 ret;
}
''')
for mode in ('functional', 'detailed'):
    code, text = command([str(root / 'research/hopper-ptx/stage8/readiness_execution_probe'),
                          str(scoped), mode, 'scoped_labels'], 'scoped_labels-' + mode + '.log')
    if code or 'SCOPED_LABEL_RESULT status=PASS' not in text:
        raise ValueError(f'scoped labels: {mode} result failed')
undefined = run / 'undefined_label.ptx'
undefined.write_text(header + 'bra.uni missing_label;' + tail)
code, text = command([str(root / 'research/hopper-ptx/stage3/frontend_probe'),
                      str(undefined), 'readiness_kernel'], 'undefined_label-parser.log')
if not code or 'does not appear in assembly code' not in text:
    raise ValueError('undefined label was not rejected')
result = {'status': 'PASS', 'scope': 'parser only; execution unsupported',
          'positive_cases': len(positive), 'negative_cases': len(negative),
          'strict_execution_cases': 2 * len(positive), 'scoped_label_modes': 2,
          'undefined_label_rejected': True, 'cases': results}
(run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('FRONTEND_CONTRACTS status=PASS positive=' + str(len(positive)) +
      ' negative=' + str(len(negative)) + ' strict_execution=' + str(2 * len(positive)))
