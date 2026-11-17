#!/usr/bin/env python3
"""Compare completed Qwen64 forwards with the frozen 24-call audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / 'runs/hopper-stage-7/qwen-code-diagnosis-20261112/all-kernels-audit.json'


def read(run, name, count):
    summary_path = run / 'stage7-summary.json'
    acceptance_path = run / 'independent-acceptance.json'
    summary = json.loads(summary_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    assert summary['case'] == acceptance['case'] == name
    assert summary['completed'] and summary['launches_completed'] == count
    assert acceptance['execution_checks_passed'] and acceptance['launches'] == count
    assert len(summary['rows']) == len(acceptance['rows']) == count
    assert acceptance['numerical']['passed']
    assert acceptance['numerical']['failed_elements'] == 0
    assert acceptance['numerical']['nonfinite_elements'] == 0
    for checked, row in zip(acceptance['rows'], summary['rows']):
        for key in ('launch_index', 'module', 'role', 'ncu_ms'):
            assert checked[key] == row[key], (name, key)
        assert math.isclose(checked['sim_ms'], row['sim_ms'], abs_tol=1e-12)
    return summary, {
        'summary': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        'acceptance': hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
    }


def compare(name, baseline, candidate, count, alignment):
    after, after_hashes = read(candidate, name, count)
    assert len(baseline['rows']) == count
    rows = []
    for index, (old, new) in enumerate(zip(baseline['rows'], after['rows'])):
        for key in ('launch_index', 'module', 'entry', 'role', 'grid', 'block',
                    'cluster', 'shared_bytes', 'ncu_ms'):
            assert old[key] == new[key], (name, index, key)
        assert index == old['launch_index']
        if name == 'qwen64-t4096':
            assert old['nsys_ms'] == new['nsys_ms']
        else:
            assert 'nsys_ms' not in old and 'nsys_ms' not in new
        old_us, new_us, ncu_us = (old['sim_ms'] * 1000,
                                  new['sim_ms'] * 1000,
                                  old['ncu_ms'] * 1000)
        nsys_us = old.get('nsys_ms')
        nsys_us = None if nsys_us is None else nsys_us * 1000
        rows.append(dict(case=name, launch_index=index,
                         paired_launch_index=alignment[index], module=old['module'],
                         role=old['role'], entry=old['entry'],
                         baseline_us=old_us, candidate_us=new_us,
                         ncu_us=ncu_us,
                         baseline_vs_ncu_percent=(old_us / ncu_us - 1) * 100,
                         candidate_vs_ncu_percent=(new_us / ncu_us - 1) * 100,
                         nsys_us=nsys_us,
                         baseline_vs_nsys_percent=(None if nsys_us is None else
                                                   (old_us / nsys_us - 1) * 100),
                         candidate_vs_nsys_percent=(None if nsys_us is None else
                                                    (new_us / nsys_us - 1) * 100)))
    return dict(case=name, baseline=baseline['path'], candidate=str(candidate),
                candidate_hashes=after_hashes,
                rows=rows, baseline_timing=baseline['forward_timing'],
                candidate_timing=after['forward_timing'])


def value(number, signed=False):
    if number is None:
        return '—'
    return f'{number:+.2f}' if signed else f'{number:.3f}'


def markdown(results):
    lines = ['# Qwen64 修复前后逐调用设备时间', '',
             '单位为 µs；误差 = 模拟 / 同调用硬件时间 − 1。'
             ' 128 无可用 nsys 逐核样本。',
             '各模拟时间为首 CTA 开始至最后 CTA 完成；'
             '不含核外主机提交空档。', '']
    for result in results:
        lines.extend([f"## {result['case']}", '',
                      '| 调用 | 对应序号 | 模块 | 算子 | 修前 | 修后 | ncu | 修前/ncu % | 修后/ncu % | nsys | 修前/nsys % | 修后/nsys % |',
                      '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for row in result['rows']:
            lines.append('| {launch_index} | {pair} | {module} | {role} | {baseline_us:.3f} | '
                         '{candidate_us:.3f} | {ncu_us:.3f} | '
                         '{before_ncu} | {after_ncu} | {nsys} | '
                         '{before_nsys} | {after_nsys} |'.format(
                             **row,
                             pair=('—' if row['paired_launch_index'] is None
                                   else row['paired_launch_index']),
                             before_ncu=value(row['baseline_vs_ncu_percent'], True),
                             after_ncu=value(row['candidate_vs_ncu_percent'], True),
                             nsys=value(row['nsys_us']),
                             before_nsys=value(row['baseline_vs_nsys_percent'], True),
                             after_nsys=value(row['candidate_vs_nsys_percent'], True)))
        lines.append('')
    lines.extend(['## 主要调用', '',
                  'FC1、FC2、gating、ncu 内核和占比至少 5% 的调用，'
                  '以及按 4096 修后相对 nsys 绝对差额最大的六个普通调用。', '',
                  '| 调用 | 算子 | 修前 | 修后 | ncu | 修后/ncu % | nsys | 修后/nsys % |',
                  '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    long = next(result for result in results if result['case'] == 'qwen64-t4096')
    rows = long['rows']
    ncu_sum = sum(row['ncu_us'] for row in rows)
    special = {row['launch_index'] for row in rows
               if row['module'] == 16 or 'gating' in row['role']
               or row['ncu_us'] >= 0.05 * ncu_sum}
    ordinary = sorted((row for row in rows if row['launch_index'] not in special),
                      key=lambda row: abs(row['candidate_us'] - row['nsys_us']),
                      reverse=True)[:6]
    selected = special | {row['launch_index'] for row in ordinary}
    for row in rows:
        if row['launch_index'] not in selected:
            continue
        lines.append('| {launch_index} | {role} | {baseline_us:.3f} | '
                     '{candidate_us:.3f} | {ncu_us:.3f} | {ncu} | '
                     '{nsys_us:.3f} | {nsys} |'.format(
                         **row, ncu=value(row['candidate_vs_ncu_percent'], True),
                         nsys=value(row['candidate_vs_nsys_percent'], True)))
    lines.extend(['', '## 范围', '',
                  'ROI 与直接 CUDA event 设备跨度分别保留在冻结审计和候选 '
                  'stage7-summary.json；它们不等于单核时间求和。'
                  '显式 DMA 和真实主机提交空档未由该模拟计时表测得。'
                  '进程墙钟与峰值 RSS 见各运行包装目录的 .time 和 .compare.time。', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=AUDIT)
    for label in ('candidate-128', 'candidate-4096', 'output-dir'):
        parser.add_argument('--' + label, type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text())
    alignment = audit['alignment_128_4096']
    paired_128 = {row['index128']: row['index4096'] for row in alignment
                  if row['index128'] is not None}
    paired_4096 = {row['index4096']: row['index128'] for row in alignment}
    assert len(paired_128) == 22 and len(paired_4096) == 24
    assert [i for i, pair in paired_4096.items() if pair is None] == [0, 11]
    results = [compare('qwen64-t128', audit['cases']['128'],
                       args.candidate_128.resolve(), 22, paired_128),
               compare('qwen64-t4096', audit['cases']['4096'],
                       args.candidate_4096.resolve(), 24, paired_4096)]
    source = {'audit': str(audit_path),
              'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest()}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = [row for result in results for row in result['rows']]
    with (output / 'per-launch.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'per-launch.md').write_text(markdown(results) + '\n')
    (output / 'per-launch.json').write_text(
        json.dumps(dict(source=source, cases=results), indent=2,
                   ensure_ascii=False) + '\n')
    print(output / 'per-launch.md')


if __name__ == '__main__':
    main()
