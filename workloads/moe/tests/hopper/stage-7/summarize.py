#!/usr/bin/env python3
"""Match SIM_KERNEL device intervals to the frozen per-launch ncu/nsys records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MOE = ROOT / 'workloads/moe'
sys.path.insert(0, str(HERE.parent / 'stage-2'))
from read_forward import load


def write_timing_table(run, result):
    """Keep the two kernel intervals visibly separate in the human table."""
    fields = ['case', 'launch_index', 'role', 'module', 'start_cycle',
              'end_cycle', 'execution_cycles', 'execution_us',
              'simulation_interval_cycles', 'simulation_interval_us',
              'outside_cta_interval_cycles', 'ncu_us',
              'execution_vs_ncu_percent', 'host_kernel_wall_seconds']
    with (run/'kernel-times.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result['rows']:
            writer.writerow(dict(
                case=result['case'], launch_index=row['launch_index'],
                role=row['role'], module=row['module'],
                start_cycle=row['start_cycle'], end_cycle=row['end_cycle'],
                execution_cycles=row['cycles'], execution_us=row['sim_ms']*1000,
                simulation_interval_cycles=row['simulation_interval_cycles'],
                simulation_interval_us=row['simulation_interval_ms']*1000,
                outside_cta_interval_cycles=row['outside_cta_interval_cycles'],
                ncu_us=row['ncu_ms']*1000,
                execution_vs_ncu_percent=row['ncu_relative_error']*100,
                host_kernel_wall_seconds=row['simulation_wall_seconds']))
    lines = [f"# {result['case']} 计时", '',
             '执行时间统一为首个 CTA 发出至 kernel 完成；模拟区间还包含启动等待和边界周期。两列不能混用。',
             'H100 单核来自 ncu；与完整前向的无 profiler CUDA event 基线分开列出。', '',
             '| 调用 | 算子 | 模块 | 执行周期 | 执行 µs | 含启动等待的模拟周期 | 模拟区间 µs | 区间外加周期 | H100 ncu µs |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        lines.append(f"| {row['launch_index']} | {row['role']} | {row['module']} | "
                     f"{row['cycles']} | {row['sim_ms']*1000:.3f} | "
                     f"{row['simulation_interval_cycles']} | {row['simulation_interval_ms']*1000:.3f} | "
                     f"{row['outside_cta_interval_cycles']} | {row['ncu_ms']*1000:.3f} |")
    timing = result['forward_timing']
    lines += ['', '完整前向从前后两个 event 的设备周期戳直接相减，不由 kernel 时间求和。']
    if timing['simulation_end_to_end_ms'] is not None:
        lines += [f"模拟前向：{timing['simulation_end_to_end_ms']:.6f} ms，"
                  f"{timing['simulation_event_elapsed_cycles']} cycles。",
                  f"H100 首次前向：{timing['hardware_first_forward_cuda_event_ms']:.6f} ms；"
                  f"预热后：{timing['hardware_warm_cuda_event_ms']:.6f} ms。",
                  f"原始比值差：相对首次 {timing['event_vs_hardware_first_percent']:+.2f}%，"
                  f"相对预热后 {timing['event_vs_hardware_warm_percent']:+.2f}%。",
                  '模拟器未建模显式 DMA 耗时和真实主机提交间隙；上述差额包含模型覆盖范围差异。',
                  f"主机执行前向耗时：{timing['host_forward_wall_seconds']:.3f} s。"]
    else:
        lines.append(timing['simulation_end_to_end_status'])
    if result['completed']:
        lines += [f"单核执行时间求和：{result['kernel_sum_sim_ms']:.6f} ms；"
                  f"含启动等待的单核区间求和：{result['kernel_sum_simulation_interval_ms']:.6f} ms。"
                  '两项均保留为求和值，不命名为端到端时间。']
    (run/'timing.md').write_text('\n'.join(lines) + '\n')


def summarize(run):
    manifest = json.loads((run / 'run.json').read_text())
    name = manifest['workload']
    log = (run / 'stdout.log').read_text()
    modules = json.loads((MOE/'artifacts/workloads.json').read_text())['modules']
    forward = load(MOE/'artifacts/cases'/name/'forward.bin')
    reference = next(c for c in json.loads((MOE/'evidence/hardware-counters.json').read_text())['cases'] if c['case'] == name)
    nsys = next((c for c in json.loads((MOE/'evidence/kernel-times.json').read_text())['cases'] if c['case'] == name), None)
    baseline = next(c for c in json.loads((MOE/'evidence/performance-baseline.json').read_text())['cases'] if c['case'] == name)
    events = []
    previous_dram = [0, 0]
    previous_l2 = {}
    markers = list(re.finditer(r'^SIM_KERNEL (.*)$', log, re.M))
    config = (run/'gpgpusim.config').read_text()
    def option(name):
        return int(re.search(r'^-' + name + r'\s+(\d+)', config, re.M)[1])
    atom = option('gpgpu_dram_buswidth') * option('gpgpu_dram_burst_length') * option('gpgpu_n_mem_per_ctrlr')
    for i, marker in enumerate(markers):
        e = dict(pair.split('=', 1) for pair in marker[1].split())
        stats = log[marker.end():markers[i+1].start() if i+1 < len(markers) else len(log)]
        # These frozen forwards execute kernels sequentially. Preserve the
        # simulator's interval as well as the first-CTA execution interval:
        # gpu_sim_cycle also includes the configured pre-CTA launch delay.
        e['simulation_interval_cycles'] = int(re.search(
            r'^gpu_sim_cycle = (\d+)', stats, re.M)[1])
        # DRAM totals are global; L2 breakdown accumulates per printed stream
        # key. Include setup_anchor when advancing both sets of checkpoints.
        stream = int(re.search(r'^kernel_stream_id = (\d+)', stats, re.M)[1])
        dram = [int(re.search(r'total dram ' + kind + r' = (\d+)', stats)[1])
                for kind in ('reads', 'writes')]
        sectors = {'read': 0, 'write': 0}
        for kind, status, count in re.findall(r'L2_cache_stats_breakdown\[(\w+)\]\[(\w+)\] = (\d+)', stats):
            if status not in ('HIT', 'HIT_RESERVED', 'MISS', 'SECTOR_MISS'):
                continue
            if kind in ('GLOBAL_ACC_R', 'CHIPLET_ACC_R'):
                sectors['read'] += int(count)
            elif kind in ('GLOBAL_ACC_W', 'CHIPLET_ACC_W'):
                sectors['write'] += int(count)
        e['kernel_stream_id'] = stream
        e['l2_stream_cumulative_sector_bytes'] = {
            kind: count * 32 for kind, count in sectors.items()}
        if stream == (1 << 64) - 1:
            # cache_stats::print_stats uses -1 to print every stream. These
            # combined totals cannot advance any individual stream checkpoint.
            delta = {'read': None, 'write': None}
            e['l2_previous_stream_cumulative_sector_bytes'] = None
            e['l2_attribution_error'] = (
                'kernel_stream_id is UINT64_MAX: the log combines all stream '
                'keys, so this kernel L2 increment cannot be attributed.')
        else:
            prior = previous_l2.get(stream, {'read': 0, 'write': 0})
            delta = {kind: (count - prior[kind]) * 32
                     for kind, count in sectors.items()}
            assert all(count >= 0 for count in delta.values()), (stream, delta)
            e['l2_previous_stream_cumulative_sector_bytes'] = {
                kind: count * 32 for kind, count in prior.items()}
            e['l2_attribution_error'] = None
            previous_l2[stream] = sectors
        e['traffic'] = dict(dram_read_bytes=(dram[0]-previous_dram[0])*atom,
                            dram_write_bytes=(dram[1]-previous_dram[1])*atom,
                            l2_data_read_sector_bytes=delta['read'],
                            l2_data_write_sector_bytes=delta['write'])
        previous_dram = dram
        if 'setup_anchor' not in e['entry']:
            events.append(e)
    rows = []
    for i, (e, launch, ref) in enumerate(zip(events, forward, reference['launches'])):
        assert e['entry'] == modules[launch['module']]['entry'] == ref['entry']
        for key, expected in (('grid', launch['shape'][:3]), ('block', launch['shape'][3:6]), ('cluster', launch['shape'][6:9])):
            assert tuple(map(int, e[key].split(','))) == tuple(expected), (i, key, e)
        cycles = int(e['end']) - int(e['start'])
        ms = cycles / float(e['core_MHz']) / 1000
        ref_ms = ref['metrics']['gpu__time_duration.sum'] / 1e6
        row = dict(launch_index=i, module=launch['module'], entry=e['entry'],
                   role=ref['role'], grid=e['grid'], block=e['block'], cluster=e['cluster'],
                   shared_bytes=int(e['shared']), dynamic_shared_bytes=launch['shape'][9],
                   start_cycle=int(e['start']),
                   end_cycle=int(e['end']), cycles=cycles, core_MHz=float(e['core_MHz']),
                   simulation_interval_cycles=e['simulation_interval_cycles'],
                   simulation_interval_ms=e['simulation_interval_cycles']/float(e['core_MHz'])/1000,
                   outside_cta_interval_cycles=e['simulation_interval_cycles']-cycles,
                   sim_ms=ms, ncu_ms=ref_ms, ncu_relative_error=ms/ref_ms-1,
                   wgmma=int(e['wgmma']), tensor_math_ops=int(e['wgmma'])*262144,
                   simulation_wall_seconds=float(e['wall_end'])-float(e['wall_start']),
                   traffic=e['traffic'],
                   kernel_stream_id=e['kernel_stream_id'],
                   l2_stream_cumulative_sector_bytes=e['l2_stream_cumulative_sector_bytes'],
                   l2_previous_stream_cumulative_sector_bytes=e['l2_previous_stream_cumulative_sector_bytes'],
                   l2_attribution_error=e['l2_attribution_error'],
                   hardware_traffic=dict(dram_read_bytes=ref['metrics']['dram__bytes_read.sum'],
                                         dram_write_bytes=ref['metrics']['dram__bytes_write.sum'],
                                         l2_read_bytes=ref['metrics']['lts__t_sectors_op_read.sum']*32,
                                         l2_write_bytes=ref['metrics']['lts__t_sectors_op_write.sum']*32),
                   hardware_tensor_math_ops=ref['metrics']['sm__ops_path_tensor_src_bf16_dst_fp32.sum'])
        if nsys:
            r = nsys['launches'][i]
            assert r['entry'] == e['entry']
            row.update(nsys_ms=r['measured_mean_ns']/1e6,
                       nsys_relative_error=ms/(r['measured_mean_ns']/1e6)-1)
        rows.append(row)
    expected = len(forward)
    assert len(events) <= expected
    completed = len(events) == expected and manifest.get('exit_code') == 0 and 'MOE_ROI_END' in log
    result = dict(case=name, completed=completed, launches_completed=len(events),
                  summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  launches_expected=expected, rows=rows,
                  registered_frozen_entries=sorted(set(re.findall(r'__cudaRegisterFunction (moe_kernel_\w+)', log))),
                  numerical_comparison=json.loads((run/'comparison.json').read_text()) if (run/'comparison.json').exists() else None,
                  forward_timing=dict(
                      hardware_source='workloads/moe/evidence/performance-baseline.json',
                      hardware_first_forward_cuda_event_ms=baseline['first_forward_cuda_event_ms'],
                      hardware_warm_cuda_event_ms=baseline['warm_cuda_event_ms'],
                      hardware_scope='CUDA events around the complete frozen forward; initialization and final output download excluded. Warm result is the median of three ten-forward group means after three warmup forwards.',
                      simulation_end_to_end_ms=None,
                      simulation_end_to_end_status='Not measured: the frozen launcher disables CUDA events in simulation mode. Host gaps and explicit DMA duration are not modeled.'),
                  timing_scope='First CTA admission to kernel completion; excludes host and launch queue gaps. Explicit DMA duration is not modeled.',
                  simulation_interval_scope='Per-kernel gpu_sim_cycle in these sequential runs; includes configured pre-CTA launch delay and interval boundaries. Hardware timestamp alignment is not established.',
                  traffic_scope='Sequential kernels: global DRAM totals differenced per completion, times configured burst bytes. L2 GLOBAL/CHIPLET data read/write sector totals differenced against the previous checkpoint with the same printed kernel_stream_id, including setup_anchor. UINT64_MAX prints all keys, so its kernel L2 increment is null with an attribution error and does not advance individual stream histories. Raw cumulative totals and stream keys are retained. Auxiliary miss bookkeeping and instruction fetches excluded.',
                  ncu_conditions=json.loads((MOE/'evidence/hardware-counters.json').read_text())['conditions'])
    if completed:
        before, roi = log.split('MOE_ROI_BEGIN', 1)
        roi = roi.split('MOE_ROI_END', 1)[0]
        checkpoints = r'^gpu_tot_sim_cycle\s*=\s*(\d+)'
        roi_cycles = (int(re.findall(checkpoints, roi, re.M)[-1]) -
                      int(re.findall(checkpoints, before, re.M)[-1]))
        span = rows[-1]['end_cycle'] - rows[0]['start_cycle']
        result['forward_timing'].update(
            simulation_roi_cycles=roi_cycles,
            simulation_roi_ms=roi_cycles/rows[0]['core_MHz']/1000,
            simulation_roi_scope='Cumulative GPU-cycle checkpoint difference across the forward ROI; includes modeled launch delays. Explicit DMA duration and host gaps are not modeled, so this is not hardware-equivalent end-to-end time.',
            simulation_device_span_cycles=span,
            simulation_device_span_ms=span/rows[0]['core_MHz']/1000,
            simulation_device_span_scope='First kernel first CTA to last kernel completion, including modeled inter-kernel gaps; excludes the first launch delay. This is not complete-forward CUDA-event time.')
        sim = sum(r['sim_ms'] for r in rows)
        ref = sum(r['ncu_ms'] for r in rows)
        result.update(kernel_sum_sim_ms=sim, kernel_sum_ncu_ms=ref,
                      kernel_sum_simulation_interval_ms=sum(r['simulation_interval_ms'] for r in rows),
                      kernel_sum_ncu_relative_error=sim/ref-1)
        result['major_calls'] = [r['launch_index'] for r in rows if r['module']==16 or 'gating' in r['role'] or r['ncu_ms']/ref >= 0.05]
        if nsys:
            ref = sum(r['nsys_ms'] for r in rows)
            result.update(kernel_sum_nsys_ms=ref, kernel_sum_nsys_relative_error=sim/ref-1)
    if (run/'event-timing.json').exists():
        event = json.loads((run/'event-timing.json').read_text())
        assert completed
        assert event['start_cycle'] <= rows[0]['start_cycle']
        assert event['end_cycle'] >= rows[-1]['end_cycle']
        result['forward_timing'].update(
            simulation_end_to_end_ms=event['device_elapsed_ms'],
            simulation_end_to_end_status='Measured directly from stream-ordered CUDA-event device cycle timestamps.',
            simulation_event_start_cycle=event['start_cycle'],
            simulation_event_end_cycle=event['end_cycle'],
            simulation_event_elapsed_cycles=event['elapsed_cycles'],
            simulation_event_scope=event['scope'],
            host_forward_wall_seconds=event['host_forward_wall_seconds'],
            event_vs_hardware_first_percent=(event['device_elapsed_ms']/baseline['first_forward_cuda_event_ms']-1)*100,
            event_vs_hardware_warm_percent=(event['device_elapsed_ms']/baseline['warm_cuda_event_ms']-1)*100,
            comparison_scope='Raw timing differences, including scope differences: simulator does not model explicit DMA duration or real host submission gaps. Hardware first and warm baselines have different cache/warmup conditions; neither ratio alone establishes timing-model accuracy.')
    (run/'stage7-summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    write_timing_table(run, result)
    print(name, 'complete' if completed else 'partial', len(events), '/', expected)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+')
    a = p.parse_args()
    for run in a.runs:
        summarize(run.resolve())
