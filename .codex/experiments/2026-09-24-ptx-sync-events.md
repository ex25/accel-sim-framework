# P1 PTX synchronization event smoke

## Objective / hypothesis

- **Objective**：确认 nested GPGPU-Sim 的 `bar.*`/`membar.*` timing path 能产生
  P0-compatible JSONL event，并在 GPU-local completion 后保持 kernel 正常完成。
- **Hypothesis**：在已有 `cais_add` PTX kernel 前插入 `membar.cta; bar.sync 0;`
  后，应该得到每个 warp 的 release/arrival/acquire event，以及一次 CTA
  barrier wakeup；不会改变功能 checksum。
- **Evidence level**：`Verified / simulator`，不是 real H100 hardware。

## Fixed inputs

- Main repository branch/commit：`feat/ptx-sync-events-p0-p5` / `f160e68`。
- Nested GPGPU-Sim branch/commit：`feat/ptx-sync-events-p1-p5` / `ef16060b`。
- Workload source：`experiments/cais_add_ptx/build/control.ptx` 和
  `experiments/cais_add_ptx/build/cais_add_ptx`。
- Config：`gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config`
  与同目录 `config_hopper_islip.icnt`。
- Runtime：release `libcudart.so` 由 `cmake --install gpu-simulator/build/release`
  安装；实际机器环境为 CUDA 12.8 / GCC 13.4。

## Reproduction

输出目录必须在 repository 外，避免把 trace/log 写进 Git：

```bash
python3 experiments/ptx_sync_events/run_smoke.py \
  --output-dir /tmp/accelsim-ptx-sync-events-20260924
```

脚本位置：`experiments/ptx_sync_events/run_smoke.py`。脚本会：

1. 从 `control.ptx` 生成临时 `sync.ptx`，只在 kernel opening brace 后插入
   `membar.cta;` 和 `bar.sync 0;`；
2. 复制 config 到外部 output directory；
3. source `activate-accel-sim.sh`，设置 `PTX_SIM_USE_PTX_FILE=1`、
   `PTX_SIM_KERNELFILE`、`PTX_SIM_EVENT_FILE` 和 `PTX_SIM_RANK=0`；
4. 运行 `cais_add_ptx 256 1`；
5. 用 `CommunicationEvent.from_dict` 解析 `events.jsonl`，检查 event IDs、
   event kinds 和 `time_domain`。

## Result / acceptance

实际运行结果：

```text
PTX_SYNC_SMOKE_PASS events=25 output=/tmp/accelsim-ptx-sync-docs-20260924
```

预期 event count：

| kind | count |
|---|---:|
| `memory_release` | 8 |
| `memory_acquire` | 8 |
| `local_barrier_arrive` | 8 |
| `barrier_wakeup` | 1 |

`run.log` 中 kernel result 为 `status=PASS`；`summary.json` 和 `events.jsonl`
保存在 output directory，不提交到 repository。所有 event 的
`metadata.time_domain` 为 `sim_cycle`。

## Interpretation / limitations

- **Verified**：PTX parser、warp issue、现有 barrier/membar release path 和 P0
  event parser 可以串联运行。
- **Unknown**：`sim_cycle` 到 global ps 的转换尚未实现；不能把该 output 直接
  当成 network timing。
- **Needs hardware validation**：SM90/H100 instruction timing、NVLink、NVSwitch、
  NCCL 和 NVLS 行为。
- 复制标准 config 到不完整的临时目录可能触发已有 cuobjdump/config setup
  failure；脚本因此同时复制 `config_hopper_islip.icnt`，并使用项目 activation
  script。
