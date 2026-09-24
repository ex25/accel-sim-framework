# 项目路线与阶段计划

更新时间：2026-09-24  
当前主分支：`feat/ptx-sync-events-p0-p5`  
当前 nested GPGPU-Sim 分支：`feat/ptx-sync-events-p1-p5`

## 目标与边界

项目目标按以下顺序推进：

1. **当前系统目标**：建立可复现的 DGX-H100 类 8-GPU + 4-NVSwitch + NVLS
   logical simulator；先证明事件、collective、topology、transport 和完成
   顺序，再谈性能。
2. **工作负载目标**：在系统稳定后运行缩小且可复现的 prefill、training 或
   等价 compute/communication workload，检查 throughput、通信占比、stall、
   overlap 和 scaling 是否符合预期。
3. **最终研究目标**：在上述基础上复现并扩展 ACIS/CAIS、DySHARP 类 custom ISA、
   in-switch computing、dynamic `multimem` addressing 和 switch-side reduction
   设计。论文复现不应建立在未经验证的 DGX-H100 model 之上。

分层 simulator（架构分工见 [network-model.md](network-model.md) 和 [gpu-execution.md](gpu-execution.md)）：

```text
PTX GPU execution
  → synchronization/communication event
  → collective semantics
  → NVSwitch topology/protocol
  → BookSim transport timing
  → completion and compute/communication overlap
```

必须区分三种 evidence：

| Evidence | 可以证明 | 不能证明 |
|---|---|---|
| `Verified / unit` | schema、dependency、bytes/flits、状态机 | GPU 或 NVLS fidelity |
| `Verified / simulator` | 当前 GPGPU-Sim/PTX path 的可执行行为 | 真实 H100 timing |
| `Synthetic network` | topology、routing、collective ordering 和模型趋势 | physical NVLink/NVSwitch/NVLS |
| `Real hardware` | 指定 GPU/system 上的 correlation | 其他 SKU 或未测 topology |

本机只有 1× A800 PCIe，没有 NVLink peer，因此当前工作最多形成
`Synthetic` H100/NVSwitch/NVLS model；不能把它称为真实 DGX H100 validation。

## 当前完成状态

| 阶段 | 状态 | 产物 |
|---|---|---|
| P0 event schema | 已完成 | `communication_events.py` |
| P1 GPU-local PTX sync events | 已完成 | nested `shader.cc` JSONL producer |
| P2 point-to-point completion | 已完成 | `point_to_point_bridge.py` |
| P3 Ring all-reduce | 已完成 | `ring_collective.py` |
| P4 8-GPU/4-NVSwitch facade | 已完成 | `nvswitch_transport.py` |
| P5 NVLS reference completion | 已完成 | `nvls_collective.py` |
| GPU event stream → collective engine | 未完成 | 下一阶段 |
| Embedded BookSim per-event timing | 未完成 | 下一阶段 |
| H100/NVSwitch hardware calibration | 未完成 | 需要目标硬件 |
| Prefill/training workload sanity | 未开始 | 依赖 P10 和固定 workload/config |
| ACIS/CAIS/DySHARP reproduction | 未开始 | 依赖系统与 workload validation |

## 后续唯一执行顺序

### P6：online event/collective integration

把 P1 JSONL producer 从离线文件消费改成 online adapter，完成：

1. `sim_cycle → global ps` 的明确 clock conversion；
2. `COMM_SUBMIT`、`EventCompletion` 与 GPU warp/scoreboard wakeup 的双向映射；
3. 一个最小 2-rank PTX workload 的 submit/inject/complete 回归；
4. 失败、取消、超时和重复 event 的处理。

进入条件：P1/P2 tests 继续通过。退出条件：一个 PTX workload 能产生
communication event，并让模拟 GPU 在 completion 后继续执行。

### P7：embedded BookSim transport

在不复制 BookSim source 的前提下，给 `28f4329` reference checkout 增加一个
narrow facade，暴露 `create/inject/advance_to/poll_completed/stats`。必须分别
验证 packetization、flit count、clock ratio、queue、routing 和 completion。

进入条件：P6 online path 通过。退出条件：同一事件在 reference adapter 和
BookSim facade 的 ordering/bytes/flits 可比较；仍标记为 `Synthetic network`。

### P8：NVSwitch protocol model

在 P4 graph 之上增加 link bandwidth、switch port、buffer/credit、arbitration、
contention、multicast resource 和 route statistics。所有参数必须来自 config，
不能硬编码为“DGX H100 真值”。

### P9：NVLS semantic model

把 P5 的 abstract reduction 替换为可配置 reduction operand、multicast
replication、switch-side completion 和 failure semantics。`multimem` PTX
parser/frontend 是否需要扩展，必须先有 source/ISA evidence 再实现。

### P10：compute/communication overlap

将 kernel、stream、barrier、collective channel 和 network completion 放进同一
global timeline，验证 Ring/NVLS 对 compute stall、overlap 和 completion 的影响。

### P11：H100/HGX/DGX validation

需要真实 H100/H200 multi-GPU system。每次 run 记录 GPU count、SKU、driver、
CUDA、NCCL、Fabric Manager、`nvidia-smi topo -m`、NVLink/NVSwitch state、
benchmark、message sizes、channels、algorithm、output path 和 commit SHA。

### P12：prefill/training workload sanity

在固定的 8-GPU topology、clock、message size 和 model configuration 下，加入
受控的 prefill 与 training workload（优先选择可缩小、可重复的 kernel/collective
组合），记录 compute、communication、stall、overlap、throughput 和 scaling。
先验证 simulator 实现的一致性与趋势，不把没有硬件校准的结果称为 DGX-H100
绝对性能。

进入条件：P10 timeline 和固定 workload/config/evidence contract 通过；不要求
先拥有真实 H100，但结果必须标明 `Synthetic` 或 `Needs hardware validation`。
退出条件：至少
一个 prefill 和一个 training-style workload 可重复运行，结果能按阶段归因，
并明确哪些参数仍是 synthetic 或 hypothesis。

### P13：ACIS/CAIS 与 DySHARP reproduction/extension

以 [custom-isa.md](custom-isa.md)
为设计入口，逐项
实现论文中的 custom ISA、GPU endpoint、multimem/event schema、switch-side
reduction、scheduler/runtime assumption 和 workload generator。每项必须分别
验证 parser/functional/dependency/timing、network semantics、application
metric，并记录论文公开信息与自行假设的区别。

进入条件：P12 workload sanity 通过，且已有公开或硬件数据可用于至少一组
calibration。退出条件：论文 baseline、改进方案和 ablation 均能用固定 commit、
config、workload 和 output path 重复运行；无法从论文确认的细节必须标为
`Unknown` 或 `Hypothesis`。

## 停止条件与文档要求

- 每个阶段先记录 baseline，再改动，再运行 focused test 和 regression，最后
  更新本文件与 [implementation.md](implementation.md) 的对应章节；不要为
  新 phase 再创建一个 implementation Markdown。
- 新功能必须在 `implementation.md` 写明：功能、修改文件/symbol、调用路径、
  配置、before/after、测试命令、结果、evidence、限制和下一步。
- 新实验必须在 `.codex/experiments/` 写明：objective、hypothesis、workload、
  topology、rank、message size、SHA、command、output directory、metrics、
  acceptance criteria 和复现命令。raw trace/log 只能放在 Git 外。
- 不得把 A800 结果写成 H100 结果，不得把 synthetic BookSim/NVLS 写成真实
  NVSwitch/NVLS 实现。
