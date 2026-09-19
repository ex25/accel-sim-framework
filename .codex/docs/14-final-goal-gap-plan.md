# 最终目标差距分析与分阶段计划

更新时间：2026-09-19

## 1. 最终目标的准确表述

最终目标不是“让 BookSim 跑出两个节点”，而是建立一个可以研究以下问题
的分层模拟器：

```text
H100/H200 compute timing
→ NCCL collective lifecycle
→ NVLink/NVSwitch topology and contention
→ NVLS multicast and switch-side reduction
→ collective completion and compute/communication overlap
```

研究对象优先限定为单节点 2/4/8 GPU AllReduce，比较 Ring 和 NVLS。跨节点
InfiniBand、完整 DGX 集群和所有 NCCL collective 暂不属于第一目标。

需要永久区分三种结果：

1. `Software pipeline validation`：tracer、parser、orchestrator 和 simulator
   能够运行。
2. `Synthetic model validation`：bytes、flits、routing、reduction、ordering
   和 scaling 逻辑正确。
3. `Hardware correlation`：模型能够和真实 H100 NVSwitch 的 NCCL、计时器和
   counters 对齐。

只有第三类结果才能支持“接近真实 H100 NVLS 性能”的表述。

## 2. 当前已经有什么

### Verified

- Accel-Sim release build 成功。
- 本机 A800 上完成 CUDA vectoradd → NVBit → raw trace → `.tracez` →
  Accel-Sim SM80 replay。
- 一条真实 A800 `LDG.E.64` 已经完成从 NVBit producer 到 parser、warp、
  memory coalescing、GPU memory interconnect 和 completion 的源码 mapping。
- 外部 BookSim `28f4329` 可以完成两个 logical endpoint 的 AnyNet smoke 和
  9 组 packet/link-latency sweep。
- bytes/flits、rational clock、completion ordering 的 reference contract
  tests 通过。
- 当前项目已经明确保留 compute trace，并将未来 communication event、
  collective semantics 与 BookSim transport 分层。

### Source-audited but incomplete

- Accel-Sim checkout 有 SM90/Hopper 的部分 opcode、TMA、WGMMA、mbarrier、
  cluster/DSM 和 trace fields；这不等于已经支持 H100 NVLink/NVSwitch/NVLS。
- nested GPGPU-Sim 有 BookSim2-derived `intersim2`，但它服务 shader↔memory，
  不是 GPU↔GPU fabric。
- 当前 tracer/context trace 有 CTA、cluster、stream 和 context 文件边界，
  但没有已验证的 rank、peer、collective、NVSwitch queue 或 switch reduction
  event schema。

### 未实现或未验证

- multi-GPU logical GPU contexts 和 rank orchestrator。
- NCCL collective lifecycle/event capture。
- GPU-to-GPU NVLink transport model。
- H100 third-generation NVSwitch topology、ports、routing、buffer 和
  contention。
- multicast resource/slot allocation。
- switch-side arithmetic reduction 和 reduction completion。
- `multimem` instruction parsing/semantics/timing。
- H100 NVSwitch hardware correlation。

## 3. 目前环境最多能做到哪一步

当前本机是 1× A800 PCIe、Ampere/sm_80，没有 NVLink peer。因此本机最多可以
可靠完成：

1. A800 real trace pipeline 和 Ampere replay。
2. trace parser/frontend 的 synthetic Hopper-record tests；可以静态检查或
   编译面向 `sm_90` 的代码，但不能在 A800 上执行 H100 SASS。
3. logical 2/4/8 GPU orchestrator。
4. BookSim two-endpoint、Ring transport、synthetic NVSwitch topology。
5. synthetic multicast、reduction、completion 和 compute/communication
   overlap model。
6. analytical bounds、unit tests、deterministic regression 和公开资料趋势
   比较。

本机不能可靠完成：

- H100/SM90 real trace generation。
- H100 `multimem` execution 或真实 NVLS kernel behavior。
- NVLink 4 link rate、credit、serialization 和 protocol overhead calibration。
- NVSwitch physical route、queue、multicast slot、switch reduction latency。
- NCCL Ring/NVLS hardware algorithm selection and performance correlation。

因此当前环境的合理终点是：

> 完成一个证据标注为 `Synthetic` 的 H100/NVSwitch/NVLS research model，
> 并把所有需要真实 H100 的 calibration hooks 和 experiment scripts 准备好。

## 4. 要接近真实结果还需要什么

### 硬件

最低需要一个可访问的 H100/H200 NVSwitch 节点，优先 8-GPU HGX/DGX 配置；
至少要能运行 2/4/8 GPU 子集。单张 H100 只能验证 SM90 trace，不能验证
NVSwitch/NVLS。

### 软件和系统信息

- GPU 型号、SKU、GPU 数量、NVSwitch 数量和 topology。
- Driver、CUDA、NCCL、Fabric Manager、NVLink/NVSwitch firmware 版本。
- Fabric Manager/NVLSM 状态、NVLink active state 和 partition 配置。
- 固定 NCCL commit/version 和 `nccl-tests` commit。

H100 NVSwitch 系统还需要 Fabric Manager 配置 NVLink/NVSwitch memory fabric；
NVIDIA 文档指出 H100 及后续 DGX/HGX NVSwitch 系统支持 NVLink multicast 和
reduction，并且 NVSwitch 有有限的 multicast slots。citeturn8view0

### 观测和 calibration 数据

- `nccl-tests` 的 Ring/NVLS AllReduce latency、algBW、busBW。
- NCCL debug logs、selected algorithm、channels、chunk/protocol 信息。
- Nsight Systems 的 kernel/collective timeline。
- Nsight Compute/CUPTI/NVML/NVLink counters。
- NVLink/NVSwitch utilization、packet/byte counters、stall 和 error state。
- 多种 message size、GPU count、dtype 和 stream/overlap 配置。

### 源码和 trace

- 固定 NCCL source commit，重点保存 `src/transport/nvls.cc`、algorithm
  selection、channel/tree setup 和 device kernel path。
- 在 H100 上生成真正的 SM90/NCCL trace，验证 tracer 是否能保留 NVLS 相关
  opcode、地址和同步信息。
- 保存 raw trace、post-processed trace、kernels list 和 metadata，但大文件
  不应直接提交到 Git。

NCCL 当前 `nvls.cc` 的初始化路径会检查 GPU count、`NCCL_NVLS_ENABLE` 和
  multicast capability，并建立 NVLS tree/channel；其 setup 还涉及 multicast
  memory、UC backing 与 MC partition 的绑定，不是单纯的一条网络 route。
  citeturn9view1turn9view0

## 5. 单卡 trace 是否足够

结论：不够模拟真实多卡 NVLS；但可以作为每个 logical rank 的 compute
workload template。

### 单卡 trace 可以提供什么

- 一条 GPU 内部 instruction/warp/CTA 时间序列。
- SASS opcode、PC、mask、register dependency、memory address。
- 单卡 kernel 的 issue、cache、memory 和 barrier timing。
- 在“每个 rank 运行完全相同 compute”的假设下，复制 compute workload。

### 单卡 trace 缺少什么

- rank/device/peer identity。
- NCCL communicator、collective id、algorithm/protocol/channel。
- send/recv、reduce-scatter、all-gather 的 message partition。
- collective start、ready、dependency、barrier 和 completion。
- NVLink endpoint、NVSwitch route、link contention、buffer/credit state。
- multicast group、multicast slot 和 switch-side reduction operand。
- compute/communication overlap 以及 NCCL kernel 与应用 kernel 的关系。

所以“复制单卡 trace 八次”最多等价于“八个相同的 compute template”，不等价
于真实 8-GPU execution。

### `multimem` 为什么是关键缺口

PTX 文档定义 `multimem.ld_reduce`、`multimem.st` 和 `multimem.red` 操作
multimem address；它们访问多个 memory locations，且要求 `sm_90` 或更高。
普通 `ld/st` 不能替代 multimem address 访问。citeturn8view2

A800 是 `sm_80`，因此当前 A800 trace 不可能包含真实可执行的 H100
`multimem.*` 指令。当前 Accel-Sim/GPGPU-Sim/tracer source search 也没有
`multimem` consumer semantics。

即使未来 H100 trace 中出现 `multimem`，SASS trace 仍然只能告诉我们 GPU 侧
发出了什么 instruction；它不能直接观察 NVSwitch 内部 queue、switch-side
arithmetic、fabric contention 或 reduction completion。那些必须由 NVSwitch/
NVLS semantic model 和硬件 counters 补充。

## 6. 推荐的数据模型

不要把所有通信语义硬塞进普通 SASS trace。推荐 hybrid model：

```text
per-rank compute trace
        +
communication event trace
        +
topology/configuration metadata
        +
BookSim transport backend
        +
NVSwitch/NVLS semantic layer
```

每个 communication event 至少包含：

```text
collective_id
rank / device_id
operation: AllReduce / ReduceScatter / AllGather / Broadcast
algorithm: Ring / Tree / NVLS / NVLSTree
protocol / channel
src / dst / multicast_group
message_bytes / dtype / chunk_id
ready_time / dependency_ids
completion_time
reduction_op and reduction_precision
```

数据来源按优先级分开：

- compute trace：NVBit/Accel-Sim trace。
- collective lifecycle：NCCL wrapper、NCCL debug/instrumentation、host API
  event capture。
- kernel timeline：Nsight Systems/CUPTI。
- switch/fabric state：hardware counters、Fabric Manager/NVLink telemetry。
- switch reduction timing：真实硬件 calibration 加 semantic model；不能从
  普通 SASS trace 推断。

## 7. 分阶段计划和进入条件

### P0 — 研究基础和证据边界（已完成）

做什么：建立 branch、`.codex/`、规则、baseline、环境记录、source map。

怎么做：保存 Git/环境/BookSim SHA；分离 Verified、Synthetic 和 Needs
hardware validation。

如何验证：新 session 只读 `.codex/` 即可恢复项目目标、状态、限制和下一步。

进入下一阶段：repository clean，baseline 和 source-of-truth 文档可恢复。

### P1 — A800 trace pipeline（已完成）

做什么：跑通 CUDA → NVBit → raw trace → post-process → `.tracez` → replay。

怎么做：使用 `sm_80` 小 benchmark，保存命令、kernel metadata、trace 统计和
   replay output。

如何验证：A800 vectoradd replay exit 0，trace/kernel/request/reply 统计可复现。

进入下一阶段：真实 trace 可生成、可解析、可重放，且不是只读源码。

### P2 — 单指令 autopsy 和 Hopper gap audit（当前已完成初版）

做什么：把一个真实 trace record 映射到 parser、warp、issue、memory 和
completion；逐项记录 SM90/Hopper 需要的字段和缺口。

怎么做：使用 file:line/function mapping；另准备 H100 trace fixture 需求。

如何验证：Mermaid call graph、字段解码、source lines 和 absent-field list。

进入下一阶段：明确 compute trace 能提供什么，不能提供什么。

### P3 — communication event schema 和 reference orchestrator

做什么：实现独立 communication event schema、logical GPU context、rank、
stream dependency 和 collective lifecycle。

怎么做：先用 JSON/CSV synthetic event；compute trace 只作为 rank workload
   template；实现事件 ready/inject/complete 状态机。

如何验证：2/4/8 rank 的 bytes conservation、ordering、barrier、completion
   unit tests；同一 compute trace 复制时明确标注 identical-compute assumption。

进入下一阶段：一个 collective event 可以无歧义地表示为多个 rank/chunk/
dependency，而不是依赖普通 SASS trace 猜测。

### P4 — BookSim endpoint facade 和 clock coupling

做什么：把 BookSim 变成可由 event-driven orchestrator 调用的 backend。

怎么做：对 external BookSim 做最小 upstream-tracked facade，暴露
`create/inject/advance/poll/stats`；使用 rational GPU/network clock 和 global
picosecond time；不把 NVSwitch semantics 写进 generic router。

如何验证：packetization、clock ratio、completion ordering、backpressure、
bytes/flits 和 two-endpoint delivery tests；与现有 analytical checker 对比。

进入下一阶段：logical GPU0→GPU1 event 能被注入、推进、完成并返回稳定 stats。

### P5 — Ring AllReduce baseline

做什么：先实现 2/4/8 logical GPU Ring AllReduce。

怎么做：collective layer 生成 reduce-scatter/all-gather phases，BookSim 只
   负责 packet/flit transport；加入 message size、chunk、channel 和 overlap。

如何验证：bytes conservation、每个 rank completion、理论带宽/latency bound、
   GPU count/message size scaling、无 contention 与有 contention 对照。

进入下一阶段：Ring 的数据流、完成条件和 scaling trend 稳定，且不依赖 NVLS。

### P6 — NVSwitch transport/topology model

做什么：建立 H100-like single-node NVSwitch topology、ports、link latency、
   bandwidth、routing、VC/buffer/credit 和 contention。

怎么做：参数来自 NVIDIA 文档和 H100 硬件 measurements；先 transport，后
   multicast resource；所有参数记录来源和 evidence label。

如何验证：2/4/8 GPU route、link utilization、queue occupancy、packet/byte
   conservation；与 Ring hardware/public trend 做数量级比较。

进入下一阶段：NVSwitch transport 假设和 statistics 可被单独校准，且没有把
   generic BookSim latency 直接称成 NVSwitch latency。

### P7 — NVLS semantic model

做什么：加入 multicast group、multicast slots、multimem operation class、
   switch-side reduction、reduction completion 和 NVLSTree/NVLS AllReduce。

怎么做：先 synthetic reduction operator（sum/min/max）和 deterministic
   completion；再按照 NCCL pinned source 的 setup/channel/chunk 行为对齐；
   `multimem` 作为 GPU-side event 与 switch reduction event 分开表示。

如何验证：reduction 数值/operand conservation、multicast fanout、slot
   exhaustion、reduction completion ordering、Ring vs NVLS scaling trend。

进入下一阶段：synthetic NVLS model 在 unit、analytical 和 BookSim levels
   一致，并且每个 switch-side assumption 都有显式参数。

### P8 — Compute/communication integration

做什么：把 per-rank compute trace、NCCL-like event、BookSim transport 和
   collective completion 接成统一 timeline。

怎么做：实现 event-driven orchestrator；处理 stream dependency、GPU stall、
   communication/compute overlap 和 kernel completion；统计 GPU/network/
   switch/reduction utilization。

如何验证：无通信、无 overlap、完全 serialized、完全 overlapped 四个控制组；
   对比 compute-only 与 communication-only 的结果。

进入下一阶段：控制组行为可解释，collective completion 不会早于依赖事件。

### P9 — H100 trace and hardware calibration

做什么：在真实 H100/H200 NVSwitch 节点生成 H100/NCCL traces，运行 Ring/NVLS
   `nccl-tests`，收集 timeline/counters/topology。

怎么做：固定 hardware/software inventory、NCCL commit、driver/CUDA/Fabric
   Manager；分别测试 2/4/8 GPU、message size、dtype、Ring/NVLS、overlap。

如何验证：先校准单 link，再 topology/transport，再 Ring，最后 NVLS reduction
   and collective completion；报告误差、置信范围和未解释部分。

进入下一阶段：只有当模型与多组 H100 measurements 在预先定义的误差范围内
   相关，才可以声称“接近真实”。

### P10 — Application correlation and research experiments

做什么：选择真实 NCCL/application workload，比较 Ring/NVLS 的 latency、
   bandwidth、utilization、stall、overlap 和 end-to-end runtime。

怎么做：每个实验记录 Accel-Sim SHA、BookSim SHA、config、workload、GPU 数量、
   collective、algorithm、message size、seed、命令、日期和 hardware/model
   level。

如何验证：real hardware、trace replay、simulator 三者的 trend 和误差分解。

## 8. 最重要的决策

当前最合理的路线不是“先把单卡 SASS trace 扩展成八卡 trace”，而是：

```text
H100/可复用 compute trace
        +
独立 communication event trace
        +
Multi-GPU orchestrator
        +
NVSwitch/NVLS semantic layer
        +
BookSim transport backend
```

这样可以在 A800 上先完成软件和 synthetic validation，同时保留未来 H100
trace、NCCL metadata 和硬件 counters 的接入口。它不能消除真实 H100 校准的
需要，但能避免在没有硬件时把不可观测信息伪装成 trace 内容。

## 9. 当前结论

当前环境足以把模型做到 P8 前半段：完成 communication schema、logical
multi-GPU、BookSim transport、Ring、NVSwitch/NVLS synthetic semantics 和
compute/communication orchestrator。要进入 P9 的真实结果阶段，必须获得
真实 H100/H200 NVSwitch 系统、H100 trace、NCCL/Fabric Manager/NVLink telemetry
和固定软件版本。
