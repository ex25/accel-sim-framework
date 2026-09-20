# 多 GPU、定制 ISA 与 in-switch computing 应该如何模拟

本文回答一个比“如何跑通 A800 trace”更重要的问题：如果研究目标是
H100 + NVSwitch + NVLS，甚至要修改 ISA、GPU 微架构和交换机，那么应该怎样
决定 workload 从哪里来、GPU 之间的调度怎样产生、哪些部分需要真实 trace、
哪些部分必须由模拟器自己生成。

先给结论：

> 真实 GPU trace 是“一个真实执行结果的记录”，不是“完整系统的设计规范”。
> 它可以告诉我们某个 GPU 执行了哪些指令以及这些指令在该次运行中的先后关系，
> 但通常不能告诉我们 NVSwitch 内部队列、credit、multicast/reduction 状态，
> 也不能在硬件已经不存在新指令时凭空产生新 ISA 的执行。

因此，本项目采用“PTX 主线 + SASS 验证线”的分层方法：

\`\`\`text
PTX execution-driven GPU design input
        +
独立的多 GPU communication / collective event trace
        +
修改后的 GPU ISA / microarchitecture model
        +
NVSwitch / NVLS semantic model
        +
BookSim cycle-level transport model
        +
真实 A800/H100 SASS trace（仅用于验证已有硬件行为）
\`\`\`

这不是降低研究严谨性，而是把“硬件能够直接观察的事实”和“新硬件设计
必须假设的行为”明确分开。

---

## 1. 先区分三种不同的 trace

很多论文只写“we use Accel-Sim to simulate multi-GPU”，但这里的 trace
可能指完全不同的东西。

### 1.1 真实 GPU 指令 trace（验证线）

这是我们目前在 A800 上做的事情：

\`\`\`text
CUDA program
  → real GPU executes it
  → NVBit observes SASS instructions
  → raw trace
  → post-processing
  → Accel-Sim replays the trace
\`\`\`

这种 trace 的优点是指令来自真实 GPU；缺点是它只记录被 NVBit 观察到的
GPU 执行信息，不是 NVSwitch 的内部 trace。

### 1.2 模拟器输入的通信事件 trace

这是我们后续应该新增的第二种 trace。它不是 SASS 行，而是类似下面的
高层事件：

\`\`\`text
time = 1200 ns
rank 0 → multicast group {1,2,3}
collective = 17
operation = dispatch
chunk = 4
bytes = 4096
\`\`\`

它描述“哪个 GPU 在什么时候发起了什么通信”，然后由 NVSwitch/NVLS
模型和 BookSim 决定具体经过哪些端口、队列和 flit。

### 1.3 PTX execution-driven design input（主线）

PTX 模式不需要真实 GPU 产生新指令。我们可以在模拟器中定义一个新的
PTX-level opcode，让 PTX parser、功能执行、依赖和 timing model 直接执行它。
它适合研究还不存在的 GPU/ISA，但不能自动代表真实 H100 的 SASS 调度。

### 1.4 为新硬件设计生成的 synthetic workload

如果我们提出了真实 GPU 不存在的 \`dymultimem.ld_reduce\` 或新的 CAIS
指令，就不能等待 A800/H100 产生真实 trace。此时需要：

1. 优先修改 PTX parser、PTX IR 和功能执行路径，使模拟器认识新 opcode；
2. 修改 GPU timing/resource/dependency model；
3. 明确定义新指令的操作数、延迟、依赖、内存语义和网络副作用；
4. 由 workload generator、修改后的 compiler/runtime 或 synthetic event
   generator 产生这些事件；
5. 只有在实验明确需要时，才为新指令建立 synthetic SASS trace。

这类输入必须标记为 \`Synthetic\` 或 \`Hypothesis\`，不能称为真实硬件 trace。

---

## 2. 多 GPU 的“调度”到底从哪里来

先把调度拆成三层，否则“GPU 调度”会混成一个词。

### 2.1 应用/运行时调度

这一层决定：

- 哪些数据属于哪个 rank；
- 哪个 rank 放在哪个 GPU；
- 什么时候调用 AllReduce、AllGather 或 Dispatch；
- NCCL communicator 中有哪些参与者；
- 哪些 stream/event 形成先后依赖。

真实系统中，这部分通常来自 CUDA runtime、NCCL 和应用框架。

### 2.2 GPU 内部调度

这一层决定：

- CTA 什么时候被放到某个 SM；
- warp 什么时候被 scheduler 发射；
- 某条 load/reduce 指令什么时候到达 LSU；
- 指令依赖、scoreboard、barrier 什么时候允许继续执行。

Accel-Sim 的 trace-driven frontend 可以从真实 SASS trace 重放一部分结果，
但如果研究的是新的 scheduler 或新的跨 GPU 同步机制，就必须修改
simulator 的 scheduler/state machine，不能只复制原 trace。

### 2.3 网络调度

这一层决定：

- packet 进入哪个 GPU endpoint；
- 走哪个 NVSwitch port；
- 哪个 virtual channel 获得 credit；
- multicast 复制多少份；
- reduction 等待几个 operand；
- response 什么时候返回 GPU。

这部分不在普通单 GPU SASS trace 中，应该由 NVSwitch semantic layer 和
BookSim transport layer 产生。

因此，真实多 GPU 模拟至少需要：

\`\`\`text
应用/collective schedule
  → GPU instruction schedule
  → network packet schedule
\`\`\`

不能把三者都假设成“从一个单卡 trace 自动推导”。

---

## 3. CAIS 和 DySHARP 论文实际说明了什么

### 3.1 CAIS

CAIS 论文明确说明了以下内容：

- 模拟 8-GPU、4-NVSwitch 的 DGX-H100 类拓扑；
- 扩展 Accel-Sim，加入 Hopper-specific architectural features；
- 集成定制 BookSim2；
- 修改 Accel-Sim 和 BookSim2 支持 NVLS \`multimem\`；
- 在 BookSim2 router 中加入 in-switch multicast/reduction；
- 在 Accel-Sim Hub 处理 multimem address 到 virtual address 的转换；
- 使用 \`ld.cais\` / \`red.cais\` 一类新的 PTX-level ISA 扩展；
- 加入 TB-group synchronization、request throttling 和 graph-level
  dataflow optimization；
- 使用 CUTLASS GEMM 和缩小后的 LLM 配置；
- 用真实 NCCL AllReduce 的 1–16 GB 数据与模拟器结果比较，报告平均误差
  3.87%。

论文还给出了 NVLink/NVSwitch 的模型参数，例如 16B flit、250 ns 单跳、
40 KB per-port merge table 和 virtual channel 数量。

但论文正文没有完整说明：

- 每条模拟指令是否来自 NVBit trace；
- 是否有公开的 trace generator；
- 多 GPU rank 的具体 event format；
- 修改后的 Accel-Sim/BookSim2 代码是否公开；
- \`ld.cais\` 如何从 compiler 生成到 simulator 输入；
- 每个 rank 的具体 CTA/warp 时间如何获得。

所以这些细节的正确标签是 \`Unknown\`，不能自行补成“肯定是 trace 重放”。

CAIS 论文明确写的是“custom Accel-Sim + custom BookSim2 + custom ISA/
microarchitecture + compiler/runtime support”，而不是“只把一份单卡 trace
复制八次”。

来源：[CAIS HPCA 2026 paper](https://chenzhangsjtu.github.io/files/2026-HPCA-CAIS.pdf)

### 3.2 DySHARP

DySHARP 论文明确说明了：

- 模拟 GH200 NVL32 类 32-GPU、9-NVSwitch、fat-tree 拓扑；
- 集成 BookSim2 和定制 Accel-Sim；
- 按 H200 参数配置 GPU；
- 加入动态 multimem addressing；
- 修改 packet format、GPU LSU、switch route/reduction logic 和 Hub memory
  manager；
- 使用 \`dymultimem.st\` 和 \`dymultimem.ld_reduce\`；
- 用 persistent megakernel 表示任务列表，软件方式绕过原本的硬件 TB
  scheduler；
- 使用 token readiness、Dispatch/GEMM/Combine pipeline 和 SM partitioning；
- 对 DeepSeek-V3 使用 S/M/L 配置，并将 token expert distribution 建模为
  标准差为 0.032 的正态分布；
- 报告其模型在 DGX-H100 上进行验证，GEMM 和 DeepEP communication operator
  的平均误差在 6% 以内。

这说明 DySHARP 的 GPU 之间行为不是由一份普通单卡 trace 自动产生的。
至少从论文公开描述看，它有自己的：

\`\`\`text
task list
→ persistent megakernel
→ token readiness
→ dispatch/computation/combine schedule
→ custom multimem instruction
→ custom switch packet/reduction
\`\`\`

但是，论文同样没有在正文中完整说明每条基础 GPU 指令是否来自 NVBit、
是否使用 execution-driven PTX、是否使用内部 trace generator。因此这一点
仍然必须标记为 \`Unknown\`，只能通过作者代码、补充材料或联系作者确认。

来源：[DySHARP arXiv paper](https://arxiv.org/abs/2605.05607)

---

## 4. 如果我们修改 ISA，应该怎样模拟

新增一条指令不是只在 \`trace_opcode.h\` 中加一个枚举。至少要经过下面六层。

### 第一步：定义指令语义

先写清楚：

\`\`\`text
指令名称：dymultimem.ld_reduce
输入寄存器：数据地址、target list、target count
输出寄存器：reduced value
内存语义：读取多个 GPU 的同一 algebraic index 并求和
完成条件：所有目标 operand 到达 switch 且结果返回
异常/ordering：是否可重排，是否需要 acquire/release
\`\`\`

如果语义没有写清楚，后面无法定义 latency、scoreboard 或 packet。

### 第二步：决定输入来源

有三种选择：

| 方式 | 适合用途 | 是否是真实硬件行为 |
|---|---|---|
| 真实 H100 trace | 重放已有 H100 指令 | 是，但只能覆盖已有指令 |
| 修改后的 compiler/runtime 生成 trace | 验证新 ISA workload | 否，属于 proposed design |
| 独立 synthetic event/instruction generator | 先做模型和边界测试 | 否，属于 synthetic |

我们本地 A800 不能产生 SM90 \`multimem\` 或 CAIS/DySHARP 自定义指令，
因此新 ISA 的第一版只能使用后两种方式。

### 第三步：修改 trace/frontend

如果继续使用 Accel-Sim trace-driven 路径，需要修改：

\`\`\`text
opcode enum / opcode map
→ trace parser
→ instruction metadata
→ register dependency
→ latency/resource mapping
→ trace warp issue
\`\`\`

对于普通计算指令，trace 行可能足够；但对 \`multimem\`，trace 还必须带有：

\`\`\`text
collective_id
rank
target group
algebraic address
reduction operator
expected operand count
\`\`\`

这些字段不应该偷偷塞进普通的 \`memory address\` 字段，因为它们不是普通
local load/store 的属性。

### 第四步：修改 GPU 微架构模型

新指令可能需要：

- LSU 新队列，例如 \`MultimemQ\`；
- 新 scoreboard dependency；
- 新的 issue/retire 条件；
- Hub address translation；
- GPU endpoint packet builder；
- communication completion callback；
- 与 CTA/warp barrier 的关系。

DySHARP 论文中的 \`MultimemQ\`、Hub AL-TLB 和 token tables 就属于这一层。

### 第五步：修改 NVSwitch/NVLS 语义层

BookSim 只知道 packet、flit、router、VC 和 cycle。它不知道：

- 哪些 packet 属于同一个 reduction group；
- 需要等待几个 operand；
- multicast 目标集合是什么；
- reduction 完成后应该向哪个 GPU 返回；
- address 到目标 GPU 本地地址怎样映射。

因此需要在 BookSim 之上建立：

\`\`\`text
NVSwitchSemanticLayer
  - multicast group table
  - reduction session table
  - expected operand count
  - partial result buffer
  - completion event
\`\`\`

### 第六步：校准和验证

新 ISA 不能只验证“程序跑完了”。至少要逐层验证：

1. opcode 能被 parser 识别；
2. register dependency 正确阻塞/释放；
3. packet bytes/flits 正确；
4. multicast 复制数量正确；
5. reduction operand 数量正确；
6. completion 只发生一次；
7. 2/4/8 GPU scaling 趋势合理；
8. 与公开硬件数据或真实 H100/NVSwitch 结果比较。

---

## 5. 一台 DGX、8 份 GPU trace 仍然看不到什么

即使将来拿到一台真实 DGX-H100，并记录 8 张 GPU 的 trace，也不能保证
得到完整 NVSwitch/NVLS 信息。

### 5.1 通常可以看到的内容

- 每张 GPU 的 kernel 和 SASS 指令；
- CTA/warp/PC/opcode；
- local/global memory addresses；
- 某些 \`multimem\` 指令；
- GPU 侧发起请求的大致时间；
- CUDA stream 和部分同步行为；
- NCCL kernel 的软件侧调用痕迹。

### 5.2 通常看不到或不能直接还原的内容

- NVSwitch 内部每个 input/output queue 的 occupancy；
- credit 消耗和返回的精确时间；
- 内部 arbitration 顺序；
- physical route 的每次选择；
- multicast 在 switch 内部复制的瞬间；
- reduction buffer 是否命中、等待或 eviction；
- switch-side reduction 的 partial result 生命周期；
- Fabric Manager 内部状态；
- 真实 collective 在硬件内部的隐式 completion 条件。

这些需要额外来源：

\`\`\`text
NCCL debug/log metadata
NVIDIA hardware counters
Nsight Systems / Nsight Compute
公开 NVLink/NVSwitch 参数
真实 collective latency/bandwidth
模拟器假设
\`\`\`

所以“8 GPU trace”仍然不是“8 GPU network trace”。它更准确地说是：

\`\`\`text
8 份 GPU endpoint-side execution observations
\`\`\`

网络内部状态仍需要模型。

---

## 6. 我们项目应该采用的模型边界

建议固定为下面的五层：

\`\`\`text
Layer 1: Workload / runtime
  rank、stream、collective、dependency、completion

Layer 2: GPU compute model
  PTX instruction、CTA、warp、LSU、cache、HBM、SM scheduler
  （SASS trace 作为已有硬件验证输入）

Layer 3: GPU endpoint / protocol adapter
  local memory request → NVLink packet
  communication completion → warp/collective release

Layer 4: NVSwitch / NVLS semantic model
  route、multicast、reduction、address mapping、session state

Layer 5: BookSim transport
  packet、flit、VC、buffer、credit、arbitration、cycle
\`\`\`

这样做的好处是：

- 先不改 trace format，也能做 2-GPU communication event；
- 先不实现 NVLS，也能验证 BookSim transport；
- 先用 synthetic event，再接真实 H100 trace；
- 修改 ISA 时只扩展 Layer 2/3，不把所有逻辑塞进 BookSim；
- 修改 switch 时只扩展 Layer 4，不伪装成 GPU memory timing。

推荐的输入形式是：

\`\`\`text
PTXWorkload(rank, CTA, warp, instruction, local memory behavior)
或 SASSComputeTrace(rank, CTA, warp, instruction, local memory behavior)
CommunicationEvent(rank, peer/group, collective, chunk, bytes, ready_time)
\`\`\`

而不是立即把所有内容合并为一个巨大的 SASS trace。

---

## 7. 我们现在能做到哪一步

### A800 本地环境可以验证

- NVBit 安装和 instrumentation；
- CUDA → raw trace → \`.tracez\`；
- Accel-Sim trace parser/replay；
- GPU 内部 cache/L2/DRAM timing；
- synthetic two-endpoint BookSim transport；
- 新 trace/event schema 的单元测试。

### A800 不能验证

- H100 SM90 \`multimem\`；
- H100 WGMMA/TMA/mbarrier 的真实性能；
- NVLink 4；
- H100 NVSwitch；
- NVLS multicast/reduction；
- NCCL 8-GPU collective timing。

### 要比较接近真实结果，至少还需要

1. H100/H200 的 SM90 trace 生产能力；
2. 真实 2/4/8 GPU NVLink/NVSwitch 系统；
3. NCCL 版本、驱动、CUDA、Fabric Manager 和拓扑记录；
4. Ring/NVLS 的真实 AllReduce 或 ReduceScatter/AllGather 数据；
5. 新 ISA 设计的 compiler/runtime 或明确的 synthetic generator；
6. NVLink/NVSwitch queue、VC、buffer、latency、bandwidth 的校准数据。

没有这些时，可以继续做模型设计和趋势实验，但结果只能称为
\`synthetic network validation\` 或 \`public-data comparison\`。

---

## 8. 对我们下一阶段的直接决定

下一阶段不要先复制 8 份 \`vectoradd\` trace，也不要先修改一大批 SASS
opcode。应该先跑通 SM90 PTX baseline，再实现一条最小 custom PTX instruction：

1. 先定义一条 custom PTX instruction 的语义、依赖和完成条件；
2. 明确定义 \`CommunicationEvent\`，包括 rank、collective、chunk、bytes、
   ready/completion；
3. 建立 2 个 logical GPU 的 orchestrator；
4. 用 synthetic compute duration + communication event 驱动 BookSim；
5. 验证 point-to-point 和 Ring 的 bytes、ordering、completion；
6. 再加入 NVSwitch multicast/reduction；
7. 再把真实单卡/H100 compute trace 接到兼容的 validation boundary；
8. 最后用真实 SM90/NCCL 结果校准模型。

达到下面条件后，才适合进入自定义 ISA 阶段：

- 2/4/8 logical GPU 的 collective event model 已经通过 unit/synthetic tests；
- Ring transport 和 collective completion 正确；
- NVSwitch route/multicast/reduction 的状态机有独立测试；
- compute 与 communication overlap 有可重复的时间线；
- 至少有一组真实或公开数据可以校准 latency/bandwidth；
- 新 ISA 的语义、trace 表示、GPU endpoint 和 switch 行为已经写成设计文档。

否则即使模拟器输出了一个数字，也很难说明这个数字来自哪个假设。
