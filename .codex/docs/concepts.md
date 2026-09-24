# 给项目研究者的概念和调用路径说明

这是一份面向人的入门文档。它不假设读者已经熟悉 CUDA、GPU 微体系结构、
NCCL 或 BookSim。阅读顺序是：先理解 GPU 如何执行一个 kernel，再理解一条
trace instruction 如何进入 Accel-Sim，最后理解多 GPU collective 为什么需要
额外的通信事件和 NVSwitch/NVLS 模型。

## 先建立一张总图

我们的目标系统可以先看成五层：

```text
CUDA/PTX workload → GPU execution → communication event
                                  → collective/NVSwitch/NVLS
                                  → BookSim transport → GPU completion
```

A800 trace 路径有真实 A800 的 software-pipeline evidence；multi-GPU、
NVSwitch/NVLS 与 BookSim coupling 仍处于 reference/synthetic 阶段。最重要的分界是：

- Accel-Sim 主要回答“一个 GPU 内部的指令多久完成”。
- BookSim 主要回答“一个已经形成的网络 packet/flit 如何经过网络”。
- Multi-GPU/NVSwitch/NVLS 层回答“为什么产生这些通信、谁参与、何时完成、
  是否在交换机中做 reduction”。

## CUDA 程序是怎样在 GPU 上运行的

### Kernel

CUDA 程序通常由 CPU 端代码和 GPU 端代码组成。CPU 端发起一个 GPU 函数，
这个 GPU 函数叫 kernel。比如 vector addition kernel 会把两个数组的每个
元素相加。

一次 kernel launch 包含：

- grid：这次任务总共有多少个工作块。
- block：一个工作块包含多少个线程。
- stream：GPU 用来安排异步工作的顺序队列。
- kernel arguments：输入数组、输出数组和长度等参数。

当前 A800 vectoradd 的一个 kernel 有 98 个 CTA，每个 CTA 有 1,024 个
线程。

### CTA / thread block

CTA 是 Cooperative Thread Array 的缩写，在 CUDA 编程模型里通常就是
thread block。它是一组可以一起运行、可以使用 block-level barrier 的线程。

一个 block 会被放到某个 SM 上执行。不同 block 通常可以并行执行，但 block
之间不能直接假设任意执行顺序。

### Warp 和 lane

GPU 不会一次独立调度 1,024 个线程，而是把线程分成大小通常为 32 的 warp。

- warp：硬件调度的基本 SIMD/SIMT 单位。
- lane：warp 中的一个线程位置，编号 0 到 31。
- warp instruction：一条 SASS 指令通常由整个 warp 共同发出。

因此一条 trace record 通常描述“一条 warp 指令”，而不是只描述一个线程。

### SM

SM 是 Streaming Multiprocessor。它包含 warp scheduler、register file、
整数/浮点/tensor 执行单元、load/store pipeline 和 L1/shared memory 等。

H100 的 SM 是 SM90；A800 的 SM 是 SM80。两者不是同一代硬件，所以 A800
trace 不能直接证明 H100 的 instruction timing。

## PTX、SASS、opcode 和 trace

### PTX 和 SASS 的区别

PTX 是 CUDA 编译器生成的中间表示，类似“面向 GPU 的汇编源代码”。真正运行
时，PTX 会被编译成某一代 GPU 的机器指令，也就是 SASS。

例如：

```text
PTX:  ld.global.u64
SASS: LDG.E.64
```

PTX 可能在不同 GPU 上生成不同 SASS。因此研究 H100 instruction behavior
时，最好使用 H100 编译并执行产生的 trace。

### opcode

opcode 是指令的操作类型。例如 `MOV` 是移动，`IADD` 是整数加法，`LDG` 是
global memory load，`STG` 是 global memory store，`QGMMA/HGMMA/IGMMA` 是
Hopper GMMA 类别，`SYNCS` 是 Hopper synchronization/mbarrier 类别。

`LDG.E.64` 中的核心 opcode 是 `LDG`，`.E.64` 是描述访问形式和宽度的修饰
部分。

### PC

PC 是 Program Counter，表示当前指令在 kernel machine code 中的位置。它不是
内存数据地址，而是“下一条要执行哪条指令”的地址。

### active mask 和 predicate mask

一个 warp 有 32 个 lane，但分支之后不一定每个 lane 都执行同一条指令。

- active mask：当前 warp 中哪些 lane 仍然活跃。
- predicate mask：这条指令的条件谓词允许哪些 lane 执行。
- 最终执行 mask：两者的交集。

当前真实记录里的 `ffffffff` 是 32-bit mask，32 位都是 1，表示 32 个 lane
都参与这条指令。

### register 和 register dependency

寄存器是 GPU 内部很快的临时存储。例如：

```text
R4 = R2 + R3
```

这里 `R2/R3` 是输入，`R4` 是输出；下一条读取 `R4` 的指令必须等待它完成，
这就是 register dependency。模拟器用它判断 warp 是否 ready。

当前 Accel-Sim parser 将 SASS `R4` 映射为内部寄存器索引 `R5`，因为
GPGPU-Sim 的内部寄存器编号从 1 开始，而 SASS 从 `R0` 开始。这是编号约定
变化，不是程序真的多了一个寄存器。

## NVBit trace 是什么

NVBit 是 NVIDIA 的 binary instrumentation framework。它允许 tracer 在 GPU
指令附近插入记录函数。

当前 tracer 记录 CTA、warp、PC、active/predicate mask、opcode、寄存器、memory
width、每个 lane 的地址，以及 Hopper cluster/TMA/mbarrier 相关 metadata。

它记录的是“GPU 指令执行信息”，不是完整的应用执行时间线。普通 trace record
没有 NCCL collective、NVSwitch queue 或 switch reduction 字段。

## 一条真实 A800 指令怎样进入 Accel-Sim

NVBit 记录 SASS，post-processing 生成 `.tracez`；Accel-Sim 的 parser、
warp issue、cache/interconnect、memory completion 依次消费它。具体 producer、
consumer symbol、`LDG.E.64` 调用路径与 A800 smoke 证据集中记录在
[gpu-execution.md](gpu-execution.md)，此处只保留概念层次。

## CTest、hotspot trace 和 request/reply packets

### CTest

CTest 是 CMake 的测试运行器。只有通过 `add_test()` 注册的测试才会被发现。
因此 `ctest → No tests found` 只表示当前 build 没有注册 CTest case，不能解释
成“所有测试通过”。所以本项目还需要显式运行 Python unit test、shell smoke
和 simulator replay。

### hotspot trace

hotspot 是用于产生规则计算/内存负载的 CUDA benchmark。项目里的 V100 hotspot
trace 可以检查 parser、kernel launch、shader/cache/memory path 是否结束；但
它不能证明 A800、H100、NVLink 或 NVLS，因为 trace/config 来自 V100 场景。

### request/reply network packets

当前 GPU memory system 中：

- request packet：shader 发给 memory partition 的请求，例如读 cache line。
- reply packet：memory system 返回数据或 write acknowledgement。

之前 A800 replay 的 75,000 request 和 75,000 reply 是 GPU 内部 memory path
统计，不是 NVLink packet、NVSwitch packet、NCCL message 或 NVLS reduction。

## 多 GPU 和 NCCL 的基本概念

### device、rank、peer

- device：物理 GPU，例如 CUDA device 0。
- rank：一个 collective participant 的逻辑编号；不能简单等同于 device id。
- peer：通信另一端的 GPU。

单卡 trace 知道 kernel 在某个 CUDA context 中运行，但不自动知道它在 NCCL 中
是 rank 3，或者 peer 是哪张 GPU。

### communicator

NCCL communicator 是参与通信组的上下文，维护 rank 数量、当前 rank、连接方式、
channels、算法和 transport resources。没有 communicator 信息，模拟器无法知道
一次 message 应该等待哪些参与者。

### collective

collective 是一组 GPU 共同参加的通信操作：

- AllReduce：每个 GPU 提供数据，最后每个 GPU 得到相同的归约结果。
- AllGather：每个 GPU 提供一段数据，最后每个 GPU 得到全部数据。
- ReduceScatter：先归约，再把不同结果片段分给不同 GPU。
- Broadcast：一个 GPU 把数据发送给多个 GPU。

collective 有参与者集合、统一完成条件和通常固定的算法阶段。

### Ring、reduce-scatter、all-gather

Ring AllReduce 通常分两段：

1. reduce-scatter：数据片段沿 ring 传递并累加，最终每个 rank 拥有一个已归约
   片段。
2. all-gather：每个 rank 把自己的归约片段沿 ring 传播，最终所有 rank 拥有
   完整结果。

因此 AllReduce 不是一条大 packet，而是多个阶段、chunk 和方向的通信事件。

### message、partition、chunk、channel

- message：一次 collective 需要传输的逻辑数据，例如 1 MiB。
- partition：把大 message 切成多个部分。
- chunk：实际调度/传输的较小片段。
- channel：NCCL 并行推进通信的逻辑通道；多个 channel 增加并行度，也增加
  network contention。

### dependency、completion、overlap

- dependency：事件 B 必须等事件 A 完成。
- collective completion：所有 rank 都满足 collective 完成条件。
- compute/communication overlap：GPU 一边计算，一边通过另一条 stream 或
  pipeline 进行通信。

只复制 compute trace 会丢失这些时间关系，因此无法得到正确 end-to-end runtime。

## NVLink、NVSwitch、BookSim 和 AnyNet

### NVLink 和 NVSwitch

NVLink 是 GPU 之间的高速互联链路，定义物理带宽、链路方向和协议。NVSwitch
是连接多张 GPU 的交换芯片，可能包含 ports、routing、buffers、queues、flow
control、credits 和 multicast/reduction 资源。

### packet、flit、credit

- packet：一次逻辑网络消息。
- flit：packet 被切成的固定大小传输单元。
- credit：接收端告诉发送端还有多少 buffer 空间，防止溢出。

BookSim 通常以 flit 为 timing 单位，因此需要：

```text
message bytes → packetization → flit count → route/arbitration → delivery
```

### AnyNet

AnyNet 是 BookSim 的通用拓扑描述方式。当前 prototype 用两个 router 表示：

```text
logical GPU 0 → router 0 ↔ router 1 ← logical GPU 1
```

它只说明 generic packet transport 可以运行，不表示真实 H100 NVSwitch 的端口、
路由或 latency。

### multicast 和 switch-side reduction

multicast 是“一次发送，多个 endpoint 收到”；真实 NVSwitch 的 multicast 有
资源限制，不能简单复制 packet。switch-side reduction 则是在交换设备中执行
加法等运算，例如 `value0 + value1 + value2`，因此还需要 reduction resource、
operand ordering、operator、precision 和 completion timing。

## NVLS 是什么

NVLS 可以先理解成利用 NVLink multicast 和 NVSwitch reduction 能力的 collective
transport/algorithm。它至少涉及：

1. communicator 判断 NVLS 支持。
2. 分配和绑定 multicast memory。
3. 建立 NVLS channels/tree。
4. GPU 使用 multimem 相关操作访问特殊 memory mapping。
5. NVSwitch 执行 multicast/reduction。
6. 所有参与者完成后 collective 才结束。

所以 NVLS 必须同时描述 GPU-side instruction、memory mapping、network transport、
switch reduction 和 collective completion。

## endpoint/clock contract

GPU 与网络可能使用不同频率；通信事件应携带统一的 global timestamp，
不能把一个 GPU cycle 直接当成一个 BookSim cycle。为什么使用整数
picoseconds、如何逐 network edge 推进和何时返回 completion，见
[network-model.md](network-model.md)。

## 如何理解当前结果

- `75,000 request/reply packets`：GPU memory model 的内部请求/响应，不是 NVLink。
- `BookSim latency = 12 cycles`：当前 AnyNet 配置下的 synthetic latency，不是
  H100 NVLink latency。
- `trace replay PASS`：trace 可以被 simulator pipeline 读取并完成 timing simulation，
  不代表 trace 包含 NCCL/NVLS，也不代表硬件性能相关。
- `H100 support partial`：源码有 SM90 config/opcode/handling，但还没有真实 H100
  trace 和硬件 correlation。

## Hopper 支持与多 GPU 边界

SM90 config、Hopper opcode map、GMMA/TMA/mbarrier/cluster fields 有部分源码
路径；但 `cluster_rank` 不是 NCCL rank，TMA multicast 不是 NVLS multicast，
现有 `intersim2` 也只建模 GPU-memory traffic。详细 feature 矩阵和限制见
[gpu-execution.md](gpu-execution.md)。

## 读完后应该形成的判断

1. trace 是 GPU instruction workload，不是完整 distributed execution。
2. 当前 `icnt` 是 GPU 内部 memory network，不是 NVLink。
3. request/reply packet 统计不能直接解释成 NCCL traffic。
4. AnyNet 是 BookSim generic topology，不是 NVIDIA topology。
5. `multimem` 是 GPU-side instruction，但 NVLS 还需要 switch-side semantics。
6. 当前最合理的数据模型是 compute trace + communication event trace。
7. A800 可以完成软件和 synthetic 阶段，但真实 H100/NVLS 仍需要 H100 NVSwitch。
