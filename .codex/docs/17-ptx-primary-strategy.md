# PTX 主线与 SASS 验证线

更新时间：2026-09-20

## 结论

从现在开始，项目采用两条明确的路线：

```text
PTX execution-driven
  → 新 ISA / 新微架构 / 新调度 / in-switch computing 设计

真实 SASS trace
  → A800/H100 现有硬件行为验证、编译器影响和参数校准
```

PTX 是设计主线，不是因为它比 SASS 更接近真实 H100，而是因为 CAIS、
DySHARP 这类工作要研究的指令、packet 和调度在真实 GPU 上不存在，无法靠
NVBit 等待出一份真实 SASS trace。SASS 仍然重要，但主要用于验证“已经存在的
硬件行为”。

## 为什么 SASS 不再作为新设计主线

SASS trace 来自真实 NVIDIA GPU：

```text
CUDA/NCCL 程序
  → 真实 GPU 产生机器指令
  → NVBit 记录 SASS
  → Accel-Sim 重放
```

这条路径能保留真实的寄存器分配、机器指令顺序和实际地址，因此适合 H100
已有 kernel 的校准。但 A800/H100 不会产生我们新定义的
`ld.cais`、`red.cais` 或 `dymultimem.*`。人工写一份 SASS trace 虽然可行，
但那已经是 synthetic input，不能再称作真实 trace 验证。

## 为什么 PTX 适合新 ISA

PTX 模式由 GPGPU-Sim 解析并执行，模拟器可以直接知道：

- 指令的输入和输出；
- active thread 和分支结果；
- 功能语义；
- register dependency；
- timing/resource mapping；
- 指令何时完成。

因此我们可以先定义一条 proposed instruction，再修改 parser、功能语义、
执行资源和 timing，而不必等待硬件支持它。当前仓库的 PTX 入口主要位于
`gpu-simulator/gpgpu-sim/src/cuda-sim/`；新指令通常涉及 `opcodes.def`、
`ptx.l`、`instructions.cc`、PTX IR 和 shader timing。

PTX 的限制也必须保留：它不是 H100 的真实 SASS。PTX 可能无法反映真实的
寄存器分配、机器级调度和闭源库优化。因此 PTX 结果必须标注为
`Synthetic`、`Hypothesis` 或 `Proposed design`，直到有真实 H100 数据校准。

## CAIS/DySHARP 对本项目的启发

CAIS 明确采用 PTX-level `ld.cais`/`red.cais` 扩展，同时修改 Accel-Sim、
BookSim2、switch merge logic、compiler 和 runtime。DySHARP 进一步加入
`dymultimem.st`、`dymultimem.ld_reduce`、target list、packet format、LSU、
Hub address manager 和 token-centric scheduler。两篇论文都不是“复制一份
单卡 SASS trace 就得到多卡系统”。论文没有公开完整的输入生成和前端实现，
因此我们不假设它们使用了某一种未明确写出的 trace 模式。

我们的实现应先复现它们共同的结构，而不是先猜它们的内部脚本：

```text
PTX instruction / synthetic instruction
  → GPU issue/dependency/completion
  → communication event
  → NVSwitch/NVLS semantic layer
  → BookSim packet/flit timing
```

## 多 GPU 的输入边界

PTX 只负责 GPU 内部执行，不能自动提供 rank、NCCL communicator 或
NVSwitch 状态。多 GPU 输入需要单独的 orchestrator/event schema：

```text
rank/device
collective and algorithm
channel/chunk
target or multicast group
bytes and ready time
dependency and completion
```

因此第一版不把所有信息塞进普通 SASS trace，也不复制单卡 trace 冒充多卡。
推荐让 PTX GPU context 和 communication event 共用一个全局时间，再由
BookSim 负责 transport timing，由 NVSwitch/NVLS layer 负责 multicast、
reduction 和 completion semantics。

## 新的阶段顺序

1. 保留 A800 SASS smoke test，确认软件工具链。
2. 跑通并记录 SM90 PTX baseline。
3. 实现一条没有网络副作用的 custom PTX instruction。
4. 为该指令定义 communication event，并接入两个 logical GPU。
5. 连接 BookSim transport，验证 Ring 前的点对点和时间转换。
6. 实现 Ring，再实现 NVSwitch multicast/reduction。
7. 实现 NVLS-style completion 和 compute/communication overlap。
8. 最后用 H100 SASS、NCCL 和真实 NVSwitch 数据校准。

任何自定义 ISA 论文实验都必须同时给出：指令语义、输入生成方式、微架构
假设、网络模型假设、验证层级和不能证明的内容。
