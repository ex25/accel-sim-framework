# 阶段 1 实现方案：算对数值、参与线程和路由索引

本文件是待执行方案，不是完成报告。当前没有本阶段代码改动或运行验收。依赖[阶段 0](stage-0.md)的加载、传参和单模块入口；配套：[阶段 1 验收](../validation/stage-1.md)。

## 完成后应当能做什么

独立启动真实的类型转换、top-k、计数和重排内核，算出正确的数值与索引，同时产生有效周期。top-k 选择每个 token 分给哪些专家；计数和重排把这些 token 排成专家矩阵的输入。这里若把专家编号或位置算错，后面的矩阵乘法即使实现正确，也会计算错误的数据。

warp 是一起调度的 32 个线程，lane 是线程在其中的编号 0～31。**member mask 是一个 32 位整数，第 i 位为 1 表示 lane i 参与本次协作。**例如 `0xffff0000` 指定 lane16～31，不是“前16个活动线程”。predicate 是控制某条指令是否执行的真假值，与 member mask 是不同输入。

```text
BF16输入 → 类型转换 → 专家分数/人工分数
                         → top-k及排序 → 专家计数 → 行映射/重排
                               ↑              ↑
                          ballot投票      shuffle交换、redux求和
```

真实 gating 矩阵乘法属于阶段 2。本阶段用人工分数独立测试路由，不等待 gating，更不声称冻结完整启动器已经能执行前向的一部分。

## 工作范围和具体交付

| 指令或能力 | 本阶段动作及必须得到的结果 | 验收 |
| --- | --- | --- |
| `cvt.f32.bf16`、`cvt.rn.bf16.f32`、`cvt.rn.bf16x2.f32` | 扩展类型和转换；分别得到正确 FP32、单 BF16、两个 BF16 的位模式 | [数值](../validation/stage-1.md#numeric) |
| `shfl.sync.bfly/idx/up.b32` | 扩展显式 mask、数据交换和可选 predicate 输出；原地交换不读到已被覆盖的输入 | [线程协作](../validation/stage-1.md#collectives) |
| `vote.sync.ballot.b32` | 在已有 vote 上接通 member mask，返回参与线程的真假位图 | [线程协作](../validation/stage-1.md#collectives) |
| `redux.sync.add.s32`、`elect.sync` | 新增集合求和和唯一 leader；保留所需双输出及 `_` 丢弃数据输出 | [线程协作](../validation/stage-1.md#collectives) |
| `bar.warp.sync` | 在已有 bar 家族内增加 warp 范围，等待指定线程并保证它们的访存顺序 | [warp 屏障](../validation/stage-1.md#warp-barrier) |
| `bfind.shiftamt.u32`、`%lanemask_le` | 扩展最高有效位查找的返回方式，补特殊寄存器取值 | [位运算](../validation/stage-1.md#bits-loads) |
| `ld.global.L1::evict_first.b32/b8`、`ld.global.L1::evict_last.b32` | 接受实际缓存提示，保留正确宽度、地址、谓词和访存请求 | [带提示的读取](../validation/stage-1.md#bits-loads) |
| H100 PTX 的 SHFL 时间参数 | 补齐第六项，结果依赖和发射间隔真实进入周期模型 | [时序](../validation/stage-1.md#timing) |
| 普通算术、分支、load/store、shared atomic、`bar.sync` | 复用已有实现，随真实消费者回归；不宣称全部旧指令已通过 | [真实内核](../validation/stage-1.md#consumers) |

## 已核实的静态事实与影响

这里的“影响”说明代码为什么需要处理；不是当前 workload 已运行复现的错误。

| 当前代码事实 | 对当前目标意味着什么 | 处理位置 |
| --- | --- | --- |
| 词法/类型表没有所需 BF16 类型；`cvt_impl` 当前只读一个源操作数 | 不能把 BF16 当 FP16，也不能照旧入口只取打包转换的第一个值 | 下文数值转换 |
| `shfl_impl` 用已执行线程数选择 lane，逐次读其它线程寄存器并立刻写回；没有读取第四个源 member mask，可选 predicate 留有 TODO | 部分 mask 的真实 lane、原地交换及双输出都需要接通，不能只接受 `.sync` 拼写 | 下文集合操作 |
| `vote_impl` 使用函数静态累积状态和当前活动掩码，没有按显式 member mask 归组 | 新同步形式需要按本次 warp/参与集合计算；不能直接把 trace 或活动线程集当答案 | 下文集合操作 |
| `bar_impl` 的 `.sync` 操作数按 CTA barrier 编号处理 | `bar.warp.sync -1` 的 -1 是参与位图，不能当 CTA 屏障编号 | 下文 warp 屏障 |
| `bfind_impl` 的 shiftamt 变换仅为注释；向下循环的索引是无符号类型 | 需要补返回值；输入0时当前循环不能按预期在 -1 结束，属于可直接定位的实现问题，尚未运行复现 | 下文位运算 |
| `%lanemask_le` 取值调用 `feature_not_implemented`；H100 integer 时间配置仅五项，而 decode 读取六项并访问第六项 | 前者缺功能值；后者 SHFL 参数没有被完整赋值，不能据此评价 shuffle 时间 | 下文位运算/时序 |

<a id="numeric"></a>

## 1. BF16 转换：保留位表示、舍入和两个值的顺序

BF16 是16位浮点格式，与32位 FP32 共用相同的指数位宽，但小数精度更少。它与同为16位的 FP16 编码不同。`cvt.rn` 的 rn 表示取最近可表示值；恰在两个值中间时，选择保留部分最低位为偶数的那个，不能总是截断。

例如 FP32 位模式 `0x3f808000` 在两个 BF16 值中间，结果为 `0x3f80`；`0x3f818000` 的结果为 `0x3f82`。转换错误会改变矩阵输入，也可能改变分数接近时的路由选择。

### 具体改法

1. 在 `ptx.l`、`ptx.y`、`ptx_ir.*` 增加当前需要的 BF16/BF16x2 类型识别和宽度信息，分别为16/32位。寄存器容器继续使用已有 `u16/u32` 位存储，不新增主机 BF16 对象，也不把 BF16 映射成 `F16_TYPE`。
2. 在 `cvt_impl` 进入旧通用转换表前，分派这三种转换。旧表按原类型编号索引；不能新增编号后直接越界访问，也不为未使用的 BF16 算术补整张转换矩阵。
3. `cvt.f32.bf16` 将源16位放到32位结果高半部，低16位清零；用位复制写入结果，不做整数数值转换。真实代码可能把结果放入 `.b32` 寄存器，再通过 `mov.b32` 解释成 FP32，这条路径也要保留。
4. FP32→BF16 的有限值可用无符号位运算：`(bits + 0x7fff + ((bits >> 16) & 1)) >> 16`，取结果16位。指数全1的值单独按 PTX 规则保留无穷/NaN类别，不能让舍入把 NaN 变成零或无穷；不扩展成完整浮点运算库。
5. `cvt.rn.bf16x2.f32 d,a,b` 必须读取两个 FP32 源：转换 a 后放高16位，转换 b 后放低16位。预解码必须把 a、b 都登记为读取依赖，目的为一个32位寄存器；不能只改功能函数而漏掉第二个源的调度依赖。[官方转换规则](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

**SASS 复用：**借鉴 Hopper `F2F/F2FP` 的普通 ALU 分类，复用现有寄存器依赖和转换执行管线。trace 不生成数值，三个转换的结果必须由 PTX 自己算；无需实现 BF16 tensor core，本阶段也不更改 WGMMA。

<a id="collectives"></a>

## 2. shuffle、ballot、redux 和 elect：先确定参与者，再计算结果

shuffle 交换线程寄存器，例如 lane0 读取 lane16 的数；ballot 把一组真假值变成位图；redux 对一组整数求和；elect 从参与者中选一个线程负责后续单次操作。**同步集合与当前正在执行的线程集合不必相同**，不能用 `member_mask & active_mask` 缩小必须等到的参与者，再宣布完成。

### 解析、功能执行和依赖如何衔接

1. 保留 shfl 的四个源 `a,b,c,membermask`；vote 为 predicate 与 membermask；redux 为整数与 membermask；elect 为 membermask。`.sync` 要保存为本类指令的同步属性，不能因为共用选项解析把它们误分类成 CTA barrier。
2. 接入 `core_t::execute_warp_inst_t` / `ptx_exec_inst` 的 warp 执行上下文，按**实际 lane 编号**收集本次指令参与者的输入。先读取指令 predicate，再保存参与者的源值和 mask，最后统一分发结果；不能靠 `done_threads` 的计数猜 lane，也不能在全 warp 输入尚未读完前覆盖源寄存器。
3. 首选在一次已汇合的动态 warp 指令中用最多32项临时输入完成计算。对当前真实内核遇到的分批到达，若旧 SIMT 调度不能保证合法汇合，再补每 warp 的最小到达状态：指令 PC、同一 PC 的执行轮次、member mask、arrived mask 和输入值。PC 是指令位置，轮次区分循环中反复执行的同一位置；完成后清除该次状态。
4. 分批到达时，已到达 lane 留在等待点，尚未到达的合法分支仍须能执行；不能挂起整个 warp 后又等待它自己的其它分支。这里是实现当前消费者的约束，**尚未确认需要重写 SIMT 调度**。执行 agent 先用真实 top-k 的 mask/控制流取得证据；已有汇合路径足够就不新增等待调度器。
5. 结果写回及预解码覆盖数据、可选 predicate 两个目的。复用现有双目的操作数表示，并核对 `set_operand_value` 对 predicate 的内部编码：当前低位0表示真，1表示假。对 `_` 不分配寄存器、不建立写依赖。不能把布尔真直接当内部数值1写入。
6. 仍让该指令经过正常流水、scoreboard（记录哪些寄存器尚未就绪）和写回。功能值在发射阶段计算不代表后续指令可以跳过它的时间依赖。

### 每条指令的确定算法

| 指令 | 要实现的算法 |
| --- | --- |
| shfl 三种模式 | 保留旧索引算法，明确 `b=b&31`、`seg=(c>>8)&31`、`clamp=c&31`；`lo=lane&seg`，`hi=lo\|(clamp&(~seg&31))`。bfly 的源为 `lane^b`，idx 为 `lo\|(b&(~seg&31))`，两者判断源≤hi；up 的源为 `lane-b`，判断源≥hi。up 的 c=0 与 bfly 的 c=31 含义不同，不能统一改成一个边界公式。边界判断为假时读自己的原始输入；predicate 只反映边界判断，不是“源线程是否参与”的检查 |
| `vote.sync.ballot.b32` | 对参与且未退出的 lane，按逻辑 predicate 真值设置对应位；其余位为0，向参与者广播同一个结果。支持实际使用的 `!predicate`，不要沿用跨 warp 的函数静态累积状态 |
| `redux.sync.add.s32` | 对参与者的32位值用无符号累加保存低32位，再把同一位模式写给参与者；避免用 C++ 有符号溢出实现 PTX 截断 |
| `elect.sync` | 确定性选 member 集合中最低的有效 lane；所有参与者得到同一个 leader 编号，仅该 lane 的 predicate 为真。同一个 mask 重复选择保持一致，不要求与硬件选中相同编号 |

算法依据：[workload 指令说明](../../workloads/moe/docs/ptx-instructions.md)及[官方 shuffle](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。合法源必须参与；未定义的非参与源不补零恢复，也不建立非法 mask 测试矩阵。

**SASS 复用：**复用 warp 发射、寄存器端口、依赖和 ALU/专用流水。Hopper 映射中 SHFL/VOTE 为 ALU，REDUX 为专用单元4，ELECT 为专用单元1；可借用这些时间分类，具体接法见下文。SASS 的活动掩码来自已有轨迹，不能代替 PTX 的 predicate 求值和汇合。

<a id="warp-barrier"></a>

## 3. warp 屏障：等指定线程，不扩大成整个 CTA

`bar.warp.sync mask` 保证指定线程到达后，才越过这个同步点；同时保证这些线程此前的访存与之后的访存按要求排序。它与只交换寄存器的 shfl/vote 不同。例如线程先写 shared，再通过这个屏障读取另一参与者写入的位置，读者必须看到本轮数据。

具体实现：在 bar 解析中保存 warp 范围；`bar_impl` 按 mask 求值，不按 barrier id 求值。复用上节实际 lane/到达集合处理，以 CTA、warp、本次屏障身份隔离状态。当前冻结形式为全 mask `-1`，先保证这条真实路径；另用一个合法部分 mask 小测试确认没有误接 CTA barrier。

功能存储沿用同 CTA 的 shared；屏障只在必要的先前访存顺序已满足后放行。时序层核对已发射 store/相关访存的完成记录，复用已有等待和调度入口；不能仅因功能存储早写了值，就把屏障当零成本空操作。无需实现一般弱内存模型，也不增加未使用的跨设备可见性机制。

**SASS 复用：**可借鉴已有 barrier 的到达、等待和释放位置，但不直接调用“等待整个 CTA”的分支。普通 `bar.sync` 保留原 CTA 语义，并随真实排序/重排程序回归。

<a id="bits-loads"></a>

## 4. 位运算和缓存提示：接通当前使用的具体形式

**`%lanemask_le`：**返回编号不大于自己的线程位图。lane0 为 `0x00000001`，lane16 为 `0x0001ffff`，lane31 为 `0xffffffff`。在 `ptx_thread_info::get_builtin` 补取值，可用 `0xffffffffu >> (31-lane)`，避免左移32位。复用既有特殊寄存器读取及普通 move 计时，不补其它未使用的 lanemask。

**`bfind.shiftamt.u32`：**bfind 原本找最高有效1所在位置；shiftamt 返回把它移到 bit31 所需的左移位数。例如1→31、8→28、`0x80000000`→0；0保留 `0xffffffff`。增加 `.shiftamt` 解析和指令属性，在旧查找后作变换；旧循环改成能正常结束的有符号下标，掩码使用对应宽度的无符号值。只修这条共用循环及本次变体，不扩大成所有位操作重写。SASS/普通整数 ALU 时序可以复用。

**`L1::evict_first/last`：**这是建议缓存先淘汰或后淘汰某类数据的提示，不是新地址空间，也不改变读取的字节数。在 lexer/选项解析增加这两种形式；`ld_impl` 仍按 b8/b32 读取1/4字节，预解码和访问生成仍为普通 global load。先保留已有 L1/L2 缓存通路，明确记录尚未细化逐出优先级；不要把 evict_first 自动翻译成绕过 L1 的 `.cg`。可复用 SASS 的缓存/合并/返回模型，只有主要流量或时间差异指向提示策略时再细化。

<a id="timing"></a>

## 5. 时间参数与真实消费者同时接通

latency 是结果多久可用，initiation interval 是同一执行单元多久能接下一条操作；两者不能混成一个延迟。当前 H100 PTX integer 配置的顺序是 ADD、MAX、MUL、MAD、DIV、SHFL，最后一项缺失。

1. 沿 Hopper SASS SHFL 的 ALU 分类，以已有 integer ALU 的4周期 latency、2周期 initiation 作为初始模型，分别把两行补为 `4,4,4,4,21,4` 和 `2,2,2,2,2,2`。这是复用现有模型的初值，不是新增硬件实测结论，不改其它五项。
2. 新增 redux/elect 在 PTX 的 decode 中接到对应既有专用流水；H100 PTX 若未启用单元1/4，只搬用外层 `trace.config` 对应 BRA/UDP 单元配置和时间参数：分别4/4与4/1。不复制整个 trace 配置，更不提前引入阶段6的 GMMA 资源。PTX decode 自己设置所用参数，不能期待只有 trace reader 才读取的选项自动生效。
3. 转换、vote、bfind 和 load 继续用已有对应流水；mask 寄存器也属于源依赖，双目的 predicate 也属于写依赖。warp 屏障的等待来自参与者及访存状态，不用一个随意常数替代等待。
4. 用有依赖 shuffle 链检查 latency，用多条独立链/足够 warp 检查 initiation；仅为微测改变第六项，确认参数影响对应行为。真实内核看主要时间和访问量，不为每条低占比指令追求精确硬件周期。

先完成 BF16→FP32、ballot/lanemask、bfly/up/redux，运行转换→top-k/排序→计数/映射；再补所需反向转换和一个 SwiGLU。只由专家使用的 idx/elect/bar.warp/打包转换可晚于首批普通内核交付，但最迟在阶段6相关消费者前完成，报告不得提前写“阶段1全部通过”。

## 暂缓范围与实际记录

| 暂缓项 | 暂缓原因和重开条件 |
| --- | --- |
| 其它 BF16 算术、redux/vote 模式、其它特殊寄存器 | 冻结九例没有相应消费者；实际指令出现再扩展 |
| 完整独立线程调度、任意分歧组合和非法 mask 恢复 | 当前先验证真实合法汇合；消费者确需分批到达就局部补齐，不能跳过卡住的真实内核 |
| 全部特殊浮点组合、NaN载荷穷举 | 保留基本类别和合法舍入，使用少量定向值；不新增无关随机压力测试 |
| eviction hint 精细优先级 | 数据和正常缓存通路优先；主要时间/流量证据指向它再重开 |

阶段1仅需一组真实转换、路由、计数/重排及相关微测，不逐一单测110模块。首个真实普通内核就记录初始化、模拟、比较墙钟和峰值 RSS；这些样本不能推算 WGMMA 的 CPU 计算成本。

## 源码和证据入口

| 位置 | 执行 agent 查看/修改的具体内容 |
| --- | --- |
| [ptx.l](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l)、[ptx.y](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y)、[ptx_ir.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.cc)、[ptx_ir.h](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h)、[opcodes.def](../../gpu-simulator/gpgpu-sim/src/cuda-sim/opcodes.def) | 类型/选项/新 opcode、双目的和源依赖表示 |
| [instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) | `cvt_impl`、`shfl_impl`、`vote_impl`、`bar_impl`、`bfind_impl` 及新增语义 |
| [abstract_hardware_model.cc](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc)、[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc)、[ptx_sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_sim.cc) | warp逐线程执行、predicate处理、预解码、时间分类和特殊寄存器 |
| [H100 PTX配置](../../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config)、[Hopper opcode表](../../gpu-simulator/ISA_Def/hopper_opcode.h)、[H100 trace配置](../../gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config) | SHFL第六项及可复用流水映射；SASS初值的证据 |
| [workloads.json](../../workloads/moe/artifacts/workloads.json)、[模块用途](../../workloads/moe/docs/ptx-modules.md) | 模块身份、entry、指针参数位置和真实消费者 |

实际改动、指令/接口、复用及自测填写到 [阶段 1 实际实现记录](../execute/implementation/stage-1/README.md)；独立运行结论填写到 [实际验收记录](../execute/validation/stage-1/README.md)。本文件保留实现方案，不重复维护实际状态。
