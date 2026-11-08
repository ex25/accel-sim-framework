# 阶段 6 实现方案：完整矩阵结果、异步计算与寄存器额度

本文件是待执行方案，当前没有本阶段实现或运行验收。[配套验收](../validation/stage-6.md)。本阶段可先用普通 shared 写入准备矩阵，与阶段 3～5 分别推进；两条开发依赖线最终用于同一个专家内核。

## 完成后应当能做什么

WGMMA 是由同一 CTA 内连续四个 warp、共 128 个线程协作的异步矩阵乘加。当前指令计算 `D[64,128] = A[64,16] × B[16,128] + D[64,128]`，8192 个 FP32 结果分给 128 个线程，每线程 64 项。“异步”表示发起后可以继续执行允许重叠的工作，直到相应 wait 才要求旧计算完成。

本阶段既要算对全部结果，也要保留这种重叠。把每次 WGMMA 改成“算完、写完、再发下一次”，即使数值正确，也会把专家内核的时间算长。

```text
四个 warp 到齐，取得本次矩阵描述符和累加器
  → 按 shared 布局读取 A/B → 计算一份逻辑结果
  → 按 GMMA 资源推进周期 → 分批服务寄存器写回
  → 对应操作完成 → 满足条件的 wait 返回
后续合法 WGMMA 可排在同一累加链上；普通指令不能提前读取未完成结果。
```

## 已核实事实及影响

以下是源码静态核查，未在当前 PTX workload 上运行复现。

| 当前事实 | 对当前目标的影响 | 本阶段处理 |
| --- | --- | --- |
| tuple/手写 decode 有 8 项边界，公共寄存器数组另有 24/32 等边界 | parser 即使收下 64 个名字，后续仍可能漏掉寄存器或依赖 | 沿解析、读入、依赖、写回完整扩容 |
| 普通 scoreboard 记录尚未写完的寄存器，遇到读写冲突就阻塞 | 连续 WGMMA 累加不能直接套用普通逐条完成规则 | 仅为同一 warpgroup 的有序异步累加增加归属信息 |
| collector 会清除已服务的写回项，并在端口冲突时返回 false；上层 `shader_core_ctx::writeback` 忽略返回值且提前减少 GMMA outstanding | 扩成 64 项后，不能在还有结果未服务时宣布整条指令完成 | 保留分批进度，最后一批完成后才释放依赖和完成计数 |
| 旧 WMMA 处理 16×16 FP16；已有 SASS GMMA 无本次 PTX 数值执行 | 不能把旧路径更名就当 BF16 64×128×16 已支持 | 借用组织和性能资源，新增当前形状的数值/布局 |
| SASS GMMA 的发起、outstanding 和等待以 warp 为入口，PTX 配置也不会自动加载外层 trace.config | 四 warp 的共同完成及当前 GMMA 时间需明确接通 | 显式 warpgroup 状态、PTX 路由与配置 |

## 指令与交付范围

| 指令/链路 | 要得到的行为 | 状态 |
| --- | --- | --- |
| `ldmatrix.sync.aligned.x4.m8n8.shared.b16` | 四张 8×8 b16 矩阵，按 lane 分成每线程四个 b32 | 待实现 |
| `stmatrix.sync.aligned.x4.m8n8.shared.b16` | 按上述分布把寄存器写回 shared | 待实现 |
| `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16` | 解码 A/B 描述符，计算、保留并传递全部 64 项/线程 | 待实现 |
| `wgmma.fence.sync.aligned` | 把此前普通寄存器操作与后续 WGMMA 的寄存器使用排序 | 待实现 |
| `wgmma.commit_group.sync.aligned` | 为本组未提交操作划定完成组，不等待 | 待实现 |
| `wgmma.wait_group.sync.aligned` | 当前 N=1/0，等待较旧组或全部已提交组 | 待实现/适配 |
| `setmaxnreg.dec/inc.sync.aligned.u32` | 当前 dec40/inc232，四 warp 到齐后变更额度，不足则等待归还 | 待实现 |
| `fence.proxy.async.shared::cta` | 普通 shared 写入能被随后异步矩阵读取看到 | 与阶段 5 共用一次实现 |
| 64 项解析、依赖、读写端口、完成统计 | 不能截断；端口耗时与实际服务相符 | 待扩展/修正 |

<a id="registers"></a>

## 1. 从 64 个名字到 64 个完成结果

寄存器 tuple 是 PTX 中 `{%f0, %f1, ...}` 这样的列表。IR 是模拟器解析后保存的指令；scoreboard 记录未完成的读写依赖；collector 按寄存器 bank/端口收集输入或服务输出。这些对象保存的是不同环节的信息，不能只改其中一个数组。

### 具体改法

1. 从真实模块 16 取一条完整 WGMMA，列出目标 tuple、两个描述符、scale_d 谓词及其余立即数。保持源文本 tuple 的顺序；实际寄存器名字可以递减，不能按编号排序后当成矩阵坐标。
2. 更新 `operand_info` 的 tuple 表示/构造、`ptx.y` 列表规则、`ptx_decode_inst` 手写展开、公共 `in/out` 与 `arch_reg.src/dst`、collector 存储及计数。用实际数量遍历；累加器需要读入时，64 项加描述符/谓词也必须装得下。描述符为一个逻辑 64 位 PTX 寄存器，沿现有宽寄存器约定处理，不凭字节数擅自变成两个逻辑编号。
3. 沿 scoreboard 的 reserve/check/release 和 collector 的输入/输出路径逐个替换残留的小容量遍历。优先统一现有容量与计数，不重构整个 IR；普通短指令不能因此产生 64 次虚构读写。
4. 复用 collector 对已完成输出设置 `arch_reg.dst=-1` 的进度机制。PTX WGMMA 在 collector 返回 false 时保留指令，下一周期继续未完成项；不得再次读取输入、重算矩阵或重复统计已完成项。
5. 上层只在所有输出服务结束后减少 outstanding、释放对应依赖、记完成并清空指令。依赖信息使用保留的逻辑目标列表，不能用已清空的端口进度数组代替。现有 LSU 条件写回可借鉴。
6. 同时接通写回停顿对 GMMA 结果入口的容量约束：优先复用现有流水寄存器的 ready/占用状态，不能让不允许停顿的上游覆盖尚未写完的结果。如果必须在 GMMA 出口保留待写回项，只保存该路径所需的有界状态，不另建所有指令通用的重试系统。

**SASS 复用：**沿用 bank 映射、读写仲裁、collector 和流水资源。SASS trace 的少量依赖字段不能代替 PTX 的真实 64 项目标。若改了共用写回逻辑，验证受影响 SASS；不要无依据把所有旧指令改成新执行方式。

<a id="layout"></a>

## 2. 明确矩阵的地址和线程分工

### ldmatrix/stmatrix 的四张小矩阵

对矩阵编号 `q=0..3`，lane `8q+r`（`r=0..7`）提供第 r 行的 shared 地址，该行包含八个 b16 元素、共 16B。lane `4r+c`（`c=0..3`）的第 q 个 b32 寄存器保存该行第 `2c` 个元素的低 16 位和第 `2c+1` 个元素的高 16 位；stmatrix 做对应存储。

收齐一个 warp 的地址/值后执行这个分配。每 warp 四张矩阵实际访问 512B，把每个地址交给现有 shared bank/端口计时，不把它简化成每线程一个标量访问。加载和存储各自验坐标，不能只做往返抵消错误。

**SASS 复用：**复用 shared 访存合并、bank 冲突和服务周期。新增 PTX 的 lane 分布和真实 b16 值读写；不另造 shared 缓存。

### WGMMA 的 64 位矩阵描述符

这里的描述符是一个 64 位值，用来说明 shared 中矩阵从哪里开始、各组元素相隔多少，以及是否打乱存放位置以减少 bank 冲突。它不是阶段 5 的 128B tensor map，也不是普通指针。

| 字段 | 解码 | 用途 |
| --- | --- | --- |
| bits 0..13 | 左移 4 得到字节起点 | 矩阵在 shared 的起始位置 |
| bits 16..29 | 左移 4 得到 leading offset（LBO） | 无 swizzle 的当前 K-major 布局中，跨 K 小块的间距 |
| bits 32..45 | 左移 4 得到 stride offset（SBO） | 跨下一组八行的间距 |
| bits 49..51 | base offset | 非默认起点的 swizzle 相位 |
| bits 62..63 | 0 无；1=128B；2=64B；3=32B swizzle | 选择地址排列；优先覆盖真实 64B/128B |

当前 BF16、`trans_a=trans_b=0` 使用 K-major：A 的行是 m，B 的行是 n，行内沿 k 存放。不能把 B 的 shared 存储当普通 `B[k][n]` 连续行。

对 BF16，先在**未作 swizzle 的规范坐标**中求相对字节位置。无 swizzle 时，当前形状可用 `u=(r%8)*16+(r/8)*SBO+2*(k%8)+(k/8)*LBO`，整数除法；64B/128B swizzle 时，K=16 落在一个行块内，使用 `u=(r%8)*W+(r/8)*SBO+2*k`，W 为 64/128。**K-major 的 swizzle 形式不使用 LBO**，不能为了“使用所有字段”硬加这个偏移。

再按描述符起点和 base offset 确定 swizzle 相位，映射到实际 shared 地址。复用阶段 5 的 16B 块排列运算，但描述符的子矩阵起点需要自己的解码：不能把起点加上 u 后一律再 XOR，也不能每次从相位 0 开始。实现前把冻结 PTX 生成的 64B/128B 描述符各解出一张 16B 块地址表，特别包含非零起点；与公开规范的相位规则和 TMA 所写地址对应后再接入执行。此步骤是实现检查项，尚无运行结果。

只写一个由 TMA/WGMMA 共用的低层 swizzle 位排列辅助函数；tensor map 的全局寻址与矩阵描述符的字段解码仍各归原处。格式和布局依据：[PTX 矩阵 shared 布局](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout)。

### 8192 项结果如何分给线程

令 CTA 内该 warpgroup 的线程号 `t=0..127`、tuple 位置 `j=0..63`，`w=t/32`、`l=t%32`。当前 FP32 结果对应：

```text
m = 16*w + l/4 + 8*((j%4)/2)
n = 8*(j/4) + 2*(l%4) + j%2
该线程 tuple[j] = D[m,n]
```

例如线程 0 的第 0、1、2、3 项分别是 `(0,0)、(0,1)、(8,0)、(8,1)`；第 4 项是 `(0,8)`。线程 127 的第 63 项是 `(63,127)`。循环按位置 j 分发，不看 `%f` 后面的数字大小。

**SASS 复用：**没有可直接复用的 PTX 描述符/数值布局；旧 WMMA 可借鉴收齐线程和访问 shared 的组织。FP32/BF16 基础值转换复用阶段 1。

<a id="async"></a>

## 3. 一次计算、连续累加和异步组

### 状态与功能执行

用 `(kernel、CTA、CTA 内 warpgroup、动态序号)` 标识一次逻辑操作。CTA 内每四个连续 warp 组成一个 warpgroup；不能直接用整个 SM 的 warp 编号除以四代替。循环中的同一个 PC 会多次执行，因此 PC 不是动态身份。

在现有线程求值入口收集四 warp 的描述符、scale_d、目标 tuple，并在四 warp 到齐后形成一次运算。只实现当前形状与 `scale_a=scale_b=1、trans_a=trans_b=0`；scale_d 为 0 时不读旧 D，为 1 时使用有序的旧累加值。

在建模的 shared 读取完成点取得本次 A/B 值，按 `m、n、k` 循环进行 BF16 转 FP32 和累加，再按上面的分布生成结果。一次逻辑运算只计算一份，不为四 warp 各算整张矩阵；保存仍在使用的输入/结果直到相应消费者不再需要，随后释放。CPU 花多久不决定 GPU 完成周期，完成由性能模型事件决定。

### 有序累加不等于逐条同步

- 在每个 warpgroup 保留未提交操作列表、已提交组序号和组内完成计数。`commit_group` 把此前未提交操作收进一个组，本身不等待。
- 对同一累加器链，后一条 WGMMA 可以使用前一条的逻辑结果并继续排队，即使前一条尚未完成全部周期/写回；它仍受 GMMA 发起资源和队列容量限制。
- 后继 WGMMA 使用的逻辑累加值与普通线程可读的已完成寄存器值分开保存。前一操作写回不能覆盖后一操作已生成的逻辑累加值；保留动态写者序号，最后一次写回之后普通消费才看到最终值。只保存仍被链上操作使用的结果，不按四 warp 复制整份矩阵。
- 这个例外只允许同一 warpgroup、同一有序累加链。scoreboard 为异步目标保留最新写者或未完成引用，早期写者结束不能释放后期写者的依赖。普通 add/load/store 等仍按已有依赖规则，不能提前读取结果。
- `wgmma.fence` 在四 warp 到齐后建立普通寄存器操作到 WGMMA 的顺序；它不代替 shared 的 proxy fence，也不代替等待已有 WGMMA 完成。
- `wait_group N` 在该 warpgroup 共同到达后，根据已提交序号等待所有比保留的最新 N 组更旧的组完成。N=1 允许最新组仍在执行，N=0 等完全部已提交组。组完成包括计算和全部结果写回服务，不能只看第一个 warp 的 outstanding。
- PTX 用自身 opcode 选择 WGMMA、TMA bulk 或普通 cp.async 的组状态，不沿用“最近一次出现的组类型”猜测队列。

这是当前消费者必需的并行语义；不建立通用任务图调度器，也不通过关闭 scoreboard 或每条隐式 wait0 来绕开。

**SASS 复用：**保留 GMMA 发射/完成事件及资源限制；四 warp 共同操作、动态累加归属、真实数值及完整写回必须补充。shared 发布沿阶段 5 实现，只保留一个功能入口。

<a id="timing"></a>

## 4. 计算资源与周期从一开始接入

当前一条逻辑运算计 `2×64×128×16=262144` 次 math op。四 warp 是参与者，不是四张额外矩阵。shared 输入请求、寄存器服务和计算流水分别记入自身资源，不能在一份 warp 费用之外又多加四份整组费用。

优先沿现有 SASS GMMA 的四 warp 发射方式接入 PTX，但以一个逻辑操作关联四个参与者，完成功能计算只做一次。核对每个参与者实际占用的专用资源、延迟与发起间隔，并在测试报告列出两种计数：逻辑矩阵操作数、warp 管线发射数。

`trace_driven.cc` 对 N=128 动态算出的延迟/发起间隔为 64/64；外层 `trace.config` 的 GMMA 默认是 128/32。PTX 必须显式选择当前形状的参数，把专用资源加入它实际读取的 H100 配置，不能只复制默认数值。沿已有 GMMA 资源绑定，避免再造一份独立 tensor 单元从而把容量算双倍。这些是初始建模参数，不是已校准的 Hopper 精确时序。

**SASS 复用：**专用执行单元、发起限制、统计和 shared/register bank 计时均可借鉴/复用；用微测先确认因果关系，真实 FC1/FC2 在阶段 7 校准。额外物理 bank 重分配暂缓。

<a id="quota"></a>

## 5. setmaxnreg：在同一个 CTA 内重新分配额度

这里的“额度”是物理寄存器预算，不是 PTX 能访问的最大虚拟寄存器编号。当前专家 CTA 有 384 线程，资源记录为每线程 168 个物理寄存器，共 64512。生产者 warpgroup 搬数据，两个消费者 warpgroup 算矩阵，所以生产者归还额度供消费者使用。

| 状态 | 生产者 128 线程 | 消费者 1 | 消费者 2 | CTA 空闲池 |
| --- | ---: | ---: | ---: | ---: |
| 初始，每线程 168 | 21504 | 21504 | 21504 | 0 |
| 生产者 dec 到 40 | 5120 | 21504 | 21504 | 16384 |
| 两消费者各 inc 到 232 | 5120 | 29696 | 29696 | 0 |

在 CTA 执行状态保存初始物理分配、三个组当前额度、空闲量和等待申请。立即数 40/232 是**新额度**：一次 inc 需要 `128×(232-168)=8192`，不是再申请 `128×232`。

四 warp 到齐后才改变额度；申请不足只暂停申请组，生产者和其它可执行 warp 继续调度。dec 归还后重新检查等待申请，满足时分配并唤醒；不从别的 CTA 借。CTA 的总物理驻留预算仍为 64512，不因内部归还就向 SM 虚构释放一部分容量。CTA 最终退出时按原分配归还一次。

初始化额度来自核实的 ptxas/资源信息，不来自 PTX 声明的虚拟寄存器个数。额度改变不删 PTX 值、不截断高编号寄存器；沿已有 bank/端口计时，不新增复杂物理搬家模型。

**SASS 复用：**已有 CTA 资源占用与归还逻辑可复用；当前按组的额度、等待和唤醒需要新增。资源事实及跨 SM 驻留仍按阶段 3。

## 实施顺序、暂缓项与源码入口

先完整寄存器链路、普通 shared 准备的单次 WGMMA；ldmatrix/stmatrix 可分别实现。单次矩阵和阶段 5 非组播 TMA 可用就立即参加[首次联测](../validation/stage-5.md#early-integration)，不等本阶段全部完成。随后补连续累加、wait1/0、额度及完整流水。

暂缓其它 WGMMA 形状/类型/转置、未消费的矩阵存取变体、额外物理 bank 重分配、非法 warpgroup 分歧的恢复处理。不把当前 64 项结果、非零描述符起点、wait1/0、额度不足等待列为 corner case；真实内核会用到。

| 入口 | 修改职责 |
| --- | --- |
| [ptx.y](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y)、[ptx_ir.h](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h)、[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) | 指令/tuple、解码与操作数完整性 |
| [instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) | PTX 矩阵求值、lane/descriptor 解码，与异步状态衔接 |
| [abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) | 共用指令数组与动态操作信息 |
| [scoreboard.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.cc) | 异步目标归属及普通依赖约束 |
| [shader.h](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h)、[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) | warpgroup、GMMA 发起完成、collector 写回、配额等待 |
| [trace_driven.cc](../../gpu-simulator/trace-driven/trace_driven.cc)、[trace.config](../../gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config) | 参照现有 GMMA 动态时间与资源设置 |

## 实际交付记录

待执行 agent 填写：各指令及容量链路的实际改动/提交，布局、异步、配额和联测报告，SASS 回归、时序近似、墙钟/RSS。当前无实现、修复或验证结果。
