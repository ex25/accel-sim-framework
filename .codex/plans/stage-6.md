# 阶段 6：矩阵布局、64 项结果与异步 WGMMA

状态：仅方案；本轮无实现、修复或运行验收。[对应验收](../validation/stage-6.md)。基本计算依赖阶段 1，不以 TMA、普通 cp.async 或 ldmatrix 全部完成为前置。

## 本阶段解决什么

专家 GEMM 的一次 WGMMA 由四个 warp、128 个线程协作，输出 64×128 个数，每线程保存 64 项 FP32 结果。必须把这些结果完整保存并允许连续异步累加；只扩大 parser 或每次算完才发下一次，都不能满足当前内核。

指令表示（IR）保存操作数；scoreboard 记录哪些寄存器尚未写完；collector 收集执行所需的寄存器值；bank/端口模型限制每周期能读写多少。这几处都必须认识完整的64项结果。

## 当前事实及影响

| 静态事实 | 对实现的影响 |
| --- | --- |
| tuple、decode、指令数组、collector 等有 8/32 项上限 | 仅 parser 接受 64 项仍会在依赖或写回漏掉结果；要沿整条链路修改 |
| 普通 scoreboard 把待写寄存器当作后续读/写冲突 | 不能直接用于同一 WGMMA 累加器的连续异步累加，否则会逐条等待 |
| 旧 WMMA 是 16×16 FP16 路径 | 可借鉴协作代码，不能当 BF16 m64n128k16 数值算法 |
| GMMA 资源在外层 trace.config，PTX 入口未自动加载 | 需要显式接入 PTX 专用资源，否则时间可能落入不匹配的管线 |
| 已有 GMMA wait 初次比较方向不一致、后续仅看本 warp outstanding | 需要按四 warp 的组完成重做适配，不能只改 opcode 名称 |

均为静态核查，不代表已有 PTX WGMMA 错误运行记录。[容量及等待依据](../hopper-notes.md#existing-waits)

## 指令清单

| 指令/链路 | 动作 | 状态 |
| --- | --- | --- |
| `ldmatrix.sync.aligned.x4.m8n8.shared.b16`、`stmatrix.sync.aligned.x4.m8n8.shared.b16` | 当前非转置四矩阵的 lane 映射与 shared 请求 | 待新增 |
| `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16` | A/B shared 描述符解码、BF16×BF16、FP32 累加、64 项结果 | 待新增 |
| `wgmma.fence.sync.aligned`、`wgmma.commit_group.sync.aligned`、`wgmma.wait_group.sync.aligned` | 寄存器顺序、异步提交、wait1/0 | 待新增/适配 |
| `setmaxnreg.dec.sync.aligned.u32`、`setmaxnreg.inc.sync.aligned.u32` | 当前 dec40/inc232、组同步及额度等待唤醒 | 待新增 |
| `fence.proxy.async.shared::cta` | shared→async 发布；与阶段 5 共用 | 待新增或复用另一线交付 |
| parser→decode→scoreboard→collector→写回/统计 | 64 输入/输出相关项完整传递 | 待扩展/修正 |

## 如何实现

### 1. 先打通完整寄存器链路

列出所有固定上限和遍历，不只改 `MAX_OUTPUT_VALUES`：tuple 构造/解析、手写 decode、`in/out`、arch `src/dst`、输入输出计数、scoreboard、collector、写回端口和统计。累加器在 scale_d 为真时同时是输入，64 项加上 descriptor/predicate 等实际输入也要能表达。按实际数量循环，不给所有普通指令虚增 64 次寄存器访问。

优先在现有表示上统一容量/计数；无必要不做整个 IR 容器化重构。现有 bank/端口照用；`opndcoll_rfu_t::writeback` 已用清除已服务目标的方式保存进度，扩容时保留它，并检查依赖释放与实际完成一致，不另造写回重试系统。用 64 项组件检查证明没有静态容量缺口，再由真实 WGMMA 消费全结果验收。

### 2. 按当前布局实现矩阵存取

`ldmatrix/stmatrix.x4`：lane 0..7、8..15、16..23、24..31 分别提供四张矩阵的行地址；每张矩阵第 i 行的 8 个 b16 值分给 lane 4i..4i+3，每个 lane 一个 packed b32。先收齐 32 lane 的地址/输入，再读写，避免按每线程独立连续访问处理。

每 warp 实际 512B shared 访问进入已有 bank/端口模型，不以每 lane 一个标量访问替代。加载与存储分别用手工坐标检查，不能只验证互相往返，因为两者同样错位可能互相抵消。

WGMMA 使用公开格式的 **64 位矩阵描述符**，与阶段 5 内部 **128B tensor map** 不同。解码起始地址、leading/stride offset、swizzle 等实际字段，按当前 A/B 布局映射 shared 元素。落实前对照冻结 PTX 构造这些位的代码及[原指令说明](../../workloads/moe/docs/ptx-instructions.md)，不能把 descriptor 直接当指针，也不能假设 PTX 寄存器编号递增就是矩阵坐标递增。

地址、leading offset、stride offset 分别取 bits 0..13、16..29、32..45，再左移4恢复字节单位；bits49..51是swizzle base offset，bits62..63选择swizzle。当前布局实现须同时使用这些字段，不能只取低位基址。具体对应关系见[PTX 8.7矩阵描述符格式](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。

### 3. 一次逻辑运算只计算一次

以 CTA 内 warpgroup 身份和动态发起序号收齐四 warp；PC 不能单独区分循环中的多次计算。收齐之前不执行四份矩阵乘法。捕获 A/B descriptor、scale_d 与结果寄存器对应关系，在建模的 shared 读取点取得输入；之后保留本次仍需要的数据和结果，避免 shared 复用改变已经读完的操作。

当前只实现 m64n128k16、BF16 输入、FP32 累加，scale_a/b=1、trans_a/b=0，scale_d 的真假都必须支持。先用直接清楚的循环实现独立功能计算，再按 lane/tuple 映射分发 8192 个结果。复用必要 tile 状态，不为每条 warp 指令复制一整份矩阵。硬件累加顺序不保证与 CPU 标量逐项 FMA 逐位相同；精确小值先验布局，真实输出仍按既定容差，不擅自放宽。

### 4. 异步计算、依赖和时间一起接入

每个 warpgroup 维护未提交操作、已提交组及完成进度。普通寄存器写入经 `wgmma.fence` 排序后作为初始累加器；后续同一累加器的 WGMMA 读取前一逻辑操作的有序结果，允许继续入队，不走普通 WAW 的逐条等待。只对这一受控路径放行，不能全局关闭 scoreboard。

普通指令仍受待完成结果依赖约束；异步操作完成及输出服务后发布对应值，依赖引用/所有权不能因较早操作完成而误释放后续写入。使用现有 scoreboard 加最小异步归属信息，避免另建一套所有指令的依赖系统。

commit 定义组边界而不等待；wait1 只要求旧组完成，wait0 排空已提交组。显式 WGMMA opcode 选择自己的组；不能把 TMA bulk 的等待状态混入。shared 输入何时读完与结果何时完成须一致记录，正式程序在合法等待后复用输入，不能靠错误读取未完成数据制造重叠。

沿现有 GMMA 专用资源接入延迟、发起间隔及写回；配置进入 PTX 实际加载文件。一次逻辑运算为 `2×64×128×16=262144` math ops。可以保留四 warp 的管线发射，但总工作量不能乘四，也不能只计一个 warp 的资源成本。现有 SASS 64/64 初值先按此口径核对，不当已校准参数。

这里的64/64来自 `trace_driven.cc` 对N=128的动态计算；外层 `trace.config` 的GMMA默认值为128/32，不能只复制配置就认为得到了本形状的64/64。两处口径都要在PTX路由中明确。

### 5. 配额与虚拟寄存器值分开

采用每 CTA 寄存器池和每 warpgroup 配额。当前资源为 384×168=64512；生产者 dec 到40，归还 `128×(168-40)`，两个消费者各 inc 到232，共需 `2×128×(232-168)`，两者相等。inc/dec 的立即数是新额度，不是增减量。

同一 warpgroup 一致到达才改变额度；inc 不足时仅暂停申请组，生产者仍可执行 dec，归还后唤醒。额度从本 CTA 池取得，不跨 CTA 借；物理配额不作为 PTX 虚拟寄存器编号上限。占用使用核实的物理资源，而不是 PTX 声明的全部虚拟寄存器数。

## 交付顺序与 SASS 复用

先容量链路与单次 WGMMA（普通 shared 准备输入）；矩阵存取可独立补。单次通过且阶段 5 非组播可用，立即参加[首次联测](../validation/stage-5.md#early-integration)。再交付连续累加、wait1/0、配额和双缓冲；首次联测不等这些全部完成。

可复用 SASS 的 GMMA 资源、issue/completion 入口、bank/端口和统计；旧 WMMA 只借鉴协作组织。必须新增 PTX 数值、descriptor/lane 布局、四 warp 逻辑组和完整寄存器链路。现有本 warp wait、固定少量结果和 trace 给出的依赖信息不能直接沿用。

## 暂缓与源码入口

暂缓其它 WGMMA 形状/类型/转置、额外物理 bank 重分配、非法 warpgroup 分歧处理和未使用的 setmaxnreg 数值组合。当前 64 项、连续累加、wait1/0、额度不足等待与归还唤醒全部必须做。只在真实主要耗时差额指向暂缓细节时重开。

[ptx_ir.h](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h)、[ptx.y](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y)、[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc)、[abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h)、[scoreboard.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.cc)、[shader.h](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h)、[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)、[trace.config](../../gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config)。

## 实际交付记录

待填写：指令和容量链路实际改动、单次/异步/配额/联测各自结果、时序近似、墙钟/RSS、提交与报告。当前：无。
