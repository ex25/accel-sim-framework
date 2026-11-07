# Hopper PTX 调查依据与指令映射

2026-11-05 调查；2026-11-06 整理。本文件是**追溯附录**：保存当前代码与环境事实、可复用的源码位置、范围边界及 202 种指令的阶段归属；不另行定义路线、决定或验收规则。先读 [README](README.md)，再按需查[实施计划](hopper-plan.md)、[验证方案](hopper-validation.md)、[已确认的取舍](hopper-planning-decisions.md)。原始指令语义仍以 workload 的[指令清单](../workloads/moe/docs/ptx-instructions.md)为准。

阶段实现细节现位于[plans/](plans/README.md)，测试与结果记录位于[validation/](validation/README.md)；本文件继续只保存调查事实和总指令映射。

本文件中的“当前事实”来自静态核查、现有证据或明确标注的环境探针；具体运行故障仍需测试确认。后续实现要求和可能出错的场景集中在[实施计划的对应阶段](hopper-plan.md#stages)，术语解释和讨论问答见[技术问答](hopper-qa.md)。

## 调查范围与可追溯性

- 外层仓库：`b4bb6be6060ab4a594819563579bfc5617585d51`。
- 内层 GPGPU-Sim：`4195dd4f229cbf03f4f38c49139ae94d01d23f03`。
- 读取根 `AGENTS.md`、两份规划输入、workload README及四份范围/模块/性能文档，沿加载、执行、访存和调度路径做静态核查，并读取冻结PTX、启动器、JSON证据。
- 未构建、未运行模拟 workload。仅对本机真实driver encoder可用性做只读探针，以及按launcher协议读取九份forward.bin。静态发现不等于运行验证；已有API导出不等于已正确实现。
- 调查不修改模拟器；静态判断须由对应阶段的最小程序和真实 entry 验证。

<a id="current-gaps"></a>

## 当前代码与环境的核查事实

以下行号对应调查提交。表中标注“静态核查”的结论仅描述现有代码；本轮未运行模拟器验证其触发与结果。

| 主题 | 位置 | 当前查到的事实 | 后续处理位置 |
| --- | --- | --- | --- |
| 架构后缀 | [cuda_runtime_api.cc](../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) L3329；[ptx_loader.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_loader.cc) L363–369 | 静态核查：架构被转为整数，ptxas 命令用 `sm_%u` 构造目标 | [阶段 0](hopper-plan.md#stage-0) |
| 模块管理的现有结构 | [cuda_runtime_api.cc](../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) L688、3796–3861；[ptx_parser.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_parser.cc) L102–144 | 静态核查：有 default 名称/缓存路径，parser 复用全局符号表；尚无“只保留最后模块”的运行结论 | [阶段 0](hopper-plan.md#stage-0) 测试 |
| 64B 参数及声明 | [ptx_parser.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_parser.cc) L530–539；[TE PTX](../workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx) 参数声明 | 静态核查：align 语法已有，参数对齐检查只接受到 16；实际 PTX 包含 64B 对齐及 `.noreturn` 外部声明 | [阶段 0](hopper-plan.md#stage-0) |
| 直接 launch | [cuda_runtime_api.cc](../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) L2053–2081；[launcher.cu](../workloads/moe/source/launcher.cu) L362 | 静态核查：CUDA≥10 的直接入口不建立所依赖的配置栈；冻结程序直接调用该入口 | [阶段 0](hopper-plan.md#stage-0) |
| 动态 shared | [cuda_api_object.h](../gpu-simulator/gpgpu-sim/libcuda/cuda_api_object.h) L136–167；[cuda_runtime_api.cc](../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) L1017、4247、4426；[shader.cc](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) L4406–4509 | 静态核查：config 保存了动态字节，grid 创建/驻留链路未完整传递；FuncSetAttribute 忽略设置并返回成功 | [阶段 0](hopper-plan.md#stage-0) |
| generic 地址范围 | [abstract_hardware_model.h](../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) L796–814；[cuda-sim.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) L392–422 | 静态核查：仍用 96KiB/80SM 常量；当前配置为 132SM，TE 使用超过 96KiB 的 shared 偏移；shared 存储模板的 16KiB 是分页大小 | [阶段 3](hopper-plan.md#stage-3) |
| 主机接口 | [runtime-interface-audit.json](../workloads/moe/evidence/runtime-interface-audit.json)；[launcher.cu](../workloads/moe/source/launcher.cu) L290–318、349–364 | 接口核查记录：driver-entry 与 ExC 未导出；启动器先编码描述符，再注册 110 模块，最后启动内核 | [阶段 3](hopper-plan.md#stage-3) / [5](hopper-plan.md#stage-5) |
| 特殊寄存器 | [ptx.l](../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l) L250；[ptx_sim.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_sim.cc) L258–260 | 静态核查：`%lanemask_le` 只有词法入口，取值未实现 | [阶段 1](hopper-plan.md#stage-1) |
| shuffle 配置 | [cuda-sim.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) L748、766–779、1022；[H100 配置](../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config) | 静态核查：integer latency/initiation 解析和使用六项，配置只有五项 | [阶段 1](hopper-plan.md#stage-1) |
| 显式复制时间 | [stream_manager.cc](../gpu-simulator/gpgpu-sim/src/stream_manager.cc) L140–169；[gpu-sim.cc](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) L240、2531 | 静态核查：stream copy 直接完成，perf_sim_memcpy 用于填充 L2，没有完整 DMA 时序证据 | [阶段 0](hopper-plan.md#stage-0) 核查；性能口径见验证方案 |

## 可复用的现有源码

此表只定位现有代码。PTX 需要新增的功能和适配方案见右侧阶段。

| 能力组 | 已有功能代码或数据结构 | 已有时序代码 | 实施位置 |
| --- | --- | --- | --- |
| BF16、标量、warp 协作 | [instructions.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc)：cvt L3036、bfind L1719、bar L1514、shfl L5361、vote L6418 | 现有 SIMT、scoreboard、算术、shared/global 管线 | [阶段 1](hopper-plan.md#stage-1) |
| 普通 cp.async | 当前缺少该 PTX 功能执行入口 | [trace_driven.cc](../gpu-simulator/trace-driven/trace_driven.cc) L385 的 LDGSTS 入口；[shader.cc](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) L1073、1136、2092、2253 的组与请求返回处理 | [阶段 2](hopper-plan.md#stage-2) |
| cluster/DSMEM | [cuda-sim.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) L2130–2192 当前给线程本 CTA 的 shared | [abstract_hardware_model.h](../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) L282、1555 已有 cluster 字段，另有普通调度/存储模型 | [阶段 3](hopper-plan.md#stage-3) |
| mbarrier | [abstract_hardware_model.h](../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) L318–359 有 arrivals/tx/phase 数据结构 | [shader.cc](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) L3174、3723–3874 有完成状态更新 | [阶段 4](hopper-plan.md#stage-4) |
| TMA/tensor map | 当前缺少 PTX 描述符求值与真实数据搬运实现 | [abstract_hardware_model.cc](../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc) L287、550、767 有地址合并/完成信息；[shader.cc](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) L2955、3086 有 OOB/byte mask/返回处理及组播通路 | [阶段 5](hopper-plan.md#stage-5) |
| ldmatrix/stmatrix | 当前缺少所需 PTX 入口；旧 WMMA 有线程协作代码 | 现有 shared bank/端口/延迟模型 | [阶段 6](hopper-plan.md#stage-6) |
| WGMMA/寄存器 | [instructions.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) L1903、2045 的旧 WMMA 为 16×16/FP16 累加，所需形式为 BF16/FP32 m64n128k16 | [hopper_opcode.h](../gpu-simulator/ISA_Def/hopper_opcode.h) L218、[trace_driven.cc](../gpu-simulator/trace-driven/trace_driven.cc) L646–680、shader L1110/2178 有 GMMA 资源和完成入口 | [阶段 6](hopper-plan.md#stage-6) |

SASS 轨迹提供执行路径/活动线程、寄存器编号及部分操作数值、CTA/cluster 身份、TMA 全局地址、OOB 字节、mbarrier 和 multicast mask；这些不由当前 PTX 功能路径自动生成。[trace_driven.cc](../gpu-simulator/trace-driven/trace_driven.cc) L259、346；[NVBit tracer](../util/tracer_nvbit/tracer_tool/tracer_tool.cu) L1494、1550 使用运行时解析 API。仓库没有可直接用于无 GPU PTX 路径的独立 tensor map 解码器。

<a id="existing-waits"></a>

## 现有等待与资源代码的静态事实

此表记录当前代码行为；它不等于已运行验证这些行为在 PTX workload 中造成了具体故障。所需适配和错误场景在对应阶段。

| 代码事实 | 源码位置 | 实施位置 |
| --- | --- | --- |
| mbarrier phase 输入右移 31 位 | [shader.h](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h) L1984 | [阶段 4](hopper-plan.md#stage-4) |
| try_wait 在 issue 前读取 trace 操作数；重试条件包含恒真的逻辑或项，按无限重试处理 | [shader.cc](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) L1270、1315 | [阶段 4](hopper-plan.md#stage-4) |
| bulk/GMMA 初次 wait 判断与后续更新的比较方向不一致 | shader L1194、1210、5043 | [阶段 5](hopper-plan.md#stage-5) / [6](hopper-plan.md#stage-6) |
| GMMA 后续等待只检查本 warp 的全部 outstanding | shader L5053 | [阶段 6](hopper-plan.md#stage-6) |
| TMA outstanding 在 global store ack 时清除，没有独立的 shared 源读完事件 | shader L4975 | [阶段 5](hopper-plan.md#stage-5) |
| 等待队列由 last_depbar_group_type 选择 | shader 的 DEPBAR 分支 | [阶段 7](hopper-plan.md#stage-7) 联合验证 |
| GMMA 专用资源在外层 trace.config，PTX run.py 只复制内层配置 | [trace.config](../gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config) L24–30；[run.py](../workloads/moe/run.py) | [阶段 6](hopper-plan.md#stage-6) |

已确认的容量限制分布在：[ptx_ir.h](../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h) L588–676的8项tuple；[ptx.y](../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y) L607–611；[cuda-sim.cc](../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) L1175–1217的手写decode；[abstract_hardware_model.h](../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) L66–67、1274、1422–1424的常量及硬编码数组；[scoreboard.cc](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.cc) L82、127；[shader.h](../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h) L1343的collector及写回/统计遍历。完整容量链路与异步依赖处理见[阶段 6](hopper-plan.md#stage-6)。

<a id="resource-facts"></a>

## Workload 和资源事实

资源字段须分开解释：专家 GEMM 的 H100 记录为动态 shared 215040B、静态 shared 0B、驱动保留 1024B、总分配 216064B。驱动保留量不是 PTX 静态声明，资源验收见[阶段 3 验证](hopper-validation.md#stage-3)。每 SM shared 配置为 233472B，仅两个 CTA 的动态部分就已超容量，因此该内核需要跨 SM 放置。

九份forward.bin各有4个tensor map，均为rank2；当前dtype为UINT16/BF16，box为64×64、64×128、32×64，element stride为1，无interleave，swizzle为64B/128B，L2 promotion为128B，OOB为零填。初始dim≤128、global stride=0；真实地址、dim和stride由设备端replace更新。初始值与设备端更新序列的实现见[阶段 5](hopper-plan.md#stage-5)。

实际PTX还会写global_dim ord0..4、global_stride ord0..3，虽然rank仍为2；实际更新包含这些 ordinal。见[TE PTX](../workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx) L1061–1131及后续更新路径。L5872先发TMA、L5876再登记expect。这一实际顺序由[阶段 4](hopper-plan.md#stage-4)的计数设计和[阶段 5](hopper-plan.md#stage-5)的真实搬运测试覆盖。

本机CUDA12.8头文件仅公开opaque对象；真实libcuda探针在未初始化时编码返回`CUDA_ERROR_NOT_INITIALIZED`，`cuInit(0)`返回`CUDA_ERROR_NO_DEVICE`。这是当前主机的结果，不推广到所有环境。官方说明对象不透明、通过CUDA API/PTX访问，支持内部语义实现而非猜测原生编码的选择。[CUDA12.8.1 tensor map文档](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

实际九例专家启动均为grid66×2×1、block384×1×1、cluster1×2×1、动态shared215040B。H100记录为168 regs/thread、allocated shared216064B，寄存器和shared都限制一CTA/SM。**这是该内核的资源占用结论，不是Hopper的一般CTA数上限。** 当前寄存器总额度64512恰好对应两个232/thread消费者组和一个40/thread生产者组；具体配额等待要求见[阶段 6](hopper-plan.md#stage-6)。依据[hardware-counters.json](../workloads/moe/evidence/hardware-counters.json)的模块16 launch记录及[TE PTX](../workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx) L956、5113。

官方语义核查仅限影响本路线的条目：`setmaxnreg.inc`额度不足等待，bulk `.read`只等descriptor及源读取而不要求目标写回完成；WGMMA/cluster/mbarrier等其余形式以workload逐项说明及其中官方链接为验收依据。[PTX8.7相关条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)、[bulk wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)。

## 已确定的范围与建模边界

这里记录当前采用的范围和近似，并非已发现故障清单；运行中何时需要复查写在右列。多模块行为由阶段 0 测试；模拟墙钟和 RSS 从最早测试开始记录，在真实普通内核、gating 和代表性专家可运行时分别更新，见[验收记录规则](validation/README.md)。

| 项目 | 当前处理 | 何时重新处理 |
| --- | --- | --- |
| 额外物理register bank重分配 | 暂缓；已有bank/端口计时继续使用，完整寄存器结果和额度交接必须做 | 有证据表明它能解释主要GEMM时间差额时 |
| cache eviction/L2 hint的精细替换策略 | 保留类型、读取/缓存通路与正确数据；若现有策略未准确表达hint，明确记录近似 | 相关真实内核出现大的L2/DRAM/耗时差额且能归因时 |
| 完整GPC/任意cluster拓扑 | 当前二CTA跨SM、共同驻留、目标映射和必要通信时序；不扩任意规模/多卡 | 当前TE的主要差额由放置/通信解释，或workload范围变更时 |
| 显式DMA的详细引擎模型 | 保持正确数据/顺序/缓存影响，单列未计时间；20%初期口径不混入未建模DMA | 必要复制能解释显著时间差额时；不是把TMA内核搬运也当小问题 |
| 全部PTX8.7、H200、其它WGMMA形状/类型 | 不在九例范围，暂缓 | 明确新增消费者时 |
| 指令表写法与实际常量差异 | 实物优先：普通wait有6，bulk.read有1；mapa文中0..3应按实际二CTA的rank0/1 | 本计划已覆盖，后续可做文档小勘误；不改冻结PTX |
| 全九例自然运行完整设备时间 | 已接受现有ncu初期口径；不伪造缺失的DMA/间隙数据 | 需要更严格性能结论时补采 |

记录新小问题时只写“触发条件→结果/时间影响→暂缓原因→重开条件”。数值错误、错地址、漏搬运、提前解除等待、强制异步串行化和64项截断不属于可放过的小问题。通过当前阶段必要检查后停止无关扩测。

<a id="instruction-map"></a>

## 指令覆盖与阶段对应

下表由原指令文档的202个条目生成，只维护对应阶段，不复制其完整语义。34个新增、12个扩展、154个已有路径待回归、2个失败分支全部分配。已有路径在阶段1最小基础测试及其实际消费者阶段2/7回归，不能把表中“已有”读成已经验收。

<details>
<summary>需新增：34 种写法</summary>

| 完整写法 | 实施/回归阶段 |
| --- | --- |
| `barrier.cluster.arrive.relaxed.aligned` | 阶段3 cluster/DSMEM |
| `barrier.cluster.wait.aligned` | 阶段3 cluster/DSMEM |
| `cp.async.bulk.commit_group` | 阶段5 TMA/tensor map |
| `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` | 阶段5 TMA/tensor map |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` | 阶段5 TMA/tensor map |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint` | 阶段5 TMA/tensor map |
| `cp.async.bulk.wait_group.read` | 阶段5 TMA/tensor map |
| `cp.async.cg.shared.global` | 阶段2 普通异步拷贝 |
| `cp.async.commit_group` | 阶段2 普通异步拷贝 |
| `cp.async.wait_group` | 阶段2 普通异步拷贝 |
| `elect.sync` | 阶段1 warp协作 |
| `fence.mbarrier_init.release.cluster` | 阶段4 mbarrier |
| `fence.proxy.async.shared::cta` | 阶段5/6共用发布顺序 |
| `fence.proxy.tensormap::generic.acquire.gpu` | 阶段5 TMA/tensor map |
| `ldmatrix.sync.aligned.x4.m8n8.shared.b16` | 阶段6 矩阵/寄存器 |
| `mapa.shared::cluster.u32` | 阶段3 cluster/DSMEM |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | 阶段4 mbarrier |
| `mbarrier.arrive.shared::cluster.b64` | 阶段4 mbarrier |
| `mbarrier.arrive.shared::cta.b64` | 阶段4 mbarrier |
| `mbarrier.expect_tx.shared::cta.b64` | 阶段4 mbarrier |
| `mbarrier.init.shared::cta.b64` | 阶段4 mbarrier |
| `mbarrier.try_wait.parity.shared::cta.b64` | 阶段4 mbarrier |
| `redux.sync.add.s32` | 阶段1 warp协作 |
| `setmaxnreg.dec.sync.aligned.u32` | 阶段6 矩阵/寄存器 |
| `setmaxnreg.inc.sync.aligned.u32` | 阶段6 矩阵/寄存器 |
| `stmatrix.sync.aligned.x4.m8n8.shared.b16` | 阶段6 矩阵/寄存器 |
| `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned` | 阶段5 TMA/tensor map |
| `tensormap.replace.tile.global_address.shared::cta.b1024.b64` | 阶段5 TMA/tensor map |
| `tensormap.replace.tile.global_dim.shared::cta.b1024.b32` | 阶段5 TMA/tensor map |
| `tensormap.replace.tile.global_stride.shared::cta.b1024.b64` | 阶段5 TMA/tensor map |
| `wgmma.commit_group.sync.aligned` | 阶段6 矩阵/寄存器 |
| `wgmma.fence.sync.aligned` | 阶段6 矩阵/寄存器 |
| `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16` | 阶段6 矩阵/寄存器 |
| `wgmma.wait_group.sync.aligned` | 阶段6 矩阵/寄存器 |

</details>

<details>
<summary>需扩展：12 种写法</summary>

| 完整写法 | 实施/回归阶段 |
| --- | --- |
| `bar.warp.sync` | 阶段1（TE专用项最迟阶段6前） |
| `bfind.shiftamt.u32` | 阶段1（TE专用项最迟阶段6前） |
| `cvt.f32.bf16` | 阶段1（TE专用项最迟阶段6前） |
| `cvt.rn.bf16.f32` | 阶段1（TE专用项最迟阶段6前） |
| `cvt.rn.bf16x2.f32` | 阶段1（TE专用项最迟阶段6前） |
| `ld.global.L1::evict_first.b32` | 阶段1（TE专用项最迟阶段6前） |
| `ld.global.L1::evict_first.b8` | 阶段1（TE专用项最迟阶段6前） |
| `ld.global.L1::evict_last.b32` | 阶段1（TE专用项最迟阶段6前） |
| `shfl.sync.bfly.b32` | 阶段1（TE专用项最迟阶段6前） |
| `shfl.sync.idx.b32` | 阶段1（TE专用项最迟阶段6前） |
| `shfl.sync.up.b32` | 阶段1（TE专用项最迟阶段6前） |
| `vote.sync.ballot.b32` | 阶段1（TE专用项最迟阶段6前） |

</details>

<details>
<summary>已有路径，待回归：154 种写法</summary>

| 完整写法 | 实施/回归阶段 |
| --- | --- |
| `abs.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `add.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `add.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `add.s64` | 阶段1基础；阶段2/7随消费者回归 |
| `and.b16` | 阶段1基础；阶段2/7随消费者回归 |
| `and.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `and.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `and.pred` | 阶段1基础；阶段2/7随消费者回归 |
| `atom.shared.add.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `bar.sync` | 阶段1基础；阶段2/7随消费者回归 |
| `bfe.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `bfi.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `bra` | 阶段1基础；阶段2/7随消费者回归 |
| `bra.uni` | 阶段1基础；阶段2/7随消费者回归 |
| `brev.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `cvt.s64.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `cvt.u16.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `cvt.u32.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `cvt.u64.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `cvta.global.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `cvta.to.global.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `div.full.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `div.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `div.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `ex2.approx.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `fma.rn.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.b16` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.b8` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.nc.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.v2.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.v2.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.global.v4.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.s8` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.u8` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.v2.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.param.v2.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.u8` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.v2.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.v2.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.v4.b16` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.v4.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.v4.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `ld.shared.v4.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `mad.lo.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `max.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `min.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.pred` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.u16` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `mov.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `mul.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `mul.hi.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `mul.lo.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `mul.lo.s64` | 阶段1基础；阶段2/7随消费者回归 |
| `mul.wide.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `mul.wide.u16` | 阶段1基础；阶段2/7随消费者回归 |
| `mul.wide.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `neg.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `not.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `not.pred` | 阶段1基础；阶段2/7随消费者回归 |
| `or.b16` | 阶段1基础；阶段2/7随消费者回归 |
| `or.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `or.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `or.pred` | 阶段1基础；阶段2/7随消费者回归 |
| `popc.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `prmt.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `rem.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `rem.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `ret` | 阶段1基础；阶段2/7随消费者回归 |
| `selp.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `selp.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `selp.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `selp.u16` | 阶段1基础；阶段2/7随消费者回归 |
| `selp.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `selp.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.eq.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.eq.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.eq.s16` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.eq.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.eq.s64` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.ge.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.ge.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.gt.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.gt.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.gt.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.gt.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.le.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.le.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.le.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.le.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.leu.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.lt.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.lt.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.lt.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.nan.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.ne.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.ne.s16` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.ne.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.ne.s64` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.neu.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `setp.num.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `shl.b16` | 阶段1基础；阶段2/7随消费者回归 |
| `shl.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `shl.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `shr.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `shr.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `shr.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.b16` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.b8` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.v2.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.global.v4.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.param.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.param.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.u8` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.v2.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.v2.b64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.v2.u64` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.v4.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `st.shared.v4.u32` | 阶段1基础；阶段2/7随消费者回归 |
| `sub.f32` | 阶段1基础；阶段2/7随消费者回归 |
| `sub.s32` | 阶段1基础；阶段2/7随消费者回归 |
| `sub.s64` | 阶段1基础；阶段2/7随消费者回归 |
| `xor.b32` | 阶段1基础；阶段2/7随消费者回归 |
| `xor.pred` | 阶段1基础；阶段2/7随消费者回归 |

</details>

<details>
<summary>仅断言失败路径：2 种写法</summary>

| 完整写法 | 实施/回归阶段 |
| --- | --- |
| `call.uni` | 阶段0加载；阶段1前失败行为 |
| `trap` | 阶段0加载；阶段1前失败行为 |

</details>

