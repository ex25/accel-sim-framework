# 阶段 0：加载、参数与普通启动

状态：仅方案；本轮无实现、修复或运行验收。[对应验收](../validation/stage-0.md)。

## 本阶段解决什么

让小型 CUDA 程序能够把正确的 PTX、参数和 shared 资源交给周期模型。否则后续指令测试尚未开始计算，就可能在加载或启动处失败。本阶段交付独立最小程序和单模块测试入口，不声称冻结完整前向已经能跑。

## 当前事实及影响

| 已核实的静态事实 | 对当前目标的影响 | 本阶段处理 |
| --- | --- | --- |
| 架构以整数存储、ptxas 命令按 `sm_%u` 生成 | `sm_90a` 会丢失 a 后缀，后续含架构专用指令的模块无法按原目标处理 | 保留原目标字符串，数值字段只用于能力排序 |
| 参数对齐检查仅接受到 16B，真实 entry 有 64B 对齐结构体 | 无法接受实际参数声明；若仅绕过检查而布局未改，可能读错结构体字段 | 参数分配、绑定和拷贝使用同一对齐结果 |
| CUDA≥10 的直接 launch 依赖未由它建立的配置栈 | 冻结启动器直接调用 API 时，不能取得本次正确 grid/block/shared | 理顺直接入口和编译器入口的配置来源 |
| 动态 shared 没有完整进入 grid 创建/资源占用；函数 opt-in 被忽略 | 申请的空间和可驻留 CTA 数不可靠，后续专家内核尤其受影响 | 接通每次 launch 的实际字节数和函数许可上限 |
| 默认模块名与共享符号表同时存在 | 是否影响多模块加载尚待测试，不能据此断言前面的模块被覆盖 | 两模块交替调用后再决定是否修正 |

以上没有在当前模拟 workload 上运行复现；来源见[调查依据](../hopper-notes.md#current-gaps)。模块含义和例子见[QA](../hopper-qa.md#modules)。

## 指令与接口范围

| 对象 | 要做的事 | 当前状态 |
| --- | --- | --- |
| `.target sm_90a`、`.param .align 64`、`.noreturn` 外部声明 | 扩展加载、语法和参数布局 | 待扩展 |
| `cudaLaunchKernel`、编译器 `<<<>>>` 入口、`cudaFuncSetAttribute` | 修正启动参数传递和 opt-in 语义 | 待修正/验证 |
| fatbinary 注册、entry 绑定、global/constant 初始化 | 验证现有路径；仅在测试证明有问题时局部修正 | 待验证 |
| `ld.param`、`mov`、普通 `ld/st`、`bar.sync`、`ret` | 用已有指令完成参数与 shared 回写 | 现有路径待回归 |
| `call` 到 `__assertfail`、`trap` | 声明可加载；触发时明确失败，不假成功 | 现有/缺口由测试确认，最迟阶段 1 前完成 |

## 如何实现

1. **先建立可重复构建。** 按根 AGENTS.md 配置 debug 构建；保留工具链输出。最小测试必须确认加载内层仓库生成的 libcudart，并启用 PTX 周期模式。工具链实际失败才处理，不先升级整套依赖。
2. **保留目标与布局。** 沿 cuobjdump section→PTX 加载→ptxas 资源提取传递完整目标名；已有整数能力比较可以保留。沿 parser 的参数声明→`function_info` 参数配置→`cudaSetupArgumentInternal` 核对 64B 对齐。不要只提高断言上限，也不要按 C++ 结构体 `sizeof` 猜 PTX ABI。
3. **把 launch 配置传到同一条已有启动路径。** `cudaLaunchKernelInternal` 从显式实参建立本次配置并绑定参数；编译器入口消费它自身的配置。先检查编译器桩实际走哪条入口，再选择直接传配置或已有 push/pop 的最小修正，不能两边重复压栈。继续复用 stream 队列和 kernel 创建，不新增另一套执行器。
4. **区分许可与实际分配。** 函数属性保存允许的动态 shared 上限，launch 保存本次实际字节数；后者传入 kernel/grid 对象并进入 CTA shared 分配和驻留计算。PTX 静态声明、动态部分分别保存。H100 的 1024B 驱动保留量属于占用开销，不作为 PTX 静态地址区添加给程序；资源核对在阶段 3 完成。
5. **以两模块测试决定是否改模块系统。** 跟踪 fatbinary handle→host 函数标记→entry→资源信息。若唯一 entry 和各自全局标记均正确，保留共享表；若失败，定位具体名称/缓存键/资源归属后局部修正。不预先重写链接器，不把最后一次返回的表指针误解为只含最后一个模块。
6. **建立一个可复用单模块入口。** 独立测试程序只注册被测 fatbinary，并按该 entry 的实际参数布局传参。它用于原始 PTX 的测试，不修改冻结程序绕过 110 模块注册或 tensor map 编码。真实 entry 写死的维度仍要满足。

## 能复用哪些 SASS 性能模型

加载和参数 ABI 是 PTX/运行时工作，SASS trace 已给出内核与操作数，不能替代这一部分。可复用普通 grid/CTA 调度、shared 容量限制、stream 队列和周期统计。显式 host copy 继续保留数据、顺序和 L2 影响；现有 `perf_sim_memcpy` 不是完整 DMA 耗时模型，不把它作为已实现 DMA 的证据。

## 暂缓与交付

| 暂缓项 | 原因与重开条件 |
| --- | --- |
| 通用跨模块重名符号链接、所有 CUDA 版本 ABI | 当前消费者是冻结模块和本机工具链；只有实际模块测试暴露相关需求才做 |
| 通用 CUDA Graph、多流并发和 CPU 提交耗时 | 九例不以此为验收范围；不能影响当前 stream 顺序 |
| 完整 DMA 引擎模型 | 初期比较 kernel 时间和；显式复制正确性必须保留，时间单列 |

交付后可以执行两模块交替启动及单模块测试；开始记录初始化/模拟墙钟和 RSS。进入阶段 1，不等完整 Hopper 语法齐备。

## 源码入口

- [cuda_runtime_api.cc](../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc)：`cudaLaunchKernelInternal`、`cuobjdumpParseBinary`、fatbinary 注册与函数属性。
- [ptx_loader.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_loader.cc)、[ptx_parser.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_parser.cc)、[ptx.y](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y)：目标、对齐、外部声明。
- [cuda_api_object.h](../../gpu-simulator/gpgpu-sim/libcuda/cuda_api_object.h)、[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)：launch config、CTA 资源限制。

## 实际交付记录

待执行 agent 填写：已实现/修复的接口与指令、提交、对应验收报告、尚未完成内容。当前：无。
