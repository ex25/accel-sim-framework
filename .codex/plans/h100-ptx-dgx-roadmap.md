# H100 PTX → Dense → DGX H100/NVLS → DeepEP/MoE 路线

确认日期：2026-10-13。本文固化用户在设计讨论及最终“确认”中接受的范围。
**计划已确认，阶段 10–19 均未开始；本文不是实现或性能结果。**

2026-10-13 后续审计修订：阶段 0–2 保留窄范围验收，3–9 已撤下整体验收并补续做
计划。当前 [从阶段 3 开始逐阶段补齐，最终到单卡 Dense 阶段 14](single-card-dense.md)，
15–19 延后。本页为长期路线，当前阶段边界与验收顺序以该单卡计划为准。当前事实见
[实现与结果总览](../docs/implementation/README.md)，历史计划见
[Hopper PTX 阶段 0–9](hopper-ptx.md)。租机操作要求见
[H100/DGX 短租测试手册](../docs/h100-dgx-rental-runbook.md)。

## 1. 最终交付与当前缺口

目标是建立可作为研究 baseline 的 **PTX execution-driven H100**，复用并核查
现有 SASS 路径的共享时序后端，最终扩展为支持 NVLS 的 DGX H100。
“支持 H100 SASS”不能直接推出“PTX 路径已经准确建模 H100”。

| 交付 | 当前基础 | 仍需完成的关键内容 |
| --- | --- | --- |
| 单卡完整 Dense block | 受限 Hopper 指令、普通 attention 数值测试 | 真实生成 PTX 的 norm/QKV/RoPE/causal attention/projection/FFN/residual，完整输出及资源时序 |
| 原版 FA3 | 冻结源码/内核前端解析及部分指令语义 | 实际动态执行路径、descriptor/layout、同步与资源、完整输出、详细时序与 H100 对照 |
| Python/PyTorch/Triton 入口 | 阶段 9 文本 PTX module/launch harness | 先编译与执行适配器，后所需 CUDA Driver shim；真实 Python 程序和依赖图验收 |
| H100 性能 baseline | A800 兼容实现与模型趋势证据 | 同一 H100 实现/launch 的校准、独立留出验证、运行成本测量 |
| DGX/NVLS | 多 rank 与网络模型基础，端点功能性服务 | 依赖集成、真实 payload、BookSim 路由器归约资源及背压、硬件算法与拓扑验证 |
| DeepEP/MoE | 参数化 MoE 诊断场景 | V2.5 真实 dispatch/combine 路径、BF16、专家计算、完整 block 与 DGX 对照 |

“原生运行”在本路线中特指：真实 Python 程序生成/选择真实 kernel，经明确适配后
在模拟器执行 PTX。不是承诺任意未修改 PyTorch wheel、cuBLAS/cubin 或全部 CUDA API
的二进制兼容。适配层必须公开，不能把 CPU reference 运算当作模拟执行。

## 2. 已确认的工作负载与决策

### 2.1 单卡先做 Dense inference prefill forward

顺序固定为：RMSNorm → QKV → RoPE → causal MHA → output projection → residual
→ RMSNorm → SwiGLU FFN → residual。第一版不包括训练、反向、decode、GQA、varlen。

| 档位 | B | S | hidden | query heads | head dimension | FFN intermediate |
| --- | --- | --- | --- | --- | --- | --- |
| small | 1 | 128、512 | 1024 | 8 | 128 | 2816 |
| large | 1 | 2048 | 4096 | 32 | 128 | 11008 |

输入/权重 FP16，关键累加 FP32。RMSNorm epsilon、RoPE 参数、权重与随机种子、
bias、各处 cast/归约顺序、算子级容差在阶段 10 明确冻结，不交给框架默认值漂移。
Dense 第一版不扩成 BF16；后续 DeepEP 的 BF16 是单独的工作负载及 ISA 验收。

先完成语义等价的完整 attention/Dense，随后单独完成冻结原版 FA3。
现有冻结 FA3 为 noncausal，必须保留其回归，并为目标 causal FA3 建立独立 manifest。
等价 attention 通过不代表原版 FA3 通过。

算子实现固定为可保存 PTX 的 Triton/CUTLASS 等明确来源；保存源代码、编译器、
编译选项、PTX、资源与 launch 参数。硬件和模拟器比较同一实现；不能用模拟器简化
GEMM 与硬件默认 cuBLAS 的时间差作为模拟误差。

### 2.2 性能验收原则

- 在测量前冻结 calibration/held-out 列表。每个关键 held-out 微基准误差均须
  ≤20%，使用 `abs(T_sim - median(T_hw)) / median(T_hw)`；报告波动与重复次数。
  不用平均误差掩盖失败，不因看到硬件结果而删测试。
- 调参只使用 calibration 集；若依据 held-out 结果调参，旧 held-out 已失去独立性，
  必须明确披露并另建未参与调参的验证集，保留原失败结果。
- 完整 Dense 的逐算子及端到端时间独立报告；微基准达标不自动代表 Dense 达标。
  端到端和后续网络的数值性能门限尚未确认，应在对应硬件采集前登记，不能套用
  微基准的 20% 或事后宣布通过。未设门限时只可提交测量报告，不能声称准确度验收。
- 完整详细仿真目标约一天内完成。分别记录仿真 wall time、模拟 GPU 时间、峰值
  内存；缩小 shape、抽样或解析模式只能用于开发回归，不能替代完整 block 验收。
- 单卡 H100 与 DGX 分别租用；优先匹配最终 DGX 的 H100 SXM。若测到的是 PCIe
  或不同配置，独立命名数据集，不能跨 SKU 直接套参数。

### 2.3 多卡与 NVLS 边界

先完成 AllReduce、ReduceScatter、AllGather，再接 DeepEP dispatch/combine，最后
完整 MoE。初版 NVLS 模型限定 **8 GPU、SUM、FP16/BF16 输入、FP32 累加器、固定
chunk、单个 collective**；仍须支持 collective 内多个 chunk 与竞争/背压测试。

FP32 累加器是选定的模型设计，不是已证实的 NVSwitch 内部行为。必须对照实际软件
路径测试舍入/归约顺序；与硬件不一致的配置单列为假设模型，不能作为硬件已验证
baseline。chunk 大小、归约吞吐、buffer/VC 数量与流水线延迟进入参数表，标明
实测/公开来源/假设及敏感性范围，验证前冻结。

NVLS 的最终模型必须位于 **BookSim 路由器数据路径**：入口接纳、归约槽位、buffer、
VC/credit、背压、转发和完成通知。端点一次性附加归约延迟只能是诊断基础。
AllGather 是数据搬运，不凭名称赋予 switch reduction；TMA 的 CTA multicast
也不是跨 GPU NVLS。NCCL 设置 NVLS 开关不证明该次 collective 实际选中了 NVLS，
必须保存算法/协议选择证据。参见 [NCCL NVLS 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)。

### 2.4 DeepEP 版本与精度

选定 DeepEP V2.5、单节点 NVLink high-throughput/prefill 路径，暂不覆盖 RDMA/
跨节点及旧 V1/NVSHMEM 路径。阶段 10 固定不可变 commit，不能把 `main` 当版本锁。

2026-10-13 核对的上游说明使用统一 EPBuffer，包含运行时 JIT；combine 输入要求
BF16，且专家端应用路由权重后交给 combine 求和。因此需要 BF16 数据链；若用显式
转换衔接 FP16 专家，转换与数值影响必须在硬件/模拟器两侧一致计入，禁止隐式省略。
当前上游工具链要求也明显高于现有 CUDA 12.8 环境，应按冻结 revision 单独准备。
这些要求依据 [DeepEP 上游说明](https://github.com/deepseek-ai/DeepEP/blob/main/README.md)，
后续报告须改为引用冻结 SHA 对应内容。

不能把 EPBuffer combine 自动等同于 NVLS switch reduction。先追踪实际 kernel/
通信算法，再决定哪些事件进入网络、哪些计算留在 GPU。既有 8 GPU/16 experts/
top-2/每 GPU 128 tokens/hidden 4096 场景可作为诊断起点；其 capacity=1.25/drop
及参数化 GEMM 不是已确认的原生 MoE 配置，阶段 18 须冻结实际路由与专家规格。

## 3. 仓库职责、环境与集成原则

| 所属 | 负责内容 |
| --- | --- |
| accel-sim-framework + 内层 gpgpu-sim | PTX/ISA、单 GPU 功能执行、Tensor Core/TMA、cache/HBM/shared、资源与调度、Dense/专家 kernel、设备端通信语义 |
| multigpu-sim | rank worker、物理时间协调、跨 GPU 地址/事件映射、DGX 拓扑、BookSim NVSwitch、通信算法和完整多卡工作负载集成 |

跨仓库事件 ABI 至少包含版本、rank/peer 或 rank 集合、地址空间与范围、bytes、dtype、
stream、序号、collective/chunk 标识、token/epoch、提交和完成时间及状态。规定时间
单位、clock-domain 转换、pause/resume、失败传播、一次且仅一次完成以及 payload
所有权。设备可见完成必须晚于 payload 可见，不能只传一个 host AllToAll 延迟。

本次文档整理时的代码锚点（均不包含未提交改动）：

| 仓库 | HEAD |
| --- | --- |
| accel-sim-framework | `f9faa73a4f7907db88db450e54000a502ecf39d2` |
| 内层 gpgpu-sim | `f4fe783ac7fc4f1cdaa9efa72a3575537d4fba36` |
| multigpu-sim | `147a44e12727d8aa9b4a59af399d67da9059b82f` |

multigpu-sim 当前固定的依赖为 accel-sim `d930ad6d02c09bb56867132583735aba0389cff4`、
gpgpu-sim `91880c53383d5a6a6742bfb1be2c5f34e39c7871`、BookSim
`28f43299f1706a3160ffac721ca461d74eb6e618`，含已 materialize 的改动。
不能以“同一项目”假定已包含本分支能力；阶段 16 必须显式处理 pin、patch/overlay
与版本冲突，保留当前用户修改，不直接 reset 或替换子模块。

主要开发环境是 **Conda `in-switch`**，工作区是 `/home/wangjz/in-switch`；二者不是
同一概念。执行脚本记录实际 Python/compiler/CUDA 路径和版本；新工具链优先用单独
可复现环境，不能静默升级主环境。

**遇到 sudo、驱动、系统包、性能计数器或其他权限阻塞：AI 先诊断，再编写可审阅的
最小脚本，由用户/管理员执行。** 脚本应提供 preflight/dry-run、明确系统改动、幂等
检查、日志和失败退出；AI 不自行 sudo、不自动更换驱动。记录阻塞证据，并继续不依赖
权限的工作。阶段必需检查未完成时保持阻塞，不能靠忽略权限错误标记通过。

## 4. 阶段依赖与执行计划

以下编号延续本研究的 0–9；不重命名 multigpu-sim 自身 M 系列里程碑。
当前先按 3 → 4 → 5 → 6 → 7 → 8 → 9 补齐组件，再按 10 → 11 → 12 → 13 → 14
完成单卡应用；验收边界和移交项以单卡计划为准，不能让 3–9 依赖 10–15 的整应用
结果。下表是后续阶段，不能用来跳过阶段 3。15–19 当前不实施。

| 阶段 | 交付 | 状态 | 核心退出条件 |
| --- | --- | --- | --- |
| 10 | 可执行规格与版本/验收冻结 | 未开始 | 工作负载、版本、证据格式、验证集和工具链差异全部显式登记 |
| 11 | 完整 Dense 功能执行 | 未开始 | 三个固定 shape 的所有中间值和最终输出通过 |
| 12 | Dense 详细时序 | 未开始 | 完整详细执行、资源守恒、成本报告，尚未声称 H100 准确 |
| 13 | Python 适配器与 Driver shim | 未开始 | 同一真实 Python 程序进入 PTX 模拟执行，无隐式 CPU 计算替代 |
| 14 | 单卡 H100 校准与验证 | 未开始 | 独立微基准逐项 ≤20%，完整 Dense 对照与运行预算验收 |
| 15 | 冻结原版及 causal FA3 | 未开始 | 独立功能/时序/硬件证据；等价实现不能代替 |
| 16 | 多卡依赖集成与基础 collective | 未开始 | 2/8 rank 真实 payload、事件协议及 ring/tree 基础对照 |
| 17 | BookSim 路由器 NVLS | 未开始 | 8 rank 归约、buffer/credit/背压与完成顺序通过模型验收 |
| 18 | DeepEP V2.5 真实路径 | 未开始 | 冻结 API、BF16、实际 dispatch/combine 数值和依赖通过 |
| 19 | 完整 MoE 与 DGX 硬件验证 | 未开始 | block、collective、DeepEP 结果与 DGX 对照及 baseline 发布条件通过 |

### 阶段 10：冻结规格和证据协议

1. 盘点真实代码动态路径，建立 opcode/modifier/API/layout 支持矩阵：已有且验证、
   已有未覆盖、未实现。以实际生成 PTX 为准，不只检索 opcode 名称。
2. 固定 Dense 数学规格、reference、每算子容差、输入/权重/seed、Triton/CUTLASS/
   FA3/PyTorch/DeepEP SHA 或包 hash、编译器和 launch；列出兼容版本组合。
3. 为 CUDA/PTX 新版本、JIT、BF16、tensor-map 编码、stream/CPU 同步做依赖检查。
   记录现有环境与目标工具链差异；只写出无 sudo 安装方案及管理员脚本需求。
4. 冻结微基准 calibration/held-out：算术/转换、Tensor Core shape、shared/bank、
   cache/HBM、TMA/mbarrier、多 CTA 与资源占用；覆盖吞吐、依赖链延迟、重叠和瓶颈。
   固定计时区间、重复次数、warmup、时钟策略及异常值规则。
5. 建立版本化 event ABI 草案、硬件参数登记表、结果 manifest/schema、失败分类和
   短租优先级。未确定参数明确标 pending 及关闭它的阶段，不填写虚构常量。

交付：冻结 manifest、支持矩阵、校准/验证用例列表、reference 生成与检查命令、
工具链可行性记录。退出：所有验收输入可重建，未解决依赖有明确责任阶段和阻塞条件。
此阶段不要求声称当前模拟器能运行完整 workload。

### 阶段 11：真实 kernel 的完整 Dense 功能路径

1. 在 `research/hopper-ptx/` 新增后续阶段 harness 与 PTX/metadata 导出；覆盖
   RMSNorm、投影 GEMM、RoPE、causal attention、SwiGLU、residual。
2. 沿真实动态执行路径补内层 parser/执行语义/地址与 layout；补所需 descriptor、
   共享内存布局、同步和转换。未知行为保留带位置的严格错误。
3. 从单算子到完整 block，保存各层中间 tensor，与冻结 reference 比较；测试
   causal mask、边界 tile、数值极值和不同 seed，避免仅用零/常量输入。
4. 验证三个固定 shape 无跳算、无 host reference 代算、无 hidden fallback。

退出：功能输出按预注册容差通过；有每算子 max/mean error、失败位置及全 block
结果。负例应验证未支持路径确实拒绝，但负例数量不能替代成功运行证据。
性能此时只记录，不以功能通过推导详细时序正确。

### 阶段 12：完整 Dense 的详细时序与资源

1. 将 PTX 动态指令接入共享 H100 时序后端，检查 issue/scoreboard、Tensor Core
   执行与依赖、TMA 请求/事务完成、mbarrier、cache/shared/HBM 和资源占用。
2. 按生成 kernel 的实际需要补 setmaxnreg、warpgroup/CTA 并发等行为；不把 parser
   接受当作资源模型。需要 cluster/DSMEM 时补对应详细时序，否则明确未覆盖。
3. 加入守恒与顺序检查：访存字节、请求完成、barrier transaction、warpgroup
   commit/wait、寄存器/shared 限制，测试竞争和异步重叠。
4. 先跑小回归，再跑三个完整 Dense shape；记录逐算子 cycles、GPU 时间、
   wall time、RSS、瓶颈和统计文件。优化宿主执行成本须保留语义/模型等价性。

退出：完整详细执行成功，无未解释丢失事务/提前完成；提供时间与资源解释及预计
运行成本。若超一天预算，先分析宿主热点、事件调度和输出开销，不缩小正式验收 shape。
H100 误差仍为未验证。

### 阶段 13：真实 Python 入口，先适配器后 shim

1. 先实现 PyTorch/Triton 编译与执行适配器，保留真实 JIT 产物、kernel 参数、
   张量布局和依赖图；Python 只承担宿主逻辑/reference，不替模拟器执行目标算子。
2. 据实际调用清单逐项扩展 Driver shim：context、device attributes、分配/释放、
   copy、文本 PTX module/function、参数、必要 stream/event、错误与同步。
   未需 cubin/库接口保持明确不支持，不能返回假成功。
3. 同一 Python workload 支持硬件与模拟后端配置；比对 kernel/launch 序列和结果。
   验证缓存命中/未命中、重新编译、对象生命周期、多次运行及异步依赖。
4. 保存 API 支持矩阵和运行方法，区分“适配器通过”“所需 shim 通过”和未覆盖 API。

退出：真实 Python 程序完成全部固定 Dense shape；输出、调用/launch 序列与阶段
11/12 一致，功能及详细模式均有证据，异常和 unsupported 路径可诊断。

### 阶段 14：单卡 H100 测量、校准、留出验证

1. 按短租手册准备硬件 bundle 并本地演练；核对 H100 SKU、驱动、时钟/功耗/MIG、
   工具链和测量权限。A800 仅验证通用包装/传输，不能证明 Hopper kernel 能执行。
2. 采集冻结 micros、全部 Dense shapes、输出及逐算子/整体 CUDA-event 时间；
   CPU/JIT/传输耗时单列，不和 GPU kernel 时间混为一谈。
3. 用 calibration 集更新可解释参数，记录参数来源与版本；冻结参数后一次评估
   held-out，并保留逐项误差与波动。无法解释的模型偏差回到对应实现修复。
4. 在主环境复跑完整详细仿真，报告一天预算与算子/整体性能门限是否满足。

退出：关键独立微基准各 ≤20%，数值通过；Dense 性能有独立预注册门限和结果，
完整 run 达到运行预算。无硬件、工具限制或超标时保持未验收/阻塞，不以 A800 替代。

### 阶段 15：原版 FA3 单独闭环

1. 从已冻结 noncausal FA3 建立动态指令、tensor-map、swizzle、同步、寄存器/共享
   资源的缺口清单；先解决实际触发的 elect/prefetch/front-end-only 等严格失败。
2. 先功能再详细执行原冻结内核，检查全输出及 pipeline 生命周期；增加多 tile、
   多 CTA/warpgroup、边界与压力例，保留旧组件回归。
3. 为 Dense 的 causal 目标冻结独立原版 FA3 变体，重新导出与验证，不能仅修改
   mask 标签。把该变体接回完整 block，与等价 attention 路线分别记账。
4. 对同一原版 kernel 做 H100 测量。若阶段 14 时尚不可运行，另备短租包/窗口，
   不复用等价 kernel 的硬件数据为原版背书。

退出：noncausal 冻结目标、causal 目标各有解析/功能/详细/硬件状态，完整 causal
block 通过；额外 dtype、shape、backward 等未测范围继续列为未覆盖。

### 阶段 16：集成单卡研究分支和多卡基础路径

1. 在保留 dirty 依赖的前提下冻结迁移清单，整理 pin/patch/overlay；记录三个依赖
   与两个研究仓库的版本，先通过 multigpu-sim 原有回归，再验证 H100 PTX 能力。
2. 落实事件 ABI、统一物理时间、地址映射、pause/resume 和 exactly-once 完成；
   用 2 rank 再 8 rank 测试乱序、重试/重复、错误传播和不同 stream 依赖。
3. 实现真实 payload 的 AllReduce/ReduceScatter/AllGather 基础算法和对照路径。
   归约与通信实际工作须归属 GPU 或网络，禁止仍由未计时 host reference 完成。
4. 建立 DGX H100 配置及 tensor-parallel Dense 诊断，明确 sharding 和 collective
   位置；保留 qv100/参数化场景作为旧诊断，不能改名充当 H100 验证。

退出：2/8 rank 正确性、完成顺序、字节与时间守恒、单卡与原多卡回归通过；真实
H100 PTX 分支在 worker 中运行，支持矩阵与限制可查。此时多卡硬件准确度未验证。

### 阶段 17：路由器 NVLS 模型

1. 在 BookSim 路由器数据路径加入 collective/chunk 识别、参与者状态、归约槽、
   有限 ingress/egress buffer、归约吞吐/延迟及转发/广播规则。
2. 归约完成与输入到齐、资源竞争、credit 和下游背压绑定；实现 SUM 的 dtype/
   舍入规则，FP32 累加模型显式命名，与端点功能版区分。
3. 接入 8 rank 固定 chunk 单 collective；检查慢 rank、热点、有限槽位、tail
   chunk、长短消息、credit 枯竭和恢复，验证无提前通知、丢包、重复贡献或死锁。
4. 记录所有模型常量、拓扑/链路聚合假设、时钟单位；扫描 chunk、buffer、吞吐、
   带宽与延迟敏感性，建立 ring/tree 与 NVLS 的等工作量比较。

退出：输出与协议/资源不变量通过；有可解释流量/队列/归约利用率、缩放趋势和
模型级性能结果。硬件归约精度和实际 NVLS 路径的确认留在阶段 19，不能提前标 H。

### 阶段 18：DeepEP V2.5 dispatch/combine

1. 在冻结版本上追踪 EPBuffer、JIT、CPU 同步、stream/event、内存注册和设备端
   通信路径；固定单节点 NVLink 模式并检测实际没有走 RDMA 依赖。
2. 导出真实 kernel PTX 与 launch，补所需新 PTX/BF16/通信语义。跨卡 load/store、
   原子或同步映射到事件/payload，不整体替换成 host 估计延迟。
3. 保留 token→expert 路由、alignment/padding、源 rank 元数据、权重位置、实际
   token count、combine 顺序及 CPU 可见完成；测试空 expert、偏斜及热点。
4. 冻结 MoE 规格：专家数/top-k/容量或不丢弃策略、每 rank tokens、专家 FFN
   维度、dtype、路由 seed/分布与计时范围；明确与旧 golden 场景的差异。
5. 先真实 dispatch→reference expert→combine 做隔离诊断，再使用模拟专家
   kernel 闭环。reference expert 诊断单独标记，不能用于最终性能结果。

退出：真实 dispatch/combine 数值、BF16 链路、元数据和依赖通过，实际路径清楚；
adapter、JIT cache/toolchain、算法选择和 unsupported 范围均有记录。

### 阶段 19：完整 MoE、DGX 硬件验证与 baseline

1. 组合 norm→attention→MoE（dispatch→专家 FFN→combine）及 residual 的完整
   forward，保留 Dense 对照。按阶段 18 冻结的路由与 dtype 做全输出检查。
2. 准备 DGX 独立短租包：8 GPU 拓扑/链路与时钟证据、基础 collective/NVLS
   选择探针、DeepEP、完整工作负载；数据太大时优先数值、时间和必要日志。
3. 分开采集网络 calibration/held-out，验证链路与算法、switch 归约舍入/顺序、
   背压/缩放趋势；逐项对照预注册门限。无法验证的物理常量保持假设标记。
4. 分别报告 ring/tree、NVLS、DeepEP 实际算法和完整 block。NVLS 未实际参与
   DeepEP 的路径不能写成“DeepEP 的 NVLS 加速”；任何实验改造单独作为变体。
5. 输出完整可复现包、参数来源、误差/失败表及有效范围，冻结可比较 baseline。

退出：单卡、多卡、DeepEP、MoE 四层结果可追溯，正确性与预注册性能门限通过。
如果只有端点功能/假设时序或软件路径不一致，交付对应研究模型，不声称已准确复现
真实 DGX H100。新优化必须使用同一工作负载、算法边界、测量范围及校准版本比较。

## 5. 每阶段报告与验收材料

实现开始时在 `docs/implementation/` 下建立对应编号报告，首页固定写：状态、
本阶段验收范围、外层/内层配对 commit、环境、实际结果摘要、证据入口、未完成项。
多卡阶段追加 multigpu-sim/BookSim 及依赖 patch 的版本。阶段提交遵循现有仓库规则；
本文档确认本身不代表授权自动提交或实施全部阶段。

结果至少包括：

- `manifest.json`：workload、实现/编译版本、输入与 PTX hash、配置、launch、
  时间单位、计时边界、机器型号、运行 ID、命令与退出状态。
- `results.json` 与逐 case CSV：reference/模拟/硬件输出误差，cycles/时间、
  重复统计、wall time/RSS、阈值、通过/失败/跳过原因。
- 原始 stdout/stderr、参数文件、必要统计/trace、输出及校验和。失败也归档；
  大文件可存外部持久目录，但报告必须给可访问路径及 hash，不能只指向 `/tmp`。
- 证据等级独立列：S=静态/解析，F=功能，M=详细模型，H=硬件对照。
  S/F/M/H 不是互相推导；只有范围内所有门槛通过才标“已验收”。

所有命令在报告中区分“计划执行”与“已执行”，预期值与实测值分栏。新脚本名在
创建、检查并实际运行前不得作为现成工具宣称可用。总览只链接最终验收入口，历史
失败保留在阶段报告中，防止再次把计划、旧失败和最终结果混在一起。

## 6. 风险和停止条件

| 风险 | 提前检测与处理 | 不可接受的替代 |
| --- | --- | --- |
| 新工具链产出超出 PTX 前端能力 | 阶段 10 保存真实 PTX 并盘点，阶段 11/18 补实际路径 | 降低版本后冒称已支持冻结目标 |
| 完整 FA3 同步/资源复杂 | 单独阶段 15，按动态路径逐项补语义/时序 | 等价 attention 通过就标 FA3 完成 |
| 仿真超一天 | 阶段 12 提前测量，分析宿主热点并回归 | 用缩小 shape 充当正式结果 |
| H100/DGX 时间短或权限不足 | 预检、自动恢复/打包、优先级和用户执行权限脚本 | 租机期间临时大规模修环境或丢失原始结果 |
| NVSwitch 内部参数不可观测 | 参数来源/假设、敏感性、匹配实际算法 | 把自选 FP32 累加/拓扑参数写成硬件事实 |
| 多仓依赖分叉 | 显式迁移 pins/patches，单卡与多卡双回归 | 覆盖 dirty 子模块 |
| 结果不达标 | 保留失败，定位功能/模型/测量原因并复验 | 更换验证集、隐藏 fallback 或只报平均数 |

下一步入口是 [阶段 3 实现报告](../docs/implementation/03-frontend-foundation.md)；
本次仅更新规划与验收约定，没有开始上述实现或租机。
