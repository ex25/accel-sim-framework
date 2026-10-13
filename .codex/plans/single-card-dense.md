# 当前执行计划：补齐单卡完整 Dense block

更新：2026-10-13。此前执行顺序整理只改文档；同日的新实现会话从阶段 3 开始，
实际结果以 [阶段 3 报告](../docs/implementation/03-frontend-foundation.md) 为准。
本文件是下一次 Codex 实现的唯一当前执行顺序；替代旧文档中“阶段 0–9 全部完成，
直接继续后面”的推断。长期目标保留在 [总路线](h100-ptx-dgx-roadmap.md)。

## 当前做到哪里

已有文本 PTX harness、部分 Hopper 功能/详细模型、noncausal 普通 attention
数值测试、受限 Driver module/launch。**尚无完整 Dense block 执行结果**。
原 FA3 只有 parser readiness；普通 attention 的 detailed 测试仅 1 row、1 head，
不是完整 attention grid。`in-switch` 当前未安装 torch/triton，阶段 9 的框架清单
是静态分析。证据、源码与状态见 [实现总览](../docs/implementation/README.md)。

阶段 0–2 保留原窄范围验收；同日审计曾将 **3–9 撤下整阶段完成标记，恢复“进行中”**。
已有测试 PASS 保留，不篡改原始结果。重新打开的原因分两种：缺失原有成功执行证据
（如阶段 3 的 ldmatrix 数值），以及已有组件子集尚未覆盖当前完整应用要求。
不是断言旧测试全部错误，也不要求重做已经验证的能力。

随后本次阶段 3 实现按其当前组件合同通过全部 gate：42 个 ldmatrix 数值 case
及前端、参数/label、普通 kernel、WGMMA 回归，证据见阶段 3 首页。**阶段 3 已验收，
4–9 仍进行中**；本会话停止，下一实现会话从 4 开始。MMA arithmetic、ldmatrix
timing 及完整应用仍按承接阶段处理，不从组件通过推导完整 Dense。

## 当前目标和停止点

单 GPU，FP16 输入/权重、关键累加 FP32，inference prefill forward、causal MHA：

`RMSNorm → QKV → RoPE → causal attention → O projection → residual → RMSNorm
→ SwiGLU FFN → residual`

| case | B | S | hidden | heads | head dimension | FFN intermediate |
| --- | --- | --- | --- | --- | --- | --- |
| dense-small-128 | 1 | 128 | 1024 | 8 | 128 | 2816 |
| dense-small-512 | 1 | 512 | 1024 | 8 | 128 | 2816 |
| dense-large-2048 | 1 | 2048 | 4096 | 32 | 128 | 11008 |

逐步交付三种明确结论：

1. **完整 Dense 功能可运行**：阶段 11；全部 shape、全部输出及关键中间值通过。
2. **完整 Dense 详细模型及真实 Python 入口可运行**：阶段 12/13；不是单 CTA
   抽样，不是 host reference 代算，功能与详细两种模式都能完成。
3. **单卡 H100 性能 baseline 可验收**：阶段 14；同实现硬件对照、关键独立微基准
   每项误差 ≤20%、完整 Dense 性能单独评估，最大正式 case 详细仿真约一天内完成。

近期实现到阶段 14 为止。原版 FA3 完整执行（阶段 15）、DGX、NVLS、DeepEP、MoE
均延后；当前 attention 可以采用已确认的语义等价真实 Triton/CUTLASS 实现。
不把“等价 attention 完成”改写为“原版 FA3 完成”。无 H100 数据时可完成前两项，
第三项保持阻塞并准备租机包，不能误标整条单卡路线验收。

## 执行顺序与阶段边界（按用户最新纠正）

从 **阶段 3** 开始，严格按 **3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14**
逐阶段完成。阶段 3 已集中完成并报告入口，**下一实现会话集中阶段 4**；不得
直接从阶段 10 开始，也不能把 3–9 当成可以跳过的缺口登记册。

此前“先做 10–14、其中顺便修补 3–9”的安排已撤销。为避免前序验收依赖后序结果
的循环，组件与整应用交付分开列出；所有移交项有承接阶段，不删除待办。本次修改
只调整计划，没有把任何阶段重新标为通过。

| 阶段 | 本阶段必须完成的交付 | 后续集成验收（仍未完成） |
| --- | --- | --- |
| [3](../docs/implementation/03-frontend-foundation.md) | 前端、sm_90a/tuple/type、受支持 ldmatrix 逐 lane 数值及负例 | Dense 新指令在 10 盘点、11 补齐；矩阵详细资源在 6/12 |
| [4](../docs/implementation/04-mbarrier-and-sync.md) | 多 barrier/phase/transaction 的功能和组件级详细同步 | cluster-addressed 行为归 7；完整 Dense pipeline 归 12 |
| [5](../docs/implementation/05-tma.md) | 冻结 TMA descriptor/layout、异步/并发、multicast 功能及详细验证 | 公开 descriptor API 归 9；Dense 新形式归 11/12；原版 FA3 特有形式归 15 |
| [6](../docs/implementation/06-wgmma.md) | 冻结矩阵形式/layout、数值、多组/CTA/SM 并发与资源成本 | 全量 Dense GEMM 归 11/12；H100 准确度归 14 |
| [7](../docs/implementation/07-cluster-and-dsmem.md) | setmaxnreg/资源、cluster/DSMEM detailed、组件级 launch 元数据 | 公开 Driver launch API 归 9；完整应用归 12/13 |
| [8](../docs/implementation/08-a800-and-model-validation.md) | 普通 attention 全 grid 功能/详细、A800 证据、严格聚合器 | causal Dense 归 11/12；H100 校准归 14；原版 FA3 执行归 15 |
| [9](../docs/implementation/09-runtime-integration.md) | 真实 Python 算子/短链 adapter、所需 Driver API、资源/stream/lifecycle | 三个完整 Dense shape 的 Python 集成归 13 |
| [10](../docs/implementation/10-dense-contract.md) | 基于 3–9 正式冻结完整 Dense 规格、版本、用例和能力缺口 | 不代表新 kernel 已可执行 |
| [11](../docs/implementation/11-dense-functional.md) | 补齐真实 Dense 新需求，三个完整 block 功能执行 | 不代表 timing 或硬件准确 |
| [12](../docs/implementation/12-dense-timing.md) | 全 grid 详细执行、资源/并发与运行预算 | 不代表 Python 入口或 H100 准确 |
| [13](../docs/implementation/13-dense-python-runtime.md) | 复用 9，三个完整 block 的 Python 功能/详细集成 | 新增 API 缺口须实际修复 |
| [14](../docs/implementation/14-dense-h100-validation.md) | H100 同实现校准/独立验证与单卡收口 | 15–19 延后 |

每阶段先冻结自身支持清单、PTX/输入/reference 和退出条件。需要真实生成 PTX 时，
在当前阶段做最小编译/导出准备并记录版本；不要求先完成阶段 10，也不将局部准备
冒称完整 Dense 规格冻结完成。现有明确支持的形式必须补实证，不能删掉来绕过失败。

阶段 10 汇总 `required-capabilities`，复用 3–9 成果。新发现的真实 Dense 指令/
layout/API 缺口在 11–13 实际修补、回归并更新组件报告，不能沿用旧组件验收自动
标为支持。每项移交记录来源、承接阶段、当前实现状态及未来成功用例。

## 后续 Codex 的操作规则

1. 先读本文件、总览和当前阶段报告，记录两仓库 SHA/dirty 状态。当前首项为
   **阶段 4，进行中**。阶段 3 已按新版合同通过，本会话不自动开始阶段 4 或阶段 10。
2. 仅实现当前阶段及必要的最小前置准备，避免重复实现/重复验收。旧功能
   的正例继续回归，原“未实现必须拒绝”的负例只有在功能真正补齐后才能改为正例；
   非法语法/非法状态负例继续保留。
3. 组件测试可用手写 PTX，但不能据此宣称真实编译器或完整应用通过。应用阶段
   使用冻结的真实 PTX 和全部正式 shape，不能把矩阵计算偷偷换成标量路径。
4. 报告填写实际命令、版本、完整结果、失败与证据后才进入待验收；验收脚本必须
   检查必需 case 集合，missing/skipped/timeout 均不能使当前 gate 通过。
5. 主要环境为 Conda `in-switch`。依赖缺失先给出隔离、可重建方案，不静默升级。
   sudo/驱动/计数器权限阻塞时，AI 写最小可审阅脚本（含 preflight/dry-run、日志、
   失败退出及恢复说明），用户/管理员运行；继续不依赖权限的工作。
6. 保留 `experiments/`、内层 `.serena/` 和已有修改，不重写旧实现提交来改变历史
   验收结论。新实现按阶段记录配对版本和完整报告。此前只提交文档的安排是历史
   审计范围；本次实现会话按当前阶段修改源码、测试和报告，旧实验构建产物不提交，
   不推送远端。

## 全局验收合同

- 数值：阶段 10 预先冻结算子/中间值/最终输出的 atol/rtol 与参考运算 cast。
  逐元素检查并报告最大/均方误差及首个失败；拒绝非预期 NaN/Inf、错 shape、漏元素。
  旧 attention 的 `0.002 + 0.002*abs(ref)` 只用于旧用例，不能直接充当全 block 门限。
- 执行：保存 PTX/launch/API manifest 和全量执行计数，证明所有层和全部 grid
  实际执行。关键 tensor 来自 simulator 内存；reference 仅用于比较。
- 模型：每条所需动态指令有可解释的 functional/timing/resource 归属；内存字节、
  异步事务、barrier phase、commit/wait、资源占用守恒，无空操作隐藏未实现语义。
- 性能：硬件与模拟器同 kernel/launch/dtype/计时边界；校准与 held-out 分离，
  H100关键微基准逐项 ≤20%。全 block 门限在测量前冻结，未设门限不得宣称准确度通过。
- 证据：持久目录保存输入/输出或可访问位置、manifest、逐 case JSON/CSV、日志、
  hashes、配对代码 SHA、dirty patch/hash 与实际加载的 runtime hash。只存 `/tmp`
  或只存一个 PASS 摘要不足以验收。文档计划与实际结果分栏。

本轮审计记录见 [JSON](../docs/implementation/evidence/single-card-audit-20261013.json)：
56 个保存文件的 hash 一致；仅验证证据完整性，不代表执行覆盖充分。本轮未重跑
模拟器或硬件。原始 PASS 不变，当前验收结论已按上述范围重新判断。
