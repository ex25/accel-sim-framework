# H100/H200 PTX 支持研究与实施计划

> 2026-10-13 状态修订：阶段 3–9 已重开，不能按旧测试 PASS 推断已完成。
> 当前从阶段 3 开始逐阶段完成，执行 [单卡计划](single-card-dense.md) 的 3→14 顺序。
> 阶段 3–9 先补组件，10–14 再完成完整 Dense；下一实现会话先完成阶段 3。

## 1. 研究问题

当前 Accel-Sim README 将 Hopper 的 TMA、WGMMA、mbarrier、cluster 和 HBM 配置描述
为 SASS trace-driven 支持。研究目标不是复制 `H100-SASS`/`H200-SASS` 配置，而是
让文本 PTX execution-driven 路径生成同等的执行语义和动态元数据，再复用已有的
共享时序模型。

当前已知事实：基线 PTX lexer 只有旧 WMMA；SASS trace 前端直接提供 TMA 地址、字节
数、mbarrier、cluster mask 和 GMMA commit 信息；共享 `warp_inst_t`、shader 和
LDST 已有一部分 Hopper 时序状态；现代 CUDA module/launch API 仍有 stub。因此
工作必须按功能、同步、时序、应用入口分层推进。

## 2. Git 和文档基线

- 外层 `.gitignore` 已单独提交：`907c0c8 chore: ignore Serena project metadata`。
- 外层研究分支：`research/hopper-ptx`，从更新后的本地 `dev` 创建。
- 内层研究分支：`research/hopper-ptx`，从内层 `dev=91880c53` 创建。
- 既有 `experiments/` 不属于本计划的基线变更。
- 内层 `feat/ptx-sync-events-p1-p5` 和 `feat/cais-add-ptx` 不自动合并。

## 3. 阶段门控

| 阶段 | 主题 | 必须产生的结果 |
|---|---|---|
| 0 | 基线与分支 | 两仓库 SHA、工具环境、文档入口可复现 |
| 1 | workload 冻结 | FA3 revision、PTX、shape、指令清单固定 |
| 2 | 文本 PTX harness | 最小 PTX kernel 可解析、执行、写回 |
| 3 | 前端基础 | opcode、类型、操作数和 `sm_90a` 处理明确 |
| 4 | mbarrier/同步 | phase、arrival、transaction、wait 状态正确 |
| 5 | TMA | descriptor、地址展开、OOB、shared 写和 multicast 正确 |
| 6 | WGMMA | 128-thread 语义、累加器、commit/wait 和时序连接正确 |
| 7 | cluster/DSMEM | cluster rank、远程 shared、fence 和资源行为明确 |
| 8 | A800/模型验证 | reference、SASS oracle、A800 趋势对照完成 |
| 9 | runtime integration | 作为独立阶段评估 PyTorch/Triton module/launch API |

阶段 0-8 是文本 PTX 研究主线；阶段 9 不得反向降低前面阶段的严格错误策略。

## 4. 通用验收协议

每个阶段必须提交对应报告，包含：目标、前置条件、源码依据、操作命令、代码逻辑、
预期结果、实际结果、测试输出摘要、失败日志、限制和下一阶段入口。报告状态只有在
所有验收项完成后才能改为 `已验收`。

功能验收使用 CPU/reference 和确定性断言；同步验收检查 phase/count/token 和阻塞
顺序；时序验收检查事件、统计项、瓶颈和趋势；硬件准确度没有 H100/H200 数据时不
承诺。失败 gate 按原因保持 `进行中` 或 `阻塞`，不得推进依赖它的下一 gate；
未使用功能的延期须按当前单卡计划显式登记，不能伪造其完成状态。

## 5. A800 对照原则

固定相同 attention 数学问题、输入、shape 和精度；Hopper 使用 FA3 或对应 Hopper
kernel，A800 使用架构兼容实现（优先 FA2 或普通 CUDA kernel）。分别记录功能结果、
kernel 时间、访存规模和瓶颈。报告实现差异，不能用固定 A800/H100 比例进行线性校准。

## 6. 已确认的后续路线（2026-10-13）

阶段 0–2 保留基线/workload/harness 的窄范围验收；3–9 撤下整体完成，补齐计划写在
各报告首页。当前执行 [从阶段 3 开始的单卡计划](single-card-dense.md)，逐阶段
验收后再推进。阶段 10–14 均为 `未开始`，不得跳过重开的 3–9。
[长期 10–19 路线](h100-ptx-dgx-roadmap.md) 保留原版 FA3 及多卡后续目标，暂不推进。
短租准备与权限处理见 [H100/DGX 手册](../docs/h100-dgx-rental-runbook.md)。
