# Hopper PTX 规划文档导航

本目录保存固定九个单卡 MoE 用例的调查和实施规划。本轮未实现模拟器功能；阶段通过条件见验证方案。

| 想了解什么 | 去哪个文档 | 文档职责 |
| --- | --- | --- |
| 看总体路线和依赖 | [实施概览](hopper-plan.md) | 当前缺口、能力依赖、阶段交付 |
| 让执行 agent 实现某一阶段 | [分阶段计划](plans/README.md) | 指令、实现步骤、SASS复用、问题影响和暂缓项 |
| 看九例性能口径和基线 | [验证概览](hopper-validation.md) | 阶段检查概览、性能参照、停止条件 |
| 执行测试和填写实际结果 | [分阶段验收](validation/README.md) | 输入、预期、功能/时序检查、运行证据和墙钟预算 |
| 哪些取舍已经确定 | [已确认的决定](hopper-planning-decisions.md) | 用户确认的目标、精度和实现边界 |
| 当前代码查到了什么、依据在哪里 | [调查依据与指令映射](hopper-notes.md) | 当前代码事实、源码位置、可复用部分、范围边界和 202 条指令阶段映射 |
| 查阅讨论中的技术解释 | [技术问答](hopper-qa.md) | 保存问题与回答，按主题链接到实施阶段 |
| 最初要求规划工作交付什么 | [原始规划要求](hopper-planning-prompt.md) | 任务输入记录，供追溯原始要求 |

首次阅读建议先看实施计划；遇到术语看技术问答，准备测试时看验证方案。需要核查现有缺口时查调查依据，需要确认路线时查已确认的决定。

workload 的原始材料：

| 材料 | 用途 |
| --- | --- |
| [workload README](../workloads/moe/README.md) | 冻结用例、运行入口和数值比较 |
| [执行要求](../workloads/moe/docs/execution-requirements.md) | 指令以外需要支持的运行能力 |
| [指令清单](../workloads/moe/docs/ptx-instructions.md) | 202 种 PTX 写法的参数与语义 |
| [模块用途](../workloads/moe/docs/ptx-modules.md) | 110 个模块的用途和调用顺序 |
| [性能说明](../workloads/moe/docs/performance.md) | H100 基线和采集条件 |

路线取舍以已确认的决定为准，当前实施顺序以实施计划为准。原始规划要求保留为任务输入，不作为另一份实施计划维护。
