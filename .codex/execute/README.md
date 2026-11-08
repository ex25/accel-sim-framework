# Hopper PTX 实际执行与验收记录

本目录记录**实际做过什么、实际验证到哪里**；方案仍在 [plans](../plans/README.md)，测试要求仍在 [validation](../validation/README.md)。目录建立时阶段 0～7 均未实施、未运行模拟验收，以下空表不是完成证据。

给新 agent 发送 [执行 prompt](implementation-prompt.md) 或 [验证 prompt](validation-prompt.md)，另附“当前进度、这轮负责哪个阶段、已有提交或报告”。两份 prompt 对阶段 0～7 通用，不需要逐阶段改写正文。

| 阶段 | 实际实现记录 | 实际验收记录 |
| --- | --- | --- |
| 0：加载与启动 | [README](implementation/stage-0/README.md) | [README](validation/stage-0/README.md) |
| 1：基础数值与 warp 协作 | [README](implementation/stage-1/README.md) | [README](validation/stage-1/README.md) |
| 2：普通 cp.async | [README](implementation/stage-2/README.md) | [README](validation/stage-2/README.md) |
| 3：cluster 与 shared 身份 | [README](implementation/stage-3/README.md) | [README](validation/stage-3/README.md) |
| 4：mbarrier | [README](implementation/stage-4/README.md) | [README](validation/stage-4/README.md) |
| 5：tensor map 与 TMA | [README](implementation/stage-5/README.md) | [README](validation/stage-5/README.md) |
| 6：矩阵、异步 WGMMA 与配额 | [README](implementation/stage-6/README.md) | [README](validation/stage-6/README.md) |
| 7：真实专家与完整九例 | [README](implementation/stage-7/README.md) | [README](validation/stage-7/README.md) |

## 分工与状态

- 执行 agent 修改模拟器、补必要测试并做构建和自测，维护 implementation 下对应阶段 README。自测不能冒称独立验收。
- 验证 agent 阅读实际代码与测试，独立检查预期和运行证据，维护 validation 下对应阶段 README；可补充或修正验收测试。发现模拟器错误时报告复现、影响和修正位置，由执行 agent 修复，再由验证 agent 复验。这是用户于 2026-11-08 确认的分工。
- 实现状态用“未开始 / 进行中 / 部分完成 / 已完成”；验收状态用“未运行 / 进行中 / 未通过 / 部分通过 / 通过”。填完成或通过必须指明具体范围、代码版本和证据。仅完成早期可运行子集时，其余项仍保留未完成。
- 旧验收结论不因代码已修就自动改为通过。实现记录注明修复及影响范围，验证记录绑定复验版本和结果；无需因文档或无关改动重跑测试。

## 每阶段只维护两份实际记录

implementation 的 README 先说明现在能做什么，再写“对应计划哪一节 → 修改/新增哪个文件和函数 → 为什么改 → 行为如何变化 → 哪些指令/接口受影响 → 自测证据”。注明实际复用的 SASS/共用模型及必要近似，不把拟实现的步骤复制成完成报告。

validation 的 README 先给结论和阻塞，再写“被验版本 → 实际测试与独立预期 → 输出/同步/时序结果 → 通过范围 → 未覆盖与原因”。不要只填退出码。每个问题用具体触发条件、可见后果、证据和所属阶段说明；没有复现的静态疑点单列，不能冒充运行故障。

实际记录以这里为准，原计划/验收方案文末只放链接。实现需要调整方案时更新相关段落及理由，不把整个概览再抄一遍。跨阶段问题指定一个负责阶段，其它阶段引用，不新建重复的问题库。

## 证据放哪里、保存多少

测试源码复用阶段 0 入口和已有目录；需要新增时放 `workloads/moe/tests/hopper/stage-N/`，不为每阶段创建一个框架。运行日志、输出、大型中间数据留在 `workloads/moe/runs/hopper-stage-N/<test-or-run>/` 或原九例运行目录，README 链接具体报告和日志。目录建立本身不代表已有测试或结果；大型产物不搬进 .codex，也不随文档提交。

保存能复跑的命令、实际加载库/模式、配置及散列、工具链、外层与独立内层仓库版本；未提交代码注明基准和被测 diff/散列。测试代码/输入和预期来源也要能定位。沿既有记录保存设备周期、初始化/模拟/比较墙钟及 RSS，不为元数据另造管理系统。

正文用简短中文解释动作、原因和后果。源码定位、命令、日志和详细结果放同一 README 后部的证据区或链接的运行报告；有助理解时用简图，不逐行复述代码。

## 范围和停止

遵循 [AGENTS.md](../../AGENTS.md)、[原规划原则](../hopper-planning-prompt.md)及[已确认决定](../hopper-planning-decisions.md)。不为未使用变体、非法输入穷举、假想扩展添加防御代码或验证；通过本阶段必要功能、同步、主要时序和受影响回归后停止扩测。

运行阻塞、数值失败、错误同步/地址和明显主要时序失真需要修正；低影响问题记录触发条件、影响、暂缓理由和重开条件后继续。约 20% 只用于可比时间，数值沿原标准。用户指定本轮阶段；通过后说明解锁什么，不自行开始下一阶段。
