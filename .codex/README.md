# Hopper PTX 研究资料

这套资料服务于 Accel-Sim 与 GPGPU-Sim 的配对研究分支：

- 外层仓库分支：`research/hopper-ptx`
- 内层仓库分支：`research/hopper-ptx`
- 外层起点：`dev`，`.gitignore` 提交后为 `907c0c8`
- 内层起点：`dev`，`91880c53`

## 阅读顺序

1. [当前实现与结果总览](docs/implementation/README.md)：先看实际能力、数值结果、证据链接和未完成范围。
2. [研究规则](rules/hopper-ptx.md)：修改与验收要求。
3. [当前单卡 Dense 执行计划](plans/single-card-dense.md)：从阶段 3 开始逐阶段补齐；阶段 3 的当前组件合同已通过，下一实现会话从阶段 4 开始，不跳到 10。
4. [H100/DGX 短租测试手册](docs/h100-dgx-rental-runbook.md)：租机前准备、权限处理、现场预算和结果回收。
5. [长期路线：H100 PTX → DGX/NVLS](plans/h100-ptx-dgx-roadmap.md) 与 [原总计划](plans/hopper-ptx.md)：保留背景；当前执行顺序以上述单卡计划为准。
6. 按总览链接读取对应阶段报告的最终结果；旧失败记录和计划属于历史。
7. [基线源码证据](docs/evidence.md) 描述内层 `91880c53` 的初始缺口，不代表当前实现。

2026-10-13 重新审计：阶段 **0–2** 保留原窄范围验收，阶段 **3–9** 已撤下
整体验收、恢复 `进行中`，各报告首页列缺口与重新验收条件。当前只有组件和普通
attention 结果，尚无完整 Dense block、原版 FA3 执行或真实 Python 应用验收。

同日的新实现会话已完成 [阶段 3 的新版组件验收](docs/implementation/03-frontend-foundation.md)：
42 个逐 lane ldmatrix 数值 case 零误差及全部前端/相关回归通过。阶段 4–9 仍为
进行中，ldmatrix detailed timing 和 mma.sync 算术仍未实现。当前会话到此停止。

阶段 10–14 已建立具体实施文档，实现均为 `未开始`；15–19 延后。主环境是 Conda `in-switch`；
遇到 sudo/系统权限阻塞，AI 准备最小可审阅脚本，由用户/管理员执行，并保存日志。

## 状态约定

阶段状态只能是：`未开始`、`进行中`、`待验收`、`已验收`、`阻塞`。
状态为 `已验收` 前，必须写出实际命令、结果摘要、证据路径和代码 commit。

验收状态只覆盖报告中实际测试的范围。解析、功能、时序、硬件验证必须分开记录；
未完成的结果继续明确写为“未实现”或“未验证”。
