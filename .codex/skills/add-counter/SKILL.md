---
name: add-counter
description: 端到端新增 simulator statistic/counter，包括 declaration、initialization、increment、registration、test 与 documentation。
---

## 使用时机与输入

用于新的 GPGPU-Sim/Accel-Sim statistic、hardware-like counter 或 debug metric。
先确定 owner 是 shader core、cluster、memory partition、network、kernel 还是
global object，并记录 counter name、unit/type、owner、increment event、reset/
lifetime、aggregation、output label、baseline 与 test workload。

## Workflow

1. 在同一 subsystem 搜索现有 counter，跟踪 declaration → constructor/reset →
   increment/read → registration/reporting；不要建立平行 statistics path。
2. 确认 integer/float、per-object/aggregate、per-kernel/epoch reset、sum/max/
   average，并定义一次 increment 的含义。
3. 只修改 owning header/source、stats registration 与 reporting file，保持命名
   和 output convention。
4. 添加 deterministic focused test/replay 检查 label/value；有意义时增加 zero-event case。
5. 运行 focused test 与最小 regression，比较 baseline 并分类 failure。
6. 在 `docs/implementation.md` 与 `docs/changelog.md` 记录 unit、scope、reset、increment path、
   file/symbol、command/result 与 limitation。
7. 执行 `git diff --check`，使用具体 subject，如 `feat(stats): count ...`，body
   写 what/why/files/how/tests/limits。

完成条件：counter 安全初始化、每个目标 event 恰好 increment 一次、unit/scope
明确、focused evidence 充分，并已为后续 session 记录。
