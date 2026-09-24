---
name: session-bootstrap
description: 在 context loss 后恢复 Accel-Sim/NVLS research project，报告有 evidence 支撑的状态，并选择下一项安全任务。
---

## 使用时机

每个新 session 开始时或 context compaction 后使用。

## 步骤

1. 编辑前读取 `AGENTS.md`、`.codex/README.md`、全部 `.codex/rules/` 与
   `.codex/knowledge/project-state.md`，不依赖 chat memory。
2. 只读取 task-relevant knowledge、docs 与 skill；沿 source-of-truth link
   查找，不加载所有历史 log。
3. 记录 main/nested repository 的 status、branch、HEAD 与 remote。
4. 检查并保留用户修改；不得 reset、clean 或 stash。
5. 编辑前报告 goal、branch、baseline SHA、hardware、milestone、completed work、
   blocker、evidence level 与 next step。
6. proposed ISA/microarchitecture task 必须先读 `ptx-design-baseline` 与
   `docs/roadmap.md`、`docs/implementation.md`。
7. state 改变时同步更新 `project-state.md`、相关主题 docs 与
   `docs/changelog.md`。

完成时应让新 session 无需之前对话即可识别 repository、baseline、hardware
limit、milestone、blocker 与 next action。
