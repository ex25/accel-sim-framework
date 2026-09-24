---
name: document-change
description: 将 Accel-Sim research change 记录到正确的 source of truth，使 rationale、evidence 与 limitation 在 context loss 后仍可恢复。
---

## 路由

- Stable policy → `.codex/rules/`
- Repeated workflow → `.codex/skills/`；`.agents/skills/` 只放 symlink
- Compact current fact/state → `.codex/knowledge/`
- Architecture、call path、alternative、diagram → `.codex/docs/`
- Design choice → `.codex/docs/decisions.md`
- Chronology/commit → `.codex/docs/changelog.md`
- Experiment metadata/entrypoint → `.codex/experiments/`

## 语言

解释性正文默认使用中文。PTX、SASS、opcode、pipeline、BookSim、NVLink、
NVSwitch、NVLS、API、symbol、path、config、metric、evidence label 等专有名词
保留英文。代码、命令、路径与原始输出不得翻译。

## 步骤

1. 每类信息只选择一个 source of truth，不把相同段落复制到多个文件。
2. 记录 date、branch、relevant SHA、command、test、evidence label、assumption、
   limitation 与 next step。
3. code/model change 要同步更新 `.codex/docs/implementation.md` 对应章节，包含 problem、
   design、alternative、file/class/function、call path、configuration、before/
   after、test、validation、commit 与 future work。
4. 保持 `.codex` 高信号：不保存 raw trace、binary、build output 或长 log。
5. 运行 `git diff --check`，检查 path/link；milestone 或 blocker 改变时更新
   `project-state.md`。
