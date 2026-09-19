# Research documentation index

## Purpose

This directory is the human-readable record of why the simulator is changed,
what the current checkout actually does, and which claims require hardware
validation. `.codex/knowledge/` is the compact reload path; this index is the
full reading path.

## Recommended reading order

1. `项目进展总览.md` — 中文历史总览、当前证据边界和下一步路线。
2. `14-final-goal-gap-plan.md` — 最终目标差距、环境上限和分阶段退出条件。
3. `15-concepts-for-readers.md` — 面向人的中文概念、trace 调用路径和术语解释。
4. `00-project-overview.md` — scope, terminology, and staged strategy.
5. `01-accelsim-architecture.md` — current simulator layers and call paths.
6. `02-trace-pipeline.md` — NVBit → trace → frontend → timing model.
7. `03-hardware-capability-matrix.md` — A800/H100/NVSwitch validation boundary.
8. `04-single-vs-multi-gpu-trace.md` — why compute traces do not encode a
   collective execution by themselves.
9. `05-h100-architecture-support.md` — source audit of SM90 support.
10. `06-nvlink-nvswitch-nvls.md` — hardware/model concepts (to be expanded with
   primary sources).
11. `07-current-support-gap-analysis.md` — capability matrix and evidence.
12. `08-booksim-architecture.md` — BookSim components and embedding options.
13. `09-accelsim-booksim-integration.md` — clock/data-boundary alternatives.
14. `10-design-options.md` — compute/communication architecture comparison.
15. `11-implementation-roadmap.md` — milestone exit criteria.
16. `12-experiment-plan.md` — synthetic and future hardware experiments.
17. `13-validation-plan.md` — evidence ladder and H100/NVSwitch plan.

## Current navigation state

- Milestone: M2 A800 trace pipeline, two-endpoint transport, trace autopsy, and
  endpoint/clock reference contract.
- Branch/base: `feat/nvbit-a800-booksim-nvlink` / upstream `d930ad6`.
- Baseline: release build and real A800 vectoradd trace/replay pass; CTest still
  has no registered tests.
- Blockers: no physical second GPU/NVLink peer and no H100/NVSwitch system.
- BookSim: separate `28f4329` checkout; standalone two-node AnyNet smoke/sweep
  passes, but no Accel-Sim adapter or clock coupling exists.
- Next: tracked per-cycle BookSim facade and one timestamped logical GPU event.

## Source-of-truth map

| Information | Primary location |
|---|---|
| Persistent rules | `.codex/rules/` |
| Current state/checklist | `.codex/knowledge/project-state.md` |
| Local environment | `.codex/knowledge/local-environment.md` |
| Trace facts | `.codex/knowledge/trace-format.md` |
| Full architecture/research | `.codex/docs/` |
| Design decisions | `.codex/docs/DECISIONS.md` |
| Phase history | `.codex/docs/CHANGELOG.md` |
| Experiment metadata | `.codex/experiments/` |

## Skill ownership

Skills are edited only under `.codex/skills/`. Every entry under
`.agents/skills/` must be a symlink to the matching canonical directory; do
not maintain two independent copies. The skill definitions include their
inputs, commands, artifacts, validation, and completion criteria, while the
rules define the stable policy they must follow.
