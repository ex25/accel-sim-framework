# Research documentation index

## Purpose

This directory is the human-readable record of why the simulator is changed,
what the current checkout actually does, and which claims require hardware
validation. `.codex/knowledge/` is the compact reload path; this index is the
full reading path.

## Recommended reading order

1. `00-project-overview.md` — scope, terminology, and staged strategy.
2. `01-accelsim-architecture.md` — current simulator layers and call paths.
3. `02-trace-pipeline.md` — NVBit → trace → frontend → timing model.
4. `03-hardware-capability-matrix.md` — A800/H100/NVSwitch validation boundary.
5. `04-single-vs-multi-gpu-trace.md` — why compute traces do not encode a
   collective execution by themselves.
6. `05-h100-architecture-support.md` — source audit of SM90 support.
7. `06-nvlink-nvswitch-nvls.md` — hardware/model concepts (to be expanded with
   primary sources).
8. `07-current-support-gap-analysis.md` — capability matrix and evidence.
9. `08-booksim-architecture.md` — BookSim components and embedding options.
10. `09-accelsim-booksim-integration.md` — clock/data-boundary alternatives.
11. `10-design-options.md` — compute/communication architecture comparison.
12. `11-implementation-roadmap.md` — milestone exit criteria.
13. `12-experiment-plan.md` — synthetic and future hardware experiments.
14. `13-validation-plan.md` — evidence ladder and H100/NVSwitch plan.

## Current navigation state

- Milestone: M2 A800 trace pipeline plus standalone two-endpoint transport.
- Branch/base: `feat/nvbit-a800-booksim-nvlink` / `85cccb1`.
- Baseline: release build and real A800 vectoradd trace/replay pass; CTest still
  has no registered tests.
- Blockers: no physical second GPU/NVLink peer and no H100/NVSwitch system.
- BookSim: separate `28f4329` checkout; standalone two-node AnyNet smoke/sweep
  passes, but no Accel-Sim adapter or clock coupling exists.
- Next: real-instruction trace autopsy, endpoint adapter, clock conversion, and
  analytical byte/flit checks.

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
