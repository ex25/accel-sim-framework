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

- Milestone: M0/M1, repository and baseline audit.
- Branch/base: `research/nvls-accelsim-study` / `d930ad6`.
- Baseline: release build succeeds; CTest has no registered tests; prior local
  end-to-end V100 trace outputs exist but are not a newly rerun regression.
- Blockers: GPU is busy; NVBit SDK/tracer artifact is absent; no H100/NVSwitch.
- BookSim: separate `28f4329` checkout, analysis only.
- Next: source-grounded capability and trace autopsy, then minimal synthetic
  transport design.

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
