---
name: implement-feature
description: Implement one scoped Accel-Sim/BookSim research feature with design, focused tests, evidence, documentation, and an explanatory Git commit.
---

## Gate

Use only after reconnaissance identifies a bounded gap and the phase exit record
exists. Do not start full NVSwitch/NVLS implementation with this workflow.

## Workflow

1. Write problem, motivation, current behaviour, alternatives, chosen design,
   clock/ownership assumptions, and before/after call path.
2. Define the smallest edit and focused test, including baseline and regression
   commands, before editing.
3. Change only named files/classes/functions. Preserve unrelated PTX, SASS
   trace, and memory paths. When the feature is explicitly a proposed PTX ISA
   or microarchitecture change, modify the PTX path deliberately and document
   the SASS validation boundary.
4. Run focused tests, then the relevant regression. Classify failures as
   BASELINE FAILURE, environment/dependency failure, or branch regression.
5. Update implementation note, project-state, DECISIONS, CHANGELOG, and compact
   knowledge as appropriate.
6. Inspect git diff --check, git diff --stat, full diff, and all repository
   statuses. Commit one logical unit.
7. Re-read the final diff as a reviewer and record the resulting SHA.

## Required report

What changed; why; architecture; files and important classes/functions; call
path; configuration; before; after; tests and results; limitations; validation
status; commit SHA; and next step. The commit body must briefly say what
functionality changed, which files/symbols changed, how it changed, tests, and
limits.
