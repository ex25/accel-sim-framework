---
name: hopper-ptx-research
description: Execute the staged Hopper H100/H200 PTX research workflow in the paired Accel-Sim and GPGPU-Sim branches.
---

# Hopper PTX Research Workflow

Use this skill only for work on the paired `research/hopper-ptx` branches.

## Before work

Read `.codex/rules/hopper-ptx.md`, `.codex/plans/hopper-ptx.md`, the current
stage report, and the inner repository's `.codex/README.md`. Confirm both Git
repositories and record their HEAD commits.

## Stage protocol

1. State the current stage and its acceptance criteria.
2. Inspect the relevant source symbols before editing.
3. Make the smallest change that satisfies the stage.
4. Run the stage-specific commands and tests.
5. Update the stage report with actual files, logic, output, limitations, and evidence.
6. Mark `已验收` only when every criterion passes; otherwise mark `阻塞`.
7. Do not begin the next stage while the current report is not `已验收`.

## Technical method

Prefer the existing PTX parser, `ptx_instruction`, `warp_inst_t`, shader timing,
LDST, cluster, and mbarrier abstractions. The intended data flow is:

```text
PTX text -> lexer/parser -> PTX instruction/functional operands
          -> warp_inst_t Hopper metadata
          -> shared shader/LDST/mbarrier timing model
```

For every new instruction, separately test parser acceptance, functional result,
synchronization state, memory effects, timing statistics, and failure behavior.
