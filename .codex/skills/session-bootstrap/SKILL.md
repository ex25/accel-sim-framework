---
name: session-bootstrap
description: Restore the Accel-Sim/NVLS research project after context loss, report its evidence-backed state, and select the next safe task.
---

## When to use

Run at the start of every new Codex session or after a context compaction.

## Steps

1. Read AGENTS.md, .codex/README.md, every file in .codex/rules/, and
   .codex/knowledge/project-state.md before editing. Do not depend on chat
   memory.
2. Read only the task-relevant knowledge, docs, and skill. Follow links from
   the source of truth instead of loading every historical log.
3. Capture main and nested repository status, branch, HEAD, and remotes with
   git status --short --branch; git branch --show-current; git log -1
   --oneline --decorate; and git -C gpu-simulator/gpgpu-sim status.
4. Check for user modifications. Preserve them; never reset, clean, or stash.
5. Report before editing: goal, branch, baseline SHA, hardware, milestone,
   completed work, blockers, evidence level, and next step.
6. If the task concerns a proposed ISA or microarchitecture, read
   `.codex/skills/ptx-design-baseline/` and
   `.codex/docs/17-ptx-primary-strategy.md` before selecting a SASS-oriented
   workflow.
7. When state changes, update project-state.md, relevant docs, and
   CHANGELOG.md.

## Artifacts and completion

Produce a concise session report and, when needed, updated state/changelog.
A fresh session must identify repositories, baseline, hardware limits,
milestone, blockers, and next action without the previous conversation.
