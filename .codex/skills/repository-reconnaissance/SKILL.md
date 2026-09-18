---
name: repository-reconnaissance
description: Audit Accel-Sim, nested GPGPU-Sim, BookSim, Git history, dependencies, and local changes before design or implementation.
---

Capture `git status --short --branch`, branch, remotes, HEAD, recent history,
submodule status, nested repository status, and relevant ignored/build paths.
Read `.claude/` and `AGENTS.md` without copying them. Inspect build entrypoints,
tracer/frontend boundaries, and dependency SHAs. Classify each conclusion as
verified, observed, inferred, unknown, or needing hardware validation. Produce
a compact report in `.codex/knowledge/` or `.codex/docs/` and do not modify
simulator core code during reconnaissance.
