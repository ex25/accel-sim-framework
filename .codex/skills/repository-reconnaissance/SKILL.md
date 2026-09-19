---
name: repository-reconnaissance
description: Audit Accel-Sim, nested GPGPU-Sim, BookSim, Git history, dependencies, and local changes before design or implementation.
---

## When to use

Use before a new research phase, feature design, dependency decision, or when
repository state is uncertain. Do not modify simulator core code.

## Steps

1. For every repository capture status, branch, remotes, HEAD, recent history,
   submodule status, diff stat, ignored build paths, and untracked files.
2. Locate and read AGENTS.md, .claude, .codex, .agents, build files,
   gpu-simulator, util/tracer_nvbit, trace parser/frontend, and BookSim.
3. Verify nested GPGPU-Sim and external BookSim SHA/status independently.
   Record whether each is a submodule, nested repository, symlink, or checkout.
4. Build a source map for launcher, NVBit producer, post-processor, parser,
   trace shader core, shader/memory path, interconnect, configs, tests, and
   statistics.
5. Inspect representative original commits with git show --stat --format=fuller
   and use their concrete scope as the commit-message reference.
6. Label conclusions Verified, Observed, Inferred, Hypothesis, Unknown, or
   Needs hardware validation.
7. Write compact state plus only reusable detailed facts to .codex.

## Artifacts and validation

Record paths, commands, SHAs, blockers, and a safe next step in the relevant
knowledge/doc. Run git diff --check and re-check every repository status.
Never claim a dependency or feature exists from a README alone.
