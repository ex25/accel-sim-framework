# Repository instructions

This repository is the Accel-Sim research checkout for the in-switch project.

Before doing work, read:

1. `.codex/README.md`
2. `.codex/rules/`
3. `.codex/knowledge/project-state.md`

Then load only the skill and research documents relevant to the task. The
`.claude/` directory is historical project guidance; do not copy it blindly.

Keep the Accel-Sim repository and nested GPGPU-Sim working repository distinct.
Do not reset, clean, stash, force-push, or delete user work. Core simulator
changes require a design note, tests, a regression result, and a Git commit.

Every research claim must be labelled `Verified`, `Observed`, `Inferred`,
`Hypothesis`, `Unknown`, or `Needs hardware validation` where ambiguity could
change an implementation or an experiment.

The project memory source of truth is `.codex/`; update it whenever a fact,
decision, limitation, or next step could matter to a later session.
