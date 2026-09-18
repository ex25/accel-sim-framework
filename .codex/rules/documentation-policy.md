# Documentation and source-of-truth policy

`Context is temporary; the repository is persistent.` If a fact, call path,
model assumption, hardware parameter, test procedure, decision, limitation, or
next step can affect future work, capture it in `.codex/` before closing the
phase.

Use one primary home for each kind of information:

| Location | Purpose | Required style |
| --- | --- | --- |
| `.codex/rules/` | Stable policy every session follows | Normative, actionable, rarely changed |
| `.codex/skills/` | Repeatable workflow | Inputs, steps, commands, artifacts, validation, exit criteria |
| `.codex/knowledge/` | Fast-loading current facts | Compact, structured, evidence labels, current SHAs/state |
| `.codex/docs/` | Human-readable research record | Rationale, call paths, alternatives, diagrams, citations |
| `.codex/experiments/` | Reproducible experiment entrypoints | Metadata, commands, run index; raw output elsewhere |
| `AGENTS.md` | Thin bootstrap pointer | Do not duplicate the rules |

An implementation that changes future work updates the relevant knowledge and
implementation note in the same logical phase as the code. Record important
architecture choices in `DECISIONS.md` and chronological work in
`CHANGELOG.md`. Link to the source of truth instead of copying five divergent
summaries.

Write evidence labels (`Verified`, `Observed`, `Inferred`, `Hypothesis`,
`Unknown`, `Needs hardware validation`) next to claims. Include exact file,
class/function, command, SHA, and date when they are needed to reproduce the
claim. Do not put terminal dumps, binaries, multi-GB traces, or temporary
thoughts in `.codex/`.
