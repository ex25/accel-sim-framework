---
name: document-change
description: Record an Accel-Sim research change in the correct source of truth so rationale, evidence, and limitations survive context loss.
---

## Routing

Stable policy -> .codex/rules/.
Repeated workflow -> .codex/skills/ (canonical); .agents/skills/ is symlinked.
Compact current fact/state -> .codex/knowledge/.
Architecture, call path, alternatives, or diagrams -> .codex/docs/.
Design choice -> .codex/docs/DECISIONS.md.
Chronology/commit -> .codex/docs/CHANGELOG.md.
Experiment metadata/entrypoint -> .codex/experiments/.

## Steps

1. Choose one source of truth; do not copy paragraphs into multiple files.
2. Add date, branch, all relevant SHAs, commands, tests, evidence label,
   assumptions, limitations, and next step.
3. For code/model changes update the numbered implementation note with problem,
   design, alternatives, files/classes/functions, call path, configuration,
   before/after, tests, validation, commit, and future work.
4. Keep .codex high-signal: no raw traces, binaries, build output, or long logs.
5. Run git diff --check, verify paths/links, and update project-state when the
   milestone or blocker changes.
