# Design decisions

## D001 — Keep research memory in `.codex/`

- Date: 2026-09-18
- Context: future Codex sessions must recover without chat history.
- Decision: rules, reusable workflows, compact facts, full research, and
  experiment metadata live under `.codex/`; `AGENTS.md` is a thin loader.
- Why: one persistent source of truth with progressive disclosure.
- Validation: perform context-loss reload checks at each milestone.

## D002 — Expose canonical `.codex/skills` through `.agents/skills`

- Date: 2026-09-18
- Context: project requirements name `.codex/skills`, while current Codex
  discovery scans repository `.agents/skills`.
- Decision: author skills once under `.codex/skills`; add symlinks under
  `.agents/skills`.
- Trade-off: Git records links and some tools may not follow them; duplicated
  skill copies are forbidden because they can diverge.
- Evidence: official Codex skills documentation (accessed 2026-09-18) states
  repository discovery scans `.agents/skills` and each skill requires
  `SKILL.md` with `name` and `description`.

## D003 — BookSim remains an external analysis target for M0/M1

- Date: 2026-09-18
- Context: `booksim2` is a separate checkout at `28f4329` and no adapter exists.
- Decision: do not vendor/copy it during repository reconnaissance. Decide
  submodule/external checkout/adapter after reading its build and API.
- Why: preserves upstream history and prevents premature coupling.

## D004 — Separate compute traces from communication events

- Date: 2026-09-18
- Context: one SASS trace lacks collective rank/topology/transport state.
- Decision: preserve current compute trace format initially; represent future
  communication as an independent event model or orchestrated layer unless
  source evidence proves a compatible extension is necessary.
- Validation: trace autopsy and multi-GPU capability audit are required before
  implementation.
