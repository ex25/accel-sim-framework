# Research changelog

## 2026-09-18 — M0/M1 context and baseline audit

- Branch: `research/nvls-accelsim-study`, base `d930ad6`.
- Added: `AGENTS.md`, `.codex/` rules/skills/knowledge/docs skeleton, and
  `.agents/skills` symlinks.
- Verified: main and BookSim repository boundaries; A800/driver/CUDA/topology;
  activated toolchain; release build succeeds.
- Observed: CTest has no registered tests; prior simulation artifacts exist;
  NVBit SDK/tracer binary is absent; GPU is busy.
- Not done: no core simulator change, no trace generation, no H100/NVSwitch
  validation, no BookSim integration.
- Commit: `4421d43` (`docs(codex): establish project memory and research rules`).
