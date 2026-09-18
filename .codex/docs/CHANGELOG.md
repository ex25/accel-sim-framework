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

## 2026-09-18 — Existing BookSim-derived interconnect discovered

- Source evidence: nested GPGPU-Sim `src/intersim2`,
  `gpgpu-sim/src/gpgpu-sim/icnt_wrapper.cc`,
  `InterconnectInterface`, and `GPUTrafficManager`.
- Result: BookSim-derived timing already exists for shader↔memory traffic and
  is selected by `-network_mode 1`; no evidence yet of GPU↔GPU/NVLink/NVSwitch/
  NVLS semantics.
- Impact: future work should extend/reuse this boundary rather than blindly
  introduce a second BookSim copy.
- Standalone reference smoke: `make -C src -j8` and `src/booksim
  src/examples/singleconfig` completed; output was recorded as transport-only.
- Commit: `99189a8` (`docs(accelsim): document existing booksim interconnect boundary`).

## 2026-09-18 — Clarify trace/NVLink gap evidence

- Source search found no `multimem` symbols; NVLink references are monitoring
  helpers, not GPU-to-GPU simulator timing. Per-context trace files are not rank
  or fabric events.
- Commit: `40ede7c` (`docs(accelsim): clarify gpu trace and nvlink gaps`).
