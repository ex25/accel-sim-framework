# Research changelog

## 2026-09-19 — A800 trace and BookSim two-endpoint transport

- Branch: `feat/nvbit-a800-booksim-nvlink`, base `85cccb1`.
- Installed NVBit 1.8 under the `in-switch` conda environment and made the
  tracer SDK path overridable instead of copying the SDK into the repository.
- Verified A800 SDK vectoradd -> NVBit -> post-process -> `.tracez` ->
  Accel-Sim SM80 replay. Replay exit 0, 6,824 cycles, 98 CTAs, and 75,000
  request/reply packets.
- Added a standalone BookSim two-node AnyNet prototype plus a nine-case link
  latency/packet-size sweep. Smoke and sweep pass against BookSim `28f4329`.
- Validation boundary: real A800 software pipeline plus synthetic BookSim
  transport only; no physical NVLink, NVSwitch, NVLS, or H100 validation.
- Commit: `4cad91d` (`feat(transport): verify A800 trace and add two-GPU BookSim model`).

## 2026-09-19 — Strengthen persistent rules and reusable skills

- Commit: `edf95cf` (`docs(codex): strengthen research rules and reusable skills`).
- Completed: expand `.codex/rules/` using the checked-in `.claude/rules/`
  source map; replace short skill stubs with executable workflows; add the
  existing `add-counter` workflow; keep `.agents/skills/` as symlinks to the
  canonical `.codex/skills/` directories.
- Commit policy now requires a specific subject plus a short body describing
  what functionality changed, why, files/classes/functions, how the before/
  after behaviour differs, tests/results, and limitations. The examples were
  checked against representative upstream history (`0db0445`, `9be0e3c`,
  `d7f397a`).
- Validation: all nine skills pass
  `skill-creator/scripts/quick_validate.py`; no simulator core was modified.
- Evidence: `Verified` for file/link structure and skill validation;
  `Observed` for the upstream commit style examples.

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

## 2026-09-18 — Baseline V100 trace replay

- Command: existing `sim_run_12.8/.../QV100-SASS/justrun.sh`, with output
  captured as `baseline-replay-20260918.log` in the ignored run directory.
- Result: exit 0 in about 40 seconds; request/reply packet counts both 29,005.
- Evidence: verified software trace replay only; source trace is V100 and the
  selected config uses the existing GPU memory interconnect, not NVLink/NVLS.
- Commit: `12005db` (`test(trace): record baseline v100 replay`).
