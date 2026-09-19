# Research changelog

## 2026-09-19 — Trace/transport analysis and project history

- Branch: `feat/nvbit-a800-booksim-nvlink`, based on upstream `d930ad6`.
- Commit: current phase (`docs(analysis): audit trace and define transport contract`;
  see the branch `HEAD`, because recording a commit's own SHA would change that
  SHA).
- Added the real A800 `LDG.E.64` source-level autopsy from NVBit emission
  through `.tracez`, parser, warp instruction, memory coalescing, GPU memory
  interconnect, and completion gating.
- Defined the timestamped BookSim endpoint API and rational global-picosecond
  clock contract without modifying Accel-Sim or BookSim core code.
- Added analytical bytes/flits checks, the dependency-free endpoint contract
  and its unit tests, revised phase-oriented commit policy, and the Chinese
  project history document.
- Validation: analytical checker passed 9 synthetic sweep rows; endpoint
  contract tests passed 3 tests; BookSim two-node smoke passed.
- Limits: all network values remain synthetic; no embedded BookSim facade,
  H100, NVSwitch, NCCL, or NVLS validation.

## 2026-09-19 — A800 trace and two-endpoint BookSim transport

- Commit: `dc7ffa9` (`feat(transport): verify A800 trace and add two-GPU BookSim model`).
- Installed NVBit 1.8 inside the `in-switch` conda environment and made the
  tracer SDK path configurable through `NVBIT_PATH`.
- Verified A800 SDK vectoradd → NVBit → post-process → `.tracez` → Accel-Sim
  SM80 replay: exit 0, 6,824 simulated cycles, 98 CTAs, and 75,000 request /
  reply packets.
- Added the standalone BookSim AnyNet two-logical-GPU topology and nine-case
  link-latency/packet-size sweep against external BookSim SHA `28f4329`.
- Limits: generic transport only; no physical NVLink, NVSwitch, NVLS, NCCL, or
  H100 validation.

## 2026-09-18 — Repository memory, baseline, and capability audit

- Commit: `b659f94` (`docs(codex): establish research memory and baseline audit`).
- Audited repository boundaries, `.claude/`, `AGENTS.md`, nested GPGPU-Sim,
  external BookSim, build dependencies, and local A800/CUDA/toolchain state.
- Established `.codex/` rules, skills, knowledge, docs, experiment metadata,
  `.agents/skills` symlinks, and persistent evidence labels.
- Documented the existing GPGPU-Sim `src/intersim2` BookSim-derived
  shader↔memory path and separated it from future GPU↔GPU/NVLink work.
- Recorded release build success, the existing V100 trace replay, and the
  absence of registered CTest cases.
- Limits: this audit predates the real A800 trace and does not establish
  Hopper/NVSwitch/NVLS behavior.

## Upstream base

- Accel-Sim base: `d930ad6` (`Revert slurm.sim to the pre-2.0 template`).
- Hopper support reference in the checkout: upstream commit `0db0445`.
- External BookSim reference: `/home/wangjz/in-switch/booksim2`, SHA `28f4329`.
- Nested GPGPU-Sim reference: separate checkout, SHA `91880c53` at the time of
  the audit; it is not a tracked submodule of this repository.

The three project commits above intentionally replace the earlier fine-grained
changelog/state/SHA follow-up commits. Future work should add one commit per
completed logical phase, with implementation, tests, documentation, state, and
changelog together.
