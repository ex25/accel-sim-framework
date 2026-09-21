# Research changelog

## 2026-09-21 — Complete minimal single-GPU custom PTX instruction

- Added proposed `cais.add.u32 dst, src1, src2` semantics to nested GPGPU-Sim
  commit `8f3c35fa`: lexer/opcode registration, unsigned modulo-32 functional
  execution, and integer ADD timing/resource mapping.
- Added CUDA-6+ external-PTX override support for synthetic proposed opcodes.
  Stock `ptxas` cannot assemble `cais.add`, so this focused override skips
  PTXInfo/occupancy assembly; that is an explicit harness limitation.
- Added the SM90 fixture generator, dependent-chain workload, 1/32-iteration
  launcher cases, and reusable control/custom result checker.
- Verified both iteration pairs: control/custom cycles, instructions, IPC,
  checksum, and zero error matched exactly; the existing SM90 baseline also
  remained at 5501 cycles, 1654784 instructions, IPC 300.8151, and PASS.
- Limits: this instruction has no memory/network effect and provides no real
  GPU, SASS, occupancy, multi-GPU, NVLink, NVSwitch, NVLS, or BookSim evidence.
  CTest still has no registered tests; existing build warnings remain.

## 2026-09-20 — Complete SM90 PTX execution-driven baseline

- Added a bounded CUDA 12.8 `compute_90` PTX-only workload and launcher using
  `H100-PTX` with `SM90_H100/gpgpusim.config`.
- Verified the unmodified nested GPGPU-Sim PTX path: parser acceptance,
  functional checksum, issue/timing, and simulator completion. Result: 5,501
  cycles, 1,654,784 simulated instructions, IPC 300.8151, max error
  `3.5762786865234375e-07`.
- Added the source-level call-path record from `ptx_loader`/lexer/parser to
  `ptx_instruction`, functional opcode dispatch, shader issue, and the timing
  pipeline. No nested simulator source was modified.
- Limits: this is SM90 simulator evidence only; native A800 execution is not
  applicable, and H100/H200/Hopper/NVSwitch/NVLS validation remains pending.

## 2026-09-20 — Complete single-A800 SASS/PTX gap matrix

- Added `experiments/sass_ptx_gap/`: a dual-code-object SM80 CUDA workload
  suite with bounded FMA, memory, control-flow, atomic, GEMV, and GEMM cases;
  added a reusable metric collector and the `SASS_PTX_GAP` launcher definition.
- Generated 12 real A800 NVBit traces, replayed all 12 with `A100-SASS`, and
  ran all 12 with `A100-PTX`. Native checksums and both replay paths passed;
  the result table is in `.codex/experiments/2026-09-20-sass-ptx-a800-gap.md`.
- Observed cycle deltas of −2.6% to +19.0%; GEMV/GEMM instruction and IPC
  deltas remained large despite matching high-level global read/write totals.
- Classified the initial CUDA 12.8 `-lineinfo` PTX `.loc/inlined_at` abort as a
  parser compatibility baseline failure. The passing rerun omits compile-time
  `-lineinfo` and uses `TRACE_LINEINFO=0`.
- Limits: single A800 only; no multi-GPU, NVLink/NVSwitch/NVLS, H100, or
  absolute A800 calibration claim.

## 2026-09-20 — Confirm PTX-primary design strategy

- Decision: use PTX execution-driven mode as the main path for proposed ISA,
  GPU microarchitecture, scheduler, and in-switch-computing experiments;
  retain SASS/NVBit traces as the validation/calibration path for existing
  A800/H100 behaviour.
- Updated: `.codex/README.md`, `knowledge/project-state.md`, Accel-Sim and
  research-scope rules, design options, roadmap, experiment/validation plans,
  multi-GPU trace guidance, and the Chinese project history.
- Added: `docs/17-ptx-primary-strategy.md` and the reusable
  `skills/ptx-design-baseline/` workflow, exposed through the `.agents/skills/`
  symlink.
- Rationale: real hardware cannot produce a SASS trace for a proposed
  instruction such as CAIS/DySHARP-style operations; PTX can define functional
  semantics and timing, while SASS preserves compiler/machine-level validation.
- Validation: documentation-only structural checks and `git diff --check`; no
  simulator core was changed and no hardware claim was added.
- Evidence: PTX/SASS boundary is source-verified; PTX as the design path is a
  project decision; CAIS/DySHARP exact private input-generation details remain
  `Unknown`.

## 2026-09-20 — Clarify external workload provenance and custom-ISA methodology

- Commit: current phase; the SHA is recorded by the following Git history after
  this phase is committed.
- Corrected `02-trace-pipeline.md`: the A800 `vectoradd.cu` source is the
  external NVBit SDK sample at
  `$CONDA_PREFIX/opt/nvbit-1.8/test-apps/vectoradd/vectoradd.cu`, not a file in
  the Accel-Sim repository. Added the SDK Makefile, `sm_80` override, artifact
  paths, historical-command limitation, and a reproducible command recipe.
- Added `16-simulating-custom-multigpu-isa.md`, a reader-facing explanation of
  real instruction traces, communication-event traces, synthetic new-ISA
  workloads, multi-GPU scheduling layers, CAIS/DySHARP methodology evidence,
  and the required ISA/microarchitecture/NVSwitch validation boundaries.
- Updated the research index so the new methodology document is part of the
  six-month recovery path.
- Evidence labels: the vectoradd source location is `Verified`; the exact
  historical replay shell command is `Unknown/Observed limitation`; paper
  details not stated in CAIS or DySHARP remain `Unknown` rather than inferred.
- Validation: documentation diff/path checks and later phase regression are
  required before commit. No simulator core was changed.

## 2026-09-19 — Complete P2 audit and add reader-facing concepts guide

- Commit: current phase; the SHA is recorded by the following Git history after
  this phase is committed.
- Completed the source audit of the current SM90/H100 path: binary-version
  dispatch, Hopper opcode map, GMMA latency mapping, TMA metadata/handling,
  mbarrier/SYNCS requirements, cluster fields, explicit TODOs, unsupported
  paths, and the absence of `multimem`/GPU↔GPU/NVSwitch/NVLS semantics.
- Added `.codex/docs/15-concepts-for-readers.md`, a Chinese tutorial that
  introduces GPU execution, trace fields, the real A800 call path, CTest,
  hotspot, request/reply packets, NCCL collective terms, AnyNet, endpoint
  clocks, and NVLS before using those terms in the research plan.
- Updated the documentation index, project state, and H100 capability audit.
- Validation: release build completed; the existing missing `stubgen` helper is
  tolerated by the build; endpoint unit tests passed 3 tests; analytical checker
  passed 9 rows.
- Limits: this remains source-level P2 evidence; real H100 trace replay and
  hardware correlation still require H100/H200 NVSwitch access.

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
