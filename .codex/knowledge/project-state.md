# Project state (reload first)

Updated: 2026-09-20

## Identity

- Goal: Accel-Sim + BookSim research platform for H100 NVLink/NVSwitch/NVLS.
- Main repo/branch: `accel-sim-framework`,
  `feat/nvbit-a800-booksim-nvlink`.
- Main baseline SHA: `d930ad6` (`Revert slurm.sim to the pre-2.0 template`).
- Nested GPGPU-Sim: separate ignored checkout, `91880c53`, branch `dev`.
- BookSim reference checkout: `/home/wangjz/in-switch/booksim2`, `28f4329`,
  branch `master`; not integrated. The nested GPGPU-Sim checkout already has a
  BookSim2-derived `src/intersim2` with a GPU-specific adapter.

## Environment

- One NVIDIA A800 80GB PCIe, Ampere/sm_80 record; `nvidia-smi topo -m` shows
  only the local GPU and CPU affinity, with no NVLink peer.
- Driver 570.133.20; driver-reported CUDA 12.8; toolkit `/usr/local/cuda`,
  `nvcc` 12.8.61.
- Activated `in-switch`: Python 3.11.16, GCC/G++ 13.4.0, CMake 3.31.8.
- NVBit 1.8 is installed inside the `in-switch` environment at
  `$CONDA_PREFIX/opt/nvbit-1.8`; the tracer accepts this path through the
  overridable `NVBIT_PATH` Make variable.
- GPU was idle for the 2026-09-19 smoke test. Do not kill or interfere with
  other processes if occupancy changes.

## Baseline evidence

- `cmake --build gpu-simulator/build/release -j8`: PASS on 2026-09-18.
- `ctest --test-dir gpu-simulator/build/release --output-on-failure`: no tests
  found (not a pass/fail regression result).
- Fresh replay of the existing V100 `hotspot` trace via its `justrun.sh`:
  PASS/exit 0 in about 40 seconds; output included 29,005 request and 29,005
  reply network packets. This is a V100 trace replay, not A800/H100 evidence.
- Optional `stubgen` step was missing but is explicitly tolerated by the build.
- Existing `sim_run_12.8/` outputs and `IN-SWITCH-ENV.md` report a prior full
  10/10 V100 trace replay; this session has not rerun that matrix. Treat it as
  `Observed`, not new regression evidence.
- Real A800 NVBit vectoradd trace, post-processing, and SM80 replay: PASS.
  Replay completed in 6,824 simulated cycles and about 16 seconds CPU time.
- Single-A800 SASS/PTX gap matrix: 12 bounded CUDA cases (including GEMV 64/128/256
  and GEMM 16/32/64) all passed native functional checks, A800 tracing, SASS
  replay, and PTX replay after removing CUDA-12.8 `-lineinfo`. PTX-vs-SASS cycle
  delta was −2.6% to +19.0%; GEMV instruction/IPC deltas were about −36% to
  −42%, and GEMM deltas about −20% to −29%. Full evidence:
  `.codex/experiments/2026-09-20-sass-ptx-a800-gap.md`.
- SM90 PTX baseline: a CUDA 12.8 `compute_90` PTX-only kernel passed the
  unmodified `H100-PTX` / `SM90_H100` path with 5,501 cycles, 1,654,784
  simulated instructions, IPC 300.8151, and functional max error
  `3.5762786865234375e-07`. This is **Verified** simulator software evidence,
  not H100 hardware validation. Full record:
  `.codex/experiments/2026-09-20-sm90-ptx-baseline.md`.
- Standalone BookSim two-endpoint smoke/sweep: PASS at reference SHA `28f4329`.
  This validates generic transport only, not physical NVLink.

## Confirmed execution strategy

- PTX execution-driven mode is the primary path for proposed ISA,
  microarchitecture, scheduler, and in-switch-computing experiments.
- SASS/NVBit traces are the validation path for existing A800/H100 behaviour,
  compiler effects, and hardware calibration.
- PTX does not by itself provide multi-GPU rank/collective/NVSwitch semantics;
  those still require an event schema, orchestrator, semantic layer, and
  BookSim transport.
- A custom instruction that does not exist in hardware must enter through PTX or
  a clearly labelled synthetic instruction/event generator, not an imagined
  “real” SASS trace.

## Status checklist

- [x] repository audit and research branch
- [x] `.claude/` audit
- [x] `AGENTS.md` audit/creation
- [x] `.codex/` infrastructure established
- [x] A800 environment capture
- [x] baseline build attempt
- [x] one pre-generated trace replay (V100 source)
- [ ] baseline test with registered test cases
- [x] A800 real trace smoke test
- [x] trace pipeline/source mapping (one real A800 `LDG.E.64` autopsy)
- [x] Accel-Sim architecture audit (PTX/SASS boundary documented)
- [x] Hopper/H100 support audit (P2 source audit complete; hardware correlation pending)
- [x] trace-format audit (version-6 record and parser fields)
- [x] single-vs-multi-GPU answer backed by source (compute trace is not distributed execution)
- [x] current multi-GPU/NVLink/NVSwitch/NVLS capability matrix (conservative unknowns recorded)
- [x] NCCL NVLS source study (primary-source review; pinned local NCCL checkout pending)
- [x] Existing BookSim-derived `intersim2` boundary identified
- [x] BookSim architecture and dependency/provenance decision
- [x] clock/data coupling decision (timestamped endpoint adapter design)
- [x] Multi-GPU architecture design (transport/semantic-layer boundary)
- [x] synthetic transport prototype
- [ ] Ring baseline
- [ ] NVSwitch model
- [ ] NVLS semantics prototype
- [ ] compute + communication integration
- [ ] statistics/experiment automation
- [x] analytical/public-data validation (synthetic BookSim accounting only)
- [x] PTX-primary/SASS-validation strategy confirmed
- [x] single-A800 SASS/PTX frontend gap matrix (bounded GEMV/GEMM)
- [x] SM90 PTX execution-driven baseline
- [ ] minimal custom PTX instruction
- [ ] PTX instruction-to-communication-event boundary
- [ ] real H100 validation

## Active blockers and next action

1. The A800 vectoradd source is external to this repository at
   `$CONDA_PREFIX/opt/nvbit-1.8/test-apps/vectoradd/vectoradd.cu`; the historical
   replay command was not preserved and must be captured in future runs.
2. The CUDA 12.8 PTX parser requires the gap experiment's compatibility
   condition (no compile-time `-lineinfo`, `TRACE_LINEINFO=0`) until the parser
   is updated; do not classify that failure as a workload gap.
3. Read `15-concepts-for-readers.md`, `16-simulating-custom-multigpu-isa.md`,
   and `17-ptx-primary-strategy.md` before changing trace format, PTX ISA, or
   network semantics.
4. Define one single-GPU custom PTX instruction and map its parser,
   functional, dependency, timing, and completion tests before adding network
   side effects.
5. Define the independent communication-event schema and logical rank/stream
   orchestrator before any Ring implementation.
6. Implement the narrow BookSim endpoint facade, then obtain H100/H200
   NVSwitch access for SM90/NVLS trace and hardware calibration.

## Evidence rule

All claims in this file are either direct observations above or links to the
full documents. Update this file at every milestone and commit it with the
work it describes.
