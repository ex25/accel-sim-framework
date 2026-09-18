# Project state (reload first)

Updated: 2026-09-18

## Identity

- Goal: Accel-Sim + BookSim research platform for H100 NVLink/NVSwitch/NVLS.
- Main repo/branch: `accel-sim-framework`,
  `research/nvls-accelsim-study`.
- Main baseline SHA: `d930ad6` (`Revert slurm.sim to the pre-2.0 template`).
- Nested GPGPU-Sim: separate ignored checkout, `91880c53`, branch `dev`.
- BookSim checkout: `/home/wangjz/in-switch/booksim2`, `28f4329`, branch
  `master`; not integrated.

## Environment

- One NVIDIA A800 80GB PCIe, Ampere/sm_80 record; `nvidia-smi topo -m` shows
  only the local GPU and CPU affinity, with no NVLink peer.
- Driver 570.133.20; driver-reported CUDA 12.8; toolkit `/usr/local/cuda`,
  `nvcc` 12.8.61.
- Activated `in-switch`: Python 3.11.16, GCC/G++ 13.4.0, CMake 3.31.8.
- GPU was busy during the audit. Do not kill or interfere with other processes.

## Baseline evidence

- `cmake --build gpu-simulator/build/release -j8`: PASS on 2026-09-18.
- `ctest --test-dir gpu-simulator/build/release --output-on-failure`: no tests
  found (not a pass/fail regression result).
- Optional `stubgen` step was missing but is explicitly tolerated by the build.
- Existing `sim_run_12.8/` outputs and `IN-SWITCH-ENV.md` report a prior full
  10/10 V100 trace replay; this session has not rerun that matrix. Treat it as
  `Observed`, not new regression evidence.
- NVBit SDK/tracer binary was not found. A800 trace smoke test: `Unknown`,
  blocked by SDK/artifact setup and GPU availability.

## Status checklist

- [x] repository audit and research branch
- [x] `.claude/` audit
- [x] `AGENTS.md` audit/creation
- [x] `.codex/` infrastructure established
- [x] A800 environment capture
- [x] baseline build attempt
- [ ] baseline test with registered test cases
- [ ] A800 real trace smoke test
- [ ] trace pipeline/source mapping
- [ ] Accel-Sim architecture audit
- [ ] Hopper/H100 support audit
- [ ] trace-format audit
- [ ] single-vs-multi-GPU answer backed by source
- [ ] current multi-GPU/NVLink/NVSwitch/NVLS capability matrix
- [ ] NCCL NVLS source study
- [ ] BookSim architecture and dependency decision
- [ ] clock/data coupling decision
- [ ] Multi-GPU architecture design
- [ ] synthetic transport prototype
- [ ] Ring baseline
- [ ] NVSwitch model
- [ ] NVLS semantics prototype
- [ ] compute + communication integration
- [ ] statistics/experiment automation
- [ ] analytical/public-data validation
- [ ] real H100 validation

## Active blockers and next action

1. Reserve/confirm GPU access before running CUDA/NVBit workloads.
2. Install or locate a compatible NVBit release and build the tracer without
   changing simulator core code.
3. Finish source-grounded audits and document the one-instruction trace
   autopsy before designing the adapter.

## Evidence rule

All claims in this file are either direct observations above or links to the
full documents. Update this file at every milestone and commit it with the
work it describes.
