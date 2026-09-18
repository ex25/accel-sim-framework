---
name: run-a800-trace
description: Generate and replay a small real CUDA/NVBit trace on the local NVIDIA A800 when the GPU and tracer toolchain are available.
---

## Boundary

This is a software-pipeline smoke test, not H100, NVLink, NVSwitch, NVLS,
or NCCL validation. Never stop or reset unrelated GPU processes.

## Preconditions

Run and record nvidia-smi, nvidia-smi -L, nvidia-smi topo -m, and nvcc
--version. Record model, compute capability, driver, CUDA, compiler, NVBit
version/path, occupancy, and a fresh output directory. If GPU, SDK, or
benchmark access is missing, document the blocker instead of substituting a
synthetic run.

## Workflow

1. Choose a tiny deterministic CUDA benchmark and record source and compile
   flags.
2. Build util/tracer_nvbit with the documented NVBit installation and make
   command.
3. Run the current run_hw_trace.py wrapper and record kernel filtering,
   trace output, spinlock mode, register-value mode, and device selection.
4. Capture raw files, kernelslist, trace version, sizes, kernel count, warnings,
   and wall time; do not commit raw traces.
5. Run post-traces-processing and record input/output paths.
6. Replay with the smallest compatible Accel-Sim config; record command, exit
   code, runtime, stats, and errors.
7. Classify failures as source, build, NVBit, runtime, parser, simulator, or
   environment failures.
8. Update trace-pipeline docs, local-environment or trace-format knowledge, and
   CHANGELOG with a compact reproducibility record.

## Completion

Exact commands, SHAs, trace/kernel metadata, outputs, failure classification,
evidence label, and limitations are recorded. Success proves only the named
CUDA -> NVBit -> post-process -> replay path on A800.
