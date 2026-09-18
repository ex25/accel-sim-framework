---
name: run-a800-trace
description: Generate and replay a small real CUDA/NVBit trace on the local NVIDIA A800 when the GPU and tracer toolchain are available.
---

Check GPU occupancy, driver/toolkit, CUDA device visibility, NVBit compatibility,
benchmark source, and clean output paths first. Never stop unrelated GPU
processes. Record source, compile/tracer commands, trace version/files/sizes,
kernel count, post-processing, simulator config, outputs, errors, and wall
times. A successful A800 run validates the software trace pipeline only; label
all Hopper/NVLink/NVSwitch/NVLS claims as unsupported. If SDK or GPU access is
missing, document the blocker and stop rather than substituting a synthetic run.
