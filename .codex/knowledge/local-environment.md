# Local environment

Updated: 2026-09-18. Commands were run on this host unless marked otherwise.

| Item | Value | Evidence/status |
|---|---|---|
| GPU | 1 × NVIDIA A800 80GB PCIe | `nvidia-smi -L`, Verified |
| GPU generation | Ampere; compute capability 8.0 (sm_80) | `nvidia-smi --query-gpu=compute_cap`, Verified |
| Driver | 570.133.20 | `nvidia-smi`, Verified |
| Driver CUDA | 12.8 | `nvidia-smi`, Verified |
| Toolkit | `/usr/local/cuda`, CUDA 12.8.61 | `nvcc --version`, Verified |
| Topology | GPU attached to CPU; no NVLink peer shown | `nvidia-smi topo -m`, Verified |
| OS | Ubuntu 20.04.6 / x86_64 (from `IN-SWITCH-ENV.md`) | Observed, refresh command pending |
| Activated Python | 3.11.16 | `activate-accel-sim.sh`, Verified |
| Activated compiler | GCC/G++ 13.4.0 | `activate-accel-sim.sh`, Verified |
| Activated CMake | 3.31.8 | `activate-accel-sim.sh`, Verified |
| Conda env | `/home/wangjz/miniconda3/envs/in-switch` | activation output, Verified |
| NVBit | SDK/tracer artifact not found under `util/tracer_nvbit` | Unknown/blocker |

The GPU reported active external Python processes during capture. No process
was stopped. A800 is suitable for validating the software trace pipeline, but
cannot validate SM90, H100 NVLink4, NVSwitch, or NVLS hardware semantics.
