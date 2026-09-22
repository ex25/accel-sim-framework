# 本地环境

更新时间：2026-09-19。除非另有说明，命令均在本主机执行。

| 项目 | 值 | Evidence/status |
|---|---|---|
| GPU | 1 × NVIDIA A800 80GB PCIe | `nvidia-smi -L`，Verified |
| GPU generation | Ampere；compute capability 8.0（sm_80） | `nvidia-smi --query-gpu=compute_cap`，Verified |
| Driver | 570.133.20 | `nvidia-smi`，Verified |
| Driver CUDA | 12.8 | `nvidia-smi`，Verified |
| Toolkit | `/usr/local/cuda`，CUDA 12.8.61 | `nvcc --version`，Verified |
| Topology | GPU 连接 CPU；没有 NVLink peer | `nvidia-smi topo -m`，Verified |
| OS | Ubuntu 20.04.6 / x86_64 | `IN-SWITCH-ENV.md`，Observed |
| Activated Python | 3.11.16 | `activate-accel-sim.sh`，Verified |
| Activated compiler | GCC/G++ 13.4.0 | `activate-accel-sim.sh`，Verified |
| Activated CMake | 3.31.8 | `activate-accel-sim.sh`，Verified |
| Conda env | `/home/wangjz/miniconda3/envs/in-switch` | activation output，Verified |
| NVBit | 1.8，显式使用 `NVBIT_PATH` build tracer | Verified |

首次无权限 container query 无法访问 driver；同一主机的 GPU-access query 显示
A800 idle、使用 4 MiB、没有运行 process。没有停止任何 process。A800 适合
验证 software trace pipeline，但不能验证 SM90、H100 NVLink4、NVSwitch 或
NVLS hardware semantics。
