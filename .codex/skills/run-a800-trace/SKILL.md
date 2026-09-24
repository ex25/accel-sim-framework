---
name: run-a800-trace
description: 在 GPU 与 tracer toolchain 可用时，于本地 NVIDIA A800 生成并 replay 小型真实 CUDA/NVBit trace。
---

## 边界

这是 software-pipeline smoke test，不是 H100、NVLink、NVSwitch、NVLS 或 NCCL
validation。不得停止或 reset 无关 GPU process。

## Preconditions

记录 `nvidia-smi`、`nvidia-smi -L`、`nvidia-smi topo -m` 与 `nvcc --version`，
包括 model、compute capability、driver、CUDA、compiler、NVBit version/path、
occupancy 与 fresh output directory。缺 GPU、SDK 或 benchmark 时记录 blocker，
不要用 synthetic run 代替。

## Workflow

1. 选择 tiny deterministic CUDA benchmark，记录 source 与 compile flags。
2. 用已记录的 NVBit installation/command build `util/tracer_nvbit`。
3. 运行当前 `run_hw_trace.py` wrapper，记录 kernel filter、trace output、
   spinlock mode、register-value mode 与 device selection。
4. 保存 raw file、kernelslist、trace version、size、kernel count、warning 与
   wall time；不提交 raw trace。
5. 运行 post-traces-processing，记录 input/output path。
6. 用最小兼容 Accel-Sim config replay，记录 command、exit code、runtime、stats、error。
7. 将 failure 分类为 source/build/NVBit/runtime/parser/simulator/environment。
8. 更新 `docs/gpu-execution.md`、local-environment/trace-format knowledge 与
   `docs/changelog.md`。

完成条件：exact command、SHA、trace/kernel metadata、output、failure classification、
evidence label 与 limitation 均已记录。成功只证明指定的 A800
CUDA → NVBit → post-process → replay path。
