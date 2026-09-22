---
name: experiment-workflow
description: 设计并记录可复现的 synthetic、trace-driven、BookSim 或未来 H100/NVSwitch experiment，明确 evidence boundary。
---

## 运行前

定义 objective、hypothesis、evidence level、workload/trace、GPU 或 synthetic
topology、rank count、collective、algorithm、message size、clock/config、seed、
所有 SHA、command、output directory、metric 与 acceptance criteria。

## Workflow

1. 在 `.codex/experiments/` 创建指向 gitignored run directory 的 metadata，
   包含 date、config、seed、SHA、command 与 output。
2. 先跑最小 case，汇总 latency、bandwidth、bytes/flits、utilization、
   reduction/completion ordering 与 failure。
3. scaling 前比较 analytical bound 与 prior baseline；优先 2/4/8 GPU synthetic
   Ring/NVLS，message size 1 KiB–256 MiB。
4. 使用 evidence ladder：unit → analytical → synthetic network → public-data
   trend → real H100/NVSwitch → trace/correlation。
5. 更新 experiment index、project state 与 limitation。不得把 copied trace、
   BookSim transport 或 synthetic NVLS 称为 real hardware result。
