# 实验计划

实验分为 PTX design path 和 SASS/hardware validation path。每个结果都必须
注明产生它的路线。

## PTX design path

1. 运行不含 custom instruction 的小型 SM90 PTX benchmark。
2. 添加一个无 network side effect 的 custom communication instruction，检查
   parser、functional result、issue timing、dependency 和 completion。
3. 让该指令生成 synthetic communication event。
4. 通过 event orchestrator 和 BookSim 运行两个 logical PTX GPU contexts。
5. 对 2/4/8 logical GPUs、Ring、NVLS-style multicast/reduction 做 sweep，
   message size 范围为 1 KiB 至 256 MiB。

Metrics 包括 instruction/cycle counts、event ready/completion time、bytes、
flits、effective bandwidth、link/switch utilization、reduction occupancy、
stall time 和 compute/communication overlap。

## SASS validation path

使用 A800 trace 验证 software pipeline；检查已有 Hopper instruction behaviour
时使用 H100/H200 trace。不得用 A800 trace 声称 H100 ISA 或 NVLS performance。
当两条路线运行同一 high-level kernel 时，应将比较视为 validation experiment，
不能把两种输入当作可互换数据。

## Reproducibility metadata

每次运行都要在 `.codex/experiments/` 中记录 Accel-Sim SHA、nested
GPGPU-Sim SHA、BookSim SHA/config、input path（PTX/SASS/synthetic）、workload、
rank count、collective/algorithm、message size、seed、command、output path、
date 和 evidence level。大型 trace 与 raw result 保留在 `.codex/` 之外。
