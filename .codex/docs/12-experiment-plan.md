# Experiment plan

First synthetic matrix: 2/4/8 logical GPUs; AllReduce; Ring then NVLS;
1 KiB–256 MiB messages; metrics include latency, effective bandwidth, bytes,
link/switch utilization, reduction utilization, and completion order. Begin
with unit/analytical checks, then BookSim transport trends. These experiments
validate model sanity only, not H100 hardware.

Every run records Accel-Sim SHA, BookSim SHA, config, workload, rank count,
collective/algorithm, size, seed, command, output path, date, and evidence
level under `.codex/experiments/`.
