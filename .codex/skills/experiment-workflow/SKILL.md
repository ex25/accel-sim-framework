---
name: experiment-workflow
description: Design and record reproducible synthetic, trace-driven, BookSim, or future H100/NVSwitch experiments.
---

Define objective, model/hardware evidence level, workload, GPU/rank count,
collective, algorithm, message sizes, topology/config, seed, SHAs, commands,
outputs, metrics, and acceptance criteria before running. Use the validation
ladder from unit tests to analytical/synthetic/public-data/real-H100 evidence.
Store metadata and an index under `.codex/experiments/`; keep raw outputs in a
gitignored run directory. Never infer NVLS hardware accuracy from transport-only
or synthetic results.
