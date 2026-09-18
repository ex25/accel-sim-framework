---
name: experiment-workflow
description: Design and record reproducible synthetic, trace-driven, BookSim, or future H100/NVSwitch experiments with explicit evidence boundaries.
---

## Before running

Define objective, hypothesis, evidence level, workload/trace, GPU or synthetic
topology, rank count, collective, algorithm, message sizes, clock/config, seed,
all SHAs, commands, output directory, metrics, and acceptance criteria.

## Workflow

1. Create metadata under .codex/experiments/ pointing to a gitignored run
   directory. Include date, config, seed, SHAs, commands, and outputs.
2. Run the smallest case first; summarize latency, effective bandwidth,
   bytes/flits, utilization, reduction/completion ordering, and failures.
3. Compare analytical bounds and the prior baseline before scaling. Prefer 2/4/8
   GPU synthetic Ring and NVLS cases and message sizes from 1 KiB to 256 MiB.
4. Use the ladder: unit -> analytical -> synthetic network -> public-data trend
   -> real H100/NVSwitch -> trace/correlation.
5. Update experiment index, project-state, and limitations. Never call copied
   traces, BookSim transport, or synthetic NVLS a real hardware result.
