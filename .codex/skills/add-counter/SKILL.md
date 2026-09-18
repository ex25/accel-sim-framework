---
name: add-counter
description: Add a simulator statistic or counter end to end, including declaration, initialization, increment sites, registration, tests, and documentation.
---

## When to use

Use for a new GPGPU-Sim/Accel-Sim statistic, hardware-like counter, or debug
metric. First identify whether the owner is a shader core, cluster, memory
partition, network, kernel, or global object.

## Inputs

Record the counter name, units/type, owner class, increment event, reset/lifetime,
aggregation semantics, output label, expected baseline, and test workload.

## Workflow

1. Search counters in the same subsystem and follow declaration -> constructor
   or reset -> increment/read -> registration/reporting. Do not create a
   parallel statistics path.
2. Confirm whether the value is integer/float, per object or aggregate, reset
   per kernel/epoch, and sum/max/average. Define what one increment means.
3. Modify the smallest owning header/source, stats registration, and reporting
   files. Preserve existing naming and output conventions.
4. Add a deterministic focused test or replay that exercises the event and
   checks label/value. Add a zero-event case when meaningful.
5. Run the focused test and smallest regression; compare baseline and classify
   failures.
6. Document units, scope, reset point, increment call path, files/symbols,
   commands/results, and limitations in the implementation note and changelog.
7. Run git diff --check and commit with a specific subject such as
   feat(stats): count ...; the body must say what/why/files/how/tests/limits.

## Completion

The counter initializes safely, increments exactly once per intended event,
reports defined units and scope, has focused evidence, and is documented for
future sessions.
