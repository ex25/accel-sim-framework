---
name: ptx-design-baseline
description: Establish or extend the PTX execution-driven path for a proposed GPU instruction or microarchitecture feature before adding multi-GPU/NVLS semantics.
---

## When to use

Use this skill for a new PTX instruction, a hypothetical GPU feature, a custom
GPU scheduler experiment, or any CAIS/DySHARP-like design that cannot be
captured by a real A800/H100 SASS trace.

Do not use it as a replacement for the A800 SASS smoke test or for H100 hardware
correlation. Those are separate validation workflows.

## Inputs

- current branch and main/nested repository SHAs;
- PTX or CUDA workload and target configuration;
- instruction semantic note: operands, result, ordering, latency, completion,
  and possible communication side effect;
- baseline PTX command and expected output;
- evidence label: `Verified`, `Synthetic`, `Hypothesis`, or `Needs hardware validation`.

## Steps

1. Read `.codex/rules/accel-sim-policy.md`,
   `.codex/docs/17-ptx-primary-strategy.md`, and the relevant implementation
   note before editing.
2. Run the unmodified PTX baseline and record command, config, exit code,
   cycle/instruction summary, and any baseline failure.
3. Map the current path:
   PTX lexer/parser → PTX IR → functional semantics → warp instruction →
   shader timing/issue → memory or completion.
4. Define the smallest instruction syntax and semantic contract. Do not add
   rank, packet, multicast, or reduction state until the single-GPU instruction
   path is tested.
5. Modify only the named parser/opcode/IR/semantics/timing files. Preserve the
   SASS trace path unless the design explicitly requires a shared interface.
6. Add focused tests for parser recognition, functional result, register
   dependency, issue latency, resource selection, and completion.
7. If the instruction emits a communication event, serialize that event through
   an explicit schema; never hide collective state in an ordinary memory address.
8. Run the focused test, the PTX baseline, and the smallest relevant regression.
9. Update the implementation note, `.codex/knowledge/project-state.md`,
   `DECISIONS.md` when the boundary changes, and `CHANGELOG.md`.
10. Run `git diff --check`, inspect both repository statuses, and commit one
    complete logical phase.

## Artifacts produced

- PTX baseline command and output metadata;
- semantic contract for the instruction;
- source call-path note with file/function references;
- focused parser/functional/timing tests;
- optional communication-event fixture;
- implementation note and evidence label.

## Completion criteria

The instruction is recognized and executed in PTX mode, dependency and timing
tests pass, baseline behaviour remains classified, and the documentation states
what PTX proves and what still requires real SASS/H100 validation. A successful
PTX run alone never proves hardware accuracy.
