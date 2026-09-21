# Accel-Sim + BookSim NVLS research memory

## Project goal

Extend Accel-Sim with a research-grade GPU interconnect model backed by
BookSim, then study single-node H100/H200 NVLink + NVSwitch + NVLS collective
behaviour. The intended first target is 2/4/8-GPU AllReduce, comparing Ring and
NVLS. The project has two deliberately different input paths:

1. **PTX-first design path** — the main path for new ISA, new GPU
   microarchitecture, custom scheduling, and in-switch computing designs that
   do not exist in current NVIDIA hardware.
2. **SASS validation path** — real NVBit traces from A800/H100 used to check
   existing GPU behaviour, compiler effects, and calibration assumptions.

This is a staged research project, not a claim that the current checkout already
models NVSwitch or NVLS.

## Current state

- Main repository: `accel-sim-framework`, branch
  `feat/nvbit-a800-booksim-nvlink`, based on upstream `d930ad6`.
- Nested GPGPU-Sim is a separate ignored Git checkout on
  `feat/cais-add-ptx` at `8f3c35fa`; it is not a tracked submodule in the
  Accel-Sim repository.
- Accel-Sim already contains a BookSim2-derived `gpu-simulator/gpgpu-sim/src/intersim2`
  implementation, selected through `-network_mode 1`; it models the existing
  GPU-to-memory interconnect. The separate `../booksim2` checkout at `28f4329`
  is an upstream/reference checkout, not yet a dependency or source change in
  this branch.
- The release build already exists and rebuilt successfully with the activated
  `in-switch` environment. CTest currently discovers no tests.
- One physical NVIDIA A800 80GB PCIe is present. It is Ampere/sm_80 in the
  current hardware record, and has no NVLink peer on this host.
- NVBit 1.8 is installed inside the `in-switch` conda environment. A real A800
  vectoradd trace, post-processing, and SM80 replay passed on 2026-09-19.
- A standalone two-endpoint BookSim AnyNet prototype and 1/4/8-cycle link,
  1/4/16-flit sensitivity sweep passed. It is generic transport, not calibrated
  physical NVLink or an embedded Accel-Sim integration.

## How a new session starts

Read this file, `.codex/rules/`, and
`.codex/knowledge/project-state.md`. Check Git status for both the main and
nested repositories. For implementation work, read the relevant design note
and skill before editing. Never infer H100/NVSwitch/NVLS validation from A800
or synthetic BookSim results.

## Directory guide

- `rules/`: persistent policy and evidence labels.
- `skills/`: reusable project workflows. `.codex/skills/` is the canonical
  editing location; `.agents/skills/` contains only symlinks so Codex skill
  discovery sees the same files without creating a second copy. Current
  workflows include bootstrap, reconnaissance, A800 trace, trace autopsy,
  PTX design baseline, BookSim analysis, feature implementation, documentation,
  experiments, and counter addition.
- `knowledge/`: compact, high-signal current facts.
- `docs/`: human-readable architecture, research, decisions, and changelog.
- `experiments/`: experiment specifications and reproducibility metadata only;
  raw traces and large outputs stay outside `.codex/` and are gitignored.

## Current milestone

M2 A800 trace pipeline, standalone two-endpoint BookSim transport, real trace
instruction autopsy, analytical checks, and the endpoint/clock reference
contract are complete. M3's SM90 PTX baseline and minimal single-GPU
`cais.add.u32` custom-instruction experiment are complete. The next milestone
is the separate instruction-to-communication-event boundary. BookSim
integration remains a separate transport milestone; no multi-GPU or network
side effect has been added to the custom instruction.

## Important decisions

1. Use PTX execution-driven mode as the primary path for proposed ISA and
   microarchitecture changes. Do not wait for unavailable hardware traces for
   instructions that do not exist in hardware.
2. Keep real SASS traces as a validation/calibration path, not as the primary
   input format for a hypothetical GPU.
3. Keep compute traces and future communication events conceptually separate;
   do not replicate one single-GPU trace and call it a multi-GPU execution.
4. Reuse/analyse the existing BookSim-derived `intersim2` boundary before
   introducing another copy. Treat it as a generic network timing backend;
   NVSwitch topology, multicast, reduction, and NVLS completion semantics
   remain a separate layer.
5. Preserve the existing Accel-Sim trace/frontend path for SASS validation;
   extend the PTX frontend or add a synthetic instruction/event input only when
   the research design requires it.

See `.codex/docs/DECISIONS.md` for rationale and status.

## Main limitations today

The current checkout has documented Hopper SM90/TMA/WGMMA/mbarrier/cluster
support, but this audit has not yet established that it has multi-GPU
execution, NVSwitch, NVLS, or `multimem` timing semantics. README/release-note
claims are not treated as implementation evidence. No H100 or NVSwitch system
is available locally.

## Next task

Define and test the PTX-instruction-to-communication-event schema and its
single-GPU completion contract. Preserve `cais.add.u32` as the local
parser/functional/timing control. Only after the event boundary has focused
tests should it connect to the existing two-endpoint BookSim prototype. Do not
implement Ring, NVSwitch, or NVLS before both boundaries have focused tests.

## Detailed entry points

- Research index: `.codex/docs/README.md`
- Chinese project history: `.codex/docs/项目进展总览.md`
- State/checklist: `.codex/knowledge/project-state.md`
- Local machine: `.codex/knowledge/local-environment.md`
- Accel-Sim trace facts: `.codex/knowledge/trace-format.md`
- Capability matrix: `.codex/docs/03-hardware-capability-matrix.md`
- Decisions/history: `.codex/docs/DECISIONS.md`, `.codex/docs/CHANGELOG.md`
