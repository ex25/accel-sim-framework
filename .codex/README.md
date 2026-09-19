# Accel-Sim + BookSim NVLS research memory

## Project goal

Extend Accel-Sim with a research-grade GPU interconnect model backed by
BookSim, then study single-node H100/H200 NVLink + NVSwitch + NVLS collective
behaviour. The intended first target is 2/4/8-GPU AllReduce, comparing Ring and
NVLS. This is a staged research project, not a claim that the current checkout
already models NVSwitch or NVLS.

## Current state

- Main repository: `accel-sim-framework`, branch
  `feat/nvbit-a800-booksim-nvlink`, base research state `85cccb1`.
- Working tree was clean before this documentation phase.
- Nested GPGPU-Sim is a separate ignored Git checkout at `91880c53`; it is not
  a tracked submodule in the Accel-Sim repository.
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
  BookSim analysis, feature implementation, documentation, experiments, and
  counter addition.
- `knowledge/`: compact, high-signal current facts.
- `docs/`: human-readable architecture, research, decisions, and changelog.
- `experiments/`: experiment specifications and reproducibility metadata only;
  raw traces and large outputs stay outside `.codex/` and are gitignored.

## Current milestone

M2 A800 trace pipeline is complete and the first standalone two-endpoint
BookSim transport prototype passes. The next exit criteria are a real trace
instruction autopsy plus a documented endpoint/clock adapter design before any
simulator-core integration.

## Important decisions

1. Keep compute traces and future communication events conceptually separate;
   do not replicate one single-GPU trace and call it a multi-GPU execution.
2. Reuse/analyse the existing BookSim-derived `intersim2` boundary before
   introducing another copy. Treat it as a generic network timing backend;
   NVSwitch topology, multicast, reduction, and NVLS completion semantics
   remain a separate layer.
3. Preserve the existing Accel-Sim trace/frontend path until evidence shows
   that a new event format is necessary.

See `.codex/docs/DECISIONS.md` for rationale and status.

## Main limitations today

The current checkout has documented Hopper SM90/TMA/WGMMA/mbarrier/cluster
support, but this audit has not yet established that it has multi-GPU
execution, NVSwitch, NVLS, or `multimem` timing semantics. README/release-note
claims are not treated as implementation evidence. No H100 or NVSwitch system
is available locally.

## Next task

Autopsy one instruction from the real A800 trace and design an explicit
BookSim endpoint/clock adapter with analytical byte/flit checks. Do not
implement Ring, NVSwitch, or NVLS before that boundary is validated.

## Detailed entry points

- Research index: `.codex/docs/README.md`
- State/checklist: `.codex/knowledge/project-state.md`
- Local machine: `.codex/knowledge/local-environment.md`
- Accel-Sim trace facts: `.codex/knowledge/trace-format.md`
- Capability matrix: `.codex/docs/03-hardware-capability-matrix.md`
- Decisions/history: `.codex/docs/DECISIONS.md`, `.codex/docs/CHANGELOG.md`
