# Accel-Sim + BookSim NVLS research memory

## Project goal

Extend Accel-Sim with a research-grade GPU interconnect model backed by
BookSim, then study single-node H100/H200 NVLink + NVSwitch + NVLS collective
behaviour. The intended first target is 2/4/8-GPU AllReduce, comparing Ring and
NVLS. This is a staged research project, not a claim that the current checkout
already models NVSwitch or NVLS.

## Current state

- Main repository: `accel-sim-framework`, branch
  `research/nvls-accelsim-study`, base `d930ad6`.
- Working tree was clean before this documentation phase.
- Nested GPGPU-Sim is a separate ignored Git checkout at `91880c53`; it is not
  a tracked submodule in the Accel-Sim repository.
- BookSim is a separate checkout at `../booksim2`, `28f4329`; it is not yet a
  dependency or source change in this branch.
- The release build already exists and rebuilt successfully with the activated
  `in-switch` environment. CTest currently discovers no tests.
- One physical NVIDIA A800 80GB PCIe is present. It is Ampere/sm_80 in the
  current hardware record, and has no NVLink peer on this host.
- NVBit SDK/tracer artifacts were not found during this audit; real A800 trace
  generation is therefore pending environment preparation and GPU availability.

## How a new session starts

Read this file, `.codex/rules/`, and
`.codex/knowledge/project-state.md`. Check Git status for both the main and
nested repositories. For implementation work, read the relevant design note
and skill before editing. Never infer H100/NVSwitch/NVLS validation from A800
or synthetic BookSim results.

## Directory guide

- `rules/`: persistent policy and evidence labels.
- `skills/`: reusable project workflows. `.agents/skills/` contains symlinks so
  Codex's repository skill discovery can find the canonical `.codex/skills/`
  content without duplicating it.
- `knowledge/`: compact, high-signal current facts.
- `docs/`: human-readable architecture, research, decisions, and changelog.
- `experiments/`: experiment specifications and reproducibility metadata only;
  raw traces and large outputs stay outside `.codex/` and are gitignored.

## Current milestone

M0 repository/context infrastructure plus M1 baseline audit. The next exit
criteria are: capture a reproducible A800 trace smoke test if the GPU can be
reserved, map one real trace instruction through source code, and complete the
current-code capability audit before any simulator-core feature work.

## Important decisions

1. Keep compute traces and future communication events conceptually separate;
   do not replicate one single-GPU trace and call it a multi-GPU execution.
2. Treat BookSim as a generic network timing backend. NVSwitch topology,
   multicast, reduction, and NVLS completion semantics remain a separate layer.
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

Finish source-grounded audits of trace format, Hopper support, multi-GPU/NVLink
search results, and the BookSim architecture. Then record a minimal adapter
design and a synthetic transport validation plan. Do not implement NVLS first.

## Detailed entry points

- Research index: `.codex/docs/README.md`
- State/checklist: `.codex/knowledge/project-state.md`
- Local machine: `.codex/knowledge/local-environment.md`
- Accel-Sim trace facts: `.codex/knowledge/trace-format.md`
- Capability matrix: `.codex/docs/03-hardware-capability-matrix.md`
- Decisions/history: `.codex/docs/DECISIONS.md`, `.codex/docs/CHANGELOG.md`
