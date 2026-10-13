# Hopper PTX Research

This repository is the Accel-Sim side of the paired `research/hopper-ptx`
branches. Read these files before changing code:

1. `.codex/README.md`
2. `.codex/rules/hopper-ptx.md`
3. `.codex/plans/single-card-dense.md` (current execution order and gates)
4. The current stage file under `.codex/docs/implementation/`

The current entry point is stage 3. Complete and validate stages in order:
3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14.
Do not skip reopened stages or treat historical PASS summaries as acceptance.
The next implementation session is scoped to stage 3; report its results and
stage 4 readiness before starting further implementation.
Each stage must satisfy its current component-level acceptance contract with
commands, results, and durable evidence before advancing. Full Dense, Python
block integration, H100 calibration, and original FA3 integration have explicit
owners in stages 11–15; they must not create circular prerequisites for 3–9.
Minimal specification/PTX-export preparation may be recorded within the current
stage, but does not mark any later stage complete. See `single-card-dense.md`.
Keep the inner
`gpu-simulator/gpgpu-sim` repository on its paired branch and record both
repository commits in every implementation-stage report.
