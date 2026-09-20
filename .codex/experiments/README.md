# Experiment metadata index

Keep experiment specifications and compact metadata here. Raw traces, logs,
checkpoints, and large result tables belong in gitignored run directories. Each
metadata file must pin Accel-Sim and BookSim SHAs, configuration, workload,
rank/GPU count, collective/algorithm, message sizes, seed, commands, output
path, date, and hardware/model evidence level.

Completed entries:

- `2026-09-19-a800-nvbit-booksim-nvlink2.md`: real A800 vectoradd trace,
  post-processing/replay, and standalone BookSim two-endpoint smoke/sweep.
- `2026-09-20-sass-ptx-a800-gap.md`: single-A800 dual-code-object matrix
  comparing SASS trace replay and PTX execution-driven replay, including
  bounded GEMV/GEMM sizes and the CUDA 12.8 lineinfo parser failure.
- `2026-09-20-sm90-ptx-baseline.md`: CUDA 12.8 `compute_90` PTX-only kernel
  accepted by the unmodified `H100-PTX`/`SM90_H100` execution-driven path.
