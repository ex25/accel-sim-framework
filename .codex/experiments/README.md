# Experiment metadata index

Keep experiment specifications and compact metadata here. Raw traces, logs,
checkpoints, and large result tables belong in gitignored run directories. Each
metadata file must pin Accel-Sim and BookSim SHAs, configuration, workload,
rank/GPU count, collective/algorithm, message sizes, seed, commands, output
path, date, and hardware/model evidence level.

Completed entries:

- `2026-09-19-a800-nvbit-booksim-nvlink2.md`: real A800 vectoradd trace,
  post-processing/replay, and standalone BookSim two-endpoint smoke/sweep.
