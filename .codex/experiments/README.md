# Experiment metadata index

Keep experiment specifications and compact metadata here. Raw traces, logs,
checkpoints, and large result tables belong in gitignored run directories. Each
metadata file must pin Accel-Sim and BookSim SHAs, configuration, workload,
rank/GPU count, collective/algorithm, message sizes, seed, commands, output
path, date, and hardware/model evidence level.

The first planned entry is a tiny A800 trace smoke test, currently blocked by
NVBit artifact availability and GPU occupancy.
