# Project overview

The research target is a layered simulator: Accel-Sim models GPU compute and
memory timing; a future multi-GPU orchestrator models rank/stream/dependency and
collective lifecycle; an NVLink/NVSwitch semantic layer models endpoints,
topology, multicast, and reduction; BookSim supplies generic packet/flit,
router, channel, contention, and network-clock timing.

The first useful result is not a full H100 emulator. It is a reproducible
single-GPU baseline, an evidence-backed gap analysis, and a minimal logical
GPU0→GPU1 transport experiment. Ring AllReduce comes before NVLS. NVLS comes
only after transport, completion ordering, and reduction semantics have focused
tests.
