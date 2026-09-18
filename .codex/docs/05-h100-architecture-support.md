# H100/Hopper support audit

The checkout README/release notes claim Accel-Sim 2.0 support for H100/H200
features including TMA, WGMMA, `mbarrier`, threadblock clusters/DSM, chiplet or
uGPU memory partitioning, and SM90 configs. Source search confirms Hopper opcode
maps, trace fields for cluster IDs/rank, TMA multicast metadata, and mbarrier/
WGMMA handling in the trace-driven layer.

The source audit currently confirms: `SM90_H100` sets compute capability 9.0
and Hopper-specific pipeline/memory parameters; `hopper_opcode.h` maps Hopper
opcodes; tracer/parser structs carry cluster IDs/rank and TMA multicast fields;
and `trace_driven.cc` contains TMA, mbarrier, spinloop, cluster, and GMMA
handling. No `multimem` symbol was found in the Accel-Sim/GPGPU-Sim/tracer
source search.

This is not yet a complete validation. The audit must map each feature to
configuration, producer trace requirement, parser field, simulator consumer,
tests, and evidence level. In particular, SM90 support does not imply multi-GPU
NVLink/NVSwitch/NVLS support; absence of a current `multimem` path is not a
claim that a future parser could never add one.
