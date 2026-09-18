# H100/Hopper support audit

The checkout README/release notes claim Accel-Sim 2.0 support for H100/H200
features including TMA, WGMMA, `mbarrier`, threadblock clusters/DSM, chiplet or
uGPU memory partitioning, and SM90 configs. Source search confirms Hopper opcode
maps, trace fields for cluster IDs/rank, TMA multicast metadata, and mbarrier/
WGMMA handling in the trace-driven layer.

This is not yet a complete validation. The audit must map each feature to
configuration, producer trace requirement, parser field, simulator consumer,
tests, and evidence level. In particular, SM90 support does not imply multi-GPU
NVLink/NVSwitch/NVLS support.
