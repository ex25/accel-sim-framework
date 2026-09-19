# Hardware validation boundary

The local A800 is an Ampere software-pipeline instrument. It can validate
CUDA/NVBit instrumentation, trace generation/post-processing, parser replay,
builds, and automation. It cannot validate SM90 instructions (WGMMA/TMA/
mbarrier/cluster semantics), NVLink 4, H100 NVSwitch, multicast memory,
`multimem`, or NVLS performance.

An H100/H200 can validate Hopper trace production and consumption. A single
H100 still cannot validate multi-GPU routing, NVSwitch contention, or NVLS
collectives. Those claims require the stated number of GPUs and topology; a
BookSim transport model or copied single-GPU trace is not a hardware result.

For each hardware run record GPU model/count, compute capability, driver,
CUDA, NCCL, Fabric Manager, topology (`nvidia-smi topo -m`), NVLink/NVSwitch
state, benchmark, environment variables, command, and output path. State what
the run proves and what it does not prove. Never present A800/Ampere results as
H100/Hopper results, synthetic NVLS as real NVLS execution, or a transport-only
model as a real NVSwitch implementation.
