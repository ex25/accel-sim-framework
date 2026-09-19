# Current support gap analysis

| Capability | Current evidence | Status |
|---|---|---|
| Single-GPU SASS trace replay | trace parser/trace-driven code and working release build | implemented/verified build; workload replay needs fresh run |
| Hopper SM90 trace fields | Hopper opcode map, TMA/cluster/mbarrier code | partial source-verified; H100 correlation pending |
| Multi-GPU trace metadata | tracer has per-CUcontext files/counters; no rank/device/fabric fields found | trace/context support only; multi-GPU semantics not established |
| Multi-GPU simulation contexts | no logical GPU/rank orchestration path found | unknown/not established |
| BookSim-derived GPU-to-memory transport | `intersim2`, `icnt_wrapper`, `InterconnectInterface`, `GPUTrafficManager` | implemented for memory path; verified source |
| GPU-to-GPU/NVLink transport | only NVLink monitoring helpers/metrics; no GPU-rank/peer/fabric timing path | not established |
| NVSwitch topology | no simulator topology/queue implementation; NVIDIA FM docs describe hardware fabric requirements | unknown/not implemented |
| NVLS reduction/multicast | NCCL primary-source setup reviewed; no project semantic model | source-understood, simulator not implemented |
| `multimem` semantics | no `multimem` symbol in current source search; PTX requires SM90+ | not implemented/needs explicit support |
| NCCL collective lifecycle | NCCL `nvls.cc` setup reviewed; no project event schema/orchestrator | source-understood, simulator not implemented |

The conservative `unknown/not implemented` labels mean the project has not
verified a usable simulator path. NVIDIA's current Fabric Manager documentation
states that H100-and-later DGX/HGX NVSwitch systems support multicast and
reduction, with finite multicast resources; that is hardware capability
evidence, not Accel-Sim implementation evidence.

This conservative matrix intentionally does not convert release-note language
into implementation claims. It will be updated with file/function evidence.
