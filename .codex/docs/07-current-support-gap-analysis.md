# Current support gap analysis

| Capability | Current evidence | Status |
|---|---|---|
| Single-GPU SASS trace replay | trace parser/trace-driven code and working release build | implemented/verified build; workload replay needs fresh run |
| Hopper SM90 trace fields | Hopper opcode map, TMA/cluster/mbarrier code | partial source-verified; correlation pending |
| Multi-GPU trace metadata | tracer has per-CUcontext files/counters; no rank/device/fabric fields found | trace/context support only; multi-GPU semantics not established |
| Multi-GPU simulation contexts | no logical GPU/rank orchestration path found | unknown/not established |
| BookSim-derived GPU-to-memory transport | `intersim2`, `icnt_wrapper`, `InterconnectInterface`, `GPUTrafficManager` | implemented for memory path; verified source |
| GPU-to-GPU/NVLink transport | no GPU-rank/peer/fabric path established | unknown/not established |
| NVSwitch topology | no evidence recorded yet | unknown |
| NVLS reduction/multicast | no evidence recorded yet | unknown/not implemented for this project |
| `multimem` semantics | no `multimem` symbol in current source search; PTX requires SM90+ | not implemented/needs explicit support |
| NCCL collective lifecycle | no evidence recorded yet | unknown |

This conservative matrix intentionally does not convert release-note language
into implementation claims. It will be updated with file/function evidence.
