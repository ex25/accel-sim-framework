# Current support gap analysis

| Capability | Current evidence | Status |
|---|---|---|
| Single-GPU SASS trace replay | trace parser/trace-driven code and working release build | implemented/verified build; workload replay needs fresh run |
| Hopper SM90 trace fields | Hopper opcode map, TMA/cluster/mbarrier code | partial source-verified; correlation pending |
| Multi-GPU trace metadata | not established by this audit | unknown |
| Multi-GPU simulation contexts | no source-verified path yet | unknown/not established |
| NVLink transport | README/release-note claims require code-path audit | unknown |
| NVSwitch topology | no evidence recorded yet | unknown |
| NVLS reduction/multicast | no evidence recorded yet | unknown/not implemented for this project |
| `multimem` semantics | no evidence recorded yet | unknown |
| NCCL collective lifecycle | no evidence recorded yet | unknown |

This conservative matrix intentionally does not convert release-note language
into implementation claims. It will be updated with file/function evidence.
