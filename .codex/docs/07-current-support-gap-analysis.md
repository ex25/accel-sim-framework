# 当前 support gap analysis

| Capability | 当前证据 | Status |
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

保守使用 `unknown/not implemented` 表示项目尚未验证可用 simulator path。
NVIDIA 当前 Fabric Manager 文档说明 H100 及更新的 DGX/HGX NVSwitch systems
支持 multicast 与 reduction，且 multicast resources 有限；这是 hardware
capability evidence，不是 Accel-Sim implementation evidence。

该矩阵不会把 release-note 描述直接转成 implementation claim，后续只根据
明确的 file/function evidence 更新。
