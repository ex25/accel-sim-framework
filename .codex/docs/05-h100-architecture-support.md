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

## P2 source audit result

| Feature | Configuration / producer | Consumer path | Current evidence |
|---|---|---|---|
| SM90/H100 | `SM90_H100/gpgpusim.config`, binary version 90 | `trace_driven.cc:198-200` selects `Hopper_OpcodeMap` | source verified; no H100 replay |
| GMMA | `OP_QGMMA/HGMMA/IGMMA`, Hopper opcode map | `parseSASSInstruction`, latency/initiation mapping at `trace_driven.cc:634-673` | partial model; H100 trace needed |
| TMA | `UTMALDG/UTMASTG/UTMAREDG/UBLK*` and TMA trace metadata | `trace_parser.cc:227-275`, `trace_driven.cc:345-368,560-605` | partial model; TMA prefetch prints not implemented |
| mbarrier/SYNCS | register-value trace plus address metadata | `trace_driven.cc:722-858`, shader TMA/mbarrier state | partial model; unsupported variants abort or warn |
| cluster/DSM fields | CTA/cluster IDs and cluster rank in NVBit record | `TracezReader`, `trace_warp_inst_t`, shader state | fields are present; this is not NCCL rank |
| `multimem` | no producer/parser/opcode symbol found | no consumer path | not implemented |
| NVLink/NVSwitch/NVLS | no GPU↔GPU event or fabric path found | no consumer path | not implemented |

Important source-level limitations are explicit in the current code: TMA
prefetch is treated as a no-op (`trace_driven.cc:585-589`), the GMMA model rejects
unknown tile sizes (`:651-663`), cluster fence handling is still marked TODO
(`:715-719`), and unknown opcode strings assert in
`parse_from_trace_struct()` (`:298-313`). These are reasons to require a real
H100 trace fixture before claiming complete Hopper support.

The complete reader-facing explanation is in
`.codex/docs/15-concepts-for-readers.md`. The audit remains conservative:
SM90 support does not imply multi-GPU NVLink/NVSwitch/NVLS support, and absence
of a current `multimem` path does not prevent a future implementation.
