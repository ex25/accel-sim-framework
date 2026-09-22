# H100/Hopper support audit

当前 checkout 的 README/release notes 声称 Accel-Sim 2.0 支持 H100/H200 的
TMA、WGMMA、`mbarrier`、threadblock clusters/DSM、chiplet 或 uGPU memory
partitioning，以及 SM90 configs。源码搜索确认了 Hopper opcode maps、cluster
ID/rank trace fields、TMA multicast metadata，以及 trace-driven layer 中的
mbarrier/WGMMA handling。

源码 audit 已确认：`SM90_H100` 设置 compute capability 9.0 与 Hopper-specific
pipeline/memory parameters；`hopper_opcode.h` 映射 Hopper opcodes；tracer/parser
struct 携带 cluster ID/rank 与 TMA multicast fields；`trace_driven.cc` 包含
TMA、mbarrier、spinloop、cluster 与 GMMA handling。在 Accel-Sim、GPGPU-Sim
和 tracer 源码中未找到 `multimem` symbol。

## P2 source audit 结果

| Feature | Configuration / producer | Consumer path | 当前证据 |
|---|---|---|---|
| SM90/H100 | `SM90_H100/gpgpusim.config`、binary version 90 | `trace_driven.cc:198-200` 选择 `Hopper_OpcodeMap` | source verified；无 H100 replay |
| GMMA | `OP_QGMMA/HGMMA/IGMMA`、Hopper opcode map | `parseSASSInstruction`，`trace_driven.cc:634-673` 的 latency/initiation mapping | partial model；需要 H100 trace |
| TMA | `UTMALDG/UTMASTG/UTMAREDG/UBLK*` 与 TMA trace metadata | `trace_parser.cc:227-275`、`trace_driven.cc:345-368,560-605` | partial model；TMA prefetch prints 未实现 |
| mbarrier/SYNCS | register-value trace 与 address metadata | `trace_driven.cc:722-858`、shader TMA/mbarrier state | partial model；unsupported variants 会 abort 或 warn |
| cluster/DSM fields | NVBit record 中的 CTA/cluster ID 与 cluster rank | `TracezReader`、`trace_warp_inst_t`、shader state | fields 存在；不是 NCCL rank |
| `multimem` | 未找到 producer/parser/opcode symbol | 无 consumer path | 未实现 |
| NVLink/NVSwitch/NVLS | 未找到 GPU↔GPU event 或 fabric path | 无 consumer path | 未实现 |

当前代码明确存在 source-level limitations：TMA prefetch 被当作 no-op
（`trace_driven.cc:585-589`）；GMMA model 拒绝未知 tile size（`:651-663`）；
cluster fence handling 仍标记为 TODO（`:715-719`）；未知 opcode string 会在
`parse_from_trace_struct()`（`:298-313`）中触发 assert。因此，在声称完整
Hopper support 前必须取得真实 H100 trace fixture。

面向读者的完整解释见 `.codex/docs/15-concepts-for-readers.md`。结论保持保守：
SM90 support 不代表 multi-GPU NVLink/NVSwitch/NVLS support；当前没有
`multimem` path 也不妨碍未来实现。
