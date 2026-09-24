# 014 — P3 NCCL-style Ring all-reduce schedule

P3 adds a deterministic Ring reference scheduler over the P2 bridge. A plan
with `N` ranks and `B` bytes per rank emits `2*(N-1)*N` directed
`COMM_SUBMIT` events: `N-1` reduce-scatter steps followed by `N-1`
all-gather steps. Each transfer is `B/N` bytes and carries collective, chunk,
channel, phase, step and dependency metadata.

For every rank, step `s>0` depends on its predecessor's step `s-1`; the first
all-gather step depends on the final reduce-scatter step. The P0 orchestrator
therefore exposes only the current Ring frontier to P2 transport.

**Verified**: 4 focused tests cover four-rank completion, chunk/dependency
mapping, invalid plans, and eight-rank message accounting. The four-rank plan
completes through the P2 reference bridge with 1,536 transferred bytes and 48
32-byte flits.

**限制**：没有实现 reduction arithmetic、NCCL kernel protocol、channel
parallelism、overlap policy 或真实 NVLink timing；P3 是 collective dependency
reference，后续 P4 才会替换 transport/topology semantics。
