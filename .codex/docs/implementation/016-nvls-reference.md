# 016 — P5 NVLS multicast/reduction completion reference

P5 adds `NVLSReducePlan` and `NVLSReferenceEngine` on top of the P0 event
orchestrator. Each source emits one `multicast` ingress event to a selected
NVSwitch endpoint with the full `multicast_group`; the abstract switch
reduction event is released only after every ingress completion dependency is
satisfied.

For eight ranks and `B` bytes per rank this reference transfers `8B` ingress
bytes, compared with the Ring reference's `14B` directed transfers. The
reduction event is a completion boundary, not arithmetic: `metadata.reduction`
is explicitly `abstract`.

**Verified**: 3 focused tests cover 8-rank ingress/reduction completion, Ring
versus NVLS byte accounting, and rejection of an unattached rank. An 8-rank,
1,024-byte plan completes through the deterministic endpoint after all eight
ingress transfers.

**限制**：没有实现 NVLS hardware multicast replication, in-switch reduction
arithmetic, SHARP/NVLink protocol, switch queueing, or H100 validation. P5
therefore demonstrates the event/dependency boundary needed for later hardware
calibration, not NVLS performance accuracy.
