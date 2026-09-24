# 015 — P4 NVSwitch topology and transport facade

P4 adds a configurable GPU-to-NVSwitch graph and wraps the P2 endpoint bridge
with deterministic route selection. The default reference shape is 8 logical
GPUs and 4 logical NVSwitches, with every GPU attached to every switch. The
shape is configuration, not a claim that the local simulator has measured the
DGX H100 physical port topology.

Each point-to-point event is route-checked before registration. A route is
`gpu → nvswitch → gpu`, and `channel_id` selects a common switch modulo the
number of common switches. Custom disconnected graphs fail explicitly instead
of silently falling back to the GPU-memory `intersim2` path.

**Verified**: 4 focused tests cover default 8×4 connectivity, custom
disconnection, pre-registration route rejection, and an 8-rank Ring completed
through the wrapper with 112 routed transfers and 14,336 bytes.

**限制**：link bandwidth, arbitration, switch queueing, port-level topology,
multicast, and NVLS reduction are not modeled. P4 remains a deterministic
topology facade over the reference endpoint; BookSim integration and calibrated
DGX H100 parameters are still pending.
