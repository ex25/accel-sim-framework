# BookSim and interconnect policy

BookSim is a generic interconnection-network timing engine, not an NVIDIA
NVSwitch implementation. Keep the standalone BookSim repository and the
existing Accel-Sim `gpu-simulator/gpgpu-sim/src/intersim2` code clearly
identified; the latter already models a GPU-memory interconnect and is not
evidence that GPU-GPU NVLink/NVSwitch exists.

Pin and record the inspected upstream SHA. Prefer a submodule or external
checkout plus a narrow adapter; do not copy/vendored-paste BookSim before a
decision in `DECISIONS.md`. Any local patch must be separately named and
upstream-trackable.

When studying or embedding BookSim, locate the configuration, topology/network,
router, channel, virtual channel, flit/packet, routing, traffic manager,
simulation step, injection/ejection, and statistics code. Expose an adapter
with explicit operations such as create, inject, advance, query completion,
and collect stats rather than letting GPU code call internal globals.

Keep these layers separate and test them separately:

1. endpoint/collective semantics (rank, message, dependency, completion);
2. NVSwitch topology/protocol (ports, bandwidth, latency, routing, multicast);
3. NVLS reduction semantics (group, arithmetic, reduction completion);
4. BookSim transport timing (flits, buffers, VCs, contention, cycles).

Never equate “BookSim integrated” with “NVSwitch implemented,” or “NVSwitch
implemented” with “NVLS implemented.” Document the Accel-Sim/BookSim clock
conversion; never silently assume one GPU cycle equals one network cycle.
