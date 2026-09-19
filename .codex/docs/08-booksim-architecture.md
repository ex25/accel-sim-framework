# BookSim architecture study

There are two relevant copies. The reference checkout is `../booksim2`
(`28f4329`, `master`). Accel-Sim's nested GPGPU-Sim checkout contains a
BookSim2-derived `src/intersim2` with extra `InterconnectInterface`,
`GPUTrafficManager`, and GPGPU-Sim build integration. It is already used for
the GPU-to-memory network when `-network_mode 1` selects it.

Source-verified core components in the reference checkout are `Network`,
topology implementations, `Router`/`Channel`, `Flit`/`Credit`, VC/buffer,
`TrafficManager`, routing/configuration, `ReadInputs/Evaluate/WriteOutputs`,
and `_Step()` simulation advancement. The standalone run completed on
2026-09-18: `make -C src -j8` passed, and `src/booksim
src/examples/singleconfig` produced latency/throughput statistics and then
ended with the example's convergence/unstable-simulation message. This
validates the standalone executable path only.

The existing GPU adapter already has the desired conceptual surface:
`create_network`, `inject_packet`, `advance_cycle`, `query_completion`, and
`collect_stats` (concretely `CreateInterconnect`, `Push`, `Advance`, `Pop`,
`Busy`, and stats calls). The open question is whether to extend this boundary
for logical GPU endpoints or add a separate orchestrator rather than overload
memory-node assumptions.

## Two-GPU NVLink transport prototype

The first prototype uses the standalone reference checkout, not a copied
source tree. At BookSim `28f4329`, `AnyNet` creates two routers, one endpoint
per router, and one bidirectional router link. Files live under
`tools/booksim_nvlink/`; `bitcomp` with two nodes maps node 0 to node 1 and
node 1 to node 0 deterministically.

The smoke test passed with BookSim's historical success status 255 (its
`main()` returns `-1` when `TrafficManager::Run()` returns true; the wrapper
normalizes it). It verified both node listings, finite packet latency, two
hops, and matching injected/accepted rates. The one-flit, one-cycle-link case
reported 12 network cycles.

The sweep varied link latency 1/4/8 cycles and packet size 1/4/16 flits.
Latency increased monotonically: 12/13.4118/15.2941 cycles for one flit and
27/30.2941/35.9412 cycles for 16 flits. This is a generic transport sanity and
sensitivity result, not physical NVLink calibration. It does not yet provide
an embedded Accel-Sim API, clock coupling, protocol overhead, or bandwidth
mapping.
