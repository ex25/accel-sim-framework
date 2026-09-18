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
