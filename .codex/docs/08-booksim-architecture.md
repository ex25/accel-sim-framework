# BookSim architecture 研究

当前有两个相关版本。reference checkout 是 `../booksim2`
(`28f4329`, `master`). Accel-Sim's nested GPGPU-Sim checkout contains a
BookSim2-derived `src/intersim2` with extra `InterconnectInterface`,
`GPUTrafficManager`, and GPGPU-Sim build integration. It is already used for
the GPU-to-memory network when `-network_mode 1` selects it.

reference checkout 中经 source verification 的 core components 包括 `Network`、
topology implementations, `Router`/`Channel`, `Flit`/`Credit`, VC/buffer,
`TrafficManager`, routing/configuration, `ReadInputs/Evaluate/WriteOutputs`,
and `_Step()` simulation advancement. The standalone run completed on
2026-09-18: `make -C src -j8` passed, and `src/booksim
src/examples/singleconfig` produced latency/throughput statistics and then
ended with the example's convergence/unstable-simulation message. This
validates the standalone executable path only.

现有 GPU adapter 已提供所需的概念接口：
`create_network`, `inject_packet`, `advance_cycle`, `query_completion`, and
`collect_stats` (concretely `CreateInterconnect`, `Push`, `Advance`, `Pop`,
`Busy` 和 stats calls）。待决定的问题是：为 logical GPU endpoint 扩展该边界，
还是新增独立 orchestrator，以免复用并扭曲 memory-node assumptions。

## Two-GPU NVLink transport prototype

首个 prototype 使用 standalone reference checkout，不复制 source tree。在
BookSim `28f4329` 中，`AnyNet` 创建两个 router，每个 router 一个 endpoint，
并用一条 bidirectional router link 连接。文件位于
`tools/booksim_nvlink/`; `bitcomp` with two nodes maps node 0 to node 1 and
node 1 to node 0 deterministically.

smoke test 以 BookSim 历史上的 success status 255 通过（其
`main()` returns `-1` when `TrafficManager::Run()` returns true; the wrapper
会被 wrapper normalize）。测试确认了两个 node listing、有限 packet latency、
two hops，以及一致的 injected/accepted rates。one-flit、one-cycle-link case
为 12 network cycles。

该 sweep 改变 link latency（1/4/8 cycles）和 packet size（1/4/16 flits）。
Latency increased monotonically: 12/13.4118/15.2941 cycles for one flit and
27/30.2941/35.9412 cycles for 16 flits. This is a generic transport sanity and
sensitivity result, not physical NVLink calibration. It does not yet provide
an embedded Accel-Sim API, clock coupling, protocol overhead, or bandwidth
mapping.
