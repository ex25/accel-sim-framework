# Accel-Sim ↔ BookSim integration

GPU memory path 已有 Accel-Sim↔BookSim adapter，因此 multi-GPU communication
的候选研究边界为：

```mermaid
flowchart TD
  A[Accel-Sim compute contexts] --> B[Multi-GPU orchestrator]
  B --> C[Collective/NVSwitch semantic layer]
  C --> D[BookSim transport timing]
```

可复用的 lower-level interface 是
`InterconnectInterface::CreateInterconnect/Push/Pop/Advance/Busy`, backed by
`GPUTrafficManager` 和 `Network`。它目前映射 shader/memory device ID，并将
`mem_fetch` type 转成 flit type；不提供 rank、collective、peer GPU、multicast
或 switch-side reduction semantics。

当前 GPU cycle coupling 已通过 source verification：`gpgpu_sim::cycle()` 使用
`next_clock_domain()`, invokes cluster `icnt_cycle()` on the ICNT mask, moves
L2 requests/replies with `icnt_pop`, and calls `icnt_transfer()` on the ICNT
mask. The Hopper config declares separate core/interconnect/L2/DRAM
frequencies。这是 clock-domain hook，还不是 multi-GPU global-time contract。

未来 collective layer 与 Accel-Sim 的 clock coupling 目前只是 design
decision，不是 implementation claim。adapter 应暴露精简 transport API：

```text
create_network(config, endpoint_count)
inject_message(src, dst, bytes, tag, ready_time_ps) -> injection_status
advance_to(time_ps)
poll_completed() -> {tag, src, dst, bytes, completion_time_ps}
read_stats() -> transport_stats
```

adapter 负责 packetization（`ceil((bytes + protocol_overhead) / flit_bytes)`）、
endpoint tag，以及 tail-flit ejection 时的 completion。BookSim 只负责
topology、routing、virtual channel、buffering、arbitration 和 transport
latency。Rank/collective lifecycle、multicast group、reduction operand 与
NVSwitch/NVLS semantics 位于该边界之上。

推荐 clock contract 采用 event-driven 模式，并用整数 picoseconds 表示 global
time。当 GPU 与 network period 以 rational value 表示时，`advance_to(t)` 在
下一个 network edge `<= t` 时推进 BookSim；rational accumulator 可避免
floating-point drift。GPU callback 在自身 GPU-domain timestamp 入队
communication event，并在对应 global time 消费 completion。该设计支持不同
GPU/interconnect clock，并保持相同 timestamp 下的 ordering。lockstep 或
fixed batch 只能在复用同一 conversion contract 时作为优化。绝不能默认把
一个 Accel-Sim cycle 当成一个 BookSim cycle。

外部 BookSim `TrafficManager` 当前暴露 `Run()`，但仍将 per-cycle `_Step`、
injection 和 completion queue 设为 protected。因此 packet-granular embedded
adapter 需要一个小型 upstream-tracked facade（或暴露这些操作的 subclass）；
调用 standalone executable 不能替代 clock-coupled injection。本阶段只记录
该边界，不修改 BookSim 或 Accel-Sim core code。
