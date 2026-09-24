# 已实现功能记录

本文件是唯一的 implementation record。旧的分散阶段文件已合并到这里；以后
新增功能直接追加到对应阶段，不再建立一份孤立的阶段文档。

## P0 — communication event schema

- **Commit**：`55932a3`（主仓库）。
- **功能**：`CommunicationEvent`、`EventCompletion`、`LogicalGPUContext` 和
  `CommunicationEventOrchestrator`；支持 `WAITING → READY → INJECTED → COMPLETED`、
  dependency release、rank/device/stream/kernel、chunk/channel、barrier token 和
  `ready_time_ps`。
- **代码**：`tools/booksim_nvlink/communication_events.py`；测试为
  `test_communication_events.py`。
- **调用路径**：future PTX/SASS issue hook → event → orchestrator → transport
  adapter → completion。P0 没有修改 GPGPU-Sim 或 BookSim。
- **验证**：`python3 tools/booksim_nvlink/test_communication_events.py`（4/4），
  `python3 tools/booksim_nvlink/test_transport_contract.py`（3/3）。
- **Evidence / 限制**：`Verified / unit`；不证明 PTX、NCCL、NVLink、NVSwitch 或
  NVLS 行为。

## P1 — PTX synchronization event producer

- **Commit**：nested GPGPU-Sim `ef16060b`，主仓库文档随 `692531f` 提交。
- **功能**：在 timing-visible warp issue/completion boundary 产生
  `local_barrier_arrive`、`barrier_wakeup`、`memory_release`、`memory_acquire`。
  只有设置 `PTX_SIM_EVENT_FILE` 才写 JSONL；`PTX_SIM_RANK` 指定 logical rank。
- **代码/symbol**：
  `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc` 的
  `shader_core_ctx::issue_warp`、`barrier_set_t::warp_reaches_barrier`、
  `shader_core_ctx::warp_waiting_at_mem_barrier`。
- **时间**：当前 `ready_time_ps` 实际是 simulator cycle，metadata 中标记
  `time_domain=sim_cycle`；不能直接送入要求 global ps 的 P2 bridge。
- **验证**：
  `cmake --build gpu-simulator/build/release -j8`；
  `cmake --install gpu-simulator/build/release`；
  `python3 experiments/ptx_sync_events/run_smoke.py --output-dir /tmp/accelsim-ptx-sync-events-<date>`。
  Smoke 会注入 `membar.cta; bar.sync 0;`，检查 25 个 JSONL event、四种 event
  kind、P0 schema round-trip 和 `status=PASS`。
- **Evidence / 限制**：`Verified / simulator`；不实现 NCCL、BookSim、NVLink、
  NVSwitch、NVLS，也不做真实 H100 validation。运行需要 release runtime、
  `gpgpusim.config` 和 `config_hopper_islip.icnt`。

## P2 — point-to-point event bridge

- **Commit**：`f214b80`。
- **功能**：`COMM_SUBMIT` 的 submit/inject/complete bridge；验证单 source、单
  destination、正 bytes，并拒绝 `time_domain=sim_cycle`。
- **代码**：`tools/booksim_nvlink/point_to_point_bridge.py`；测试
  `test_point_to_point_bridge.py`。
- **验证**：3 个 focused tests；通过 `ReferenceEndpointAdapter` 检查 ready time、
  flit accounting、completion metadata 和状态机。
- **Evidence / 限制**：`Verified / unit`；这是 Python deterministic reference，
  不是 embedded BookSim，也没有两个 GPGPU-Sim process online coupling。

## P3 — NCCL-style Ring reference schedule

- **Commit**：`974f6b0`。
- **功能**：`RingAllReducePlan` 生成 reduce-scatter + all-gather 两阶段；每个
  rank/chunk/step 带 predecessor dependency。N 个 rank、B bytes/rank 产生
  `2*(N-1)*N` 个 `COMM_SUBMIT`，每次传输 `B/N` bytes。
- **代码**：`tools/booksim_nvlink/ring_collective.py`；测试
  `test_ring_collective.py`。
- **验证**：4 个 focused tests；4-rank plan 经 P2 bridge 完成，1,536 bytes、48
  flits；8-rank accounting 通过。
- **Evidence / 限制**：`Synthetic network`；不实现 reduction arithmetic、NCCL
  kernel protocol、channel overlap 或真实 NVLink timing。

## P4 — 8-GPU/4-NVSwitch topology facade

- **Commit**：`2caff9a`。
- **功能**：可配置 GPU→NVSwitch graph；默认 8 logical GPUs、4 logical
  NVSwitches、full attachment。按 `channel_id` 选择 common switch，断开 route
  在 event registration 前失败。
- **代码**：`tools/booksim_nvlink/nvswitch_transport.py`；测试
  `test_nvswitch_transport.py`。
- **验证**：4 个 focused tests；8-rank Ring 通过 wrapper 完成 112 条 route、
  14,336 bytes。
- **Evidence / 限制**：`Synthetic network`；不是已校准的 DGX H100 port topology，
  没有 bandwidth、buffer、credit、arbitration、contention 或 multicast slots。

## P5 — NVLS multicast/reduction reference

- **Commit**：`f160e68`。
- **功能**：`NVLSReducePlan` 为每个 source 生成一次 multicast ingress，所有
  ingress completion dependency 满足后才释放 `COLLECTIVE_COMPLETE`。8-rank、
  B bytes/rank 的 reference ingress 为 `8B`，Ring 为 `14B`。
- **代码**：`tools/booksim_nvlink/nvls_collective.py`；测试
  `test_nvls_collective.py`。
- **验证**：3 个 focused tests；1,024-byte/rank、8-rank engine 完成，并检查
  8,192 ingress bytes、256 flits 和 reduction completion ordering。
- **Evidence / 限制**：`Synthetic network`；`metadata.reduction=abstract`，不
  实现 NVLS arithmetic、replication protocol、SHARP/NVLink、queueing 或 H100
  validation。

## 共同 regression

```bash
python3 tools/booksim_nvlink/test_communication_events.py
python3 tools/booksim_nvlink/test_transport_contract.py
python3 tools/booksim_nvlink/test_point_to_point_bridge.py
python3 tools/booksim_nvlink/test_ring_collective.py
python3 tools/booksim_nvlink/test_nvswitch_transport.py
python3 tools/booksim_nvlink/test_nvls_collective.py
```

当前测试是 focused/unit/synthetic evidence；CTest 没有 registered tests。下一
阶段必须先完成 [roadmap.md](roadmap.md) 的 P6 online event integration，再引入真实
BookSim timing。
