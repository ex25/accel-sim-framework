# P2–P5 reference collective checks

## Objective / hypothesis

- **Objective**：验证 P2 point-to-point、P3 Ring、P4 8-GPU/4-NVSwitch facade 和
  P5 NVLS multicast/reduction completion 可以在同一 deterministic reference
  layer 中串联。
- **Hypothesis**：8-rank Ring 使用 `2*(8-1)=14` directed rounds of rank-sized
  traffic，即 `14B`；NVLS reference 每个 source 只产生一次 switch ingress，即
  `8B`，并且 reduction completion 必须等待全部 ingress completion。
- **Evidence level**：`Synthetic network` / `unit`，不是 real NVSwitch/NVLS。

## Fixed inputs

- Main repository：branch `feat/ptx-sync-events-p0-p5`，commit `f160e68`。
- Topology：`NVSwitchTopology()`，8 logical GPUs、4 logical NVSwitches、每个
  GPU attached to all four switches。
- Network clock：`ClockDomain.from_frequency_mhz(1000)`。
- Message：8 ranks，`1024 bytes/rank`，32-byte flit，fixed latency 0。
- Code：
  `tools/booksim_nvlink/point_to_point_bridge.py`、`ring_collective.py`、
  `nvswitch_transport.py`、`nvls_collective.py`。

## Reproduction

```bash
python3 experiments/event_collectives_reference/run_reference.py
```

脚本位置：`experiments/event_collectives_reference/run_reference.py`。它先让
Ring 通过 P4 wrapper 完成，再让 NVLS engine 完成，并检查 total bytes/flits 和
collective completion。focused unit tests 可分别运行：

```bash
python3 tools/booksim_nvlink/test_point_to_point_bridge.py
python3 tools/booksim_nvlink/test_ring_collective.py
python3 tools/booksim_nvlink/test_nvswitch_transport.py
python3 tools/booksim_nvlink/test_nvls_collective.py
```

## Result / acceptance

实际输出：

```text
REFERENCE_COLLECTIVES_PASS ring_bytes=14336 ring_flits=448 nvls_ingress_bytes=8192 nvls_ingress_flits=256 topology=8gpu-4nvswitch
```

Acceptance criteria：Ring/NVLS 都完成；Ring bytes=`14*1024`；NVLS ingress
bytes=`8*1024`；所有 NVLS ingress 完成后才进入 `COLLECTIVE_COMPLETE`；断开
topology 必须在 event registration 前失败。

## Interpretation / limitations

- **Verified**：event dependency、route validation、bytes/flits accounting 和
  completion ordering。
- **Synthetic**：8×4 topology、1 GHz network clock、32-byte flit 和 latency 0
  都是 reference parameters。
- **未实现**：BookSim per-cycle timing、port/buffer/credit/arbitration、NCCL
  kernel protocol、NVLS arithmetic/replication、真实 H100 calibration。
