# 007 — NVBit A800 trace 与 BookSim two-GPU transport prototype

## 问题与设计

当时尚未在本地 A800 上运行 software trace pipeline，也需要在加入 NVSwitch/
NVLS 前完成最小 network experiment。

NVBit 1.8 安装在 `in-switch` conda environment；tracer Makefile 接受显式
`NVBIT_PATH`，不复制 SDK，也不依赖机器专用 symlink。Workload 使用 SDK
vectoradd，随后 post-process 并用 Accel-Sim release binary replay。

Network 部分将 BookSim 固定在 SHA `28f4329`，不 fork source。使用 AnyNet 与
TrafficManager，把 node 0/1 作为两个 logical GPU，每个 endpoint 一个 router，
中间一条 bidirectional router link；使用 `bitcomp` 产生 deterministic 双向 traffic。

## Alternatives

- Copy BookSim 到 Accel-Sim：拒绝，容易遮蔽 provenance。
- 立即扩展 Accel-Sim `intersim2`：推迟，先独立设计 endpoint/clock API。
- Uniform traffic：拒绝，无法保证双向 traffic。

## 修改文件

- `util/tracer_nvbit/tracer_tool/Makefile`：可覆盖 `NVBIT_PATH`。
- `tools/booksim_nvlink/nvlink2_config`：two-endpoint configuration。
- `tools/booksim_nvlink/nvlink2_network`：AnyNet topology。
- `tools/booksim_nvlink/run_nvlink2.sh`：smoke test/status normalization。
- `tools/booksim_nvlink/sweep_nvlink2.sh`：parameter sweep 与 CSV。
- `tools/booksim_nvlink/README.md`：reproduction 与 interpretation。
- `.codex` trace/BookSim/environment/state documents：evidence 与限制。

## Call path

```text
vectoradd → NVBit tracer_tool.so → raw trace.xz
  → post-traces-processing → tracez/kernelslist.g → accel-sim.out
  → trace_parser → shader/memory timing → statistics

run_nvlink2.sh → BookSim main → AnyNet → TrafficManager::Run
  → TrafficManager::_Step → Network ReadInputs/Evaluate/WriteOutputs
  → packet/flit statistics
```

## 结果

- A800 vectoradd 与 post-processing：pass。
- Accel-Sim replay：exit 0、6,824 cycles、75,000 request/reply packets。
- BookSim smoke：`NVLINK2_SMOKE_PASS`；one-flit/link-latency-1 为 12 cycles、
  two hops。
- BookSim sweep：`NVLINK2_SWEEP_PASS`；9 个 case 全部写出 CSV。
- Implementation commit：`dc7ffa9`。

## 限制与 evidence

- **Verified**：A800 software trace pipeline 与 generic BookSim transport。
- **Inferred**：two-router topology 是有用的初始 endpoint model。
- **Needs hardware validation**：physical NVLink latency/bandwidth、H100
  topology、NVSwitch routing、multicast、NCCL 与 NVLS。BookSim status 255 是
  wrapper normalize 的 legacy success convention。

下一步是 autopsy 一条真实 trace instruction，再定义 endpoint adapter 与 clock
conversion，之后才能将 BookSim transport 接入 Accel-Sim。
