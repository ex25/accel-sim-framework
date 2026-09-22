# 008 — Trace autopsy 与 BookSim endpoint/clock contract

## 问题与设计

A800 trace 与 standalone BookSim prototype 已通过，但还缺少两类证据：一条
真实 instruction 如何被 Accel-Sim 消费的 source-level proof，以及 future
timestamped GPU-to-GPU transport 的精确边界。Ring 工作前还需要可重复的
BookSim analytical sanity check。

Trace autopsy 使用 2026-09-19 A800 vectoradd 中真实 `LDG.E.64` record，跟踪
NVBit emission、text/address compression、`.tracez` parsing、warp instruction
construction、memory coalescing、GPU memory interconnect injection 与 completion
gate。完整 call graph 与 field decoding 在 `.codex/docs/02-trace-pipeline.md`。

Future communication boundary 定义为 transport-only endpoint adapter：
`create_network`、`inject_message`、`advance_to`、`poll_completed`、`read_stats`。
adapter 把 bytes packetize 为 flits，并以 tail-flit delivery 表示 completion；
collective/rank/multicast/reduction semantics 位于 BookSim 之上。Clock contract
使用 integer picosecond global clock 与 rational GPU/network period conversion。
这只是 design artifact，还不是 Accel-Sim hook。

## Alternatives

- `TrafficManager::Run()` 无法提供 per-event injection/cycle advancement，因为
  相关 BookSim method 是 protected。
- 将一个 GPU cycle 等同为一个 BookSim cycle 在 clock domain 不同时无效。
- 不修改完整 BookSim；只有 embedded transport 开始时才加入 narrow tracked
  facade。

## 修改文件

- `.codex/docs/02-trace-pipeline.md`：真实 record decoding 与 Mermaid call graph。
- `.codex/knowledge/trace-format.md`：精简 parser/consumer facts。
- `.codex/docs/09-accelsim-booksim-integration.md`：adapter API 与 clock contract。
- `.codex/docs/DECISIONS.md`：D006 clock/adapter decision。
- `tools/booksim_nvlink/check_nvlink2_analytical.py`：CSV accounting/trend checker。
- `tools/booksim_nvlink/transport_contract.py` 与
  `test_transport_contract.py`：packet/clock/completion reference contract/tests。

## 测试与结果

```bash
python3 tools/booksim_nvlink/check_nvlink2_analytical.py \
  /tmp/accelsim-booksim-nvlink2-sweep/nvlink2_sweep.csv \
  --flit-bytes 32 --link-flits-per-cycle 1
```

结果为 `ANALYTICAL_CHECK_PASS`（9 rows）。1/4/16 flits 分别对应
32/128/512 bytes；所有 latency 高于 serialization lower bound，并随 packet
size 和 configured link latency 单调增加。

Contract unit tests 也通过 packetization、1.25 GHz exact clock 与 two-message
completion ordering。固定 delay 是 synthetic reference，不是 BookSim measurement。

## Evidence 与限制

- **Verified**：一条真实 A800 compute-trace instruction 及其 Accel-Sim source
  path。
- **Verified**：synthetic BookSim sweep 的 arithmetic/trend checks。
- **Synthetic**：所有 BookSim network latency 与 bandwidth value。
- **Needs hardware validation**：NVLink protocol rate、link clock、NVSwitch
  topology/queue、NCCL、multicast 与 NVLS reduction。

下一步是在 BookSim protected per-cycle API 上实现最小 standalone endpoint
facade，保留已测试的 packetization/clock/completion contract，然后连接一个
two-GPU logical communication event。
