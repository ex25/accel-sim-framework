# 008 — Trace autopsy and BookSim endpoint/clock contract

## Problem

The A800 trace and standalone BookSim prototype were validated, but the
project still needed (1) a source-level proof of how one real instruction is
consumed by Accel-Sim and (2) a precise boundary for future timestamped
GPU-to-GPU transport. Raw BookSim latency also needed a repeatable analytical
sanity check before Ring work.

## Design

The autopsy uses the real `LDG.E.64` line from the 2026-09-19 A800 vectoradd
trace. It follows NVBit emission, text/address compression, `.tracez` parsing,
warp instruction construction, memory coalescing, GPU memory interconnect
injection, and completion gating. The complete call graph and field decoding
are in `.codex/docs/02-trace-pipeline.md`.

The future communication boundary is a transport-only endpoint adapter:
`create_network`, `inject_message`, `advance_to`, `poll_completed`, and
`read_stats`. It packetizes bytes into flits and reports tail-flit delivery;
collective/rank/multicast/reduction semantics stay above BookSim. The adapter
uses an integer-picosecond global clock with rational GPU/network period
conversion. This is a design artifact, not an implemented Accel-Sim hook.

## Alternatives

- Reusing `TrafficManager::Run()` cannot provide per-event injection or cycle
  advancement because the relevant BookSim methods are protected.
- Treating one GPU cycle as one BookSim cycle is invalid when clock domains
  differ and would make latency calibration opaque.
- Patching the full BookSim simulator was deferred; a narrow tracked facade is
  required only when embedded transport implementation begins.

## Files changed

- `.codex/docs/02-trace-pipeline.md`: real record decoding and Mermaid call
  graph.
- `.codex/knowledge/trace-format.md`: condensed parser/consumer facts.
- `.codex/docs/09-accelsim-booksim-integration.md`: adapter API and clock
  contract.
- `.codex/docs/DECISIONS.md`: D006 records the clock/adapter decision.
- `tools/booksim_nvlink/check_nvlink2_analytical.py`: CSV accounting and trend
  checker.
- `tools/booksim_nvlink/README.md`: checker reproduction command.
- `tools/booksim_nvlink/transport_contract.py` and
  `test_transport_contract.py`: dependency-free packet/clock/completion
  reference contract and unit tests.

## Tests and results

```bash
python3 tools/booksim_nvlink/check_nvlink2_analytical.py \
  /tmp/accelsim-booksim-nvlink2-sweep/nvlink2_sweep.csv \
  --flit-bytes 32 --link-flits-per-cycle 1
```

Result: `ANALYTICAL_CHECK_PASS`, 9 rows. The sweep reports 32/128/512 bytes
for 1/4/16 flits, all latency values exceed the serialization lower bound, and
latency increases monotonically with packet size and configured link latency.

The contract unit tests also pass for packetization, a 1.25 GHz exact clock,
and two-message completion ordering. Their fixed delay is a synthetic
reference, not a BookSim measurement.

## Limitations and validation status

- `Verified`: one actual A800 compute-trace instruction and its current
  Accel-Sim source path.
- `Verified`: arithmetic/trend checks on the synthetic BookSim sweep.
- `Synthetic`: all BookSim network latency and bandwidth values.
- `Needs hardware validation`: NVLink protocol rate, link clock, NVSwitch
  topology/queues, NCCL, multicast, and NVLS reduction.

## Next step

Implement the smallest standalone endpoint facade (without Accel-Sim core
changes) over BookSim's protected per-cycle API, preserving the tested
packetization/clock/completion contract. Then connect a two-GPU logical
communication event and compare its accounting with the analytical checker.
