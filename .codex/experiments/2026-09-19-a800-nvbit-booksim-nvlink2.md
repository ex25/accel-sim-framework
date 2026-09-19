# 2026-09-19 A800 trace and BookSim NVLink2 smoke

## Identity

- Date: 2026-09-19
- Branch/base: feat/nvbit-a800-booksim-nvlink / 85cccb1
- Accel-Sim baseline: d930ad6; working branch documentation/code pending commit
- Nested GPGPU-Sim: 91880c53
- BookSim: external checkout 28f4329
- Hardware: one NVIDIA A800 80GB PCIe, sm_80; no NVLink peer
- Evidence: real A800 software trace plus synthetic BookSim transport

## A800 trace run

- Workload: NVBit 1.8 test-apps/vectoradd/vectoradd.cu
- Compile target: sm_80
- Trace: one kernel, 98 CTAs, 1024 threads/CTA, 50,066 reported instructions
- Temporary output: /tmp/in-switch-a800-trace-20260919
- Raw trace: 271,492 bytes
- Post-processed tracez: 1,083,875 bytes
- Replay result: exit 0, 6,824 cycles, 1,502,112 simulated instructions,
  75,000 request packets, and 75,000 reply packets
- Simulator wall time: approximately 16 seconds
- Validation: software pipeline only; not Hopper or inter-GPU communication

## BookSim run

- Topology: two AnyNet routers, one logical GPU endpoint each, one
  bidirectional router link
- Traffic: bitcomp, which maps node 0 to 1 and node 1 to 0
- Smoke command: BOOKSIM_ROOT=/home/wangjz/in-switch/booksim2
  tools/booksim_nvlink/run_nvlink2.sh
- Sweep command: BOOKSIM_ROOT=/home/wangjz/in-switch/booksim2
  tools/booksim_nvlink/sweep_nvlink2.sh
- Temporary outputs: /tmp/accelsim-booksim-nvlink2-smoke and
  /tmp/accelsim-booksim-nvlink2-sweep
- Smoke result: pass, two hops, 12 cycles for one flit/link latency one
- Sweep result: all nine link-latency and packet-size cases pass
- Seed: BookSim default deterministic seed from this checkout
- Validation: generic cycle-level transport only; no NVLink bandwidth or
  protocol calibration

## Result interpretation

Latency increases with both link latency and packet size, which is the expected
sanity trend. Accepted rate remains fixed because the low offered load and run
configuration are unchanged. This supports the topology and wrapper as a
minimal research scaffold, but it does not yet establish model accuracy.
