# 2026-09-19 A800 trace 与 BookSim NVLink2 smoke

## 固定信息

- Date：2026-09-19
- Branch/base：`feat/nvbit-a800-booksim-nvlink` / upstream `d930ad6`
- Accel-Sim baseline：`d930ad6`
- Nested GPGPU-Sim：`91880c53`
- BookSim：external checkout `28f4329`
- Hardware：一张 NVIDIA A800 80GB PCIe，sm_80；没有 NVLink peer
- Evidence：真实 A800 software trace + synthetic BookSim transport

## A800 trace run

- Workload：NVBit 1.8 `test-apps/vectoradd/vectoradd.cu`
- Compile target：`sm_80`
- Trace：一个 kernel、98 CTAs、每个 CTA 1024 threads、50,066 reported
  instructions
- Temporary output：`/tmp/in-switch-a800-trace-20260919`
- Raw trace：271,492 bytes
- Post-processed tracez：1,083,875 bytes
- Replay：exit 0，6,824 cycles、1,502,112 simulated instructions、75,000
  request packets、75,000 reply packets
- Simulator wall time：约 16 秒
- Validation boundary：只验证 software pipeline，不验证 Hopper 或 inter-GPU
  communication

## BookSim run

- Topology：两个 AnyNet router，每个 router 一个 logical GPU endpoint，一条
  bidirectional router link
- Traffic：`bitcomp`，node 0→1 与 node 1→0
- Smoke command：`BOOKSIM_ROOT=/home/wangjz/in-switch/booksim2 tools/booksim_nvlink/run_nvlink2.sh`
- Sweep command：`BOOKSIM_ROOT=/home/wangjz/in-switch/booksim2 tools/booksim_nvlink/sweep_nvlink2.sh`
- Temporary outputs：`/tmp/accelsim-booksim-nvlink2-smoke` 与
  `/tmp/accelsim-booksim-nvlink2-sweep`
- Smoke：pass，two hops，one flit/link latency one 时为 12 cycles
- Sweep：9 个 link-latency/packet-size cases 全部通过
- Seed：该 checkout 的 BookSim default deterministic seed
- Validation boundary：只验证 generic cycle-level transport；没有 NVLink
  bandwidth 或 protocol calibration

## 结果解释

Latency 随 link latency 和 packet size 同时增加，符合预期 sanity trend。
由于 offered load 很低且 run configuration 不变，accepted rate 保持固定。
该结果支持 topology/wrapper 作为最小 research scaffold，但不能证明 model
accuracy。
