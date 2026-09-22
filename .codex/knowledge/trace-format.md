# Trace-format facts

Status：2026-09-19 对一条真实 A800 trace record 完成 source audit。

## Producer

`util/tracer_nvbit/tracer_tool/tracer_tool.cu` 使用 NVBit callback 处理 CUDA
initialization/event/function loading，并写出 per-context instruction record。
`common.h` 包含 active/predicate mask、virtual PC、opcode ID、CTA/cluster ID，
以及普通 address 或 TMA metadata。Producer 还会 serialize Hopper cluster
fields 与 TMA multicast metadata。

## On-disk 与 parser path

Text path 由 `kernelslist.g`、kernel trace file 和 `trace_parser.{h,cc}` 组成。
`trace_parser.cc` 解析 kernel/grid/block/stream 与 cluster header、instruction
opcode/operand 和 address compression。Binary `.tracez` path 由
`tracez_format.h` 描述，并通过 `warp_trace_stream.h` stream。

## Simulator consumer path（已 audit 一条真实 instruction）

`gpu-simulator/accel-sim.cc` / `gpu-simulator/main.cc` 创建 trace framework；
`trace-driven/trace_driven.{h,cc}` 创建 `trace_kernel_info_t`、
`trace_shader_core_ctx`、`trace_shd_warp_t` 和 `trace_warp_inst_t`。已验证的
A800 `LDG.E.64` path 为：

```text
TracezReader::open/get
  → inst_trace_t::parse_from_string()
  → trace_shd_warp_t::get_next_trace_inst()
  → trace_warp_inst_t::parse_from_trace_struct()
  → func_exec_inst()
  → warp_inst_t::generate_mem_accesses()
  → simt_core_cluster::icnt_inject_request_packet()
  → icnt_push → cache/L2/DRAM timing
```

Warp 只有在 stream exhausted，且 `get_next_inst()` 中 pipeline/barrier/store/
scoreboard checks 清空后才完成。

代表性 raw record：
`48 0 0 17 0 0 0 0 0 0 0 00a0 ffffffff 1 R4 LDG.E.64 1 R4 8 1
0x7f7490524700 8 0 0 0 NoVal`。这是 trace version 6、lineinfo disabled、
32 lanes 全 active、8-byte global load、base+stride address，且没有 recorded
register value。完整 source mapping 见 `.codex/docs/02-trace-pipeline.md`。

## Existing network boundary

`gpu-simulator/gpgpu-sim/src/gpgpu-sim/icnt_wrapper.cc` 选择 local crossbar
（`network_mode 2`）或 BookSim-derived `intersim2`（`network_mode 1`）。
`InterconnectInterface::Push/Pop/Advance/Busy` 与 `GPUTrafficManager` 将
shader/memory node 之间的 `mem_fetch` traffic packetize。这是 GPU-internal
memory interconnect 的 verified source evidence，不是 GPU-to-GPU NVLink 或
NVSwitch evidence。

## 尚未建立的内容

当前 record 不包含 rank、NCCL collective lifecycle、peer GPU、NVSwitch queue
state、switch reduction timing 或 physical fabric contention。源码搜索未找到
`multimem` symbol；NVLink references 位于 hardware-monitoring helper，而不是
GPU-to-GPU simulator transport。是否存在独立 multi-GPU event path，必须在
设计前继续进行 source search。
