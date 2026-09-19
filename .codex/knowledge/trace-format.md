# Trace-format facts

Status: source-audited for one real A800 trace record, 2026-09-19.

## Producer

`util/tracer_nvbit/tracer_tool/tracer_tool.cu` uses NVBit callbacks for CUDA
initialization/events/function loading and writes per-context instruction
records. `common.h` contains active/predicate masks, virtual PC, opcode ID,
CTA/cluster IDs and either regular addresses or TMA metadata. The producer also
serializes Hopper cluster fields and TMA multicast metadata.

## On-disk and parser path

The text path is organized around `kernelslist.g`, kernel trace files, and
`trace_parser.{h,cc}`. `trace_parser.cc` parses kernel/grid/block/stream and
cluster headers, instruction opcode/operand data, and address compression. The
binary `.tracez` path is described by `tracez_format.h` and streamed by
`warp_trace_stream.h`.

## Simulator consumer path (one real instruction audited)

`gpu-simulator/accel-sim.cc` / `gpu-simulator/main.cc` create the trace framework;
trace classes in `trace-driven/trace_driven.{h,cc}` create
`trace_kernel_info_t`, `trace_shader_core_ctx`, `trace_shd_warp_t`, and
`trace_warp_inst_t`. For the verified A800 `LDG.E.64` record, the call path is
`TracezReader::open/get` -> `inst_trace_t::parse_from_string()` ->
`trace_shd_warp_t::get_next_trace_inst()` ->
`trace_warp_inst_t::parse_from_trace_struct()` -> `func_exec_inst()` ->
`warp_inst_t::generate_mem_accesses()` ->
`simt_core_cluster::icnt_inject_request_packet()` -> `icnt_push` and ordinary
cache/L2/DRAM timing. A warp completes only after stream exhaustion and the
pipeline/barrier/store/scoreboard checks in `get_next_inst()`.

Representative raw record:
`48 0 0 17 0 0 0 0 0 0 0 00a0 ffffffff 1 R4 LDG.E.64 1 R4 8 1
0x7f7490524700 8 0 0 0 NoVal`. It is trace version 6, lineinfo disabled,
all 32 lanes active, an 8-byte global load, base+stride addresses, and no
recorded register values. Full source mapping is in
`.codex/docs/02-trace-pipeline.md`.

## Existing network boundary

`gpu-simulator/gpgpu-sim/src/gpgpu-sim/icnt_wrapper.cc` selects either the
local crossbar (`network_mode 2`) or the BookSim-derived `intersim2` interface
(`network_mode 1`). `InterconnectInterface::Push/Pop/Advance/Busy` and
`GPUTrafficManager` packetize `mem_fetch` traffic between shader and memory
nodes. This is verified source evidence for a GPU-internal memory interconnect,
not for GPU-to-GPU NVLink or NVSwitch.

## What is not established

The current record does not contain rank, NCCL collective lifecycle, peer GPU,
NVSwitch queue state, switch reduction timing, or physical fabric contention as
ordinary per-instruction fields. The source search found no `multimem` symbols;
NVLink references are in hardware-monitoring helpers rather than a GPU-to-GPU
simulator transport. Whether any separate multi-GPU event path exists must be
verified by source search before design decisions.
