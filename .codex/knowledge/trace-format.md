# Trace-format facts (initial audit)

Status: partial source audit, 2026-09-18.

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

## Simulator consumer path (source names, not yet a complete autopsy)

`gpu-simulator/accel-sim.cc` / `gpu-simulator/main.cc` create the trace framework; trace classes in
`trace-driven/trace_driven.{h,cc}` create `trace_kernel_info_t`,
`trace_shader_core_ctx`, `trace_shd_warp_t`, and `trace_warp_inst_t`. The trace
instruction becomes a `warp_inst_t`-derived object and follows the ordinary
shader, cache, interconnect, and memory timing paths. Exact file/line/function
mapping for one real instruction remains a next milestone.

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
ordinary per-instruction fields. Whether any separate multi-GPU event path
exists must be verified by source search before design decisions.
