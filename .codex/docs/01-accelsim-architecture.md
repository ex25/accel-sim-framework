# Accel-Sim architecture (initial map)

This is an initial source map, not a completed audit. The repository contains
an Accel-Sim trace-driven wrapper over the nested GPGPU-Sim timing/functional
model. `main.cc`/`accel-sim.cc` provide the trace-mode entry and orchestration;
`trace_parser` and `warp_trace_stream` load command/kernel/warp records;
`trace_shader_core_ctx` feeds `trace_warp_inst_t` into the shader pipeline.

The timing path then uses GPGPU-Sim shader core, cache, interconnect, memory
partition, and DRAM components. The current in-simulator interconnect is a
memory-system path; this is not evidence of GPU-to-GPU NVLink/NVSwitch support.
The exact cycle call graph and completion path will be expanded after the trace
autopsy.
