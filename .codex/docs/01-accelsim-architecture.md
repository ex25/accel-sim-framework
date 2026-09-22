# Accel-Sim architecture（初始映射）

这是初始 source map，不是完整 audit。仓库通过 Accel-Sim trace-driven wrapper
调用 nested GPGPU-Sim 的 timing/functional model。`main.cc`/`accel-sim.cc`
提供 trace mode 入口和 orchestration；`trace_parser` 与 `warp_trace_stream`
加载 command、kernel 和 warp records；`trace_shader_core_ctx` 将
`trace_warp_inst_t` 送入 shader pipeline。

后续 timing path 使用 GPGPU-Sim 的 shader core、cache、interconnect、memory
partition 和 DRAM components。`icnt_wrapper.cc` 在 local xbar 与
BookSim-derived `intersim2` 之间选择；
`InterconnectInterface::Push/Pop/Advance` 和 `GPUTrafficManager` 将 shader 与
memory node 之间的 `mem_fetch` request packetize。该路径已验证为 GPU 内部
memory-system path，不能作为 GPU-to-GPU NVLink/NVSwitch support 的证据。
完整 instruction-to-memory completion path 由后续 trace autopsy 文档补充。
