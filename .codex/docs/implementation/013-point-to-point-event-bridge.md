# 013 — P2 point-to-point event bridge

P2 将 P0 `COMM_SUBMIT` 生命周期接到现有 `ReferenceEndpointAdapter`。桥接器只接受
单源、单目的、正字节数的事件；`pump(now_ps)` 在 event ready 后注入 packet，并把
transport completion 回送给 orchestrator。完成记录保留 event token、flit 数和
reference backend 标识。`time_domain=sim_cycle` 被拒绝，防止把 P1 GPU cycles
误当作 ps。

**Verified**：3 个 focused tests 覆盖 submit→inject→complete、ready time 和
无效输入；P0 schema 与 transport regression 继续通过。

**限制**：目前是 Python deterministic reference transport，不是嵌入式 BookSim、
NVLink 或两个正在执行的 GPGPU-Sim 实例。P1 JSONL 与 P2 的 runtime online
producer/consumer 尚未连接；这会在多 GPU clock coupling 阶段完成。

下一步 P3 在本接口上实现 NCCL-style Ring 分块/轮次调度。
