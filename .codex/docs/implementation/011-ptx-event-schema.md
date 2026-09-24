# 011 — PTX-to-communication event schema（P0）

## 问题与动机

**Verified**：当前 `cais.add.u32` 只完成 single-GPU register semantics，没有
communication side effect；现有 BookSim contract 也只描述 transport accounting。
在修改 simulator core、NCCL 或 BookSim 之前，需要一个可以独立测试的
instruction-to-event boundary。

## 设计

新增 `tools/booksim_nvlink/communication_events.py`，定义：

- `CommunicationEvent`：GPU、rank、stream、kernel、PTX/SASS location、scope、
  collective、chunk/channel、barrier token、dependency 和 global `ready_time_ps`；
- `EventKind`：local barrier、memory ordering、communication submit/inject/
  complete、barrier wakeup 与 collective completion；
- `EventCompletion`：transport/fabric 返回的 completion token 和 timestamp；
- `LogicalGPUContext`：logical device/rank identity；
- `CommunicationEventOrchestrator`：`WAITING → READY → INJECTED → COMPLETED`
  的确定性状态机。

事件 schema 与 BookSim、NCCL 和 GPGPU-Sim 解耦。`ready_time_ps` 使用整数或
`Fraction` 的 global picosecond domain；不能假设 GPU cycle 等于 network cycle。
当前 module 不实现 packetization、NCCL algorithm、NVSwitch route 或 NVLS
reduction timing。

## Call path

```text
PTX/SASS issue hook（future）
  → CommunicationEvent
  → CommunicationEventOrchestrator.submit
  → future NCCL/fabric adapter.mark_injected
  → EventCompletion
  → future GPU mbarrier/scoreboard wakeup
```

P0 只实现中间 contract 和 reference state machine；没有修改 PTX frontend、
SASS trace format、nested GPGPU-Sim 或 existing GPU-memory `intersim2`。

## 修改文件

- `tools/booksim_nvlink/communication_events.py`
- `tools/booksim_nvlink/test_communication_events.py`

## 验证

命令：

```bash
cd tools/booksim_nvlink
python3 test_communication_events.py
cd ../..
python3 tools/booksim_nvlink/test_transport_contract.py
```

结果：P0 focused tests 4/4 通过；既有 transport contract 3/3 通过。
这是 **Verified / unit** evidence，只证明 schema、依赖释放、状态转移、
timestamp round-trip 和 duplicate context 检查；不证明真实 PTX、NCCL、
NVLink、NVSwitch 或 NVLS 行为。

## 限制与下一步

- 当前 event producer 仍未接入 GPGPU-Sim；P1 才处理本地 barrier、mbarrier、
  fence 和 GPU warp wakeup。
- 当前 `metadata` 是稳定的 string tuple，后续若需要结构化 reduction metadata
  要扩展 schema 和 version。
- 下一阶段必须先完成 P1 focused tests，再进入 P2 point-to-point transport。
