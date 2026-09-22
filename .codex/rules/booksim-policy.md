# BookSim 与 interconnect policy

BookSim 是 generic interconnection-network timing engine，不是 NVIDIA
NVSwitch implementation. Keep the standalone BookSim repository and the
existing Accel-Sim `gpu-simulator/gpgpu-sim/src/intersim2` code clearly
identified; the latter already models a GPU-memory interconnect and is not
evidence that GPU-GPU NVLink/NVSwitch exists.

固定并记录 inspected upstream SHA。优先使用 submodule 或 external
checkout plus a narrow adapter; do not copy/vendored-paste BookSim before a
decision in `DECISIONS.md`. Any local patch must be separately named and
upstream-trackable.

研究或嵌入 BookSim 时，必须定位 configuration、topology/network、
router, channel, virtual channel, flit/packet, routing, traffic manager,
simulation step, injection/ejection, and statistics code. Expose an adapter
with explicit operations such as create, inject, advance, query completion,
and collect stats rather than letting GPU code call internal globals.

以下 layer 必须分离并分别测试：

1. endpoint/collective semantics (rank, message, dependency, completion);
2. NVSwitch topology/protocol (ports, bandwidth, latency, routing, multicast);
3. NVLS reduction semantics (group, arithmetic, reduction completion);
4. BookSim transport timing (flits, buffers, VCs, contention, cycles).

绝不能把“BookSim integrated”等同于“NVSwitch implemented”，也不能把
“NVSwitch implemented”等同于“NVLS implemented”。记录 Accel-Sim/BookSim
clock conversion；不得默认一个 GPU cycle 等于一个 network cycle。
