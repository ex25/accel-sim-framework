# Design options

| Option | Strength | Risk | Role in this project |
|---|---|---|---|
| Extend SASS trace | one artifact; close to existing replay path | mixes instruction and collective semantics; cannot capture a new instruction from real hardware | validation-only or explicitly synthetic experiments |
| PTX execution-driven input | can execute a proposed ISA and generate functional control flow/address behaviour without hardware trace | less faithful to real compiler register allocation and machine scheduling | **primary design path** |
| Compute + communication event traces | explicit and replayable rank/collective lifecycle | needs synchronization and schema design | network/collective input alongside PTX |
| High-level orchestrator | clean rank/stream/collective lifecycle | more integration work | required for multi-GPU execution |
| Hybrid PTX + events + SASS validation | separates proposed design from hardware evidence | multiple input paths must share timing contracts | **chosen overall strategy** |

## 已选择的方向

proposed ISA、GPU microarchitecture、scheduler 和 in-switch-computing behaviour
使用 PTX execution-driven mode。rank/stream/collective lifecycle 使用独立的
communication-event model 和 multi-GPU orchestrator。BookSim 只处理 generic
packet/flit transport；NVSwitch/NVLS semantics 位于 BookSim 之上。真实
A800/H100 SASS trace 只用于验证已有 compute behaviour 和校准假设。SASS
trace 永远不能替代包含 proposed instruction 的 PTX workload。
