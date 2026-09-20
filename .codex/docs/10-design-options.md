# Design options

| Option | Strength | Risk | Role in this project |
|---|---|---|---|
| Extend SASS trace | one artifact; close to existing replay path | mixes instruction and collective semantics; cannot capture a new instruction from real hardware | validation-only or explicitly synthetic experiments |
| PTX execution-driven input | can execute a proposed ISA and generate functional control flow/address behaviour without hardware trace | less faithful to real compiler register allocation and machine scheduling | **primary design path** |
| Compute + communication event traces | explicit and replayable rank/collective lifecycle | needs synchronization and schema design | network/collective input alongside PTX |
| High-level orchestrator | clean rank/stream/collective lifecycle | more integration work | required for multi-GPU execution |
| Hybrid PTX + events + SASS validation | separates proposed design from hardware evidence | multiple input paths must share timing contracts | **chosen overall strategy** |

## Chosen direction

Use PTX execution-driven mode for proposed ISA, GPU microarchitecture, scheduler,
and in-switch-computing behaviour. Use an independent communication-event model
and a multi-GPU orchestrator for rank/stream/collective lifecycle. BookSim handles
generic packet/flit transport; NVSwitch/NVLS semantics remain above BookSim.
Use real A800/H100 SASS traces only to validate existing compute behaviour and
calibrate assumptions. A SASS trace is never treated as a substitute for a PTX
workload that contains a proposed instruction.
