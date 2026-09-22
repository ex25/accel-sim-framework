# PTX-first implementation roadmap

项目目前分为两条路线。PTX 路线是 hypothetical ISA/microarchitecture 和
in-switch computing 的主设计路线；SASS 路线用于验证 A800/H100 已存在的
behaviour。以下每个 milestone 都必须记录 input source 和 evidence level；
synthetic PTX result 不是 H100 hardware result。

| Milestone | Work | Exit criterion | Hardware |
|---|---|---|---|
| M0 | Repository/Codex memory and baseline | branch, SHAs, rules, build/test evidence committed | none/A800 optional |
| M1 | A800 SASS pipeline | CUDA→NVBit→trace→replay reproducible | A800 |
| M2 | Trace autopsy and Hopper gap | one instruction mapped source-to-completion; unsupported fields documented | A800/source |
| M3 | PTX execution-driven baseline | SM90 PTX benchmark runs; PTX parser→functional execution→timing path documented | none; A800 optional |
| M4 | Minimal custom PTX instruction | parser, semantics, latency, dependency, issue and completion tests pass for one instruction | none |
| M5 | Communication-event schema | rank/device/stream/collective/chunk/ready/completion schema has fixtures and validation | none |
| M6 | Logical GPU orchestrator | two PTX GPU contexts can run with explicit dependencies and deterministic event ordering | none |
| M7 | BookSim transport facade | timestamped two-endpoint injection/completion, clock conversion and byte/flit accounting pass | none |
| M8 | Ring baseline | 2/4/8 logical ranks, message sweep, completion and analytical bounds pass | none; synthetic |
| M9 | NVSwitch semantic layer | topology, route, VC/buffer assumptions and endpoint mapping have focused tests | none; synthetic |
| M10 | NVLS semantics | multicast group, reduction session, operand count and completion state machine pass | none; synthetic |
| M11 | PTX compute/communication overlap | custom PTX communication instruction and network completion release GPU work correctly | none; synthetic |
| M12 | SASS validation bridge | existing H100/A800 SASS compute path can feed the same event/timing interfaces | A800/H100 |
| M13 | Hardware calibration | Ring/NVLS `nccl-tests`, counters and simulator results are compared | H100 NVSwitch |
| M14 | Research extension | CAIS- or DySHARP-like ISA/scheduler experiment has reproducible workload and ablations | H100 preferred; synthetic first |

## Milestone gate format

Before advancing, record:

- input source: PTX, real SASS, synthetic event, or hardware measurement;
- files/classes/functions changed;
- before/after call path;
- focused tests and baseline classification;
- limitations and evidence label;
- Codex execution time and user review time;
- commit SHA and next decision.

当前已完成 M3 与 M4。下一阶段是 M5 communication-event schema，而不是全面
重写 BookSim/NVSwitch：先定义 event contract，再逐步接入 transport。
