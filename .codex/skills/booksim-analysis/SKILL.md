---
name: booksim-analysis
description: 研究 standalone/embedded BookSim 的 build boundary、API、timing loop 与 statistics，同时避免把 BookSim 混同为 NVSwitch。
---

## Workflow

1. 独立确认 external BookSim 与 nested `src/intersim2` 的 status/branch/remote/SHA。
2. 定位 configuration、topology/network、router、channel、VC、packet/flit、
   routing、TrafficManager、injection/ejection、cycle advancement 与 stats。
3. 跟踪真实 simulation loop：input、Evaluate、ReadInputs、WriteOutputs、step、
   completion。
4. 运行最小 standalone smoke test，记录 config、seed、packet、latency、cycle、stats。
5. 在 copy code 前比较 external checkout、submodule、subtree、narrow patch，并
   将 reproducibility/upstream tracking/maintenance 写入 `DECISIONS.md`。
6. 设计 create/configure、inject、advance、completion、stats adapter，定义
   packet ownership 与 clock conversion。
7. 分离 endpoint/collective semantics、NVSwitch topology、multicast/reduction
   与 BookSim transport timing。

完成时必须固定 SHA、source symbol、command、decision、adapter boundary、clock
assumption，并声明 transport smoke test 不是 NVLS hardware evidence。
