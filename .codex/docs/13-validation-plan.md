# Validation 计划

Validation 必须区分 proposed-design path 与 real-hardware path。

## PTX/synthetic design validation

1. PTX parser and custom opcode recognition.
2. Functional semantics and register/dependency behaviour.
3. Issue latency, initiation interval, resource and completion invariants.
4. Communication-event schema, rank ordering and collective lifecycle.
5. BookSim packet bytes/flits, routing, clock conversion and completion.
6. Ring byte conservation and analytical latency/bandwidth bounds.
7. NVSwitch multicast/reduction operand counting and exactly-once completion.
8. 2/4/8 logical-GPU scaling and compute/communication overlap timelines.

以上只能建立 `unit`、`analytical` 和 `synthetic network` evidence。

## SASS and hardware validation

1. A800: NVBit → SASS trace → replay software compatibility.
2. H100/H200: real SM90 SASS trace and existing Hopper kernel correlation.
3. H100 NVSwitch: `nccl-tests` Ring/NVLS latency, algBW, busBW, topology and
   counters.
4. Compare H100 hardware, real H100 trace replay, and PTX/synthetic model with
   error decomposition.

只有第二部分的 level 3–4 才能支持真实 H100 NVLink/NVSwitch/NVLS behaviour
的结论。PTX result 在明确 assumptions 和 synthetic status 时可以支持
proposed architecture claim，但不能单独证明 hardware accuracy。
