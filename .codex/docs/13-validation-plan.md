# Validation plan

Validation must distinguish the proposed-design path from the real-hardware
path.

## PTX/synthetic design validation

1. PTX parser and custom opcode recognition.
2. Functional semantics and register/dependency behaviour.
3. Issue latency, initiation interval, resource and completion invariants.
4. Communication-event schema, rank ordering and collective lifecycle.
5. BookSim packet bytes/flits, routing, clock conversion and completion.
6. Ring byte conservation and analytical latency/bandwidth bounds.
7. NVSwitch multicast/reduction operand counting and exactly-once completion.
8. 2/4/8 logical-GPU scaling and compute/communication overlap timelines.

These establish `unit`, `analytical`, and `synthetic network` evidence only.

## SASS and hardware validation

1. A800: NVBit → SASS trace → replay software compatibility.
2. H100/H200: real SM90 SASS trace and existing Hopper kernel correlation.
3. H100 NVSwitch: `nccl-tests` Ring/NVLS latency, algBW, busBW, topology and
   counters.
4. Compare H100 hardware, real H100 trace replay, and PTX/synthetic model with
   error decomposition.

Only levels 3–4 of the second section can support claims about real H100
NVLink/NVSwitch/NVLS behaviour. A PTX result can support a proposed architecture
claim when its assumptions and synthetic status are explicit; it cannot by
itself establish hardware accuracy.
