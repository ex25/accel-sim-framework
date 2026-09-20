# Single-GPU versus multi-GPU trace

## Initial conclusion

`Verified source boundary / needs workload validation`: a single-GPU compute SASS trace can be reused
as a workload template for logical ranks only when the research question
explicitly permits identical compute. It is not a multi-GPU execution record.

Replication does not supply rank/device identity, peer addresses, NCCL algorithm,
collective launch/completion, message partitioning, synchronization, topology,
routing, contention, switch reduction, multicast, or compute/communication
overlap. Those belong in independent communication events and an orchestrator
unless a verified existing trace path provides them. The current tracer's
apparent multi-context support is per-`CUcontext`: `ctx_kernelid`,
context-specific files, and context lifecycle callbacks. Its trace records
include CTA/cluster identity, not an explicit NCCL rank, CUDA device ID, peer
endpoint, or fabric event. Release-note wording about “multi-GPU trace” must
therefore not be treated as multi-GPU simulation evidence.

Candidate designs (to be evaluated): (A) extend SASS traces, (B) PTX workload
+ communication event trace, (C) high-level event orchestrator, (D) PTX-primary
hybrid with SASS validation. The confirmed choice is D: use PTX for proposed
ISA/microarchitecture and new GPU behaviour, add explicit communication
semantics outside ordinary instruction records, and preserve SASS for existing
hardware validation.
