# Validation plan

1. Unit invariants: packet bytes, routing, ordering, reduction, completion.
2. Analytical bounds: bandwidth/latency scaling and byte conservation.
3. Synthetic BookSim Ring/NVLS trend comparison.
4. Public hardware data as trend context, never ground truth.
5. Real H100 NVSwitch `nccl-tests` Ring/NVLS with complete inventory,
   topology, NCCL/Fabric Manager versions, and counters.
6. Application trace + simulator correlation.

Only levels 5–6 can support claims about real H100 NVSwitch/NVLS behaviour.
