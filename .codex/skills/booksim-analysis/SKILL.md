---
name: booksim-analysis
description: Study standalone and embedded BookSim to determine its build boundary, embeddable API, timing loop, and statistics without confusing it with NVSwitch.
---

## Workflow

1. Verify external BookSim status, branch, remote, and SHA. Separately inspect
   gpu-simulator/gpgpu-sim/src/intersim2; do not merge their identities.
2. Read build files and locate configuration, topology/network, router, channel,
   virtual channel, packet/flit, routing, traffic manager, injection/ejection,
   cycle advancement, and statistics symbols.
3. Trace the actual simulation loop, including input loading, Evaluate,
   ReadInputs, WriteOutputs, traffic-manager step, and completion.
4. Build and run the smallest standalone smoke test when possible. Capture
   config, seed, injected/delivered packets, latency, cycles, and stats.
5. Compare external checkout, submodule, subtree, and narrowly patched fork.
   Record reproducibility, upstream tracking, build isolation, and maintenance
   in DECISIONS.md before copying code.
6. Design an adapter with create/configure, inject, advance, completion, and
   stats operations; define packet ownership and clock conversion.
7. Keep endpoint/collective semantics, NVSwitch topology, multicast/reduction,
   and BookSim transport timing as separate layers.

## Completion

Pin SHA, source symbols, commands, decision, adapter boundary, clock
assumptions, and the limitation that transport smoke tests are not NVLS
hardware evidence.
