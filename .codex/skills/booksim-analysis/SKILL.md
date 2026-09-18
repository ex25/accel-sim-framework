---
name: booksim-analysis
description: Study the checked-out BookSim network simulator and determine its embeddable API, build boundary, timing loop, and statistics.
---

Pin the inspected SHA. Locate network, topology, router, channel, VC/flit/
packet, traffic manager, routing, configuration, simulation clock, and stats
classes/functions. Run only a small standalone smoke test if dependencies and
permissions permit. Compare external checkout, submodule, subtree, and vendored
options; do not copy BookSim before recording a decision. Explicitly separate
BookSim transport timing from future NVSwitch/NVLS semantics.
