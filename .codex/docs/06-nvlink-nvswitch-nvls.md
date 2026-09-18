# NVLink, NVSwitch, and NVLS (research notes)

This document is a placeholder for a primary-source-grounded hardware study.
The permanent boundary is already fixed: BookSim transport timing is not an
NVSwitch model; an NVSwitch topology is not NVLS; and a SASS trace cannot by
itself expose switch queues or switch-side reduction timing.

The study must cover H100/NVLink generation, third-generation NVSwitch,
multicast memory, `multimem`, NVLink SHARP/NVLS, NCCL algorithm selection, and
the distinction between transport, reduction, multicast, and collective
completion. All hardware numbers require a cited NVIDIA/NCCL source and an
evidence label.
