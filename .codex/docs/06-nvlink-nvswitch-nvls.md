# NVLink, NVSwitch, and NVLS (research notes)

Initial primary-source facts (accessed 2026-09-18):

- NVIDIA Fabric Manager documents DGX/HGX H100 as eight H100 GPUs with four
  third-generation NVSwitches and says Fabric Manager/NVLSM configure routing,
  GPU-side routing, port maps, and forwarding tables.
- The same guide states H100-and-later NVSwitch systems support NVLink multicast
  and reduction, with finite multicast slots.
- PTX ISA 8.1+ defines `multimem.ld_reduce`, `multimem.st`, and `multimem.red`
  over multimem addresses; the instructions require `sm_90` or newer and a
  multimem address cannot be accessed by ordinary `ld/st`.
- Current NVIDIA NCCL `src/transport/nvls.cc` checks GPU count/multicast
  capability, uses CUDA multicast APIs, sets up NVLS trees/channels, and binds
  unicast backing into multicast partitions. This is a communicator setup and
  transport/semantic path, not just a packet route.

Sources: [Fabric Manager guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/),
[PTX multimem instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html),
[NCCL nvls.cc](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc),
and [NCCL device API](https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/usage/deviceapi.rst).

The permanent boundary is already fixed: BookSim transport timing is not an
NVSwitch model; an NVSwitch topology is not NVLS; and a SASS trace cannot by
itself expose switch queues or switch-side reduction timing.

The study must cover H100/NVLink generation, third-generation NVSwitch,
multicast memory, `multimem`, NVLink SHARP/NVLS, NCCL algorithm selection, and
the distinction between transport, reduction, multicast, and collective
completion. All hardware numbers require a cited NVIDIA/NCCL source and an
evidence label.
