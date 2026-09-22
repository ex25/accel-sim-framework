# NVLink、NVSwitch 与 NVLS 研究笔记

初步 primary-source 事实（2026-09-19 核查）：

- NVIDIA Fabric Manager 文档将 DGX/HGX H100 描述为八张 H100 GPU 和四个
  third-generation NVSwitch，并说明 Fabric Manager/NVLSM 配置 routing、
  GPU-side routing、port map 与 forwarding table。
- 同一指南说明 H100 及更新的 NVSwitch systems 支持 NVLink multicast 和
  reduction，且 multicast slots 数量有限。
- PTX ISA 8.1+ 在 multimem address 上定义 `multimem.ld_reduce`、
  `multimem.st` 与 `multimem.red`；这些指令要求 `sm_90` 或更新架构，普通
  `ld/st` 不能访问 multimem address。
- 当前 NVIDIA NCCL `src/transport/nvls.cc` 检查 GPU count/multicast
  capability，使用 CUDA multicast APIs，建立 NVLS trees/channels，并把
  unicast backing 绑定到 multicast partitions。这是 communicator setup 和
  transport/semantic path，不只是 packet route。
- 当前 NCCL 源码还要求至少两张 GPU，并受 `NCCL_NVLS_ENABLE` 控制；它会
  报告 multicast support 并创建 NVLS tree connections。因此 simulator 除
  data packets 外，还必须建模 communicator setup 与 resource state。

Sources: [Fabric Manager guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/),
[PTX multimem instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html),
[NCCL nvls.cc](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc),
and [NCCL device API](https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/usage/deviceapi.rst).

固定边界如下：BookSim transport timing 不是 NVSwitch model；NVSwitch topology
不是 NVLS；SASS trace 本身也无法暴露 switch queue 或 switch-side reduction
timing。

研究必须覆盖 H100/NVLink generation、third-generation NVSwitch、multicast
memory、`multimem`、NVLink SHARP/NVLS、NCCL algorithm selection，以及
transport、reduction、multicast、collective completion 之间的区别。所有
hardware number 都必须引用 NVIDIA/NCCL source，并标注 evidence label。本次
核查的 source 包括 NVIDIA
[Fabric Manager guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/),
[PTX multimem section](https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html),
and [NCCL `nvls.cc`](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc).
