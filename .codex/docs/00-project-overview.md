# 项目概览

本项目的研究目标是构建分层 simulator：Accel-Sim 负责 GPU compute 与 memory
timing；未来的 multi-GPU orchestrator 负责 rank、stream、dependency 和
collective lifecycle；NVLink/NVSwitch semantic layer 负责 endpoint、topology、
multicast 与 reduction；BookSim 负责通用 packet/flit、router、channel、
contention 和 network clock timing。

第一个有价值的结果不是完整 H100 emulator，而是可复现的 single-GPU
baseline、有证据支持的 gap analysis，以及最小 logical GPU0→GPU1 transport
实验。先实现 Ring AllReduce，再研究 NVLS；只有 transport、completion
ordering 和 reduction semantics 都有 focused tests 后，才能进入 NVLS。
