# Hardware validation boundary

本地 A800 是 Ampere software-pipeline instrument，可以验证 CUDA/NVBit
instrumentation、trace generation/post-processing、parser replay、build 与
automation。它不能验证 SM90 instruction（WGMMA/TMA/mbarrier/cluster）、NVLink
4、H100 NVSwitch、multicast memory、`multimem` 或 NVLS performance。

H100/H200 可以验证 Hopper trace production/consumption；单张 H100 仍不能验证
multi-GPU routing、NVSwitch contention 或 NVLS collective。这些 claim 必须使用
明确数量与 topology 的 GPU；BookSim transport model 或 copied single-GPU trace
不是 hardware result。

每次 hardware run 必须记录 GPU model/count、compute capability、driver、CUDA、
NCCL、Fabric Manager、topology（`nvidia-smi topo -m`）、NVLink/NVSwitch state、
benchmark、environment variable、command 与 output path，并明确说明 run 能证明
和不能证明什么。不得把 A800/Ampere result 当作 H100/Hopper result，不得把
synthetic NVLS 当作 real NVLS execution，也不得把 transport-only model 当作
real NVSwitch implementation。
