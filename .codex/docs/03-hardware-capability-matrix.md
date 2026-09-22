# Hardware capability matrix

| 环境 | 可开发/构建 | 可验证 | 无法证明 |
|---|---|---|---|
| CPU-only | parser、unit tests、BookSim、analytical model | deterministic software behaviour | GPU trace、SASS timing、NVLink |
| 本地 A800（1× PCIe） | 真实 Ampere CUDA/NVBit trace pipeline、replay/config/build | Ampere software path 与 trace plumbing | SM90、H100 NVLink4、NVSwitch、NVLS |
| 单张 H100/H200 | Hopper trace generation 与 SM90 model correlation | SASS/TMA/WGMMA/mbarrier/cluster path | multi-GPU switch/fabric/reduction |
| 多张无 NVSwitch 的 H100 | 受 topology 限制的 peer/NCCL software experiments | 有限的 direct-link collective behaviour | NVSwitch/NVLS switch reduction |
| H100 NVSwitch domain | NCCL Ring/NVLS、topology 与 hardware counters | 目标 collective/fabric correlation | 超出实测 topology/config 的结论 |

本地主机属于第二行。所有实验 metadata 和正文都必须明确保留这一证据边界。
