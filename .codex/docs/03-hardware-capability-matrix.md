# Hardware capability matrix

| Environment | Develop/build | Validate | Cannot establish |
|---|---|---|---|
| CPU-only | parser, unit tests, BookSim, analytical model | deterministic software behaviour | GPU trace, SASS timing, NVLink |
| Local A800 (1× PCIe) | real Ampere CUDA/NVBit trace pipeline, replay/config/build | Ampere software path and trace plumbing | SM90, H100 NVLink4, NVSwitch, NVLS |
| Single H100/H200 | Hopper trace generation and SM90 model correlation | SASS/TMA/WGMMA/mbarrier/cluster path | multi-GPU switch/fabric/reduction |
| Multiple H100 without NVSwitch | peer/NCCL software experiments subject to topology | limited direct-link collective behaviour | NVSwitch/NVLS switch reduction |
| H100 NVSwitch domain | NCCL Ring/NVLS, topology and hardware counters | target collective/fabric correlation | claims beyond measured topology/config |

The local host currently falls in the second row. Results must carry this
boundary into experiment metadata and prose.
