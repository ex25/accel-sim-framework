# Hardware validation boundary

- Never present A800/Ampere results as H100/Hopper results.
- Never call a BookSim transport experiment a real NVSwitch result.
- Never call synthetic NVLS semantics or copied traces a real NVLS execution.
- An H100 trace can validate Hopper/SM90 trace consumption, but only a system
  with the required GPU topology can validate NVLink, NVSwitch, or NVLS.
- Record GPU model, count, driver, CUDA, NCCL, Fabric Manager, topology, and
  environment for every hardware experiment.
