# Design options

| Option | Strength | Risk |
|---|---|---|
| Extend SASS trace | one artifact | mixes instruction and collective semantics; compatibility risk |
| Compute + communication event traces | explicit and replayable | synchronization/merge design |
| High-level orchestrator | clean rank/stream/collective lifecycle | more integration work |
| Hybrid | preserves existing compute path while adding events | requires clear ownership |

Initial hypothesis: hybrid compute trace + communication event/orchestrator;
BookSim handles transport timing, while NVSwitch/NVLS semantics sit above it.
This remains a design hypothesis until current multi-GPU code and NCCL evidence
are audited.
