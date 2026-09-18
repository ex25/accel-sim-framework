# Accel-Sim ↔ BookSim integration

No adapter exists yet. The candidate boundary is:

```mermaid
flowchart TD
  A[Accel-Sim compute contexts] --> B[Multi-GPU orchestrator]
  B --> C[Collective/NVSwitch semantic layer]
  C --> D[BookSim transport timing]
```

Clock coupling is unresolved. Options are lockstep cycles, explicit rational
clock ratios, event-driven global time, or batched advancement. The choice must
be justified from event ordering and performance needs; never silently equate
one Accel-Sim cycle with one BookSim cycle.
