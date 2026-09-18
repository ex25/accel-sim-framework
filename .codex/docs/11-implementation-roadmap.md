# Implementation roadmap

| Milestone | Exit criterion | Hardware |
|---|---|---|
| M0/M1 context + baseline | state, rules, branch, build evidence committed | none/A800 optional |
| M2 A800 trace | tiny CUDA→NVBit→trace→replay recorded | A800 |
| M3 trace autopsy | one real instruction mapped file→function→completion | trace source GPU |
| M4 capability gap | source-backed Hopper/multi-GPU/NVLink matrix | none |
| M5 BookSim study | pinned SHA, API/build/timing notes | none |
| M6 transport smoke | logical GPU0→GPU1 bytes/latency/stats | none |
| M7 adapter | minimal Accel-Sim↔BookSim clock/data tests | none |
| M8 Ring | 2/4/8 logical ranks and message scaling | none |
| M9 NVSwitch semantics | topology/ports/flow-control assumptions tested | none |
| M10 NVLS | multicast/reduction/completion tests | none; synthetic |
| M11 correlation | NCCL Ring/NVLS and counters compared | H100 NVSwitch |

Each milestone needs input, changed files/classes, focused tests, Codex wall
time, user review time, risks, and a commit before advancing.
