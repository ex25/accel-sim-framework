---
name: trace-pipeline-analysis
description: Map an Accel-Sim NVBit trace from an on-disk instruction through parsing, warp state, shader timing, memory, and completion.
---

Start from a real trace line or tracez record. Identify its producer struct and
writer, parser function, internal instruction class, warp/CTA owner, issue and
execution path, memory request path when applicable, and completion/statistics
consumer. Cite file paths and line numbers. Draw a Mermaid call graph and label
unverified edges. Keep the result in `.codex/docs/02-trace-pipeline.md` and a
short fact summary in `.codex/knowledge/trace-format.md`.
