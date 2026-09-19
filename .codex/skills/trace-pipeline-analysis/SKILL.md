---
name: trace-pipeline-analysis
description: Map one real Accel-Sim NVBit trace instruction through parsing, warp state, issue, memory, and completion with source evidence.
---

## Inputs

One on-disk trace instruction, trace version and kernel metadata, current
Accel-Sim/GPGPU-Sim SHAs, and the source checkout.

## Workflow

1. Identify the producer struct/writer in util/tracer_nvbit/tracer_tool,
   including inst_trace_t, callbacks, field order, memory/address/register
   capture, and versioning.
2. Identify post-processing/compression and the exact record consumed by
   trace-parser.
3. Follow parser functions into trace_shader_core_ctx, trace_warp_t,
   instruction/warp state, CTA/kernel ownership, and scheduling.
4. For compute, follow issue, functional/timing unit, completion, and stats.
   For memory, additionally follow address/coalescing, mem_fetch, cache,
   interconnect, memory partition, response, and warp completion.
5. Check active mask, predicate, registers, control flow, barriers, streams,
   and Hopper-specific fields. State absent fields explicitly.
6. Cite file and line evidence and draw a Mermaid flow from trace record to
   parser, warp, issue, memory/unit completion, and kernel statistics.
7. Store the full autopsy in .codex/docs/02-trace-pipeline.md and compact
   fields in .codex/knowledge/trace-format.md.

## Validation

Use a real record where possible, distinguish verified edges from unverified
edges, and include replay command/config. An Ampere trace is not proof of
Hopper-specific trace fields.
