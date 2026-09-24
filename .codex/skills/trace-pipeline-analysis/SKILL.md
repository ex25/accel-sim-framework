---
name: trace-pipeline-analysis
description: 用 source evidence 把一条真实 Accel-Sim NVBit trace instruction 映射到 parsing、warp state、issue、memory 与 completion。
---

## 输入

一条 on-disk trace instruction、trace version/kernel metadata、当前 Accel-Sim/
GPGPU-Sim SHA 与 source checkout。

## Workflow

1. 找到 `util/tracer_nvbit/tracer_tool` 的 producer struct/writer，包括 field
   order、memory/address/register capture 与 versioning。
2. 确认 post-processing/compression，以及 `trace-parser` 实际消费的 record。
3. 沿 parser 进入 `trace_shader_core_ctx`、warp/instruction state、CTA/kernel
   ownership 与 scheduling。
4. Compute instruction 跟踪 issue、functional/timing unit、completion、stats；
   memory instruction 额外跟踪 coalescing、`mem_fetch`、cache、interconnect、
   memory partition、response 与 warp completion。
5. 检查 active mask、predicate、register、control flow、barrier、stream 与
   Hopper-specific fields，并明确缺失字段。
6. 引用 file/line evidence，绘制 record→parser→warp→issue→completion Mermaid。
7. 完整 autopsy 写入 `docs/gpu-execution.md`，compact facts 写入
   `knowledge/trace-format.md`。

优先使用真实 record；区分 verified/unverified edge。Ampere trace 不能证明
Hopper-specific trace field。
