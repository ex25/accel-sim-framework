# Research 变更日志

## 2026-09-24 — 完成 PTX-to-communication event schema（P0）

- 新增 `CommunicationEvent`、`EventCompletion`、`LogicalGPUContext` 与
  `CommunicationEventOrchestrator`，定义 GPU event 的 identity、dependency、
  ready/inject/complete lifecycle 和 global picosecond timestamp。
- 新增 4 个 P0 focused unit tests；既有 3 个 transport contract tests 继续通过。
- 本阶段没有修改 PTX parser、nested GPGPU-Sim、SASS trace format、BookSim 或
  GPU-memory `intersim2`。这是 **Verified / unit** schema evidence，不是
  NCCL/NVSwitch/NVLS 或真实 H100 evidence。

## 2026-09-22 — 统一 `.codex` 文档语言

- 将 README、docs、experiments、knowledge、rules 与 skills 中的解释性正文统一
  为中文；PTX、SASS、BookSim、NVLink、NVSwitch、NVLS、API、symbol、路径、
  command、metric 和 evidence label 保留英文。
- 重写长期入口、project state、experiment index、implementation index、
  DECISIONS、CHANGELOG、trace facts 与主要 experiment/implementation records，
  删除重复的英文摘要并保留 SHA、command、result、evidence 与 limitation。
- 增加 D008 与 documentation policy 的语言规范，后续 document-change 默认使用
  中文，不翻译 code/path/raw output。
- Validation：`git diff --check` 通过；本阶段没有修改 simulator code，也没有
  产生新的 hardware/accuracy claim。

## 2026-09-21 — 完成最小 single-GPU custom PTX instruction

- Nested GPGPU-Sim commit `8f3c35fa` 新增 `cais.add.u32`：lexer/opcode
  registration、unsigned modulo-32 functional execution，以及 integer ADD
  timing/resource mapping。
- 修复 CUDA 6+ external-PTX loader，使 `PTX_SIM_USE_PTX_FILE` 真正选择
  synthetic proposed opcode。stock `ptxas` 无法 assemble `cais.add`，因此
  focused override 暂时跳过 PTXInfo/occupancy assembly；这是明确的 harness
  limitation。
- Main repository 新增 dependent-chain workload、control/custom PTX fixture
  generator、1/32-iteration launcher case 和 reusable result checker。
- Control/custom 两组结果的 cycles、instructions、IPC、checksum 和 zero
  error 完全一致；修改后 SM90 baseline 仍为 5501 cycles、1654784
  instructions、IPC 300.8151、PASS。
- 限制：无 memory/network effect、无真实 GPU/SASS/occupancy/multi-GPU/
  NVLink/NVSwitch/NVLS/BookSim evidence；CTest 仍没有 registered tests。

## 2026-09-20 — 完成 SM90 PTX execution-driven baseline

- 新增有界 CUDA 12.8 `compute_90` PTX-only workload，并使用 `H100-PTX` /
  `SM90_H100/gpgpusim.config`。
- 未修改的 nested GPGPU-Sim PTX path 通过 parser、functional checksum、
  issue/timing 与 simulator completion：5501 cycles、1654784 instructions、
  IPC 300.8151、max error `3.5762786865234375e-07`。
- 记录 `ptx_loader`/lexer/parser、PTX IR、functional dispatch、shader issue
  和 timing pipeline 的 source call path。
- 限制：这是 SM90 simulator evidence，不是 H100/H200 hardware validation。

## 2026-09-20 — 完成 single-A800 SASS/PTX gap matrix

- 新增 dual-code-object SM80 CUDA workload suite，覆盖 FMA、memory、control
  flow、atomic、GEMV 和 GEMM；新增 metric collector 与 `SASS_PTX_GAP` launcher。
- 12 个真实 A800 NVBit trace、SASS replay 与 PTX replay 全部通过 functional
  check。周期差 −2.6% 至 +19.0%；GEMV/GEMM 的 instruction/IPC delta 仍明显。
- 记录 CUDA 12.8 `-lineinfo` 导致 PTX `.loc/inlined_at` abort 的 parser
  compatibility baseline failure；通过移除 compile-time `-lineinfo` 与设置
  `TRACE_LINEINFO=0` 复跑成功。
- 限制：仅 single A800；不涉及 multi-GPU、NVLink/NVSwitch/NVLS、H100 或
  absolute calibration。

## 2026-09-20 — 确认 PTX-primary design strategy

- 决定 proposed ISA、GPU microarchitecture、scheduler 和 in-switch-computing
  使用 PTX execution-driven mode；SASS/NVBit trace 作为已有 hardware 的
  validation/calibration path。
- 更新 design option、roadmap、experiment/validation plan、project state 和
  research scope。
- 原因：真实 hardware 无法产生 proposed instruction 的 SASS trace；PTX 可
  直接表达 functional semantics 与 timing，SASS 保留 compiler/machine-level
  validation。

## 2026-09-19 — 完成 trace/transport analysis 与 project history

- 完成真实 A800 `LDG.E.64` 的 producer→parser→warp→memory coalescing→GPU
  memory interconnect→completion source mapping。
- 确认 nested GPGPU-Sim 现有 `intersim2` 是 GPU-to-memory transport，而不是
  GPU-to-GPU NVLink/NVSwitch model。
- 完成 BookSim AnyNet two-endpoint smoke/sweep 与 endpoint/clock reference
  contract；这些是 generic/synthetic transport evidence。
- 建立 `.codex` rules、skills、knowledge、docs 与 experiments 的 source-of-
  truth 结构。

## 2026-09-19 — 完成 H100/Hopper P2 source audit

- 审计 binary-version dispatch、Hopper opcode map、GMMA latency mapping、TMA
  metadata、mbarrier/SYNCS、cluster fields 与明确的 TODO/unsupported path。
- 结论保持保守：SM90 source support 不等于 multi-GPU、NVSwitch、NVLS 或
  `multimem` timing support。

## 2026-09-19 — 完成 A800/NVBit trace pipeline

- 在单张 A800 上完成 CUDA → NVBit → raw trace → `.tracez` → Accel-Sim SM80
  replay；software pipeline 通过。
- 外部 BookSim checkout 完成 standalone two-node transport smoke/sweep。
- 限制：本地没有第二张 GPU/NVLink peer；没有 H100/NVSwitch hardware
  correlation。

## 语言与记录约定

后续 `.codex` 文档默认使用中文；PTX、SASS、BookSim、NVLink、NVSwitch、NVLS、
API、symbol、路径、命令、metric 和 evidence label 保留英文，以避免术语含义
漂移。每个 phase 应将代码、测试、state、implementation note 与本日志一起
提交。
