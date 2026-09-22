# 项目状态（重新进入 session 时先读）

更新时间：2026-09-22

## 项目身份

- 目标：以 Accel-Sim + BookSim 构建 H100 NVLink/NVSwitch/NVLS research platform。
- Main repository/branch：`accel-sim-framework` /
  `feat/nvbit-a800-booksim-nvlink`。
- Nested GPGPU-Sim：独立 ignored checkout，branch `feat/cais-add-ptx`，commit
  `8f3c35fa`；不是 tracked submodule。
- BookSim reference checkout：`/home/wangjz/in-switch/booksim2`，`28f4329`，
  branch `master`；未集成。

## 本地环境

- 一张 NVIDIA A800 80GB PCIe，Ampere/sm_80；没有 NVLink peer。
- Driver 570.133.20；CUDA toolkit 12.8.61；`nvcc` 12.8.61。
- `in-switch` environment：Python 3.11.16、GCC/G++ 13.4.0、CMake 3.31.8。
- NVBit 1.8 位于 `$CONDA_PREFIX/opt/nvbit-1.8`；tracer 支持通过
  `NVBIT_PATH` 覆盖位置。
- GPU 运行实验时必须避免影响其他 process。

## Baseline evidence

- `cmake --build gpu-simulator/build/release -j8`：2026-09-21 通过。
- `ctest --test-dir gpu-simulator/build/release --output-on-failure`：没有发现
  tests；这不是完整 regression pass。
- 真实 A800 NVBit vectoradd trace、post-processing 与 SM80 replay：通过。
- Single-A800 SASS/PTX gap matrix：12 个有界 case，包含 GEMV/GEMM；native、
  trace、SASS replay 和 PTX replay 全部通过。PTX-vs-SASS cycle delta 为
  −2.6% 至 +19.0%；详见 `.codex/experiments/2026-09-20-sass-ptx-a800-gap.md`。
- SM90 PTX baseline：CUDA 12.8 `compute_90` PTX-only kernel 在 `H100-PTX` /
  `SM90_H100` 上通过，5501 cycles、1654784 instructions、IPC 300.8151，
  `max_error=3.5762786865234375e-07`。这是 **Verified** simulator software
  evidence，不是 H100 hardware validation。
- `cais.add.u32`：4096-element、1/32 iteration dependent chain 的 parser、
  functional、dependency、timing 与 completion 通过；control/custom 的
  cycle、instruction、IPC、checksum 完全一致；详见
  `.codex/experiments/2026-09-21-cais-add-ptx.md`。
- Standalone BookSim two-endpoint smoke/sweep：在 `28f4329` 通过；只证明
  generic transport，不证明 physical NVLink。

## 已确认的 execution strategy

- proposed ISA、microarchitecture、scheduler 和 in-switch-computing 以 PTX
  execution-driven mode 为主线。
- SASS/NVBit trace 用于已有 A800/H100 behaviour、compiler effect 与 hardware
  calibration。
- PTX 不自动提供 multi-GPU rank/collective/NVSwitch semantics；这些需要
  event schema、orchestrator、semantic layer 和 BookSim transport。
- hardware 不存在的 custom instruction 必须从 PTX 或明确标记的 synthetic
  instruction/event generator 进入，不能伪装成真实 SASS trace。

## 状态清单

- [x] repository audit 与 research branch
- [x] `.codex/` infrastructure
- [x] A800 environment capture
- [x] release build attempt
- [x] A800 real trace smoke test
- [x] trace pipeline/source mapping
- [x] Hopper/H100 source audit（hardware correlation pending）
- [x] single-vs-multi-GPU trace boundary
- [x] BookSim-derived `intersim2` boundary
- [x] BookSim architecture 与 dependency decision
- [x] clock/data coupling decision
- [x] multi-GPU architecture design
- [x] synthetic transport prototype
- [x] PTX-primary/SASS-validation strategy
- [x] single-A800 SASS/PTX frontend gap matrix（GEMV/GEMM）
- [x] SM90 PTX execution-driven baseline
- [x] minimal custom PTX instruction
- [ ] PTX instruction-to-communication-event boundary
- [ ] Ring baseline
- [ ] NVSwitch model
- [ ] NVLS semantics prototype
- [ ] compute + communication integration
- [ ] statistics/experiment automation
- [ ] real H100 validation

## 当前 blocker 与下一步

1. A800 vectoradd source 在外部 NVBit SDK；历史 replay command 未完整保存。
2. CUDA 12.8 PTX parser 对 `-lineinfo` 生成的 `.loc ... inlined_at` 仍存在兼容
   性限制；trace workflow 使用 `TRACE_LINEINFO=0`。
3. 先定义独立的 PTX instruction-to-communication-event schema 与 completion
   contract，并为它添加 focused tests。
4. 保留 `cais.add.u32` 作为无副作用 control；不要在 event boundary 之前加入
   network side effect。
5. event boundary 通过后，再定义 logical rank/stream orchestrator 与
   BookSim endpoint facade；最后才进入 Ring/NVSwitch/NVLS。

## Evidence 规则

本文件中的事实必须是直接 observation 或链接到完整记录，并使用
`Verified`、`Observed`、`Inferred`、`Hypothesis`、`Unknown`、
`Needs hardware validation` 等 evidence label。每个 milestone 都要同步更新
本文件并与对应 phase 一起提交。
