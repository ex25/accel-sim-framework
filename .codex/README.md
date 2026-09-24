# Accel-Sim + BookSim NVLS 研究记忆

## 项目目标

项目目标分三层：先实现可复现的 DGX-H100 类 8-GPU + 4-NVSwitch + NVLS
simulator；再在其上运行 prefill/training workload，检查性能趋势和实现一致性；
最后扩展到 ACIS/CAIS、DySHARP 类 custom ISA 与 in-switch computing 设计复现。
首个系统目标是 2/4/8-GPU AllReduce，对比 Ring 与 NVLS。项目有两条明确分工的
输入路线：

1. **PTX-first design path**：用于现实硬件尚不存在的新 ISA、新 GPU
   microarchitecture、custom scheduling 与 in-switch computing design。
2. **SASS validation path**：使用 A800/H100 的真实 NVBit trace，检查已有 GPU
   behaviour、compiler effect 与 calibration assumption。

当前仓库是分阶段 research project，不能声称已经完整建模 NVSwitch 或 NVLS。

## 文档语言规范

`.codex/README.md`、`docs/`、`experiments/`、`knowledge/`、`rules/` 与
`skills/` 默认使用中文撰写。PTX、SASS、opcode、pipeline、trace、BookSim、
NVLink、NVSwitch、NVLS、API、class/function、config、metric、evidence label、
branch、commit 等专有名词保留原始英文，不做生硬直译。代码、命令、路径、
symbol 和原始输出必须保持原样。

## 当前状态

- Main repository：`accel-sim-framework`，branch
  `feat/ptx-sync-events-p0-p5`，基于 upstream `d930ad6`。
- Nested GPGPU-Sim 是独立 ignored Git checkout，当前在
  `feat/ptx-sync-events-p1-p5`；它不是 Accel-Sim submodule。
- Nested GPGPU-Sim 已含 BookSim2-derived `src/intersim2`，由
  `-network_mode 1` 选择，用于 GPU-to-memory interconnect。外部
  `../booksim2` checkout `28f4329` 只是 upstream/reference checkout。
- Release build 已通过；CTest 当前没有 registered tests。
- 本机只有一张 NVIDIA A800 80GB PCIe（Ampere/sm_80），没有 NVLink peer。
- NVBit 1.8 位于 `in-switch` conda environment。真实 A800 vectoradd trace、
  post-processing 与 SM80 replay 已通过。
- A800 single-GPU SASS/PTX gap matrix 已完成，包含 GEMV/GEMM。
- SM90 PTX execution-driven baseline 与最小 `cais.add.u32` custom instruction
  已完成。后者只有本地 register semantics，不产生 network side effect。
- Standalone two-endpoint BookSim AnyNet prototype 与 sensitivity sweep 已通过；
  这只是 generic transport evidence，不是 physical NVLink calibration。

## 新 session 启动方式

依次阅读本文件、`.codex/rules/` 和 `.codex/knowledge/project-state.md`，然后
检查 main 与 nested repository 的 Git status。实现前阅读对应 design note 与
skill。不得把 A800 或 synthetic BookSim result 推断成 H100/NVSwitch/NVLS
validation。

## 目录说明

- `rules/`：每个 session 必须遵守的稳定 policy。
- `skills/`：可复用 workflow；`.agents/skills/` 只保存指向 canonical skill
  的 symlink。
- `knowledge/`：高密度、快速加载的当前事实。
- `docs/`：按主题维护的 `roadmap.md`、`implementation.md`、概念/架构文档、
  `decisions.md` 与 `changelog.md`；不按 phase 自动新增文件。
- `experiments/`：实验 specification 和 reproducibility metadata；raw trace、
  log、checkpoint 与大结果保存在 `.codex/` 之外。

## 当前 milestone

M2 A800 trace pipeline、M3 SM90 PTX baseline、M4 `cais.add.u32` 和 P0–P5
communication/collective reference layers 已完成。下一阶段按
`.codex/docs/roadmap.md` 的 P6 开始 online event/collective integration；BookSim
integration、real H100 calibration 和 hardware validation 仍未完成。

## 关键决定

1. proposed ISA 与 microarchitecture 以 PTX execution-driven mode 为主线。
2. 真实 SASS trace 只作为已有 hardware 的 validation/calibration path。
3. compute trace 与 communication event 分离；复制 single-GPU trace 不能构成
   multi-GPU execution。
4. 在引入新的 BookSim copy 前，先复用并分析现有 `intersim2` boundary。
5. 保留 Accel-Sim 现有 SASS validation path；只有 research design 需要时才
   扩展 PTX frontend 或增加 synthetic instruction/event input。

详细 rationale 见 `.codex/docs/decisions.md`。

## 当前限制

当前 checkout 有已记录的 Hopper SM90/TMA/WGMMA/mbarrier/cluster support，但
没有完成 real H100 correlation，也没有证明 multi-GPU execution、NVSwitch、
NVLS 或 `multimem` timing semantics。README/release-note claim 不能替代源码与
实验 evidence。本机没有 H100 或 NVSwitch system。

## 下一项任务

执行 `.codex/docs/roadmap.md` 的 P6 online event/collective integration。P0–P5
reference layers 已有 focused tests，但不能替代真实 BookSim 或 H100/NVSwitch
validation。

## 详细入口

- Research index：`.codex/docs/README.md`
- 唯一计划：`.codex/docs/roadmap.md`
- 实现记录：`.codex/docs/implementation.md`
- 当前 state/checklist：`.codex/knowledge/project-state.md`
- 本机环境：`.codex/knowledge/local-environment.md`
- Trace facts：`.codex/knowledge/trace-format.md`
- GPU/trace：`.codex/docs/gpu-execution.md`
- Network/BookSim：`.codex/docs/network-model.md`
- Concepts/custom ISA：`.codex/docs/concepts.md`、`.codex/docs/custom-isa.md`
- Decision/history：`.codex/docs/decisions.md`、`.codex/docs/changelog.md`
