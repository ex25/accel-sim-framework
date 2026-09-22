# Research 文档索引

## 用途

本目录记录 simulator 为什么修改、当前 checkout 实际实现了什么，以及哪些
结论仍需 hardware validation。`.codex/knowledge/` 是快速 reload 入口；本索引
是完整阅读路径。文档默认使用中文，专业术语、symbol、API、config 与 metric
保留英文原文。

## 推荐阅读顺序

1. `项目进展总览.md` — 中文历史总览、当前 evidence boundary 和路线。
2. `14-final-goal-gap-plan.md` — 最终目标差距、环境上限与分阶段 exit criteria。
3. `15-concepts-for-readers.md` — GPU、trace、BookSim、NCCL、NVLS 基础概念。
4. `16-simulating-custom-multigpu-isa.md` — multi-GPU scheduling、custom ISA 与
   microarchitecture 修改的模拟方法。
5. `17-ptx-primary-strategy.md` — PTX 主线、SASS validation line 与 milestone。
6. `18-sass-ptx-a800-gap.md` — A800 single-GPU SASS/PTX gap 实验证据与限制。
7. `00-project-overview.md` — scope、terminology 与 staged strategy。
8. `01-accelsim-architecture.md` — 当前 simulator layer 与 call path。
9. `02-trace-pipeline.md` — NVBit → trace → frontend → timing model。
10. `03-hardware-capability-matrix.md` — A800/H100/NVSwitch validation boundary。
11. `04-single-vs-multi-gpu-trace.md` — 为什么 compute trace 不等于 distributed
    execution。
12. `05-h100-architecture-support.md` — SM90 support 的 source audit。
13. `06-nvlink-nvswitch-nvls.md` — NVLink、NVSwitch 与 NVLS 概念。
14. `07-current-support-gap-analysis.md` — capability matrix 与 evidence。
15. `08-booksim-architecture.md` — BookSim component 与 embedding options。
16. `09-accelsim-booksim-integration.md` — clock/data boundary alternatives。
17. `10-design-options.md` — PTX-primary architecture choice。
18. `11-implementation-roadmap.md` — PTX-first milestone exit criteria。
19. `12-experiment-plan.md` — PTX design 与 SASS validation experiment。
20. `13-validation-plan.md` — design evidence 与 hardware evidence ladders。

Implementation record 位于 `implementation/`；当前 PTX 相关记录是
`009-sm90-ptx-baseline.md` 与 `010-cais-add-ptx.md`。

## 当前导航状态

- Milestone：M3 SM90 PTX baseline 与 M4 `cais.add.u32` 已完成；下一步是
  instruction-to-communication-event boundary。
- Branch/base：`feat/nvbit-a800-booksim-nvlink` / upstream `d930ad6`。
- Baseline：release build 与真实 A800 vectoradd trace/replay 通过；CTest 仍无
  registered tests。
- Blocker：没有第二张 GPU/NVLink peer，也没有 H100/NVSwitch system。
- BookSim：外部 `28f4329` checkout 的 two-node AnyNet smoke/sweep 通过，但还
  没有 Accel-Sim adapter 或 clock coupling。
- Next：定义独立 communication-event schema 与 focused completion tests，
  再连接 two-endpoint BookSim prototype；暂不声称 Ring/NVSwitch/NVLS behaviour。

## Source-of-truth 映射

| 信息 | Primary location |
|---|---|
| 持久规则 | `.codex/rules/` |
| 当前 state/checklist | `.codex/knowledge/project-state.md` |
| 本机环境 | `.codex/knowledge/local-environment.md` |
| Trace facts | `.codex/knowledge/trace-format.md` |
| Architecture/research | `.codex/docs/` |
| Design decisions | `.codex/docs/DECISIONS.md` |
| Phase history | `.codex/docs/CHANGELOG.md` |
| Experiment metadata | `.codex/experiments/` |

## Skill 管理

Skill 只在 `.codex/skills/` 下编辑；`.agents/skills/` 必须是指向 canonical
directory 的 symlink，不维护两份内容。skill 定义 input、step、command、
artifact、validation 与 completion criteria；rules 定义稳定 policy。
