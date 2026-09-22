---
name: implement-feature
description: 实现一个有边界的 Accel-Sim/BookSim research feature，配套 design、focused test、evidence、documentation 与 explanatory Git commit。
---

## Gate

仅在 reconnaissance 找到 bounded gap 且 phase exit record 存在后使用。不得用此
workflow 直接开始完整 NVSwitch/NVLS implementation。

## Workflow

1. 写明 problem、motivation、current behaviour、alternative、chosen design、
   clock/ownership assumption 与 before/after call path。
2. 编辑前定义最小修改与 focused test，包括 baseline/regression command。
3. 只修改指定 file/class/function，保留无关 PTX、SASS trace 与 memory path；
   proposed PTX ISA/microarchitecture 要明确记录 SASS validation boundary。
4. 先跑 focused test，再跑相关 regression；失败分类为 `BASELINE FAILURE`、
   environment/dependency failure 或 branch regression。
5. 更新 implementation note、project-state、必要时的 DECISIONS、CHANGELOG 与
   compact knowledge。
6. 检查 `git diff --check`、`git diff --stat`、完整 diff 与全部 repository status。
7. 提交一个 logical unit，复读最终 diff，并记录 SHA。

## Required report

必须记录 what、why、architecture、files/symbol、call path、configuration、
before/after、tests/results、limitation、validation status、commit SHA 与 next step。
Commit body 也要简述 what/files/how/tests/limits。
