---
name: ptx-design-baseline
description: 在加入 multi-GPU/NVLS semantics 前，为 proposed GPU instruction 或 microarchitecture feature 建立/扩展 PTX execution-driven path。
---

## 使用时机

用于 new PTX instruction、hypothetical GPU feature、custom scheduler，或无法由
真实 A800/H100 SASS trace 捕获的 CAIS/DySHARP-like design。它不能替代 A800
SASS smoke test 或 H100 hardware correlation。

## 输入

- current branch 与 main/nested repository SHA；
- PTX/CUDA workload 与 target config；
- semantic note：operands、result、ordering、latency、completion、可能的
  communication side effect；
- baseline PTX command 与 expected output；
- evidence label：`Verified`、`Synthetic`、`Hypothesis` 或
  `Needs hardware validation`。

## 步骤

1. 编辑前读 `rules/accel-sim-policy.md`、`docs/17-ptx-primary-strategy.md` 与
   relevant implementation note。
2. 运行 unmodified PTX baseline，记录 command、config、exit code、cycle/
   instruction summary 与 baseline failure。
3. mapping 当前 path：PTX lexer/parser → PTX IR → functional semantics → warp
   instruction → shader timing/issue → memory 或 completion。
4. 定义最小 instruction syntax/semantic contract；single-GPU path 测试前不得
   加 rank、packet、multicast 或 reduction state。
5. 只修改 named parser/opcode/IR/semantics/timing file；除非 design 明确要求，
   保留 SASS trace path。
6. 为 parser recognition、functional result、register dependency、issue latency、
   resource selection 与 completion 添加 focused test。
7. instruction 产生 communication event 时，使用 explicit schema 序列化；不要把
   collective state 隐藏在普通 memory address 中。
8. 运行 focused test、PTX baseline 与最小相关 regression。
9. 更新 implementation note、`project-state.md`、边界改变时的 `DECISIONS.md` 与
   `CHANGELOG.md`。
10. 执行 `git diff --check`，检查两个 repository status，并提交一个完整 phase。

## Completion criteria

PTX mode 能识别并执行 instruction；dependency/timing test 通过；baseline
behaviour 已分类；文档明确 PTX 能证明什么、仍需真实 SASS/H100 验证什么。
成功的 PTX run 不能单独证明 hardware accuracy。
