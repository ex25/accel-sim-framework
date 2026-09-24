# Git workflow 与 commit record

## Repository boundary

Accel-Sim checkout、nested `gpu-simulator/gpgpu-sim` 与 external BookSim 是独立
repository。每个 command 都要明确在哪个 repository 运行，并记录 relevant
SHA。只能在 `research/*`、`feat/*`、`fix/*`、`experiment/*` branch 工作，不得
直接修改 `main`、`master` 或 `dev`。

编辑前记录：

```text
git status --short --branch
git branch --show-current
git remote -v
git log -1 --oneline --decorate
git log --oneline --decorate -n 20
git submodule status
git -C gpu-simulator/gpgpu-sim status --short --branch
```

保留用户修改。未经用户对精确目标授权，不得用 `reset --hard`、`clean`、
`checkout`、`restore`、`stash` 隐藏修改，也不得 force-push。

## Commit cadence 与 logical scope

每个完成的 logical phase 提交一次，不按文件、command 或小修复拆分。一个
phase 通常包含 implementation、focused tests、相关 documentation、state/
checklist 和 `docs/changelog.md`。独立 typo 或 generated-SHA 修正应 fold 到 phase commit。

一个 phase commit 应回答一个 review question。只有真正独立的 experiment 或
simulator feature 才拆分。Commit 前批量检查完整 diff。禁止 `update`、
`changes`、`work`、`misc` 或裸 `fix` 等模糊 subject。

Subject 使用有 scope 的 imperative form，通常不超过 72 characters：

```text
docs(accelsim): document trace pipeline
test(trace): record A800 replay smoke test
feat(booksim): add embedded network adapter
fix(nvls): correct collective completion ordering
```

## Commit body

feature、bug fix、model change 或非平凡 documentation phase 的 body 至少包含：

```text
What: 解决的 capability 或 defect
Why: research/correctness 原因
Files / symbols: 主要 file、class、function、config、script
How: implementation/documentation path 与重要 before/after behaviour
Tests: exact command、baseline、新结果、output location
Limits: unsupported hardware、synthetic assumption、follow-up
```

读者不打开 chat transcript 也应能回答“改了哪些文件、增加了什么、如何实现”。
重大 feature 的相同信息还要进入 `.codex/docs/implementation.md` 对应章节。

## Commit gate

1. 重新检查 main 与 nested repository status。
2. 运行 `git diff --check`，检查 `git diff --stat` 与完整 diff，排除 generated
   file、无关 formatting 和用户修改。
3. 运行 focused test 与最小 regression；pre-existing failure 标为
   `BASELINE FAILURE`。
4. 同一 phase 更新 source-of-truth、`project-state.md`（milestone 改变时）与
   `docs/changelog.md`。不要为了 pending SHA 创建第二个 commit；private branch 可
   amend phase commit，或在下一 phase 记录。
5. 提交完整 phase，记录 SHA/test result，并确认 working tree clean（明确保留的
   用户修改除外）。

## History maintenance

Research branch 应在六个月后仍可阅读。若 phase 被错误拆为多个 local commit，
且尚未 push 或成为其他 branch base，可在 handoff 前 squash。只重写 project
branch，不得重写 `main`、`dev` 或 shared remote branch。

禁止提交 trace、build tree、binary、simulator output 或大型 log；只有明确用于
reproducibility 的小 fixture 可以例外。
