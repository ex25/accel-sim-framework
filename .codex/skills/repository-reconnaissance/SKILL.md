---
name: repository-reconnaissance
description: 在 design/implementation 前 audit Accel-Sim、nested GPGPU-Sim、BookSim、Git history、dependency 与 local change。
---

## 使用时机

新 research phase、feature design、dependency decision 或 repository state
不确定时使用；不修改 simulator core。

## 步骤

1. 对每个 repository 记录 status、branch、remote、HEAD、recent history、
   submodule、diff stat、ignored build path 与 untracked file。
2. 读取 `AGENTS.md`、`.claude`、`.codex`、`.agents`、build file、
   `gpu-simulator`、`util/tracer_nvbit`、trace frontend 与 BookSim。
3. 独立确认 nested GPGPU-Sim 与 external BookSim SHA/status，注明 submodule、
   nested repository、symlink 或 checkout 类型。
4. 建立 launcher、NVBit producer、post-processor、parser、trace shader core、
   shader/memory path、interconnect、config、test 与 statistics source map。
5. 用 `git show --stat --format=fuller` 阅读代表性 commit，参考其具体 scope。
6. 使用 evidence label 标注结论，并将 compact state/可复用事实写入 `.codex`。
7. 运行 `git diff --check`，重新检查所有 repository status。

不得仅根据 README 声称 dependency 或 feature 已存在。
