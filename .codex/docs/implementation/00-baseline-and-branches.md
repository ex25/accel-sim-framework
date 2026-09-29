# 阶段 0：基线与分支

状态: 已验收
日期: 2026-09-29
外层 commit: `907c0c8`
内层 commit: `91880c53`

## 目标

建立不混入既有实验提交的配对研究分支，并让后续模型能自动找到研究资料。

## 实际操作

- 外层只提交 `.gitignore` 的 `.serena/` 忽略规则。
- 外层创建并切换 `research/hopper-ptx`。
- 内层从本地 `dev` 创建并切换 `research/hopper-ptx`。
- 保留外层未跟踪 `experiments/`。
- 添加 `AGENTS.md`、`.codex/` 规则、skill、计划、证据和阶段模板。

## 验收

- `.gitignore` commit 只包含一个文件，commit message 为
  `chore: ignore Serena project metadata`。
- 两个仓库的当前分支都是 `research/hopper-ptx`。
- 内层没有带入 `feat/ptx-sync-events-p1-p5` 的两个实验提交。
- 文档入口和阶段门控规则已经存在。

## 下一阶段入口

阶段 1 只允许冻结 workload 和工具链，不修改模拟器代码。
