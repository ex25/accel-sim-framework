# Documentation 与 source-of-truth policy

`Context is temporary; the repository is persistent.` 任何会影响未来工作的 fact、
call path、model assumption、hardware parameter、test procedure、decision、
limitation 或 next step，都必须在 phase 结束前记录到 `.codex/`。

## 语言规范

`.codex` 的解释性正文默认使用中文。PTX、SASS、opcode、pipeline、trace、
BookSim、NVLink、NVSwitch、NVLS、API、class/function、config、metric、evidence
label、branch、commit 等专有名词保留英文；代码块、命令、路径、symbol 与原始
输出保持原样。优先保证中文语义自然，不做生硬逐词翻译。

## Source-of-truth 路由

| Location | 用途 | 要求 |
|---|---|---|
| `.codex/rules/` | 每个 session 遵守的 stable policy | normative、actionable、很少变化 |
| `.codex/skills/` | repeatable workflow | input、step、command、artifact、validation、exit criteria |
| `.codex/knowledge/` | 快速加载的 current facts | compact、structured、包含 evidence label 与当前 SHA/state |
| `.codex/docs/` | 面向人的 research record | rationale、call path、alternative、diagram、citation |
| `.codex/experiments/` | reproducible experiment entrypoint | metadata、command、run index；raw output 放在外部 |
| `AGENTS.md` | thin bootstrap pointer | 不复制 rules |

改变未来工作的 implementation 必须在同一 logical phase 更新相关 knowledge 与
implementation note。重要 architecture choice 写入 `DECISIONS.md`，chronology
写入 `CHANGELOG.md`。优先链接 source of truth，避免复制出多份漂移 summary。

所有 claim 旁应使用 `Verified`、`Observed`、`Inferred`、`Hypothesis`、
`Unknown`、`Needs hardware validation` 等 evidence label。复现需要时必须记录
exact file、class/function、command、SHA 与 date。不得把 terminal dump、binary、
multi-GB trace 或临时想法放入 `.codex/`。
