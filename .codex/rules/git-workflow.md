# Git workflow and commit records

## Repository boundaries

The Accel-Sim checkout, nested `gpu-simulator/gpgpu-sim`, and any external
BookSim checkout are separate repositories. Always identify the repository in
which a command is run and record each relevant SHA. Work on a
`research/*`, `feat/*`, `fix/*`, or `experiment/*` branch, never directly on
`main`, `master`, or `dev`.

Before editing, capture:

```text
git status --short --branch
git branch --show-current
git remote -v
git log -1 --oneline --decorate
git log --oneline --decorate -n 20
git submodule status
git -C gpu-simulator/gpgpu-sim status --short --branch   # when present
```

Preserve user modifications. Do not use `reset --hard`, `clean`, `checkout`,
`restore`, or `stash` to hide them, and never force-push, unless the user gives
explicit authorization for the exact target.

## One logical commit

One commit should answer one review question. Keep source, tests, and the
closely related documentation in the same commit when they describe one
change; split unrelated cleanup, formatting, and research phases. Do not use
vague subjects such as `update`, `changes`, `work`, `misc`, or bare `fix`.

Use an imperative subject with a useful scope, normally no more than 72
characters:

```text
docs(accelsim): document trace pipeline
test(trace): record A800 replay smoke test
feat(booksim): add embedded network adapter
fix(nvls): correct collective completion ordering
```

The upstream history is the style reference, not a requirement to copy issue
numbers. For example, `Accel-Sim 2.0: full NVIDIA Hopper (H100/H200) support`
describes a release-scale feature, `add the same range filter in tracer tool
to spinlock tool` names the concrete behaviour, and `[Fix] Fix potential
deadlock caused by the last EXIT instruction` names the failure mechanism.
Prefer the same specificity for this project.

## Commit body: what changed, where, and how

For every feature, bug fix, model change, or non-trivial documentation phase,
the commit body must briefly include the following. A one-line typo-only
commit may omit the body, but its subject still has to be specific.

```text
What:
  The capability or defect addressed.
Why:
  The research or correctness reason for changing it.
Files / symbols:
  Main files, classes, functions, configs, and scripts changed.
How:
  The implementation or documentation path, including important before/after
  behaviour.
Tests:
  Exact commands, baseline result, new result, and relevant output location.
Limits:
  Unsupported hardware, synthetic assumptions, or follow-up work.
```

The body need not be verbose, but it must let a reviewer answer “which files
changed, what functionality was added, and how was it implemented?” without
opening the chat transcript. The same information belongs in the appropriate
`.codex/docs/implementation/NNN-*.md` note when the change is a substantial
feature.

## Commit gate

Before committing:

1. Re-check status in the main and nested repositories.
2. Run `git diff --check` and inspect `git diff --stat` plus the full diff for
   accidental generated files, unrelated formatting, or user edits.
3. Run the focused test, then the smallest relevant regression. Record a
   pre-existing failure as `BASELINE FAILURE`, not as a regression.
4. Update the source-of-truth documentation, `.codex/knowledge/project-state.md`
   if the milestone changed, and `.codex/docs/CHANGELOG.md` with a pending
   commit entry if the SHA is not known yet.
5. Commit, record the resulting SHA and test result in `CHANGELOG.md`, and
   verify `git status` is clean (apart from intentionally preserved user work).

Never commit traces, build trees, binaries, simulator output, or large logs
unless a small fixture is explicitly required for reproducibility.
