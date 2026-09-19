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

## Commit cadence and logical scope

Commit once per completed logical phase, not once per file, command, or small
follow-up edit. A phase normally includes the implementation, focused tests,
closely related documentation, state/checklist updates, and its changelog
entry. For example, the A800 trace smoke test, its reproducibility notes, and
the corresponding environment update belong to one phase commit. A typo or a
single generated-SHA correction should normally be folded into the phase
commit rather than creating a new history entry.

One phase commit should answer one review question. Split only genuinely
independent work, such as an unrelated experiment or a separate simulator
feature. Before committing, batch all edits belonging to the phase and inspect
the complete diff. Do not use vague subjects such as `update`, `changes`,
`work`, `misc`, or bare `fix`.

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
the commit body must briefly include the following. A one-line typo-only edit
should be folded into the surrounding phase; standalone typo commits are
discouraged.

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
   if the milestone changed, and `.codex/docs/CHANGELOG.md` in the same phase
   change. Do not create a second commit merely to replace a pending SHA; when
   the branch is private, amend the phase commit or record the SHA in the next
   phase entry.
5. Commit the complete phase once, record its SHA and test result, and verify
   `git status` is clean (apart from intentionally preserved user work).

## History maintenance

Research branches are expected to be readable six months later. If a phase was
accidentally split into several local commits and has not been pushed or used
as another branch's base, squash those commits before handoff. Preserve the
logical phase boundary and rewrite only the project branch; never rewrite
`main`, `dev`, or a shared remote branch. The final squashed commit must retain
the detailed body above and the changelog must list the final SHA.

Never commit traces, build trees, binaries, simulator output, or large logs
unless a small fixture is explicitly required for reproducibility.
