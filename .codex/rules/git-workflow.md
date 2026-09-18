# Git workflow

- Work on a research/feature/experiment branch, not `main`, `master`, or `dev`.
- Before changing files, record `git status`, branch, remotes, HEAD, and nested
  repository status. Preserve user modifications; never reset, clean, stash, or
  force-push without explicit authorization.
- One commit addresses one logical problem. Use messages such as
  `docs(accelsim): document trace pipeline` or
  `feat(booksim): add embedded network adapter`.
- Before each commit: run the relevant tests, inspect `git diff`, update docs,
  and record the commit SHA in `.codex/docs/CHANGELOG.md`.
- Keep build artifacts, traces, and large results out of Git unless explicitly
  required as a small reproducibility fixture.
