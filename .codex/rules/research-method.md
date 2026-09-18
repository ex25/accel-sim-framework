# Research method

- Inspect the checked-out source, Git history, build files, and executable
  behaviour before making architectural claims.
- Label evidence: `Verified` means reproduced in this checkout or on the stated
  hardware; `Observed` means recorded but not independently reproduced in this
  session; `Inferred` means a source-based interpretation; `Hypothesis` means a
  design proposal; `Unknown` means no evidence yet; `Needs hardware validation`
  means only a real target system can settle it.
- Treat README, release notes, and comments as claims to verify against code.
- Separate software-pipeline validation, synthetic network sanity checks,
  public-data trend comparison, and real H100/NVSwitch correlation.
- Capture commands, SHAs, configs, outputs, and limitations in the repository;
  do not rely on chat context.
