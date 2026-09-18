# Research method and evidence labels

Work in this order unless a documented blocker requires a change:

```text
understand -> inspect history/source -> reproduce baseline -> identify gap
-> design alternatives -> smallest implementation -> focused test
-> regression -> document -> synthetic/analytical calibration
-> hardware validation
```

Read the checked-out source, Git history, build files, README/release notes,
and executable behaviour before making architectural claims. Treat README,
comments, issue summaries, and papers as claims to verify against code. For
external hardware or software facts, prefer NVIDIA official documentation,
PTX/NVBit/CUDA/NCCL source or documentation, BookSim source, and primary
papers; record URL/version/date in the research document.

Label every non-trivial conclusion:

* `Verified`: reproduced in this checkout or on the named hardware;
* `Observed`: recorded from an existing artifact but not independently rerun;
* `Inferred`: source-based interpretation with no direct runtime proof;
* `Hypothesis`: proposed design or expected behaviour;
* `Unknown`: searched but not established;
* `Needs hardware validation`: only the target H100/NVSwitch system can settle it.

Keep four claims separate: software trace-pipeline validation, synthetic
network/model sanity, public-data trend comparison, and real H100/NVSwitch
correlation. Capture commands, SHAs, configs, output summaries, assumptions,
and limitations in the repository; the chat is not evidence storage.
