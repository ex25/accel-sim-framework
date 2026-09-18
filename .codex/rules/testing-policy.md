# Testing and evidence policy

Every core change has a focused test and the smallest relevant regression.
Documentation-only changes still run structural checks (links or file
existence where applicable) and `git diff --check`.

## Baseline first

Before changing behaviour, capture the baseline command, SHA, configuration,
exit code, wall time when meaningful, and a short output summary. If the same
command fails before the change, label it `BASELINE FAILURE`; distinguish it
from an environment failure (missing CUDA/NVBit/GPU/dependency) and a genuine
branch regression. Never silently turn a missing test suite into a pass.

For simulations record workload/trace, GPU model or synthetic topology, config
file, random seed, main and nested repository SHAs, output directory, and
evidence level. Keep large outputs outside Git and store only metadata and a
small fixture when needed.

## Validation levels

Use explicit labels: `unit`, `replay smoke`, `analytical`,
`synthetic network`, `public-data comparison`, or `real hardware`. A BookSim
packet-delivery test can establish transport correctness; it cannot establish
NVSwitch fidelity. An A800 trace replay can establish software-pipeline
compatibility; it cannot establish Hopper SASS or NVLS behaviour.

## Test report

Every implementation note and commit body should state the command, expected
result, actual result, failure classification, and limitation. For timing
changes include bytes/flits, latency/completion ordering, and counter checks;
for trace changes include trace version, kernel count, and parser/replay
result.
