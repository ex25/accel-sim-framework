# 007 — NVBit A800 trace and BookSim two-GPU transport prototype

## Problem

The software trace pipeline had not been exercised on the available A800, and
the project needed a minimal network experiment before adding NVSwitch or NVLS.

## Design

Install NVBit 1.8 under the in-switch conda environment. Make the tracer
Makefile accept an explicit SDK core path so the repository needs no SDK copy
or machine-specific symlink. Use the SDK vectoradd as a real workload, then
post-process and replay it with the existing Accel-Sim release binary.

For the network, keep BookSim external at SHA 28f4329. Use AnyNet and
TrafficManager without forking. Model two logical GPUs as node 0/node 1, one
router per endpoint, and one bidirectional router link. Use bitcomp for
deterministic bidirectional traffic, plus smoke/sweep scripts.

## Alternatives

- Copy BookSim into Accel-Sim: rejected; it obscures provenance.
- Extend Accel-Sim intersim2 immediately: deferred until the endpoint/clock API
  is designed independently.
- Use uniform traffic: rejected because it does not guarantee both directions.

## Files changed

- util/tracer_nvbit/tracer_tool/Makefile: overridable NVBIT_PATH.
- tools/booksim_nvlink/nvlink2_config: two-endpoint configuration.
- tools/booksim_nvlink/nvlink2_network: AnyNet topology.
- tools/booksim_nvlink/run_nvlink2.sh: smoke test/status normalization.
- tools/booksim_nvlink/sweep_nvlink2.sh: parameter sweep and CSV.
- tools/booksim_nvlink/README.md: reproduction and interpretation.
- .codex trace/BookSim/environment/state documents: evidence and limits.

## Call paths

Trace:

vectoradd -> NVBit tracer_tool.so -> raw trace.xz ->
post-traces-processing -> tracez/kernelslist.g -> accel-sim.out ->
trace_parser -> shader/memory timing -> statistics.

BookSim:

run_nvlink2.sh -> BookSim main -> AnyNet -> TrafficManager::Run ->
TrafficManager::_Step -> Network ReadInputs/Evaluate/WriteOutputs ->
packet/flit statistics.

## Before / after

Before, NVBit setup was a blocker and no two-endpoint BookSim experiment
existed. After, the A800 software pipeline is verified and a reproducible
two-node transport prototype plus sensitivity sweep exists. No Accel-Sim
multi-GPU or calibrated NVLink model was added.

## Tests and results

- A800 vectoradd and post-processing: pass.
- Accel-Sim replay: exit 0; 6,824 cycles; 75,000 request/reply packets.
- BookSim smoke: NVLINK2_SMOKE_PASS; one-flit/link-latency-1 is 12 cycles and
  two hops.
- BookSim sweep: NVLINK2_SWEEP_PASS; all nine cases produced CSV output.
- git diff --check: required before commit.

## Limitations and validation status

Verified: A800 software trace pipeline and generic BookSim transport.
Inferred: the two-router topology is a useful first endpoint model.
Needs hardware validation: physical NVLink latency/bandwidth, H100 topology,
NVSwitch routing, multicast, NCCL, and NVLS. BookSim status 255 is a legacy
success convention normalized by the wrapper.

## Commit

- Implementation: `4cad91d` (`feat(transport): verify A800 trace and add
  two-GPU BookSim model`).
- Documentation record: the follow-up commit is listed in the research
  changelog.

## Next

Autopsy one real trace instruction, then define an endpoint adapter and clock
conversion before connecting BookSim transport to Accel-Sim.
