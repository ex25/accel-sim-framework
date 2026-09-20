# 009 — SM90 PTX execution-driven baseline

## Problem

The project had selected PTX as the design path, but had not yet demonstrated
that a real CUDA 12.8 `compute_90` PTX code object could pass the SM90/H100
configuration and had not recorded the parser-to-timing source path.

## Design

Add a tiny SM90 PTX-only CUDA workload. It has one deterministic kernel with a
global load/store, predicated branch, dependent FP32 FMA chain, host-side
checksum, and a bounded 4096×32 launch. The Makefile intentionally omits
compile-time `-lineinfo` because the current CUDA 12.8 PTX parser rejects the
emitted `.loc ... inlined_at` form observed in the SASS/PTX gap experiment.

The launcher uses the existing `H100-PTX` composition and
`SM90_H100/gpgpusim.config`; no simulator core or nested repository source is
modified.

## Files and symbols

- `experiments/sm90_ptx_baseline/sm90_ptx_baseline.cu`: workload and checksum.
- `experiments/sm90_ptx_baseline/Makefile`: `compute_90 -> compute_90` build.
- `util/job_launching/apps/define-sm90-ptx-baseline.yml`: one bounded launcher case.
- `.codex/experiments/2026-09-20-sm90-ptx-baseline.md`: commands, output,
  source call path, evidence labels, and limitations.
- PTX loader/parser: `gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_loader.cc:169–203`,
  `ptx.l`, `ptx.y`, and `ptx_parser.cc:102–144`.
- PTX IR and timing metadata:
  `cuda-sim/ptx_ir.cc:1211–1228` and `cuda-sim/cuda-sim.cc:747–805`.
- Functional dispatch: `cuda-sim/cuda-sim.cc:1793–1887`, `opcodes.def`,
  and `instructions.cc`.
- Issue/timing: `gpgpu-sim/shader.cc:1055–1072`, `2032–2065`, `2809–2830`,
  and `gpu-sim.cc:2234`.

## Before / after

Before, the project had only the SM80 A800 SASS/PTX comparison and a source
audit. After this phase, a real CUDA 12.8 SM90 PTX code object (`.target sm_90`,
PTX 8.7) is parsed and simulated by `H100-PTX`, producing a passing checksum,
functional completion, and timing counters. This proves the local simulator
software path only.

## Tests and results

```bash
make -C experiments/sm90_ptx_baseline clean all inspect
python3 util/job_launching/run_simulations.py \
  -B SM90_PTX_BASELINE -C H100-PTX \
  -N sm90-ptx-baseline -r /tmp/accelsim-sm90-ptx-baseline -n
```

Result: build/`cuobjdump` PASS; PTX simulation exit 0; 5,501 simulated cycles;
1,654,784 simulated instructions; IPC 300.8151; checksum max error
`3.5762786865234375e-07`; `RESULT ... status=PASS`; simulator exit detected.

Control: existing SM80 `vec_add 4096` under the same H100-PTX config also
passed at 3,739 cycles and IPC 23.0051. The control is not evidence of SM90
instruction support.

## Limitations and validation status

- **Verified:** local CUDA 12.8 `compute_90` PTX parsing, functional semantics,
  issue/timing, and completion on nested GPGPU-Sim `91880c53`.
- **Observed:** current H100 config is a simulator model with configured SM90
  parameters; it is not hardware calibration.
- **Needs hardware validation:** native H100/H200 execution, Hopper SASS,
  WGMMA/TMA/mbarrier/cluster accuracy, HBM/NVLink/NVSwitch/NVLS.
- **Unknown:** whether every CUDA 12.8 SM90 PTX feature used by future custom
  instructions is accepted; the baseline covers only the selected kernel.

## Next

Define one single-GPU custom PTX instruction without a network side effect.
Specify operands, result, ordering, latency, issue resource, register
dependency, and completion before touching communication events. Add focused
parser, functional, dependency, timing, and completion tests; then rerun this
baseline and the prior A800 SASS/PTX matrix.
