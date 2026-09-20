# SM90 PTX execution-driven baseline

Date: 2026-09-20
Evidence level: **Verified** for the local unmodified GPGPU-Sim software path;
**Needs hardware validation** for real SM90/H100 behaviour.

## Objective

Establish the smallest reproducible PTX execution-driven baseline on the
repository's SM90/H100 configuration before adding a proposed instruction or a
communication side effect. This is a simulator baseline, not a claim that the
local A800 executes SM90 code.

## Pinned inputs

| Item | Value |
|---|---|
| Accel-Sim checkout | `6732dae12d5749b8342a4608aea825896a262a42` before this phase commit |
| Nested GPGPU-Sim | `91880c53383d5a6a6742bfb1be2c5f34e39c7871`, branch `dev` |
| Config | `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config` |
| Launcher config | `H100-PTX` = `SM90_H100` + `PTX` |
| PTX toolchain | CUDA 12.8.61, PTX `.version 8.7`, `.target sm_90` |
| Local hardware | one A800 SM80; not used to execute the SM90 binary |
| Raw output | `/tmp/accelsim-sm90-ptx-baseline/` |

## Workload and commands

The workload is `experiments/sm90_ptx_baseline/sm90_ptx_baseline.cu`: one
deterministic FP32 kernel with global load/store, predicated branch, dependent
`fma`, and a fixed 4096-element/32-iteration launch. It is compiled as PTX
only for SM90, with no compile-time `-lineinfo`:

```bash
source /home/wangjz/in-switch/activate-accel-sim.sh
make -C experiments/sm90_ptx_baseline clean all inspect
export SM90_PTX_BASELINE_ROOT=$PWD/experiments/sm90_ptx_baseline
python3 util/job_launching/run_simulations.py \
  -B SM90_PTX_BASELINE -C H100-PTX \
  -N sm90-ptx-baseline -r /tmp/accelsim-sm90-ptx-baseline -n
```

The generated run directory was executed locally with the generated
GPGPU-Sim `libcudart.so` on `LD_LIBRARY_PATH`; the final simulator output is
`/tmp/accelsim-sm90-ptx-baseline/sm90_ptx_baseline/4096_32/H100-PTX/direct-run.log`.

Before adding the SM90-only workload, an unmodified `SM90_H100`/`H100-PTX`
run of the existing SM80 `vec_add 4096` case also passed: 3,739 cycles,
86,016 simulated instructions, IPC 23.0051, and `RESULT ... status=PASS`.
This control is **Observed** and is not the SM90 instruction baseline.

## Result

The SM90 PTX binary was accepted and functionally/timing simulated by the
unmodified PTX path:

```text
gpu_tot_sim_cycle = 5501
gpu_tot_sim_insn  = 1654784
gpu_tot_ipc       = 300.8151
RESULT test=sm90_baseline size=4096 aux=32 checksum=3.708071458181e+02 \
  max_error=3.576278686523e-07 kernel_ms=3000.000000 status=PASS
GPGPU-Sim: *** exit detected ***
```

The simulator config printed compute capability major/minor `9/0`. This is
**Verified** parser, functional execution, issue/timing, and completion
evidence for this checkout. The `kernel_ms` value is simulated event time; it
must not be compared with A800 native event time.

## PTX-to-timing call path

The current path is:

```text
embedded PTX from CUDA fatbin
  → ptx_loader.cc:gpgpu_ptx_sim_load_ptx_from_string (169–203)
  → ptx.l lexer / ptx.y grammar
  → ptx_instruction constructor (ptx_ir.cc:1211–1228)
  → set_opcode_and_latency (cuda-sim.cc:747–805, FMA at 943–969)
  → shader_core_ctx::issue_warp (shader.cc:1055–1072)
  → exec_shader_core_ctx::func_exec_inst / execute_warp_inst
  → ptx_thread_info::ptx_exec_inst (cuda-sim.cc:1793–1887)
  → opcodes.def dispatch and instructions.cc implementation
  → functional memory result / register dependency
  → shader_core_ctx::execute and pipelined_simd_unit::cycle
  → gpgpu_sim::cycle (gpu-sim.cc:2234 onward)
```

Important boundaries:

- `ptx_loader.cc` calls the reentrant lexer/parser and aborts on parser errors;
  parser recognition is not hardware validation.
- `ptx_instruction::set_opcode_and_latency` maps PTX opcode and type to
  `SP_OP`, `LOAD_OP`, etc. and reads latency/initiation values from the config.
- `ptx_thread_info::ptx_exec_inst` executes one lane's functional semantics,
  tracks memory address/space, updates the PC and PTX instruction statistics,
  then returns memory metadata to the warp instruction.
- `shader_core_ctx::issue_warp` stamps dynamic issue state and invokes the
  functional execution; the timing pipeline later applies result-bus and
  functional-unit latency. This is why a new instruction needs parser, IR,
  functional, resource, dependency, and completion work.

## Baseline failure classification and limits

- The earlier CUDA 12.8 `-lineinfo` failure (`.loc ... inlined_at`) remains a
  **BASELINE FAILURE** of parser compatibility and is avoided here by omitting
  `-lineinfo` and setting `TRACE_LINEINFO=0` for trace workflows.
- Native execution of the SM90-only binary on the local A800 is intentionally
  not attempted: the A800 is SM80. H100/H200 hardware correlation is **Needs
  hardware validation**.
- No simulator core files or nested GPGPU-Sim files were changed in this
  phase. No rank, packet, multicast, reduction, NVSwitch, NVLS, or BookSim
  state is hidden in this single-GPU baseline.

## Exit decision

The SM90 PTX parser→functional→timing baseline is complete. The smallest next
implementation is a single-GPU custom PTX instruction with explicit operands,
result, ordering, latency, dependency, and completion semantics; its network
event must remain a separate schema and must not be added until this focused
instruction path is tested.
