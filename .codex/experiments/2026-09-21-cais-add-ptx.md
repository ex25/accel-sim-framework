# `cais.add.u32` single-GPU PTX extension

Date: 2026-09-21
Evidence level: **Verified** for the local CUDA 12.8 + nested GPGPU-Sim
software path; **Needs hardware validation** for any real GPU implementation.

## Objective and contract

Validate the smallest proposed PTX instruction before adding any communication
event or BookSim/NVSwitch behavior. The instruction is deliberately local:

```ptx
cais.add.u32 %r_dst, %r_src1, %r_src2;
```

It computes unsigned 32-bit modulo addition, writes one register, has no memory
or network side effect, and participates in normal register RAW dependencies.
The prototype reuses the configured integer `ADD` latency, initiation interval,
and `INTP_OP` execution resource. This is a proposed simulator contract, not a
claim about NVIDIA hardware or a CAIS production ISA.

## Pinned inputs

| Item | Value |
|---|---|
| Accel-Sim before phase | `b6c241d` (`test(ptx): establish SM90 execution-driven baseline`) |
| Nested GPGPU-Sim branch | `feat/cais-add-ptx` |
| Nested GPGPU-Sim commit | `8f3c35fa` (`feat(ptx): add single-GPU cais.add prototype`) |
| CUDA toolchain | CUDA 12.8.61, PTX 8.7, `.target sm_90` |
| Simulator config | `H100-PTX` / `SM90_H100/gpgpusim.config` |
| Local hardware | one A800 SM80; not used to execute this SM90 fixture |
| Raw result directory | `/tmp/accelsim-cais-add-final-20260921/` |

## Files changed

Nested GPGPU-Sim (separate Git checkout):

- `src/cuda-sim/ptx.l`: recognize the `cais.add` mnemonic.
- `src/cuda-sim/opcodes.def`: register `CAIS_ADD_OP` and dispatch it to
  `cais_add_impl`.
- `src/cuda-sim/instructions.cc`: implement modulo-32 register addition with
  no memory/communication effect.
- `src/cuda-sim/cuda-sim.cc`: map the opcode to the integer ADD timing/resource
  class.
- `libcuda/cuda_runtime_api.cc`: make `PTX_SIM_USE_PTX_FILE` work on the
  CUDA-6+ external-PTX loader path. Proposed opcodes cannot be sent through
  stock `ptxas`, so the synthetic override intentionally skips PTXInfo loading;
  a production extension needs matching assembler/occupancy metadata.

Main Accel-Sim repository:

- `experiments/cais_add_ptx/cais_add_ptx.cu`: deterministic dependent-chain
  workload and host checksum.
- `experiments/cais_add_ptx/make_custom_ptx.py`: generate control PTX and a
  fixture with exactly one `add.u32` rewritten to `cais.add.u32`.
- `experiments/cais_add_ptx/Makefile`: bounded `compute_90 -> compute_90`
  build and fixture inspection.
- `experiments/cais_add_ptx/compare_runs.py`: compare control/custom functional
  and timing counters and require iteration scaling.
- `util/job_launching/apps/define-cais-add-ptx.yml`: 4096-element cases with
  one and 32 dependent additions.

## Build and run

```bash
cd /home/wangjz/in-switch/accel-sim-framework
source /home/wangjz/in-switch/activate-accel-sim.sh
cmake --build gpu-simulator/build/release -j8
cmake --install gpu-simulator/build/release
make -C experiments/cais_add_ptx clean all inspect
```

The install step matters: it places the rebuilt `libcudart.so` under the path
used by `activate-accel-sim.sh`. The fixture build printed
`PTX_FIXTURE_PASS marker_count=1` and inspection showed control `add.u32` and
custom `cais.add.u32` in otherwise identical PTX.

Generate standard run directories with:

```bash
export CAIS_ADD_PTX_ROOT=$PWD/experiments/cais_add_ptx
python3 util/job_launching/run_simulations.py -B CAIS_ADD_PTX -C H100-PTX \
  -N cais-add-final -r /tmp/accelsim-cais-add-final-20260921 -n
```

For each generated directory, select the fixture explicitly and run the binary
from that directory so its `gpgpusim.config` is used:

```bash
source /home/wangjz/in-switch/activate-accel-sim.sh
export PTX_SIM_USE_PTX_FILE=1
export PTX_SIM_KERNELFILE=$CAIS_ADD_PTX_ROOT/build/control.ptx  # or custom.ptx
cd /tmp/accelsim-cais-add-final-20260921/<case>/H100-PTX
$CAIS_ADD_PTX_ROOT/build/cais_add_ptx 4096 1  > control-iter1.log
$CAIS_ADD_PTX_ROOT/build/cais_add_ptx 4096 32 > control-iter32.log
```

Use `custom.ptx` for the custom pair, then validate all four logs:

```bash
python3 $CAIS_ADD_PTX_ROOT/compare_runs.py \
  control-iter1.log custom-iter1.log control-iter32.log custom-iter32.log
```

## Results

The external-PTX marker appeared for both fixtures, so these are not the
earlier misleading runs that silently parsed embedded `add.u32`:

| Fixture | Iterations | Cycles | Instructions | IPC | Checksum | max error |
|---|---:|---:|---:|---:|---:|---:|
| control `add.u32` | 1 | 3765 | 114688 | 30.4616 | 7061451698760 | 0 |
| custom `cais.add.u32` | 1 | 3765 | 114688 | 30.4616 | 7061451698760 | 0 |
| control `add.u32` | 32 | 4293 | 622592 | 145.0249 | 19941060158848 | 0 |
| custom `cais.add.u32` | 32 | 4293 | 622592 | 145.0249 | 19941060158848 | 0 |

All four host checks printed `status=PASS`; the reusable checker printed:

```text
CAIS_ADD_TEST_PASS iter1_cycles=3765 iter32_cycles=4293 \
iter1_insn=114688 iter32_insn=622592
```

The unchanged SM90 PTX baseline was rerun after the nested modification:

```text
gpu_tot_sim_cycle = 5501
gpu_tot_sim_insn  = 1654784
gpu_tot_ipc       = 300.8151
RESULT test=sm90_baseline ... max_error=3.576278686523437e-07 ... status=PASS
```

## Reasonableness and boundaries

- **Verified:** lexer/parser acceptance, register functional semantics,
  dependent-chain completion, integer timing/resource mapping, and simulator
  completion for the synthetic opcode.
- **Reasonable for this milestone:** control/custom equality is expected because
  the prototype intentionally aliases ordinary integer ADD timing. The higher
  32-iteration count demonstrates that the dependent loop is not optimized away
  and that simulated work scales with the chain length.
- **Not established:** hardware latency, compiler scheduling, SASS encoding,
  occupancy effects, or communication behavior. No packet, rank, stream,
  multicast, reduction, NVLink, NVSwitch, NVLS, or BookSim event is emitted.
- **Known implementation boundary:** stock CUDA `ptxas` rejects
  `cais.add.u32`; the external-PTX harness skips PTXInfo/occupancy assembly for
  this proposed opcode. This is acceptable for the focused execution test but
  must be replaced by a real assembler/metadata path before production use.
- **Regression status:** release CMake build/install passed. CTest reports no
  registered tests (`No tests were found!!!`). Existing warnings remain: CUDA
  internal-header/pointer-cast warnings and missing optional `stubgen`.

## Next step

Define the separate PTX-instruction-to-communication-event schema and its
single-GPU completion contract. Keep the event side effect out of this opcode
until parser/functional/timing tests for the event boundary exist; do not yet
implement Ring, NVSwitch, or NVLS.
