# Accel-Sim 与 GPGPU-Sim policy

## Repository 与 build boundary

Accel-Sim 提供 launcher、NVBit tracer、trace post-processing、configs、
and trace-driven wrapper. The nested `gpu-simulator/gpgpu-sim` repository owns
the simulator core and has its own Git history. Do not describe a change in one
repository as a change in the other; record both SHAs when both are involved.

baseline build 使用 checkout 已记录的 environment，并将
the build directory outside source control:

```bash
source ./gpu-simulator/setup_environment.sh
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/release \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ./gpu-simulator/build/release -j"$(nproc)"
```

The expected simulator is
`gpu-simulator/bin/release/accel-sim.out` (or the equivalent configured build
path). If the checkout's CMake options differ, record the actual command.

## 两条 execution path 及其归属

在 design 与 documentation 中保持两条 path 分离：

* PTX/functional execution: CUDA/PTX is parsed and executed functionally by
  GPGPU-Sim (`src/cuda-sim`, `ptx_parser`, `ptx_thread_info`, and
  `gpgpu_sim::cycle`).
* SASS trace replay: NVBit writes a trace, post-processing creates the replay
  format, and Accel-Sim's trace frontend feeds `trace_shader_core_ctx` and the
  normal shader/memory timing model.

本 research project 以 PTX execution-driven mode 作为 proposed ISA、GPU
for proposed ISA, GPU microarchitecture, scheduler, and in-switch-computing
features. It lets the simulator execute a design that does not exist on a real
GPU. SASS remains the validation/calibration path for behaviour that exists on
A800/H100, including compiler register allocation, machine instruction order,
and real memory addresses.

不得从 PTX support 推断 trace support，不得从 H100 config 推断 PTX accuracy，
or multi-GPU support from a single-GPU trace configuration. A PTX instruction
extension still requires parser, functional semantics, timing/resource,
dependency, and (when applicable) network side-effect changes. PTX mode alone
does not provide rank, NCCL, NVSwitch, or NVLS semantics.

## Core source map

追踪 change 时从当前 checkout 的真实 symbol 开始：

* `gpgpu_sim` / `gpgpu_sim::cycle`: global simulation loop and kernel/stream
  scheduling;
* `shader_core_ctx`, `simt_core_cluster`, `shd_warp_t`: CTA/warp state and
  shader pipeline;
* `trace_shader_core_ctx`, `trace_warp_t`, `trace_parser`: trace-driven
  injection and instruction state;
* `mem_fetch`, cache classes, `memory_partition_unit`, DRAM, and
  `icnt_wrapper`: memory request path;
* `InterconnectInterface` and `src/intersim2`: the existing GPU-memory
  interconnect boundary, not an NVLink/NVSwitch model.

research note 使用 `rg` 与 source line reference。不要只依赖 class
names alone: follow one call path from parser to issue, memory completion, and
statistics.

## Change discipline

添加 proposed instruction 前先定义 operands、result、ordering、
latency, completion condition, and network side effects before editing code.
Prefer a small PTX-mode prototype or synthetic instruction/event generator.
Only add a SASS trace opcode when a real hardware trace exists or when the
synthetic trace format is explicitly part of the experiment and is labelled
`Synthetic`.

任何 trace-format 或 tracer change 都必须同步更新 producer、writer/parser、
version/compatibility notes, a small fixture or replay test, and the
source-to-consumer call-path document. Any timing-model change must state the
clock domain, queue ownership, and completion condition. Keep NVLink,
NVSwitch, and NVLS additions behind explicitly named layers rather than
silently changing the existing memory interconnect.
