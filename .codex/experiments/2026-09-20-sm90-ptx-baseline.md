# SM90 PTX execution-driven baseline

日期：2026-09-20

Evidence level：本地未修改 GPGPU-Sim software path 为 **Verified**；真实
SM90/H100 behaviour 为 **Needs hardware validation**。

## 目标

在加入 proposed instruction 或 communication side effect 前，先在仓库的
SM90/H100 configuration 上建立最小可复现 PTX execution-driven baseline。
这是 simulator baseline，不表示本地 A800 可以执行 SM90 code。

## 固定输入

| 项目 | 值 |
|---|---|
| Accel-Sim checkout | `6732dae12d5749b8342a4608aea825896a262a42` |
| Nested GPGPU-Sim | `91880c53383d5a6a6742bfb1be2c5f34e39c7871`，branch `dev` |
| Config | `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config` |
| Launcher | `H100-PTX` = `SM90_H100` + `PTX` |
| PTX toolchain | CUDA 12.8.61，PTX `.version 8.7`，`.target sm_90` |
| Local hardware | 一张 A800 SM80；不执行该 SM90 binary |
| Raw output | `/tmp/accelsim-sm90-ptx-baseline/` |

## Workload 与命令

`experiments/sm90_ptx_baseline/sm90_ptx_baseline.cu` 是 deterministic FP32
kernel，包含 global load/store、predicated branch、dependent `fma`，固定为
4096 elements/32 iterations。编译为 SM90 PTX-only，并省略 compile-time
`-lineinfo`：

```bash
source /home/wangjz/in-switch/activate-accel-sim.sh
make -C experiments/sm90_ptx_baseline clean all inspect
export SM90_PTX_BASELINE_ROOT=$PWD/experiments/sm90_ptx_baseline
python3 util/job_launching/run_simulations.py \
  -B SM90_PTX_BASELINE -C H100-PTX \
  -N sm90-ptx-baseline -r /tmp/accelsim-sm90-ptx-baseline -n
```

生成的 run directory 使用 simulator `libcudart.so` 完成执行；最终 output
在 `/tmp/accelsim-sm90-ptx-baseline/sm90_ptx_baseline/4096_32/H100-PTX/`。

## 结果

```text
gpu_tot_sim_cycle = 5501
gpu_tot_sim_insn  = 1654784
gpu_tot_ipc       = 300.8151
RESULT test=sm90_baseline size=4096 aux=32 checksum=3.708071458181e+02 \
  max_error=3.576278686523437e-07 kernel_ms=3000.000000 status=PASS
```

SM90 PTX binary 被接受，并完成 functional/timing simulation。Simulator config
打印 compute capability `9/0`。这证明当前 checkout 的 parser、functional
execution、issue/timing 和 completion path，不证明 H100 hardware accuracy。

## PTX 到 timing 的调用路径

```text
embedded PTX from CUDA fatbin
  → ptx_loader.cc:gpgpu_ptx_sim_load_ptx_from_string
  → ptx.l lexer / ptx.y grammar
  → ptx_instruction constructor
  → set_opcode_and_latency
  → shader_core_ctx::issue_warp
  → func_exec_inst / execute_warp_inst
  → opcodes.def dispatch 与 instructions.cc implementation
  → register/memory result 与 dependency
  → shader execute / pipelined_simd_unit::cycle
  → gpgpu_sim::cycle
```

`ptx_loader.cc` 负责 parser；`ptx_instruction::set_opcode_and_latency` 根据
opcode/type 选择 `SP_OP`、`LOAD_OP` 等 resource，并读取 config latency；
`ptx_thread_info::ptx_exec_inst` 执行 functional semantics；shader pipeline
负责 issue、dependency、latency 与 completion。

## 失败分类与限制

- CUDA 12.8 `-lineinfo` 产生的 `.loc ... inlined_at` 会触发当前 parser abort，
  属于 **BASELINE FAILURE**；trace workflow 使用 `TRACE_LINEINFO=0`。
- 本地 A800 是 SM80，未执行 SM90 binary；H100/H200 correlation 为
  **Needs hardware validation**。
- 本阶段没有加入 rank、packet、multicast、reduction、NVSwitch、NVLS 或
  BookSim state。

## Exit decision

SM90 PTX parser→functional→timing baseline 已完成。下一步是定义一条有明确
operands、result、ordering、latency、dependency 和 completion 的 single-GPU
custom PTX instruction；network event 必须保持独立 schema。
