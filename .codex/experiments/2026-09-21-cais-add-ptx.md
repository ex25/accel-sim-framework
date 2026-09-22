# `cais.add.u32` single-GPU PTX extension 实验

日期：2026-09-21

Evidence level：本地 CUDA 12.8 + nested GPGPU-Sim software path 为
**Verified**；任何真实 GPU implementation 都是 **Needs hardware validation**。

## 目标与指令 contract

在添加 communication event、BookSim 或 NVSwitch behaviour 前，先验证最小
proposed PTX instruction：

```ptx
cais.add.u32 %r_dst, %r_src1, %r_src2;
```

该指令执行 unsigned 32-bit modulo addition，写入一个 register，不产生
memory/network side effect，并参与正常 register RAW dependency。prototype
复用 integer `ADD` 的 latency、initiation interval 和 `INTP_OP` resource。
这是 proposed simulator contract，不代表 NVIDIA hardware 或正式 CAIS ISA。

## 固定输入

| 项目 | 值 |
|---|---|
| 实验前 Accel-Sim | `b6c241d`（`test(ptx): establish SM90 execution-driven baseline`） |
| Nested GPGPU-Sim branch | `feat/cais-add-ptx` |
| Nested GPGPU-Sim commit | `8f3c35fa`（`feat(ptx): add single-GPU cais.add prototype`） |
| CUDA toolchain | CUDA 12.8.61、PTX 8.7、`.target sm_90` |
| Simulator config | `H100-PTX` / `SM90_H100/gpgpusim.config` |
| 本地 hardware | 一张 A800 SM80；不用于执行该 SM90 fixture |
| Raw result directory | `/tmp/accelsim-cais-add-final-20260921/` |

## 修改文件

Nested GPGPU-Sim（独立 Git checkout）：

- `src/cuda-sim/ptx.l`：识别 `cais.add` mnemonic。
- `src/cuda-sim/opcodes.def`：注册 `CAIS_ADD_OP`，dispatch 到
  `cais_add_impl`。
- `src/cuda-sim/instructions.cc`：实现 modulo-32 register addition，不产生
  memory/communication side effect。
- `src/cuda-sim/cuda-sim.cc`：映射到 integer ADD timing/resource class。
- `libcuda/cuda_runtime_api.cc`：让 `PTX_SIM_USE_PTX_FILE` 在 CUDA 6+
  external-PTX loader path 中生效。由于 stock `ptxas` 无法处理 proposed
  opcode，该 synthetic override 暂时跳过 PTXInfo loading；production
  implementation 仍需要 assembler/occupancy metadata。

Main Accel-Sim repository：

- `experiments/cais_add_ptx/cais_add_ptx.cu`：deterministic dependent-chain
  workload 与 host checksum。
- `experiments/cais_add_ptx/make_custom_ptx.py`：生成 control PTX，并确保只把
  一个 `add.u32` 改为 `cais.add.u32`。
- `experiments/cais_add_ptx/Makefile`：有界 `compute_90 -> compute_90` build
  与 fixture inspection。
- `experiments/cais_add_ptx/compare_runs.py`：比较 control/custom functional
  与 timing counters，并要求 iteration scaling。
- `util/job_launching/apps/define-cais-add-ptx.yml`：4096-element、1/32 次
  dependent addition cases。

## 编译

```bash
cd /home/wangjz/in-switch/accel-sim-framework
source /home/wangjz/in-switch/activate-accel-sim.sh
cmake --build gpu-simulator/build/release -j8
cmake --install gpu-simulator/build/release
make -C experiments/cais_add_ptx clean all inspect
```

`cmake --install` 不可省略，它会把 rebuilt `libcudart.so` 安装到
`activate-accel-sim.sh` 实际使用的目录。fixture build 输出
`PTX_FIXTURE_PASS marker_count=1`；inspection 确认 control 中为 `add.u32`，
custom 中为 `cais.add.u32`，其余 PTX 保持一致。

## 运行

先生成标准 run directory：

```bash
export CAIS_ADD_PTX_ROOT=$PWD/experiments/cais_add_ptx
python3 util/job_launching/run_simulations.py -B CAIS_ADD_PTX -C H100-PTX \
  -N cais-add-final -r /tmp/accelsim-cais-add-final-20260921 -n
```

在对应 `H100-PTX` run directory 中显式选择 fixture：

```bash
source /home/wangjz/in-switch/activate-accel-sim.sh
export PTX_SIM_USE_PTX_FILE=1
export PTX_SIM_KERNELFILE=$CAIS_ADD_PTX_ROOT/build/control.ptx  # 或 custom.ptx
cd /tmp/accelsim-cais-add-final-20260921/<case>/H100-PTX
$CAIS_ADD_PTX_ROOT/build/cais_add_ptx 4096 1  > control-iter1.log
$CAIS_ADD_PTX_ROOT/build/cais_add_ptx 4096 32 > control-iter32.log
```

custom pair 使用 `custom.ptx`，随后检查四个 log：

```bash
python3 $CAIS_ADD_PTX_ROOT/compare_runs.py \
  control-iter1.log custom-iter1.log control-iter32.log custom-iter32.log
```

## 结果

control/custom log 均出现 external-PTX marker，因此不是早期静默解析
embedded `add.u32` 的无效结果：

| Fixture | Iterations | Cycles | Instructions | IPC | Checksum | max error |
|---|---:|---:|---:|---:|---:|---:|
| control `add.u32` | 1 | 3765 | 114688 | 30.4616 | 7061451698760 | 0 |
| custom `cais.add.u32` | 1 | 3765 | 114688 | 30.4616 | 7061451698760 | 0 |
| control `add.u32` | 32 | 4293 | 622592 | 145.0249 | 19941060158848 | 0 |
| custom `cais.add.u32` | 32 | 4293 | 622592 | 145.0249 | 19941060158848 | 0 |

四次 host check 都输出 `status=PASS`。checker 输出：

```text
CAIS_ADD_TEST_PASS iter1_cycles=3765 iter32_cycles=4293 \
iter1_insn=114688 iter32_insn=622592
```

修改后复跑既有 SM90 PTX baseline：

```text
gpu_tot_sim_cycle = 5501
gpu_tot_sim_insn  = 1654784
gpu_tot_ipc       = 300.8151
RESULT test=sm90_baseline ... max_error=3.576278686523437e-07 ... status=PASS
```

## 合理性与边界

- **Verified**：lexer/parser acceptance、register functional semantics、
  dependent-chain completion、integer timing/resource mapping，以及 simulator
  completion。
- **本阶段结果合理**：prototype 有意复用 ordinary integer ADD timing，因此
  control/custom 应完全一致；32-iteration 的 cycle/instruction 高于
  1-iteration，说明 dependent loop 没有被优化掉，simulated work 随 chain
  length 增长。
- **尚未建立**：hardware latency、compiler scheduling、SASS encoding、
  occupancy effect 或 communication behaviour。当前不会产生 packet、rank、
  stream、multicast、reduction、NVLink、NVSwitch、NVLS 或 BookSim event。
- **已知 implementation boundary**：stock CUDA `ptxas` 拒绝
  `cais.add.u32`；external-PTX harness 因此跳过 PTXInfo/occupancy assembly。
  这适用于 focused execution test，但不是 production solution。
- **Regression status**：release CMake build/install 通过。CTest 输出
  `No tests were found!!!`。已有 CUDA internal-header/pointer-cast warnings
  与缺少可选 `stubgen` 的情况保持不变。

## 下一步

定义独立的 PTX-instruction-to-communication-event schema，以及 single-GPU
completion contract。在 event boundary 有 focused tests 前，不给该 opcode
加入通信副作用，也不实现 Ring、NVSwitch 或 NVLS。
