# 009 — SM90 PTX execution-driven baseline

## 问题

项目已经选择 PTX 作为 proposed ISA 的 design path，但此前没有证明 CUDA
12.8 `compute_90` PTX code object 可以通过 SM90/H100 configuration，也没有
记录 parser 到 timing 的 source path。

## 设计

新增一个有界 SM90 PTX-only CUDA workload：global load/store、predicated
branch、dependent FP32 FMA chain、host checksum，固定为 4096 elements/32
iterations。Makefile 不使用 compile-time `-lineinfo`，因为 CUDA 12.8 PTX
parser 会拒绝 `.loc ... inlined_at` 形式。

Launcher 使用现有 `H100-PTX` composition 与 `SM90_H100/gpgpusim.config`；本
阶段没有修改 simulator core 或 nested repository source。

## 文件与 source path

- `experiments/sm90_ptx_baseline/sm90_ptx_baseline.cu`：workload/checksum。
- `experiments/sm90_ptx_baseline/Makefile`：`compute_90 -> compute_90` build。
- `util/job_launching/apps/define-sm90-ptx-baseline.yml`：bounded launcher case。
- `.codex/experiments/2026-09-20-sm90-ptx-baseline.md`：command、output、call
  path、evidence 与 limitation。
- PTX loader/parser：`ptx_loader.cc`、`ptx.l`、`ptx.y`、`ptx_parser.cc`。
- PTX IR/timing：`ptx_ir.cc`、`cuda-sim.cc`。
- Functional dispatch：`cuda-sim.cc`、`opcodes.def`、`instructions.cc`。
- Issue/timing：`shader.cc` 与 `gpu-sim.cc`。

完整 path 为：embedded PTX → lexer/parser → `ptx_instruction` →
`set_opcode_and_latency` → shader issue → functional opcode dispatch → register/
memory result → dependency/timing pipeline → `gpgpu_sim::cycle()` completion。

## Before / after

Before：只有 SM80 A800 SASS/PTX comparison 与 source audit。After：真实 CUDA
12.8 SM90 PTX code object（`.target sm_90`、PTX 8.7）可被 `H100-PTX` 解析并
模拟，functional checksum、completion 和 timing counters 均可观测。

## 测试结果

```bash
make -C experiments/sm90_ptx_baseline clean all inspect
python3 util/job_launching/run_simulations.py \
  -B SM90_PTX_BASELINE -C H100-PTX \
  -N sm90-ptx-baseline -r /tmp/accelsim-sm90-ptx-baseline -n
```

结果：5501 cycles、1,654,784 simulated instructions、IPC 300.8151、
`max_error=3.5762786865234375e-07`、`RESULT ... status=PASS`，并正常检测到
simulator exit。

## 限制与 exit decision

- **Verified**：本地 CUDA 12.8 SM90 PTX parsing、functional execution、
  issue/timing 与 completion。
- **Observed**：当前 H100 config 是 simulator model，不是 hardware calibration。
- **Needs hardware validation**：native H100/H200 execution、Hopper SASS、
  WGMMA/TMA/mbarrier/cluster accuracy、HBM/NVLink/NVSwitch/NVLS。
- **Unknown**：未来 custom instruction 使用的所有 CUDA 12.8 PTX feature 是否
  都能被当前 parser 接受。

Baseline 已完成。下一步是定义 operands、result、ordering、latency、dependency
和 completion 明确的 single-GPU custom PTX instruction，并让 network event
保持独立 schema。
