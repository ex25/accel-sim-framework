# 010 — 最小 single-GPU custom PTX instruction

## 问题

项目已经建立 PTX-primary strategy 和 SM90 execution baseline，但尚未证明
custom opcode 能完整经过 parser、functional execution 与 timing path。在加入
CAIS/in-switch communication semantics 前，需要先定义并验证一条 operand、
result、dependency、timing 和 completion 行为都明确的本地指令。

## 设计

`cais.add.u32 dst, src1, src2` 是有意保持最小范围的 prototype：执行 unsigned
32-bit modulo addition，写入一个 register，不产生 memory 或 network side
effect，并复用 integer `ADD` 配置的 latency、initiation interval 与 `INTP_OP`
resource。

测试 workload 为每个 thread 构造 1 次或 32 次 dependent chain。首先由 `nvcc`
生成 PTX，然后 fixture generator 只把其中一个 `add.u32` 改为
`cais.add.u32`；其余 instruction 保持一致。

## 实现与调用路径

Nested GPGPU-Sim commit `8f3c35fa` 修改五个文件：lexer token、opcode table、
functional implementation、timing mapping，以及 CUDA 6+ external-PTX loader。
loader 修改是必需的，因为原有 `PTX_SIM_USE_PTX_FILE` 只覆盖 pre-CUDA-6
path；如果不修改，control/custom 对比会静默解析 embedded `add.u32`，从而
产生无效的“通过”结果。

实验性 opcode 无法通过 stock `ptxas`，因此 external fixture path 暂时跳过
PTXInfo loading。有效调用路径为：

```text
external PTX fixture → CUDA-6+ loader override → ptx.l lexer/parser
  → CAIS_ADD_OP PTX IR → cais_add_impl register update
  → integer ADD timing/resource class → shader issue/dependency/completion
```

Main repository 中的 workload、fixture generator、launcher 和 checker 见
`.codex/experiments/2026-09-21-cais-add-ptx.md`。

## 验证结果

4096-element 的 control/custom 组合在 1 次与 32 次 dependent iteration 下均
通过。每一对的 cycles、instruction count、IPC、checksum 完全一致，且
`max_error=0`；32 次 case 的工作量高于 1 次 case。修改后，既有 SM90
baseline 仍为 5501 cycles、1654784 instructions、IPC 300.8151，功能检查通过。

这些是 **Verified** simulator software evidence。真实 ISA、latency、throughput、
occupancy 或 SASS encoding 仍是 **Needs hardware validation**。该指令按设计
没有 communication semantics。

## 构建说明与限制

构建必须依次执行 `cmake --build ...` 与 `cmake --install ...`；后者把新
`libcudart.so` 安装到 activated environment 使用的目录。release build 可以
完成，但保留已有 CUDA internal-header/pointer-cast warnings，并容忍缺少可选
`/home/wangjz/.local/bin/stubgen`。CTest 当前没有 registered tests。

为 proposed external opcode 跳过 PTXInfo 是 test harness boundary，不是
production solution。未来需要补充 assembler/occupancy metadata，并单独定义
instruction 如何产生 communication event。本阶段没有修改 multi-GPU 或
network behaviour。
