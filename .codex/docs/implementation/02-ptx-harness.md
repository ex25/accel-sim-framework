# 阶段 2：文本 PTX harness

状态: 已验收
日期: 2026-10-08
外层 implementation commit: 本报告所在的阶段提交（见 `git log -- .codex/docs/implementation/02-ptx-harness.md`）
内层 commit: `792d3ab967d63c0721754e15f7153f4910a5873c`

## 目标与前置条件

阶段 1 已验收，冻结 workload 和输入位于
`research/hopper-ptx/stage1/artifacts/`。本阶段验证一条最小文本 PTX
execution-driven 入口，不声称 Hopper 指令已经实现。外层和内层都在
`research/hopper-ptx`；未跟踪的既有 `experiments/` 未修改。

## 源码依据

- 内层 `792d3ab` 的 `src/cuda-sim/ptx_parser.cc::gpgpu_context::init_parser`
  建立文本 PTX parser，`ptx.l::ptx_error` 输出文件名和 lexer 行号。
- 同一 commit 的 `src/cuda-sim/cuda-sim.cc::cuda_sim::gpgpu_opencl_ptx_sim_init_grid`
  按 `gpgpu_ptx_sim_arg` 逆序装入 kernel 参数并调用 `function_info::finalize`。
- 同一 commit 的 `cuda_sim::gpgpu_cuda_ptx_sim_main_func` 为每个 CTA 创建
  `functionalCoreSim`，执行 PTX 并保留全局内存结果；`gpgpu_t::gpu_malloc`、
  `memcpy_to_gpu` 和 `memcpy_from_gpu` 提供内存路径。

## 实现与数据流

`research/hopper-ptx/stage2/stage2_harness.cc` 创建现有 `gpgpu_context`，调用
`gpgpu_ptx_sim_init_perf`、`gpgpu_ptx_sim_load_ptx_from_filename`，查找
`stage2_kernel`，固定 128-thread block 和三个参数（两个 64-bit device pointer、
一个 32-bit N），通过模拟器 global memory 分配/拷贝后调用 functional executor。
`stage2_kernel.ptx` 执行 `ld.global`、整数加法、`.shared` 写读、`bar.sync` 和
`st.global`；harness 读取 global result buffer，与 CPU reference 比较并输出稳定的
`RESULT` 行。三个 fixture 故意触发 opcode、modifier 和 type 的 parser 错误。

## 命令、预期和实际结果

```bash
make -C research/hopper-ptx/stage2 all
research/hopper-ptx/stage2/run.sh /tmp/stage2-run-direct2
research/hopper-ptx/stage2/parser_errors.sh /tmp/stage2-parser-errors3
```

构建成功。运行日志显示：

```text
GPGPU-Sim PTX: finished parsing EMBEDDED .ptx file .../stage2_kernel.ptx
HARNESS kernel=stage2_kernel grid=1,1,1 block=128 params=3 input=0xc0000000 output=0xc0000200
GPGPU-Sim: Done functional simulation (2304 instructions simulated).
RESULT status=PASS n=128 checksum=161408 first=1007 last=1515
```

CPU reference 为 `output[i] = input[i] + 7 + i`，128/128 个元素通过。三个错误
日志都含有完整 fixture 路径和 `:6 Syntax error:`：

```text
.../stage2/bad_opcode.ptx:6 Syntax error:
.../stage2/bad_modifier.ptx:6 Syntax error:
.../stage2/bad_type.ptx:6 Syntax error:
PARSER_ERRORS status=PASS directory=/tmp/stage2-parser-errors3
```

证据文件：`/tmp/stage2-run-direct2/run.log`、
`/tmp/stage2-parser-errors3/{bad_opcode,bad_modifier,bad_type}.log`。

## 范围、失败和限制

当前构建环境没有 NVIDIA 驱动，故没有原生 GPU 对照。阶段 harness 使用旧 PTX
算术/全局内存/共享内存/同步路径；仓库中没有可直接复用的旧 WMMA 文本 fixture，
因此 WMMA 仅记录为未验证，不能把本阶段结果解释为 Hopper opcode 结果。独立构建的
inner runtime 已确认可构建，但本阶段只需链接现有 `libcudart.so` 导出入口，未改动
内层源码。

## 下一阶段入口

阶段 3 可基于 `stage1/artifacts/instruction_manifest.json` 的 parser candidate
集合，逐条增加 `sm_90a` opcode、modifier、type 和严格失败测试；在此之前不进入
mbarrier、TMA 或 WGMMA 语义实现。
