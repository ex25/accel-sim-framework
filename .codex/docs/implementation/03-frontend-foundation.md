# 阶段 3：PTX 前端基础

状态: 已验收

前置条件：阶段 1、2 已验收；两个仓库保持 `research/hopper-ptx`。本阶段先实现
矩阵指令前端和 `ldmatrix.m8n8.b16` 功能 reference，WGMMA 只提供校验后的 IR 入口，
功能严格失败，完整执行留到阶段 6。预期结果为：正反例通过、64/88 个累加器 tuple
不截断、`sm_90a` 完整保留、普通 kernel 和 shared matrix reference 通过。

## 目标

为后续 Hopper 指令提供可靠的 lexer、parser、类型、寄存器 tuple 和 `warp_inst_t`
入口；区分旧 WMMA、普通 MMA 和 WGMMA。

## 操作

- 扩展 opcode/token 和 modifier grammar。
- 增加 workload 所需 BF16、TF32、FP8 等类型。
- 扩展 vector operand 长度和寄存器 tuple 校验。
- 增加 `ldmatrix` 和必要的 shared layout 语义。
- 验证 `sm_90a` 在 loader/ptxas 路径中完整保留。

## 代码逻辑

```text
PTX token -> ptx_instruction -> operand_info/type metadata
           -> functional implementation
           -> warp_inst_t arch-op and Hopper metadata
```

## 验收

- 每条新增 opcode 有正例、错误 modifier 和错误 operand 测试。
- reference 数值结果通过。
- 尚未有 timing 实现的指令显式标记为“功能通过、时序未通过”。
- 未支持指令严格失败，不静默映射成普通 ALU 或旧 WMMA。

## 实际实现

内层仓库 `gpu-simulator/gpgpu-sim` 在配对分支的 commit
`16f592795466f13d82463daecbbc8ae2ba89828a` 完成前端改动。外层 implementation
commit 为本报告所在外层提交（见 `git log -- .codex/docs/implementation/03-frontend-foundation.md`）。

- `src/cuda-sim/ptx.l`、`ptx.y` 和 `opcodes.def` 将 `ldmatrix`、`mma.sync` 和
  WGMMA fence/async/commit/wait 分成独立 opcode；完整 modifier 字符串由 lexer
  送入严格校验，错误 modifier 不会落入旧 WMMA 或 ALU。
- `ptx_parser.cc` 校验矩阵 tuple 的寄存器数量、类型和宽度，支持最多 128 个
  tuple 元素；`ptx_matrix_metadata` 写入 `warp_inst_t`，保留矩阵形状、转置、
  累加器寄存器和功能/时序支持状态。
- 新增 BF16、TF32、E4M3、E5M2 的 token、大小和 bit-width metadata。它们只作为
  指令类型使用，变量仍使用 `.b16/.b32` 等合法寄存器声明。
- `ptx_version::set_target` 去除 `sm_` 前缀后解析数值，同时保存完整的
  `sm_90a` 字符串；loader 遇到 parser 错误会拒绝部分模块。
- `ldmatrix` 提供 shared-memory `.b16` reference；WGMMA、`mma.sync` 的功能和
  时序实现仍严格失败，作为后续阶段入口。

## 验证命令与结果

在外层仓库执行：

```bash
rm -rf /tmp/hopper-stage3-build
cmake -S gpu-simulator/gpgpu-sim -B /tmp/hopper-stage3-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/hopper-stage3-build --target cudart -j2
make -C research/hopper-ptx/stage3 clean all
```

构建成功。使用 SM90 配置目录
`gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100` 和
`LD_LIBRARY_PATH=/tmp/hopper-stage3-build:/usr/local/cuda/lib64` 运行：

```bash
frontend_probe frontend_types.ptx frontend_types
frontend_probe frontend_ldmatrix.ptx frontend_ldmatrix
frontend_probe frontend_wgmma.ptx frontend_wgmma
```

三个正例均输出 `FRONTEND status=PASS target=sm_90a target_number=90`；分别得到
`mma.sync`、`ldmatrix`、`wgmma.fence/wgmma.mma_async/wgmma.commit_group/wgmma.wait_group`
指令，WGMMA 的 `m64n176` 88 元素累加器 tuple 完整保留。反例：

```bash
frontend_probe frontend_bad_modifier.ptx frontend_bad
frontend_probe frontend_bad_operand.ptx frontend_bad
```

两条命令均非零退出；错误日志分别包含 `unsupported matrix opcode/modifier` 的
源文件位置和 `matrix register tuple length mismatch`、PTX 行号。parser 错误不会
再返回部分 function。

正反例输出保存在 `/tmp/stage3-types.log`、`/tmp/stage3-ldmatrix.log`、
`/tmp/stage3-wgmma.log`、`/tmp/stage3-badmod.log` 和 `/tmp/stage3-badop.log`；
运行目录中的配置副本为 `/tmp/hopper-stage3-probe-run`。`ldmatrix` 的功能实现已
接入 shared-memory `.b16` 行读取和寄存器写回；本阶段探针验证其 parser/IR，数值
执行回归仍由下面的普通功能 harness 覆盖。

普通功能 reference 使用阶段 2 harness 和新 runtime：

```bash
make -C research/hopper-ptx/stage2 clean all CXX=/usr/bin/g++ \
  SIM_RUNTIME=/tmp/hopper-stage3-build/libcudart.so
stage2_harness stage2_kernel.ptx
```

结果为 `RESULT status=PASS n=128 checksum=161408 first=1007 last=1515`，日志证据
为 `/tmp/stage3-harness.log`。

## 限制与下一阶段入口

阶段 3 通过的是 lexer/parser、类型 metadata、tuple 校验、`sm_90a` 保留和普通
reference。`mma.sync` 与 WGMMA 仍为“功能未通过、时序未通过”；`ldmatrix` 只有
shared-memory reference，尚未接入 Hopper shared layout/timing 统计。mbarrier、TMA、
WGMMA synchronization 和 cluster 语义必须等待阶段 4-7，不能在前端阶段静默近似。
