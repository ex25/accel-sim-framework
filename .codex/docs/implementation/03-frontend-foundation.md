# 阶段 3：PTX 前端基础

## 当前审计与执行入口（2026-10-13）

状态: 进行中

**下一实现会话从本阶段开始，完成后报告阶段 4 入口。** 撤下整体验收的直接原因：
parser/IR 正例有效，但 ldmatrix 的数值证据不足；报告引用的阶段 2 整数 kernel
没有执行 ldmatrix。BF16/TF32/FP8 token 也不等于对应算术已实现。

### 本阶段如何补齐

1. 先冻结当前支持清单及测试矩阵，核对 `ptx_parser.cc`、`instructions.cc` 与
   `stage3` fixture：明确 ldmatrix 形状/x1/x2/x4/transpose 的实际支持范围、
   类型/tuple/opcode 的前端能力。不能删除已有支持声明来绕过数值验证。
2. 在 stage3 增加真实执行 harness，使用非均匀 shared 输入，回读所有 lane 的
   完整 fragment，与独立 reference 逐项比较；覆盖受支持形式、地址/边界和负例。
3. 补 parser/operand/type/label/tuple 缺陷，并做 sm_90a、长 tuple、合法/非法
   modifier、参数和 label 回归。需要真实编译 PTX 时在本阶段最小导出并冻结版本，
   不要求先验收阶段 10。
4. 支持矩阵分别标 parser、functional、timing。保留普通 kernel 回归，但不能
   用它代替矩阵数值；WGMMA 当前已在阶段 6 有受限实现，不重复写成完全未实现。

### 本阶段可验收条件

- [ ] 已声明受支持的 ldmatrix 形式真正执行，所有 lane 输出满足独立 reference。
- [ ] 新增前端形式有正例及非法 operand/modifier 负例，tuple/target/参数无截断。
- [ ] 普通 kernel 回归通过；保存实际命令、配对版本、全部输出及持久证据。
- [ ] 支持矩阵明确每项 S/F/M 状态，未实现形式仍拒绝，不把 parser 当功能通过。

### 后续移交（未完成，不是本阶段的循环前置条件）

真实 Dense 新需求由阶段 10 盘点、11 补齐；mma.sync 若被使用，必须在阶段 11/12
补功能/详细正例。WGMMA/layout 资源由阶段 6/12 验证。本阶段通过只证明前端及
上述矩阵加载组件，不证明全部 Tensor Core/完整 Dense。完成本阶段前不进入阶段 4。

当前执行顺序见 [单卡 Dense 计划](../../plans/single-card-dense.md)。

<details>
<summary>历史实现、测试结果和旧计划（不作为当前验收状态或执行顺序）</summary>

前置条件：阶段 1、2 旧范围通过；两个仓库保持 `research/hopper-ptx`。本阶段先实现
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

</details>
