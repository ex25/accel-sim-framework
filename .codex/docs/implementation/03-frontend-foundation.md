# 阶段 3：PTX 前端基础

## 当前实现与验收（2026-10-13）

状态: 已验收

验收范围：当前前端及静态 shared ldmatrix 组件合同。

本会话从阶段 3 开始，只实施本阶段；阶段 4 保持进行中，未开始其补齐实现。
原撤销原因保留：旧 probe 只解析；阶段 2 整数 kernel 不执行 ldmatrix。
本次真实执行进一步发现功能 fetch 时序门控及未初始化动态 warp ID 问题。
BF16/TF32/FP8 token 仍不代表对应算术已实现。

### 初始核查与实际修复

[初始检查表与冻结合同](../../../research/hopper-ptx/stage3/CHECKLIST.md) 区分已有
有效证据、实现但缺测试、缺失检查和后续能力；未删除原声明的任何矩阵加载形式。

- 内层 `src/cuda-sim/cuda-sim.cc::ptx_fetch_inst`：只有详细模式检查 matrix timing；
  功能模式可以进入原有 ldmatrix 实现，mma.sync 的执行仍严格失败。
- `src/abstract_hardware_model.{h,cc}::core_t::getExecuteWarp`：功能 fetch 绑定动态
  warp ID；原先直接复制静态指令，collective 线程索引使用未初始化值，实际触发崩溃。
- `src/cuda-sim/instructions.cc::ldmatrix_impl`：保留完整 64 位地址及 offset，
  先验证所有 provider 的 16 字节行在声明 shared 区内，再写寄存器。原来的 u32
  截断和只查 alignment 均不足。所有 provider 先读完，支持地址与目的寄存器 alias。
- `src/cuda-sim/ptx_parser.cc`：地址只接受 32/64 位整数/bit 寄存器或 shared 符号；
  补全五种 MMA 的 m/n/k metadata。`ptx_exec_inst` 对不足 full warp 给出带位置诊断。
- 外层 `stage3/ldmatrix_harness.cc` 实际写 simulator global→PTX shared→ldmatrix→
  global→host 回读；host reference 只比较，不生成 simulator 输出。生命周期与
  stream_manager 的 functional launch/finish 一致，详细 launch 先做 PDOM 分析。
- `run_acceptance.py` 独立按矩阵元素 scatter 到 lane/halfword，严格核对 case 集合、
  全输出及 ptxas。`run_suite.py` 串行构建/检查，以 runtime hash、ldd 和 source hash
  固定版本；缺项、异常、timeout 不写 PASS gate。stage6/8 回归脚本允许指定同一
  SIM_RUNTIME/PYTHON_BIN，避免重新加载历史 `/tmp/hopper-stage3-build`。

### 当前支持矩阵（S=解析，F=功能执行，M=详细时序）

| 形式 | S | F | M | 证据 / 边界 |
| --- | --- | --- | --- | --- |
| ldmatrix m8n8 x1/x2/x4 b16，normal/trans，shared/shared::cta（12 种） | 通过 | 通过 | 未实现，严格拒绝 | 42 数值 case；完整 32 lane；静态 shared/full warp |
| 地址 b32/b64/u32/u64/s32/s64、offset、shared 符号、目的 alias | 通过 | 通过 | 同上 | 全输出；分散/permuted 行、最后合法行、未使用 provider、重复行 |
| sm_90a 与参数 | 通过 | 地址参数/拷贝通过 | label/参数回归通过 | target_number=90，完整 target 字符串；2368 字节参数及 guard |
| MMA m16n8：k16 f16/bf16、k8 tf32、k32 e4m3/e5m2 | 通过 | 未实现 | 未实现 | shape/type/tuple metadata 和正负例；不能当成算术结果 |
| BF16/TF32/E4M3/E5M2 类型入口 | 通过 | 无单独算术验收 | 无单独算术验收 | 合法 bit 寄存器 tuple；不声明原生这些类型变量/算术均已支持 |
| WGMMA m64n128/176k16 f32/f16/f16，SS/RS，fence/commit/wait | 通过 | 既有受限实现回归 | 既有受限模型回归 | 64/88 元素 tuple 完整；16 数值回归、36 负例；阶段 6 整体仍进行中 |
| generic-address ldmatrix、其他 shape/type、predication | 拒绝 | 未实现 | 未实现 | parser 负例/严格匹配；正确拒绝不算实现，按后续需求移交 |
| 动态 shared ldmatrix | 未验证 | 未验证 | 未实现 | 不在原静态 shared 支持合同内；在 10 盘点、11/12 补正式需求 |

片段 reference 依据 [NVIDIA PTX ISA 8.7 ldmatrix](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix)。
手写组件 PTX 经 CUDA 12.8 ptxas 编译，不冒称真实 Dense kernel/PTX 已导出或阶段 10 已完成。

### 版本、运行环境与可复现命令

入口外层 `63d6e475d381308f1afd18cc1bdd3b163d07f4ef`，内层
`f4fe783ac7fc4f1cdaa9efa72a3575537d4fba36`，两者均为 `research/hopper-ptx`。
最终外层为**本报告所在阶段提交**，配对内层为
`eca83b30959381ee14bb26384a0c4f01c3d35c01`。最终 gate 执行时外层仍为上述入口
HEAD 加本阶段 working tree，内层只有用户未跟踪 `.serena/`，无代码 dirty。
外层提交后只有用户未跟踪 `experiments/`；不提交这两个目录。
实际 dirty patch、47 个源码文件 hash、ldd、工具输出和逐条 argv/cwd 见
[versions.json](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen/versions.json)、
[source-sha256.json](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen/source-sha256.json)、
[commands.json](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen/commands.json)。

实际加载 `/tmp/hopper-stage3-acceptance-build/libcudart.so`（5,196,976 字节），
SHA256 `7ce26b6fbec28ff680efcf1f18b846c35856cf82b1e66e4444d97614ffe3552c`。
日志 build banner 为 `gpgpu-sim_git-commit-eca83b30_modified_0.0.`；ldd 也指向该库。
重现时 build/host 工具不同可能改变 binary hash，必须重新记录，不能复用本次 hash。
用户已有 `experiments/` 与内层 `.serena/` 保留，不 reset、不覆盖。

实际环境为 Conda `/home/wangjz/miniconda3/envs/in-switch/bin/python3`，Python 3.11.16、
NumPy 2.4.6；torch/triton 未安装。编译器 `/usr/bin/g++` 9.4.0，CUDA nvcc 12.8.61、
ptxas 12.8，CMake 3.31.8、Bison 3.8.2、Flex 2.6.4；未升级环境，未修改系统权限。

```bash
/home/wangjz/miniconda3/envs/in-switch/bin/python3 \
  research/hopper-ptx/stage3/run_suite.py /tmp/stage3-reproduce \
  --build /tmp/stage3-reproduce-build
```

该命令保存配置/日志并实际执行下列过程（完整参数见 commands.json）：

```bash
cmake -S gpu-simulator/gpgpu-sim -B /tmp/stage3-reproduce-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/stage3-reproduce-build --target cudart -j2
make -C research/hopper-ptx/stage3 -B all \
  SIM_RUNTIME=/tmp/stage3-reproduce-build/libcudart.so
```

随后跑 stage3 数值/前端、stage2 整数、stage8 参数/label、stage6 WGMMA 回归。
本次正式运行命令：

```bash
/home/wangjz/miniconda3/envs/in-switch/bin/python3 \
  research/hopper-ptx/stage3/run_suite.py \
  research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen
```

[最终 gate](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen/gate.json)
为 `PASS`；[完整执行日志](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/suite-frozen.log)
为 `STAGE3_GATE status=PASS`。没有 missing、skip、timeout 或只解析的必需数值项。

| 检查 | 实际结果 |
| --- | --- |
| ldmatrix 42 数值 case | 每例 2 CTA × 2 warp × 32 lane；共 21,504 个输出 word，25,600 个实际加载 halfword；全部 reference 精确一致，max/RMS error=0 |
| 非法运行地址/active mask | 7 个负例；错位、整行越界、最后 provider、部分行越界、高 64 位地址、分歧、partial warp 均拒绝 |
| 合法 ldmatrix detailed | 1 个未实现边界检查，明确报 timing not implemented；不计入数值/M 实现 |
| 新前端合同 | 12 正例、35 非法 modifier/operand 负例；五种 MMA 形状、SS/RS 64/88 tuple、完整 sm_90a |
| 普通整数回归 | 128 元素；checksum=161408，first=1007，last=1515 |
| 参数与 label | 参数 4 正例/5 负例及 copy guard；两模式 scoped label 输出 3、undefined label 拒绝 |
| 已有后续前端边界回归 | 12 parser 正例、19 非法负例、24 functional/detailed 严格拒绝检查；不是合法功能实现 |
| 既有 WGMMA 回归 | 16 全矩阵/空 group 数值场景、36 负例和 timing verifier 通过；非空结果 max_error=0；不恢复阶段 6 整体验收 |

完整输入/PTX/output/reference、逐项误差、退出码、日志均在 `frozen/components/`；
其余回归分别在 `frozen/params/`、`labels-frontend/`、`wgmma/`，普通结果为
`frozen/ordinary.log`。[持久文件校验](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/verification.json)
确认 932 个保存文件 hash、47 个源码 hash、实际 runtime hash 和全部 42 份
output/reference 二进制相等。正式范围内没有未完成必需项。

### 失败与限制（不隐藏失败、不按历史 PASS 验收）

`stage3/evidence/acceptance-20261013/` 保存完整开发记录，中间运行目录另存为
[exploratory.tar.gz](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/exploratory.tar.gz)
（解包保留 baseline/components/final/accepted/release 等原路径），原文件本机亦保留：

- 首次 harness 编译失败（gpu 基类没有 detailed cycle），已改用 gpgpu_sim。
- 旧 runtime 在 ldmatrix 前因 timing guard 失败；此时没有验证高位地址或越界，
  `baseline.json` 的 FAIL 不代表数值已执行。
- 修复 fetch 后，harness 先补正确 functional lifecycle，随后暴露动态 warp ID 崩溃。
- 首个 detailed harness 没有 PDOM 分析、未执行 kernel；gate 正确失败，已加 PDOM
  和 kernel completion 检查。未把输出 sentinel 当作成功。
- 首批 ptxas 拒绝 shared 符号直接作 add 操作数；改为 mov 地址到寄存器后相加。
- `accepted/labels-frontend/store_coords-parser.log` 因重链接并行读库报 file too short；
  该批次无效，后续 suite 串行且锁定同一构建目录，核对运行前后 source hash。

无 H100 数值或周期测量；ldmatrix timing/bank/resource 模型、动态 shared、通用
generic 地址、mma.sync 算术等仍没有本阶段成功证据。详细拒绝合法 ldmatrix 仅证明
边界被保护，不算 M 通过。

### 本阶段可验收条件

- [x] 已声明受支持的 ldmatrix 形式真正执行，所有 lane 输出满足独立 reference。
- [x] 新增前端形式有正例及非法 operand/modifier 负例，tuple/target/参数无截断。
- [x] 普通 kernel 回归通过；保存实际命令、配对版本、全部输出及持久证据。
- [x] 支持矩阵明确每项 S/F/M 状态，未实现形式仍拒绝，不把 parser 当功能通过。

### 后续移交与阶段 4 入口

真实 Dense 新需求由阶段 10 盘点、11 补齐；mma.sync 若被使用，必须在阶段 11/12
补功能/详细正例。WGMMA/layout 资源由阶段 6/12 验证。本阶段通过只证明前端及
上述矩阵加载组件，不证明全部 Tensor Core/完整 Dense。

阶段 4 下一会话先冻结多 barrier/phase/transaction 合同，核对
`mbarrier_impl` 与 `get_ptx_tma_cta` 的单 binding/固定 offset 假设，再实现通用 CTA
状态和 detailed issue/writeback/completion 的恰好一次记账。验证多地址/多 phase、
延迟 producer、乱序 completion、wait/token/parity、drop/invalidate 和跨 launch 隔离；
复用 stage4 功能及 stage5 TMA 回归。cluster-addressed 归 7，完整 Dense pipeline 归 12。
本会话不开始这些实现，也不把已有 stage4/5 PASS 重新写为整阶段验收。

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
