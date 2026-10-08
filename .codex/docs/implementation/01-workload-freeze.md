# 阶段 1：冻结 FlashAttention-3 workload

状态: 已验收
日期: 2026-10-08
外层 implementation commit: 本报告所在的阶段提交（见 `git log -- .codex/docs/implementation/01-workload-freeze.md`）
内层 commit: `792d3ab967d63c0721754e15f7153f4910a5873c`

## 目标

固定一个可复现的 FA3 前向 kernel、文本 PTX、shape、dtype 和工具链，避免实现过程
不断变化输入而无法判断缺口是否真实解决。

## 前置条件与范围

- 阶段 0 已验收；两个仓库均在 `research/hopper-ptx`。
- 本阶段没有修改模拟器源代码。仓库已有未跟踪 `experiments/` 保持不变。
- FA3 源码来自 `Dao-AILab/flash-attention` revision
  `94e22c906678e5483fa0e9e24d8e787bc2c0ed4c`（2026-10-06）；CUTLASS 子模块为
  `7127592069c2fe01b041e174ba4345ef9b279671`。

## 操作

1. 记录 FA3 源码 revision 和构建命令。
2. 固定 CUDA toolkit、`ptxas`、编译器和 `-arch=sm_90a`。
3. 保存 PTX、ptxas resource usage 和输入 checksum。
4. 使用脚本提取 opcode、modifier、type、vector operand、barrier 和 descriptor。
5. 把每条指令放入 `missing/parser/functional/timing/validated` 分类。

冻结选择为 FA3 Hopper `run_flash_fwd<90,128,128,1,half_t,half_t,false,...>`：
固定长度 B=1、S_Q=S_K=128、H=8、D=D_V=128、FP16、BSHD、非 causal、非 varlen、
非 split、单 CTA cluster。源码适配器为
[`research/hopper-ptx/stage1/frozen_fwd.cu`](../../../research/hopper-ptx/stage1/frozen_fwd.cu)。

## 代码逻辑

本阶段不改模拟器。输出是后续 parser 的需求集合，不能把 SASS trace 中的 opcode
名称直接当作 PTX 文本语法。

## 输出与数据流

- `frozen_fwd.cu` 只实例化上游 launcher；NVCC 生成
  [`artifacts/frozen_fwd.ptx`](../../../research/hopper-ptx/stage1/artifacts/frozen_fwd.ptx)。
- `kernel_metadata.cu` 重新实例化同一模板并输出
  [`kernel_metadata.json`](../../../research/hopper-ptx/stage1/artifacts/kernel_metadata.json)：
  block `[384,1,1]`、dynamic shared `216064` bytes、参数 `2368` bytes、MMA threads
  `256`、producer threads `32`。
- `extract_ptx_manifest.py` 对 PTX 的 2917 条指令和 99 种完整形式逐条记录；当前
  分类为 `missing=23`、`parser=76`、`functional=0`、`timing=0`、`validated=0`。
  `parser` 只表示基线 lexer/grammar 的静态候选，不表示运行成功。
- `generate_inputs.py` 生成固定 little-endian binary16、BSHD 输入。Q/K/V 均为
  `[1,128,8,128]`，公式为 `((i*73+offset)%257-128)/128`，offset 分别为
  17/53/101；每个文件的 SHA256 写在 `artifacts/inputs/inputs.json`。

## 验收

- 另一个模型能用文档重建同一 PTX 文件。
- kernel 名称、线程块、shape、dtype 和输入数据固定。
- `sm_90a` target 和资源输出未被数字化为 `sm_90`。
- 指令清单覆盖 FA3 实际 kernel，而非只列 README 特性。

## 实际命令、结果和证据

```bash
git -C /tmp/flash-attention-stage1 rev-parse HEAD
git -C /tmp/flash-attention-stage1/csrc/cutlass rev-parse HEAD
/usr/local/cuda/bin/nvcc --version
/usr/local/cuda/bin/ptxas --version
/usr/local/cuda/bin/nvcc -ccbin /usr/bin/g++ -std=c++17 -O3 --use_fast_math \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 -arch=sm_90a -ptx \
  -I/tmp/flash-attention-stage1/hopper \
  -I/tmp/flash-attention-stage1/csrc/cutlass/include \
  research/hopper-ptx/stage1/frozen_fwd.cu \
  -o research/hopper-ptx/stage1/artifacts/frozen_fwd.ptx
ptxas -v -arch=sm_90a -o /tmp/fa3-build/frozen_fwd.cubin \
  research/hopper-ptx/stage1/artifacts/frozen_fwd.ptx
python3 research/hopper-ptx/stage1/extract_ptx_manifest.py \
  research/hopper-ptx/stage1/artifacts/frozen_fwd.ptx \
  --lexer gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l \
  -o research/hopper-ptx/stage1/artifacts/instruction_manifest.json
python3 research/hopper-ptx/stage1/generate_inputs.py \
  research/hopper-ptx/stage1/artifacts/inputs
```

结果：PTX 头部为 `.target sm_90a`、`.version 8.7`；`ptxas` 成功，报告 168 registers、
16 barriers、8-byte stack、12-byte spill stores/loads。PTX SHA256 为
`0ee436723623d4f2d3363b36cc4ed9d5ecdb96a6c449367a66356c705a351a76`。固定输入的
Q/K/V SHA256 分别为 `0b2b...a8802`、`34ad...2259c`、`bed69...fb3b`（完整值见
`inputs.json`）。重复构建的输入文件和 PTX 指令清单一致；NVVM 会把源文件路径编码
进内部匿名符号，因此不同输出目录的 PTX 字节 SHA 会变化，不能把该匿名符号差异误判
为 workload 差异。

证据路径：`artifacts/frozen_fwd.ptx`、`artifacts/ptxas.log`、`artifacts/nvcc.log`、
`artifacts/instruction_manifest.json`、`artifacts/kernel_metadata.json`、
`artifacts/inputs/inputs.json`。

## 限制与下一阶段入口

没有 GPU 运行、模拟器执行或 H100/H200 实测；因此功能、同步、时序和硬件准确度均为
“未验证”。基线 parser 对 23 种 Hopper 形式没有 token/modifier 入口，正是阶段 3
的需求集合。阶段 2 只能使用该冻结 PTX 的最小 harness；不得把本报告的 parser
candidate 当成功解析。

## 失败处理

缺少源码、PTX 或固定输入时状态为 `阻塞`，不得开始阶段 2。
