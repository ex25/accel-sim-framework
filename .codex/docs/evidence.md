# 已确认源码证据

以下证据以外层当前配对分支所记录的内层基线 `91880c53` 为准。

## PTX 前端缺口

- `gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l:84-86` 只识别 `wmma`、`wmma.load`、
  `wmma.store`。
- `src/cuda-sim/opcodes.def:55-57` 将这些旧 WMMA 接到 `MMA_OP`。
- `src/cuda-sim/instructions.cc:1903` 的 `mma_impl` 使用单 warp 和 `16x16` 矩阵。
- `ptx.l:286-289` 的类型列表没有完整 Hopper datatype，`.f16x2` 仍带 TODO 映射。
- `ptx.y:612-615` 的 vector operand 只覆盖有限长度，不能直接假设能容纳 WGMMA tuple。
- 定向搜索没有找到基线的 `wgmma`、`mbarrier`、`cp.async.bulk`、TMA、cluster、
  `mapa`、`setmaxnreg` 前端和功能实现。

## 可复用共享模型

- `src/abstract_hardware_model.h:155-160` 已有 `FENCE_OP`、`SYNCS_OP`、`TMA_OP`、
  `STAS_OP`、`ARRIVES_OP`。
- `abstract_hardware_model.h:317-364` 保存 mbarrier 的 pending count、transaction
  count 和 phase。
- `src/gpgpu-sim/shader.cc:3723`、`:3772`、`:3814` 实现 mbarrier init、arrive、
  complete transaction。
- `warp_inst_t` 已保存 cluster、TMA、GMMA commit 等字段；这些字段应成为 PTX 和
  SASS 的共同接口，而不是再造一套 PTX 专用 timing path。

## SASS 动态输入的含义

- `gpu-simulator/trace-driven/trace_driven.cc:253` 的 trace 解析填充共享
  `warp_inst_t`。
- `:346-357` 从 trace 提供 TMA 展开地址、mbarrier 地址、byte/OOB count 和
  multicast mask。
- `:634-672` 根据 GMMA N 维度和 commit group 设置时序。
- `:722-857` 依赖 register tracing 构造 mbarrier 操作数。

## 独立阻塞

- `src/libcuda/cuda_runtime_api.cc:4753-4802` 的 module load/function query 仍是
  warning + `CUDA_SUCCESS` stub。
- 现代 PyTorch/Triton 原生入口因此必须列为独立 runtime integration 阶段。
- `src/cuda-sim/ptx_loader.cc:346-358` 使用数字 SM 版本调用 ptxas，需要单独验证
  `sm_90a` 字符串的保留。

## 模型限制

现有 SASS Hopper 模型仍有简化：TMA shared 写侧、try-wait 返回/阻塞行为、部分
cluster/proxy fence 和 workload 特例都必须在阶段 4-8 记录，不能把 SASS 结果直接
当成完整 PTX 正确性 oracle。
