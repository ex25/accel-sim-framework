# 上游来源与许可证

本 workload 调用 Megatron 的单层 MoE，并导出/重新编译其实际使用的 GPU 内核。没有替换框架的数学实现。对应版本和每个模块的来源记录于 [workloads.json](../artifacts/workloads.json) 与 [编译记录](../evidence/build.json)。

| 上游 | 固定来源 | 许可证 |
| --- | --- | --- |
| Megatron-LM / Core 0.12.0 | [MoELayer 源码](https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/transformer/moe/moe_layer.py) | [Megatron-LM.txt](Megatron-LM.txt) |
| Transformer Engine 2.9.0 | [官方 BF16 grouped GEMM](https://github.com/NVIDIA/TransformerEngine/blob/70f536662ae10a62a54f4ed1ba92e3314c5cfd69/transformer_engine/common/gemm/cutlass_grouped_gemm.cuh) | [TransformerEngine.txt](TransformerEngine.txt) |
| CUTLASS | [TE 对应子模块提交](https://github.com/NVIDIA/cutlass/tree/57e3cfb47a2d9e0d46eb6335c3dc411498efa198) | [CUTLASS.txt](CUTLASS.txt) |
| PyTorch NGC 25.04 | [对应源码提交](https://github.com/pytorch/pytorch/tree/79aa17489c)；构建使用容器内 TensorTopK.cu、Sort.cu | [PyTorch.txt](PyTorch.txt) |
| Triton | NGC PyTorch 携带的编译器；实际 PTX 直接从 JIT 模块导出 | [Triton.txt](Triton.txt) |

CUDA/NGC 工具由其各自许可约束；框架版本与编译来源记录于工作负载清单和构建记录中。
