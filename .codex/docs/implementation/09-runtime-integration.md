# 阶段 9：运行时集成

状态: 未开始

## 目标

独立评估 PyTorch/Triton 原生 module load、function query、launch、动态 shared memory、
cluster launch 和 TMA descriptor API。

## 当前已知阻塞

基线 `cuda_runtime_api.cc:4753-4802` 中多个 `cuModuleLoad*` 和
`cuModuleGetFunction` 是 warning + `CUDA_SUCCESS` stub。因此补齐 PTX opcode 后，
仍不能声明完整 Python 应用可运行。

## 操作

- 列出目标框架实际调用的 driver/runtime API。
- 为 module、function、launch 参数和 descriptor 建立可观察状态。
- 让不支持的 API 返回明确错误，而不是假成功。
- 只有文本 PTX 阶段稳定后才接入动态应用。

## 验收

- module load、function query、launch 参数和错误路径均可测试。
- API 错误不会伪装成 `CUDA_SUCCESS`。
- 文档明确区分“文本 PTX 支持”和“PyTorch/Triton 原生支持”。
