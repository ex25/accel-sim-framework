# 阶段 2：文本 PTX harness

状态: 未开始

## 目标

建立不依赖 PyTorch/Triton 动态 module API 的最小文本 PTX 执行入口。

## 操作

- 核查 `PTX_SIM_KERNELFILE`、`ptx_parser` 和现有 execution-driven 入口。
- 先运行普通算术、全局内存、共享内存和旧 WMMA kernel。
- 固定参数布局、内存分配、线程块尺寸和结果导出格式。
- 对故意错误的 opcode、modifier 和类型检查行号错误。

## 代码逻辑

```text
PTX file -> parser -> kernel metadata -> thread launch
         -> functional execution -> global result buffer
```

本阶段不伪造 Hopper 指令结果；harness 能运行旧指令只证明入口可用。

## 验收

- 最小 kernel 结果与 CPU/reference 一致。
- 参数、地址和线程索引可观测。
- parser 错误包含文件和行号。
- 测试命令、配置和输出文件被写入阶段记录。
