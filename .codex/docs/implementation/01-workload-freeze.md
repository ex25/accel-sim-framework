# 阶段 1：冻结 FlashAttention-3 workload

状态: 未开始

## 目标

固定一个可复现的 FA3 前向 kernel、文本 PTX、shape、dtype 和工具链，避免实现过程
不断变化输入而无法判断缺口是否真实解决。

## 操作

1. 记录 FA3 源码 revision 和构建命令。
2. 固定 CUDA toolkit、`ptxas`、编译器和 `-arch=sm_90a`。
3. 保存 PTX、ptxas resource usage 和输入 checksum。
4. 使用脚本提取 opcode、modifier、type、vector operand、barrier 和 descriptor。
5. 把每条指令放入 `missing/parser/functional/timing/validated` 分类。

## 代码逻辑

本阶段不改模拟器。输出是后续 parser 的需求集合，不能把 SASS trace 中的 opcode
名称直接当作 PTX 文本语法。

## 验收

- 另一个模型能用文档重建同一 PTX 文件。
- kernel 名称、线程块、shape、dtype 和输入数据固定。
- `sm_90a` target 和资源输出未被数字化为 `sm_90`。
- 指令清单覆盖 FA3 实际 kernel，而非只列 README 特性。

## 失败处理

缺少源码、PTX 或固定输入时状态为 `阻塞`，不得开始阶段 2。
