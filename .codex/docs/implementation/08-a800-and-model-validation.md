# 阶段 8：A800 与模型验证

状态: 未开始

## 目标

在没有 H100/H200 实测的条件下，使用 reference、SASS oracle 和 A800 兼容实现检查
功能正确性、事件一致性和性能趋势。

## 操作

- 固定相同 attention 数学问题、输入、shape 和 dtype。
- Hopper 使用 FA3 或对应 Hopper kernel。
- A800 使用 FA2 或普通 CUDA 兼容实现。
- 记录 kernel time、总时间、访存规模、occupancy 和瓶颈。
- 比较 CPU/reference、PTX、SASS trace 和 A800 结果。

## 验收

- 数值结果达到统一误差标准。
- Hopper 指令事件、同步状态和访存统计有解释。
- A800 仅用于趋势和合理性，不做固定倍率校准。
- 报告明确写出架构、编译器和算法实现差异。
