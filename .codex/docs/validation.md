# 验证总表

## 证据层级

1. Parser：文本 PTX 能否被接受，错误能否定位到文件和行号。
2. Functional：寄存器、内存、矩阵和 predicate 结果是否与 reference 一致。
3. Synchronization：phase、arrival、transaction、token、wait 顺序是否一致。
4. Timing：事件、队列、latency/II、访存和瓶颈统计是否可解释。
5. Hardware comparison：A800 同问题兼容实现的趋势对照；无 H100/H200 时不宣称硬件校准。

## 记录格式

每个测试记录 command、输入 checksum、配置、外层/内层 commit、结果文件路径和结论。
失败输出保留，不用后续成功结果覆盖。

## A800 对照

相同数学问题、输入、shape、dtype，使用各架构可执行的 kernel。报告实现差异，分别
统计功能结果和性能趋势，不把 A800/H100 当固定换算比例。
