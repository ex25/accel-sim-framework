# 验证总表

当前验收范围和执行顺序以 [单卡 Dense 计划](../plans/single-card-dense.md) 及
[实现总览](implementation/README.md) 为准。阶段 3–9 当前为进行中；原始 evidence
中的 PASS 只属于当时用例集合，不可覆盖新的必需项。

## 证据层级

1. Parser：文本 PTX 能否被接受，错误能否定位到文件和行号。
2. Functional：寄存器、内存、矩阵和 predicate 结果是否与 reference 一致。
3. Synchronization：phase、arrival、transaction、token、wait 顺序是否一致。
4. Timing：事件、队列、latency/II、访存和瓶颈统计是否可解释。
5. Hardware comparison：A800 同问题兼容实现的趋势对照；无 H100/H200 时不宣称硬件校准。

## 记录格式

每个测试记录 command、输入 checksum、配置、外层/内层 commit、结果文件路径和结论。
失败输出保留，不用后续成功结果覆盖。

完整 Dense 必须检查正式三个 shape 的全 grid、全部输出元素、所有层及必需 API。
聚合器校验 required case 集合；missing、skip、timeout 或仅 parser 通过不能算成功。
数值容差由阶段 10 预先冻结；H100关键 held-out 微基准逐项 ≤20%，完整 block 门限
独立预注册。报告分别给功能、详细模型、Python 入口、硬件对照四个结论。

## A800 对照

相同数学问题、输入、shape、dtype，使用各架构可执行的 kernel。报告实现差异，分别
统计功能结果和性能趋势，不把 A800/H100 当固定换算比例。
