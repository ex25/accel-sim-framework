# 阶段 5 验收：tensor map 与 TMA

状态：未运行。[实现方案和完整指令表](../plans/stage-5.md)；[公共规则](README.md)。

## 本阶段要避免什么

搬运发起不代表输入已到 shared；输出发起也不代表 shared 源已读完。若混淆，前者可能算到上一块数据，后者可能把覆盖后的数据写出去；若总等 global ack 才复用，则会少算流水重叠。这些是需要验证的实现保证，当前没有对应 PTX 故障复现。

## 交付一：描述符和非组播输入

| 测试 | 怎么构造 | 输出与时间判据 |
| --- | --- | --- |
| 内部128B对象 | 经正式 driver-entry 编码；复制到参数/global/shared，复制副本后只修改一份 | address/dim/stride等保留；两份不串改，不依赖原host指针；支持当前rank2及inactive ordinal替换 |
| 修改/发布/取得 | 两个不同矩阵基址与步长，按实际replace→cp_fenceproxy→acquire顺序切换 | TMA读取新对象；128B复制按一次warp协作计量，不产生32份重复流量 |
| 三种box与两种swizzle | 当前64×64、64×128、32×64和64B/128B组合；行列放非对称b16标记 | 普通shared观察解出正确元素，global有效地址/字节正确；预期不能调用被测swizzle函数 |
| 部分/全OOB | 当前rank2合法边界tile；先在shared放非零标记 | 越界部分为零，有效部分正确；全OOB没有DRAM请求，零填写完后屏障仍完成 |
| 真实字节完成 | 使用阶段4的expect/arrive/try_wait，用实际TMA返回替换组件注入 | 有效字节交付及OOB零填完成后才更新屏障；到达/字节未齐不放行 |

通过后就具备首次联测的输入侧条件，不等组播和输出完成。

<a id="early-integration"></a>

## 首次交叉测试：TMA→mbarrier→WGMMA→普通写出

**入口条件：**阶段5非组播输入及阶段6单次WGMMA各自通过，阶段1所需基础指令可用。沿已支持的合法cluster启动，其中一个目标CTA完成本测试，其它参与者按需完成同步；不改变最终专家的两CTA范围。

测试用足够存放 A=64×16、B=16×128 的合法 TMA box（可以带当前布局需要的填充），不能为凑微测创造未实现的 TMA 形状。A/B 使用可精确表示、行列不对称的小值；主机以普通二维矩阵独立计算8192个预期结果。

```text
主机编码输入描述符
        ↓
TMA搬入A/B → 对应mbarrier等待 → WGMMA fence/单次运算
                                      ↓
                             commit → wait0 → 普通global store
```

检查全部128线程×64项结果、实际shared布局和所需发布顺序。使用已完成的mbarrier acquire保证TMA输入可消费；普通shared写入时才走相应proxy fence，不能用一种fence替代全部顺序。记录输入交付、屏障通过、WGMMA发起/完成与普通输出时间。

再用第二组不同数据合法重复一次，确认不是沿用旧tile/descriptor。**不依赖组播、TMA输出、ldmatrix/stmatrix或setmaxnreg**；WGMMA结果直接普通store。这一小测试仅证明两线首次接通，不能替代阶段6的异步流水/配额或阶段7真实专家验收。

本测试由阶段5保存一份报告，阶段6链接它，不运行两套相同联测。失败定位为地址布局、数据完成或寄存器结果后，归回相应实现阶段修正。

## 交付二：两 CTA 组播

使用当前两个rank的mask=1、2、3，合法发起并在目标屏障满足后观察shared。每目标数据与独立预期一致，同offset其它cluster不混。global读取量与目标shared交付量分列：两个目标不意味着两份相同global读，每个目标完成通知仍各一次。空mask不属于本次测试。

## 交付三：输出与缓冲区复用

shared先存A，合法发布后发TMA输出并commit；read wait满足后覆盖为B，检查global仍为A，再写B得到第二份结果。分别覆盖bulk read wait1/0，并记录源读完与global ack两个时刻。

在微测中延长global写确认延迟：源读完后仍应允许复用shared，最终内核结束必须等所需写确认。不要在等待前故意覆盖共享源制造未定义输入。数据检查只能证明值，事件/周期检查同时证明没有把源读取简化为零耗时或把ack强制当read完成。

## 回归、停止与结果

修改TMA/屏障共用模型后运行受影响SASS最小回归；记录缺失trace造成的证据边界。仅测试当前rank2、box、swizzle和两个目标，不扩任意tensor map。上述交付通过后进入完整双缓冲与专家集成。

| 项目 | 当前状态 | 后续填写 |
| --- | --- | --- |
| 编码/replace/发布指令 | 未验证 | 指令实现/修复与提交 |
| 非组播/实际mbarrier完成 | 未验证 | 布局、有效/OOB字节、时序 |
| 首次交叉测试 | 未运行 | 唯一报告链接、另一线提交 |
| 组播/输出/commit/read wait | 未验证 | 数据、双完成点、SASS回归 |
