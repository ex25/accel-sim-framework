# 阶段 4：mbarrier 到达、字节与阶段

状态：仅方案；本轮无实现、修复或运行验收。[对应验收](../validation/stage-4.md)。远端操作依赖阶段 3 的 CTA 身份与 shared 查找；本地状态可先做。

## 本阶段解决什么

mbarrier 是放在 shared 地址上的屏障：既要等参与者到达，也要等登记的搬运字节完成。两者都齐了才进入下一轮。phase 表示轮次，parity 是该轮次的奇偶值。专家内核循环复用同一屏障，因此只存一个 ready 布尔值不够。

## 当前事实及影响

已有 `mbarrier_t` 保存到达数、有符号字节计数和 phase，可复用；**不需要再造一套屏障类型**。现有 SASS 入口从 trace 读取地址/phase，phase 解码有右移 31 位，try_wait 在发射前按无限重试处理。若原样用于 PTX，会在操作数尚未求值时检查，或把 PTX 的 0/1 phase 解错，无法正确返回真假。[源码事实](../hopper-notes.md#existing-waits)

这些是静态适配缺口，未运行证明当前 PTX 已发生卡死。已有 `expect_on` 只加计数，执行 agent 还需将阶段转换检查放到需要的状态变更后，覆盖完成先于 expect 的实际顺序。

## 指令范围与拟动作

| 指令 | 功能及接入动作 | 状态 |
| --- | --- | --- |
| `mbarrier.init.shared::cta.b64` | 初始化 expected/pending arrivals、tx=0、phase=0 | 待新增 PTX 入口 |
| `mbarrier.arrive.shared::cta.b64`、`mbarrier.arrive.shared::cluster.b64` | 对本地/目标 CTA 减到达数；正确处理 `_` | 待新增 PTX 入口 |
| `mbarrier.expect_tx.shared::cta.b64` | 增加本轮期望完成字节，不减少到达数 | 待新增 PTX 入口 |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | 同一次操作先登记字节再到达 | 待新增 PTX 入口 |
| `mbarrier.try_wait.parity.shared::cta.b64` | 使用 PTX 求得的地址和 parity，输出 predicate | 待新增 PTX 入口/适配等待 |
| `fence.mbarrier_init.release.cluster` | 发布此前屏障初始化 | 待新增 |

## 如何实现

1. **只有一份状态。** 用阶段 3 的 kernel/cluster/目标 CTA/shared offset 定位已有屏障对象。PTX 指令求值后调用统一状态变更，TMA 返回也调用同一对象。不能在功能执行和 shader 的同一指令处理各减一次到达数。
2. **保留有符号字节数。** 复用已有 `int32_t tx_count`，不增加负值报错或钳制到零。真实 TE 有先发 TMA、后 expect 的顺序：返回先发生时 tx 可暂为负，登记后配平。到达为零且 tx 为零才翻转 phase、重装下一轮 arrivals；`arrive.expect_tx` 不在两个子动作之间误翻转。
3. **完成身份贯穿请求。** 发起搬运时记住它通知的屏障和本轮身份，返回时更新一次对应字节数。可复用现有请求 uid 和目标信息；不能等返回时根据“当前 warp 最近使用的屏障”猜目标。普通完成顺序由正确计数保证，不另加丢弃旧回调的恢复路径。
4. **PTX parity 在入口归一。** PTX 传 0/1，SASS 位编码保留在 SASS 适配入口；内部以一致含义比较当前/紧邻完成阶段。当前 arrive 输出都是 `_`，只需保留副作用并正确解析丢弃目标；不为未使用的 token 消费指令扩展功能。
5. **try_wait 先保证能返回正确 predicate。** 从 PTX 寄存器读取地址/parity/hint，观察统一状态；未完成允许返回假并让原 PTX 循环继续。不能直接进入依赖 trace 值的无限等待。hint 是纳秒提示：优先复用已有暂停/唤醒机制实现有界等待，在完成或提示期限结束时重新观察；时间换算使用实际 core 频率。若先采用允许返回假的非挂起实现，明确记录轮询近似，并在真实专家主要等待时间出现偏差时修正，不把 hint 原值当周期。
6. **发布只做本指令保证的事。** 初始化 fence 保证此前初始化对本 cluster 后续使用有序，不等价于发布所有 shared 数据。与到达/等待相关的 release/acquire 沿当前存储完成状态接入；地址空间 `shared::cluster` 不意味着无限扩大内存排序范围。

## SASS 可以复用什么

复用 `mbarrier_t`、cluster 屏障查找表、已有 TMA `complete_on` 回调和调度暂停位置。SASS 操作数、phase 位编码、无限 try_wait 和针对其它内核的特殊处理不能照搬。主要修改是 PTX 求值接入、唯一状态更新与正确完成条件，不重写无关 trace 模型。

## 交付与暂缓

先以 CUDA 小程序验证零事务字节的本地/远端到达、轮询和重复 phase；再用组件测试调用正式完成接口验证非零计数。此时未具备 TMA，不能声称搬运通过；阶段 5 必须用真实返回重验。

暂缓未使用的 arrive_drop、test_wait token 形式、任意旧 phase 查询和非法重初始化恢复。负计数、重复 phase、延迟远端到达是实际消费者需要，不能暂缓。调试日志仅保留屏障身份、前后计数和时间，正式测时关闭详细输出。

## 源码入口与实际交付记录

[abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) 的 `mbarrier_t`、`ClusterMbarriersLookupTable`；[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) 的 `check_trywait_ready`、LSU 屏障处理/返回；[shader.h](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h) 的 phase 适配。

待填写：六种 mbarrier 与 fence 实现/修复、组件和 CUDA 证据、hint 建模选择及影响、提交。当前：无。
