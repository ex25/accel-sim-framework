# 阶段 4：mbarrier 与同步

状态: 未开始

## 目标

把 PTX 的 barrier 初始化、到达、transaction 和等待语义接入现有 mbarrier 状态机。

## 操作

- 实现 init、arrive、arrive_drop、expect_tx、complete_tx、invalidate。
- 维护 pending arrival count、transaction count 和 phase。
- 处理 token/predicate 返回值与软件轮询控制流。
- 复用 cluster mbarrier lookup table 和 shader/LDST 更新函数。
- 专门测试现有 try-wait 的阻塞/writeback 简化。

## 验收

- 单 CTA phase 转换确定。
- transaction 完成前不能错误释放等待线程。
- try-wait 成功和失败路径均有断言。
- PTX 状态转移与合成 SASS trace 对照一致；差异单独记录。
