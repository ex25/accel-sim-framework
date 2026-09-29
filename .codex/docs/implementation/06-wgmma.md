# 阶段 6：WGMMA

状态: 未开始

## 目标

实现 128-thread warpgroup 的 WGMMA 功能语义，并连接现有 GMMA latency、II 和
commit/wait timing。

## 操作

- 定义 warpgroup 参与规则、寄存器布局和 accumulator 更新。
- 解析 descriptor/寄存器操作数和 N 维度。
- 实现 commit group、wait group 和依赖关系。
- 把 PTX commit 标志映射到共享 `warp_inst_t`。
- 分离矩阵结果测试和周期统计测试。

## 验收

- 单指令、多指令累加结果与 reference 一致。
- 非完整 warpgroup、错误 tuple 和 modifier 严格失败。
- wait 不提前释放未完成 GMMA。
- latency/II 随 N 维度变化，且统计项可解释。
