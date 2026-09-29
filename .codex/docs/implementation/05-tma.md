# 阶段 5：TMA 与异步内存

状态: 未开始

## 目标

从 PTX descriptor 生成与 SASS trace 等价的 TMA 地址、字节计数、OOB 和 multicast
元数据，并完成 shared-memory 语义。

## 操作

- 解析 tensor-map descriptor。
- 计算 global-to-shared 地址展开和边界填充。
- 计算 byte count、OOB byte count 和 mbarrier 关联。
- 实现 shared 写侧、异步完成和 multicast CTA mask。
- 将结果填入 `warp_inst_t`，复用现有 LDST/mbarrier timing。

## 验收

- 一维、二维和越界 descriptor 与 reference 一致。
- TMA completion 发生后 mbarrier 才可满足等待。
- shared-memory 写入和 multicast 目标正确。
- 发起、等待、完成和访存统计可以区分。
