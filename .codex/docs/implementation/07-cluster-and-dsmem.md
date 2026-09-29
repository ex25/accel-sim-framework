# 阶段 7：Cluster、DSMEM 与寄存器配置

状态: 未开始

## 目标

补齐 Hopper cluster 地址、CTA rank、远程 shared、cluster barrier/fence 和必要的
`setmaxnreg` 资源语义。

## 操作

- 接入 cluster dimensions、cluster ID、CTA rank 和 launch metadata。
- 实现 `mapa`、cluster shared address 和 DSMEM 访问。
- 实现 cluster barrier、proxy fence 和 multicast 目标。
- 评估 `setmaxnreg` 对寄存器资源、occupancy 和 warpgroup 的影响。

## 验收

- 单 CTA、双 CTA 和多 CTA rank/address 正确。
- DSMEM 远程访问与 reference 一致。
- barrier/fence 不提前完成。
- 不支持的资源行为明确失败或记录为独立限制。
