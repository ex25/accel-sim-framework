# 阶段 14：单卡 H100 校准与完整 Dense 收口

状态: 未开始

前置 gate：[阶段 13](13-dense-python-runtime.md) 真实入口通过；冻结规格、数值
容差、calibration/held-out 与测量范围均可追溯。实际命令/结果/证据：尚无。

## 实施步骤

1. 按 [短租手册](../h100-dgx-rental-runbook.md) 实现 prepare/preflight/run/
   summarize/collect 脚本并演练，包括逐 case 超时、resume、失败也打包、下载校验。
   手册列出的接口当前尚不存在；必须在报告补真实脚本路径后才可让用户运行。
2. 为 micros 与三个 Dense case 准备 H100 bundle，保存真实 PTX/launch 与编译产物；
   验证硬件和模拟器同实现。若硬件侧优化/资源变化，重新冻结双方版本再测。
3. 核实 H100 SKU/驱动/时钟/功耗/MIG/资源竞争，记录 warmup、重复与计时条件；
   单独计 kernel、整体 GPU、host/JIT/copy，profile 的干扰数据分开保存。
4. 使用 calibration 数据定位 Tensor Core/shared/cache/HBM/同步/occupancy
   偏差，记录参数来源与变更理由；冻结参数后评估 held-out，每项关键微基准误差
   `abs(T_sim - median(T_hw)) / median(T_hw) ≤ 0.20`。
5. 独立报告每个 Dense shape 的逐算子与端到端误差、波动和瓶颈；全 block 性能
   门限须在硬件结果揭晓前冻结，尚无统一已确认数值，不能事后选有利阈值。
6. 同一最终参数/runtime 复跑最大 case 全 block，测约一天运行预算，复核数值。
   如果模型修补使用了 held-out 数据，声明该集已参与调参并重新准备独立验证。

## 验收条件

- [ ] 三个 case 完整数值通过；所有关键微基准留出项各自 ≤20%，不只报告平均数。
- [ ] Dense 逐算子/整体时间、预注册门限、波动与结论齐全；算法/实现/计时范围一致。
- [ ] 最大正式 case 全详细仿真 ≤24 小时，CPU/内存/runtime/配置及结果可重现。
- [ ] 实测输出、日志、manifest 和校验和已回收到持久目录，失败/缺失情况完整。
- [ ] 总览清楚给出功能、详细、Python、硬件四项状态，不能将其中一项代替其他项。

没有 H100 数据时保留硬件阻塞，完成可离线准备的 bundle；A800 不能代替 H100。
需要 sudo 时 AI 编写可审阅最小脚本交用户/管理员执行，不自行改驱动/权限。
失败门限继续定位，不删除失败用例，不把阶段 8 的 A800 PASS 搬来填本阶段结果。

## 实施后填写与停止点

配对提交/参数版本、硬件环境、实际命令、每 case 数值/性能表、完整原始结果与
hash、运行预算、有效范围及未支持项。通过后停止在**单卡完整 Dense baseline**；
原版 FA3（阶段 15）和多卡阶段不在本轮实现范围，等待后续用户指令。
