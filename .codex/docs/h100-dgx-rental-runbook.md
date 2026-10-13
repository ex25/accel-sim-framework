# H100 / DGX H100 短租测试手册

状态：**已确认的准备方案，脚本尚未实现，未执行租机测试**。本手册服务于
[后续路线阶段 14、15、19](../plans/h100-ptx-dgx-roadmap.md)，不替代阶段验收。
目标是在 1–2 小时窗口内完成预定采集并收回证据；租用由用户决定和操作。

## 1. 两类机器分别准备

| 测试包 | 机器条件 | 必需采集 | 不可替代的事实 |
| --- | --- | --- | --- |
| 单卡 H100 | 核实 SKU、显存、驱动、时钟/功耗、MIG 状态；优先 SXM | 冻结微基准、完整 Dense、原版 FA3（准备完成后另项）、数值与计时 | A800 不能验证 Hopper 指令；PCIe 数据不是 SXM 数据 |
| DGX H100 | 完整 8 GPU、NVSwitch/NVLink 拓扑、无其他作业竞争，兼容通信库 | collective/NVLS 路径、DeepEP、完整 MoE/必要 TP Dense | 8 张独立 GPU 不等于 DGX；设置 NVLS 环境变量不证明实际采用 |

订机前向供应方确认环境和权限要求。首次登录后必须重新探测，不能只信商品名称。
单卡窗口不安排 DGX collective；完整 trace/profiler 属于加分项，不保证都能在一小时内完成。

## 2. 租机前必须完成的准备

### 2.1 冻结输入、软件和测量协议

1. 固定全部源码 revision、包/wheel hash、CUDA/PTX/driver 兼容矩阵、编译 flags、
   kernel/PTX、输入/权重和 launch。原版 FA3 与等价 attention 使用不同 case ID。
2. 将 calibration、held-out、smoke、profile 用例分开列入 manifest；登记每个
   case 的预期耗时、优先级、超时、重复数、warmup、容差和计时范围。
3. 准备 FP32/FP64 reference、输入和输出检验；冻结 CPU/JIT/H2D/D2H 与 GPU
   kernel/collective 的分项计时方法，不把 profiler 扰动数据混入普通 timing。
4. 固定模型参数版本。硬件先采原始数据，回本地才调参；不可边看 held-out 边改模型。

### 2.2 提前构建与 JIT 处理

在兼容构建环境准备 wheels、host extension、可预编译的二进制与 PTX、依赖包及
离线安装说明。现有 `in-switch`/CUDA 12.8 不能直接视作 DeepEP V2.5 环境。

截至 2026-10-13 上游说明列出 CUDA ≥13.1、PyTorch ≥2.10、NCCL ≥2.32.3、
Python ≥3.10、支持 C++20 的编译环境；实际执行以阶段 10 冻结 revision 为准，
并确认 PyTorch/DeepEP 使用兼容的同一 NCCL 库。参见
[DeepEP 安装说明](https://github.com/deepseek-ai/DeepEP/blob/main/README.md)。

DeepEP 包含运行时 JIT，host extension 可提前构建，但首次在目标架构上执行可能
仍需编译。准备与版本/架构匹配的缓存、保留必要 compiler，预留有上限的首次 warmup；
缓存不能跨不兼容环境强行复用。warmup 超时应保存日志并跳过该组，不在租机窗口中
临时大改核心实现。需确认单节点 NVLink 模式实际启动，不意外依赖 RDMA 配置。

### 2.3 预期脚本接口（待实现）

下表是阶段实现要求，**不是当前可运行命令**。脚本完成后在阶段报告补真实路径与
运行示例；禁止写出一个尚不存在的“一键脚本”让用户在租机后试错。

| 接口职责 | 必须行为 |
| --- | --- |
| prepare | 打包 manifest/输入/PTX/依赖，检查 hash、磁盘与目标兼容性，生成预计时间预算 |
| preflight | 输出机器/拓扑/软件/权限 JSON，区分 fatal 与仅影响 profiling 的限制 |
| admin-setup | 最小权限修复，dry-run、明确改动与前后检查；仅用户/管理员运行 |
| run | 选择单卡或 DGX preset，逐 case 超时、checkpoint、resume，保留退出码和失败日志 |
| summarize | 比较数值并聚合时间；失败/跳过不得从结果表消失 |
| collect | 即使部分 case 失败仍打包，生成校验和、目录清单和完成状态 |

runner 不依赖一个 SSH 会话存活，应能重连读取状态；支持停止后保存已完成结果。
不要让一个失败 case 阻塞全部采集，也不要用 shell 忽略错误把整包标 PASS。

### 2.4 本地演练门槛

- 在 Conda `in-switch` 验证打包、manifest、输出检查、超时/失败分支和恢复。
- A800 上只验证架构兼容 smoke 及采集流程；明确无法验证的 sm_90a/JIT/NVLS 项。
- 使用模拟故障验证断线/超时后可归档，测量上传和下载时间、产物大小。
- 没有 H100 可预热时，标明首次目标编译风险与预留时间；必要时先安排独立 smoke
  窗口，不能承诺未知 JIT 耗时下所有实验必定在一小时结束。
- 正式包不依赖现场改核心源码。每个必需 case 都有清晰的通过条件和失败后操作。

## 3. 首次登录 preflight

记录设备 UUID/SKU、GPU 数量、显存、驱动/CUDA/runtime、实际 Python 和包路径、
容器或环境 hash、时钟/功耗/温度、MIG/ECC 状态与其他进程；DGX 额外保存拓扑、
NVLink 状态、NCCL/DeepEP 版本、选中算法/协议及所需权限。

运行最小分配/copy/kernel/synchronize smoke；DGX 再运行最小 8 rank collective
和真实 DeepEP dispatch/combine。检查输出，不以进程成功退出代替正确性。
NVLS 设置成功但算法日志显示回退时，标记 fallback 并单独归档，不能归入 NVLS 数据。

需要 sudo/系统权限时，AI 生成最小脚本供用户/管理员运行，包含日志、dry-run 与
恢复说明；不自动改驱动、系统包或功耗设置。计数器权限不足但普通 timing 可运行时
继续采必要数据；若必需路径无法运行，明确该组阻塞并保留证据。

## 4. 现场时间预算

以下是优先级预算模板，不是运行时间保证。prepare 应根据实际 case 数量和测量
开销生成 preset；若必需采集无法装入窗口，提前选择两小时或拆成多次租用。

| 单卡任务 | 60 分钟模板 | 120 分钟模板 |
| --- | --- | --- |
| preflight、校验、最小 smoke/JIT | 0–10 | 0–15 |
| calibration/held-out micros | 10–30 | 15–50 |
| 固定 Dense 数值/逐算子/整体时间 | 30–45 | 50–80 |
| 已就绪 FA3 或精选 profile/补采 | 有余量才做 | 80–100 |
| 打包、下载、校验、异常补录 | 45–60 | 100–120 |

| DGX 任务 | 60 分钟模板 | 120 分钟模板 |
| --- | --- | --- |
| 8 GPU 拓扑、库、NVLS/DeepEP smoke/JIT | 0–12 | 0–20 |
| collective calibration/held-out 与实际算法证据 | 12–28 | 20–50 |
| DeepEP dispatch/combine 及偏斜路由 | 28–38 | 50–75 |
| 完整 MoE/必要 TP Dense 数值和 timing | 38–45 | 75–95 |
| 额外 profile/重测 | 有余量才做 | 95–100 |
| 打包、下载、校验 | 45–60 | 100–120 |

reserve 不得被可选 profiling 吃掉。遇到 warmup/某组超时，先保留最重要的正确性、
必要时间和路径证据；未完成 case 明确 missing，后续补租，不补造结果。
大规模 trace 只对选定 kernel 和短区间采样，保存采样范围，不能冒称完整 workload trace。

## 5. 结果目录与回收

建议每次运行独立目录，不覆盖旧结果：

```text
<run-id>/
  manifest.json
  environment.json
  topology/                # DGX 必需
  inputs/                  # 或持久存储位置与 hash
  artifacts/               # PTX、launch、compiler/build metadata
  raw/<case-id>/           # 原始日志、输出、时间样本、退出码
  results.json
  results.csv
  traces/                  # 可选，明确覆盖范围
  SHA256SUMS
```

退出租机前必须打包并下载到本地/持久存储，核对 SHA256、必需文件、case 总数、
failed/skipped/missing 状态和可解析性。只显示远端“打包成功”不等于已回收。
若下载耗时不可控，采集中分批同步必要小结果，最后再回收大文件。

## 6. 回到本地后的验收

1. 用冻结 reference 复核数值；核查硬件/模拟器 kernel、launch、dtype、路由和
   计时边界相同。不同实现结果可以并列，但不能当模拟误差。
2. calibration 调参后冻结版本，生成 held-out 逐项误差表与波动；失败保留。
3. 完整 Dense/MoE 分开报告 kernel 总和、关键路径、整体 GPU 时间、宿主开销、
   仿真 wall time/RSS。异步重叠时不把 kernel 时间简单相加当端到端。
4. DGX 明确真实拓扑、算法、归约 dtype/舍入与模型假设；没有实际 NVLS 证据的组
   不得支撑 NVLS 准确度结论。
5. 在对应阶段报告链接原始证据、hash、命令和配对版本；全部门槛通过后更新总览。
   数据缺失、预注册门限未过或性能路径不匹配，保持未验收并列补采清单。
