# 阶段7实际验收：三个128-token完整前向

最新状态：**通过，限定三个128-token单卡收尾范围。**Qwen64/128、Phi/128、Mixtral/128均完成22次前向调用，全量输出0项超原容差，主要kernel与内核总和误差均在当前目标内。按用户约定结束单卡精调；六个4096/8192例、硬件等价端到端精度仍暂缓，不能宣布原九例计划全部完成。

**未运行模拟的六个大用例：Qwen64/4096、Qwen64/8192、Phi/4096、Phi/8192、Mixtral/4096、Mixtral/8192。**
现有九例H100基线不代表九例模拟验收通过。
[提交归档的全部kernel/端到端表与精简证据](../../../../workloads/moe/evidence/hopper-stage7-128/README.md)

## 本轮完整前向与统一计时（已完成）

仅运行 Qwen64/128、Phi/128、Mixtral/128，各一次、不预热，三个进程并行；未运行4096/8192。
控制进程 PID `1623105`，本轮已结束。使用固定副本的 release 库 `f223cd2c…b7f25`，冻结程序、输入和数值容差不变。
启动时已核对三个进程均加载该库，之后不轮询长任务。

逐核分别记录首个 CTA 发出至完成的执行周期，以及包含启动等待的完整模拟区间；配置的5000-cycle启动延迟和边界周期不再混入第一列。
完整前向用前后 CUDA event 已记录的设备周期戳直接相减，同时保存主机墙钟时间；不使用逐核时间求和充当端到端时间。
event 周期通过 GDB 读取现有字段，未修改模拟器或冻结程序；小程序在有无观测时均得到 `5324/10648/15972` 周期检查点，前向区间为 `10648` 周期。
模拟器原 `cudaEventElapsedTime` 返回主机墙钟，故模拟运行的 `output.json/cuda_event_ms` 不可作为设备时间，采用独立的 `event-timing.json`。

结果目录：[full128-events-v13bkbds](../../../../workloads/moe/runs/hopper-stage-7/full128-events-v13bkbds/)。
已自动生成根目录 `forward-times.csv`（三例完整前向与H100首次/预热基线）、`kernel-times.csv`（66次调用）和 `final-results.json`；
各模型子目录的用例中保存 `timing.md`、`event-timing.json`、`stage7-summary.json`、全量输出、原比较器与独立数值检查结果；用例旁的 `.time` 文件记录总墙钟与最大RSS。
模拟设备时间包含建模的启动和核间等待，但显式DMA耗时与真实主机提交间隙未建模；与H100完整前向的原始差额会同时列出该范围差异。

修复前完整运行的历史状态：**部分通过；三个128例均完成运行，但完整数值与主要性能验收未通过**。
2026-11-10完成独立验收，范围为Qwen64/128、Phi/128、Mixtral/128。
使用本轮三个并行进程产生的新输出，未采用执行agent的旧Qwen结果代替。
其余六例4096/8192未运行，不能宣布阶段7九例完成。

[验收方案](../../../validation/stage-7.md) · [实现记录](../../implementation/stage-7/README.md) ·
[本轮完整结果](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/final-results.json)

## 现在能做什么、还卡在哪里

三例均在PTX周期模式完成全部22次调用，包含路由、FC1、SwiGLU、FC2和合并输出；
完整注册110个冻结模块，实际调用顺序、entry、grid/block/cluster和共享内存全部符合原记录。
原比较器与独立BF16解码检查器对全部输出得到一致结果，所有值有限。

完整数值仍失败：Qwen有2项、Phi有84项、Mixtral有123项超原容差。Qwen两项的位置和数值
与用户已暂缓的问题完全相同，保留限定可用边界；Phi/Mixtral是本轮新发现的失败，未获暂缓。
三个内核总和接近硬件，但Phi/Mixtral的gating偏慢约61%/69%，被其它调用偏快部分抵消，
不能仅凭总和宣布性能通过。下一步交执行agent定位新数值分歧和gating差额，再复验相关短测试。

## 被验版本、模式与独立预期

| 项目 | 已核对内容 |
| --- | --- |
| 外层HEAD | `a3719197c0eddacaf84d7f5e8217e0a996eb2e3e` |
| 内层HEAD | `b62607065669e4ffcf4d304d9959c29f9085c092` |
| 内层未提交diff SHA256 | `449386bf1daa28e7f0dac46fc950236f7297aa44abe3b10d111226735a5226ce` |
| 实际release库 SHA256 | `f8c4ffa0bbe194393d67389766251d48801a33fffc47a119e796196bb2a759f6`，从候选快照目录加载 |
| 实际配置 | H100、132 SM，两份运行副本与冻结候选散列一致 |
| 模式/环境 | 日志确认PTX simulation mode 0、performance simulation；CUDA12.8/GCC13.4，release；正式前向trace关闭 |
| 原程序与输入 | 二进制、三例forward/reference及110份PTX散列复核未变 |
| 数值标准 | 全量有限值，`abs(actual-ref) <= 1e-5 + 0.02*abs(ref)`，未调整容差 |

源码与构建已绑定，复用库而未机械重建。外层增加验收测试与文档；内层实现、生产配置、
冻结程序/输入/golden没有修改。原始diff和散列见
[preflight.json](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/preflight.json)，
补充测试源码、散列和差异见[handoff.json](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/handoff.json)。
各例build.json、run.json保存实际库路径、模式和命令。未执行Git提交。

## 按验收方案逐项结果

| 验收要求 | 实际结论及边界 |
| --- | --- |
| 开始前：阶段0～6 | 核对原独立验收与候选改动影响；未改的cluster/mbarrier/TMA交付语义复用原证据，公共修改另跑相关回归。见[前置核查](../../../../workloads/moe/runs/hopper-stage-7/independent-prerequisite-review-20261109.md) |
| §1：原专家计算与ABI | 当前库FC1/FC2定向实验通过；原模块16、两份1344B参数、63组表、原N/K、132CTA及双CTA资源，限制group30非空。逐专家整数参考、全部计算值及非活动/guard哨兵精确匹配。局部工作量不冒称完整原浮点权重通过 |
| §1：组合顺序与重叠 | 新真实N/K局部trace通过：输入交付/屏障早于矩阵读取，输入覆盖不早于读取完成，矩阵写回早于stmatrix消费，输出read早于源复用，最后ack早于内核结束；观察到输入搬运/矩阵执行重叠 |
| §2：加载、原启动和完成 | 三例110模块注册集合及22次调用均匹配，FC1/FC2分别在launch index16/18实际执行。全部参数形状和共享内存经独立检查；原程序同步完成、出现ROI结束和完整输出。完整前向未开详细异步trace，细粒度顺序由定向实验支持 |
| §3：范围和停止 | 按用户要求三个128例并行完成；其它六例未覆盖。必要检查结束后停止扩测，不启动下一阶段 |
| §4：全量数值 | 严格比较未通过，失败数为2/84/123；无NaN/Inf，输出均保留BF16位模式。Qwen只有既有两项暂缓，另两例不自动套用 |
| §5：逐调用时间和工作量 | 全部66行匹配同调用ncu；FC1/FC2计算量同时符合问题表独立公式、动态计数和硬件Tensor计数。三个总和接近，主要gating差额仍未解释，性能未通过 |
| §5：ROI/DMA/nsys | kernel区间为首CTA至最后完成，不将compare.py ROI比值当kernel误差；显式DMA/主机间隙未完整建模，未测硬件等价端到端时间。128例没有4096的nsys要求 |
| §6：墙钟/RSS | 保存每例time-v、每kernel墙钟和独立比较时间，见成本表。纯初始化未单独计时，内核外开销不伪装成纯初始化 |
| 受影响回归 | 当前库共享读取/32位地址、单CTA gating、标签/回跳、5项矩阵/异步及状态组件通过。未运行Hopper SASS trace；不能以组件检查宣称SASS运行通过 |

## 完整数值与执行结果

| 用例 | 完成调用 | 全量元素 | 不同值 / 超原容差 | 非有限值 | 最大绝对误差 | 相对L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen64/128 | 22/22 | 262144 | 47 / 2 | 0 | 0.000244140625 | 5.56273e-05 |
| Phi/128 | 22/22 | 524288 | 883 / 84 | 0 | 0.0009765625 | 0.000172435 |
| Mixtral/128 | 22/22 | 524288 | 1172 / 123 | 0 | 0.001953125 | 0.00021754 |

这里的最大绝对误差覆盖全部输出，不等于失败元素中的最大误差；原容差随参考值变化。
每例的原comparison.json、independent-acceptance.json和全部independent-failures.json均保留：

- [Qwen完整独立核查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/qwen64/qwen64-t128/independent-acceptance.json) · [失败位置](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/qwen64/qwen64-t128/independent-failures.json)
- [Phi完整独立核查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/phi/phi-t128/independent-acceptance.json) · [失败位置](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/phi/phi-t128/independent-failures.json)
- [Mixtral完整独立核查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/mixtral/mixtral-t128/independent-acceptance.json) · [失败位置](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/mixtral/mixtral-t128/independent-failures.json)

## 设备时间、工作量与性能判断

以下统一使用首CTA开始至最后完成，按1980MHz换算，与同launch ncu duration比较。
误差保留偏快/偏慢方向，不包括主机间隙和纯launch排队；ncu采集条件保留在摘要中。

| 用例 | 模拟内核和ms / ncu ms | 总和误差 | FC1误差 | FC2误差 | gating误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen64/128 | 0.552415 / 0.561952 | -1.70% | +2.41% | +5.14% | +23.31% |
| Phi/128 | 1.116191 / 1.089344 | +2.46% | -0.14% | +2.68% | +60.58% |
| Mixtral/128 | 1.191896 / 1.186720 | +0.44% | -2.70% | -0.72% | +68.73% |

六次专家逻辑WGMMA计数依次为Qwen354816/177408、Phi819200/409600、
Mixtral917504/458752；均符合独立问题表推导和硬件Tensor工作量，未发现漏算/重复计算。
专家周期均在约6%以内；gating的正偏差与其它调用负偏差抵消，故总和接近不代表主要调用达标。
小核不逐项套20%要求，保留它们对总和的实际影响，不通过加入固定延迟修表。

本轮大差额定位沿工作量→流量→等待/重叠进行：未发现launch/数学工作量不符；
Phi/Mixtral的普通shared标量读未应用Qwen已有合并，shared服务和缓存前序是待核线索，
尚无足以提出新延迟参数的定量证据。专家DRAM读接近硬件，但L2读量偏低约21%～24%，
不能因周期接近就声称缓存层级已校准。详见
[最终性能审查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/final-performance-review.md)、
[gating因果核查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/gating-prefix-review.md)及
[全部66次调用CSV](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/final-kernel-times.csv)。

## 定向顺序、回归与覆盖边界

[FC1/FC2独立实验](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/expert-active-report.md)
分别精确验证42240/30720计算项、1399552/1017856非活动输出哨兵及每例8064项guard；
保持M15、真实N/K，shared为215040B动态+0B静态+1024B driver预留。
[流水审查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/pipeline-validation.json)
检查两rank多轮交付、输入覆盖、矩阵读取/写回、bulk组归属、源复用和最终确认，并观察重叠。

边界：本次原专家trace未出现真正阻塞的bulk wait，也未出现read后ack前复用源；
不能宣称本实验观察到了这两种放行窗口，继续引用[阶段5独立证据](../stage-5/README.md)。
日志只显示屏障当前phase，不显示指令请求parity；局部工作量不代表完整63组原浮点输出。
旧小形状两轮证据只补充持久调度。完整三例数值失败不被局部通过覆盖。

本轮已独立通过的相关回归：

- [shared_vector、bits_loads](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/regression-shared-address/run.json)。
- [单CTA gating](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/gating-single-cta/run.json)：8192项整数参考精确。
- [作用域、U后缀和末尾回跳](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/loading/run.json)：32项精确。
- [integration、pipeline、ldmatrix、stmatrix、legacy8及状态组件](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/regression-matrix/run.json)。

## 失败复现、影响与修正交接

1. **Phi/Mixtral新增数值失败。** 原冻结程序完成后，保存输出与golden分别有84/123项超限。
   例如Phi token123/channel3546应为0.0001974106，实际−0.0001306534，误差约为允许上限的23.52倍；
   Mixtral token97/channel2641应为−0.000001102686，实际−0.0002326965，约为允许上限的23.11倍。
   两例分别涉及18/46个token。不是NaN/Inf，也不能直接归为允许的BF16舍入。
   最早产生分歧的内核尚未定位，不能无证据归到WGMMA或同步错误。
   现有输出可快速复现比较失败；尚未形成导致该差异的短PTX原因复现。
   下一步先捕获失败token对应必要输入/分组与参数，用独立数学参考定位首处分歧，
   再交相应指令/数据路径阶段修复。本轮验证方未修改模拟器取得通过。
2. **Qwen既有两项复现。** index72779、217949的实际/参考值与先前已暂缓问题完全一致，
   没有新增超限；仍保留原比较器failed状态，不写严格数值通过。
3. **gating性能。** Phi/Mixtral分别偏慢60.58%/68.73%，阻止主要性能验收；
   Qwen的23.31%作为小幅超目标保留，不为刚压进20%反复调参。
   交阶段2及其公共shared/cp.async时序路径解释，不按模型拟合系数。
4. **统计stream身份未初始化。** PTX纹理映射构造器未初始化m_streamID，host stream0
   却打印513、529、巨大值；短测loading和integration已复现，UINT64_MAX还会触发显示所有统计key。
   验收summarize.py已改为按正常key独立累计差分，哨兵填null并说明原因，保留原累计量；
   本轮最终66行没有不可归属flag。模拟器错误仍需执行agent在阶段0/common初始化正确身份。
   已确认统计归属影响，未复现由此引起的数值或调度/周期错误。最短复现和源码位置见gating报告。

复现数值失败直接使用已有输出，不必重跑数小时：

```bash
python3 workloads/moe/compare.py workloads/moe/artifacts/cases/phi-t128 \
  workloads/moe/runs/hopper-stage-7/independent-128-20261109/phi/phi-t128
python3 workloads/moe/compare.py workloads/moe/artifacts/cases/mixtral-t128 \
  workloads/moe/runs/hopper-stage-7/independent-128-20261109/mixtral/mixtral-t128
```

每例numerical-failure-review.json保存首个/最坏超差、受影响token和复现命令；
全部失败位置保留在independent-failures.json，不建立另一份重复问题库。

## 本机成本与后续预算

墙钟是本机执行时间，设备毫秒是模拟结果，两者分列。下表为本次三进程并发实测，
不当作独占性能保证，不从128样本按token倍数外推4096/8192。

| 用例 | 进程总墙钟 | 内核模拟合计 | 内核外开销（含初始化） | 原比较时间 | 峰值RSS |
| --- | --- | --- | --- | --- | --- |
| Qwen64/128 | 3:58:52 | 3:55:41 | 191.35s | 0.26s | 12.79GiB |
| Phi/128 | 8:18:21 | 8:13:14 | 306.87s | 0.38s | 13.42GiB |
| Mixtral/128 | 8:58:12 | 8:53:03 | 309.02s | 0.36s | 14.54GiB |

纯初始化未单独埋点；总墙钟减内核墙钟还包含加载、CPU提交/拷贝和下载等，不能全归初始化。
每次launch的模拟墙钟、FC1/FC2工作量及time-v均已保留，可用于同128形状排期。
若需复验，相关短回归为秒级，真实N/K局部FC1/FC2为约2～4分钟；修复后先用短复现和相关回归反馈，
再按影响范围做一次完整前向确认。另预留一次必要失败定位/复验成本。长token预算仍缺实测。

## 证据归档与停止判断

独立检查已实际执行：

```bash
python3 -B workloads/moe/tests/hopper/stage-7/acceptance.py \
  workloads/moe/runs/hopper-stage-7/independent-128-20261109/qwen64/qwen64-t128 \
  workloads/moe/runs/hopper-stage-7/independent-128-20261109/phi/phi-t128 \
  workloads/moe/runs/hopper-stage-7/independent-128-20261109/mixtral/mixtral-t128
```

退出1来自严格数值失败；三例的执行、加载/形状和独立比较交叉检查均已完成。
精确原运行命令与库/配置在各例results.json、build.json、run.json；日志、完整output.f32、
原comparison.json、stage7-summary.json、独立报告、全部失败位置及time-v均保留原目录。
汇总见final-results.json与final-kernel-times.csv。性能摘要绑定修正后summarizer散列，
修正前摘要与脚本保留在summary-before-l2-stream-fix，未覆盖执行方历史证据。

本轮必要验收结束，已保存失败复现并交接，不越权修模拟器或无条件重跑。
可确认三个128例完整PTX执行与定向数学/异步能力；不能解锁“完整数值/性能支持完成”的结论。
下一步是执行agent修复/解释上述问题，再复验；其余六例与下一阶段仍由用户指定。

## 2026-11-10 数值修复定向复验

按用户要求，不重跑完整 MoE，不重复运行原正常 token。复验当前生产算术对原失败的 Qwen 2、Phi 18、Mixtral 46 个 token 的结果，并运行四个原数值失败的短 PTX 用例；未再次租用 H100。

| 用例 | 原超差坐标 | 本次超差 | 检查最终输出数 |
| --- | ---: | ---: | ---: |
| Qwen64/128 | 2 | 0 | 4096 |
| Phi/128 | 84 | 0 | 73728 |
| Mixtral/128 | 123 | 0 | 188416 |

独立核对旧失败集合、H100 输入和输出散列；266240 个最终输出全部有限且满足原容差，原 209 处均逐位恢复参考。激活与全部已采集 FC2 输出逐位匹配 H100；另用 H100 路由权重交叉合并也全部通过。

实际 PTX 的 SwiGLU、近似数学、单条及依赖 WGMMA 四项均复现旧库数值失败、新库通过；周期/动态线程指令不变。库 SHA256 为 `4a6b5530bf40c48b52757e1ffc7a59629effd5c92013472b501cf07782c9c19e`，前后源码和库均与执行方最终快照一致。

**本次数值修复定向验收通过。**token 输出检查是生产算术回放，四个短用例是实际 PTX 模拟执行；没有把二者写成修复后的完整 MoE 前向。原完整运行记录保留，性能问题仍按原记录处理。

[本次验收报告及复现入口](../../../../workloads/moe/runs/hopper-stage-7/numerical-fix-acceptance-20261110-w06y06u5/README.md) · [独立核查 JSON](../../../../workloads/moe/runs/hopper-stage-7/numerical-fix-acceptance-20261110-w06y06u5/independent-acceptance.json)

## 当前交接：剩余性能与统计问题

先阅读项目 AGENTS.md，以及[实现记录](../../implementation/stage-7/README.md)和本报告的最新验收结论。数值修复已通过原失败 token 的定向验收，本轮保留该实现、冻结 PTX/输入/golden 和原容差。

**本轮只处理以下两项。六个 4096/8192-token 用例暂缓；不启动完整 MoE 前向，也不重复整套已通过测试。**

### 1. 定位并修正 Phi/Mixtral gating 的主要耗时偏差

已有三个128-token原前向的同调用比较：

| 模型 | gating 模块 | 模拟 / H100 时间 | 偏差 |
| --- | ---: | ---: | ---: |
| Phi | 41 | 170.446 / 106.144 μs | +60.58% |
| Mixtral | 2 | 170.291 / 100.928 μs | +68.73% |
| Qwen64 | 71 | 104.962 / 85.120 μs | +23.31% |

Phi/Mixtral 是当前主要阻塞；Qwen 的小幅超目标先记录，不为刚压进20%反复调参。专家 FC1/FC2 已在约6%以内，不为微小耗时差额重调资源。

已核对 launch 和填充后的实际计算量。优先检查两份 PTX 的真实 ptxas shared 读取降低、可证明的对齐、模拟器 shared 服务及普通 cp.async 等待，再区分缓存前序影响。两份 PTX 各有192条标量 shared 读，未触发 Qwen 已有的合并规则；这是定位线索，尚不能直接断言这些读取都可安全合并。

复用已有单CTA/单循环及独立 gating 入口，先用短实验确认原因，再运行必要的独立 gating 核。不得按模型、token数、PTX/SASS指令数比例添加耗时系数，也不通过额外同步或删除对齐约束掩盖问题。独立整数 fixture 与原前向的输入/缓存前序不同，比较时明确范围，不冒称原前向已经重测。

证据：[gating 因果核查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/gating-prefix-review.md)、[最终性能审查](../../../../workloads/moe/runs/hopper-stage-7/independent-128-20261109/final-performance-review.md)。相关实现主要在 `cuda-sim.cc::coalesce_shared_loads`、`abstract_hardware_model.cc` 的 shared 服务和 `shader.cc` 的 shared/cp.async 时序路径。

### 2. 修复 stream 统计归属，并核清剩余访存差额

已复现的错误：PTX 使用的 `kernel_info_t` 纹理映射构造器未初始化 `m_streamID`，实际 host stream0 却被打印为513、529或任意大数，甚至 `UINT64_MAX`。后者又是统计打印的“所有key”哨兵，会使单核统计混入其它内核。

在构造和实际 launch 传递处设置正确的 stream 身份，不能把所有队列一律写0。复用已有 loading/integration 短复现，核对统计身份、累计值及单核差分。现有 summarizer 的归属修正只是后处理，不能代替生产代码修复。尚未观察到该错误导致数值或调度/周期错误，不把它当成 gating 偏慢的既定原因。

另外，专家 DRAM 读取接近硬件，但 L2 读量偏低约21%～24%。先核对修正后的统计口径和缓存行为，确认差额原因及对耗时的实际影响；若是模型错误再修，若是当前近似则说明影响。没有“每个访存指标都必须压到20%”的新要求，不因专家耗时已接近便声称所有缓存流量准确。

### 完成后交什么、何时停

更新实现记录，给出原因、具体改动、必要短测试的直接命令，以及同范围的修复前后数值、周期和相关统计。只跑受本次改动影响的短回归，优先复用已有 H100/SASS 证据；实际证据不足时说明缺什么。

Phi/Mixtral gating 的主要差额得到有依据的修正或解释、stream 归属短复现通过、L2差额原因/影响记录清楚后，交独立验收。不自动推进六个大用例、不重跑完整 MoE，也不代替验证方宣布 Stage7 九例全部完成。

## 本轮 gating、stream 与 L2 解释的独立验收

**交接范围内通过。**当前库 `f223cd2c5ea7e9509a998be34afa5bb433508f23c50a58ece911e1488c9b7f25` 与执行方最终清单一致；运行前后源码和库散列相同，冻结 PTX、forward、reference、配置与清单匹配。没有修改生产代码，没有启动完整 MoE、六个大例或新的硬件调用。

### 独立 gating：在最终库复跑原 grid 与完整 K

| 模型 | 同输入修复前归档周期 | 本轮周期 | 本轮时间 / μs | 检查输出 | 数值 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen64 | 204856 | 155771 | 78.672 | 8192 | 全量有限、逐位正确 |
| Phi | 319976 | 221014 | 111.623 | 2048 | 全量有限、逐位正确 |
| Mixtral | 325403 | 209268 | 105.691 | 1024 | 全量有限、逐位正确 |

三例输出也与修复前归档逐位相同，PTX、输入和配置元数据匹配。Qwen 线程指令恰少8388608，Phi/Mixtral各少2752512，分别符合局部复制消除和shared读取合并；shared额外服务计数为524608/483408/483408。结果复现执行方记录，未发现以模型系数修改耗时。

对照旧原前向 H100 的85.120/106.144/100.928μs，量级差为−7.57%/+5.16%/+4.72%。**独立整数输入、地址及缓存前序与原前向不同，这些数值不能写成修复后原前向的正式误差。**

只读核对保存的ptxas证据：Qwen原版/删除复制的诊断版2280行指令及编码相同；Phi/Mixtral各50条LDS.128，与实现记录一致。源码保留单次引用、标签/谓词/用途限制及shared对齐证明，未移除合法性边界。

### stream 与受影响短回归

重新运行六次交错启动，host与kernel身份均为`0,1,2,1,0,2`；各流累计L2写sector为`4,4,4,8,8,8`，每次单核差分均为4。32项输出逐次检查通过，每次首CTA后周期仍为1300。stream修复通过，没有用它解释gating时长下降。

重新执行并通过：三个gating单CTA/单K片段；复制消除、额外用途与控制流三项；shared对齐循环、错位、分支与原向量四项；双启动TMA/WGMMA integration；cp.async的groups6/copy_mix。全部记录实际加载的同一最终库。未扩展到无关回归或CDP执行验收。

### L2：接受已说明的近似，不宣称差额已消除

独立从旧原始日志按统计key累计作差，重算六个专家调用，并与硬件sector/命中率交叉检查：L2读量低约21%～24%的差额仍在，不能靠stream修复宣布消失。

按硬件read+write总量、四舍五入的总命中率及未知write miss范围估算，模拟read miss与硬件估计区间接近，最大区间外距离约0.138%；DRAM读量仍接近。该区间是推算，不是新增硬件子指标；只能支持差额主要落在命中/查找口径，而不能证明全部来自fabric返回查找。

源码复核跨分区返回确实进入回填队列和缓存服务，并未直接跳过；但硬件分区映射、额外查找与计数口径仍未逐项校准。接受实现方如实保留这一边界，不增加命中计数补偿或无依据的周期修正。**本次验收不等于H100全部L2指标准确。**

上述两项生产修复及L2解释在当前交接范围内验收通过，到此停止。此前数值修复结论保留；原完整前向记录保留为历史，六个大样例继续暂缓。

[本轮逐项结果、版本与复现命令](../../../../workloads/moe/runs/hopper-stage-7/performance-statistics-acceptance-4gwar8p7/independent-acceptance.json) · [版本与源码散列](../../../../workloads/moe/runs/hopper-stage-7/performance-statistics-acceptance-4gwar8p7/provenance.json)

## 2026-11-10：修复后的Qwen64/128完整复验

**本用例通过全量数值、实际执行与主要内核时间检查。**独立重新核对110模块和22次
调用/形状/shared；262144项全部有限，0项超原容差，13项不逐位相同；最大绝对误差
1.52587890625e-5，相对L2为2.6107743112063107e-6，原两处超差均逐位恢复。

| 调用 | 模拟 / ncu μs | 有符号误差 |
| --- | ---: | ---: |
| gating | 80.147 / 85.120 | −5.84% |
| FC1 | 273.657 / 267.680 | +2.23% |
| FC2 | 145.793 / 137.728 | +5.86% |
| 全部内核执行时间之和 | 527.885 / 561.952 | −6.06% |

固定库f223cd2c…b7f25，配置/输入/参考和容差不变；两个专家工作量符合独立预期。
设备event区间1155257周期（0.583463ms），与ROI累计周期差一致；仍不宣称包含真实
DMA/主机间隙的硬件等价端到端精度。本机墙钟3:54:37、峰值RSS约12.52GiB。
全部kernel stream身份为0，未出现L2归属error；L2近似保留。

此结论仅限Qwen64/128，不覆盖尚在运行的Phi/Mixtral或六个大用例。
[本次完整分析及原始证据](../../../../workloads/moe/runs/hopper-stage-7/full128-events-v13bkbds/qwen64/qwen64-t128/qwen-scoped-review.md)

## 2026-11-10：三个128-token单卡收尾（America/Chicago）

**按用户本轮范围通过并收尾。**重新执行独立全量验收，三例各22次前向调用与110模块
集合匹配，全部1,310,720项输出有限且满足原容差，原209处超差均逐位恢复参考。
未修改模拟器、配置、输入、golden或容差，未新启动任何长模拟。

| 用例 | 全量元素 / 超原容差 | gating误差 | FC1误差 | FC2误差 | 内核和误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen64/128 | 262144 / 0 | -5.84% | +2.23% | +5.86% | -6.06% |
| Phi/128 | 524288 / 0 | +5.04% | -0.14% | +3.27% | -2.81% |
| Mixtral/128 | 524288 / 0 | +5.13% | -2.81% | -0.53% | -5.03% |

旧失败原因的数值修复、gating修复和stream身份修复已由本次完整前向复验；前置定向数学、
异步顺序及同最终库短回归继续复用，不把全量输出通过扩大为所有未测形式或全部L2计数精确。
小kernel仍有较大相对偏差；端到端的DMA/主机间隙覆盖差异及L2命中/查找近似保留。
按用户“单卡差不多即可，后续重点多卡”的明确要求，不再为这些范围继续精调或扩测。

[全部66次kernel及端到端完整表](../../../../workloads/moe/runs/hopper-stage-7/full128-events-v13bkbds/acceptance-complete.md) ·
[HTML表格](../../../../workloads/moe/runs/hopper-stage-7/full128-events-v13bkbds/acceptance-complete.html) ·
[全部kernel CSV](../../../../workloads/moe/runs/hopper-stage-7/full128-events-v13bkbds/acceptance-all-kernels.csv) ·
[端到端CSV](../../../../workloads/moe/runs/hopper-stage-7/full128-events-v13bkbds/acceptance-end-to-end.csv) ·
[独立机器可读判定](../../../../workloads/moe/runs/hopper-stage-7/full128-events-v13bkbds/final-independent-acceptance.json)

表中单核同时保存首CTA至完成的执行时间、包含启动的模拟区间及ncu参照。
端到端表独立读取CUDA event设备周期，三例分别为0.583463/1.114345/1.182654ms；
硬件首次和预热基线同时保留，但原始比值不用于宣布端到端精度通过。
三例事件区间均与各自ROI累计周期差一致，全部kernel身份为stream0且无L2归属error。

实际运行库为`f223cd2c5ea7e9509a998be34afa5bb433508f23c50a58ece911e1488c9b7f25`；
验收时内层基准为`b62607065669e4ffcf4d304d9959c29f9085c092`，未提交diff SHA256
`8e8211cd4d49bd1a3546aefed24bc427d49138eb8ebb126c3a0c71e87c373e49`。
验收时外层基准为`a3719197c0eddacaf84d7f5e8217e0a996eb2e3e`；各例build.json、batch.json
保存启动时版本/配置/实际库路径和脚本散列，本轮已复核一致。未执行Git提交。

本机总墙钟Qwen3:54:37、Phi8:22:07、Mixtral9:25:09，峰值RSS约12.52/13.39/14.22GiB；
这些是主机模拟成本，不能与上表设备毫秒混用。纯初始化未单独埋点，内核外开销合并标注。

单卡本轮无阻塞待办。剩余六例、进一步端到端/L2校准继续暂缓；下一步重点可转向多卡规划，
本次没有启动多卡实现或实验。

## 本次提交归档

模拟器独立仓库提交：`9cff9650653fbf5420f5c3c7911eb3d78c46c16a`。
它归档了最终受测源码；受测release库在提交前构建，散列仍为f223cd2c…b7f25，
验收时基准及diff保留，不因Git归档改变历史证据。外层提交同时保存全部测试入口及其必要源码依赖，
并将[全部66次kernel、端到端表与数值/版本摘要](../../../../workloads/moe/evidence/hopper-stage7-128/README.md)
归档为小型附件；完整日志/运行库/大张量仍留本地runs。

**本次提交没有六个大用例的模拟运行结果：Qwen64/4096、Qwen64/8192、Phi/4096、Phi/8192、Mixtral/4096、Mixtral/8192。**
因此只确认三个128-token单卡范围通过，不宣称原九例全部完成或多卡已验收。
