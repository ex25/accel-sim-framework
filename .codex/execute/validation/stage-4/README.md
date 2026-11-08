# 阶段 4 实际验收：mbarrier

状态：**通过**（2026-11-08 独立验收，限阶段4）。本地与远端 PTX 屏障能等待正确对象，查询能返回假和真，同一对象三轮复用正确；到达数与字节数的固定组件检查通过。未发现阻塞阶段5的实现问题。**真实 TMA 数据交付尚未验证**，须在阶段5接入实际搬运后复验。

七项 PTX 测试、四项相关回归和必要时序检查通过。汇总完成 **3844条断言**；补充独立检查完成 **3966条断言**，逐项检查 **2304个输出位置**及 **27个组件中间状态**。hint 采用方案允许的单次轮询，没有实现有界暂停；真实专家可运行后必须检查其轮询成本。本轮停止于阶段4。

[验收要求](../../../validation/stage-4.md) · [实现方案](../../../plans/stage-4.md) · [实现记录](../../implementation/stage-4/README.md) · [本轮证据](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/) · [汇总](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/summary.json) · [独立检查](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/independent-checks.json)

## 被验版本与运行身份

独立阅读九个源码文件的实际差异、状态更新/LSU服务/写回/调度通路、测试预期，以及冻结专家的真实操作数形式。前置[阶段3](../stage-3/README.md)已验收并归档为内层 `a4c35975`；本轮另跑受影响回归，没有将旧记录作为当前修改的通过证据。

复用已交付的 debug 构建：开始时18项文件散列、内层diff、运行库散列全部吻合实现证据，九份修改源码的时间均早于库生成时间，因此未重复全量构建。本轮不改模拟器、模型参数、生产配置或冻结程序；只补强测试并填写本文。保留执行 agent 的全部未提交改动，未提交、未推送、未向其它聊天发消息。Serena 工具未在本会话提供，使用本地检索与只读 GDB。

| 对象 | 实际版本及证据 |
| --- | --- |
| 外层仓库 | `89d06f0bf34ef01ecaf0ddb67acc85f593cd1c48`，开始时已有实现记录和六份阶段4测试差异。本轮修改其中三份测试及本验收记录 |
| 独立内层仓库 | `a4c359759c097f4362200e3bbadf3ff63a373a3b` 加九文件未提交差异；[inner.diff](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/inner.diff)，SHA256 `eb82541423298bb51d9413dec7e8da4ad738cdcd592514181107a27405885a1c`。验收结束再次核对完全未变 |
| 工具链与库 | CUDA12.8 / ptxas V12.8.61、GCC13.4.0；`lib/gcc-13.4.0/cuda-12080/debug/libcudart.so.12`，SHA256 `d0478c1ab9090cfefcd52de45e39ead0c26fbbbbe86c0ffa752e9c6f52e5e32f` |
| 实际模式 | 每份 run.json 保存 ldd、命令与库散列；日志确认 `mode=performance simulation`，`PTX_SIM_MODE_FUNC=0`，无外部PTX替换 |
| 配置 | H100副本仅将SM数132→2，1980MHz；SHA256 `239c7190fc3fd253913ee1a4d546a4b68b22744d38e279225b77b6abf082b354`。时序敏感性副本另改 shared latency29→129，SHA256 `3e32831824f5980ed1fc582ebf321698cc4923d7abb65c65a52e4ae61618c95f` |
| 最终测试版本 | [test-versions.json](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/test-versions.json)；最终生成PTX SHA256 `14cd3641c915be1c380eb0594f123c932c1e8029a4c9bc07b37813fa7c7445c5`；每次运行另保存测试源码副本 |

初始一致性证据：[version-check.json](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/version-check.json)。最终运行身份：[final-debug/run.json](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/final-debug/run.json)。

## 实际测试与独立预期

复用[阶段4测试入口](../../../../workloads/moe/tests/hopper/stage-4/)，预期来自验收方案固定状态表、固定launch的CTA/rank和整数标记，不调用模拟器完成判断生成参考。所有PTX测试均连续启动两次，每次输出先填 `0xdeadbeef`；独立检查覆盖全部输出位置，包括应保持哨兵值的位置。轮询次数要求为正，并另与对应事件运行实际发生的查询次数精确核对。

| 验收项 | 本轮结果与覆盖边界 |
| --- | --- |
| init、本地arrive、无hint/有hint查询 | `local`、`hint`：每CTA64线程、offset1024、count2，两个warp的leader各到达一次。每项事件运行含4次init、24次到达、476次查询，其中12次返回真；到达前后计数准确，没有功能/时序重复扣数，`_`的副作用保留 |
| 三轮复用、predicate依赖 | 同一对象phase精确变为1/2/3；parity寄存器依次0/1/0，输出11/22/33。测试预置相反predicate，紧接查询的分支与selp结果正确；每次查询服务后才发生collector写回。源码确认调用warp保持pending到写回，调度循环也在同周期再次检查等待状态 |
| 每线程到达 | `lanes`：count64、64线程分别到达。4个对象各观察64次状态更新，共256次；每次仅减1，最后一次才翻转且重装64。4次查询均为真，输出99 |
| expect/arrive.expect_tx真实PTX入口 | `zero_tx`覆盖零字节；新增 `pending_tx`和`combined_tx` 用寄存器64B分别登记及组合到达。两者均在pending0、tx64、phase0时返回假并输出99；事件分别为8次和4次状态更新，没有过早切轮。这些测试没有注入完成通知，也没有真实搬运 |
| 非零字节完成组件 | [state_component](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/final-debug/state_component.log)的27个中间状态与固定表精确一致：到达先齐、字节先齐、拆分32B完成、完成先到导致tx=-64、最后expect配平、组合更新及三轮复用。完成由正式 `complete_on` 人工调用，不是DRAM/TMA返回 |
| 远端到达与初始化发布 | `remote`：grid2×2、cluster1×2、每CTA64线程，各对象先init、fence、cluster同步。两组rank0分别通知本组rank1；两次launch共8次init、4次通知/交付/翻转。只有目标rank1的 `0x01000400` 翻转，rank0同offset始终未完成；不同cluster不混淆，全部成功查询晚于对应交付 |
| 相关回归 | 阶段3 `cluster_sync`、`ordinary`通过，覆盖原cluster三轮同步、重复launch和普通CTA；阶段2 `copy_mix`、`groups2`通过，覆盖普通异步复制返回、predicate与等待。未重跑无关数值指令或长gating |

真正通过的PTX形式是 `mbarrier.init.shared::cta.b64`、本地/远端 `mbarrier.arrive.shared::{cta,cluster}.b64`、`mbarrier.expect_tx.shared::cta.b64`、`mbarrier.arrive.expect_tx.shared::cta.b64`、无hint/寄存器hint的 `mbarrier.try_wait.parity.shared::cta.b64`，以及 `fence.mbarrier_init.release.cluster`。init count、地址、parity、非零bytes和hint均覆盖寄存器输入；arrive返回值限冻结消费者使用的 `_`。

数值与回归证据：[普通运行](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/final-debug/)、[阶段2](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/regression-stage2/run.json)、[阶段3](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/regression-stage3/run.json)。完整独立检查逻辑见 [verify.py](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/verify.py)。

## 同步、通知成本与hint近似

只读GDB记录状态更新、查询求值、collector写回、远端通知发起和交付，不调用被验函数、不改内存、不注入完成事件。四份主要事件记录逐条检查状态与时间；新增两份非零登记和一份64线程事件补查实际LSU接线。

| 改变与观测 | 本轮实际结果 |
| --- | --- |
| 生产者计算100→200次 | 本地两次kernel均从10409增至15233周期，远端从8755/8753增至11959/11957周期；没有把最后到达者一起挂起 |
| 通知延迟29→129 | 首次launch两组目标翻转由6820/8424变为7320/8924周期；所有通知均满足 `交付 >= 发起 + 配置延迟`，成功等待未越过交付。该参数同时改变shared/SYNCS管线，不能把总周期差全归为网络运输 |
| hint100ns | 1980MHz下对应198周期。实际实现是LSU服务时观察一次，未完成返回假，由原PTX循环继续；无额外198周期固定睡眠。每CTA三轮查询45/37/37次，延长生产者后89/82/82次。两种hint路径均能取得假和真，未进入SASS的发射前无限重试 |

初始化fence的通过范围是此前init已完成、本warp后续经过fence服务再进入程序原有cluster同步，随后其它CTA正确使用对象。它不证明普通shared→async数据发布或TMA数据可见。源码检查还确认arrive release等待本warp先前的store和寄存器写回；真实TMA acquire的数据交付关系尚无本阶段运行证据。

事件：[local](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/events-local/events.log)、[hint](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/events-hint/events.log)、[remote](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/events-remote/events.log)、[129周期](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/events-transport129/events.log)、[64线程](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/events-lanes/events.log)、[expect64](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/events-pending_tx/events.log)、[组合64](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/events-combined_tx/events.log)。[延长生产者](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/delay200/run.json)与[事件命令](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/event-commands.json)保留准确参数及配置散列。

## 测试补强与未覆盖

发现并修正的是测试盲点，没有复现模拟器实现错误：原host第二次launch复用第一次结果作为输入，若某个输出未再次写入，可能被旧值掩盖。现已每次重新填充哨兵后上传。另将count/parity改为寄存器输入，并补两个64B登记后查询为假的定向PTX测试，以覆盖冻结专家的实际操作数。未改变原有数值预期、输入语义或通过标准。

变更见 [validation-tests.diff](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/validation-tests.diff)；最终七项测试与延迟变体均用补强后的版本重跑。初始运行留在 `initial-tests/`、`initial-delay200/`，不混入最终汇总。事件和独立全量检查均正常完成。

尚未覆盖及交接要求：

- **真实TMA**：阶段5必须在数据实际写入shared后向捕获的正确屏障/phase扣实际字节，检查不跨轮、不重复完成及等待/复用顺序。本阶段的人工完成组件不能替代该证据。
- **SASS**：已检查bit31转换、共用状态与同offset查表/完成组件；没有现成可运行的Hopper trace，未执行真实SASS或完整LSU组播完成适配器，不能称SASS运行回归通过。
- **主要性能**：29周期端点和mbarrier按warp完成后继续均未据真实专家校准。阶段5/6首次联测及真实专家运行时记录轮询指令量/等待周期；若忙轮询或串行化明显影响主要耗时，重开hint暂停/吞吐建模，不能默认影响小。
- **范围外**：未验token消费、arrive_drop PTX、test_wait、任意旧轮查询、H200、真实专家FC1/FC2或九例完整前向，不扩非法输入或任意cluster形状。

## 本机成本与复跑

以下为本轮正常debug运行，设备周期含既有5000周期launch费用；注册是main→注册完成，kernel墙钟是提交→同步返回，包含既有host轮询。比较只统计CPU检查及结果写出，不含下载。GDB成本不混入正常运行数据。

| 测试 | 两次设备周期 | 注册墙钟 | 两次kernel墙钟 | 比较耗时 | 总墙钟 / 峰值RSS |
| --- | --- | --- | --- | --- | --- |
| local | 10409 / 10409 | 0.337s | 3.217 / 3.208s | 0.178 / 0.090ms | 6.79s / 273.2MiB |
| hint | 10409 / 10409 | 0.341s | 3.218 / 3.208s | 0.178 / 0.087ms | 6.80s / 273.5MiB |
| 64线程到达 | 5471 / 5471 | 0.335s | 0.906 / 0.905s | 0.103 / 0.079ms | 2.17s / 273.1MiB |
| remote | 8755 / 8753 | 0.336s | 2.509 / 2.507s | 0.123 / 0.085ms | 5.38s / 273.6MiB |
| expect64 | 5508 / 5509 | 0.337s | 1.006 / 1.005s | 0.115 / 0.090ms | 2.38s / 273.1MiB |

各 `.time` 保存 `/usr/bin/time -v`，汇总保留其它测试和回归成本。微测不用于估计九例预算或宣称H100耗时误差约20%；当前没有真实专家计算样本，九例预算没有新增依据。

复跑使用 [commands.sh](../../../../workloads/moe/runs/hopper-stage-4/independent-20261108/commands.sh)，默认写入同层的新目录 `independent-recheck`，需保持该目录层级；其中包括全部普通运行、四项回归、七份事件和两份汇总检查。已有构建对应本轮源码，无需为复跑重复重建。实际命令、库、配置和测试版本另保存在各run.json。

已满足阶段4“零字节PTX本地/远端、非零组件、三轮状态通过”的推进条件，可为阶段5提供屏障状态及通知基础。下一阶段由用户指定，本轮不自行开始。

## 提交归档

内层实现提交 `6e60cf3b1b2957a082ae6cad03613e8d43ef15c6`，提交内容与本轮被验diff完全一致；外层配套提交保存最终六份测试与两份实际记录。上文基准/未提交差异描述验收发生时的版本，原始散列和证据继续保留，可对应到该实现提交。

归档前再次核对模拟器diff和最终测试散列，没有功能、配置、输入或预期变化，因此未重复构建/模拟。原始运行产物不纳入Git；本地复验脚本按阶段3基准比较完整差异，避免归档后因未提交diff变空而误报。未推送，未开始下一阶段。
