# 阶段 5 实际验收：tensor map 与 TMA

状态：**部分通过**（2026-11-08 独立验收）。描述符编码/修改/发布、三种输入布局及越界零填、两 CTA 组播、输出源复用与最终写确认均通过，未复现阶段 5 的实现错误。**TMA→屏障→WGMMA→普通 store 的首次联测未运行**：[阶段 6 尚未实现单次 WGMMA](../../implementation/stage-6/README.md)，因此不能把整阶段或专家流水标为通过。

现在可以用已验收的搬运和屏障通路准备矩阵输入、回写结果。下一步是在阶段 6 单次 WGMMA 独立通过后，立即运行[首次联测](../../../validation/stage-5.md#early-integration)，比较全部 8192 项及完成顺序；不等寄存器配额等其它项目。本轮不启动阶段 6。

[验收要求](../../../validation/stage-5.md) · [实现方案](../../../plans/stage-5.md) · [实现记录](../../implementation/stage-5/README.md) · [本轮证据](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/) · [复跑命令](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/commands.sh)

## 被验版本和运行身份

独立阅读全部 14 份模拟器变更、测试及冻结专家地址片段。前置[阶段 3](../stage-3/README.md)、[阶段 4](../stage-4/README.md)已验收；本轮仍重跑受影响回归。开始时核对最终交付 diff/库，执行 `make -j8` 成功，库散列与交付一致。Serena 工具未在本会话提供，使用本地检索和只读 GDB。

| 对象 | 实际版本及证据 |
| --- | --- |
| 外层仓库 | `3eb3261021dae8c3ae8fa3b98f6422b57089ece2`；保留原有实现记录、验收方案和五份测试改动。本轮只补强 `micro.cc`、更新本文，辅助脚本/原始产物留 runs；未提交、未推送 |
| 内层仓库 | `6e60cf3b1b2957a082ae6cad03613e8d43ef15c6` 加 14 文件未提交实现；[被测 diff](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/metadata/inner.diff) SHA256 `64eea12802810c0a94172964627ae24c442f7d5461fd7bdc1f173504577a3e7b`，验收结束逐字节确认未变 |
| 构建与库 | CUDA 12.8 / ptxas V12.8.61、GCC 13.4.0、debug；实际加载 `gpu-simulator/gpgpu-sim/lib/gcc-13.4.0/cuda-12080/debug/libcudart.so.12`，SHA256 `b72f702843347742430de54866e1ee9c5800090dd9753e0aa7758d660d88794a` |
| 模式及配置 | `PTX_SIM_MODE_FUNC=0`，日志确认 `mode=performance simulation`；H100 副本只将 132 SM 改为 2 并启用 PTX_TMA 事件，保留 1980MHz、两 partition 和容量。基础配置 SHA256 `42027a216a9be5442f4330cfd5c4ca29b2fb90988935113864852574aeb64ad4` |
| 版本附件 | [源码/测试散列及库时间核对](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/metadata/versions.json)、[构建日志](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/metadata/build.log)；各 run.json 保存准确命令、配置/程序/PTX/库散列和 ldd |

未修改模拟器、模型参数、生产配置、冻结 PTX/输入或容差；未向其它聊天发送消息。写确认诊断只修改运行目录的配置，差异单独记录。

## 实际测试与数值结果

18 项 PTX 测试、必要延迟变体及五项相关回归全部通过。现有汇总脚本完成 **604 条断言**；另写独立检查，对保存结果精确核对 **155648 个 16 位值、400640 个专家 shared 字节**，连同对象/时间检查完成 **4240 条断言**：[独立结果](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/independent-checks.json)、[检查源码](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/verify.py)。没有使用模拟器的地址映射函数生成预期。

| 验收项 | 实际结果与边界 |
| --- | --- |
| 正式查询 ABI、128B 编码、三种 replace | 经正式 driver-entry 查询调用编码器；新增独立的完整 128B 字段预期，检查占位对象与有效输出对象。PTX 参数→shared→replace→global 后逐字节正确：64 位基址、实际 dim128/160、stride320B、inactive dims1/strides160B；原 host 副本不变 |
| 发布/acquire、在途轮换 | 每次 warp 发布恰好 4 个 32B 请求。`rotate` 两轮分别用 1000/20000 范围数据；同址描述符更新时，旧请求仍读原矩阵，新请求读 999，保存的另一描述符再使用仍读原矩阵。后续更新未改变旧请求快照 |
| 三种 box 与完整/部分/全越界 | `input/edge/oob` 各覆盖 32×64/64B、64×64/128B、64×128/128B。128×160 域、行步长320B；完整逐项正确，部分只有512B有效，其余精确零；全越界没有矩阵请求，仅4个描述符请求，仍交付全 tile 零数据并完成屏障 |
| shared 物理排列 | 普通 shared 读回与独立 16B 块表全量一致，保留块内字节顺序。人工 base1280/9472/17664 覆盖非零起始相位；不是只比较 TMA 保存的逻辑数据 |
| 冻结专家地址片段 | 静态核对唯一含 TMA 的冻结模块全部20处：8输出、8个64B输入、2组播、2普通128B输入。实际重放 `expert64/expert128/expert128_64` 原地址/TMA指令，13份 shared dump（含前后 guard）全量正确。覆盖 stage0/1/2、rank0/1；真实起点均为 phase0。只运行片段，未运行完整专家 |
| 组播 mask1/2/3、两个 cluster | 仅 mask 选中的 CTA 改变数据并收到完成，未选 shared 保持0xabcd；mask2 发起 rank0 不被默认通知。同一 tile 每个发起者始终132个源请求，不因mask3翻倍；每个被选目标各交付4096B，两个 cluster 身份正确 |
| 输出及 shared 源复用 | 普通 shared 写入按独立块表准备，经过 async fence→TMA→commit→read wait0 后覆写12345，再写第二个 global 域。A/B完整矩阵和域外0xeeee均正确；部分边界不写域外。两项测试共对204800个 global 元素作 host 全量比较 |
| read wait1/0、类别与最终退出 | 真实 PTX 的 C=2 时 wait1 截止1、留1个未读组仍可继续；wait0截止2，同一未读组使其等待。正式 warp 状态组件覆盖新组晚读、空组、未commit、逐lane隔离、普通cp.async不串队列；组件中只移动ack不移动read，结果正确。真实延迟写确认和退出证据见下节 |
| 相关回归 | 阶段4 `local/remote/pending_tx`、阶段2 `copy_mix/groups2` 及各自组件通过，覆盖普通 shared/global、屏障和原异步返回。没有可运行的小 Hopper SASS trace；仅 SASS 组适配组件通过，不称 SASS 运行回归通过 |

基础运行：[final-debug/run.json](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/final-debug/run.json)；专家片段：[run.json](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/expert-addresses/run.json)、[原指令/行号和地址公式](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/expert-addresses/expert-address-source.json)。回归：[阶段4](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/regression-stage4/run.json)、[阶段2](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/regression-stage2/run.json)。

实际通过的新增 PTX 形式为三种 `tensormap.replace.tile.*.shared::cta.b1024`（address/stride为b64、dim为b32）、完整 `tensormap.cp_fenceproxy...release.gpu.sync.aligned`、`fence.proxy.tensormap::generic.acquire.gpu`、`fence.proxy.async.shared::cta`、普通/组播 `cp.async.bulk.tensor.2d.shared::cluster.global...L2::cache_hint`、`cp.async.bulk.tensor.2d.global.shared::cta.bulk_group`、`cp.async.bulk.commit_group`、`cp.async.bulk.wait_group.read 0/1`。阶段4屏障均走真实 expect/arrive/try_wait，未注入搬运完成。

## 数据交付、屏障与两个完成时刻

只读 GDB 按 kernel、cluster、rank、屏障地址记录真实完成前后状态，不调用被测函数或更改模拟状态。输入调度时剩余 global 返回为0；随后 shared 服务写完数据再调用 `complete_on`。成功查询对应同一对象的实际完成，不能由另一 cluster 的成功记录代替。

| 检查 | 实际观测 |
| --- | --- |
| 普通输入与全越界 | `input32` 首轮最后返回后7747开始安排 shared，7808交付4096B并使tx4096→0、phase0→1，7809查询才成功。`oob128` 两轮均完整交付16384B并翻转，没有等待不存在的矩阵返回 |
| 两个cluster的mask2 | 两次launch共4次完成，全部属于各自 cluster 的rank1；每次4096B，rank0没有完成。查询逐对象检查，没有提前成功 |
| 同一屏障等待两块输入 | 专家普通128B片段的12次交付各16384B，每对象两次：tx32768→16384时phase保持0、查询仍假；第二块使tx→0、phase→1后才成功 |
| 正常输出重叠 | 首轮源读完28324，28338已开始覆盖 shared，而旧输出最后写确认28632才到；全量比较仍得到旧A。没有在ack时改读已经覆盖的源 |
| shared延迟29→129 | `input32` 两轮11613/11813→19503/19638周期；mask2为11810→19895，交付与成功等待均推迟。参数也影响普通 shared/SYNCS，不将总差全算作远端运输 |
| DRAM延迟194→1194 | `input32` 第二轮11813→12808；原H100写回缓存配置下输出read/ack未变，因此该变体不作为“只延迟写确认”的证据 |

逐对象事件保存在本轮证据目录的 `events-input32/`、`events-oob128/`、`events-mask2/`、`events-expert128_64/`、`events-output/`；每目录 command.json 保存命令/配置，events.log 保存完整观察。观察脚本为 [observe.py](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/observe.py)，状态结果也收录在上面的独立结果 JSON。

为验证真正的写确认延迟，另在**运行目录副本**将 L2 策略改为写穿，使 global 写经过 DRAM；在这对副本之间只改 DRAM 延迟194→10194。四次输出的发起→源读完分别仍为311/311/512/512周期，源读完→最后ack由504/504/503/503变为10503/10503/10503/10503周期（增量9999/9999/10000/10000）。源读完后仍能覆盖 shared；A/B及域外值全量正确。这里检验双完成接线，不把诊断策略当H100校准结果。

| 延迟配置下的真实退出 | 最后一个线程执行结束 | 最后一笔TMA写ack | kernel完成 |
| --- | ---: | ---: | ---: |
| 第一次launch | 43889 | 50851 | 54360 |
| 第二次launch | 111516 | 118478 | 121987 |

线程全部结束时仍有TMA写确认未到，但kernel继续等待；还要等待测试自身的普通global输出写完，故kernel完成晚于最后TMA ack。[诊断脚本/配置差异](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/run-ack-isolation.py)、[194日志](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/write-through194/output.log)、[10194日志](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/write-through10194/output.log)、[退出事件](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/events-delayed-output/events.log)。

排列参考核对了 [PTX 8.7 的官方 swizzle 表](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#swizzling-modes)；源读完和最终写完的区分核对了[官方 bulk wait 语义](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)。本轮结果来自实际模拟，不将文档定义作为运行证据。

## 测试补强、未覆盖和停止点

未发现需要执行 agent 修复的模拟器错误。修正了两处测试盲点：原编码结果的未修改字段直接沿用编码器输出作为预期，现按固定偏移独立构造128B比较；普通测试第二轮未清空读回缓冲区，现每轮先填0xcc，防止全越界零值或未选目标的旧结果掩盖未执行的读回。未改变输入语义和数值标准。[测试差异](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/metadata/validation-tests.diff)。

初始15项运行保留在 `initial-tests/`；补强后15项全部重跑，最终汇总使用 `final-debug/`。专家片段的原host已有每轮清空，本轮复跑其13份dump；各延迟目录保留自己的源码快照。最终 `micro.cc` 仅对新增块再作clang-format16.0.6空白格式调整，编译时的确切源码保存在各 `sources/`。GDB与程序的缓冲stdout可能交错，独立检查按完整OBS对象解析，不把破碎的无关trace行当事件。

剩余边界：

- **首次联测阻塞**：缺阶段6单次WGMMA；未比较8192项矩阵结果，也未证实TMA与WGMMA共同消费同一shared布局。后续联测报告在本文索引，阶段6引用同一份证据；失败回所属阶段修复后再验。
- **真实专家/性能**：未运行完整冻结FC1/FC2或九例前向，未给H100约20%时间误差结论。shared按整tile交付、32bank×4B服务量加29周期端点、远端再加29周期近似；输入描述符与数据同批计时、细粒度缓存提示及远端端口竞争尚未校准。轮询沿阶段4按warp服务，真实专家出现后须核查轮询量和主要重叠，不能默认影响小。
- **SASS与其它范围**：没有可用小Hopper trace，未跑真实SASS；已有适配组件不能代替LSU完整trace回归。未扩H200、rank>2、其它dtype/interleave/swizzle、任意mask/拓扑或native编码互换。冻结起点全部phase0，人工非零相位证据不冒称真实非零地址采样。

本阶段可独立完成的必要功能、顺序、时序敏感性和受影响回归已验完，停止扩测。保留首次联测为未验交付项，不为补它而代实现阶段6。

## 本机成本及复跑

以下为debug微测，设备周期含既有5000周期launch费用；本机墙钟与预测设备周期分列。注册为main→注册结束；launch墙钟为提交→同步返回。比较区间含检查和输出保存，output还含两次目的矩阵下载，不是纯CPU数学时间。部分必要检查并行执行，GDB成本不混入表中。

| 测试 | 两次设备周期 | 注册 | 两次launch墙钟 | 每轮比较区间 | 总墙钟 / RSS |
| --- | --- | --- | --- | --- | --- |
| input32 | 11613 / 11813 | 0.49s | 2.81 / 2.91s | 约0.48ms | 6.41s / 277.7MiB |
| input128 | 27068 / 27269 | 0.48s | 7.91 / 7.81s | 约0.83ms | 16.44s / 290.5MiB |
| mask3（两个cluster） | 11950 / 11950 | 0.49s | 4.91 / 4.81s | 约0.66ms | 10.46s / 278.7MiB |
| output（两块） | 33971 / 47237 | 0.49s | 8.92 / 12.32s | 约28.6ms | 21.99s / 277.6MiB |

各 `.time` 保存time-v，完整指标在[汇总](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/summary.json)。没有真实专家样本，不用这些微测更新九例预算。复跑入口为 [commands.sh](../../../../workloads/moe/runs/hopper-stage-5/independent-20261108/commands.sh)，默认新建 `independent-recheck`，包含基础测试、专家地址、延迟副本、回归、写确认隔离和只读事件检查。原始产物留runs，实际验收结论只维护本文。

## 提交归档

按用户要求先归档阶段5，**不将提交视为完整验收通过**。内层实现提交为 `5f74b472599c8234961a202cd11f12caa628e494`；外层配套提交保存五份测试及实现/验收文档。上文基准和未提交状态描述的是验收发生时的版本，归档后以本节及Git记录定位。

提交的模拟器差异与被验diff逐字节一致，SHA256仍为 `64eea12802810c0a94172964627ae24c442f7d5461fd7bdc1f173504577a3e7b`；归档前后独立检查均通过。只同步文档状态及归档信息，没有改变功能、配置、输入或预期，因此不重复构建/模拟。本地检查脚本改为相对运行记录中的内层基准核对差异，提交后仍可复核原始证据。

验收状态继续为**部分通过**：独立搬运项目通过；首次TMA→mbarrier→WGMMA→普通store联测未运行，阶段6单次WGMMA可用后须回到这里补验全部8192项和完成顺序。完整专家、九例前向、真实SASS轨迹及硬件时间精度仍未验。未推送，未启动阶段6。
