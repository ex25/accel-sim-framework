# 阶段 6 实际验收：矩阵、异步 WGMMA 与寄存器额度

状态：**通过**（2026-11-09，限阶段6及阶段5/6联合微测范围）。完整64项结果、当前矩阵布局、连续累加/组等待、寄存器额度、首次TMA联测和双缓冲流水均通过；没有复现需执行agent修复的模拟器错误。阶段5缺失的首次联测现已补齐，可继续阶段7真实专家集成；本轮没有启动阶段7，也不宣称九例或硬件时间精度通过。

本轮修正的是测试：补两个CTA合法启动和第二轮不同输入，并将首次联测屏障从B搬运范围内移到独立位置。最终结论只采用修正后数据；旧日志保留。[唯一联测报告](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/joint-report.md)由阶段5索引，两阶段引用同一份证据。

[验收要求](../../../validation/stage-6.md) · [实现方案](../../../plans/stage-6.md) · [实现记录](../../implementation/stage-6/README.md) · [阶段5验收](../stage-5/README.md) · [全部运行证据](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/)

## 被验版本与运行身份

独立阅读20文件实现、PTX/CPU测试、寄存器读写/依赖/组状态、配置及直接前置记录。复用交付的debug/release构建：内层diff、五份原测试、两个库的散列一致，20份修改源码均早于库生成；因此未重复全量构建。Serena工具未在本会话提供，使用本地检索和只读GDB。

| 对象 | 实际版本及证据 |
| --- | --- |
| 外层 | `3fc74557dcad398bbdf65ce71fcd928cf53c01c4`；开始时已有阶段6实现记录及五份新增测试。本轮只补强阶段6测试和验收文档，保留既有差异；未提交、未推送 |
| 独立模拟器仓库 | `5f74b472599c8234961a202cd11f12caa628e494` 加20文件未提交差异；[inner.diff](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/metadata/inner.diff) SHA256 `7dc8dd8e6b1994e125322692c4f4b3942c1eb0c320c13ee473dda59659096a8d`，验收结束确认未变 |
| 工具链与库 | CUDA12.8 / ptxas V12.8.61、GCC13.4.0。实际库均位于 `gpu-simulator/gpgpu-sim/lib/gcc-13.4.0/cuda-12080/`；debug SHA256 `2b0251ae30fe89385d7b01ab250cc41ed327f824c286bcd7589c902ccd69bc9d`，release为 `cf2f839607efb18bb779f070c35fd05e15e2c096f34ee7e7ffa6e50a5236a3bf` |
| 模式与配置 | `PTX_SIM_MODE_FUNC=0`，日志为performance simulation，实际加载模拟器libcudart；H100两SM副本、1980MHz、详细collector、既有第5专用GMMA单元，latency/init初值64/64。基础trace配置 SHA256 `5824dd03bca6f47846a64979a88d9e4385a489a035cc04130e743067d847a27b` |
| 追溯 | [原交付核对](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/metadata/version-check.json)、[最终测试散列](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/metadata/test-versions.json)、[测试补强差异](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/metadata/validation-tests.diff)。每份run.json保存命令、ldd、源码快照、配置/PTX/程序/库散列 |

验证者未修改模拟器、模型参数、生产配置、冻结程序/输入或容差，未向其它聊天发消息。所有延迟/bank/物理资源诊断差异只保存在运行目录。

## 数值、指令和依赖结果

最终release完整18项、debug首次联测及pipeline/serial通过；另跑GMMA延迟、发起间隔、shared延迟、DRAM延迟和32bank变体。独立检查用数学坐标生成C，再逆向定位线程/tuple，BF16用独立舍入判定；未调用模拟器布局函数。精确核对 **576144 项结果**，连同状态/事件执行 **921303 条检查**：[独立结果](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/independent-checks.json)、[检查脚本](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/verify.py)。计数包含不同配置和重复launch，不是完整workload输出数。

| 项目 | 实际通过的内容 |
| --- | --- |
| 64项解析、decode和依赖 | 非连续寄存器列表及冻结专家 `%f7420…%f7357` 完整逐项比对，涵盖j7/8/31/32/63和实际输入/目标元数据。scale_d=0不功能性读取未初始化D；initial经普通寄存器写入和wgmma.fence后得到C+D0；groups得到X=C、Y=2C |
| 分批写回与最后依赖 | 64bank时每warp64目标分4批，32bank分8批。最后批以前不完成、不解除依赖；只读GDB观察未服务项、完整逻辑目标列表、pending位和真实scoreboard。连续累加有多个写者时第一次release后仍保留依赖；正式组件另检查最后写者释放和普通后继冲突 |
| 独立ldmatrix/stmatrix | `ldmatrix.sync.aligned.x4.m8n8.shared.b16`、对应stmatrix分别用普通shared或直接packed寄存器准备，四张8×8矩阵每元素精确正确。GDB保存每lane四地址，32lane共128个不同4B地址，覆盖恰好512B；不是只验证往返 |
| WGMMA数值和布局 | `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16` 全8192项精确正确。64B/128B swizzle、普通及1280/1296非零起点/相位、跨八行组均通过；K-major未使用LBO。输入以独立16B块表准备，未用被测描述符解码器生成 |
| 四warp与动态身份 | 到达偏斜后才形成同一次读取，compute时arrived/collected均为15；每逻辑矩阵只算一份，262144 math op与四次warp发射分列。同PC循环两轮正确，未混用前轮状态 |
| 异步累加、fence/commit/wait | 两次有序累加得到2C；长延迟时后一条在前条写完前发起。真实PTX wait1在旧组完成后、新组尚未完成时返回，wait0等待新组；组件控制第四warp晚完成，并确认普通cp.async和TMA bulk队列不被矩阵等待排空 |
| setmaxnreg dec40/inc232 | 实际ptxas元数据168/thread、384线程，总64512；两消费者先等待空闲0，生产者归还16384后各获8192，最终40/232/232。384个标记、高虚拟编号及200个存活值正确，邻CTA组件池未变。诊断中启用已有SM资源计数，实际两次退出各将64512归还为0，矩阵CTA状态同时清空 |
| 首次联测与完整流水 | 两CTA、两轮不同输入的TMA→mbarrier→WGMMA→普通store通过；debug/release全部输出逐字节一致。pipeline/serial各轮全量比较16384项BF16位模式且彼此一致，输入/输出缓冲按各自wait复用。详见唯一联测报告 |
| 受影响回归 | 当前源码下阶段5完整15项、三个冻结地址片段及组件通过，阶段2 copy_mix/groups2通过；legacy8覆盖原八寄存器WMMA load.c/store.d路径和普通依赖。无可运行Hopper SASS trace，未称SASS运行回归通过 |

[release-all](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/release-all/run.json)、[两CTA debug](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/joint-cluster-debug/run.json)、[debug流水](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/joint-debug/run.json)、[阶段5](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/stage5-current/run.json)、[冻结地址片段](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/stage5-expert/run.json)、[阶段2](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/regression-stage2/run.json)。阶段5另由独立块表重查155648个16位值和400640字节，检查128个动态搬运/发布操作：[stage5-checks.json](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/stage5-checks.json)。

## 主要时序与资源

| 单参数变化/对照 | 观测及含义 |
| --- | --- |
| GMMA latency64→256，init64 | chain首轮40184→40375周期；前条31476发射、31736完成，后条31617已发射，保留异步累加重叠 |
| init64→128，latency256 | chain首轮40375→40439；两条首次发射间隔141→205周期，没有只改配置而不接资源 |
| 实际wait1 | 长延迟groups第一组31736完成，四warp的wait1均在31736放行，较新组到32018才全部完成 |
| 搬运与计算重叠 | latency256的pipeline当前矩阵6000发射、6260写完，下一块TMA输入6196交付；不是只从CPU函数调用顺序推断重叠 |
| shared29→129 | 最终两CTA integration为14646/14641→15245/15245；pipeline为9817/10220→11209/11592。该参数也影响矩阵shared读取，不能把总差全算作网络 |
| DRAM194→600 | pipeline第二轮10220→10504，有实际DRAM读取；首轮无DRAM读取，周期不变。最终integration每轮重新上传输入，DRAM读数两轮均为0，周期保持14646/14641，不作为DRAM延迟增长证据 |
| register banks64→32 | single64每warp写回4→8批，首轮40043→40049。所有批服务量合计64，完成事件只在最后一批 |
| 相同工作流水/串行 | 9817/10220 对10656/11261周期，数值和位模式一致。代表当前微测的合法重叠，不是H100误差校准 |

详细trace在对应run；[汇总](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/summary.json)逐launch保存到达、读取、发射、写回、wait和TMA交付。`events-chain/`、`events-ldmatrix/`、`events-stmatrix/`、`events-integration/`保存只读GDB事件，JSONL与程序stdout分离以免缓冲交错。`events-quota-resources/`仅追加 `-gpgpu_concurrent_kernel_sm 1`，用于核对实际物理资源归还；不是改变生产配置或调参取得通过。

## 测试修正与证据边界

复核发现原 `integration` 的B目标为 `[16384,32768)`，屏障却放在32000，落入会被TMA完整写入的区域。即便该位置属于计算不用的填充，也不能与活屏障重叠。模拟器屏障状态另存在表中，因而旧测试能退出、数值也正确；这不构成合法程序的通过证据。本轮把屏障移到40000，并加入三个shared区间互不重叠的独立检查。未为非法程序新增模拟器检查或恢复逻辑。

另外，原测试只启动一个CTA、两轮同输入。本轮改为cluster(1,2,1)，rank1正常参加同步；第二轮A加1/B加2，CPU预期同步按定义计算。最终完整18项、debug联测/流水、相关shared/DRAM变体与联测GDB均在修正后重跑。其它不执行integration的独立测试复用同版本有效证据；各sources快照保存准确测试版本。

旧错误地址的初始运行和检查保留在 `before-barrier-fix/`，不纳入最终汇总或通过结论。该问题属于测试输入错误，未复现模拟器在合法输入下出错。修正不改变冻结程序、输入、矩阵容差或阶段标准。新增C++按clang-format16.0.6处理，最终纯空白差异与编译快照的关系保存在版本附件。

仍未覆盖：真实专家FC1/FC2、全部九例、一般浮点数据的冻结golden、真实SASS trace、H200和其它矩阵形状/类型/转置。冻结tuple和描述符标志有静态核对，人工非零布局不冒称完整真实专家执行。详细collector改变了普通指令的服务时间，64/64 GMMA、shared整块交付和远端端点仍是初始近似；阶段7须结合真实主要内核检查工作量、轮询、重叠和时间误差。

上述边界不替代当前已通过的数值、必要同步及主要重叠检查。阶段5/6的必要项目已完成，停止无关扩测；下一阶段由用户指定。

## 本机成本与复跑

设备周期含既有5000周期launch费用；本机墙钟另列。注册为main→注册结束，模拟为提交→同步，比较含CPU检查及写出；pipeline比较区间还含TMA结果下载。部分运行并行，GDB成本未混入下表。

| release样本 | 两次设备周期 | 注册秒 | 两次模拟秒 | 两次比较秒 | 总墙钟秒 / RSS KiB |
| --- | --- | ---: | --- | --- | --- |
| single64 | 40043 / 55884 | 0.544 | 7.2136 / 10.0156 | 0.0037 / 0.0038 | 17.82 / 297540 |
| integration | 14646 / 14641 | 0.544 | 3.7099 / 3.6083 | 0.0041 / 0.0041 | 7.92 / 314224 |
| pipeline | 9817 / 10220 | 0.556 | 1.5104 / 1.6056 | 0.0069 / 0.0069 | 3.73 / 329968 |
| quota | 11237 / 11771 | 0.557 | 1.8105 / 1.9060 | 0.0003 / 0.0003 | 4.32 / 301260 |

原始time-v和HOST_TIME均保留，完整数据见汇总。没有真实尺寸专家计算样本，不用这些微测更新九例预算，也不给H100约20%时间误差结论。复跑见 [commands.sh](../../../../workloads/moe/runs/hopper-stage-6/independent-20261109/commands.sh)，默认写入新目录，包含独立矩阵、首次联测、完整流水、受影响回归、时序变体和只读观察。原始产物留runs，实际状态只维护本阶段记录，并由阶段5链接同一联测报告。

## 提交归档

按用户要求归档本阶段实现、测试和阶段5/6联合验收。内层实现提交为 `b62607065669e4ffcf4d304d9959c29f9085c092`；外层配套提交保存五份阶段6测试及两阶段实现/验收文档。上文基准、未提交状态和各run.json描述验收发生时的版本；归档不改变原始证据。

内层提交内容与被验diff逐字节相同，SHA256仍为 `7dc8dd8e6b1994e125322692c4f4b3942c1eb0c320c13ee473dda59659096a8d`，测试散列也完全未变。仅补归档说明、澄清旧自测的合法性边界；没有修改通过验收的模拟行为或测试预期，因此不重复构建/模拟。独立检查脚本按运行记录中的内层基准核对完整差异，提交后仍能验证原始结果。

通过结论限阶段5/6所列指令、微测、首次联测、双缓冲及受影响回归；完整真实专家、九例前向、真实SASS轨迹及硬件时间精度仍未验。旧屏障重叠运行继续保留并排除在最终结论外。未推送，未启动阶段7。
