# 阶段 7 实际实现：真实专家与完整九例

状态：**已实现并交付三个128-token单卡用例，独立验收已在该范围通过；原九例计划尚未全部完成。**
**Qwen64/4096、Qwen64/8192、Phi/4096、Phi/8192、Mixtral/4096、Mixtral/8192这六个大用例没有运行模拟。**
本轮基准为外层`a3719197c0eddacaf84d7f5e8217e0a996eb2e3e`、内层`b62607065669e4ffcf4d304d9959c29f9085c092`。
以下保留开发过程中的失败、自测及修复历史，当前结论以最新验收和提交归档为准。

[实现方案](../../../plans/stage-7.md) · [验收要求](../../../validation/stage-7.md) · [独立验收记录](../../validation/stage-7/README.md)

## 当前进度

Qwen64/128、Phi/128、Mixtral/128在最终固定release库上均完成全部22次前向调用，共1,310,720项输出0项超原容差；原209处超差均恢复参考。gating、FC1、FC2主要调用误差均在约6%以内。数值、共享读取/临时复制降低、stream身份及完整前向均已复验。小kernel/L2近似和端到端模型范围保留，按用户约定结束单卡精调；六个4096/8192大用例没有运行模拟，多卡尚未实施。见[独立验收](../../validation/stage-7/README.md)和[归档结果表](../../../../workloads/moe/evidence/hopper-stage7-128/README.md)。

| 对应计划 | 实际改动 | 原因及行为 |
| --- | --- | --- |
| 加载及消费者前置 | 内层 `ptx.l`、`cuda-sim.cc`、`ptx_ir.*` | 接受冻结PTX的整数U后缀；按符号身份解析局部标签，避免不同内联块的同名LAB_WAIT/DONE混用。此修复归属阶段0。另将trap连接到退出节点，避免断言失败路径在控制流分析中越过函数末尾。 |
| FC1/FC2独立预期 | 新增 `workloads/moe/tests/hopper/stage-7/expert.py`、`expert.cc` | 复用阶段0 fatbinary注册与阶段2原forward读取；保存两份原1344B参数、分组尺寸及描述符编码来源，执行原模块16。CPU整数乘加独立生成全部BF16预期并检查输出域外哨兵。小形状仅作功能证据，不用于硬件误差。 |
| 逐调用设备时间及成本 | 内层 `abstract_hardware_model.h`、`gpu-sim.cc`、`shader.cc` | 从首个CTA实际驻留至内核最后完成记录统一全局周期、频率、形状、逻辑WGMMA数和主机单调时间；排除launch延迟，保留核内等待。复用原周期模型和完成点。 |

## 自测证据

运行产物保存在 [本轮目录](../../../../workloads/moe/runs/hopper-stage-7/)。原程序、PTX和输入未修改。构建采用既有环境（CUDA12.8、GCC13.4）与debug `make -j8`。

- 原程序首次诊断：[run.json](../../../../workloads/moe/runs/hopper-stage-7/qwen128-first/run.json)、[行缓冲定位](../../../../workloads/moe/runs/hopper-stage-7/metadata/loading-linebuffered.log)。加载到模块106时遇到 `0x3340U` 等常量，被旧词法器中止。
- 专家首次加载：[日志](../../../../workloads/moe/runs/hopper-stage-7/expert-small-fc1-v2/expert.log)。局部标签不在函数顶层符号表，旧汇编器断言失败；同时核查出同名标签必须按作用域区分。

## 已通过的必要自测

- [作用域与U后缀回归](../../../../workloads/moe/runs/hopper-stage-7/loading-debug-v3/run.json)：两个内联块复用LAB_WAIT/DONE，各自循环并消费十六/十进制U常量，32个输出正确。开发版曾误用保留名字 `%tid`，已修正；失败目录保留但不作为通过证据。
- [阶段0回归](../../../../workloads/moe/runs/hopper-stage-7/regression-stage0-debug/run.json)：八项符合预期，包含assert/trap的明确失败出口。
- [阶段6相关回归](../../../../workloads/moe/runs/hopper-stage-7/regression-stage6-debug/run.json)：single64、chain、integration、pipeline、ldmatrix、stmatrix、legacy8及状态组件通过。两次launch设备周期与此前记录完全一致；例如integration 14646/14641、pipeline 9817/10220、single64 40043/55884。
- [小形状FC1](../../../../workloads/moe/runs/hopper-stage-7/expert-small-fc1-release/run.json)：原63个非空分组及132CTA启动，M沿原数据，N=256/K=128；131072项BF16精确一致，128项/组域外哨兵未改变。逻辑WGMMA=2016，设备执行18901周期，主机模拟约255秒，带详细事件；属于功能证据。
- [小形状FC2的早期运行](../../../../workloads/moe/runs/hopper-stage-7/expert-small-fc2-release-before/run.json)：N=128/K=256，65536项及哨兵精确一致，逻辑WGMMA=4032，设备执行20378周期。该运行在仲裁优化前构建的release库上执行，启动时源码已开始优化，库SHA与源码diff不完全对应，优化后重跑见下，早期运行不作为最终版本通过证据。

## 运行成本与等价优化

两次只读调用栈采样定位到寄存器bank读取仲裁，见[采样2](../../../../workloads/moe/runs/hopper-stage-7/metadata/expert-small-sample2.txt)、[采样3](../../../../workloads/moe/runs/hopper-stage-7/metadata/expert-small-sample3.txt)。旧wavefront实现只给每bank队首建立一个请求，且已经取消collector输出互斥，所以最终总会授予所有无写占用的bank。改为直接按bank取队首，保留写优先、FIFO及每周期每bank一次服务，删除随之无用的仲裁矩阵。不改变设备模型参数或结果；上述七项相关回归周期完全相同。此修改涉及SASS共用寄存器服务，未有可运行trace，不能称SASS运行验收通过。

原始debug完整前向和两份debug小专家诊断已被release运行替代并显式中止，保留日志，不算数值通过或完整耗时样本。[debug构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-arbiter-debug.log)、[release构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-arbiter-release.log)均完成。小形状只用于功能与等价性检查；真实尺寸成本见文末，不用小形状外推九例时间。

## 后续与限制

用户当前要求暂缓两项数值分歧，优先用小kernel解释并修复gating偏差。本轮已完成读取合并修复及小测试、原尺寸独立gating和相关回归，证据见文末；没有启动耗时数小时的完整MoE。原容差与阶段验收边界保持，后续完整前向和其余八例另行推进，不能把独立测试通过写成完整阶段通过。

## 本轮加载故障的后续修复

原完整前向完成前7个调用后，在模块75建立控制流时崩溃：[失败运行](../../../../workloads/moe/runs/hopper-stage-7/qwen128-release/run.json)。其正常路径ret，断言失败路径以trap结束，末尾跟调试标签。`connect_basic_blocks`原本只将ret/exit当终止指令，因而读取trap后不存在的指令。现将trap同时作为基本块终点和退出路径；不跳过断言或trap执行。

- [原模块75定向自测](../../../../workloads/moe/runs/hopper-stage-7/normalize-debug/run.json)：权重8192项、标记8192项全部精确；[作用域/U后缀/末尾trap组合](../../../../workloads/moe/runs/hopper-stage-7/loading-trap-debug/run.json)32项正确；[trap实际失败回归](../../../../workloads/moe/runs/hopper-stage-7/trap-runtime-debug/run.json)保留明确非零失败。
- [FC2仲裁优化后重跑](../../../../workloads/moe/runs/hopper-stage-7/expert-small-fc2-release/run.json)：65536项及全部域外哨兵精确，20378周期与优化前相同；实际模拟240.22秒，进程248.19秒，峰值RSS约3.69GiB。
- 小形状测试进一步将参数1208的调度tile总数按实际M/N和双CTA列簇重新计算；正式参数、原PTX和输入不改。新测试另外保存`abi.json`、原/实际参数字段表和完整实际尺寸，旧小形状结果保留为早期证据。[FC1最终测试](../../../../workloads/moe/runs/hopper-stage-7/expert-small-fc1-final/run.json)已精确通过；[FC2最终测试](../../../../workloads/moe/runs/hopper-stage-7/expert-small-fc2-final/run.json)也已精确通过。
- [阶段2复制回归](../../../../workloads/moe/runs/hopper-stage-7/regression-stage2-release/run.json)的copy_mix、groups2及组状态组件通过。
- 最新[debug构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-trap-debug.log)和[release构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-trap-release.log)成功。

## 当前性能观察与入口

[原Qwen64/128前一次运行](../../../../workloads/moe/runs/hopper-stage-7/qwen128-trapfix/run.json)完成专家之前16个调用后遇到下述回跳问题，已显式中止，不算完整通过。逐调用对照工具为 `python3 workloads/moe/tests/hopper/stage-7/summarize.py <run-dir>`，按launch位置核对entry、grid/block/cluster，并保存设备起止、逻辑矩阵工作量、同调用ncu/nsys与访存数据。DRAM累积计数按相邻完成点作差；L2按kernel UID的GLOBAL/CHIPLET数据请求分列，不把组播shared写入量当DRAM字节。

首个完整运行的gating为334757设备周期，即0.16907ms，对ncu0.08512ms约+98.6%。这一主要差额尚未解释，不能声明性能支持完成。早期真实FC1的只读采样在83936周期时已生成33792次逻辑WGMMA；该次运行后来中止，不作为成本样本。完成运行的工作量与成本见文末。硬件perf采样被本机perf_event权限拒绝，未修改系统设置，改用独立诊断测试上的只读调用栈。

## 原专家第二轮的取指修复

原模块16的最后一条物理指令是回跳分支。功能执行已将SIMT分支栈移到正确目标，但取指使用的预测地址仍是函数末尾之外。旧decode遇到空指令不生成有效buffer项，使scheduler已有的空指令恢复分支永远不能触发，部分warp停住、其它warp轮询。证据见[双轮停住状态](../../../../workloads/moe/runs/hopper-stage-7/metadata/two-tiles-stack.txt)及[分支/目标](../../../../workloads/moe/runs/hopper-stage-7/metadata/two-tiles-targets.txt)。

`shader_core_ctx::decode`现在将空取指作为一个有效空buffer项，使用现有scheduler控制流恢复和流水计数；不改分支目标、异步等待或设备延迟。该前置修复归属阶段0。

- [末尾回跳回归](../../../../workloads/moe/runs/hopper-stage-7/loading-tailbranch-debug/run.json)：32项精确，包含作用域/U后缀和不执行的trap路径。
- [原专家双轮FC1](../../../../workloads/moe/runs/hopper-stage-7/expert-two-tiles-fixed/run.json)：原63组及132CTA，N=512/K=128，262144项及全部哨兵精确；进程348.72秒，RSS5359664KiB。此前[同形状停住运行](../../../../workloads/moe/runs/hopper-stage-7/expert-two-tiles/run.json)停在2112次逻辑WGMMA，已中止，不能作为完整成本样本。
- [阶段6相关回归](../../../../workloads/moe/runs/hopper-stage-7/regression-tailbranch-stage6/run.json)：七项及状态组件全部通过。
- 阶段1的numeric/collectives/bits_loads通过；同次命令误填三个不存在的test名，故[原run.json](../../../../workloads/moe/runs/hopper-stage-7/regression-tailbranch-stage1/run.json)含三项入口错误，不能写成全通过。正确的warp_full/warp_partial/legacy已[另行通过](../../../../workloads/moe/runs/hopper-stage-7/regression-tailbranch-warp/run.json)。
- [debug构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-tailbranch-debug.log)、[release构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-tailbranch-release.log)成功；测试run.json记录精确库SHA、源码差异和配置。

旧[真实FC1](../../../../workloads/moe/runs/hopper-stage-7/expert-real-fc1/run.json)停在33792次WGMMA，35分钟后中止，不作代表性完成预算。[修复后真实FC1](../../../../workloads/moe/runs/hopper-stage-7/expert-real-fc1-tailfix/run.json)及[真实FC2](../../../../workloads/moe/runs/hopper-stage-7/expert-real-fc2-tailfix/run.json)均已完成并精确通过；完整成本见文末。

## FP32吞吐核查

当前H100/H200的四个FP32分区原发起间隔为2周期，只提供64个线程结果/SM/周期。NVIDIA[Hopper调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#improved-fp32-throughput)及[CUDA12.8吞吐表](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#arithmetic-instructions)给出SM90 add/mul/FMA为128结果/SM/周期，而min/max及INT32仍为64。只将PTX FP32 ADD/MUL/MAD间隔改为1，保留MAX、DIV、INT、延迟及其它资源。此项是硬件资源纠正，尚不能仅凭理论值宣称解释了gating差额。

新增`fp32.ptx`有4warp、每线程16个独立FMA累加器：[原配置](../../../../workloads/moe/runs/hopper-stage-7/fp32-before/run.json)和[纠正后](../../../../workloads/moe/runs/hopper-stage-7/fp32-after/run.json)的2048结果全部正确，设备周期7982→6471。阶段2真实gating的[原配置](../../../../workloads/moe/runs/hopper-stage-7/gating-fp32-before/run.json)与[纠正后](../../../../workloads/moe/runs/hopper-stage-7/gating-fp32-after/run.json)均精确通过：Qwen331840→321592周期，Mixtral354074→325403周期。这项纠正只解释约3%/8%的设备时间，不能解释原先近两倍的Qwen gating偏差。原正式前向、输入和容差未变。

## 当前版本的流水与主机成本检查

真实尺寸专家的短采样见[调用栈](../../../../workloads/moe/runs/hopper-stage-7/metadata/real-tailfix-profile.txt)，显示周期推进、矩阵寄存器写回及指令访存地址的分配/复制成本。后者原本每线程最多8地址，却为每个线程分配动态数组，并在合并请求时再复制一份。现沿用原8地址上界，改为内嵌数组+实际长度，合并请求直接读取；跨sector处理仍用局部地址副本，不修改原始地址。没有改变地址顺序、请求或设备服务时间。

- [阶段6相关回归](../../../../workloads/moe/runs/hopper-stage-7/regression-address-stage6/run.json)：integration、pipeline、ldmatrix、stmatrix、legacy8及状态组件通过；[阶段2](../../../../workloads/moe/runs/hopper-stage-7/regression-address-stage2/run.json)copy_mix/groups2及组状态组件通过。
- [FC2](../../../../workloads/moe/runs/hopper-stage-7/expert-address-fc2/run.json)：65536项及哨兵精确，20318周期，进程227.06秒、RSS3721308KiB。此版本已使用纠正后的FP32配置，不能把与旧配置相差60周期误归因于地址存储。
- 同一纠正后配置下，[地址优化前双轮FC1](../../../../workloads/moe/runs/hopper-stage-7/expert-pipeline-context/run.json)与[优化后](../../../../workloads/moe/runs/hopper-stage-7/expert-pipeline-final/run.json)的262144项及哨兵均精确，逻辑WGMMA均4032，设备周期均25743。主机模拟356.00→309.58秒，进程364.09→317.68秒；这只是小形状功能/等价性样本，不用来承诺九例成本。

异步事件沿用已有PTX_TMA日志，新增`PTX_CONTEXT`把动态指令UID与kernel、SM、CTA、cluster/rank、warp、PC和周期关联，并补输入barrier及WGMMA描述符。正式前向不开该组件。检查入口：

```bash
python3 workloads/moe/tests/hopper/stage-7/pipeline.py \
  workloads/moe/runs/hopper-stage-7/expert-pipeline-final
```

[流水结果](../../../../workloads/moe/runs/hopper-stage-7/expert-pipeline-final/pipeline.json)检查原模块cluster0双CTA的24次输入交付、64次逻辑矩阵读取、256次stmatrix消费及384次输出缓冲复用：对应数据交付和屏障成功先于矩阵读取，shared读取服务结束后才dispatch，寄存器写回先于stmatrix，输出read完成后才复用源。32次输出中16次完全越界，无global写请求；其余均有最终确认。所有cluster最后一次TMA确认29762周期，内核30744周期结束。该调用保留原132CTA资源启动，形状缩小；真实尺寸流水证据仍待代表性调用完成。

## 原完整前向的最新状态与交接入口

新增`forward.py`只顺序调用原`run.py`、原`compare.py`和逐调用摘要，固定原容差，数值失败立即停止。它在每例启动前保存双方基准、工作区差异、实际库/配置散列，运行和比较分别记录time-v。默认顺序沿计划九例，可用`--cases`只选择当前待排查项，不改变冻结workload。

```bash
source workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh
source gpu-simulator/gpgpu-sim/setup_environment release
python3 workloads/moe/tests/hopper/stage-7/forward.py \
  --run-dir workloads/moe/runs/hopper-stage-7/nine-final
```

最终构建的[阶段0八项回归](../../../../workloads/moe/runs/hopper-stage-7/regression-final-stage0/run.json)、[阶段1六项相关回归](../../../../workloads/moe/runs/hopper-stage-7/regression-final-stage1/run.json)及[末尾回跳/作用域/U组合](../../../../workloads/moe/runs/hopper-stage-7/loading-final/run.json)全部符合预期；两个仓库`git diff --check`通过。

### 首次完整运行：执行完成，数值失败

[原Qwen64/128](../../../../workloads/moe/runs/hopper-stage-7/forward-final/qwen64-t128/run.json)已完成全部22次调用、注册110份模块并出现ROI结束标记。原[comparison.json](../../../../workloads/moe/runs/hopper-stage-7/forward-final/qwen64-t128/comparison.json)检查262144项，其中259379项超限，最大绝对误差0.0855712890625，相对L2为1.19712。退出码0只表示程序运行完，不能代表结果正确。

数值标准保持 `abs(actual-ref) <= 1e-5 + 0.02*abs(ref)`：容差是允许的误差上限，并非测到的误差；例如参考值0.01时允许相差0.00021。未修改冻结程序、PTX、初始化、golden或此标准。

[逐调用摘要](../../../../workloads/moe/runs/hopper-stage-7/forward-final/qwen64-t128/stage7-summary.json)的设备时间仅作失败运行的诊断：

| 调用 | 模拟设备毫秒 | ncu毫秒 | 有符号差额 | 逻辑WGMMA | 本机模拟秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gating | 0.163917 | 0.085120 | +92.57% | 0 | 217.78 |
| FC1 | 0.274413 | 0.267680 | +2.52% | 354816 | 8480.90 |
| FC2 | 0.145860 | 0.137728 | +5.90% | 177408 | 4467.14 |

全部22次调用的[完整表格与交互对比图](../../../../workloads/moe/runs/hopper-stage-7/comparison-forward-final/comparison.html)、[逐行CSV](../../../../workloads/moe/runs/hopper-stage-7/comparison-forward-final/comparison.csv)、[表格图片](../../../../workloads/moe/runs/hopper-stage-7/comparison-forward-final/comparison-table.png)及[耗时/偏差图](../../../../workloads/moe/runs/hopper-stage-7/comparison-forward-final/timing-comparison.png)已从上述摘要导出，统一使用修复前失败版本的数据，不混用本次尚未完成的复验结果。

内核总和0.612874ms，对ncu0.561952ms为+9.06%；总和接近不能掩盖主要调用gating的差额，且本次数值失败，不能写性能验收通过。gating执行1330528条PTX warp指令，硬件对应1019680条SASS指令；PTX拆包/位转换和shared服务等待尚未定量解释全部差额。不能拿条数比例调延迟。`compare.py`的CUDA-event ROI字段另有主机间隙，不能替代上述设备口径。

### 首个数值分歧：32位地址读到了错误位置

原初始化的路由转换和gating全部精确通过；top-k首次产生错误。原模块3用32位寄存器计算shared地址。`add.s32 r, 8, -4`的低32位正确为4，但模拟器更宽的寄存器存储保留了高位进位。`ptx_thread_info::get_operand_value`把整个64位存储当作地址，于是从`0x100000004`读取，实际应该读`0x4`。第四个入选专家由此读到错误前缀计数，覆盖已有槽位，后面的路由、计数与专家输出随之错误。

[修复前定向运行](../../../../workloads/moe/runs/hopper-stage-7/routing-actual-qwen-v3/stdout.log)复现top-k错误；[临时地址抓取](../../../../workloads/moe/runs/hopper-stage-7/routing-topk-diagnose/stdout.log)保留正确位置4已写入3、随后却从`0x100000004`读到0的证据。原token0应选专家17/5/30/34，却丢掉5并留下未写槽位。临时逐指令输出已经删除。

修复在内层`src/cuda-sim/instructions.cc`的公共访存操作数求值处：按地址寄存器声明宽度截断32位地址及其偏移，64位地址沿用原行为，独立于本条load/store的数据宽度。这是阶段1的必要寻址修复，不在top-k或MoE层增加特判，也不改变设备资源参数。

新增`routing.cc`/`routing.py`复用冻结模块70/71/3/4/72～77、原launch参数和Qwen64/128初始化配方，独立CPU生成gating与前四名预期，检查完整路由链。可用`cpu-scores`跳过已单独核对的gating以缩短定位；该选项只用于定向自测，不用于正式前向或性能结论。

- [原始输入完整路由](../../../../workloads/moe/runs/hopper-stage-7/routing-address32-full/run.json)：转换131072项、gating8192项精确一致，随后全部top-k/排序/指数/归一化/标记及专家计数通过；本次实际执行原模块70/71，不使用CPU scores替代。
- [修复后定向路由](../../../../workloads/moe/runs/hopper-stage-7/routing-address32-fixed/run.json)：512项top-k值/专家编号配对及排序、512项指数、8192项路由权重和两种标记、64项专家计数全部通过；该次使用独立CPU scores。
- 阶段1 `generate_micro.py`的bits_loads新增64线程shared地址回归：地址经加正数再加负数，分别通过正常32位地址和带内部进位的32位地址读回64位数据。[release](../../../../workloads/moe/runs/hopper-stage-7/address32-micro-release/run.json)与[debug](../../../../workloads/moe/runs/hopper-stage-7/address32-micro-debug/run.json)均通过；同项也覆盖原64位global地址和字节读取。
- [阶段1相关回归](../../../../workloads/moe/runs/hopper-stage-7/address32-regression-stage1/run.json)：numeric、collectives、warp_full、warp_partial、legacy全部通过；[阶段6相关回归](../../../../workloads/moe/runs/hopper-stage-7/address32-regression-stage6/run.json)：integration、ldmatrix、stmatrix、legacy8及状态组件通过。
- [debug构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-address32-debug.log)、[release构建](../../../../workloads/moe/runs/hopper-stage-7/metadata/build-address32-release.log)均完成。最终release库SHA256为`87c507e0676a33d4fbcb8a8ca9e7d26df45140b2ed026e72070588456ca`，对应内层diff SHA256为`3beede30823ca6e76e7f25c9f8d1e7254f1822505f0a2b35e1897d6e3d497159`；本段自测和完整复验均记录该构建。

[修复后原完整前向](../../../../workloads/moe/runs/hopper-stage-7/forward-address32-fixed/qwen64-t128/run.json)已结束，构建与配置散列见[build.json](../../../../workloads/moe/runs/hopper-stage-7/forward-address32-fixed/qwen64-t128.build.json)。原二进制、真实输入、原比较器均不变；比较仍有2项超限，继续停止其余八例，详见文末最新结果。

### 已完成真实尺寸样本与成本

| 样本 | 独立数值结果 | 逻辑WGMMA | 进程墙钟 | 峰值RSS |
| --- | --- | ---: | ---: | ---: |
| [FC1真实尺寸](../../../../workloads/moe/runs/hopper-stage-7/expert-real-fc1-tailfix/run.json) | 1441792项BF16精确，域外哨兵通过 | 354816 | 2:35:47 | 10449800KiB |
| [FC2真实尺寸](../../../../workloads/moe/runs/hopper-stage-7/expert-real-fc2-tailfix/run.json) | 1048576项BF16精确，域外哨兵通过 | 177408 | 1:09:38 | 9864252KiB |
| [首次完整前向](../../../../workloads/moe/runs/hopper-stage-7/forward-final/qwen64-t128.time) | 完成22次调用，最终数值失败 | 上述两次合计532224 | 3:51:21 | 13204284KiB |

两个真实尺寸专家样本使用独立小整数输入，验证原模块、原分组及完整N/K尺寸，不能称为原始权重的golden通过。FC1库SHA为`ffc08ed1…`，其后发生FP32资源及等价地址存储优化；FC2库SHA为`42498524…`，两次均早于本次32位地址修复，不冒称最后构建的结果。对应run.json/inner.diff/config记录精确版本；地址修复后的原前向负责当前版本真实数据复验。

当前有效预算只有Qwen64/128尺寸：完整重跑参考约4小时，比较约0.23秒；该数值失败样本的耗时可用于同尺寸排期，不能作为支持通过。初始化/其它调用/进程开销约占两专家模拟之外的11分钟，未单独测出的部分不伪填初始化值。剩余八例的形状、有效行填充和轮询比例不同，须先取得Qwen64/4096样本再修订预算，不将token倍数直接乘成保证；另预留至少一次失败定位与同尺寸重跑。

当前尚未达到阶段7完成边界：修复后首例仍有2项数值超限；其余八例全量数值、真实尺寸动态流水、gating主要时间差额、4096的nsys交叉检查和剩余预算均待完成。独立验收记录保持原状态，不填“通过”。

定向复验入口（先激活既有in-switch/GCC13.4/CUDA12.8环境）：

```bash
source gpu-simulator/gpgpu-sim/setup_environment release
python3 workloads/moe/tests/hopper/stage-7/routing.py \
  workloads/moe/runs/hopper-stage-7/routing-recheck
python3 workloads/moe/tests/hopper/stage-1/run.py \
  --run-dir workloads/moe/runs/hopper-stage-7/address-recheck --test bits_loads
python3 workloads/moe/tests/hopper/stage-7/forward.py \
  --run-dir workloads/moe/runs/hopper-stage-7/forward-recheck --cases qwen64-t128
```


## 小内核时间偏差的定向排查

用户要求以数据和代码解释小内核的巨大负偏差。本轮只运行最小的转换/清零诊断，正式前向继续使用原配置和同一库；没有按模型名或内核名调延迟，也未把下列另一时间范围替换原计划的设备执行指标。

### 已确认：前表排除了模型内已有的启动等待

前表的`SIM_KERNEL`起点在`issue_block2core`首CTA驻留处；原`gpu_sim_cycle`统计区间还包括CTA之前的等待。本次H100配置的`gpgpu_kernel_launch_latency=5000`进入`kernel_info_t::m_kernel_TB_latency`，`select_kernel`只在其归零后选出内核。`decrement_kernel_latency`逐周期减计数。因此前表排除的并非一个未建模开销，而是模型已等待、但统计范围未包含的一段时间。`stream_manager.cc`另有主机侧launch计数，不应与此设备逐周期等待混为一谈。

从修复前完整日志逐项提取，全部22次调用的原`gpu_sim_cycle`均比前表多5002周期，即2.526263μs；5000来自配置，另2是实测的区间边界差，本轮不为低影响的2周期继续扩测。修复后已完成的前16次调用也有同一差值。[两种时间范围完整表](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/timing-scopes.md)和[含修复后前缀的CSV](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/timing-scopes.csv)保留原数据，未覆盖前一版表。

| 修复前调用 | 首CTA后μs（前表） | 包含原启动等待的区间μs | ncu μs | 前表偏差 → 另一范围偏差 |
| --- | ---: | ---: | ---: | ---: |
| 路由权重转换 | 0.625 | 3.151 | 3.168 | −80.28% → −0.54% |
| 路由权重清零 | 0.188 | 2.715 | 2.752 | −93.15% → −1.36% |
| 路由标记清零 | 0.179 | 2.705 | 2.560 | −93.02% → +5.67% |
| dispatch | 1.126 | 3.652 | 3.584 | −68.59% → +1.90% |
| SwiGLU | 1.873 | 4.399 | 4.128 | −54.63% → +6.56% |
| combine | 4.260 | 6.786 | 4.320 | −1.39% → +57.09% |

原首CTA后总和612.874μs，包含原启动等待的区间合计668.452μs；19个非gating/FC1/FC2调用按算术比较落在±20%的数量由1变为15。这不是新的性能通过结论：正式旧运行本身数值失败，硬件数据没有首CTA时间戳，不能宣称5000周期恰好等于硬件的某一段开销。它证明前表的−93%不能直接解释为执行资源少算了93%，并显示combine等残余差额仍需另查。

### 对照实验：启动等待、取指/常量缓存、L2完成路径

诊断使用冻结模块70/73/74和原grid/block/参数，对转换131072项、两项清零各8192项检查全部输出。复用已有fatbinary及注册/launch辅助函数，源码为[scope.cc](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/scope.cc)，驱动为[run_scope.py](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/run_scope.py)，详细结果为[scope-results.json](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/scope-results.json)。五组实验全部输出正确：

| 只改变的选项 | 权重清零首CTA后周期 | 原统计区间周期 | 能证明什么 |
| --- | ---: | ---: | --- |
| 启动等待0周期 | 373 | 375 | 执行段不含启动等待 |
| 原启动等待5000周期 | 373 | 5375 | 与完整前向清零周期一致 |
| 启动等待10000周期 | 373 | 10375 | 统计范围差额随配置精确变化 |
| 关闭理想取指/常量缓存，其余沿原配置 | 1426 | 6428 | 缓存近似确实压短执行段 |
| L2延迟237→337，其余沿原配置 | 473 | 5475 | 访存延迟实际参与内核完成 |

配置`gpgpu_perfect_inst_const_cache=1`使`shader_core_ctx::fetch`直接把取指当HIT，使`ldst_unit::constant_cycle`直接移除常量/参数访存请求；真实缓存未命中没有走服务路径。只关闭这一开关，转换1036→2533周期、权重清零373→1426、标记清零354→676，说明这是可测量的模型近似。诊断转换的地址/缓存前序与正式前向不同，只比较同一诊断的前后值，不混用其1036与正式1237周期。当前没有对应硬件miss成本分解，不能把这组差值写成对真实误差的精确归因，也没有据此翻转正式配置。

增加L2延迟100周期后，两项清零均恰好多100周期；加上全量输出检查，可排除这两项在当前测试中完全跳过global写服务的解释。代码中`hardware_done`要求`stores_done`且流水为空，写请求确认通过`store_ack`减少未完成计数。H100 L1写策略为write-through，相关路径不会按write-back本地完成来提前确认。本轮只证明已测的三项与对应代码路径，不外推全部小内核正确。

### 硬件采集条件与剩余缺口

按entry及grid/block匹配三个4096用例已有硬件记录，筛选ncu小于10μs的37次调用，ncu−nsys测量均值的中位数为1.418667μs，范围1.162667～2.058667μs；改用nsys首次观测，中位数仍为1.408μs。Qwen4096两项清零的ncu分别2.496/2.560μs，nsys测量均值1.141/1.152μs。[逐项配对CSV](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/profiler-pairs.csv)由现有冻结数据生成，没有重新采样或改写基线。

ncu为无预热、application replay三pass、cache/clock control none；nsys为六次前向中前三次预热、后三次测量，两者时间戳定义也不同。所以上述是已有测量条件差异，不能全部叫“profiler开销”，也不能从每核统一扣1.4μs。当前小核有效频率为1952.76～1991.39MHz，与配置1980MHz最大相差1.38%，不足以解释原50～90%的差额。

本机`nvidia-smi`不能连接NVIDIA驱动，不能补采同条件的空内核/首CTA/末CTA硬件时间戳与指令/常量缓存计数。下一步有硬件时先补这类边界数据，再判断启动模型与缓存近似如何分配；combine、softmax、首步行映射等残余偏差尚未解释，不通过改统一偏移使它们全部达标。当前原Qwen64/128仍在FC1，未因本轮诊断重启。

后处理`stage-7/summarize.py`已补充`simulation_interval_cycles/ms`和`outside_cta_interval_cycles`，总表另列`kernel_sum_simulation_interval_ms`；原`cycles/sim_ms/ncu_relative_error`定义和值全部保持。使用旧完整日志检查22行新增值与原计数完全一致，并逐字段核对原数值、时间、流量无变化；当前运行日志仍明确输出partial 16/22。此改动只影响摘要，不修改正在运行的模拟器库、周期推进或正式配置。

[汇总数据](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/diagnosis.json)、[构建/库散列](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/build.json)、[分析入口](../../../../workloads/moe/runs/hopper-stage-7/small-kernel-diagnosis/analyze.py)记录了确切输入、条件和限制。该诊断使用同一release库`87c507e0…`和内层diff`3beede30…`，正式模拟器源码与配置未改动。


## 完整前向时间与逐核合计的区别

用户追问表格“合计”是否端到端时间。核对后确认：`kernel_sum_sim_ms`、`kernel_sum_simulation_interval_ms`和`kernel_sum_ncu_ms`均直接累加22次调用，不能命名为端到端时间。

| Qwen64/128已有记录 | 时间μs | 统计范围 |
| --- | ---: | --- |
| 硬件ncu逐核合计 | 561.952 | 22个GPU kernel持续时间之和 |
| 硬件完整前向CUDA event，预热后 | 559.203 | 3次预热后，三组各10次前向的组均值取中位数 |
| 硬件完整前向CUDA event，未做软件预热 | 6283.780 | 另一次完整前向测量，不保证硬件缓存为空 |
| 旧模拟首CTA后逐核合计 | 612.874 | 排除逐核启动前等待 |
| 旧模拟包含原等待的逐核区间合计 | 668.452 | 22个`gpu_sim_cycle`相加 |
| 旧模拟完整前向区间的累计GPU周期差 | 668.452 | ROI前后`gpu_tot_sim_cycle`之差，1328838−5303=1323535周期 |
| 旧模拟首核首CTA至末核完成跨度 | 665.926 | 全局结束1328837−开始10304=1318533周期；包含已建模核间等待，排除首核启动前等待 |
| 模拟完整前向CUDA event | 未测量 | launcher模拟分支关闭event，显式DMA耗时与真实主机间隙未建模 |

硬件值来自冻结的[performance-baseline.json](../../../../workloads/moe/evidence/performance-baseline.json)。CUDA event围住`forward()`，包含前向区间内部出现的数据拷贝、设备等待及主机供给不及时造成的空隙，排除初始化与最终输出下载；它不是完整模型推理或进程启动到退出。event和ncu来自不同运行，不能由561.952−559.203得到负的“核间开销”，也不能将模拟未预热单次前向与硬件热态event直接判误差。未预热event较慢的原因没有细分证据，不一概归为缓存。

源码依据为`workloads/moe/source/launcher.cu:388–404,416–418`。`workloads/moe/compare.py::timing`原本也计算ROI前后累计GPU周期差，本次值恰好与22个区间之和相同；它不等于硬件CUDA event。旧`comparison.json`中的`roi_time_relative_difference`把该模型范围和硬件未预热event直接相除，不能用于端到端验收。保留历史文件，不重写既有数值。摘要脚本现在单列`forward_timing`：保留硬件两种event基线，仅完整日志生成模拟ROI累计周期差及首末核跨度，并将`simulation_end_to_end_ms`明确记为null及原因。旧22行原字段与原合计逐项保持不变，当前未完成日志不生成完整跨度。以上模拟完整数字仍来自数值失败的旧运行，修复后的完整前向仍在运行；它们只用于说明统计范围，不作为性能验收结论。

## H100配置的官方资料核对

用户要求查权威配置。本轮下载并保存四份NVIDIA官方页面，URL与字节数见[downloads.json](../../../../workloads/moe/runs/hopper-stage-7/h100-config-audit/downloads.json)，内容SHA256见[source-hashes.json](../../../../workloads/moe/runs/hopper-stage-7/h100-config-audit/source-hashes.json)。资料是硬件规格和编程约束，不是NVIDIA发布的GPGPU-Sim周期配置；以下匹配不能替代PTX性能校准。

| 项目 | 当前模型 | 官方资料／冻结硬件采集 | 判断 |
| --- | --- | --- | --- |
| SXM SM数量 | 132 | NVIDIA Hopper架构说明132；冻结H100也是132 | 匹配 |
| 每SM寄存器／线程／warp／block上限 | 65536个32位寄存器／2048／64／32 | Hopper Tuning Guide及冻结device属性相同 | 匹配 |
| L1与共享内存总容量 | 256KiB | Hopper Tuning Guide 256KB | 匹配容量，非验证延迟／结构 |
| 每SM共享内存 | 228KiB | 官方228KB；每block可用227KB，另留1KB | 当前每CTA有1KiB保留；FC1/FC2实际申请含保留216064B |
| L2总容量 | 256sets×128B×20ways×80subpartitions=52428800B | 官方50MB；冻结device同为52428800B | 匹配容量，非验证映射／延迟 |
| 共享内存划分档位 | 0,8,16,32,64,228KiB | 官方另有100,132,164,196KiB | 确认缺项 |
| 核心时钟 | 1980MHz | 冻结device nominal clock 1980MHz | 名义值匹配；真实有效频率随调用变化 |
| DRAM时钟 | 模型3106MHz | 冻结device memory clock 2619MHz，带宽3.35232TB/s；产品页3.35TB/s | 不可称逐项照搬实卡；需按实际模型验证带宽 |
| 启动5000、L1 41、共享29、L2 237、DRAM 194、GMMA 64/64 | 模型参数，单位／路径各异 | 本次官方资料没有逐项给出这些周期值 | 尚未完成硬件校准 |

权威链接：[Hopper Tuning Guide（CUDA 12.8.1）](https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html)、[Hopper架构说明](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)、[H100产品规格](https://www.nvidia.com/en-us/data-center/h100/)、[CUDA编程指南的CC9.0部分](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#compute-capability-9-0)。早期架构文章／Tuning Guide出现约3TB/s，当前产品页为3.35TB/s，冻结device属性给出的3.35232TB/s更直接对应本次基线卡，不能混用不同时间和SKU。

共享档位的实际影响由`shader_core_config::max_cta`决定：按实际最大驻留CTA计算共享用量，再选不小于该用量的最小档位，剩余容量给L1。缺少四档时，用量64KiB以上可能直接选228KiB，使L1变小。本轮按旧完整日志的资源限制、grid及132SM复算全部22项；它们只选8、32、228KiB，补齐四档不会改变这22项的选择。因此它是已确认配置缺项，但不能解释Qwen128当前偏差；更大token用例不能外推。逐项复算见[config-audit.json](../../../../workloads/moe/runs/hopper-stage-7/h100-config-audit/config-audit.json)。本轮保持正式配置与正在运行的配置不变。

内存模型当前启用`gpgpu_simple_dram_model=1`。`memory_partition_unit::simple_dram_model_cycle`按请求队列及固定返回延迟服务，未走详细DRAM行／bank时序路径。因此文件里RCD／CL等参数看起来像HBM3，并不等于正在模拟这些时序；也不能只按buswidth×clock×DDR公式宣布当前总带宽准确或直接将3106改2619。需要同卡流式带宽／延迟测量再校准。

5000周期在每个`kernel_info_t`创建时设定，当前TB额外等待为0；这是每kernel模型启动等待，不是每CTA再加5000，也不是NVIDIA承诺固定5000。当前顺序22次中配置部分共110000周期，按1980MHz为55.556μs；并发运行有重叠时不能直接相加。早期上游Hopper引入提交e10018b6的该参数是3000，当前基准HEAD为5000，进一步说明它不是不可变硬件规格。

理想取指／常量缓存的影响沿前述三项控制实验：关闭后首CTA后执行时间分别增加144.5%、282.3%、91.0%；包含现有5000启动等待的各核区间分别增加24.79%、19.59%、6.01%，绝对差为0.756、0.532、0.163μs。三项全部输出正确，但没有完整22核关闭该开关的运行，也没有对应硬件miss分解，不能把该差值说成完整前向误差。该开关未因本次敏感性检查直接翻转。


## 地址修复后的完整前向已完成：仍有2项超限

当前[run.json](../../../../workloads/moe/runs/hopper-stage-7/forward-address32-fixed/qwen64-t128/run.json)为completed/exit0，22项完成且出现ROI_END；[comparison.json](../../../../workloads/moe/runs/hopper-stage-7/forward-address32-fixed/qwen64-t128/comparison.json)仍为passed=false。262144项中262097项与硬件逐位相同，47项不同，其中45项符合原容差、2项超限。原始大面积寻址错误已消除，但不能因此填写阶段通过。

| 指标 | 地址修复前 | 当前 |
| --- | ---: | ---: |
| 超原容差元素 | 259379 | 2 |
| 不完全相同元素 | 261747 | 47 |
| 最大绝对误差 | 0.0855712890625 | 0.000244140625 |
| 相对L2误差 | 1.197116573 | 0.0000556272554 |

两个坐标均从0计：token35/channel1099参考0.0007171630859375，模拟0.0006866455078125，绝对差3.0517578125e-5，高于允许2.434326171875e-5；token106/channel861参考0.0022125244140625，模拟0.0022735595703125，绝对差6.103515625e-5，高于允许5.425048828125e-5。它们的BF16编码分别相差8与4；不能说所有差异都只有一个最低位。最大绝对差所在元素符合容差。逐项输出见[output-differences.json](../../../../workloads/moe/runs/hopper-stage-7/forward-address32-fixed/qwen64-t128/output-differences.json)，输入和原容差未改。

当前FC1执行544080周期（274.788μs），对ncu267.680μs为+2.66%；FC2执行287739周期（145.323μs），对137.728μs为+5.51%；gating324555周期（163.917μs），对85.120μs仍为+92.57%。22项执行段合计612.533μs，对ncu561.952μs为+9.00%；含原启动等待的区间合计668.111μs，对ncu算术差为+18.89%，并不构成另一口径通过。两专家逻辑WGMMA354816/177408换算运算量均与硬件计数一致，不能据此推导所有结果正确。完整[22核MD表与评估](../../../../workloads/moe/runs/hopper-stage-7/forward-address32-fixed/qwen64-t128/assessment.md)及[新摘要](../../../../workloads/moe/runs/hopper-stage-7/forward-address32-fixed/qwen64-t128/stage7-summary.json)已更新。

本次墙钟3:55:54，峰值RSS13183164KiB（12.57GiB）；比较0.22秒。实际使用release库SHA87c507e0…、同一配置，已核对运行副本配置SHA与build.json一致。无崩溃或死锁，整体driver的failed状态来自原数值比较退出1。原driver比较失败立即break，导致摘要仍停在前16项；本轮手动重生成完整摘要，并最小修改forward.py使已完成模拟即使数值失败也生成摘要，同时保留原失败状态、命令与退出码。用五组子进程替身检查全成功、模拟失败、比较失败、摘要失败、比较及摘要均失败的顺序与退出状态；均符合预期。

尚无硬件/模拟在两个坐标对应的FC1、SwiGLU、FC2、combine输入输出的配对快照，不能仅凭最终小差异宣布普通舍入误差。用户随后明确要求暂缓这两项，转向gating耗时排查；本轮保留原失败结果和容差，不重跑完整MoE或推进其余八例。Stage7继续进行中，独立验收状态未改。


## Gating主要偏差：PTX读取没有按真实编译结果合并

用户要求先用小kernel缩短反馈，两个最终数值分歧暂缓。本轮先对照冻结PTX和CUDA12.8 `ptxas -arch=sm_90a`生成的SASS，再进行单CTA/单轮循环对照；确认因果后才改模拟器，最后只跑原尺寸独立gating，没有重新运行完整MoE。

### 原因与修复

Qwen gating的冻结PTX在638～815行连续执行178条FP32 shared读取：144条标量、12条v2、22条v4，共256个32位word。真实ptxas将这段合为64条`LDS.128`，即每线程一次读取16字节。模拟器原先逐条原PTX计入发射和shared冲突服务，所以付出更多等待；这属于PTX到硬件指令之间缺少等价合并，不是gating算法算错。

该段地址为同一base加常量偏移，线程间base间隔1024字节，标量读取使一个warp的16个不同word落在同一个bank，另16个lane读取相同word可广播。原完整运行对应shared服务1524224次、warp读取124096次，额外等待1400128；这些是跨SM累计计数，不能直接从324555个内核执行周期中相减。硬件记录只有总指令/时间，没有shared bank/wavefront计数，不宣称已经测到硬件具体等待。

内层`function_info::coalesce_shared_loads`在`ptx_assemble`分配PC和建立依赖前，对SM90的连续普通shared读取做相同合并。只接受32位F32/B32/U32/S32、相同base和类型、无谓词或volatile、地址和目的寄存器不重复且不覆盖base的连续读取；原run中必须已有合法v4来证明base的16B对齐。每个完整16B组按地址顺序保留四个目的寄存器，合并为v4；不跨标签、计算、写入或同步，不合并没有对齐依据的纯标量run。当前Qwen静态减少114条读取，178→64。

功能仍走已有vector load，时间仍走原发射、scoreboard和shared服务路径，没有修改延迟、频率、缓存或原PTX。该实现仅补当前有证据的读取合并，并非完整PTX到SASS编译器；已有普通vector共享服务按首地址计算bank的近似也仍保留，不能称完整Hopper wavefront建模。剩余时间差额未定量解释，不继续靠调参数压到阈值内。

### 对照结果与范围

| 独立测试 | 修复前周期 | 修复后周期 | 模拟μs：前→后 | 冻结ncu μs | 偏差：前→后 | 数值检查 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen单CTA/单轮K=128 | 21481 | 14185 | 10.849→7.164 | 无同形状硬件 | 不用于硬件误差 | 1024输出精确，7168哨兵不变 |
| Qwen原grid8/K=2048 | 321592 | 204856 | 162.420→103.463 | 85.120 | +90.81%→+21.55% | 全8192项精确 |
| Mixtral原grid4/K=4096回归 | 325403 | 325403 | 164.345→164.345 | 100.928 | +62.83%→+62.83% | 全1024项精确 |

单CTA执行减少34.0%，本机模拟约4.4秒，适合后续快速定位。原尺寸Qwen执行减少36.3%，本机模拟147.33秒、进程2:30.41、峰值RSS1598288KiB。Qwen累计shared冲突等待1400128→524608。手工合并诊断副本和正式模拟器加载合并得到相同周期；两种尺寸的修复前、诊断副本、正式修复后输出文件SHA分别完全相同，且均与独立CPU整数参考一致。Mixtral不含当前可合并run，周期保持不变，其原有偏差未解决。

以上时段从首CTA驻留至内核完成，排除已有启动前等待。原完整MoE内gating是324555周期/163.917μs/+92.57%，独立测试基线为321592周期，缓存前序和输入不同；不能把独立修复结果写成已重测完整MoE，也不能用其替代原输入的阶段性能验收。硬件列沿用冻结原前向ncu，当前CPU整数fixture只作功能和受控性能诊断。

### 自测与复验入口

- debug `shared_vector/bits_loads/warp_full/warp_partial/legacy`全部通过；release `shared_vector/bits_loads/numeric`全部通过。新增shared_vector检查乱序目的寄存器、v2混合、重复目的寄存器、无对齐证据、谓词mask与覆盖base的边界，4096word含域外哨兵精确，预期只减少5条读取。
- release阶段6 `integration/ldmatrix/stmatrix/legacy8`及状态组件全部通过。相关debug/release构建、clang-format16检查、两个仓库diff空白检查及新增Python语法检查均通过。未执行SASS trace验收。
- 外层`stage-2/real.py`与`real.cc`新增可选`--qwen-blocks 1`、`--qwen-iterations 1`。只对诊断副本缩短K循环，保留原矩阵stride/访存布局；默认入口仍加载冻结原PTX、原grid和完整K。run.json分列冻结源与实际嵌入PTX散列。

```bash
source /home/wangjz/miniconda3/etc/profile.d/conda.sh
conda activate in-switch
export CUDA_INSTALL_PATH=/usr/local/cuda
source gpu-simulator/gpgpu-sim/setup_environment release
python -B workloads/moe/tests/hopper/stage-2/real.py \
  --run-dir workloads/moe/runs/hopper-stage-7/gating-recheck-small \
  --test qwen --full-config --qwen-blocks 1 --qwen-iterations 1
# 小测试通过后，才检查原尺寸独立gating；不是整个MoE。
python -B workloads/moe/tests/hopper/stage-2/real.py \
  --run-dir workloads/moe/runs/hopper-stage-7/gating-recheck-full \
  --test qwen --full-config
```

完整证据：[分析MD表](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/assessment.md)、[机器可读统计](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/diagnosis.json)、[小gating](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/fixed-micro/run.json)、[原尺寸gating](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/fixed-full/run.json)、[SASS反汇编](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/qwen.sass)、[debug回归](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/shared-debug/run.json)、[release回归](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/shared-release/run.json)、[矩阵回归](../../../../workloads/moe/runs/hopper-stage-7/gating-diagnosis/regression-stage6/run.json)。

修复后release库SHA256为`f8c4ffa0bbe194393d67389766251d48801a33fffc47a119e796196bb2a759f6`，debug库为`91c024b187cbeb9d4f0d4591ff12c8e6469ed1fc8deefd3cfb7e3e8682eee4ac`，两者对应内层diff `449386bf1daa28e7f0dac46fc950236f7297aa44abe3b10d111226735a5226ce`。原尺寸配置SHA `c12ba5e9d60ebe7170cc5619a9fe6363a6306615e620857fe48e8cbddc785d95`与基线相同。旧诊断packed-full启动时工作区已开始修改，但仍加载基线库87c507e0…，其真实构建来源为3beede30…，已在baseline-library/build.json和diagnosis.json明确，不用它的当时源码散列冒充构建来源。

当前交付是Qwen gating主要失真的定向修复。剩余Qwen约+21.55%、Mixtral原有差额、两个暂缓数值点、修复后的完整MoE和其余八例均未完成独立验收；不修改原阶段状态。


## 首个可用候选版交验：限定Qwen64/128

用户希望先验收一个能用的版本，本次交验范围限定为H100配置下、直接执行PTX的Qwen64/128单层MoE，并包含当前gating读取合并修复。不继续扩测九例、不再为暂缓的两项数值差异或小幅计时差额阻塞交验准备；原九例Stage7完成要求和独立验收记录均保持原样。

当前状态为**可交验，尚未独立验收通过**。已有完整前向证据来自地址修复后的上一构建，确实执行了路由、FC1、SwiGLU、FC2和combine的全部22次调用。最新构建的共享读取合并已通过小kernel、原尺寸独立gating和相关回归，但尚未在完整前向中重跑；这项差异必须由验收报告保留。

| 交验项 | 已有证据／剩余动作 |
| --- | --- |
| 可运行 | 原Qwen64/128完成全部22次调用，无崩溃／死锁；最新gating及受影响回归均完成 |
| 数值 | 原完整输出262144项中2项超原容差，用户已暂缓；比较器继续原样报告，不能写成全部通过 |
| 耗时 | FC1/FC2旧完整样本偏差+2.66%/+5.51%；最新独立gating+21.55%，完整新构建尚待复验 |
| 覆盖范围 | 只交验Qwen64/128；其它八例、完整Stage7性能支持与硬件等价端到端时间未验 |
| 版本固定 | 保存当前外层／内层HEAD与binary diff、release运行库、H100配置及散列；未执行Git提交 |

交验文件在[候选版目录](../../../../workloads/moe/runs/hopper-stage-7/usable-candidate-20261109/)，版本与已知限制见[manifest.json](../../../../workloads/moe/runs/hopper-stage-7/usable-candidate-20261109/manifest.json)。该快照保存实际库文件，而非指向后续构建的符号链接。原完整前向证据和最新独立测试分别列出，不能混称同一构建的完整通过。

验证方先核对候选版源码／库／配置，复用可靠的同版本证据，独立运行新增读取合并的小回归和单CTA gating；这两项可快速反馈。若要接受“最新构建完整Qwen前向可用”，还需只重跑一次Qwen64/128原前向（既有样本约4小时），全量比较并核对22项顺序。已知两项差异单列，新故障据实判定，不放宽容差、不改golden、不扩大到其余八例。独立验收结论仍由验证方维护，本实现记录不代填通过。

```bash
source /home/wangjz/miniconda3/etc/profile.d/conda.sh
conda activate in-switch
export CUDA_INSTALL_PATH=/usr/local/cuda
source gpu-simulator/gpgpu-sim/setup_environment release
python -B workloads/moe/tests/hopper/stage-1/run.py \
  --run-dir workloads/moe/runs/hopper-stage-7/acceptance-shared \
  --test shared_vector,bits_loads
python -B workloads/moe/tests/hopper/stage-2/real.py \
  --run-dir workloads/moe/runs/hopper-stage-7/acceptance-gating-small \
  --test qwen --full-config --qwen-blocks 1 --qwen-iterations 1
# 独立验收完整前向时使用；本次交验准备未启动此长任务。
python -B workloads/moe/tests/hopper/stage-7/forward.py \
  --run-dir workloads/moe/runs/hopper-stage-7/acceptance-qwen128 \
  --cases qwen64-t128
```

将[通用验证prompt](../../validation-prompt.md)连同本节范围和manifest交给验证方即可开始。这里只准备交验材料，没有派生验证agent或向其它聊天发送消息。


## 2026-11-10验收失败的快速数值排查（待用户查看方案，未修复）

独立验收[最新记录](../../validation/stage-7/README.md)为部分通过：Qwen既有2项暂缓，Phi新增84项、Mixtral新增123项超原容差。用户要求先查原因并展示方案，看过后才修复，性能影响在修复后再看。本轮遵守这一顺序，只新增运行目录内的CPU诊断及一个warp的PTX复现，未修改模拟器、配置、冻结输入/PTX/golden/容差，未重跑完整MoE，也未代填独立验收状态。

已有数据将两例最坏token的触发机制缩到SwiGLU激活的BF16临界舍入，但尚未取得H100的指令级中间值，不能宣布所有新增失败的唯一根因已证实。关键结果：

| 诊断 | 实际结果与边界 |
| --- | --- |
| 冻结配方独立CPU重建 | Phi原84个失败坐标84/84、Mixtral123/123逐位复现模拟输出，分别14.38/17.81秒；不调用模拟器地址、矩阵或时序代码 |
| 代表token全通道 | Phi token123与Mixtral token97的4096通道全部逐位复现模拟输出 |
| 激活边界扰动 | Phi t123/e5/k2294：gate0.78515625、up0.87109375产生0x3ef08000，恰为BF16中点，转0.46875；仅试验相邻0.470703125，47项超差→0。Mixtral t97/e2/k7899：gate−0.00390625、up0.25也恰在中点；试验相邻BF16后该token1项超差→0。这是CPU因果诊断，不是硬件中间实测或生产修复 |
| 独立排除实验 | 两代表token把combine改fmaf仍有原超差；FC2改精确点积也不能消除，当前没有依据先改WGMMA布局或同步 |
| 一个warp PTX | 提取实际指数、div、mul及BF16链，32lane/256word与CPU阶段位模式一致，主机模拟约0.122秒；只测当前模拟库，未测H100 |

源码`ex2_impl`使用CPU powf，`div_impl`的F32统一CPU除法；原Phi SwiGLU的ptxas SASS使用MUFU.EX2/MUFU.RCP。官方PTX允许ex2.approx和div.full各2 ulp，CPU与硬件近似不必逐位相同，极小FP32变化可能让BF16落在另一侧，经过FC2和加权抵消后使小参考值超容差。当前BF16平局取偶数转换本身正确，不能通过换舍入规则修复。不能凭普通CPU除法与硬件不同就断言违反PTX规范，也不能推断84/123项已经全部精确归因。

建议下一步先在可用H100上运行同一32线程微测，抓取ex2/div和激活位模式；确定哪一步首次不同后，修相应Hopper近似数学路径，保留IEEE舍入指令语义。拒绝按token/某个输入硬编码相邻BF16，或统一加减一个末位（上述两例所需方向相反）；简单把powf换成exp2f在这两例无效。本机驱动不可连接，暂缺此硬件证据。本轮已准备可在硬件运行的PTX/host，不先改实现取得表面通过。

stream统计归属问题已有独立代码证据：纹理映射构造器遗漏m_streamID，launch又未传入实际stream。建议单独初始化并从真实launch传递默认/非默认stream身份，不一律写0。它尚无数值或周期影响证据，不用来解释207项失败。待用户看过方案后再修；修后先短测试和数值复验，再检查gating性能。

[本轮诊断报告与修复建议](../../../../workloads/moe/runs/hopper-stage-7/numerical-diagnosis-20261110/assessment.md)链接全部CPU参考、207坐标结果、阶段位模式、边界扰动及来源。实际库仍为f8c4ffa0…，内层diff仍为449386bf…，与验收版本相同。第一份诊断PTX因提取工具丢首行而缺.version失败，添加注释首行后最终activation-small-final通过；失败目录保留，不计为有效小复现。


### 本机A800补测：近似除法已解释142个原失败坐标

用户指出本机A800后，沙箱外成功访问NVIDIA A800 80GB PCIe（cc8.0）。先前沙箱内看不到GPU不能据此称主机驱动不可用。真实运行库为/usr/local/cuda/lib64/libcudart.so.12；只把诊断副本target改为sm80，没有修改冻结PTX或模拟器。

一个warp对照已实测Phi输入gate0.78515625/up0.87109375：ex2结果虽差1个FP32位，但加1后的分母与模拟器完全相同；GPU div.full近似倒数比CPU除法高1个位，激活0x3ef08000→0x3ef08002，BF16从0.46875→0.470703125。此例有直接指令阶段证据，BF16转换规则本身不需要改。Mixtral此前最坏样例的那对输入在A800与模拟器相同，H100差异仍未直接实测。

对全部原失败token的230400/1318912个激活，A800原冻结SwiGLU（仅target改sm80）与测量小核的BF16结果逐项一致；与CPU/模拟器分别只有3/7个激活不同。将这些实测激活接回CPU FC2/combine，原84/123个失败坐标变成17/48个仍超差，即142个原失败坐标不再超差。它不是完整前向、没有检查其它坐标的新失败，也不能保证A800近似就是H100。剩余65项仍需继续定位。

有效批量运行每进程约0.36～0.91秒，未跑完整MoE。原核初次诊断误用FP32输入存储，已改为其要求的BF16并重跑；无效输出保留original-wrong-input，不作为证据。当前只测试和报告，继续等待用户查看方案后才修模拟器；性能影响未测。

[补测报告](../../../../workloads/moe/runs/hopper-stage-7/numerical-diagnosis-20261110/assessment.md#后续本机a800已实测确认部分差异来自gpu近似除法)、[逐步位模式](../../../../workloads/moe/runs/hopper-stage-7/numerical-diagnosis-20261110/native-a800/comparison.json)、[批量汇总](../../../../workloads/moe/runs/hopper-stage-7/numerical-diagnosis-20261110/native-a800-batch/summary.json)和[有效运行/散列](../../../../workloads/moe/runs/hopper-stage-7/numerical-diagnosis-20261110/native-a800-batch/run.json)记录细节。下一步修复方向是区分full/approx和IEEE除法、依据实测建立硬件近似倒数；不按某个gate值硬编码末位补偿，不将A800结果冒称H100验收。


### Modal H100一次批量取证：新增确认FC2累加差异（未修复）

用户提供`workloads/moe/docs/modal-h100.md`并要求节约额度。先在本机A800完成格式、原始/测量核和路由预检，再只调用一次限时H100，单容器、零重试。测试正常退出，原始数据已下载，App已完成；后续只做本地分析，不再次申请H100。没有完整MoE运行或模拟器改动。

硬件H100 80GB HBM3，测试程序1.135936秒，含输入验证1.297842秒；这不是平台计费时长，另有容器启动、打包与传输，客户端总等待89.62秒，实际费用以Modal用量为准。本批准备964544组激活probe和81664组独立数学probe，采集Qwen2、Phi18、Mixtral46个原失败token对应136个专家行的完整FC1、激活、FC2及路由权重，同时比较CUDA12.8预编译与驱动即时编译，保留驱动缓存。

| 原失败坐标 | 原模拟 | 只替换H100激活 | 只替换H100 FC2 | 两项同时替换 |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 2 | 2 | 0 | 0 |
| Phi | 84 | 17 | 67 | 0 |
| Mixtral | 123 | 48 | 75 | 0 |

替换仅是诊断数据替换，末尾仍在CPU合并，未修模拟器。FC1全量与独立CPU一致；H100与A800的实际激活及8步位模式逐项相同，两种编译路径也逐项相同。Phi23.52倍点仍由近似除法解释。Mixtral23.11倍点和最大失败绝对误差点，即使输入相同CPU激活，H100 FC2也分别输出-0.11328125/-0.271484375，而CPU顺序fmaf为-0.11376953125/-0.2734375；换H100 FC2后两点逐位恢复golden。Qwen两项也由FC2结果差异消除。

因此不能再将剩余Qwen2/Phi17/Mixtral48项归到激活函数。`ptx_matrix_op::compute()`当前按k顺序对每条WGMMA的16项逐项fmaf，其舍入结果与H100存在差异；硬件内部FP32累加器尚未采集，不宣称已确定内部次序或精度，也不能直接指定double或某个重排为修复。修复需同时处理近似数学和WGMMA累加一致性，继续等待用户查看后才改生产实现。

检查全部失败token的266240个输出，用H100激活/FC2与CPU路由权重组合，均0项超原容差；并非完整前向验收。模拟器diff仍为449386bf…，release库未变，未测新的性能。完整[报告与证据](../../../../workloads/moe/runs/hopper-stage-7/h100-numerics-20261110/report.md)、[远端原始运行](../../../../workloads/moe/runs/hopper-stage-7/h100-numerics-20261110/h100/run.json)、[原始设备日志](../../../../workloads/moe/runs/hopper-stage-7/h100-numerics-20261110/h100/batch.log)、[逐坐标分析](../../../../workloads/moe/runs/hopper-stage-7/h100-numerics-20261110/analysis.json)已保存。


### 用户优先级调整：暂缓已定位的浮点一致性修复，回到性能时序

用户明确：数值精确一致不是当前重点，允许计算精度/舍入导致的小范围差异，优先关注性能与时序。因此暂缓上节近似数学和WGMMA累加算法校准，不再以消除209个既有超容差坐标阻塞性能排查。保留原比较器、参考值、容差及失败记录，不将原严格比较改写为通过。若出现非有限输出、明显整体偏离，或数值变化导致路由/地址/指令工作量改变，仍需作为功能问题处理。

已有三例整体相对L2为0.00556%/0.01724%/0.02175%，全量输出有限；FC1/FC2工作量与硬件Tensor计数相符。本轮已定位差异位于路由之后，尚未发现改变执行路径或工作量的证据。刚才的修复方案主要改变功能值，未计划修改发射间隔、流水线延迟、shared服务或异步完成条件；它不能直接修正当前gating设备周期偏差，还可能增加主机计算成本。`shader.cc:prepare_ptx_matrix()`单独通过shared服务/read_ready门槛决定何时调用数值compute，二者不可混作同一个修复。

性能后续优先Phi/Mixtral gating：+60.58%/+68.73%，专家FC1/FC2当前均在6%以内。先核对原PTX与ptxas实际shared加载/对齐/服务量，再用小kernel区分shared服务、普通cp.async等待及缓存前序；不按耗时比例拟合参数。统计stream身份初始化错误另行处理，保持它与真实周期修复的影响边界。按已有小测试与保存的H100/ncu数据推进，暂不新增付费H100调用，不启动完整MoE长模拟。原独立验收结论保留历史事实，新的暂缓来自本轮用户指示。


### 2026-11-10：按用户最新要求恢复修复，公开 H100 模型定向验证通过

用户考虑后续层路由后明确要求恢复数值修复，并要求先查 NVIDIA 文档和公开逆向结果；本节取代上节“暂缓”的当前执行取舍。没有再次申请付费 H100，没有重跑完整 MoE，没有修改冻结 PTX、输入、golden 或原容差，也不覆盖独立验收记录。

已查阅 NVIDIA PTX 8.7 的 div/ex2/WGMMA 说明：近似数学不保证 IEEE 最近舍入；BF16 WGMMA 的内部累加次序和舍入未具体规定。因此修复目标是重现 H100，不能仅以 CPU fmaf 不同便判定违反 PTX。找到 Khattak/Mikaitis 的[《Accurate Models of NVIDIA Tensor Cores》H100 小节](https://arxiv.org/html/2512.07004v2#S4.SS1.SSS5)及[作者源码](https://github.com/north-numerical-computing/MATLAB-tensor-core/tree/79143bd29a4a71dc7c2ed47741b63572498b17f6/models)，不再试探累加顺序：BF16 按 16 项块累加，按输入指数之和对齐、保留额外 2 位，对齐和最终 FP32 输出均截断，块内不逐项归一化。其 5000 组独立公开 H100 BF16/FP32 向量由本实现逐位匹配。

特殊函数采用 [envytools 公开逆向源码](https://github.com/envytools/envytools/tree/f102b82381f3f11cee113d16374c87091db039d9/nvhw)的 EX2/RCP 多项式与系数表。该源码来自较旧硬件，已用本项目保存的 H100 探针验证再移植，未仅凭作者声明认定适用于 Hopper。保留其 MIT 声明和数值模型的 BSD 声明；未引入不透明二进制依赖。

生产改动仅涉及 `src/cuda-sim/ptx_ir.*`、`instructions.cc` 和新增的 `nvidia_math.h/hopper_math.h`。保留此前被忽略的 `.approx/.full/.ftz`；EX2/RCP 和近似除法走有来源的模型，`div.rn` 保留原路径；WGMMA 调用 16 项 BF16 块乘加。BF16 转换规则、矩阵地址、依赖、shared 服务和写回时机不变。代码没有按 token、gate 或专家补偿末位。

[离线数值回放](../../../../workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/offline-comparison.json)直接调用生产算术头文件，覆盖本批已保存专家行的全部输出，再用原路由权重、模拟器分离 FP32 乘/加方式合并：

| 用例 | 与 H100 不同的激活 / 已检查 | 与 H100 不同的 FC2 / 已检查 | 超原容差：旧 → 新 | 合并输出检查范围 |
| --- | ---: | ---: | ---: | ---: |
| Qwen64/128 | 0 / 11264 | 0 / 16384 | 2 → 0 | 4096 |
| Phi/128 | 0 / 230400 | 0 / 147456 | 84 → 0 | 73728 |
| Mixtral/128 | 0 / 1318912 | 0 / 376832 | 123 → 0 | 188416 |

这里不是直接替换硬件 FC2 数据：CPU 使用生产块累加算法重算了全部 540672 项 FC2，均与 H100 逐位一致。最终合并检查覆盖原失败 token 的全部 266240 项，不只检查原失败坐标；不是完整前向。原最严重约 23.5/23.1 倍位置及最大失败绝对误差位置均逐位恢复参考。其它位置仍可能有原容差内差异，不宣称全输出逐位相同。

81664 组已保存 H100 数学探针的五条路径、964544 组激活的八个阶段逐位全部一致；5000 组作者 H100 数据逐位一致。本机 A800 另测 65761 组输入、每组 9 条路径，包含随机位模式、非单位分子、次正规数和 FTZ，忽略 NaN 载荷后全部一致；没有把该 A800 结果冒称 H100 全域验证。

[实际 PTX 原库/新库对照](../../../../workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/ptx-small/comparison.json)四项旧版均数值失败，新版全部通过，周期/动态线程指令均完全相同：

| 定向小内核 | 原 `gpu_sim_cycle`：旧 → 新 | 动态线程指令：旧 → 新 |
| --- | ---: | ---: |
| SwiGLU 中间阶段 | 6017 → 6017 | 928 → 928 |
| EX2/RCP/三种除法 | 5764 → 5764 | 704 → 704 |
| 单条 WGMMA BF16 边界矩阵 | 40043 → 40043 | 93632 → 93632 |
| 两条依赖 WGMMA | 40184 → 40184 | 93888 → 93888 |

周期包含原模型启动等待，不与首 CTA 后范围混用。单次主机进程墙钟约 3.14→3.11、3.03→3.01、13.34→13.44、13.55→13.33 秒，包含初始化且非重复统计，不能由此保证完整 MoE 墙钟不变。最终构建在补注释/格式化后又通过同一四项，周期/指令数不变。[8 项矩阵相关回归](../../../../workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/matrix-regression/run.json)全部通过；[Qwen 路由回归](../../../../workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/routing/run.json)用 CPU 重建冻结分数跳过大 gating 核，原 top-k/排序/softmax/权重/掩码/专家计数内核通过，512 项选中专家检查无误。未称其覆盖 Phi/Mixtral 整个路由执行。

[最终版本清单](../../../../workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/manifest.json)：原库 `f8c4ffa0…`、原差异 `449386bf…`；最终库 `4a6b5530…`、包含新增头文件的内层差异 `2566a70f…`。两个仓库均未提交，保留此前其它修改。[最终构建/实执行](../../../../workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/final-smoke/run.json)、源码快照/散列及新旧差异均在同目录，两个仓库 `git diff --check` 通过。复验先运行[短测试入口](../../../../workloads/moe/tests/hopper/stage-7/h100-numerics/README.md#不再申请-h100-的修复回归)，不要先启动完整 MoE。

本轮数值定向修复和小内核时序对照完成；不替独立验收宣布整个阶段7通过。原 gating +23.31%/+60.58%/+68.73% 偏差及 stream 统计归属问题仍待处理，本次不以数值修复掩盖它们。完整[依据、范围、复现及性能记录](../../../../workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/report.md)给出原文链接和全部证据。

## 2026-11-10：Phi/Mixtral gating、stream 修复及 L2 复核（交独立验收）

本轮按最新交接只处理这两项。保留已验收的数值实现、冻结 PTX/输入/reference/容差和 H100 资源配置；未启动完整 MoE、六个大用例或付费 H100。[完整证据与直接复现命令](../../../../workloads/moe/runs/hopper-stage-7/performance-statistics-20261110/README.md)、[逐项机器可读结果](../../../../workloads/moe/runs/hopper-stage-7/performance-statistics-20261110/review.json)、[最终清单与源码快照](../../../../workloads/moe/runs/hopper-stage-7/performance-statistics-20261110/manifest.json)已保存。本节不替独立验收宣布整个 Stage7 通过。

离线 ptxas 确认两例循环各为 48 条 LDS.128，冻结 PTX 则各有 192 条标量 shared 读取；另外各有 2 条循环外 v4，不属于同一读取段，未触发旧对齐规则。已沿 `.align 16` 数组、环形缓冲偏移和线程 AND/SHL 地址计算证明对齐。生产合并逻辑只增加从所有寄存器定义推导低位零的保守证明，保留原有依赖和重排边界；每例删除 84 条标量读取，变为 28 条 v4 + 80 条标量，没有按模型或指令比例修改延迟。普通 cp.async 等待与缓存配置未改。

以下全部为同输入、同配置的**独立整数 gating**首 CTA 到最后完成范围，1980 MHz：

| 实验 | 周期：前 → 后 | 时间 μs：前 → 后 | shared 额外服务计数：前 → 后 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| Phi 单 CTA/单 K tile | 4255 → 3336 | 2.149 → 1.685 | 1684 → 964 | 前后均 0 |
| Mixtral 单 CTA/单 K tile | 4164 → 3244 | 2.103 → 1.638 | 1684 → 964 | 前后均 0 |
| Phi 原 grid/完整 K | 319976 → 221014 | 161.604 → 111.623 | 852048 → 483408 | 前后均 0 |
| Mixtral 原 grid/完整 K | 325403 → 209268 | 164.345 → 105.691 | 852048 → 483408 | 前后均 0 |

输出文件前后逐位相同。相对已有 H100 原前向 106.144/100.928 μs，修复后的独立核为 +5.16%/+4.72%，仅供量级参考。**输入、地址、缓存前序不同，原前向修复后的偏差尚未重测。**Qwen 原前向 +23.31% 保留记录，仅复查其单 CTA/单 K tile。未把独立核时间冒称新的原前向或端到端结果。

纹理快照构造器改为委托公共构造器初始化 stream，host launch 传实际 stream UID，设备 child launch 选定 child stream 后也设置身份。loading 六次交错请求的 host/kernel 身份现在一致为 `0,1,2,1,0,2`，各流累计 L2 写 sector 为 `4,4,4,8,8,8`，单核差分全为 4；旧库后面三次落入 UINT64_MAX 哨兵，已复现。每次输出通过，首 CTA 后周期前后均 1300。integration 两次从 `0/UINT64_MAX` 修正为 `0/0`，周期仍 9644/9639。小专家 FC1 的 131072 项输出前后逐位一致、guard 通过，周期均为 18754、WGMMA 均为 2016 次，无新增 shared 合并。没有证据把 stream 统计错误当作 gating 周期原因。

L2 已重新按累计 key 差分，六项结果与原报告一致；专家 key 非 UINT64_MAX，因此不能靠 stream 修复消除原 21%～24% 差额。原始日志中的 LRC 合并量仅为 6～123 MB，总读差额则为 171～799 MB，不能只归因于 LRC。结合已采的硬件读/写 sector 和约 25% 总命中率作区间分解，六项硬件 read miss 与模拟器接近，额外 read hit 与总差额相当；DRAM 读差仍仅 −0.0005%～−0.117%。详细表、估算公式与舍入边界见证据记录；不是新采的 read-miss 子指标。

[NVIDIA Nsight 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables)与[NVIDIA 员工的 fabric 解释](https://forums.developer.nvidia.com/t/whats-the-meaning-of-performance-counter-lts-t-sectors-srcunit-ltcfabric/228015/2)确认总量包含跨 L2 分区请求。当前模型计 LRC 后本地 lookup 与远端 lookup，返回则经 fabric 延迟、回填队列和 fill port 服务，不另产生一次 read lookup；地址分区/调度也未按硬件校准。这是当前近似边界，**尚不能证明硬件额外命中具体有多少属于返回查找**。原 H100 报告没有 fabric 来源细分。本机 A800 4096-sector 小探针数值通过，ncu 被驱动 `ERR_NVGPUCTRPERM` 拒绝，没有采到计数；未改驱动权限或追加付费 H100。

本轮据此不增加 L2 耗时补偿、不修改专家资源：跨分区服务没有被跳过，不能把命中计数差额直接换算为周期误差。原六项专家时间误差约 −2.70%～+5.14%，小专家前后周期相同；不外推大用例。

release 构建、三例 gating micro、四项 shared 对齐/错位/分支/原向量回归、cp.async groups6/copy_mix、stream 六启动、integration 双启动、小专家 FC1 均通过。数值相关五个生产源文件散列与前轮完全相同。最终库 `d6b3a2b1…`；完整独立 gating 在最终格式化前库 `619e4702…` 上完成，最终库重新通过相同 micro/stream/shared 检查。修复前库 `4a6b5530…` 与全部配置、输出、差异、命令均保存。本轮在此停止，交独立验收，不自动启动原前向或大用例。

## 2026-11-10：Qwen gating 的临时寄存器复制开销

已完成实现与定向自测，待独立验收。[完整记录、同范围结果与直接命令](../../../../workloads/moe/runs/hopper-stage-7/qwen-gating-20261110/README.md) · [只读核查](../../../../workloads/moe/runs/hopper-stage-7/qwen-gating-20261110/review.json) · [本轮独立代码差异](../../../../workloads/moe/runs/hopper-stage-7/qwen-gating-20261110/change.diff)。

Qwen 冻结 PTX 循环体有 512 处 `cvt.f32.bf16` 紧邻 `mov.b32`，临时结果只被复制一次。诊断副本删除复制后，ptxas 生成的全部 2,280 行 SASS 指令及编码相同，而模拟器此前仍执行这些复制。新增 SM90 加载阶段的局部复制消除：仅对普通 32 位标量结果、无条件相邻位复制、全函数单定义单读取的临时值生效；按完整寄存器引用计数保留其他用途、重复定义、向量/地址引用、谓词和标签边界。代码没有模型名称、模块编号、形状判断或时间系数。实现位于 cuda-sim.cc；ptx_ir.h 仅加一行声明，数值实现、冻结数据、原容差与配置保持不变。

| 同范围测试 | 执行周期，前 → 后 | 模拟 μs，前 → 后 | 数值 |
| --- | ---: | ---: | --- |
| Qwen 单 CTA、单 K tile | 14,185 → 11,114 | 7.164 → 5.613 | 输出及域外哨兵逐位一致 |
| Qwen 完整独立 gating | 204,856 → 155,771 | 103.463 → 78.672 | 全部 8,192 项逐位一致，最大绝对误差 0 |
| Phi 单 CTA、单 K tile | 3,336 → 3,336 | 1.685 → 1.685 | 输出及哨兵逐位一致 |
| Mixtral 单 CTA、单 K tile | 3,244 → 3,244 | 1.638 → 1.638 | 输出及哨兵逐位一致 |

完整 Qwen 的前后均在本轮、同入口、同冻结 PTX、同输入生成器和配置运行：前使用归档库 d6b3a2b18f76480d5a76a487f00e733f9571b9918a243542dcf33a3b18cd3a85，后使用 f223cd2c5ea7e9509a998be34afa5bb433508f23c50a58ece911e1488c9b7f25。耗时降低 23.96%；线程指令恰减少 `512×16×8×128=8,388,608`，共享内存额外服务计数仍为 524,608。没有按该指令数直接扣周期，而是让原调度和流水线模型执行转换后的指令。

Phi/Mixtral 没有触发新的消除，删除数均为 0。原 PTX 均没有 BF16→FP32 转换；各有 12 条 mov.b32，其中 4 条写常量，8 条集中在输出阶段，与生产该值的计算不相邻。因此新规则不改它们的 gating 指令序列，短测的周期、线程指令数和 shared 服务计数均不变，原有 shared 合并仍各为 84 条。未重跑两者完整 MoE，不能扩大为整个模型所有内核均已复测。

独立 Qwen 相对原前向 H100 的 85.120 μs 为 −7.57%，此前为 +21.55%；只作量级参考，原前向的输入、地址与缓存前序不同，本轮未复测原前向。没有启动完整 MoE、六个大例或付费 H100。三项新复制边界测试、numeric/collectives/SM60 legacy、三项 shared 回归、阶段6双 launch integration、六次真实 stream 身份回归均通过；不符合消除条件的测试周期不变。

用户指出该方法仍属局部指令组合优化，这一限制明确保留；它不是完整复制传播、寄存器分配或 ptxas 调度模型。用户随后确认避免复杂和侵入性强的改造，以当前几个模型的单卡准确性为目标，之后推进多卡。因此保留本轮小范围通用实现，停止扩展编译优化框架与执行路径；单卡阶段仍交独立验收，不自行宣布阶段7整体通过。

## 本次提交归档与覆盖范围

模拟器提交`9cff9650653fbf5420f5c3c7911eb3d78c46c16a`归档上述实现，源码与最终受测diff一致。
独立验收已确认Qwen64/128、Phi/128、Mixtral/128各22次前向和全部数值通过，
主要内核时间在约6%以内；[结果附件](../../../../workloads/moe/evidence/hopper-stage7-128/README.md)
保存完整kernel/端到端表与版本。完整日志和大型产物保留本地runs。

**六个大用例没有运行模拟：Qwen64/4096、Qwen64/8192、Phi/4096、Phi/8192、Mixtral/4096、Mixtral/8192。**
H100九例硬件基线不等于九例模拟已验证。本提交不宣称原九例阶段7全部完成，
端到端覆盖和L2近似限制保留；后续重点为多卡，未在本次开展多卡实现或验收。
