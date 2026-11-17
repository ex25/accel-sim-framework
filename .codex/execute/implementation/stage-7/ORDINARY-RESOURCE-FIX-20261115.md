# 2026-11-15 普通核资源修复与写流量诊断

> **2026-11-17收尾：**资源与L2修复已联合完成Qwen128/4096/8192完整回归并限定验收通过，全部输出零超差。8192 FC2 +21.37%保留；旧8192 event缺失末两核，当前时间戳一行修复有短测证据，未重跑完整前向。[最终结果附件](../../../../workloads/moe/evidence/hopper-stage7-qwen/README.md)。下文运行中/隔离候选描述保留其发现时点。

用户要求先结束本轮完整前向，修复其它偏差较大的kernel后统一回归。本轮没有重新启动完整前向，原8192任务与历史结果未改。本文接续[FC读取流水修复](FC-READ-PIPELINE-20261115.md)。

## 已修复并整合到主工作区

此前将整数乘加、逻辑和warp shuffle全部放在INT资源上，造成错误的资源竞争与重叠。官方Nsight Compute流水说明和本轮H100计数均确认：IMAD/IMUL属于FMA，逻辑属于ALU，shuffle和shared load属于LSU。现在Hopper整数乘加使用SP/FMA资源，shuffle使用LSU发射和写回；原指令延迟、结果依赖和寄存器写回仍保留。没有调整GMMA、TMA或HBM参数。SP合并FMAHeavy/Lite仍是近似，不能称为完整物理管线复刻。

| 短测试 | 修复前误差 | 修复后误差 |
|---|---:|---:|
| shuffle | −44.3% | −0.2% |
| shuffle + shared load | −48.3% | +1.8% |
| 逻辑 + shuffle | −18.8% | −4.0% |
| 整数乘加 + FMA | −28.7% | +12.3% |

硬件后3轮线程最大周期均值对比模拟内部循环周期；这不是完整MoE误差。11项数值检查、12项Hopper设备回归、状态组件与读取流水检查通过。原行映射PTX合成输入短测：128CTA 3538→3565，1024CTA 7563→6384周期，结果与哨兵均正确，尚未重跑原完整前向的行映射调用。

增量涉及内层四个文件：`abstract_hardware_model.h`、`cuda-sim.cc`、`shader.cc/.h`；主工作区原7个未提交文件的既有内容保留，目前共8个修改文件。源码全量散列核对见[整合清单](/home/wangjz/in-switch/accel-sim/workloads/moe/runs/hopper-stage-7/ordinary-fix-20261115/canonical-integration.json)。主工作区没有原地构建。

官方依据：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)，本轮实际读取本机2025.1.0官方文档并做H100定向NCU采样。硬件原始记录、11项完整表、失败记录与版本散列见[详细记录](/home/wangjz/in-switch/accel-sim/workloads/moe/runs/hopper-stage-7/ordinary-fix-20261115/README.md)。

## 新发现：普通写流量遗漏，仍在隔离候选验证

旧128MiB拷贝模拟68.206µs，硬件90.080µs，但模拟DRAM写85,249,664B，硬件135,840,768B，不能把全部时间差归为HBM带宽。硬件取消前序整块memset后，拷贝87.840µs、DRAM写123,581,440B；读量与模拟基本一致，写量仍有明显差额。只改初始化无法解释全部差距。

源码中跨L2写更新使用cudaMemcpy强制填标签路径，绕过正常端口/背压，并丢失脏块替换的写回事务；已有dirty hit也可能被强制填成clean。候选将其改为正常标签访问、脏块替换写回及端口收费；新分配的副本保持clean，原本dirty的本地数据保持dirty。另补在途peer更新、cache miss queue和缓存内部写回的结束判断，防止尚未完成就停止周期推进。

这属于既有模型内部事务处理修复，**没有证明H100真实拓扑就是当前双chiplet模型，也未证明能解释全部真实核残差**。候选位于独立`forward-write-candidate-*`树，尚未合主工作区。小缓存短测先区分“修结束判断”和“修脏块写回”，检查计数、数值及哨兵，再决定是否合入；不会直接启动三规模完整前向。

## 版本与比较边界

| 版本 | 冻结库SHA256前缀 | 用途 |
|---|---|---|
| 历史完整回归库 | `0b7e8ed1` | 已完成三规模完整前向、128MiB访存对照 |
| FC读取修复库 | `a82bcca0` | 普通资源修复的旧对照 |
| 普通资源最终库 | `81c7a319` | 主工作区对应版本，短回归通过 |
| L2写更新候选 | `4b1a8915` | 隔离候选，尚未验收或合入 |

模拟探针打印的legacy CUDA event毫秒是主机墙钟，不能作为GPU耗时；本轮访存表只用SIM_KERNEL活动周期和DRAM统计。完整前向端到端继续使用既有GDB CUDA-event设备周期观测，不以kernel相加代替。`clock64`旧功能返回4倍核心周期且为32位，资源微测明确除4并核实未溢出；该问题未在本轮修改，也不影响SIM_KERNEL周期。

## L2短对照的首批结果

显式设备同步后，三个写入探针的输出和哨兵全部通过，三种库的GLOBAL写请求均16384sector（32B/sector）。旧库累计DRAM写11969sector，只修结束判断为15886sector，完整候选为16332sector。由此区分：生命周期补计3917sector，正常peer写处理在相同生命周期下再恢复446sector（14272B）。三个活动kernel周期均419/519/419，没有显示计算核本身变慢，不将其称为真实前向性能改善。

每L2分片1set/1way的copy在旧库与候选均触发已有deadlock_check，现场保留，未发终止信号；该失败不属于通过验收。更高相联度copy和最终Hopper回归继续验证。永久普通资源/原行映射短测入口位于`workloads/moe/tests/hopper/stage-7/ordinary_resource_probe.py`与`row_map.cc`。L2因果实验源码、运行器、散列和完整失败日志位于上述详细记录目录。

## 最终验收与联合回归已启动

L2最终冻结库`818c97ce…`通过12项Hopper设备回归、状态组件、读取阶段校验与三段两核CUDA-event预检（每段11224周期、最终数值7）。最终两文件增量已合主工作区，全部其它内层跟踪文件散列未变；主工作区仍8个修改文件，没有原地构建。

2MiB拷贝在相同生命周期判断下，正常peer更新将DRAM写从20715恢复到44468个32Bsector，DRAM读均61682sector、原始GLOBAL写均65536sector，三版本全部输出与哨兵正确。这是脏块写回丢失的独立证据，尚不代表完整MoE性能误差已经收敛。1set/1way的旧/新死锁、4set/4way未完成后SIGTERM的失败记录均保留，极小缓存的peer进度问题尚未修复；通过的拷贝对照使用8sets/20ways，完整回归使用原默认配置。

新联合完整回归位于[combined-full-20261115](/home/wangjz/in-switch/accel-sim/workloads/moe/runs/hopper-stage-7/ordinary-fix-20261115/combined-full-20261115/full-state.json)，PID61700，128/4096/8192均不预热、各一次，逐核活动周期及直接CUDA-event设备周期分别记录。启动前宿主机无活动MoE模拟，未重启旧停止任务，原8192未动。完成后自动独立数值验收与匹配硬件/旧库比较，当前尚无完整前向新结果。
