# Hopper PTX 验证概览

更新：2026-11-07。文档导航见[README](README.md)，术语解释见[技术问答](hopper-qa.md)，执行顺序见[实施计划](hopper-plan.md)。以下是后续实现的检查与通过条件，不是本轮已完成的模拟测试。

## 每阶段怎样证明可用

每阶段都运行含目标 PTX 的最小 CUDA 程序，检查输出和周期；再用必要组合及代表性真实 entry 检查协作。测试必须只依赖本阶段和已完成阶段。现有 golden 是九例完整输出，没有逐内核中间 golden：早期使用小型已知输入和 CPU 预期，不能把硬件计数或程序退出作为数值正确性证明。

一份运行记录保存：测试名和输入、外层/内层提交号、CUDA/ptxas 版本、实际目标与加载的 libcudart、PTX 周期模式、配置文件及散列、数值差异、内核/阶段周期和主要流量。性能记录按用例、launch index、entry、grid/block/shared 区分，FC1 与 FC2 即使 entry 相同也不合并。

最小程序的整数、地址、计数、BF16 转换位模式应精确匹配。矩阵测试选可精确表示的输入做布局/依赖检查，再用一般输入检查浮点误差。完整输出沿用 `compare.py`：所有元素有限且 `abs(actual-ref) <= 1e-5 + 0.02*abs(ref)`，另报逐位一致、失败元素、最大绝对误差和相对 L2。20% 只用于时间，不放宽数值容差。

## 分阶段检查入口

详细测试、输入构造与预期放在 [validation/](validation/README.md)，实际结果表迁至 [execute 的分阶段验收记录](execute/README.md)。下列仅保留概览，当前尚无运行验收结果。

<a id="stage-0"></a>

### 阶段 0：入口和资源基础

两模块交替启动、sm_90a、64B参数、直接API/编译器入口、每次动态shared和数据顺序正确。失败分支明确报错；从首个小程序建立配置、周期、墙钟/RSS记录。

[详细验收](validation/stage-0.md) · [实现方案](plans/stage-0.md)

<a id="stage-1"></a>

### 阶段 1：基础数值与真实普通内核

BF16位模式、合法参与mask、计数/索引与集合结果正确；执行真实转换/路由/映射代表。首个真实普通内核记录初始化、模拟与比较墙钟，不外推WGMMA速度。

[详细验收](validation/stage-1.md) · [实现方案](plans/stage-1.md)

<a id="stage-2"></a>

### 阶段 2：普通 cp.async 与 gating

验证有效源/零填/谓词、普通组wait6/0和合法流水重叠；真实Mixtral/128与Qwen64/128 gating数值通过。尺寸及缓存前序可比时才计算硬件时间误差。

[详细验收](validation/stage-2.md) · [实现方案](plans/stage-2.md)

<a id="stage-3"></a>

### 阶段 3：cluster/DSMEM 基础

两个CTA身份和相同offset的存储不混；实际大shared及132SM地址正确；资源保证共同推进，退出不提前复用存储。relaxed cluster barrier不当任意shared发布。

[详细验收](validation/stage-3.md) · [实现方案](plans/stage-3.md)

<a id="stage-4"></a>

### 阶段 4：mbarrier

本地/远端到达、真假轮询、三轮phase及有符号字节计数正确；非零计数先组件验证，实际TMA返回留阶段5。当前丢弃token的副作用保留，不扩未用token消费形式。

[详细验收](validation/stage-4.md) · [实现方案](plans/stage-4.md)

<a id="stage-5"></a>

### 阶段 5：tensor map、TMA 与首次联测

128B对象复制/修改/发布、rank2输入和真实屏障完成通过后，立即做首次TMA→mbarrier→单次WGMMA→普通store联测。它不等组播、TMA输出和setmaxnreg；随后验两目标组播、输出源read完成和最终ack。

[详细验收](validation/stage-5.md) · [实现方案](plans/stage-5.md)

<a id="stage-6"></a>

### 阶段 6：矩阵、64 项结果和异步 WGMMA

先容量链路与普通shared准备的单次计算，再参加阶段5首次联测。矩阵存取分别验布局；连续累加、wait1/0、配额等待/归还和完整双缓冲必须通过。math ops/时间按一次逻辑矩阵运算核对。

[详细验收](validation/stage-6.md) · [实现方案](plans/stage-6.md)

<a id="stage-7"></a>

### 阶段 7：真实专家与九例

完整110模块和原启动器可用，逐次检查FC1/FC2，九例完整输出按既定规则通过。首个真实专家就更新墙钟/RSS预算；各例kernel和与主要调用分别比较。

[详细验收](validation/stage-7.md) · [实现方案](plans/stage-7.md)

完整用例顺序：Qwen64/128 → Qwen64/4096 → Phi/128、Mixtral/128 → Phi/4096、Mixtral/4096 → Qwen64/8192、Phi/8192、Mixtral/8192。每例独立进程，前一代表通过后推进；相关改动才重跑受影响检查。

## 性能比较的口径

**初期约 20% 比较对象是同批内核执行时间总和与主要调用。** 用户已接受将九例 ncu duration 总和作为带采集条件的设备估计参照，不宣称它是自然执行、含 DMA 的整层设备跨度。三个 4096 用例的 nsys 结果是交叉证据；大差异仍需查明，不按统一系数“去除采集开销”。

| 时间项 | 模拟端 | 参考端与用途 |
| --- | --- | --- |
| 每次内核及内核总和 | 从同次运行记录 kernel 设备开始/完成周期，排除主机提交等待和纯 launch 队列延迟；与参考起止范围一致 | 九例 ncu `gpu__time_duration.sum`；20% 初期目标，逐调用匹配 launch index |
| 前向 ROI | `MOE_ROI_BEGIN/END` 间累计设备周期差；拆出 kernel、launch/空档和显式复制贡献 | 裸跑 CUDA event 包含主机间隙；分列展示，不直接作为纯 kernel 误差 |
| 显式 DMA | 核实当前复制时间模型；记录方向/字节/顺序，未建模部分明确标记 | 三个 4096 的时间线有前向 DMA；缺失用例不填“实测零”，不能以 HBM 流量代替 |
| 主机提交、同步间隙 | 当前不要求新增通用 CPU/驱动时序模拟 | 只解释参考 event/device span 与活动时间差，不调 GPU 指令延迟吸收它 |

源码中的 `-gpgpu_perf_sim_memcpy` 主要填充 L2，stream copy 分支直接完成，不能据此声称 DMA 有完整周期模型。初期保持复制的正确性、顺序和缓存影响；仅当其耗时能解释显著整体差额时增加必要的传输建模。kernel 总和与完整 ROI 分开，不能把遗漏 DMA 藏入约 20% 结论。

计算：`T_sim_ms = cycles / core_MHz / 1000`；当前配置为 1980MHz，以每次实际配置为准。`relative_error = abs(T_sim-T_ref)/T_ref`，同时保存有符号的 `T_sim/T_ref-1`。全局 GPU 周期与跨 SM 的 elapsed avg/max/sum 不混用；后者结合该调用记录的有效频率解释，不能逐项相加当整层周期。初始缓存由真实初始化和先前内核形成，不为每次调用强制冷启动。

硬件基线如下，来自 [hardware-counters.json](../workloads/moe/evidence/hardware-counters.json)；nsys 来自 [kernel-times.json](../workloads/moe/evidence/kernel-times.json)。百分比为 ncu 总时间中的份额，非模拟预测。

| 用例 | ncu 内核和/ms | FC1 | FC2 | gating | nsys 内核和/ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixtral/128 | 1.186720 | 57.3% | 28.0% | 8.5% | — |
| mixtral/4096 | 5.884672 | 59.4% | 30.6% | 1.7% | 5.810761 |
| mixtral/8192 | 10.509216 | 58.2% | 31.2% | 1.7% | — |
| phi/128 | 1.089344 | 56.5% | 27.1% | 9.7% | — |
| phi/4096 | 3.078304 | 55.9% | 28.6% | 3.5% | 3.026607 |
| phi/8192 | 5.569152 | 56.0% | 28.5% | 3.2% | — |
| qwen64/128 | 0.561952 | 47.6% | 24.5% | 15.1% | — |
| qwen64/4096 | 1.186400 | 38.9% | 19.8% | 13.4% | 1.128616 |
| qwen64/8192 | 1.912480 | 39.1% | 19.3% | 9.8% | — |

三个 4096 的 ncu 内核和比 nsys 分别高 1.27%、1.71%、5.12%，但 ncu 是无预热 application replay，nsys 表值是预热后三次均值；跨条件差额不是纯工具开销。九例 ncu 都是三次 replay pass，不是三个独立时间样本。严禁使用采集过程中应用自身被放大的 event 时间验收。

每例报告至少包括：数值是否通过、kernel 时间和与误差方向、FC1/FC2/gating 的逐调用误差、ROI 与 DMA/主机间隙说明、采用的近似。再关注占比≥5%或能解释明显总差额的调用；较小 SwiGLU/重排作为定位证据，不要求每个小核达到20%。

校准时先解释主要差额：Tensor math ops 检查少算/重复和 tile padding；DRAM 字节、L2 sector×32 检查地址、缓存和组播；等待/重叠日志检查提前放行或串行化；最后调整有依据的延迟/吞吐。SASS 动态指令不等于 PTX 数量，TMA 流量也不等于 LSU load 数；未采集的 L1/occupancy/stall 不能当现成硬件证据。

## 通过条件与停止点

阻止执行、数值失败、错误同步/地址或主要时序失真必须在消费者阶段修正。小幅超限和低影响近似记录触发条件、影响、暂缓原因和后续触发条件，不为压到20%线以内反复调参。

全部九例数值通过，主要搬运/计算/等待/复用/重叠有证据，整体 kernel 时间和与主要调用接近初期目标且不存在被相互抵消的大偏差，即停止本轮精调。若仍有明显大差额，说明剩余原因和受影响用例，不写“支持完成”。九例自然运行完整设备时间、H200、更多数据类型或 cluster 形状不扩入本轮。

## 执行命令与记录位置

以下命令是后续实现阶段使用，本轮没有执行构建或 workload。按根 AGENTS.md 构建；本机 CUDA12.8 可用于冻结 PTX 的工具链，具体编译兼容性由阶段 0 验证。

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
cd gpu-simulator/gpgpu-sim
source setup_environment debug
make -j8
unset PTX_SIM_USE_PTX_FILE PTX_SIM_KERNELFILE CUOBJDUMP_SIM_FILE
unset MOE_TRACE_LAUNCHES
export PTX_SIM_MODE_FUNC=0
cd ../../workloads/moe
python3 run.py --case qwen64 --tokens 128 --mode sim --run-dir runs/qwen64-128
python3 compare.py artifacts/cases/qwen64-t128 runs/qwen64-128
```

`run.py` 检查库与冻结二进制/输入 hash，并限制一次无预热前向；同时核对复制的 H100 配置确实启用周期模式。`MOE_TRACE_LAUNCHES` 会额外同步，仅供定位。`compare.py` 的 ROI 差值不是上述性能验收表：后续另做简洁的逐调用汇总，读取现有 JSON/CSV 和模拟日志，不改数值比较规则。

从最早测试开始用 `/usr/bin/time -v` 记录 RSS 和墙钟时间；首个真实普通内核、gating 及代表性专家分别更新记录，规则见[验收目录](validation/README.md)。本机内存为用户确认的 256GB；仍按九个独立进程依次运行，不复制不必要的全量中间张量，也不以保存64项正确结果为内存优化对象。
