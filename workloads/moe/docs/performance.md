# Megatron-TE workload 的性能与验证

**已替换自写 MoE，九个用例在 H100 上的首次与重复前向均与框架参考逐位一致。**当前独立程序复用框架内核定义和 JIT 产物；TE/PyTorch CUDA 内核重新编译，Triton PTX 直接导出。它用于固定输入下的模拟器验证，原生框架仍用于框架层性能结论。

## 当前实测

| 模型 | token | 首次独立前向 / ms | 预热后独立 / ms | 预热后框架 / ms |
| --- | ---: | ---: | ---: | ---: |
| mixtral | 128 | 4.404 | 1.175 | 1.325 |
| mixtral | 4096 | 15.433 | 5.901 | 6.001 |
| mixtral | 8192 | 13.672 | 10.551 | 10.546 |
| qwen64 | 128 | 6.284 | 0.559 | 1.736 |
| qwen64 | 4096 | 6.684 | 1.159 | 1.771 |
| qwen64 | 8192 | 6.670 | 1.904 | 2.074 |
| phi | 128 | 9.104 | 1.079 | 1.416 |
| phi | 4096 | 8.662 | 3.054 | 3.204 |
| phi | 8192 | 25.739 | 5.612 | 5.694 |

预热后两列使用相同模型、seed、精度和单层前向：预热 3 次，每组 10 次前向，共 3 组，取组均值的中位数。首次列为无软件预热的一次前向，不表示硬件缓存为空。初始化与输出下载不计时，无 CUDA Graph。**它们来自两次 H100 任务，不是同卡交替的性能实验。**独立程序固定了主机启动计划，省去 Python 调度及部分动态元数据准备；CUDA 源码重新编译也可能改变原生指令。不能将总耗时差额全部归因于某个内核，也不能据此宣称优化了原生框架。

Mixtral/Phi 的 4096/8192 token 耗时与框架相差约 0%～5%；Qwen 对应约短 35%/8%。小 token 下主机开销占比更高。实际运行的 GPU 计算、数据类型和各用例输出均可核查。

[性能基线 JSON](../evidence/performance-baseline.json) / [CSV](../evidence/performance-baseline.csv)汇总九例首次/预热后计时、各组样本及硬件计数摘要。[完整 H100 记录](../evidence/h100-validation.json)保留数值校验与散列；[框架记录](../artifacts/workloads.json)保留版本与源码散列。设备为 H100 80GB HBM3、驱动 610.57.04。

三个模型的 4096-token 用例另有[逐次内核调用数据](../evidence/kernel-times.json)及 [SQLite 时间线](../evidence/timelines/)：按调用位置区分 FC1/FC2，保留每次耗时、grid/block、寄存器/shared 使用、内核时间总和、设备时间跨度及前向内 DMA 字节，已排除初始化与最终输出下载。**DMA 字节不是内核的 HBM/L2 访存量。**

**九例硬件计数已补齐：206 次内核调用均有下表 12 项有效指标。**[硬件计数 JSON](../evidence/hardware-counters.json) / [逐调用 CSV](../evidence/hardware-counters.csv)保存 SM 周期、DRAM 读写字节、L2 sector、SASS 指令及 Tensor 工作量；JSON 另有各用例流量汇总。九例采集前无 profiler 运行及采集后输出均与既有 H100 参考逐位一致，程序和输入未改。[采集记录及原始报告](../evidence/counters/)保留版本、命令和输出散列。本轮未采集 L1/TEX 流量、实际 occupancy 或停顿原因。

模拟器设备时间优先逐调用对照；整层 CUDA event 还包括主机提交间隙与必要复制，首次启动时间不能直接当作纯内核执行时间。硬件计数只覆盖前向内核，不含显式 DMA。

## 显存占用与模拟器主机内存

下表是固定启动程序同时保留的设备缓冲区总量，包含该层全部专家权重和中间数据；已逐例核对 `forward.bin`，原始字节数见[性能基线](../evidence/performance-baseline.json)的 `device_allocation_bytes`。单位 GiB = 2³⁰ 字节。

| 模型 | 128 token / GiB | 4096 token / GiB | 8192 token / GiB |
| --- | ---: | ---: | ---: |
| mixtral | 2.898 | 3.877 | 4.502 |
| qwen64 | 1.432 | 1.795 | 2.150 |
| phi | 2.617 | 3.127 | 3.621 |

这不是 GPU 进程总显存峰值：CUDA 上下文、模块等额外占用未采集。前文 DRAM 读写量是累计流量，同一份数据可被反复访问，不能拿它当存储容量。

PTX 模拟时，[模拟显存](../../../gpu-simulator/gpgpu-sim/src/cuda-sim/memory.h)保存在主机 RAM 中。本程序会先将全部缓冲区清零，因此这些页面都会建立存储；此外还要保存 PTX、线程寄存器和时序模型状态。专家 GEMM 声明了 10803 个 PTX 虚拟寄存器，模拟器[按线程保存实际写入的寄存器值](../../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc)，不能用 H100 的物理寄存器数估计这部分 RAM，也不表示每个线程都会用完所有声明。

**模拟器完整运行的主机内存峰值尚未实测。**按当前数据量与寄存器存储方式，单进程按 **64 GiB RAM** 规划是容量估计，不是实测最低要求；32 GiB 是否足够仍未知。九例各用独立进程依次执行，无需把九份占用相加。实现完成后，可在 `workloads/moe` 用下面命令记录峰值 RSS（进程实际驻留的主机内存）；`Maximum resident set size` 的单位为 KiB：

```bash
/usr/bin/time -v python3 run.py --case mixtral --tokens 8192 --mode sim
```

## 周期与访存指标的口径

时间线中的 `measured_cycles_at_sim_clock` 等字段按本仓库 **1980 MHz** 折算：`周期 = 时间(ns) × 1.98`，**不是实测 H100 周期**。硬件计数另存各次调用的 SM elapsed cycles 和频率：avg/max 是跨 SM 的平均值/最大值，须结合频率解释，不能直接相加当作整层周期。`sm__cycles_elapsed.sum` 累加所有 SM，也不能当作全 GPU 完成一次内核的周期。[官方指标结构](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)

<details>
<summary>已采集的 12 项最小指标：含义与单位</summary>

| 指标 | 单位及对比口径 |
| --- | --- |
| `gpu__time_duration.sum` | ns；ncu 自身测得的采集条件下内核时长 |
| `sm__cycles_elapsed.avg` | cycle；跨 SM 平均 elapsed cycles，结合该次时钟解释 |
| `sm__cycles_elapsed.max` | cycle；各 SM elapsed cycles 的最大值 |
| `sm__cycles_elapsed.avg.per_second` | Hz；该次采集的 SM 有效频率口径 |
| `dram__bytes_read.sum` | byte；DRAM 读取流量，非张量大小或 CUDA 拷贝大小 |
| `dram__bytes_write.sum` | byte；DRAM 写入流量 |
| `lts__t_sectors_op_read.sum` | sector；整个 L2 的读 sector 数，每 sector 32 byte |
| `lts__t_sectors_op_write.sum` | sector；整个 L2 的写 sector 数，每 sector 32 byte |
| `lts__t_sector_hit_rate.pct` | %；该内核的 L2 sector 命中率 |
| `smsp__inst_executed.sum` | inst；动态执行的 warp 级 SASS 指令数 |
| `sm__ops_path_tensor_src_bf16_dst_fp32.sum` | math op；BF16→FP32 Tensor 运算量，可含 tile padding |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | %；Tensor 管线相对 elapsed cycles 的活跃程度 |

</details>

实际采集使用 **ncu 2025.2.0**、application replay strict、`--cache-control none --clock-control none`，仅匹配 `moe_kernel_` 前向内核。每例无预热、一次前向，逐调用取得一份计数样本，所选指标需要 **3 个 replay pass**；这不等于三份独立统计样本。采集前另跑相同条件的无 profiler 程序。有效频率按每次调用保留，未由 ncu 锁频。[官方 CLI 选项](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

在 `workloads/moe` 中可离线查看已有报告，不申请 GPU、不重新执行：

```bash
ncu --import evidence/counters/mixtral-t4096/kernels.ncu-rep --page raw --csv --print-units base
```

TMA 有独立搬运路径，不能只用 LSU global-load 计数代表全部流量；L2/DRAM 计数反映对应存储层，组播到 shared 的总量又是另一口径。[官方 Memory Chart](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart) SASS 动态指令也不等于 PTX 指令数：Hopper 一次逻辑 WGMMA 会体现为四个 warp instructions；Tensor 运算量还可能包含尺寸填充。[官方 MMA 说明](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)

## 采集是否影响性能

**会，影响不能统一用一个百分比消除。**本次三个 4096-token 用例的 nsys event 时间，相对非采集的预热后计时分别为 Mixtral **−0.82%**、Phi **+0.37%**、Qwen64 **+0.85%**；它们来自不同运行及不同迭代长度，仅是观测差额，不能推断“采集开销不超过 1%”。

ncu 为不同指标可能重放内核并保存/恢复内存，也会受缓存清空、时钟控制、内核串行化和软件插桩影响。单 pass 时可避免保存/恢复，但精简指标不保证只有一 pass。本次使用 application replay，重跑应用以重建内存状态。ncu 自己报告的 duration 排除前后准备并避开软件插桩 pass；**ncu 运行期间应用自身的 CUDA event/主机计时不能作为自然运行时间**。[官方 Replay 与 Overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay)

本次九例采集中的应用 event 为同机采集前无预热单次运行的 **55.69～203.91 倍**，反映工具介入，不能理解为内核实际慢了这些倍。例如 Qwen64/4096 的裸跑 event 为 6.463 ms，采集时应用 event 为 1021.26 ms，而 ncu 内核 duration 总和为 1.1864 ms。应用 event 是最后一轮 replay 进程写回的时间，**不是三轮总和**；内核 duration 总和也不含主机间隙和 DMA，不能代替整层耗时。

因此保留独立无 profiler 的计时基线，计数器单独采集；只选必要指标，并记录 replay、cache、clock 设置。本次 application replay 配合 `--cache-control none` 由应用建立前序内核的缓存状态；`--clock-control none` 避免 ncu 自行改频，同时保存有效频率指标。这能降低、隔离影响，不能保证完全消除。ncu 与 nsys 的起止时间定义也有差异，不能把两者时长差额全部归为采集开销。[官方时长比较与控制说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#workload-durations)

## 为什么使用这个框架配置

采用论文所述的 **Megatron-TE**：Megatron Core 0.12.0 `MoELayer → TEGroupedMLP/GroupedLinear`。固定 TE 2.9.0 官方 BF16 CUTLASS grouped GEMM（128×128×128 tile、1×2×1 cluster），开启框架的融合重排与 SwiGLU。为取得完整设备代码，对路由及预处理启用 PyTorch Inductor，矩阵乘法后端限定 Triton；top-k/sort 使用对应 PyTorch CUDA 源码。来源及许可证见 [LICENSES](../LICENSES/README.md)。

生成时设置 `NVTE_USE_CUTLASS_GROUPED_GEMM=1`，在 `torch.inference_mode()` 下构造并运行层。`router.gating` 使用 `torch.compile(fullgraph=True)`，`router.routing` 和 `token_dispatcher.preprocess` 使用 `fullgraph=False`；编译选项为 `max_autotune=True、max_autotune_gemm_backends="TRITON"`。[层定义与初始化](../source/framework_layer.py)已保留供查阅。

这里固定的是可导出 PTX 的 Megatron-TE 版本/后端，**不是论文原版软件栈的逐版本复现**。CUDA 来源模块的 PTX header 从 8.8 降为 8.7 后，均通过 CUDA 12.8 ptxas；计算指令未改写，entry 只改为唯一名称。

框架前向包括路由、重排、专家两次 GEMM、SwiGLU 和加权合并。独立程序保留这些设备工作与必要复制，固定输入对应的主机元数据；它不是通用 PyTorch 运行时。其他 Transformer 模块仍需单独实测后外推；模拟器的周期准确度尚未验收。
