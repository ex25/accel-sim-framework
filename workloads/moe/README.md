# 固定的 Megatron-TE MoE workload

**固定 workload 阶段已完成。**三个模型配置、九个输入规模、程序、PTX 和完整参考输出均已固定；九用例在 H100 上的首次、重复前向都与框架参考逐位一致。[H100 记录](evidence/h100-validation.json)

当前只覆盖单卡、单层 MoE 前向：EP=TP=1，BF16 输入/权重/中间结果，FP32 累加，SwiGLU，无 bias、无丢 token。使用 seed=2026 的合成输入与权重；Qwen 按论文取 64 个路由专家，无共享分支。

| 模型 | H | 专家 FFN 维 I | 专家数 | top-k | token 数 |
| --- | ---: | ---: | ---: | ---: | --- |
| mixtral | 4096 | 14336 | 8 | 2 | 128 / 4096 / 8192 |
| qwen64 | 2048 | 1408 | 64 | 4 | 128 / 4096 / 8192 |
| phi | 4096 | 6400 | 16 | 2 | 128 / 4096 / 8192 |

框架为 Megatron Core 0.12.0 + TE 2.9.0 官方 CUTLASS 后端，路由/预处理使用 Inductor/Triton。版本与后端的选择、性能差异见[性能说明](docs/performance.md)。

## 文件入口

| 内容 | 用途 |
| --- | --- |
| [指令清单](docs/ptx-instructions.md) + [执行要求](docs/execution-requirements.md) | 当前九用例的模拟器实现范围；202 种写法的参数、行为、影响及官方链接 |
| [110 个模块的用途](docs/ptx-modules.md) | 按阶段分类，并列出全部模块的文件及各用例的调用顺序 |
| [artifacts](artifacts/) | 已验证的可执行程序、110 份 PTX、九套启动输入与完整参考；[清单](artifacts/workloads.json)记录对应关系 |
| [source](source/) | 当前 CUDA 启动器、内核注册表及框架层定义，供查阅启动和输入生成代码 |
| [性能基线 JSON](evidence/performance-baseline.json) / [CSV](evidence/performance-baseline.csv) | 九例首次与预热后计时、内核/DMA 摘要及硬件计数汇总；对比口径见[性能说明](docs/performance.md) |
| [硬件计数 JSON](evidence/hardware-counters.json) / [CSV](evidence/hardware-counters.csv) | 九例共 206 次内核调用的 SM 周期、DRAM/L2 流量、SASS 指令和 Tensor 工作量；[原始 ncu 报告](evidence/counters/) |
| [evidence](evidence/) | 完整数值记录、按调用位置区分 FC1/FC2 的内核耗时与时间线、构建和运行时接口核查 |
| [LICENSES](LICENSES/README.md) | 上游来源与许可证 |

`forward.bin` 保存初始化配方与固定的分配、启动、复制参数；GPU 计算仍由 PTX 执行。`reference.bf16` 保存完整参考输出，展开为 FP32 后的散列已与实际 H100 输出核对。一个前向启动 22～24 个内核，110 是九用例的去重集合；运行时从程序内加载 PTX，无需手动启动这 110 个文件。

## 实现完成后如何模拟

以下命令从项目根目录执行，使用本仓库 H100 配置和周期级 PTX 模式。运行现成程序，无需重新编译 workload，也无需安装 Megatron/PyTorch。

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
cd gpu-simulator/gpgpu-sim
source setup_environment debug
make -j8
unset PTX_SIM_USE_PTX_FILE PTX_SIM_KERNELFILE CUOBJDUMP_SIM_FILE
export PTX_SIM_MODE_FUNC=0
cd ../../workloads/moe

python3 run.py --list
python3 run.py --case mixtral --tokens 128 --mode sim --run-dir runs/mixtral-128
python3 compare.py artifacts/cases/mixtral-t128 runs/mixtral-128
```

这里清除覆盖变量，是因为仓库的 `setup_environment` 会将它们设置为示例文件名；本 workload 使用程序内嵌 PTX。`run.py` 检查加载的是模拟器 libcudart，复制 H100 配置，执行无预热的一次前向。将 `--case` 换成 `qwen64/phi`、`--tokens` 换成 `4096/8192` 即可运行其他用例，并使用对应的参考目录比较。

结果保存在 `runs/<用例>/`：`output.f32` 是输出，`stdout.log` 是模拟日志，`run.json` 记录输入与程序，`comparison.json` 报告数值差异及前向区间时间。数值默认阈值为 `atol=1e-5、rtol=0.02`，也报告是否逐位一致；严格比较可加 `--atol 0 --rtol 0`。设备周期与真实 GPU 时间的比较方式见[执行要求](docs/execution-requirements.md)。

九例数值、整层计时及逐调用硬件计数均已留档，可直接用于后续对照。`时间 × 1980 MHz` 仅为模拟频率折算周期，DMA 也不等于 HBM 访存量；无 profiler 计时与计数器采集分开保存，具体口径及采集影响见[性能说明](docs/performance.md)。模拟器的 MoE 执行路径仍待实现与验收。
