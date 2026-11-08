# 阶段 0 实际实现：加载与启动

状态：**已完成实现、自测并通过独立验收**。最小 PTX 测试已覆盖目标名、参数传递、两种启动入口、动态 shared、失败声明和两个模块交替调用。通过范围与限制见[验收记录](../../validation/stage-0/README.md)。

[实现方案](../../../plans/stage-0.md) · [验收要求](../../../validation/stage-0.md) · [独立验收记录](../../validation/stage-0/README.md) · [交付规则](../../README.md)

## 当前结论与本轮范围

PTX 周期模式现在可按完整 `sm_90a` 名称取得 ptxas 资源信息；64B 对齐和1344B参数可传递；直接 `cudaLaunchKernel` 和编译器生成的 Push/Pop 启动按本次 grid/block/shared/stream 执行；动态 shared 进入本次 kernel 的资源驻留、缓存划分、占用和释放；含 `.noreturn __assertfail` 的模块可加载，实际调用和 `trap` 都以源位置报错终止；两个独立fatbinary可按A→B→A定位正确入口与数据。

## 实际改动与计划对应

同一项有多处不同作用的改动可拆行；文件/函数写准确链接，并注明修改还是新增。没有改动就保持未开始，不能预填拟实现方案。

| 对应 plan 内容 | 实现状态 | 修改/新增的文件、函数或状态 | 作用、行为变化及涉及指令/接口 |
| --- | --- | --- | --- |
| [保留代码的完整架构目标名](../../../plans/stage-0.md#target) | 完成 | [cuda_api_object.h](../../../../gpu-simulator/gpgpu-sim/libcuda/cuda_api_object.h)、[cuda_runtime_api.cc](../../../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc)、[gpgpu_context.h](../../../../gpu-simulator/gpgpu-sim/libcuda/gpgpu_context.h)、[ptx_loader.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_loader.cc) | 以数字能力排序，同时按文件保存完整目标；ptxas 对 `sm_90a` 仍传完整目标名 |
| [参数对齐：让主机写入的位置与内核读取的位置一致](../../../plans/stage-0.md#parameters) | 完成 | [ptx_parser.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_parser.cc) `ptx_recognizer::add_function_arg` | 接受64B对齐，复用既有参数偏移算法；混合参数和1344B对象均逐字节检查通过 |
| [启动配置：使直接 API 和编译器入口都传对本次参数](../../../plans/stage-0.md#launch) | 完成 | [cuda_api_object.h](../../../../gpu-simulator/gpgpu-sim/libcuda/cuda_api_object.h)、[cuda_runtime_api.cc](../../../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) `cudaLaunchKernelInternal`、`__cudaPopCallConfiguration` | Pop回填配置并出栈；直接API建立配置，使用当前context设置参数并传播真实提交结果。四轮交替及非默认stream复制/启动/复制通过 |
| [动态 shared：区分允许申请多少与本次实际申请多少](../../../plans/stage-0.md#shared) | 完成 | [ptx_ir.h](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h)、[abstract_hardware_model.h](../../../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h)、[abstract_hardware_model.cc](../../../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc)、[shader.cc](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)、[gpu-sim.cc](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) | 函数保存许可上限、每个kernel保存本次字节数；每CTA 32800/65568B 导出驻留上限7/3，并在32→64→32KiB顺序下正确运行；外部shared地址接到entry静态shared末尾 |
| [外部失败函数声明：正常路径可加载，失败路径仍失败](../../../plans/stage-0.md#declarations) | 完成 | [ptx.l](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l)、[ptx.y](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y)、[instructions.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc)、[cuda-sim.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) | 接受 `.noreturn;`；`__assertfail` 和 `trap` 分别报源位置后终止。trap指令不再进入读取普通数值类型的未实现处理器 |
| [多模块与可复用测试入口](../../../plans/stage-0.md#modules) | 完成 | [stage-0 tests](../../../../workloads/moe/tests/hopper/stage-0/)；`cuobjdumpInit_internal` | CUDA 6+逐PTX提取不再执行旧section裁剪；A→B→A输出11、22、11。新增入口可按测试选择并记录实际库、配置和日志 |

## 实现说明与复用

实际实施后，用必要的小段解释关键数据/状态如何流动、为什么在此处改、如何连接功能与时序。逐项说明实际复用/借鉴了哪段 SASS 或共用模型、PTX 补了什么。只记录非显然的设计、主要近似和与计划的实质差别，不逐行复述代码。

没有改动SASS trace执行路径。沿用已有PTX参数符号、stream队列、CTA调度和shared页式存储；PTX新补本次launch的shared状态和外部动态shared地址。动态shared只参与原有资源占用/驻留模型，不增加搬运模型或独立延迟。`cudaFuncSetAttribute` 仅实现当前消费者需要的最大动态shared属性，其它属性仍走原警告路径。

CUDA 6+环境使用 `extract_ptx_files_using_cuobjdump_internal` 逐文件提取，不产生 section parser 所需的PTX/ELF列表；因此只在旧CUDA路径运行 section prune/merge。直接API测试中的 grid/block 参数按 `cudaLaunchKernel` ABI使用 `dim3`，与计划表中的简写相同。

## 构建与自测证据

| 自测项目/指令形式 | 被测版本与测试入口 | 实际结果及运行证据 | 当前限制 |
| --- | --- | --- | --- |
| 所有实现文件 | 内层基准 `4195dd4f229cbf03f4f38c49139ae94d01d23f03` 加工作区差异 | `export CUDA_INSTALL_PATH=/usr/local/cuda; source setup_environment debug; make -j8` 成功，CUDA 12.8、GCC 13.4.0 | 与修改共用资源计算相关的trace回归没有现成小trace；本阶段用PTX最小测试覆盖共用占用/释放 |
| 启动、算术依赖、多模块、参数、动态shared、正常/失败声明 | [stage-0 runner](../../../../workloads/moe/tests/hopper/stage-0/run.py)，证据目录 [final-rerun](../../../../workloads/moe/runs/hopper-stage-0/final-rerun/) | 默认H100参数、单SM/单chiplet测试副本，模拟库为内层 `libcudart.so.12`，`PTX_SIM_MODE_FUNC=0`；8项结果全部符合预期 | 测试配置只缩减 SM/chiplet 数量，容量和周期参数不变；错误路径按预期非零退出 |
| 动态shared资源 | 同上，另用 `--concurrent` 配置运行 `shared` | 32KiB→每CTA 32800B→驻留7；64KiB→65568B→驻留3；回到32KiB恢复7；每次三个标记为101/202/77 | [资源分支记录](../../../../workloads/moe/runs/hopper-stage-0/resource-check/)；单SM配置不用于H100性能比较 |
| 较长依赖计算 | runner `work` 与 `launch` 首次内核比较 | 64线程输出正确；100轮依赖运算输出正确，单次模拟墙钟约1秒增至约2秒，设备总周期有对应增长 | 仅检查时序随工作量增长，不对小测试设置硬件周期精度门槛 |

自测应覆盖本次必要数值、同步/访存顺序、主要时序和受影响回归；准确命令、配置、输出/周期、墙钟/RSS链接到实际报告，不把大日志复制进本文。

## 未完成、暂缓与验证反馈修复

阶段0必要子集已由执行 agent 自测并通过独立验收。阶段0不解决冻结launcher的 tensor map driver API、动态shared超过96KiB、132SM地址范围或cluster；这些按计划分别留给后续阶段。SASS小trace不可用，未声称已有SASS运行回归。

收到验证反馈时，链接对应失败记录，说明实际修正的代码、根因和影响范围；尚未复验就写“已修复，待验证 agent 复验”，不得代改独立验收结论。

## 版本、附件与交接

- 验收基准：外层 `b80607e5235e190930813353b8db0b1550b48cac`、内层 `4195dd4f229cbf03f4f38c49139ae94d01d23f03` 加当时工作区差异。阶段交付：内层提交 `62205297c2b0c7ba9ba1e275e6a9874f0c284a02`；两个仓库均以 `stage0` 标签定位配套版本，标签说明记录双方提交号。
- 测试入口：`python3 workloads/moe/tests/hopper/stage-0/run.py --run-dir workloads/moe/runs/hopper-stage-0/final-rerun --test all`；实际加载库、工具链、配置散列、命令和每项退出状态见运行目录 `run.json` 与各测试 `.log` / `.time`。
- 本机运行最大RSS约274MiB；逐程序墙钟和初始化/执行主机时间戳见相应 `.time` 与日志。设备总周期见各日志 `gpu_tot_sim_cycle`。
- 交给验证 agent：先独立运行上述 `all` 入口；重点复核目标资源命令、参数内容及偏移、shared每CTA值/驻留7↔3/资源释放、Push/Pop与直接启动、预期失败状态和A→B→A映射。验证 agent 应填写独立验收记录，不把本表自测视为独立通过。

只维护本阶段这份实际实现说明；原 plan 保留方案并链接到这里。独立验收结果以对应 validation README 为准。
