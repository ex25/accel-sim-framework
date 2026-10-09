# 阶段 8：A800 与模型验证

状态: 已验收

当前结论（2026-10-10）：按本文件解阻计划完成验收，聚合
`STAGE8_RESULT status=PASS numerical_cases=8 blockers=0`。原冻结 FA3 PTX 完整
parser readiness、参数 ABI、阶段 3–7 回归、普通 attention 数值/时序对照、真实
A800 trace/replay 和管理员 7 项 counters 均通过。最终结果与能力边界以本文末尾
“最终验收”节为准，以下旧失败记录保留为历史证据。

外层：本报告所在外层阶段 8 提交。配对内层：
`d3494f34c75bea3edfbcace59dc1f31701597fb8`。两仓库各 amend 原阶段 8 提交。

## 目标和实施前计划

阶段 7 已验收。本阶段固定阶段 1 的非 causal attention workload：B=1、S_Q=S_K=128、H=8、D=D_V=128、FP16、BSHD；另生成 S=64/256 仅用于趋势检查。CPU reference 使用 FP64 计算 `softmax(QK^T/sqrt(128))V`，统一误差标准为逐元素 `abs_error <= 2e-3 + 2e-3*abs(reference)`，并拒绝 NaN、Inf、错误 shape 和错误长度。

对照分为四层：普通 CUDA kernel 在真实 A800 上计时；等数学问题的文本 PTX 在 Accel-Sim functional/detailed 路径执行；阶段 1 冻结 FA3 PTX 做 ptxas/SASS 静态 inventory 与 parser readiness 检查；A800 NVBit SASS trace 用现有 trace parser 和 SM80 model 回放。没有把 A800 比例换算成 Hopper 周期，也没有把静态 SASS 当动态 trace。

## 前置条件、架构和工具

- 外层和内层均在 `research/hopper-ptx`；本阶段基线外层 `509a1442e29ff9aae88d4c30cf7267cf9d734613`，内层 `a7d9951175826c586f6d9f7d67380beac5468ec2`。
- 阶段 1 固定输入位于 `research/hopper-ptx/stage1/artifacts/inputs/`，Q/K/V 的完整 SHA256 和 BSHD shape 在 `inputs.json`；reference 对每次运行重新校验。
- 硬件为 NVIDIA A800 80GB PCIe、compute capability 8.0、108 SM、驱动 570.133.20；CUDA 12.8.61，`ptxas` 12.8.61，Nsight Compute 2025.1.0。A800 命令必须在沙箱外执行，因为沙箱内驱动不可见。
- A800 kernel 是阶段目录中的普通 CUDA FP16-input/FP32-accumulate 实现，使用 128-thread block（S=256 时为 256 threads）、1 CTA/query/head；它不是 FA2/FA3，也不宣称 Tensor Core 实现。文本 PTX 同样是可运行的 CUDA-core attention 对照，不冒充冻结 FA3。

## 实现和数据流

阶段目录 `research/hopper-ptx/stage8/` 包含：

- `reference.py`：生成 64/128/256 输入/reference，检查固定输入 SHA256，执行统一数值契约和负向输出契约。
- `attention.cu`、`attention.ptx`：A800 与文本 PTX 对照 kernel，输出 raw FP16 BSHD，记录 CUDA event kernel time、host transfer+kernel interval、registers、static shared、理论 occupancy、逻辑访存和 FLOPs。
- `attention_harness.cc`：在当前 PTX runtime 上运行 functional/detailed，逐 CTA/head 写回输出；detailed 同时记录模拟 cycles。
- `sass_inventory.py`：对编译器产生的冻结 FA3 `sm_90a` cubin 做静态 opcode inventory，明确 `HGMMA`、`UTMALDG`、`UTMASTG`、`SYNCS`、`WARPGROUP`、`USETMAXREG` 数量和“无动态字段”。
- `sass_oracle.cc`：用真实 PTX metadata 加合成 `HGMMA`/TMA trace fields 检查 decoder。输出标记 `origin=synthetic_decoder`，不能替代 Hopper 硬件 capture。
- `trace_audit.cc`、`trace_sim`：审计 A800 NVBit v5 文本 trace 的版本、grid/block、global 地址和 barrier，再用仓库 SM80 trace model 回放。
- `run_model.sh`、`run_a800.sh`：分别运行模型/回归和 A800/trace/profile 流程；profile 权限失败写入 `profiler-readiness.json`，不会伪造 DRAM/occupancy 数值。

## 命令

前置构建需要阶段 3 的 `/tmp/hopper-stage3-build` CMake build tree；先执行
`cmake --build /tmp/hopper-stage3-build --target cudart -j2`。模型和 parser/SASS 验证：

```bash
bash research/hopper-ptx/stage8/run_model.sh /tmp/hopper-stage8-final-model
```

真实 A800 验证（沙箱外）：

```bash
bash research/hopper-ptx/stage8/run_a800.sh /tmp/hopper-stage8-acceptance-a800
```

聚合验证（8 个数值场景、4 个 negative contract 通过，仍因缺失证据返回 exit 3）：

```bash
python3 research/hopper-ptx/stage8/verify_results.py \
  /tmp/hopper-stage8-final-model /tmp/hopper-stage8-acceptance-a800 \
  /tmp/hopper-stage8-stage-result.json
```

`run_a800.sh` 中的 Nsight Compute 使用 `gpu__time_duration.sum`、DRAM read/write bytes、achieved active warps、SM/DRAM throughput 和 registers；工具本身若无 performance-counter 权限则保留失败证据。CUDA event 只用于 unprofiled elapsed kernel time；Nsight replay overhead 不混入该时间。[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 实际结果

### 数值和趋势

- reference 生成 S=64/128/256，分别 65,536/131,072/262,144 元素；三份 reference SHA256 写在 `/tmp/hopper-stage8-final-model/reference.log`。
- A800、PTX functional 和 PTX detailed 的统一逐元素检查均通过。完整 S=128、8 heads 的 PTX functional 最大绝对误差 `2.443e-4`；A800 S=128 最大绝对误差 `2.444e-4`；S=64/128/256 的 A800 中位 kernel time 为 `43.008/142.336/556.032 us`，随序列长度单调增加。PTX detailed 单 CTA cycles 为 `218905/370567/708229`，同样单调增加。
- PTX functional 与 A800 的完整 S=128 输出直接比较也通过：最大差异 `0.00048828125`，仅 `26` 元素有 FP16 舍入差异；没有用固定 A800/Hopper 倍率做校准。
- A800 编译资源为 34 registers/thread、1540 bytes static shared、理论 occupancy 0.75。逻辑访存和 FLOPs 由算法公式记录，不能当作物理 DRAM 计数器。

### A800 硬件 trace 与模型

- NVBit capture 的一个 CTA/head 通过 `trace_audit`：`warp_instructions=6206`、`scalar_instructions=134375`、`global_read_bytes=65792`、`global_write_bytes=256`、请求 32B sectors `17424`、12 个 warp barrier events。Q/K/V global read 和 FP16 output write 与固定算法一致。
- 同一 trace 通过仓库 SM80 trace model 回放；`gpu_tot_sim_insn=134375` 与 capture 一致，`gpu_tot_sim_cycle=32979`，模拟输出正常退出。该结果验证 parser/model 的可重放性，不证明模型周期等于 A800 硬件周期。
- 冻结 FA3 `sm_90a` cubin 的静态 SASS inventory 为 2612 条，包含 `HGMMA.64x128x16.F32=22`、`HGMMA.64x176x16.F32=16`、`UTMALDG.4D=18`、`UTMASTG.5D=2`、`SYNCS.PHASECHK.TRANS64.TRYWAIT=41`、`WARPGROUP.DEPBAR.LE=4`、`USETMAXREG.*=2`。这是 compiler-produced static SASS，不是 Hopper dynamic trace。
- 阶段 6 WGMMA timing regression 的 N=128/176 wait order、阶段 5 TMA detailed regression 和 synthetic decoder oracle 均通过；oracle 明确标注 `origin=synthetic_decoder`。

## 首次验证：未通过项和失败证据（历史）

本阶段不能标记 `已验收`，原因如下：

1. Nsight Compute 返回 `ERR_NVGPUCTRPERM`，当前用户没有访问 A800 performance counters 的权限；因此实际 DRAM bytes、achieved occupancy 和 SM/DRAM bottleneck 指标仍未验证。非交互 `sudo -n ncu` 同样报告需要密码；未修改驱动或权限。
2. 冻结 FA3 PTX 可以由 `ptxas -arch=sm_90a` 编译，但当前文本 PTX parser 在 `.param .align 64` 处断言（`ptx_parser.cc::add_function_arg` 旧限制只允许 1/2/4/8/16）；因此完整 FA3 kernel 尚未通过 execution-driven parser/runtime。
模型限制：没有 H100/H200 硬件动态 trace；HGMMA/TMA decoder 结果来自真实冻结 PTX metadata 加合成动态字段，不能写成 Hopper 硬件验证。计划本来允许无 Hopper 实测，这项用于限制结论，不另加硬件验收条件。

失败日志和证据：`/tmp/hopper-stage8-acceptance-a800/ncu128.log`、`ncu128.csv`、`profiler-readiness.json`、`/tmp/hopper-stage8-final-model/frozen-fa3-parser.log`、`frozen-readiness.json`、`frozen-sass.json`、`a800-trace-audit.log`、`a800-sass-model.log`。完整 raw A800 trace 在 `/tmp/hopper-stage8-acceptance-a800/a800-traces/`。

## 首次验证：验收结论和下一阶段入口（历史）

已通过：统一 reference/numerical contract、A800 数值与趋势、PTX functional/detailed 对照、A800 SASS capture/parser/model replay、静态 Hopper SASS inventory、阶段 5/6 回归。阶段状态保持 `阻塞`，不能进入阶段 9；需要获得 Nsight counter 权限、修正 `.param .align 64` parser 后再复验。

## 首次验证：提交和限制（历史）

外层：本报告所在外层提交（validation harness、实际证据和报告在同一阶段提交）。

配对内层：本阶段没有修改模拟器源码，沿用 `a7d9951175826c586f6d9f7d67380beac5468ec2`，不制造空提交。用户已有 `experiments/` 和内层 `.serena/` 不纳入提交。没有 H100/H200 实测，不承诺硬件周期准确度；A800 kernel 是兼容趋势对照，不能代表 FA3/Hopper performance。

## 随提交保存的证据

`research/hopper-ptx/stage8/evidence/` 保存 `results.json`、`verification.log`、
`attention-model.log`、`a800-*.log`、真实 `a800-one-cta.traceg.gz`、trace 审计/回放日志、
静态 SASS inventory、synthetic oracle、GMMA wait/latency events、TMA regression、
失败 readiness 和诊断。`provenance.json` 记录 input SHA256、runtime SHA256、配对内层
SHA 和来源；`sha256.json` 校验每个证据文件。原始完整日志仍留在上述 `/tmp` 目录。

瓶颈判断只作实现解释：K 的一个 lane 沿 D 连续读取而同一 warp 的 lanes 访问不同
key，因此 trace 的 17,424 个请求 32B sector 对应 557,568 请求字节，高于 66,048
逻辑 global 字节；这说明访问模式存在 coalescing 放大，但不是实际 HBM traffic。
softmax 由单线程执行，且没有 Tensor Core/TMA；没有 counter 数据，不能定量认定
A800 的 SM/DRAM 主导瓶颈。SM80 回放的 6.2418% 模拟 occupancy 也不能代替硬件
achieved occupancy。A800 full-grid timing 与 PTX single-CTA cycle 的工作量不同，
只检查各自趋势，不计算相互倍率。

## 首次验证：源码依据（历史）

配对内层 commit `a7d9951175826c586f6d9f7d67380beac5468ec2` 的
`src/cuda-sim/ptx_parser.cc::add_function_arg` 是冻结参数 alignment 断言所在符号；
`cuda-sim.cc::ptx_thread_info::ptx_exec_inst`、`instructions.cc::ld_exec/st_impl`
负责本阶段 plain PTX 的功能执行；`gpgpu-sim/shader.cc` 复用已有 WGMMA/TMA timing。
外层基线 `509a1442e29ff9aae88d4c30cf7267cf9d734613` 的
`gpu-simulator/trace-driven/trace_driven.cc::trace_warp_inst_t::parse_from_trace_struct`
填充 HGMMA latency/II/commit、TMA mbarrier/bytes/OOB/mask；
`gpu-simulator/trace-parser/trace_parser.cc::inst_trace_t::parse_from_string` 解压捕获
的 global 地址；`gpu-simulator/accel-sim.cc::accel_sim_framework::simulation_loop`
运行 real A800 trace 的时序回放。本阶段通过外层 harness 直接调用上述源码，内层
没有阶段 8 修改。

## 阻塞分析与解阻计划

本阶段当前不是数值正确性失败，而是两个独立的验收前置条件未满足：A800 硬件性能
计数器不可访问，以及冻结 FA3 PTX 尚未通过 Accel-Sim parser readiness。二者必须分开
处理，不能用模型估算替代硬件 counter，也不能把 `ptxas` 编译成功或静态 SASS
inventory 误写成 Accel-Sim 已经能够解析、执行冻结 FA3。

### 执行环境和权限原则

本研究的主要工作环境是 `in-switch`，默认工作目录为
`/home/wangjz/in-switch/accel-sim-framework`。代码修改、parser 修复、模型构建、
functional/detailed harness、证据整理和阶段报告都优先在该环境完成；命令中的相对
路径均以该目录为基准。只有 A800 驱动、Nsight Compute 或 NVBit 明确要求真实 GPU
和沙箱外权限时，才把对应步骤迁移到有硬件访问权的外部环境，并记录外部环境和
`in-switch` 工作树之间的 commit、输入和证据对应关系。

如果某一步因为没有 `sudo`、`CAP_SYS_ADMIN`、`CAP_PERFMON` 或 NVIDIA 驱动管理权限
而无法执行，AI 不绕过权限、不猜测结果，也不把失败标记为通过。AI 应先在
`in-switch` 中生成一个最小、可审阅、可重复运行的脚本，脚本必须：

- 说明需要的权限、目标主机和预期副作用；
- 在执行前打印设备、驱动、模块参数和工具版本；
- 使用 `set -euo pipefail`，检查命令和目标设备，避免静默修改错误主机；
- 将原始 stdout/stderr、返回码和环境信息保存到指定证据目录；
- 默认只做只读检查；涉及驱动重载、模块参数或系统配置时，明确标为管理员动作，
  不在 AI 侧自动执行；
- 给出用户运行脚本、回传证据和继续验证的具体命令。

用户在有权限的环境中运行该脚本后，AI 再根据实际日志继续分析和更新报告。若用户
无法运行或权限仍未授予，阶段保持 `阻塞`，并保留可复现的失败证据。

### 1. 当前阻塞的具体原因

#### 1.1 A800 counter 权限阻塞

`run_a800.sh` 的数值运行、CUDA event 计时、NVBit trace capture 和 SM80 trace
replay 已通过；失败发生在 Nsight Compute 读取硬件 performance counters 的阶段：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

`sudo -n ncu` 也因需要密码失败。因而当前缺少真实的 `dram__bytes_read.sum`、
`dram__bytes_write.sum`、achieved active warps、SM throughput 和 DRAM throughput。
`theoretical_occupancy`、逻辑访存字节和 CUDA event 时间不能填充这些字段。

解决路径只有两类：

1. 在具备管理员权限的 A800 主机上运行 Nsight Compute，或请管理员按 NVIDIA 的
   [ERR_NVGPUCTRPERM 指南](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
   授予 `CAP_SYS_ADMIN`/`CAP_PERFMON`，或调整 `NVreg_RestrictProfilingToAdminUsers`
   后重新加载 NVIDIA 驱动；
2. 把 counter 采集迁移到已经开放 profiling 的同型号 A800，重新记录设备信息、
   驱动版本、CUDA 版本和证据 hash。

如果上述动作需要 `sudo`，下一步不是让 AI 代替用户提权，而是运行
`research/hopper-ptx/stage8/run_a800_counter_probe.sh`。这个脚本只检查当前权限、
`/proc/driver/nvidia/params`、设备和工具版本，然后运行 `ncu`，保存 CSV、日志、
设备信息和返回码。需要修改 `NVreg_RestrictProfilingToAdminUsers` 或重新加载驱动的
命令必须作为管理员操作清单提供，由用户或管理员手动执行，不能由 AI 自动调用。

不能通过 `/proc`、CUDA event、理论 occupancy 或固定倍率推导实际 DRAM traffic。
如果当前机器无法获得权限，本项应继续保持 `BLOCKED`，但其它模型、数值和 trace
证据可以独立复验。

#### 1.2 冻结 FA3 parser 阻塞

`stage1/artifacts/frozen_fwd.ptx` 可以被 `ptxas -arch=sm_90a` 编译，但 Accel-Sim
的 `ptx_recognizer::add_function_arg` 在
`gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_parser.cc:674-682` 对 alignment 只允许
`1/2/4/8/16`。冻结 kernel 的入口包含：

```ptx
.param .align 64 .b8 ...[2368]
```

因此 frontend probe 在 parser 内部 assertion 中止，`frozen-readiness.json` 为
`BLOCKED`。PTX 参数 alignment 是 ABI 布局信息，不能简单删除 `.align 64` 或强制
降为 16；这样会改变后续参数 offset，造成错误的 kernel 参数语义。

另外，修复 64-byte alignment 只解决当前首个 parser blocker。冻结 FA3 还包含
`elect.sync`、`prefetch.tensormap`、`setmaxnreg` 等现代指令或 modifier；parser
readiness、functional execution、detailed timing 是三个不同等级，必须逐级记录，
不能看到 parser probe PASS 就宣称 FA3 execution-driven 已完成。`setmaxnreg` 的
真实 register-pool redistribution 仍属于未实现语义，不能在本阶段用空操作伪造。

#### 1.3 当前环境的复现性问题

当前默认 `/home/wangjz/miniconda3/bin/python3` 没有 `numpy`，直接运行
`verify_results.py` 会在导入阶段失败。已有 `/tmp` 结果和阶段证据仍然有效，但复验
环境没有被完整固定。解阻时必须使用包含 `numpy` 的 Python 环境，并记录
`python3 --version`、`numpy.__version__`、`ptxas --version`、`ncu --version` 和
CUDA driver 信息；不能把环境导入失败混写成模型或硬件失败。

### 2. 实施顺序

#### 2.1 建立可复现的验证环境

先不修改模拟器代码，创建或选择一个能运行 `reference.py` 和
`verify_results.py` 的环境。检查以下命令：

```bash
python3 - <<'PY'
import sys, numpy
print(sys.version)
print(numpy.__version__)
PY
/usr/local/cuda/bin/ptxas --version
/usr/local/cuda/bin/ncu --version
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv
```

把实际解释器路径和依赖版本写入阶段证据，例如
`/tmp/hopper-stage8-final-model/environment.txt`。`requirements.txt` 当前没有
`numpy`，因此应在阶段报告中明确使用的外部环境；是否修改项目依赖文件要单独评估，
不能为了本次验收静默改变全局环境。

#### 2.2 修复 parser 的参数 alignment 表示

在内层 `ptx_parser.cc`、`ptx_ir.h` 和参数布局相关路径中做最小、可验证的修改：

1. 将合法 alignment 检查改为接受 PTX 允许的正数二次幂，并设置明确的上限；至少
   覆盖冻结 FA3 使用的 64，不接受 0、非二次幂、整数溢出或不合理的大值。
2. `function_info::add_config_param` 的 offset 计算使用无符号宽度安全的向上对齐，
   检查 `offset + size` 溢出；保持 `.param` 起始地址和后续参数布局一致。
3. 检查 `function_info::finalize`、参数复制和 `param_info` 的 alignment 来源，确保
   parser 记录的 64-byte 对齐会实际影响 parameter memory，而不是只绕过 assertion。
4. 增加正向 fixture：`.param .align 32`、`.param .align 64`，同时增加负向 fixture：
   非二次幂、零值、超出实现上限和导致 offset 溢出的声明。
5. 先运行 frontend probe 和参数布局断言，再运行普通 stage 3/4/5/6/7 回归，确认
   alignment 修复没有改变已有标量参数、vector 参数和 barrier/TMA harness 的布局。

预期结果是 parser 能接受冻结 kernel 的入口参数，并打印稳定的参数 offset/size；
不是立即承诺冻结 FA3 能够 functional 或 detailed 执行。

#### 2.3 建立冻结 FA3 的 parser readiness 清单

每次 parser 运行只修复一个首个失败类别，保存完整日志和版本信息。建议循环：

```bash
make -C research/hopper-ptx/stage3 all
LD_LIBRARY_PATH=/tmp/hopper-stage3-build \
  research/hopper-ptx/stage3/frontend_probe \
  research/hopper-ptx/stage1/artifacts/frozen_fwd.ptx \
  <kernel_name> > /tmp/hopper-stage8-final-model/frozen-fa3-parser.log 2>&1
```

对每个失败，记录它属于：

- lexer/parser 语法缺口；
- operand/type/modifier 缺口；
- 已能解析但 functional implementation 缺口；
- 需要 detailed timing 或 runtime API 的独立阶段缺口。

本阶段只接受有源码依据、不会伪造语义的 parser readiness 修复。若遇到
`setmaxnreg`、真实 tensor-map runtime、完整 FA3 WGMMA/TMA execution 等超出阶段 8
范围的项目，保留严格失败日志，转入阶段 9 或对应后续实现计划；不能用 no-op 让
probe 虚假通过。

#### 2.4 获取并验证 A800 counters

在管理员或开放 profiling 的 A800 环境中，由用户运行仓库中的
`research/hopper-ptx/stage8/run_a800_counter_probe.sh`，先完成最小权限检查和
counter probe，再运行完整 S=128 profile。脚本
输出应保存到 `/tmp/hopper-stage8-acceptance-a800/permission-check/` 和
`/tmp/hopper-stage8-acceptance-a800/`，不要覆盖已有失败日志：

```bash
cat /proc/driver/nvidia/params | rg 'RestrictProfiling|RmProfiling'
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv
/usr/local/cuda/bin/ncu --target-processes all --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread --csv \
  --log-file /tmp/hopper-stage8-acceptance-a800/ncu128.csv \
  research/hopper-ptx/stage8/attention \
  /tmp/hopper-stage8-acceptance-a800/inputs/128 128 1 \
  /tmp/hopper-stage8-acceptance-a800/a800-profile-128.f16le
```

如果该命令仍需要 `sudo`，由用户在有权限的终端运行对应脚本或同等命令，并把完整
日志带回 `in-switch`；AI 只负责检查日志、更新证据索引和继续后续模型验证。

检查 CSV 中每个要求的 metric 都有有效数值；若仍为 `ERR_NVGPUCTRPERM`、空值或
replay failure，生成 `profiler-readiness.json` 为 `BLOCKED`，保留原始日志，不填写
估计值。成功后重新运行 S=64/128/256 的普通 CUDA kernel，确认 counter 采集没有
改变数值 contract、输入 hash 或 trace capture。

### 3. 重新验证和证据组织

两个 blocker 可以并行准备，但验收必须在各自证据完整后统一执行：

```bash
cmake --build /tmp/hopper-stage3-build --target cudart -j2
make -C research/hopper-ptx/stage8 clean all
bash research/hopper-ptx/stage8/run_model.sh /tmp/hopper-stage8-final-model
bash research/hopper-ptx/stage8/run_a800.sh /tmp/hopper-stage8-acceptance-a800
python3 research/hopper-ptx/stage8/verify_results.py \
  /tmp/hopper-stage8-final-model \
  /tmp/hopper-stage8-acceptance-a800 \
  /tmp/hopper-stage8-stage-result.json
```

验收前逐项检查：

- `frozen-readiness.json` 为 `PASS`，且日志显示 parser 实际读取了冻结入口，而非
  替换过 alignment 或裁剪过的 PTX；
- parser alignment 正/负 fixture 均通过，已有阶段 3-7 回归没有变化；
- `profiler-readiness.json` 为 `PASS`，CSV 包含要求的硬件 counter，设备 UUID 和
  driver/CUDA 版本与报告一致；
- 八个数值场景、四个 negative contract、A800 trace audit、SM80 replay、静态 SASS
  inventory、WGMMA/TMA regression 仍然通过；
- `results.json` 状态才可从 `BLOCKED` 变为 `PASS`，且 `blockers` 为空；
- `provenance.json`、`sha256.json`、原始日志和新的环境清单同步更新。

建议把新证据放入新的唯一目录，例如 `/tmp/hopper-stage8-acceptance-final/`，避免
覆盖当前 `/tmp/hopper-stage8-final-model` 和 `/tmp/hopper-stage8-acceptance-a800`，
并在报告中同时记录旧失败证据和新成功证据。

### 4. 停止条件、降级和退出路径

以下任一情况发生时，阶段 8 继续保持 `阻塞`：

- A800 counter 仍无权限，或用逻辑字节、event 时间、理论 occupancy 替代真实 counter；
- parser 通过修改输入 PTX、删除 `.align 64`、截断冻结 kernel 或把 unsupported
  instruction 变成 no-op；
- alignment 修复改变已有参数 offset，导致 attention harness 或前置阶段回归失败；
- parser probe 通过但 functional/detailed 运行实际遇到未实现 Hopper 语义，且没有
  单独记录为阶段 9 限制；
- `verify_results.py` 无法在记录的环境中运行，或证据 hash 与 provenance 不一致；
- A800 trace/model replay、数值 contract 或负向 contract 回归失败。

当 parser readiness 和 A800 counter 均通过，且聚合验证返回 `STAGE8_RESULT
status=PASS blockers=0` 后，补写“实际结果”和“验收结论”，将状态改为 `待验收`，
再由阶段验收改为 `已验收`。在此之前不得开始阶段 9 的 runtime integration；若只有
counter 已通过而冻结 FA3 parser 未通过，或反之，仍不能解除阶段门控。

## 解阻计划执行记录（2026-10-09）

### 环境与配对版本

本节执行上面的用户计划，保留原始失败证据和计划正文。外层仍使用本报告所在的
阶段 8 提交（原 `4cc2fa0ef50c271848494cdeed7ef31acb9ea610` amend，不新增阶段
提交），配对内层为 `318094db33dbd1528a40452de5819984004b6a08`；两个仓库均在
`research/hopper-ptx`。上述旧报告中的内层 `a7d99511` 是此前测量的版本，本节新证据
使用四个参数路径源码修改后的版本，SHA 和源文件 hash 见新 `provenance.json`。

选择现成的 `/home/wangjz/miniconda3/envs/in-switch/bin/python3`，Python 3.11.16、
NumPy 2.4.6；没有安装依赖或修改全局环境/requirements.txt。模型、硬件和回归脚本
支持 `PYTHON_BIN`，默认固定这个解释器，counter 脚本另外把实际解释器写入日志。
CUDA/ptxas 12.8.61、Nsight Compute 2025.1.0、A800 UUID
`GPU-32ca4776-99e3-43b4-2f9d-fd4f90a01a3d`、驱动 570.133.20 未变。
`environment.txt` 记录复验前 HEAD，最终配对 SHA 以 `provenance.json` 为准。

### 参数 ABI 修改与验证

内层 `src/cuda-sim/ptx_parser.cc/.h` 的 `add_alignment_spec`、`add_function_arg`、
`add_identifier` 接受正数二次幂 alignment，上限明确为 1 MiB（本实现限制，不是
PTX 架构上限），拒绝零、非二次幂、过大值、整数溢出及 array-size/offset 溢出。
`g_size` 使用 `size_t` 保存自然 element alignment，数组总大小来自 symbol，避免把
数组长度当自然 alignment；函数参数 vector 大小/对齐也实际参与布局。

`ptx_ir.h` 的 `function_info::checked_param_offset/add_config_param` 先检查 unsigned
地址范围和 padding/size，再向上对齐。`cuda-sim.cc` 的 `get_args_aligned_size` 使用
同一配置设置 `param_info::m_offset` 和 symbol 地址；`add_param_data` 不再覆盖它，
`finalize` 不再按复制长度重新布局。复制尾段使用 `min(4, remaining)`，避免 1/2 字节
参数越界；截短 host 数据不会改变下一 formal 参数的地址。

新增 `param_layout_probe.cc/.py` 实际解析 PTX、读取布局并调用 `finalize`，检查内存
内容、padding 和尾部 8-byte guard。四个正例通过：32/64-byte 显式对齐（offset
0/64）、标量+v4 参数（0/2/16/32）、3-byte 数组后接 u64（0/8）、冻结入口相同的
64-byte 对齐 2368-byte 参数。五个负例通过：48、0、2 MiB、超出整数宽度、unsigned
offset 溢出；均检查诊断而不把 timeout 当成功。

最初直接复用旧阶段 harness binary 出现 ABI 不匹配的失败，已通过 clean rebuild
处理。最终 `run_param_regressions.sh` 对阶段 3-7 harness 全部 clean build 后运行，
前端正反例、mbarrier、TMA functional/detailed/multicast、WGMMA functional/detailed
及负例、cluster/DSMEM/严格失败契约全部通过：

```text
PARAM_LAYOUT_RESULT status=PASS positive=4 negatives=5 copy_guard=PASS
PARAM_REGRESSIONS status=PASS stages=3,4,5,6,7
```

### 冻结 FA3 readiness 和权限停止点

没有改动 `stage1/artifacts/frozen_fwd.ptx`。原首个失败 `.param .align 64` 已修复，
新的首个失败是原文件第 46 行：

```ptx
elect.sync %rx|%px, %r324;
```

`probe_frozen.py` 保存整个 probe 日志、kernel 名、返回码、原始输入 SHA256、首个
错误和类别（lexer/parser syntax）；返回码 1，readiness 为 `BLOCKED`。完整原文件
仍可由 ptxas 编译为 sm90a cubin，静态 SASS 仍为 2612 条。`elect.sync` 涉及 warp
选举和 active mask，不能用通用 opcode/no-op 绕过；在本阶段保持严格失败，并作为
下一次 frontend 工作项。`prefetch.tensormap` 尚未被本次 probe 到达，不能断言它已
通过；真实 setmaxnreg register-pool、tensor-map runtime 和完整 FA3 execution 也
未实现。阶段 9 尚未开始。

只读 A800 counter probe 已在沙箱外运行，记录 `RmProfilingAdminOnly: 1`，ncu 返回
1、`ERR_NVGPUCTRPERM`，脚本返回 3。离开沙箱解决驱动可见性，不提供管理员 counter
权限。本次未调用 sudo、修改驱动/module 参数、授权 capability 或重载驱动。
`run_a800_counter_probe.sh` 已准备好由用户/管理员运行：现有目录下创建唯一 attempt
子目录，保留版本/UUID/capability、stdout/stderr、CSV 和返回码。

`check_ncu_csv.py` 按 **Metric Name/Metric Value 列** 校验七项指标，要求每个出现的
数值都有限且非负，不从 ID/其他数字字段找替代值。CSV 契约测试覆盖有限值、空值、
NaN、Inf、负值、缺失项、replay 错误、第二 launch 无效值，共八类全部通过。
硬件 sweep 和聚合验证也使用这个检查器。只有 counter 与数值校验都通过才打印
probe PASS；成功后脚本再运行 S=64/128/256 的普通 kernel 数值复验。

### 最终复验命令与结果

新的唯一运行目录为 `/tmp/hopper-stage8-unblock-PRlrUlTr`，未覆盖原先两组验收目录。
实际执行：

```bash
cmake --build /tmp/hopper-stage3-build --target cudart -j2
make -C research/hopper-ptx/stage8 all
bash research/hopper-ptx/stage8/run_param_regressions.sh /tmp/hopper-stage8-unblock-PRlrUlTr/regressions-final
bash research/hopper-ptx/stage8/run_model.sh /tmp/hopper-stage8-unblock-PRlrUlTr/model-final
bash research/hopper-ptx/stage8/run_a800.sh /tmp/hopper-stage8-unblock-PRlrUlTr/a800
bash research/hopper-ptx/stage8/run_a800_counter_probe.sh /tmp/hopper-stage8-unblock-PRlrUlTr/counter-probe
/home/wangjz/miniconda3/envs/in-switch/bin/python3 research/hopper-ptx/stage8/verify_results.py \
  /tmp/hopper-stage8-unblock-PRlrUlTr/model-final \
  /tmp/hopper-stage8-unblock-PRlrUlTr/a800 \
  /tmp/hopper-stage8-unblock-PRlrUlTr/results.json
```

模型/回归和普通硬件运行返回 0；counter 脚本和聚合验收返回 3。模型脚本结束后单独
执行新增 parameter-layout/counter-contracts/probe_frozen 步骤，保存在同一个 model-final
目录；最终脚本已包含这些步骤。八个数值场景及四个负向输出契约全部通过，PTX/A800
直接最大差仍为 0.00048828125；detailed 周期为 218905/370567/708229。A800 30 次
运行的 median 为 41.984/141.312/556.032 us，仅用于趋势对照。新的真实 NVBit trace
仍为 6206 warp instructions、134375 scalar instructions、65792 read bytes、256
write bytes、17424 requested sectors、12 warp barrier events；SM80 replay 的 instruction
count 和 trace 一致。WGMMA wait-order 和四个 TMA detailed regression 通过。

```text
STAGE8_RESULT status=BLOCKED numerical_cases=8 blockers=2
```

### 新证据与下一步

新证据在 `research/hopper-ptx/stage8/evidence/unblock-20261009/`，原 evidence 保留。
该目录包含环境、results、参数/CSV契约结果、readiness、A800 sweep、trace audit、
静态 SASS、`provenance.json` 和 `sha256.json`。完整原始日志和 PTX fixture 归档在
`raw-logs.tar.gz`；新的真实 one-CTA trace 在 `a800-one-cta.traceg.gz`。归档保持原始
字节，独立文本副本仅规范尾部空白；input/source/runtime/trace hash 同步记录。

阶段 8 继续 **阻塞**，没有标为待验收或已验收。剩余工作是实现并验证实际冻结入口的
`elect.sync` 等 frontend readiness 缺口，以及由有权限的用户/管理员采集 counter。
按本计划“由用户运行”且“不能由 AI 自动调用”管理员操作的约束，用户可在有 profiling
权限的终端运行：

```bash
bash research/hopper-ptx/stage8/run_a800_counter_probe.sh /tmp/hopper-stage8-user-counters
```

若需要管理员身份，由用户或管理员自行决定并执行上述脚本；它本身没有驱动修改或
sudo 调用。得到完整 counter 后仍须重新进行聚合验证，并同时解决 frozen parser
blocker，两个条件都满足才能重新申请阶段验收。


## Counter 脚本可移植性修复（2026-10-10）

用户实际运行在 capability 记录处遇到 `rg: command not found`，`set -euo pipefail`
使脚本在 ncu 之前退出。这是脚本依赖缺陷，与 GPU counter 权限无关。修复目标是
不要求用户安装 ripgrep，让脚本到达真实硬件探测并保留准确的成功/失败判断。

`research/hopper-ptx/stage8/run_a800_counter_probe.sh` 的两处文本筛选改为系统
`grep -E`；第三处 readiness 状态判断改用已选 Python 解释器解析 JSON。已检查
整个脚本不再调用 rg。原 `/tmp/hopper-stage8-user-counters/counter-probe.log` 保留，
重复运行创建独立 attempt 子目录。

验证命令：

```bash
bash -n research/hopper-ptx/stage8/run_a800_counter_probe.sh
PATH=/usr/bin:/bin PYTHON_BIN=/home/wangjz/miniconda3/envs/in-switch/bin/python3 \
  bash research/hopper-ptx/stage8/run_a800_counter_probe.sh /tmp/hopper-stage8-user-counters
/home/wangjz/miniconda3/envs/in-switch/bin/python3 research/hopper-ptx/stage8/test_ncu_csv.py \
  /tmp/hopper-stage8-no-rg-yeqKGyvs/csv-contracts
```

该 PATH 下 `command -v rg` 无结果。真实硬件探测在沙箱外成功经过 capability、
设备、驱动、Python/NumPy 检查和输入生成，并到达 ncu；此次 run 目录为
`/tmp/hopper-stage8-user-counters/attempt-20261010T020206Z-4116313`。没有再次出现命令缺失，ncu 返回 1、脚本返回 3，实际原因仍为
`ERR_NVGPUCTRPERM` 和 `RmProfilingAdminOnly: 1`；CSV 八类契约测试通过。
因此只修复脚本可运行性，阶段 8 仍为阻塞；管理员操作仍遵循上面的用户计划。

证据位于 `research/hopper-ptx/stage8/evidence/no-rg-20261010/`，含用户失败日志、
实际重跑日志、capability、设备、readiness、CSV 契约结果、provenance/hash。
此前 `unblock-20261009` 证据及源码 hash 是历史快照；本次脚本的 SHA256 单独记录
在新 provenance。配对内层仍为 `318094db33dbd1528a40452de5819984004b6a08`，
外层为本报告所在阶段 8 提交，继续 amend，保留单一阶段提交。


## 管理员 counter 采集接入（2026-10-10）

用户按上述计划自行执行 `sudo bash research/hopper-ptx/stage8/run_a800_counter_probe.sh
/tmp/hopper-stage8-admin-counters`，UID=0，`ncu_exit=0`，七项指标有效，profile 输出及
随后的 S=64/128/256 普通 CUDA 数值复验全部通过。本次 AI 只读取用户生成的证据，
没有执行 sudo、修改驱动或重新采集管理员 counter。A800 UUID、570.133.20 驱动、
CUDA 12.8.61、ncu 2025.1.0、Python 3.11.16/NumPy 2.4.6 与原记录相符。

真实 S=128 profile CSV（grid=128×8×1，block=128×1×1）记录：

| Metric | 实测值 | 单位 |
|---|---:|---|
| gpu__time_duration.sum | 183264 | ns |
| dram__bytes_read.sum | 803712 | byte |
| dram__bytes_write.sum | 0 | byte |
| sm__warps_active.avg.pct_of_peak_sustained_active | 53.91 | % |
| sm__throughput.avg.pct_of_peak_sustained_elapsed | 7.80 | % |
| dram__throughput.avg.pct_of_peak_sustained_elapsed | 0.23 | % |
| launch__registers_per_thread | 34 | register/thread |

保留原始零写字节计数，不以逻辑 store 大小替代；该指标是此次采集的 DRAM 流量，
不能等同于 kernel 的全部逻辑访存。183.264 us 是 ncu profile 测量，与普通 CUDA
event sweep 分别记录，也不能拿来推算 Hopper 周期。

发现 counter 脚本在 ncu 成功路径没有写 `profiler-readiness.json.exit`，聚合检查
要求 `exit=0`，会误把采集成功判为阻塞。现已在所有路径统一记录实际 ncu exit；
`verify_results.py` 另外将独立 CSV 校验结果及真实 metric 值写入聚合 JSON。
原管理员目录保持不变；验证副本根据 `counter-probe.log` 的 `ncu_exit=0` 补写 exit
和证据来源，未从 PASS 推测退出码。新的脚本无需再次管理员采集即可验证此次结果。

在唯一目录 `/tmp/hopper-stage8-admin-validation-VMalMyxr` 合并证据：模型沿用
2026-10-09 的 `model-final`，硬件使用管理员采集的 CSV、profile 输出、post-profile
sweep 和 inputs；NVBit trace/replay 仍为原 2026-10-09 的真实采集，明确不是本次
sudo 脚本重采。三个 S 的 input/reference manifest 完全一致，各输出 hash 保存在
provenance；新 profile 输出再次独立 compare 通过。没有覆盖旧权限失败日志。

```bash
/home/wangjz/miniconda3/envs/in-switch/bin/python3 research/hopper-ptx/stage8/verify_results.py \
  /tmp/hopper-stage8-admin-validation-VMalMyxr/model \
  /tmp/hopper-stage8-admin-validation-VMalMyxr/hardware \
  /tmp/hopper-stage8-admin-validation-VMalMyxr/results.json
```

八个数值场景、四个负向输出契约、七项真实 counter、trace/replay、原有 TMA/WGMMA
验证均通过；CSV 八类契约与脚本 bash syntax 检查通过。聚合返回 3：

```text
STAGE8_RESULT status=BLOCKED numerical_cases=8 blockers=1
```

唯一剩余 blocker 是原冻结 PTX 第 46 行 `elect.sync` 的 lexer/parser syntax 缺口。
硬件权限 blocker 已解除，阶段 8 仍不满足全部验收条件，未开始阶段 9。

新证据在 `research/hopper-ptx/stage8/evidence/admin-counters-20261010/`，包含完整
CSV/输出、设备/工具/capability、counter readiness、profile/sweep 数值结果、聚合
result、provenance/hash；原始管理员日志另存 `raw-admin-logs.tar.gz`。配对内层保持
`318094db33dbd1528a40452de5819984004b6a08`；外层为本报告所在阶段 8 提交，继续
amend 单一阶段提交。历史证据和原始管理员目录都保留。


## 最终验收（2026-10-10）

本次按用户的 2.3 节计划继续修复唯一剩余前端 blocker。A800 counters 使用用户在
`/tmp/hopper-stage8-admin-counters` 采集的真实成功结果；AI 没有执行 sudo、修改
驱动/module/capability。普通 A800 数值 sweep、NVBit trace 和当前模型 replay 则
在新的 `/tmp/hopper-stage8-readiness-20261010/hardware` 重新运行。设备 UUID
`GPU-32ca4776-99e3-43b4-2f9d-fd4f90a01a3d`、输入和 reference 与管理员采集逐文件
一致，`attention.cu` 也与管理员采集时源码一致，因此复用 counters 有明确依据。
当前普通用户仍不能访问 counters，新的权限失败日志另外保留在
`hardware/user-counter-failure`，没有覆盖原管理员记录。

### 冻结原文逐项解阻

每次只处理原文首个失败类别并完整保存 build/parser 日志。先后结果如下：

| 迭代 | 首个失败 | 修复后的下一项 |
|---|---|---|
| 00 基线 | 第 46 行 `elect.sync` | 01 后推进到第 65 行 |
| 01 | `elect.sync` | `prefetch.tensormap` |
| 02 | 第 65 行 `prefetch.tensormap` | 第 184 行 init fence |
| 03 | `fence.mbarrier_init.release.cluster` | 第 4996 行 `stmatrix` |
| 04 | `stmatrix.sync.aligned.x4.m8n8.shared.b16` | 第 5105 行 TMA store |
| 05 | `cp.async.bulk.tensor.5d.global.shared::cta.bulk_group` | 第 5113 行 bulk group |
| 06 | bulk commit/read-wait 语法 | 第 5154 行 `griddepcontrol.wait` |
| 07 | grid dependency 语法 | inline asm 局部标签汇编断言 |
| 08 | 嵌套作用域标签地址解析 | 完整 frontend PASS |

内层 `src/cuda-sim/ptx.l`、`opcodes.def`、
`ptx_parser.cc::{add_cluster_opcode,add_tma_opcode,add_instruction,finish_vector_operand}`
仅为上述精确形式提供 lexer/operand/type/modifier 检查，保留原 opcode 字符串。
TMA store 不生成 load/mbarrier 元数据；非支持 modifier 仍失败。新语法通过
`instructions.cc::{elect_impl,prefetch_tensormap_impl,hopper_frontend_only_impl}`
明确拒绝执行，并经现有 `ptx_thread_info::ptx_exec_inst` 加上源码位置。没有 no-op。
`wait_group.read` 本阶段只接受原冻结使用的 immediate 0，其他值留待后续实现。
`elect.sync` 支持冻结使用的 32-bit register|predicate destination，sink destination
形式仍不支持；`stmatrix`、TMA store 仅支持表中冻结形式。

局部标签问题在 `cuda-sim.cc::function_info::ptx_assemble` 和
`ptx_ir.cc::symbol_table::set_label_address`：inline asm 标签属于子作用域，原来
按名字在函数表查找会断言，也可能混淆同名标签。现在 label/CFG/reconvergence
统一按 symbol identity 解析，直接写回 parser-owned label 的地址，instruction
memory padding 初始化为 null；不存在的 label 仍拒绝，错误改写 stderr 防止 abort
前缓冲丢失。这里属于实际汇编语义修复，已验证跳转结果。

原 `stage1/artifacts/frozen_fwd.ptx` 未裁剪、未替换 alignment；SHA256 仍是
`0ee436723623d4f2d3363b36cc4ed9d5ecdb96a6c449367a66356c705a351a76`。
最终 probe 读取完整入口，`target=sm_90a target_number=90 instructions=3025`，
退出码 0。3025 是包含 label 的前端条目数，不能当作动态指令数。

### 实际命令和结果

以下工作目录均为外层仓库根目录；Python 固定为已有 `in-switch` 环境，Python
3.11.16、NumPy 2.4.6，CUDA/ptxas 12.8.61、ncu 2025.1.0、driver 570.133.20。

```bash
cmake --build /tmp/hopper-stage3-build --target cudart -j2
make -C research/hopper-ptx/stage8 clean all
bash research/hopper-ptx/stage8/run_param_regressions.sh \
  /tmp/hopper-stage8-readiness-20261010/regressions-final
bash research/hopper-ptx/stage8/run_model.sh \
  /tmp/hopper-stage8-readiness-20261010/model-final
# 沙箱外运行普通测试；此脚本不调用 sudo。
bash research/hopper-ptx/stage8/run_a800.sh \
  /tmp/hopper-stage8-readiness-20261010/hardware
# 核对 UUID、全部输入、kernel 源码；只在验证副本合入原管理员 counter CSV/输出。
# actual ncu_exit=0 来自原 counter-probe.log；原始文件保持不变。
/home/wangjz/miniconda3/envs/in-switch/bin/python3 \
  research/hopper-ptx/stage8/verify_results.py \
  /tmp/hopper-stage8-readiness-20261010/model-final \
  /tmp/hopper-stage8-readiness-20261010/hardware \
  /tmp/hopper-stage8-readiness-20261010/results.json
```

全部命令完成；聚合退出 0：

```text
FRONTEND_CONTRACTS status=PASS positive=12 negative=19 strict_execution=24
PARAM_REGRESSIONS status=PASS stages=3,4,5,6,7
FROZEN_FA3_RESULT status=PASS exit=0 first=None
STAGE8_RESULT status=PASS numerical_cases=8 blockers=0
```

`frontend_contracts.py` 另运行两种模式的同名局部标签 fixture，得到 `value=3`，并
验证未定义标签失败。12 个正向项包含 `setmaxnreg`；24 次 functional/detailed
执行必须抛预期异常且 guard 不变，不能静默执行下一条 store。19 个负向项覆盖
非法 modifier、mask/destination/address/source 宽度、vector/rank 和 group operands。
参数 ABI 4 正向/5 负向及 copy guard、counter CSV 8 类契约也通过。

最终完整 S=128、8 heads functional 最大绝对误差 `0.00024430074510806143`；
八个 PTX/A800 数值场景、四个负向输出契约和 direct compare 全通过。direct compare
最大差异 `0.00048828125`，26 元素有舍入差异。detailed S=64/128/256 cycles 仍为
`218905/370567/708229`；新 A800 普通 CUDA event median 为
`41.984001/141.311996/556.031982 us`，分别单调增加。

新 NVBit trace 仍有 6206 warp instructions、134375 scalar instructions、65792
read bytes、256 write bytes、17424 requested sectors、12 warp barrier events。
当前 runtime 的 SM80 replay 同样退出成功，`gpu_tot_sim_insn=134375`、
`gpu_tot_sim_cycle=32979`。配对内层 amend 后重建 trace_sim 并再次回放；其版本
明确记录新的 inner SHA。7 项真实 counter 值见上节管理员表，CSV SHA256 与原
管理员证据一致；profile 输出独立复验通过。WGMMA wait order、TMA detailed、
synthetic SASS oracle 和冻结编译器 static SASS inventory 仍通过。

### 验收证据、提交与后续边界

最终证据：`research/hopper-ptx/stage8/evidence/acceptance-20261010/`。
`results.json`、`verification.log`、`frozen-readiness.json`、前端/ABI/counter
contracts、真实 CSV、A800 trace/replay、`iterations.json`、环境清单和 hash/provenance
均随本阶段提交保存；`raw-validation-logs.tar.gz` 包含 771 个完整日志、fixture、
配置/版本/patch 文件；新原始 trace 另保存 gzip。数值文件 hash 记录在 provenance，
原文件仍在上述 /tmp 目录。原失败证据目录保持历史内容。

配对内层提交：`d3494f34c75bea3edfbcace59dc1f31701597fb8`；外层为本报告所在阶段
8 提交。构建时 HEAD 是 amend 前版本加当前工作树，provenance 用源码 SHA256 对照
最终 inner commit；runtime SHA256 是
`d75cc6330e416dad3e041834171e6586fcae18fc09bc774c12bed613f93944da`。
外层与内层都 amend 原阶段 8 提交，完整正文记录问题、范围、测试、证据和限制。

阶段 8 标记 **已验收**，没有开始阶段 9。后续必须分别实现并验收 `elect.sync`
warp rendezvous、真实 tensor-map descriptor/cache/runtime、init fence、`stmatrix`、
TMA store/bulk completion、grid dependency 和 `setmaxnreg` register-pool 行为，才能
声称完整冻结 FA3 functional/detailed execution；当前这些路径严格失败。本阶段
没有 H100/H200 dynamic trace，不能声称 Hopper 硬件周期准确度。上述限制符合用户
计划的 parser readiness 与 execution 分级，不作为本阶段隐藏 blocker。

语法与操作数依据：[CUDA 12.8 PTX ISA 8.7 官方规范](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)，
参见 `elect.sync`、`prefetch`、`membar/fence`、`stmatrix`、`cp.async.bulk.tensor`、
bulk commit/wait 和 `griddepcontrol` 各指令节。
