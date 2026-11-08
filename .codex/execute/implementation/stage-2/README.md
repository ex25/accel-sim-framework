# 阶段 2 实际实现：普通 cp.async

状态：**已完成（实现与自测）**。普通异步拷贝能经过真实缓存/内存请求，在返回后写入 shared，并按每个线程的提交组等待；debug/release 构建、12 类 PTX 微测、分组组件测试、两个冻结 gating 全矩阵数值及受影响回归通过。[独立验收](../../validation/stage-2/README.md)已通过，覆盖 246 条结果与事件断言；当前未验收九例完整前向或硬件性能精度。

[实现方案](../../../plans/stage-2.md) · [验收要求](../../../validation/stage-2.md) · [独立验收记录](../../validation/stage-2/README.md)

## 现在能做什么

支持 `cp.async.cg.shared.global [dst], [src], 16, src_size`、`cp.async.commit_group`、`cp.async.wait_group N`。实际需求中的 src_size=0/16、predicate、空组及 N=0/2/4/6 均已自测。copy 发起后，独立计算和其它 warp 可继续执行；wait 等待所需组，保留原程序自己的 `bar.sync`。

```text
发射 copy → 保存该次 UID、CTA/shared 和 16B 数据 → L2/DRAM 请求
                    │                              ↓ 最后一个请求返回
                    └→ commit → 独立计算       写 shared → 完成该 lane 的组
                                  ↓                         ↓
                                wait ←─────────────────────┘
                                  ↓
                           原程序的同步与 shared 读取
```

阶段 0/1 以前置交付为准：外层基准 `d7056c242e48cfbf78e9283078a674a79d2d6fc8`，内层 `b25fd0a18cb01a0c112042295007a0ac2c3772ad`；两个工作区开始时干净。本轮只实施阶段 2，不启动 cluster、TMA、WGMMA 或完整前向。

## 实际改动与计划对应

下表文件属于独立内层仓库，链接从本记录定位。

| 对应 plan | 文件、函数或状态 | 实际行为与原因 |
| --- | --- | --- |
| [解析与发起](../../../plans/stage-2.md#copy) | [ptx.l](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l)、[opcodes.def](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/opcodes.def)、[ptx_ir.h](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h)/[.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.cc) | 新增三种 opcode，识别当前实际使用的完整 mnemonic，保留四个操作数和 predicate。无需改 `ptx.y`：现有 opcode/operand 语法已经足够，避免扩展未用修饰组合。 |
| 同上：依赖和动态数据 | [cuda-sim.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) `pre_decode`、`ptx_exec_inst`；[instructions.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) `cp_async_impl`；[abstract_hardware_model.h](../../../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h)/[.cc](../../../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc) `execute_warp_inst_t` | copy 没有输出寄存器；shared 地址、global 地址和 src_size 都是读取依赖。逐 lane predicate 求值后，针对本次动态指令生成一次数据载荷；捕获真实 CTA 的 shared 对象，读取有效源，其余零初始化。发起时不写 shared。 |
| [真实返回交付](../../../plans/stage-2.md#completion) | 同上 `generate_mem_accesses`/`memory_coalescing_arch`；[shader.cc](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) `ldst_unit::cycle`、`unset_depbar` | copy 是 16B global load、`.cg` 策略，复用 LDGSTS UID 请求计数。合并访问时排除零源 lane，保留执行 mask。最后返回后的共用 `unset_depbar` 入口先写 shared 再完成各 lane 的组；纯零源在正常 LSU 无请求结束路径交付，不伪造读取。copy 在派发队列排空而请求仍未返回时不重复统计完成。 |
| [逐线程分组与 wait](../../../plans/stage-2.md#groups) | 新增 [cp_async.h](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/cp_async.h) `cp_async_groups`；[shader.h](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h) `m_ptx_cp_async_groups`；`issue_warp`、`waiting` | 每 lane 保存单调提交序号、未完成组计数及 wait 截止。空组也推进序号；wait N 等编号 `< max(0,C-N)` 的组。完成后删未完成项，不按剩余队列长度重算 C，不检查 SASS 的最近组类型。只停本 warp 的后续发射，LSU 和其它 warp 继续推进。 |
| [shared 复用与退出](../../../plans/stage-2.md#reuse) | 数据载荷的 shared 指针；`reset`/`init`、`hardware_done` | 回调使用发起时捕获的 shared，不按当前硬件线程槽重新查找；warp 仍有未完成 copy 时不能释放/复用 CTA。槽位重新初始化时清空组状态。原 barrier 位置和语义保留。 |
| [真实 gating](../../../plans/stage-2.md#gating) | 外层 [stage-2 测试](../../../../workloads/moe/tests/hopper/stage-2/) `real.py`/`real.cc`/`read_forward.py` | 沿用阶段 0/1 fatbinary 注册、单模块入口和记录方式。直接解码原 `forward.bin` 参数/launch，只替换指针；独立 host 小整数矩阵和 CPU 整数累加参考。未修改冻结 PTX。 |

## 数据、分组和时间如何连接

每次动态 copy 的 32 个 lane 载荷只有一份 `shared_ptr<vector<ptx_cp_async_copy>>`；流水线和 `mem_fetch` 只共享它，没有随每个返回重复复制所有 lane 数据。LDGSTS 原有的 `m_pending_ldgsts[warp][uid]` 判断哪个动态请求完成；功能数据和目的地址来自这份载荷。同一 PC、甚至同一源地址的循环实例仍由不同 UID 区分。

组状态只存未完成项，因此不另存一份已完成组队列。commit 序号保留空组和已完成组的位置；copy 可早于 commit 完成。每 lane 的 wait 截止保持到下一次参与 wait，未来新增 copy 的组序号总不小于旧截止，不会把未参与本次 wait 的线程意外变成新等待者。已有 SIMT 调度在 warp 层停发，不引入线程独立调度或 PC 重试。

实际复用 SASS/common 的 global 合并、`.cg` 绕过 L1D、L2/DRAM 延迟和带宽、返回仲裁、UID 计数及普通 shared 消费者 bank/端口。SASS 原有 PC/地址匹配组逻辑保持原分支；PTX copy 在数据交付后直接返回，不使用它。

沿用计划的近似：只读输入在发射时快照；整条 warp 指令最后请求返回时统一交付。未细化逐 lane 最早返回、异步 shared 写端口/bank 费用，不额外发一条普通 store，不对 SM 或每次 copy 插全局同步。

## 构建与自测证据

最终数值、设备计数、主机成本与事件断言见 [summary.json](../../../../workloads/moe/runs/hopper-stage-2/summary.json)，由 [summarize.py](../../../../workloads/moe/tests/hopper/stage-2/summarize.py) 读取保存的证据并断言生成。自测不替代独立验收。

| 自测 | 结果与证据 |
| --- | --- |
| src_size=16/0、predicate=false | 16 lane 读取、8 lane 零填、8 lane 保留 `a5`；实际 global/L2 read 为 8 个 32B sector。全 warp 零源无 global/L2/DRAM read，仍交付全零并完成 wait。[数值](../../../../workloads/moe/runs/hopper-stage-2/delivery-micro/run.json)、[事件](../../../../workloads/moe/runs/hopper-stage-2/delivery-micro/events-copy_mix.log)。 |
| wait0/2/4/6 状态边界 | 人工反序完成新组，旧组 0/1 未完仍等待；先完成旧组允许最近 N 组未完成；未提交 copy 不进入 wait0，空组仍前进。组件测试明确不称为 PTX 运行。[groups_component.log](../../../../workloads/moe/runs/hopper-stage-2/delivery-micro/groups_component.log)。 |
| PTX 真实组、空组和逐 lane 序号 | 四种 N 均比较等待前两组及排空后全部组。高半 warp 先提交空组、全 warp 随后同 PC copy，交付组号低半 0/高半 1。[lane_groups](../../../../workloads/moe/runs/hopper-stage-2/delivery-lanes/)、[wait6 事件](../../../../workloads/moe/runs/hopper-stage-2/delivery-micro/events-groups6.log)。 |
| 动态身份与缓冲轮换 | 同 PC/同源/不同目的的两份 copy 同时未完成，两个 UID 分别交付；四轮双槽位、跨 lane 读取及两个 CTA 的全输出精确，shared 对象分离，8 个动态 UID 全部核销。[identity](../../../../workloads/moe/runs/hopper-stage-2/delivery-micro/events-identity.log)、[reuse](../../../../workloads/moe/runs/hopper-stage-2/delivery-micro/events-reuse.log)。 |
| 请求、交付、等待和消费 | 每 UID 的真实返回次数等于生成请求数，交付恰好一次且不早于最后返回。mixed copy 发射 5074、交付/等待释放 5326、消费 5327；全零 copy 发射 5068、交付 5071、消费 5072。[只读 GDB](../../../../workloads/moe/tests/hopper/stage-2/observe.gdb.py)。 |
| 延迟和重叠 | 仅 L2 ROP latency 237→1237，mixed copy 交付/消费 5326/5327→6326/6327；请求确实命中 L2，未用 DRAM 参数解释 L2 命中。copy 未完成时本 warp 及另一 warp 的独立整数工作都推进。单/双缓冲分别 10285/9512 周期，增延迟后 15285/12381；读取/计算期间下一块已经在途。[敏感性运行](../../../../workloads/moe/runs/hopper-stage-2/delivery-latency1237/)。这些总周期含 launch，非 GPU 精度对照。 |
| 相关回归 | 阶段 0 八项符合预期；阶段 1 numeric、collectives、bits_loads、两种 warp barrier、旧 shuffle 全部通过，覆盖普通 global load/store 与 shared 读取。[stage0](../../../../workloads/moe/runs/hopper-stage-2/stage0-regression/run.json)、[stage1](../../../../workloads/moe/runs/hopper-stage-2/stage1-delivery/run.json)。 |

早期开发日志保留：第一轮测试文件缺少 cuobjdump 所需的开头注释；第二轮使用了旧 parser 不接受的 literal vector store，改为寄存器初始化；另一次 overlap 的 CPU 参考将第二 warp 的 shared lane 坐标写错，改为 lane%32。首次 Qwen 入口将两输入对调，8192项中8024项失败；核对 CTA 的 token/expert 坐标后，修正为 param0=FP32权重、param1=BF16输入。旧输出 `(0,1)=26` 恰为参考 `(1,0)`，而正确 `(0,1)=33`；超过原错误分配的64行后还读到了其它缓冲区，故出现大数。上述是测试修正，未改冻结 PTX 或放宽预期。`stage1-regression` 曾传错入口名 bits/loads，最终有效回归为 `stage1-delivery`，不把入口查找失败记作模拟器故障。

## 真实 gating、数值与成本

三指针绑定重新核对原 PTX 地址计算，并用前序转换的输出指针和长度交叉确认：[input-binding.json](../../../../workloads/moe/runs/hopper-stage-2/input-binding.json)。Qwen 模块70转换131072个元素（权重），输出接模块71的param0；Mixtral 模块0/1分别转换524288/32768个元素（输入/权重），输出分别接模块2的param0/1。

| 模块 | param0 | param1 | param2 | 原 launch |
| --- | --- | --- | --- | --- |
| 71 / Qwen64-128 | FP32 W[64,2048]，行步长8192B | BF16 X[128,2048]，行步长4096B | FP32 Y[128,64]，行步长256B | grid=8，block=128，dynamic shared=24576B |
| 2 / Mixtral-128 | FP32 X[128,4096]，行步长16384B | FP32 W[8,4096]，行步长16384B | FP32 Y[128,8]，行步长32B | grid=4，block=64，dynamic shared=24576B |

host 构造 `X[t,k]=((3*t+k)%7)-3`、`W[e,k]=((5*e+2*k)%11)-5`，由 host 转换 Qwen 的 BF16 token 输入；CPU 独立整数累加得到 `Y[t,e]`。每个结果都要求精确相等并保留完整 actual/expected，输入并非全 1/全 0。原 launch 和参数见 [launches.json](../../../../workloads/moe/runs/hopper-stage-2/real-release/launches.json)。两例全矩阵精确相等，最大绝对误差0；完整输出、预期和 `/usr/bin/time -v` 在 [real-release](../../../../workloads/moe/runs/hopper-stage-2/real-release/)，实际库、原PTX与两份forward的散列见 [run.json](../../../../workloads/moe/runs/hopper-stage-2/real-release/run.json)。

| 自测程序 | 精确元素 | 设备周期（含launch） | 初始化/准备 | launch至同步 | 比较循环 | 进程墙钟 / 峰值RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen64/128 模块71 | 8192 | 1566772 | 0.667s | 230.991s | 0.000203s | 231.69s / 401.2MiB |
| Mixtral/128 模块2 | 1024 | 1080167 | 0.656s | 130.067s | 0.000036s | 130.75s / 290.5MiB |

设备流量单列：两例生成的 global read 请求均为98304；L2 global read sector 分别97152/73728，global write sector为1024/128，DRAM read/write均0。原始请求、L2统计和DRAM不是同一层口径，不能强行要求计数相等；这里 host 上传预填L2，小于容量的输入均为热数据。两例原PTX的循环、重复搬运和必要零填均保留。

上述成本只适用于两个128-token gating 的当前配置。尚无4096/8192 gating或专家WGMMA成本样本，不将这两个数用于承诺九例完整前向耗时。

测试沿阶段 0/1 使用 H100 配置副本，只改 SM 132→1、chiplet partition 2→1，频率/容量/时序不变。单模块人工输入和 host 上传预填 L2 的前序不等于真实前向。现有 ncu 同模块时间为 Qwen 85120ns、Mixtral 100928ns；本轮摘录在 [hardware-reference.json](../../../../workloads/moe/runs/hopper-stage-2/hardware-reference.json)，[硬件计数](../../../../workloads/moe/evidence/hardware-counters.json)包含各自流量。当前不计算可比误差、不声称约20%已达标；完整缓存前序和实际132SM的硬件校准留待集成。

## 版本、复跑与交接

本轮未提交。最终构建与微测基于内层上述提交加 stage2 工作区差异；debug、CUDA12.8 / ptxas12.8.61、GCC13.4.0、clang-format16.0.6。[构建日志](../../../../workloads/moe/runs/hopper-stage-2/development/build-final.log)，[被测差异](../../../../workloads/moe/runs/hopper-stage-2/delivery-micro/inner.diff)。实际库由 runner 的 ldd 和 performance simulation 日志确认：`lib/gcc-13.4.0/cuda-12080/debug/libcudart.so.12`，`PTX_SIM_MODE_FUNC=0`；库SHA256 `2d3e477a2f7a3946d3b58d4c52bac9171b086e26c6e9595b5dfea982aab04a81`。

真实矩阵最终使用相同源码的 release（`-O3`，保留 assert），先复跑全部12类微测：数值和设备周期均与 debug 精确相同，[release-micro](../../../../workloads/moe/runs/hopper-stage-2/release-micro/run.json)。release 库 SHA256 `772a214ba6736227e45fca9a491343bbd638b5577fa0a63cdcdb8a08457079a3`，[构建日志](../../../../workloads/moe/runs/hopper-stage-2/development/build-release.log)。

旧 `real/` 是 debug 开发记录，其中 Qwen 的测试输入绑定错误，不作数值通过证据。它启动后仅将数字类型标识换为具名 enum 并格式化；该目录逐测试构建身份另存 [per-test-versions.json](../../../../workloads/moe/runs/hopper-stage-2/real/per-test-versions.json)。最终有效输出统一使用 `real-release/` 的版本/配置记录。源码与测试、库散列见 [versions.json](../../../../workloads/moe/runs/hopper-stage-2/versions.json)，内层最终 diff SHA256 `65fa477c4ac3d3028f87c45ec0969c3fadc70c64136fe309f0429edfa5b20c7f`。

验证 agent 从 cp.async 微测和事件断言开始，再运行两个真实 gating；新目录保留原证据：

```bash
cd /home/wangjz/in-switch/accel-sim
source workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-2/run.py \
  --run-dir workloads/moe/runs/hopper-stage-2/recheck-micro
# 长矩阵推荐 release；保持同一源码和配置，先重跑必要微测。
source gpu-simulator/gpgpu-sim/setup_environment release
unset PTX_SIM_USE_PTX_FILE PTX_SIM_KERNELFILE CUOBJDUMP_SIM_FILE
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-2/run.py \
  --run-dir workloads/moe/runs/hopper-stage-2/recheck-release-micro
python3 workloads/moe/tests/hopper/stage-2/real.py \
  --run-dir workloads/moe/runs/hopper-stage-2/recheck-real
```

敏感性入口为 `run.py --l2-latency 1237 --test copy_mix,zero,overlap,single,double`。在生成目录运行 `/usr/bin/gdb -q -batch -x <绝对路径>/observe.gdb.py --args ./micro copy_mix 32 1` 保存只读事件；reuse 的 grid=2，overlap 的 block=64，其余block=32/grid=1。准确运行命令、库、配置散列和源码差异由每个 runner 的 run.json 保存。大型日志/数据不随文档提交。

暂无可运行的小 SASS LDGSTS/DEPBAR trace，因此没有 SASS 运行回归证据；默认 PTX 标记为 NONE，SASS 语义分支保持原实现。按计划暂缓 `.ca`、其它目的长度、部分源1～15B、未用 cache hint、并发改源一致性和精细异步 shared 写 bank 模型。模块61/99的wait2/4由微测覆盖，真实调用留后续九例。独立验收由验证 agent 填写，本记录不替其改结论。

阶段2必要实现及自测已完成，停止扩测。独立验收从单组copy、组边界及动态身份开始，再复跑两个gating；暂未发现阻塞这些消费者的已知问题。

## 提交归档

内层实现已提交为 `ed851bb7210ec2b207188c45fdc3d5723afdaa23`，与独立验收保存的 12 文件差异一致。外层测试、实现及验收记录由配套 `stage2` 标签定位；两个仓库的标签说明均记录双方完整提交号。上文“未提交”和“独立验收尚未运行”的历史描述按当时状态保留，最终结论以[独立验收记录](../../validation/stage-2/README.md)为准。大型运行证据保留在本地 runs 目录，不纳入提交。
