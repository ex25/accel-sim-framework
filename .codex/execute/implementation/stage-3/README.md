# 阶段 3 实际实现：cluster 与 shared 身份

状态：**已完成（实现与自测）**。两个 CTA 已能通过 ExC 以 1×2×1 启动、成组获得资源、分别拥有 shared，并按实际到达数执行重复 cluster 屏障。64 位地址布局覆盖 H100 的 132 个 SM 和 215040B 动态 shared。独立验收已通过，范围和证据见[验收记录](../../validation/stage-3/README.md)。

[实现方案](../../../plans/stage-3.md) · [验收要求](../../../validation/stage-3.md) · [独立验收记录](../../validation/stage-3/README.md)

## 当前范围与前置

本轮只交付阶段 3。阶段 0～2 已有代码与验收：外层基准 `5806f85`，内层基准 `ed851bb7`，开始时两个工作区干净。实现与验收时尚未提交，归档版本见文末；未修改冻结 PTX、输入、容差或生产 H100 配置。

```text
ExC 的本次 cluster 属性
  → kernel 按 cluster 枚举 CTA → 同时确认两份 SM 资源 → 各自初始化
  → rank 0/shared A、rank 1/shared B
       mapa(offset, rank) → 本 kernel / 本 cluster 的目标 shared 和 offset
       arrive → 通知队列 → 到达数满足 768 → 本轮 wait 可继续
```

现在具备阶段 4 远端 mbarrier 的目标解析入口。尚未执行远端 PTX load/store、mbarrier、TMA 或专家计算；它们不能由 mapa 编码和组件检查代替。

## 实际改动与计划对应

下表源码属于独立内层仓库。

| 对应 plan | 文件和入口 | 改变的行为及原因 |
| --- | --- | --- |
| [启动和身份](../../../plans/stage-3.md#launch) | [cuda_runtime_api.cc](../../../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) `cudaLaunchKernelExC`、`cudaLaunchKernelInternal`；[cuda_api_object.h](../../../../gpu-simulator/gpgpu-sim/libcuda/cuda_api_object.h) `kernel_config` | 从 CUDA 12.8 的 `cudaLaunchConfig_t` 读取 grid/block/shared/stream/cluster；复用普通 launch 的参数绑定和配置栈。属性只属于本次 kernel，接受 1×2×1 与普通 1×1×1，不支持的属性明确返回错误。 |
| 同上 | [abstract_hardware_model.h](../../../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) `cluster_id`、`cluster_ctaid`、`cluster_rank`、`increment_cta_id`；[ptx_sim.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_sim.cc) `get_builtin` | 由 CTA 坐标和 launch 维度生成身份，按同组 rank0→rank1 枚举；增加 `%cluster_ctaid.x/y/z`、`%cluster_ctarank`，补 `%smid` 的实际取值供定位。 |
| [地址](../../../plans/stage-3.md#addresses) | 同上地址常量；[instructions.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) `ld_exec`、`st_impl`、旧 `mma_ld/st_impl`；`isspacep_impl`；[cuda-sim.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) 分类 | global 堆起点 8GiB，shared 地址槽 256KiB×132，local 槽仍为每线程16KiB。load/store 的地址保持64位，不从低32位重建。`isspacep` 按无类型后缀指令分类，修复分支贯穿和内部 predicate 真值。 |
| 同上 | `cudaMallocArrayInternal` | 保留 `gpu_mallocarray` 返回的64位指针；不再从旧32位镜像字段重建。未扩展旧 texture/PTXPlus 地址通路。 |
| 同上 | [ptx.l](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l)、[opcodes.def](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/opcodes.def)、[opcodes.h](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/opcodes.h)；`mapa_impl`、`resolve_cluster_shared`、`resolve_local_shared` | 识别当前实际 mnemonic；mapa 生成 `rank<<24 | offset`，共用入口返回目标对象、CTA 和偏移。显式本地入口使用当前 rank，避免把 rank1 的普通偏移误作 rank0 地址。 |
| [资源与接纳](../../../plans/stage-3.md#scheduling) | [gpu-sim.cc](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) `issue_ptx_cluster`、`issue_block2core`；[shader.cc](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) 发射/退出 | 原容量判定先选两个不同 SM；两份都可用才连续调用原 CTA 初始化，不在两次调用之间运行调度器。每 SM 每周期仍至多接纳一个 CTA。失败不消耗 CTA 编号或占住半组。普通 CTA 沿原路径多驻留。 |
| 同上 | `shared_memory_per_cta`、`ptxinfo_smem`、`clear_ptxinfo`；[ptx_sim.h](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_sim.h) `ptx_version::set_target` | 保留 ptxas 的 system shared 来源；H100 每 CTA 补足1024B驱动保留量，已有 system 部分不重复计入。修正 `sm_90a` 数值目标解析，避免漏计此占用。寄存器仍来自 ptxas，不按函数名固定168。 |
| [屏障与生命周期](../../../plans/stage-3.md#barrier) | `kernel_info_t::ptx_cluster_state`、`create_cluster_shared`、`retire_cluster_cta`、`cluster_arrive`、`service_cluster_arrivals`、`cluster_wait_*`；`shader_core_ctx::issue_warp`、`shd_warp_t::waiting/reset/init` | kernel 持有每组两个 shared 对象，最后一个成员退出才释放。按 warp 的真实 active count 登记到达，事件交付后更新轮次计数；wait 只停本 warp。每 warp 保存轮次和所属 kernel，上一轮全部消费后删除该轮状态，其它组继续推进。 |

## 功能与时序怎样连接

复用 SASS/common 的 kernel、SM 容量判定、CTA 初始化、原 launch/TB 延迟、线程/寄存器/shared 限额和退出释放、warp 发射/暂停，以及普通 shared 的 bank/端口通路。SASS 已记录的身份和顺序不能代替 PTX 从 launch 生成身份、创建实际 shared 对象和登记到达。

与方案的等价简化：资源选择和两个 CTA 初始化在同一模拟线程调用内原子完成，无中间调度，因此不另加保留队列或第二套占用计数。当前两个成员统一分配到不同 SM，没有新增完整 GPC 拓扑。普通小 CTA 的接纳限制不变。

通知采用每 cluster 一条队列：warp 到达时提交线程数，最早在 `issue_cycle + smem_latency` 生效，collector 每 core 周期接收一条 warp 通知。H100 配置 `smem_latency=29` 用作端点延迟近似；这不是实测 DSM/GPC 网络延迟，也不借用 L2/DRAM 返回路径伪造地址请求。只有计数实际交付且达到本轮768时 wait 才能释放；不是固定等29周期便继续。参数敏感性另用129验证。

`arrive.relaxed` 不暂停发射，不增加普通 shared 发布、TMA 排空或全局同步；当前消费者每个 CTA 的384个线程均一致参与。各 warp 可在另一 warp 尚未消费旧轮时进入自己的下一轮，旧轮计数不会覆盖新轮。语义依据：[CUDA 12.8.1 PTX barrier.cluster](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)。

## 构建与自测证据

debug/release 构建通过，6类两SM测试、132SM地址采样、资源/目标解析组件、29项前置微测回归均通过。保存的数值、事件和资源证据通过 **2233条断言**：[summary.json](../../../../workloads/moe/runs/hopper-stage-3/summary.json)，由 [summarize.py](../../../../workloads/moe/tests/hopper/stage-3/summarize.py) 生成。测试源码在 [stage-3](../../../../workloads/moe/tests/hopper/stage-3/)，沿用阶段0/1的内嵌 fatbinary 注册、H100 配置复制、库检查和 `/usr/bin/time -v`，没有增加通用测试平台。

- `identity`：grid2×2×1、cluster1×2×1、block32、动态shared4096；按人工坐标检查身份。
- `mapa`：动态shared8192，偏移4096；PTX 输出交给 `cluster_component`，检查两份真实对象的不同数据、local rank、kernel隔离和单成员退出后的保留。这是 PTX 输出加组件检查。
- `addresses`：两SM、grid3×2×1、block384、动态shared215040；所有线程检查偏移0/98304/215036、generic shared/local正反转换与数据、64位global指针；重复两次launch。
- `address_sample`：原132SM、grid66×2×1、block384、动态shared215040；每CTA的lane0做同样地址检查，其它线程正常退出。实际采集SM0/79/80/131，资源接纳仍按完整384线程计算；这是地址采样程序，不是专家性能样本。
- `cluster_sync`：两组同时驻留，每组768线程，三轮arrive/独立算术/wait；仅cluster0的rank1加有限依赖链，比较delay32/128及通知延迟29/129。连续两次launch并交替普通launch。
- `ordinary`、`plain_ex`：普通小CTA多驻留；ExC的1×1×1属性与无属性配置；每次direct launch前压入另一份配置，launch后精确核对它的grid/block/shared/stream仍能恢复。

`cluster_component` 还实际用 ptxas 编译冻结模块16，提取168寄存器，向同一 `max_cta` 入口传递384线程与216064B总shared，得到每SM一个CTA。没有启动含未实现指令的真实专家，不将元数据组件检查写成专家驻留验收。

### 实际结果与主要时序

| 自测 | 实际结果和证据 |
| --- | --- |
| 身份、mapa、配置隔离 | 六类两SM测试均数值正确，direct launch的嵌套配置恢复正确；普通→ExC→普通以及两次kernel不残留rank/计数。[debug运行身份与结果](../../../../workloads/moe/runs/hopper-stage-3/debug-two-sm/run.json)、[release](../../../../workloads/moe/runs/hopper-stage-3/release-two-sm/run.json)。两种构建各目标kernel的设备周期完全一致。 |
| 大shared与高SM | 两SM全线程边界读写正确；132SM实际采样覆盖全部SM，包括0/79/80/131，66组均在同周期成对接纳到不同SM。[132SM运行](../../../../workloads/moe/runs/hopper-stage-3/high-sm/run.json)、[放置/占用/退出事件](../../../../workloads/moe/runs/hopper-stage-3/events-high-sm.log)。实际采样内核24寄存器/线程、9216寄存器/CTA、384线程、216064B shared/CTA；不是专家168寄存器的运行证据。 |
| 专家资源和mapa目标 | ptxas对冻结模块16报告168寄存器，组件核对64512寄存器/CTA、0B静态+215040B动态+1024B驱动=216064B，`max_cta=1`；PTX mapa输出解析到正确对象和值，rank1的本地入口不误指rank0；一个成员退出时其shared仍保留。[组件结果](../../../../workloads/moe/runs/hopper-stage-3/debug-two-sm/cluster_component.log)、[真实模块ptxas输出](../../../../workloads/moe/runs/hopper-stage-3/debug-two-sm/expert-ptxas.log)。驱动保留来源是既有[ncu计数](../../../../workloads/moe/evidence/hardware-counters.json)。 |
| 三轮、两组、两次launch | 每组每轮24条warp通知、每条32线程，交付/释放各768线程；共12轮无提前释放、无旧轮覆盖。首次launch慢组各轮最早释放29425/33553/36162周期；快组首轮在29009已释放。[基准事件](../../../../workloads/moe/runs/hopper-stage-3/events-sync.log)。PTX clock证明早到方在对方完成前执行了arrive之后的独立算术。 |
| 延迟敏感性 | 只把晚到方依赖链32→128，慢组三轮最早释放延后到31290/38483/47039；快组首轮仍在28923完成。[delay128](../../../../workloads/moe/runs/hopper-stage-3/events-delay128.log)。将端点延迟29→129，每个通知都至少经过相应延迟且按collector排队，慢组首次释放30392；shared普通访问也使用这一参数，不能将总周期差全部解释为通知差。[transport129](../../../../workloads/moe/runs/hopper-stage-3/events-transport129.log)。 |
| 普通CTA与生命周期 | 普通小CTA实际每SM驻留4个：128线程、3584寄存器、4224B shared；两个kernel各自创建/退出所有shared对象，未混用。[普通驻留](../../../../workloads/moe/runs/hopper-stage-3/events-ordinary.log)。 |
| 前置回归 | 阶段0八项（含预期assert/trap失败）、阶段1九项、阶段2十二项全部通过，覆盖global/local/shared、参数指针、数值/集合操作和cp.async实际返回。[stage0](../../../../workloads/moe/runs/hopper-stage-3/regression-stage0/run.json)、[stage1](../../../../workloads/moe/runs/hopper-stage-3/regression-stage1/run.json)、[stage2](../../../../workloads/moe/runs/hopper-stage-3/regression-stage2/run.json)。SASS构造器的B+D与7/3/7占用隔离组件通过：[resource_component.log](../../../../workloads/moe/runs/hopper-stage-3/metadata/resource_component.log)；这不是trace运行。未无故重跑真实长gating或完整前向。 |

最终地址区间：静态global低于 `0xf5f00000`；local在 `[0xf5f00000,0x1fdf00000)`；shared在 `[0x1fdf00000,0x200000000)`；动态global从 `0x200000000` 开始。`STATIC_ALLOC_LIMIT=0xf5f00000`，未下溢；地址槽容量没有扩大每SM实际233472B资源上限。

### 本机成本与设备周期分列

以下是release库的最小程序实测，不外推九例。设备周期包含现有5000周期launch费用；主机进程还运行了前后普通launch。注册列为进程起点到注册完成，launch列为目标提交至同步返回，比较列为CPU检查循环。构建/其它独立检查曾并行运行，墙钟是当次观测，不是空闲机器基准。

| 程序 | 目标设备周期 | 注册墙钟 | 目标launch/同步墙钟 | 比较循环 | 进程总墙钟 / 峰值RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 身份，4CTA | 6807 | 0.295s | 0.844s | 0.028s | 2.46s / 274.7MiB |
| 全线程边界，6CTA、两次launch | 31953 / 31953 | 0.295s | 6.300 / 6.441s | 0.0011 / 0.0011s | 15.62s / 278.4MiB |
| 三轮屏障，4CTA、两次launch | 30691 / 30517 | 0.298s | 6.788 / 6.735s | 0.0010 / 0.0019s | 16.39s / 281.3MiB |
| 原132SM地址采样，132CTA | 6113 | 0.678s | 6.963s | 0.012s | 11.76s / 1085.5MiB |

所有程序的debug/release成本、设备周期及初始化时间在汇总中逐项列出。这里没有真实专家计算，不能据此更新九例成本预算或声称H100时间误差达到约20%。

事件采集中，GDB 10.2的delay128进程在host已打印PASS后报告 `Cannot find user-level thread for LWP`，调试器退出非零。该记录的两次kernel、12轮到达/交付/释放、所有shared退出及数值输出完整，已由汇总逐项断言；明确保留工具退出问题，不将它改成程序失败或伪造退出码0。其它四份事件运行正常退出。

## 当前限制与交接

本阶段只覆盖冻结消费者的1×2×1和全线程一致参与屏障；未扩任意cluster形状、提前退出线程参与的屏障、远端load/store、数据/通知仍在途时的TMA生命周期。阶段4/5应复用目标解析和所有权，在实际返回路径补后两项。通知延迟及collector吞吐为上述近似，需在真实专家可运行后结合主要耗时校准；本轮不宣称约20%性能目标或九例运行预算。

旧 WMMA 地址中间值随统一布局改为64位，但本阶段未验收矩阵指令。CUDA array 分配保留完整指针；旧 texture/PTXPlus 通路不属于冻结消费者。无可运行 Hopper SASS trace，需列出实际共用构造器组件和普通PTX回归，不能称为SASS运行验收。

开发期失败和临时产物保留在 [development](../../../../workloads/moe/runs/hopper-stage-3/development/)：测试名与 `barriers` 关键字冲突；`isspacep` 解码/执行缺口；load/store地址截断；GDB首次命中PLT无参数；`sm_90a` 数值目标解析遗漏。前三者和目标字段均已在所属源码修正；旧的未计1024B结果不作最终资源证据。提前运行的全线程132SM地址测试在发现资源字段问题后终止，最终改用上述合法lane0采样以减少无关输出量。

## 版本、命令与附件

实现与自测时未提交；外层基准 `5806f857aef604e6474899181c4c77b6aca6f51c`，内层基准 `ed851bb7210ec2b207188c45fdc3d5723afdaa23` 加15个源码文件的本轮差异。内层被测diff SHA256为 `205cdab34c1e6384d9f80620ef1daf6958f251a79d381c1eb33cf369b831662c`；debug、release、132SM和cp.async回归的被测diff均与最终源码相同。外层改动为六份阶段3测试、本记录，以及阶段0的一条前置修复索引；新测试原受 `tests/` 忽略规则影响，已仅为这六份文件设置intent-to-add，使工作区差异可审阅，未创建提交。

工具链CUDA12.8/ptxas12.8.61、GCC13.4.0、clang-format16.0.6。debug与release均构建通过，release为O3且保留assert：[debug构建](../../../../workloads/moe/runs/hopper-stage-3/development/build-final-debug.log)、[release构建](../../../../workloads/moe/runs/hopper-stage-3/development/build-final-release.log)。实际库、源文件、程序及配置散列见[versions.json](../../../../workloads/moe/runs/hopper-stage-3/metadata/versions.json)；[inner.diff](../../../../workloads/moe/runs/hopper-stage-3/metadata/inner.diff)保存内层被测改动。每份run.json保存ldd和完整命令，确认加载模拟器libcudart和`PTX_SIM_MODE_FUNC=0`/performance simulation。

两SM配置仅将原H100的132SM改为2，chiplet partition仍为2；原132SM采样不改生产配置。通知敏感性副本另改smem latency29→129。相关频率、每SM容量和其它时序均保持H100值；原一SM前置runner仍按其既有方式复制配置。Serena工具在本会话不可用，使用本地检索与只读GDB。

从项目根目录复跑（使用新目录保存证据）：

```bash
source workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-3/run.py \
  --run-dir workloads/moe/runs/hopper-stage-3/recheck/debug-two-sm
source gpu-simulator/gpgpu-sim/setup_environment release
unset PTX_SIM_USE_PTX_FILE PTX_SIM_KERNELFILE CUOBJDUMP_SIM_FILE
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-3/run.py \
  --run-dir workloads/moe/runs/hopper-stage-3/recheck/release-two-sm
python3 workloads/moe/tests/hopper/stage-3/run.py --sms 132 --test address_sample \
  --run-dir workloads/moe/runs/hopper-stage-3/recheck/high-sm
```

GDB五份实际cwd、参数和日志位置见[event-commands.json](../../../../workloads/moe/runs/hopper-stage-3/event-commands.json)。事件目录从两SM运行复制micro/config/icnt；高SM目录使用原132SM配置，transport129只修改上述参数。观察器使用debug库，读取状态，不修改模拟器。汇总入口：

```bash
python3 workloads/moe/tests/hopper/stage-3/summarize.py \
  workloads/moe/runs/hopper-stage-3
```

运行产物保留runs，不纳入提交。独立验证的新根目录若要使用汇总脚本，保留相同的子目录和事件文件名；不覆盖当前证据。

验证 agent 从 `run.py` 的两SM测试和 `cluster_component` 开始，再跑132SM采样和事件断言；不要直接启动完整专家或九例前向。必要自测已通过，本轮停止，不实施阶段4。

## 提交归档

内层交付提交 `a4c359759c097f4362200e3bbadf3ff63a373a3b`，内容与独立验收的15文件差异完全一致（diff SHA256 `205cdab34c1e6384d9f80620ef1daf6958f251a79d381c1eb33cf369b831662c`）。外层配套提交保存六份测试、阶段0前置修复索引、本实现记录和独立验收记录；两个仓库的 `stage3` 标签说明保存双方完整提交号。

独立验收已通过，详见[阶段3验收记录](../../validation/stage-3/README.md)。上文未提交状态及基准描述保留当时的运行身份；归档没有改动被验实现、测试、预期或配置，不重复运行模拟。大型运行证据保留本地 runs，未纳入提交。
