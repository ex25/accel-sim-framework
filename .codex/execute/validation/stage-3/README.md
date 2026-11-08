# 阶段 3 实际验收：cluster 与 shared 身份

状态：**通过**（2026-11-08 独立验收，限阶段 3 范围）。两个 CTA 能共同获得资源，分别拥有 shared，并按实际到达的768个线程完成重复屏障；大 shared 和132个SM的地址检查通过，未发现本阶段实现阻塞。现在具备推进阶段4/5所需的启动、身份、资源和目标解析基础，本轮不自行启动下一阶段。

本轮完成 **2326条结果/事件断言**，另用独立脚本精确核对 **404824项输出**、完成53790条检查，29类前置回归全部通过：[统一汇总](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/summary.json)、[独立检查](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/independent-checks.json)。通过结论不包含远端PTX访存、mbarrier、TMA、专家计算或H100约20%的性能精度。

[验收要求](../../../validation/stage-3.md) · [实现方案](../../../plans/stage-3.md) · [实现记录](../../implementation/stage-3/README.md) · [本轮证据](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/)

## 被验版本与运行身份

独立阅读实现、测试及冻结模块 16 的实际指令形式，在新目录重跑。复用已有构建：源码、测试、debug/release 库及被测 diff 的 24 项散列核对全部相符，修改源码均早于库生成时间。本轮不改模拟器、模型参数、生产配置、冻结程序或容差；保留原有未提交修改，不提交，不向其它聊天发消息。Serena 工具未在本会话提供，使用本地检索与只读 GDB。

| 对象 | 实际版本与证据 |
| --- | --- |
| 外层仓库 | 基准 `5806f857aef604e6474899181c4c77b6aca6f51c`；原有两份实现记录和六份阶段 3 测试差异保留。本轮更新本记录，独立辅助脚本和运行产物留 runs |
| 内层仓库 | 基准 `ed851bb7210ec2b207188c45fdc3d5723afdaa23` 加 15 文件未提交差异；[inner.diff](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/inner.diff)，SHA256 `205cdab34c1e6384d9f80620ef1daf6958f251a79d381c1eb33cf369b831662c` |
| 构建及工具链 | CUDA 12.8 / ptxas V12.8.61、GCC 13.4.0；复用交付的 debug 和 O3 release（保留 assert），本轮未机械重建 |
| debug 库 | `lib/gcc-13.4.0/cuda-12080/debug/libcudart.so.12`；SHA256 `d979d017c398165c0be7cb265715ae9e851c1bee69649661f38484cde20d9410` |
| release 库 | `lib/gcc-13.4.0/cuda-12080/release/libcudart.so.12`；SHA256 `ca26dcd2c7b47e11daf2b1f70e1efa51f5fbd877ddd05cc03c253921f75864f6` |
| 实际模式 | 每份 run.json 保存 ldd、命令、库/程序/PTX/配置散列；日志确认 performance simulation，`PTX_SIM_MODE_FUNC=0`，未启用外部 PTX 替换 |
| 配置 | H100 两 SM 副本只改 SM 数 132→2，SHA256 `239c7190fc3fd253913ee1a4d546a4b68b22744d38e279225b77b6abf082b354`；原 132SM 配置 `4215119f641dfdf9b968db579129ea23c7bf653ba8d28f90f75ce03dcca4a0ed`；均为 1980MHz、每 SM shared 233472B |

[版本核对](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/version-check.json)、[构建与源码时间核对](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/build-source-check.json)、[交付版本清单副本](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/versions.json)、[本轮测试散列](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/test-versions.json)。前置记录：[阶段 0](../stage-0/README.md)、[阶段 1](../stage-1/README.md)、[阶段 2](../stage-2/README.md)；本轮另跑受影响回归，不把旧结论当作当前源码的运行证据。

## 数值、地址和资源

复用[阶段 3 测试源码](../../../../workloads/moe/tests/hopper/stage-3/)，独立核对预期：坐标/rank 来自固定 launch；shared 标签为 `100*x+10*rank+1`，local 为线程号；边界偏移固定为 0/98304/215036；算术链用 host 无符号整数运算。补充 Python 逐项重算已保存的完整相关输出，不调用模拟器映射函数生成参考。

| 验收项 | 独立结果与边界 |
| --- | --- |
| ExC、四种特殊寄存器 | grid=2×2×1、cluster=1×2×1、block=32、动态 shared=4096；四 CTA 的组内坐标/rank 精确正确。普通→ExC→普通、显式 1×1×1 与无属性 ExC 均通过；每次 direct launch 前压入的 grid/block/shared/stream 完整恢复 |
| mapa | 实际 PTX 输出分别为 `0x00001000`、`0x01001000`；组件将其解析到本 kernel、本 cluster 的两份不同 shared 对象，读取不同标签；rank1 的本地入口正确。只证明 PTX 编码加目标解析组件，未运行远端 PTX load/store |
| 大 shared、generic/local/global | 两 SM、6CTA、384线程、215040B 动态 shared，全部线程的首/中/末偏移写读、generic 正反转换及分类、真实 local 写读、64位 global 参数指针通过；两次 launch 均正确 |
| 高 SM | 原 132SM、grid66×2×1，实际覆盖全部 SM，包括 0/79/80/131；每 CTA 的 lane0 检查上述数据和地址，其余线程正常退出，资源仍按 384 线程计算。不是只打印转换公式 |
| 成组接纳 | 两 SM 多组大 shared，以及原132SM的66组，均在同周期成对接纳到不同 SM；每 SM 384线程、9216寄存器、216064B shared，未超容量。小程序实际为24寄存器/线程，不标为专家168 |
| 专家资源组件 | ptxas 实际编译冻结模块16，报告168寄存器；同一容量计算入口核对 `384×168=64512`，PTX静态0 + 动态215040 + 驱动1024 =216064B，驻留上限1。未启动尚缺指令支持的真实专家 |
| 生命周期、普通 CTA | 组件验证一个成员退出后其 shared 仍保留且下一 kernel 对象隔离；实际两次启动均创建/退出全部成员。普通小 CTA 实际每 SM 驻留4个，128线程、3584寄存器、4224B shared，没有全局限制为每SM一个CTA |

数值与完整原始输出：[debug](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/debug-two-sm/run.json)、[release](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/release-two-sm/run.json)、[132SM](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/high-sm/run.json)。[mapa/专家资源组件](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/debug-two-sm/cluster_component.log)、[真实模块 ptxas](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/debug-two-sm/expert-ptxas.log)。1024B 驱动保留量来源仍是冻结的 [ncu 资源记录](../../../../workloads/moe/evidence/hardware-counters.json)，不是伪造 PTX 静态数组。

最终区间：静态 global 低于 `0xf5f00000`；local 为 `[0xf5f00000,0x1fdf00000)`；shared 为 `[0x1fdf00000,0x200000000)`；动态 global 从 `0x200000000` 起。区间不重叠，`STATIC_ALLOC_LIMIT` 未下溢。256KiB 地址槽没有改变每 SM 的实际资源容量。

## 屏障、通信成本和资源释放

重新用只读 GDB 采集发出、交付、wait 放行及 CTA 创建/退出，不调用被测函数或强制放行。每次屏障测试为两组、每组768线程、三轮，并连续启动两次；每轮核对24条通知、每条32线程，交付和释放均为768，第一条放行不早于最后一条交付。普通算术能在 arrive 后、同伴完成前执行；其它 cluster 独立推进。

| 情形 | 本轮实际观察的设备周期 |
| --- | --- |
| 基础 delay32 | 首次 launch 慢组三轮最早放行为29425/33553/36162；快组首轮29009已完成 |
| 仅延长晚到方 delay32→128 | 慢组三轮变为31290/38483/47039；快组首轮28923已完成，未被全局阻塞 |
| 端点延迟29→129 | 每个通知均经过相应延迟，并按每周期一条排队交付；慢组首次放行30392。该参数也影响普通 shared，不能把总周期差全归于通知 |
| 两 SM、大 shared 排队 | 首次 launch 三组分别在11214/20198/29182成对接纳；前组两成员在20197/29181退出，下一组随后接纳。重复 launch 同样完成资源释放，无半组驻留或未归还槽位 |

事件：[基础](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/events-sync.log)、[晚到方](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/events-delay128.log)、[通知延迟](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/events-transport129.log)、[大shared排队](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/events-addresses.log)、[普通多驻留](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/events-ordinary.log)、[132SM](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/events-high-sm.log)。每轮及每 SM 的资源明细留日志，准确命令见 [event-commands.json](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/event-commands.json) 和[追加观察命令](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/address-event-command.json)。

源码检查确认 relaxed arrive 没有加入普通 shared 发布或异步排空；本测试用独立寄存器/global输出和事件验证参与，shared 读写另用 bar.sync，不以数据竞争推导 release 保证。语义对照 [CUDA 12.8.1 PTX barrier.cluster](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)。

通信模型以 H100 `smem_latency=29` 近似端点延迟、每 cluster 每周期交付一条 warp 通知；已验证非零成本和等待关系，未宣称精确模拟 GPC 网络。真实通信字节与主要耗时校准留阶段 5/7。

## 受影响回归和本机成本

阶段0八项、阶段1九项、阶段2十二项全部重跑通过，覆盖参数、普通 global/local/shared、旧 cvta/普通 CTA、warp协作、普通 cp.async 实际返回；assert/trap 按预期非零失败。共用资源组件再次得到7/3/7，保留 SASS 构造器的动态 shared 默认0及对象隔离。没有现成可运行的小 Hopper SASS trace，本轮仅有共用组件和 PTX 回归，不能称作 SASS 运行验收。

回归：[stage0](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/regression-stage0/run.json)、[stage1](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/regression-stage1/run.json)、[stage2](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/regression-stage2/run.json)、[资源组件](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/resource_component.log)。未为无相关改动重跑长 gating 或完整前向。

| 本轮普通运行 | 目标设备周期 | 注册墙钟 | launch→同步墙钟 | CPU比较 | 进程墙钟 / 峰值RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| release身份 | 6807 | 0.299s | 0.879s | 0.000183s | 2.50s / 274.4MiB |
| release大shared，两次 | 31953 / 31953 | 0.295s | 6.701 / 6.388s | 0.0111 / 0.0011s | 16.06s / 278.5MiB |
| release三轮屏障，两次 | 30691 / 30517 | 0.296s | 6.631 / 6.599s | 0.0013 / 0.0009s | 16.08s / 281.1MiB |
| 原132SM地址采样 | 6113 | 0.681s | 7.807s | 0.0111s | 12.72s / 1085.5MiB |

设备周期包含既有5000周期 launch 费用；注册含本机装载/初始化，launch→同步含准备和模拟，CPU比较不含下载。日志保存 HOST_TIME，独立 `/usr/bin/time -v` 保存进程总时间和 RSS。部分程序并行执行，墙钟只作为本机成本样本；这里没有真实专家计算，不外推九例预算，也不计算约20%的硬件时间误差。

## 问题、未覆盖与交接

全部必要检查及最终汇总通过；没有已复现的模拟器实现阻塞，也没有待核实的本阶段阻塞疑点。追加的 addresses 事件观察在程序两次 launch 均输出正确、全部对象退出、host打印 `PASS ... errors=0` 后，GDB 10.2 报 `Cannot find user-level thread for LWP`。原日志保留，检查完整事件而不将调试器退出信息冒充模拟器失败；普通 debug/release 的同用例均正常退出。若数值或退出事件不完整，此证据不能采用。

未覆盖：真实远端PTX load/store和mbarrier（阶段4须用真实远端副作用复验同一目标解析入口）、TMA在途的数据/通知与存储复用（阶段5）、WGMMA/旧WMMA运行、真实专家FC1/FC2、冻结九例、H200、SASS trace。未扩任意cluster形状、提前退出线程参与屏障、旧texture/PTXPlus和精确GPC拓扑；当前消费者出现或主要误差指向它们时再重开。

本轮补充了大 shared 多组资源事件与独立 Python 输出/事件检查，辅助脚本保留 runs；不改原测试预期、不放宽阶段要求。已满足阶段3的停止条件，不再扩测。阶段4/5可复用本轮验证的 kernel/cluster/shared 身份与共同接纳基础，下一步由用户指定；真实远端通知和数据返回仍需在各自阶段验收。

复跑见 [commands.sh](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/commands.sh)，使用新目录保存证据。汇总入口为 [summarize.py](../../../../workloads/moe/tests/hopper/stage-3/summarize.py) 和本轮 [verify.py](../../../../workloads/moe/runs/hopper-stage-3/independent-20261108/verify.py)；大型证据留 runs，实际结论只维护本文。

## 提交归档

内层交付提交 `a4c359759c097f4362200e3bbadf3ff63a373a3b`，提交内容与本轮被验 diff 完全一致。外层配套提交保存测试和实际记录，两个仓库的 `stage3` 标签说明保存双方完整提交号。上文基准/未提交差异描述的是验收发生时的版本；原始证据和散列保留，可对应到该提交。

本次只归档已通过的实现、测试与记录，没有修改功能、模型参数或数值预期，不重复构建或模拟。未推送；下一阶段由用户指定。
