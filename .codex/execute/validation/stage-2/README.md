# 阶段 2 实际验收：普通 cp.async

状态：**通过**（2026-11-08 独立验收，限阶段 2 范围）。普通异步拷贝的数值、逐线程分组、真实完成顺序与计算重叠符合要求；Qwen64/128 和 Mixtral/128 的原 gating 全矩阵精确正确，受影响回归通过。未发现需要执行 agent 修复的本阶段阻塞，已具备推进后续能力的条件。

完整九例、专家 GEMM、SASS trace 运行和 H100 约 20% 性能精度未验收。本轮在新目录完成 **246 条结果/事件断言**，详见[统一汇总](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/summary.json)；不自行启动下一阶段。

[验收要求](../../../validation/stage-2.md) · [实现方案](../../../plans/stage-2.md) · [实现记录](../../implementation/stage-2/README.md) · [统一证据](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/)

## 范围与被验版本

本轮独立阅读实现、测试和冻结 PTX，并在新目录重新运行。复用阶段 0/1 的前置能力和现有构建：交付的源码差异、测试及两个库散列逐项相符，修改源码均早于 release 库生成时间。未修改模拟器、模型参数、生产配置、冻结程序、输入规则或容差，未提交，也未向其它聊天发消息。Serena 工具在本会话不可用，使用本地源码及只读 GDB 检查。

| 对象 | 实际版本或运行身份 |
| --- | --- |
| 外层仓库 | 基准 `d7056c242e48cfbf78e9283078a674a79d2d6fc8`；保留执行 agent 原有实现记录与九份阶段 2 测试差异。本轮只更新本记录、增加 runs 中独立证据 |
| 内层仓库 | 基准 `b25fd0a18cb01a0c112042295007a0ac2c3772ad` 加 12 文件未提交差异；[被验 diff](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/inner.diff)，SHA256 `65fa477c4ac3d3028f87c45ec0969c3fadc70c64136fe309f0429edfa5b20c7f` |
| 构建 | 复用 CUDA 12.8 / ptxas V12.8.61、GCC 13.4.0 的 debug/release；release 开启 O3、保留 assert。两种构建均重跑全部 12 类微测，数值及设备周期相同；本轮未机械重建 |
| debug 库 | `lib/gcc-13.4.0/cuda-12080/debug/libcudart.so.12`；SHA256 `2d3e477a2f7a3946d3b58d4c52bac9171b086e26c6e9595b5dfea982aab04a81`，用于事件和回归 |
| release 库 | `lib/gcc-13.4.0/cuda-12080/release/libcudart.so.12`；SHA256 `772a214ba6736227e45fca9a491343bbd638b5577fa0a63cdcdb8a08457079a3`，用于真实 gating |
| 模式与配置 | ldd 与运行日志交叉确认模拟库及 `PTX_SIM_MODE_FUNC=0`、performance simulation；H100 副本仅 SM 132→1、chiplet partition 2→1，时钟 1980MHz。普通配置 SHA256 `e3d91789335d64379303f9233ba43cfa6c317e019115b5605c43f77672b27bbf` |

[版本核对](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/version-check.json)、[实现交付版本清单](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/versions.json)、[本轮工具链与全部测试散列](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/acceptance-versions.json)。每个 runner 的 run.json 保存准确命令、ldd、程序/库/配置散列；真实运行另保存原 PTX 和 forward.bin 散列。前置记录：[阶段 0](../stage-0/README.md)、[阶段 1](../stage-1/README.md)。

## 数值、分组和动态身份

复用[阶段 2 测试源码](../../../../workloads/moe/tests/hopper/stage-2/)，独立核查预期：完整源按组号、CTA 和 lane 编码；零填为 0；关闭谓词保留 a5；循环输出逐轮、逐槽比较。没有使用模拟器映射函数生成参考。

| 必要检查 | 本轮独立结果 |
| --- | --- |
| `cp.async.cg.shared.global` 16B、src_size=0/16、predicate | 16 lane 读取、8 lane 零填、8 lane 保留 a5；所有输出和未写区域逐字节一致。混合 copy 为 8 个 32B sector；全零 copy 无 global/L2/DRAM 读取，仍写零并完成等待 |
| commit、wait0/2/4/6 | 组件测试人为先完成新组，仍等待旧组 0/1；也验证只完成旧组即可放行、未提交组不进入 wait0、空组推进序号。组件与真实 PTX 分开记录 |
| PTX 组边界 | 四种 N 均先比较保证完成的前两组，再 wait0 比较全部组。高半 lane 提前提交空组后，首次 wait N 的截止为低半 2 / 高半 3，最终排空全部组 |
| 每线程组号 | `lane_groups` 汇合后在同 PC copy：低半 group=0、高半 group=1，全部 32 lane 的 16B 结果精确正确 |
| 同 PC/同源并发 | 两个不同 UID 在首次返回前都已发起，源相同、目的相差 512B，分别交付一次；不靠 PC 或 lane0 地址匹配 |
| 两 CTA 与缓冲复用 | 四轮双槽、跨 lane 读取及两 CTA 全输出精确；8 个 UID 全部核销，两个 shared 对象独立，相同偏移没有串写 |

数值证据：[release 12 类微测](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/micro/run.json)、[debug 12 类微测](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/debug/run.json)、[分组组件](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/debug/groups_component.log)。各目录保留完整 .actual/.expected，不只检查 PASS 字样。

源码接线与运行一致：shared 地址、global 地址和 src_size 都是读取依赖，没有普通寄存器目的；copy 动态载荷捕获发起 CTA 的 shared。发起时只暂存数据，LSU 最后返回时先写 shared、再完成各 lane 的组；未完成 copy 阻止 warp/CTA 槽位释放。组状态保留单调提交序号，完成项删除后不会改变 wait 截止。

## 完成顺序、真实访存与重叠

重新使用[只读观察器](../../../../workloads/moe/tests/hopper/stage-2/observe.gdb.py)采集 15 份事件日志：逐 UID 核对请求数与返回次数、恰好一次交付、组号、等待截止和消费者时间。所有观察只读取状态，不调用被测函数或强制放行。

| 场景 | 本轮观察的设备周期/行为 |
| --- | --- |
| 混合读取/零填 | 发起 5074 → 最后返回与 shared 交付 5326 → wait 放行 5326 → shared 消费 5327；8 次真实请求返回全部早于或等于交付 |
| 纯零填 | 发起 5068 → 正常 LSU 无请求路径交付 5071 → 消费 5072，无需虚构返回事件 |
| L2 敏感性 | 只把 ROP latency 237→1237；交付/消费从 5326/5327 变为 6326/6327，准确增加 1000。测试为热 L2 数据，未用 DRAM 参数解释命中 |
| 冷数据补验 | 仅在运行目录把 `perf_sim_memcpy` 1→0，阻止 host 上传预填 L2；混合读取产生 8 次 DRAM read，交付/消费为 5523/5524。纯零仍无 global/DRAM read，输出均精确 |
| 非阻塞 | copy 尚未交付时，同 warp 的独立整数链及另一 warp 的计算均继续推进 |
| 合法单/双缓冲 | 相同四块数据和计算，基础配置为 10285/9512 周期；增延迟后为 15285/12381。双缓冲事件显示下一块在途时当前块正在被读取，保留原 bar.sync |

事件目录：[基础 debug](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/debug/)、[L2 增延迟](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/latency1237/)、[冷数据](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/cold/)。两份诊断配置的唯一参数差异分别保存在各目录 config.diff；未改生产配置。上述总周期含 launch，不作硬件精度对照。

沿用计划允许的模型近似：只读输入在发起时快照；整条 warp 的最后一个请求返回后统一写 shared；复用 L2/DRAM、请求合并和普通 shared 消费者的 bank/端口模型，未单独细化异步 shared 写端口或逐 lane 最早交付。

## 两个真实 gating 与运行成本

直接执行冻结 PTX，保留原 entry、参数字节、grid/block/shared，仅替换设备指针。Qwen 的 BF16 输入由 host 独立预处理；其它输入为 FP32。核对原 PTX 地址运算以及 forward 中前序转换输出，确认绑定如下：

| 模块 | param0 / param1 / param2 | 原 launch |
| --- | --- | --- |
| 71 / Qwen64-128 | FP32 W[64,2048]，stride=8192B / BF16 X[128,2048]，stride=4096B / FP32 Y[128,64]，stride=256B | grid=8，block=128，dynamic shared=24576B |
| 2 / Mixtral-128 | FP32 X[128,4096]，stride=16384B / FP32 W[8,4096]，stride=16384B / FP32 Y[128,8]，stride=32B | grid=4，block=64，dynamic shared=24576B |

输入为 `X[t,k]=((3*t+k)%7)-3`、`W[e,k]=((5*e+2*k)%11)-5`；CPU 整数点积产生全矩阵参考，不使用全 0/1 或校验和。最终另由本轮 verify.py 用 Python 独立重新计算全部参考，防止只重复采信原测试的 PASS。

两个完整矩阵均无错误、全部有限、最大绝对误差为 0；Python 重新计算的 9216 个参考也全部精确相等。准确参数、原始完整输出和 time -v 见[real](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/real/)，原始指针来源交叉证据见[绑定表](../../../../workloads/moe/runs/hopper-stage-2/input-binding.json)。

| 本轮真实运行 | 精确元素 | 设备周期（含 launch） | 初始化/准备 | launch→同步 | 比较循环 | 进程墙钟 / 峰值 RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen64/128 模块71 | 8192 | 1566772 | 0.682s | 245.289s | 0.000203s | 246.01s / 401.4MiB |
| Mixtral/128 模块2 | 1024 | 1080167 | 0.643s | 135.561s | 0.000034s | 136.23s / 290.5MiB |

设备流量：两例 global read 请求均为 98304；L2 global read sector 为 97152/73728，write sector 为 1024/128，DRAM read/write 均为 0。host 上传预填 L2，输入小于缓存容量；冷数据微测另行覆盖 DRAM 返回。有效数据、合并请求、L2 sector 和 DRAM 属于不同口径，未强行要求计数相等。

初始化含输入与 CPU 参考准备；launch→同步含加载/预解码与模拟；比较循环不含下载和写文件。普通运行未挂调试器，部分短微测与事件观察同时进行，墙钟仅作本机成本样本。这两个同尺寸、同配置 gating 的模拟墙钟约 4.1/2.3 分钟，不能据此承诺更大 gating 或完整九例的成本。

## 受影响回归、未覆盖与交接

本轮重跑阶段 0 八项（assert/trap 为预期非零失败）和阶段 1 numeric、collectives、bits_loads、两种 warp barrier、legacy，覆盖普通 global load/store、shared 消费、CTA/warp 屏障及旧 shuffle。阶段 1 完整输出仍与固定参考逐字节一致：[stage0](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/stage0/run.json)、[stage1](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/stage1/run.json)、[legacy](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/stage1-legacy/run.json)。首次回归误把旧 shuffle 入口写成不存在的 old_shuffle，启动失败证据保留；使用真实入口 legacy 后通过，不计为模拟器实现回归。

没有现成可运行的小 SASS LDGSTS/DEPBAR trace，本轮未作 SASS 运行验收。静态核对默认 PTX 标记为 NONE，SASS 原完成分支仍保留；PTX 共用访存的运行回归不能替代 trace 证据。

两个 gating 使用单 SM、人工输入和独立 host 上传的热 L2 前序，与真实前向的 132SM/缓存条件不可比。已核对 ncu 对应调用：Qwen 85120ns、Mixtral 100928ns；**不计算约 20% 误差，不声称硬件性能达标**。原参照与流量见[硬件摘录](../../../../workloads/moe/runs/hopper-stage-2/hardware-reference.json)，后续有完整前序再比较纯 kernel 起止时间。

未覆盖：模块61/99的真实 gating（wait2/4已有组件和 PTX 微测）；其余七个 gating、冻结输入/完整九例、专家 FC1/FC2、H200。`.ca`、其它目的长度、部分源1～15B、未用 hint、并发改源一致性和精细异步 shared 写 bank 费用沿计划暂缓，实际消费者出现或主要误差指向它们时重开。本机成本仅适用于本次两个 128-token gating，不能外推 WGMMA 或九例完整前向预算。

本轮补充了冷 L2/真实 DRAM 的运行目录测试、独立 Python 全矩阵参考以及逐 UID/组截止断言，未放宽原验收要求。汇总脚本初版把零访问时省略的 L2 分类统计当成缺失错误，已修正为该分类计 0；总请求数、真实 DRAM 和数值比较仍单独严格检查，不把它记为模拟器失败。

[verify.py](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/verify.py) 的 246 条断言全部通过，模拟器 diff 在验收前后保持一致。当前没有已复现的实现问题或待核实的阻塞疑点；按本阶段停止条件结束扩测。现在可用独立入口执行两个真实 gating，为后续集成提供普通 cp.async 能力；下一阶段及硬件校准由用户另行指定。

复跑命令见[commands.sh](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/commands.sh)，关键事件的 cwd/完整命令见[event-commands.json](../../../../workloads/moe/runs/hopper-stage-2/independent-20261108/event-commands.json)。大型证据留 runs，本阶段实际结论只维护本文，不复制到验收方案。

## 提交归档

内层交付提交 `ed851bb7210ec2b207188c45fdc3d5723afdaa23`，提交内容与本轮被验 diff 完全一致；外层阶段 2 测试和两份实际记录由配套 `stage2` 标签定位，两个仓库的标签说明保存双方完整提交号。上文工作区/未提交状态描述的是验收发生时的版本。归档仅更新记录，没有修改被验实现、测试或预期，无需重跑模拟。原始运行证据保留本地 runs 目录。
