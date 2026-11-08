# 阶段 0 实际验收：加载与启动

状态：**通过**（2026-11-08 独立验收）。已验证小型 PTX 程序能加载正确模块、传入完整参数，按本次启动配置和动态共享内存执行，得到正确输出与设备周期。未发现阶段 0 实现阻塞，可进入阶段 1；本轮未启动后续阶段。

结论限于阶段 0。冻结九例、专家 GEMM、cluster、TMA、WGMMA 未运行；SASS 有共用构造和资源公式的组件回归，未运行 trace。

[验收要求](../../../validation/stage-0.md) · [实现方案](../../../plans/stage-0.md) · [实现记录](../../implementation/stage-0/README.md)

## 被验版本与运行身份

实现报告仅用于找到入口；下列结果来自本轮独立构建、执行和观察。验证 agent 未修改模拟器、模型参数或生产配置。验收时使用工作区改动，随后提交并以两个仓库的 `stage0` 标签归档，标签说明记录双方提交号。

| 对象 | 版本与证据 |
| --- | --- |
| 外层仓库 | 交付由 stage0 标签定位；验收基准 b80607e5235e190930813353b8db0b1550b48cac 加实现/验收记录及测试改动 |
| 内层仓库 | 交付提交 62205297c2b0c7ba9ba1e275e6a9874f0c284a02；验收基准 4195dd4f229cbf03f4f38c49139ae94d01d23f03 加 15 文件的实现；[被验 diff](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/inner.diff) SHA256 为 3dd9fc91023fd9a74053c7a3f9e840075984cb5eda83e87b536428c046e27e1f，与提交内容一致 |
| 构建 | in-switch conda 环境，CUDA 12.8 / ptxas V12.8.61、GCC 13.4.0，debug，make -j8 成功；[构建日志](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/build-activated.log)、[工具链](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/toolchain.txt) |
| 实际加载库 | 内层 lib/gcc-13.4.0/cuda-12080/debug/libcudart.so.12；[运行时 calling init](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/loader.964651) 与 runner 的 ldd 交叉确认 |
| 模式/配置 | PTX_SIM_MODE_FUNC=0，performance simulation，PTXPlus=0，资源检查开启。H100 副本仅将 clusters 132→1、chiplet partitions 2→1；计数分支副本额外启用 concurrent_kernel_sm=1，容量、延迟、吞吐和时钟未改 |
| 测试版本 | [源码](../../../../workloads/moe/tests/hopper/stage-0/)、[编译/运行命令与八项状态](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/run.json)、[源码/程序/库/配置散列](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/final-versions.json) |

默认配置 SHA256：f1dd394881905ba1e3942fbd28e2ad51077d80974c8dc30ae6a8ec4f019e1f5d；计数分支配置：f9b492c9fc7ce90be1158aaf4a92b9a8bcf78e0e5e8f228f62b0a211459e0714。单 SM 微测不作为完整 H100 性能数据。

## 逐项结果：全部通过

| 检查 | 独立预期与实际结果 | 证据 |
| --- | --- | --- |
| 运行身份/工作量 | 首次 grid=2、block=32，线程编号为 0～63；四次全部线程的 grid/block、标记和数值正确。100 轮 v=3*v+1 用主机 unsigned 运算独立比较，计算区间均增长 | [launch](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/launch.log)、[work](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/work.log) |
| 完整目标 | A=sm_90、B=sm_90a，提取文件保留目标；实际 ptxas 命令分别含 --gpu-name=sm_90、--gpu-name=sm_90a，运行写出 11、22 | [命令/入口观察](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/observe-modules.log) |
| 混合参数 | (size,offset) 为 (4,0)、(128,64)、(8,192)、(8,200)，finalize 后符号地址为 0、64、192、200；head、完整 128B、tail 与输出指针正确 | [内部状态](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/observe-parameters.log) |
| 1344B 对象 | 大小 1344、对齐 64，逐字节比较全部通过；偏移 0、63、64、127、128、1128、1136、1240、1343 分别为 3、188、195、124、131、219、19、235、188 | [参数输出](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/parameters.log) |
| 两种启动 | 四次交替得到 64、192、32、128 个正确线程结果，标记 11、22、33、44。直接提交前后栈深 0→0；Pop 为 1→0，grid/block/shared/stream 四个输出均等于栈顶 | [栈/配置](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/observe-launch.log) |
| stream 顺序 | 同一非默认 stream 内 H2D→读取并计算→D2H→同步，32 项等于 (55+i)*3+7；额外编译器启动 shared=128、非空 stream，经 Pop、直接 API 与入队正确传递 | 同上及 launch 输出 |
| shared 数据 | 函数许可始终 65536；本次 D 为 32768、65536、32768。每轮 32 个 CTA 均由线程 0 写首尾和静态标记，bar.sync 后线程 1 读出 [101,202,77]，静态/动态区不覆盖 | [shared 输出](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/shared.log) |
| 驻留/缓存 | ptxas 基数 B=32，S=233472；每 CTA B+D 为 32800、65568、32800，floor(S/(B+D)) 和实际限制均为 7、3、7。自适应缓存按本次占用选 228KiB shared 档，三轮 L1 均 28KiB | [默认资源分支](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/observe-shared.log) |
| 占用/归还 | 共用计数分支记录 96 次占用、96 次归还，每次增减精确为 B+D；峰值 229600≤233472，最终 shared/CTA 计数均 0。函数基数始终 32，各启动对象保留各自 D | [实际计数分支](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/concurrent/observe-shared-final.log) |
| 失败声明/出口 | 含五参数 extern 声明的正常路径输出 123；实际 call 报 __assertfail at stage0.2.sm_90.ptx:448 并带 CTA/线程身份，trap 报同文件第 450 行。两项被测进程非零（runner 记录 1），无成功标记 | [normal](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/normal.log)、[assert](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/assert.log)、[trap](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/trap.log) |
| 两模块 | 独立 fatbinary 正常注册后 A→B→A 得到 [11,22,11]。host 标记找到 write_a、write_b、write_a；ptxas 与 function_info 一致：寄存器均 10，shared 分别 32/64，未串用 | [数值](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/modules.log)、[映射/资源](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/observe-modules.log)、[A ptxas](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/stage0.2.sm_90.ptxas)、[B ptxas](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/timed/stage0.3.sm_90a.ptxas) |
| 共用组件回归 | SASS 所用的 stream-ID 构造函数 D 默认 0。B=32768 时驻留 7；另一个 PTX 对象 D=32768 后驻留 3，再查原对象仍 7，基数未变 | [源码](../../../../workloads/moe/tests/hopper/stage-0/resource_component.cc)、[输出](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/resource-component.log) |

字节预期为 (7*i+3)%256，参数偏移来自验收固定表，没有调用模拟器布局函数生成预期；模块数值由 global/constant 的 5+6、9+13 独立得出。实际执行包含 ld.param、普通算术/分支、global/constant/shared 访存、bar.sync 和失败调用；组件检查未冒称 SASS 运行。

## 设备周期与本机成本

观察模型首次选择内核执行时保存的 start_cycle 和完成设备计数：首个普通内核 **5001→5533，差 532 周期**；100 轮依赖运算 **5001→5959，差 958 周期**。其余三次差值由 934/440/732 增至 1363/866/1158。微测只验证工作量增加后的趋势，不套 H100 的 20% 门槛。

普通日志首轮累计周期 5534 含启动等待等区间外开销，不能当作上述 532。完整起止事件、周期和主机时间戳在 [final-summary.json](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/final-summary.json)。调试器运行的墙钟不用于预算。

| 普通运行 | 墙钟 | 峰值 RSS/KiB | 最后累计周期 |
| --- | ---: | ---: | ---: |
| launch | 4.91s | 274720 | 28265 |
| work | 5.53s | 274596 | 29972 |
| modules | 3.04s | 273596 | 17070 |
| parameters | 2.57s | 273880 | 12064 |
| shared | 4.53s | 274184 | 24317 |
| normal | 1.31s | 272968 | 5303 |
| assert / trap | 1.20s / 1.21s | 273436 / 273764 | 预期失败，无完成统计 |

首个 launch 从测试 constructor 到 main 的装载/初始化 0.4414s，首个提交到同步返回 0.9073s，work 对应 1.0074s。launch 比较用 compare_begin/end 单独记录；参数等程序的 kernel_done→下一提交或 comparison_done 包含复制/比较，未冒称纯比较时间。各 .log/.time 保留时间戳及 time -v。峰值约 268MiB，小程序不能外推专家和九例预算。

## 测试完善、问题与未覆盖

本轮只完善验收测试，未放宽预期：原 stream 内核不读取 H2D 输入，改为读取 32 项计算再 D2H；用 shared=128 经编译器启动补查 Pop 的非零参数。原 start 在 main 内漏记静态注册初始化，增加优先执行的测试 constructor 时间戳和 launch 比较时间戳。新增只读 GDB 观察及资源组件，补齐偏移、栈、资源与设备周期证据；GDB 不调用被测函数、不修改被测状态，源码行号绑定本次 diff。

八项测试、五项调试观察、资源计数和组件均通过；[固定断言脚本](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/verify.py) 已通过。没有需要交执行 agent 修复的阶段 0 问题，也没有未核实的阻塞疑点。

初次系统编译环境缺 makedepend，仅加入 conda 编译器但未配置头文件路径时缺 zlib.h；切换既有 in-switch 环境并设置 include/lib 后构建成功。首次库身份检查正确拒绝了真实 CUDA libcudart；这些诊断不计为通过运行。Serena 工具未在本会话提供，无法激活项目，改用文件与源码直接检查。

未运行 SASS trace（无现成小 trace）、冻结 110 模块/九例前向、大于 96KiB shared、132SM 地址和 tensor map/cluster 等后续能力。组件及基础 PTX 提供当前共用资源路径回归；未来修改 trace 资源来源/默认值时应补对应 trace。未扩测未用 ABI、非法输入穷举、任意 stream 并发或 DMA 精确时序。

## 复跑与交接

复跑使用新目录，保留本轮证据：

```bash
cd /home/wangjz/in-switch/accel-sim
source workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-0/run.py \
  --run-dir workloads/moe/runs/hopper-stage-0/recheck-stage0/timed --test all
```

在新运行目录执行以下观察，对 work、parameters、modules、shared 同样运行；另复制配置追加 -gpgpu_concurrent_kernel_sm 1 后观察 shared 的计数分支：

```bash
/usr/bin/gdb -q -batch \
  -x /home/wangjz/in-switch/accel-sim/workloads/moe/tests/hopper/stage-0/observe.gdb.py \
  --args ./stage0 launch
```

组件在已加载上述环境的仓库根目录构建。本次精确编译/运行参数见 timed/run.json；生成 PTX、编译器 Push→Pop→cudaLaunchKernel 包装、全部日志在同目录。

```bash
g++ -std=c++17 -g -I gpu-simulator/gpgpu-sim -I /usr/local/cuda/include \
  workloads/moe/tests/hopper/stage-0/resource_component.cc \
  -L "$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG" -lcudart -o /tmp/stage0-resource
/tmp/stage0-resource
python3 workloads/moe/runs/hopper-stage-0/independent-20261108/verify.py
```

阶段 0 必要检查已完成，停止扩测。入口可支撑阶段 1 基础数值、warp 协作及单模块真实内核测试；是否启动由用户指定。
