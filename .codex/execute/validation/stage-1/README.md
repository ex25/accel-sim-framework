# 阶段 1 实际验收：基础数值与 warp 协作

状态：**通过**（2026-11-08 独立验收，限阶段 1 范围）。九类微测、11 个冻结 PTX 模块、依赖/访存顺序/时间参数及阶段 0 回归全部符合要求，未发现实现阻塞。现在可用独立入口运行真实转换、top-k/排序、专家计数、行映射、重排和 SwiGLU；下一步可按计划实施阶段 2，本轮未启动。

完整前向、gating、专家 FC1/FC2 和 H100 约 20% 性能精度仍未验收。全部通过范围与限制如下，统一证据见[summary.json](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/summary.json)。

[验收要求](../../../validation/stage-1.md) · [实现方案](../../../plans/stage-1.md) · [实现记录](../../implementation/stage-1/README.md)

## 被验版本与运行身份

独立阅读实现、测试及冻结 PTX，再在新目录执行。复用已有 debug 构建：交付记录中的源码差异、测试、库散列全部相符，所有被修改源码均早于库生成时间。未修改模拟器、生产配置、冻结程序或原测试预期，未向其它聊天发消息。

| 对象 | 实际版本与证据 |
| --- | --- |
| 外层 | 基准 `93dc07d702e3da9df17081f062f950119d621ff5`，保留原有阶段 0/1 实现记录及未跟踪测试；本轮只更新本验收记录、增加独立证据，未提交；运行中另出现 `.gitignore` 忽略 tests/ 的改动，原样保留 |
| 内层 | 基准 `62205297c2b0c7ba9ba1e275e6a9874f0c284a02` 加 15 文件未提交差异；[被验 diff](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/inner.diff)，SHA256 `b031a60b63b1f2fa3a38de87a9161bf06267be6f3503344b98764fb1df75a7fe` |
| 构建 | CUDA 12.8 / ptxas V12.8.61、GCC 13.4.0、debug；复用[最终构建](../../../../workloads/moe/runs/hopper-stage-1/development/build-final.log)，本轮未重建 |
| 实际库 | 内层 `lib/gcc-13.4.0/cuda-12080/debug/libcudart.so.12`，SHA256 `bd0d52fe3007b792c5f729f1d820114beba580c4fec896b507477e0cb9ab8173`；各 runner 的 ldd 与 performance simulation 日志交叉确认 |
| 模式/配置 | `PTX_SIM_MODE_FUNC=0`，PTXPlus=0；H100 副本仅 SM 数 132→1、chiplet partition 2→1。普通配置 SHA256 `e3d91789335d64379303f9233ba43cfa6c317e019115b5605c43f77672b27bbf` |

[版本核对](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/version-check.json)、[本轮源码/测试/库散列](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/versions.json)。各 runner 的 `run.json` 另保存命令、配置与库散列；真实组保存原 PTX/forward.bin 散列。阶段 0 前置依据[原独立验收](../stage-0/README.md)，本轮另重跑八项回归。Serena 工具不可用，使用文件、源码和只读 GDB 检查。

## 微测与受影响回归

复用[阶段 1 测试源码](../../../../workloads/moe/tests/hopper/stage-1/)，预期来自固定整数位表和按实际 lane 的人工公式，未调用模拟器函数。新补充的只读观察与断言保存在[本轮目录](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/)，不修改被测状态。

| 项目 | 独立复跑结果 |
| --- | --- |
| 三种 BF16 转换 | 舍入中点、±0、无穷、NaN 类别、无 ftz 位扩展、两源顺序与分别舍入均精确正确；FP16→FP32→整数回归通过 |
| shfl 三种、ballot、redux、elect | 两 warp、全/高半/运行时 mask、原地 shuffle、边界 predicate、负数和、重复 leader、`_|p` 全部通过，完整输出字节一致 |
| warp 屏障 | 全 mask 和寄存器高半 mask 的 shared 互读精确正确；另一 warp 不执行屏障也正常完成；顺序证据见下节 |
| 位操作/hint load | 全 32 lane 的 lanemask_le、bfind 两种返回正确；两种 b32 hint 和 b8 的 00/7f/80/ff 正确；关闭谓词保留初值 |
| 原路径 | PTX 5.0/sm_60 原 shuffle 形式通过；阶段 0 八项符合预期，其中 assert/trap 按要求非零失败；真实排序/映射覆盖旧访存、shared atomic、CTA barrier |

[九类微测](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/micro/run.json)、[阶段 0 回归](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/stage0/run.json)。各目录保留生成 PTX、`.actual`/`.expected`、日志及 `.time`，结论不只依赖退出码。

## 同步、访存与时间模型

本轮重新观察，不直接沿用实现报告的事件数字：

- 全 warp 的 shared store 在 **5046** 完成，屏障 **5049** 放行，后续 load 在 **5062** 发射；另一 warp 的 store 到 **6470** 才完成。高半 mask 对应 **5052→5058→5071**，参与/到达均为 `ffff0000`。寄存器 mask 解码为读取源，没有目的；不进入 CTA 屏障计数。
- shuffle latency 4→16：32 条依赖链 **5570→5954**，64 条 **5794→6562**，分别增加 **384/768**，等于条数×12。固定 latency=32、initiation 2→8：总周期 **15598→21490**；64 次接收事件中四个单元最小间隔分别均为 **2/8**。
- shfl 四个寄存器源全部进入解码。redux 接单元 4 的 4/1，elect 接单元 1 的 4/4，数据/predicate 双目的与 `_` 的依赖正确。六次 redux 数据消费、六次 elect predicate 消费均不早于完成；BF16x2 等第二源 add 完成，随后 store 等转换完成。同周期先完成再发射属于合法放行。
- hint load 在两个 warp 共六次读取中保持 **4/4/1 字节**；有效 mask 均为 `fffffffe`，只有 lane1～31生成请求。b32 地址按 lane 每次加 4，b8 仅访问指定四字节，保留正常 global/cache 通路。

证据：[全 warp](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/micro/observe-warp-full.log)、[高半 warp](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/micro/observe-warp-partial.log)、[集合依赖](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/micro/observe-collective-dependencies.log)、[打包依赖](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/micro/observe-numeric-dependencies.log)、[load 宽度/地址](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/micro/observe-loads.log)。敏感性：[latency16](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/latency16/run.json)、[间隔 2](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/initiation2/observe-unit.log)、[间隔 8](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/initiation8/observe-unit.log)。

## 真实 kernel：用途、输入与预期

选 Qwen64/128 的原 entry、grid/block/shared 与参数字节，只替换指针。人工输入不依赖未运行的 gating/专家 GEMM。CPU 参考位于 `real.cc`，独立生成数学结果和布局。下面指针按原参数位置依次列出；完整标量/结构体字节和 launch 见[启动记录](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/real/launches.json)。

| 模块、用途 | 指针缓冲、标量与独立预期 | 本轮结果 |
| --- | --- | --- |
| 70：BF16→FP32 转换 | BF16[131072] → FP32[131072]，长度 131072；有限位表循环，高 16 位扩展 | 131072 项逐位一致 |
| 3：选 top-k 专家 | FP32[128,64] 分数、FP32[128,4] 值、int64[128,4] 编号；E=64、k=4、128 行；每行轮换无重复整数排列，检查前四名集合与值/编号配对 | 512 对精确一致 |
| 4：top-k 排序 | 上述值和编号两缓冲；128 行、k=4；CPU 降序参考。3 的三个、4 的两个 216B TensorInfo 结构参数保留原字节，只换指针 | 512 对精确一致 |
| 77：统计每专家 token 数 | u8[128,64] 分配表 → int64[64] 计数，E=64、tokens=128；每 token 四个不同专家，CPU 逐项计数 | 64 项精确一致，总和 512 |
| 10→38→78：构造行映射 | 10：分配表、int32[128,129] 行表、int32[64] tile 计数，标量 128/64/129；38：行表、tile 计数，标量 128/129；78：行表、宽度 129。CPU 前缀、专家偏移生成全局行号，再检查每 token 四对行号/专家及数量 | 全部精确一致，512 对无重复/遗漏 |
| 79：按专家重排 token | BF16[128,2048] 输入、BF16[512,2048] 输出、行表，标量 129/2048/2048；一对有限 BF16 元素联合编码 token 和完整列坐标，CPU 按专家遍历构造预期 | 1048576 项精确一致 |
| 80→81：chunk 映射及排列 | 80：int64[64] 计数、int64[64] 反向专家顺序、int32[512] 逆映射；81：BF16[512,2048] 输入/输出、逆映射，宽度/stride=2048。CPU 按该专家次序拼接，逐项比较 | 512 项逆映射及 1048576 项数据精确一致 |
| 82：SwiGLU 激活 | BF16[512,2816] → BF16[512,1408]，输出长度 720896；两半用不同有限小值，CPU 算 `x/(1+exp(-x))*y` 并独立舍入 | 720896 项全有限，最大绝对误差 0 |

映射检查每专家计数、前缀/全局行号、token 的四对编号与行号、逆映射和全部重排数据；不强加同值行内排序的 CPU tie order。SwiGLU 要求全部值有限并逐元素满足 `1e-5+0.02*abs(ref)`，其余离散结果和 BF16 位转换精确比较。

[真实运行元数据](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/real/run.json)；原始输出与完整比较见同目录的 [convert](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/real/convert.log)、[route](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/real/route.log)、[mapping](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/real/mapping.log)、[swiglu](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/real/swiglu.log)。

这些是独立测试入口执行的真实冻结 PTX；没有启动完整前向，也未把其它模型/尺寸的同类模块标成已验证。原完整 launcher 仍需要后续 tensor map 等能力，不能直接据此声称它已跑通上述前缀。

## 运行成本与停止判断

本轮 [verify.py](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/verify.py) 的 **33 条结果/事件记录断言通过**；直接复核输出字节、mask/地址/宽度、依赖完成顺序、时间参数敏感性与源码/库身份，不只读取 PASS 字样。当前必要检查完成，停止扩测。

| 真实运行组 | 初始化/准备 | launch→同步合计 | 比较循环 | 全进程墙钟 / 峰值 RSS |
| --- | ---: | ---: | ---: | ---: |
| 转换 70 | 1.044s | 19.528s | 0.0030s | 20.71s / 282.0MiB |
| 路由 3→4 | 0.908s | 74.789s | 0.0001s | 75.78s / 313.1MiB |
| 映射/重排七模块 | 0.941s | 384.204s | 0.0298s | 387.32s / 335.0MiB |
| SwiGLU 82 | 1.755s | 139.668s | 0.0256s | 141.78s / 301.1MiB |

初始化含 constructor 到首次提交的输入/CPU 参考准备和上传；launch→同步含预解码和模拟；比较循环不含下载/写文件。记录来自无调试器的普通运行，部分短微测同时运行，墙钟只作本机成本样本。不能由普通内核外推 WGMMA 或九例的运行预算。

设备统计独立保存在 summary：转换 **34357 周期 / 1310720 条线程指令**；top-k/排序分别 **81161/36452 周期**；七步映射/重排分别 **11385/49698/27389/103468/88088/106255/170792 周期**；SwiGLU **142419 周期 / 15228928 条线程指令**。这些日志周期**含每次 launch 等待**，不是纯 kernel 起止差。单 SM、人工输入与 ncu 采集条件不可比，本轮不计算约 20% 硬件性能误差。

当前没有已复现的实现错误或待交执行 agent 修复的阻塞。暂缓范围沿原阶段方案：任意分歧/跨动态指令分批到达未支持，当前真实路由未触发；eviction hint 精细逐出策略未建模；其它 BF16 算术、未用 redux/vote 形式、H200 未验。若后续真实消费者触发缺失参与者，必须先补汇合，不能缩小 mask 放行。

没有现成小 SASS trace，本轮未运行 trace。核查新增 warp 标记默认 false，普通 SASS membar 沿原条件；PTX 共用发射/完成路径及阶段 0 回归提供本次受影响证据，不冒称 SASS 运行通过。gating、专家 FC1/FC2、完整九例和硬件性能误差仍属于后续验收。

复跑从仓库根目录加载[既有环境](../../../../workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh)，再执行：

```bash
python3 workloads/moe/tests/hopper/stage-1/run.py --run-dir <新的微测目录>
python3 workloads/moe/tests/hopper/stage-1/real.py --run-dir <新的真实运行目录>
```

只运行一组时给 `real.py` 加 `--test convert`、`route`、`mapping` 或 `swiglu`。敏感性参数及每个完整运行命令在对应 `run.json`；GDB 观察命令另存 [commands.sh](../../../../workloads/moe/runs/hopper-stage-1/independent-20261108/commands.sh)，复跑需更换运行根目录以保留原证据。测试/输入与阈值均未修改，无需执行 agent 修复后复验。已具备推进阶段 2 普通 cp.async 与 gating 的前置条件；是否启动由用户指定。

## 提交归档

内层交付提交 `b25fd0a18cb01a0c112042295007a0ac2c3772ad`，与本次验收保存的 15 文件差异完全一致；外层文档、测试及 prompt 由配套 `stage1` 标签定位，标签说明保存两个仓库的提交号。上文未提交状态描述的是验收发生时的版本，保留用于追溯，不改变验收结论。测试源码已显式纳入 Git，大型原始运行证据仍留本地 runs 目录。

用户决定先不做硬件性能对照，[通用 prompt](../../hardware-baseline-prompt.md)保留供后续使用；本次不启动比较或后续阶段。
