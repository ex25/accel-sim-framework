# 阶段 6 实际实现：矩阵、异步 WGMMA 与寄存器额度

状态：**实现与必要自测完成；2026-11-09阶段5/6联合独立验收通过，范围与证据见[验收记录](../../validation/stage-6/README.md)**。PTX 周期模式现在能计算 `m64n128k16` 的全部 8192 项 BF16→FP32 结果，支持连续异步累加、组等待、矩阵 shared 读写和 CTA 内寄存器额度调整。TMA 输入→屏障→矩阵→输出已串接；尚未启动阶段 7 的真实专家/九例前向，也未校准硬件总时间。

实现时以阶段5现有能力为前置，没有把当时尚缺联测的阶段5视为完全验收。后续独立验收已补齐首次联测及双缓冲流水，唯一实际报告由[阶段5验收记录](../../validation/stage-5/README.md)索引；以下保留原实现与自测交付。

## 做了什么，为什么需要

原路径的 tuple、依赖和写回容量不足以容纳每线程 64 个结果，而且上层忽略写回端口未完成状态，会在结果尚未全部服务时释放依赖。本轮扩容完整链路，并让四个 warp 关联同一次逻辑矩阵操作，最后一个 warp 的全部结果写完后才完成该操作。

```text
TMA 交付 / 普通 shared 写入并发布
  → 四 warp 到齐 → collector 收集完整输入 → shared 服务完成
  → 计算一份逻辑矩阵 → 既有 GMMA 管线 → 寄存器 bank 分批写回
  → 对应组完成 → wait 返回 → 普通指令消费 / 旧缓冲复用
                   ↘ 后续有序 WGMMA 可在前条写回结束前发起
```

| 对应 plan | 实际位置（内层仓库） | 行为与原因 |
| --- | --- | --- |
| [registers](../../../plans/stage-6.md#registers) | `abstract_hardware_model.h`、`ptx_ir.h`、`ptx.y`、`ptx_parser.*`、`cuda-sim.cc::pre_decode` | tuple64，输入68、输出64，完整保存 tuple 顺序；WGMMA 表示两个描述符、谓词和64个累加器输入。scale_d=false 不功能性读取未初始化 D。 |
| registers | `shader.cc::writeback`、`opndcoll_rfu_t::writeback`、`scoreboard.*` | 保留 EX_WB 和已服务 bank 进度，只统计实际写入项；最后一批才写功能寄存器、解除依赖、完成。多个异步写者用引用数保护普通后继。 |
| [layout](../../../plans/stage-6.md#layout) | `instructions.cc::matrix_impl`、`matrix_address`、`ptx_matrix_op::compute/writeback`；新增 `cuda-sim/ptx_matrix.h` | 按 K-major 描述符读取 A/B，BF16 输入、FP32 fma；一次计算8192项，再按 thread/tuple 分配。TMA 与 WGMMA 只共用低层16B块 swizzle，描述符解码各归原处。 |
| layout | `matrix_impl`、`abstract_hardware_model.cc::generate_mem_accesses` | 分别实现 `ldmatrix/stmatrix.sync.aligned.x4.m8n8.shared.b16`。每 lane 四个4B地址都进入 shared bank 服务，共512B/warp。 |
| [async](../../../plans/stage-6.md#async) | `shader.cc::issue_ptx_matrix/prepare_ptx_matrix`、`shd_warp_t::waiting`、`ptx_matrix_warpgroup` | CTA 内按 warpgroup 和动态序号集合四 warp；fence 排序普通寄存器操作，commit 划组不等待，wait1/0 根据全部参与者的真实写回完成状态返回。cp.async、bulk、WGMMA 使用各自队列。 |
| [timing](../../../plans/stage-6.md#timing) | `specialized_unit::can_issue`、`ldst_unit::reserve_ptx_matrix_shared`、H100配置、`cuda-sim.*` | 显式启用现有第5专用 GMMA 单元和详细 collector；TMA/矩阵共用 shared 服务预留，GMMA 延迟/发起间隔分别可调。逻辑计算262144 math op与四warp发射分列。 |
| [quota](../../../plans/stage-6.md#quota) | `ptx_matrix_cta::allocate`、`issue_ptx_matrix`、CTA退出位置 | 初值来自实际 ptxas 寄存器元数据；dec40/inc232 是新额度。仅同CTA池内分配，不足仅暂停申请组，归还后唤醒；不截断 PTX 虚拟寄存器，不改变原物理驻留预算及最终回收。 |
| 必要前置修复 | `ptx.l/ptx.y`、`ptx_parser.*`、`ptx_ir.h`、`ptxinfo.l` | 原32位整数解析会截断真实64位矩阵描述符。新增64位 literal；精确识别 CUDA12.8 本轮实际出现的 C7508/C7515/C7519 信息行，保留资源数据及编译失败处理。 |

本轮具体新增指令为上述 ldmatrix/stmatrix、`wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`、`wgmma.fence/commit_group/wait_group.sync.aligned`、`setmaxnreg.dec/inc.sync.aligned.u32`。实现当前消费者的 BF16、scale A/B=1、trans A/B=0；`fence.proxy.async.shared::cta` 复用阶段5，未另加实现。

**复用与差别：**沿用 SASS 的专用执行单元、四warp发射方式、寄存器 bank 仲裁和已有可停顿流水队列；PTX 自行生成数值、64项依赖及共同完成身份。无需新增结果重试队列，也未再造一份 tensor 单元。H100 从简化 collector 改为已有详细 collector，才能实际计入全部端口服务。额度仅做预算调整，未模拟物理寄存器搬家；64/64 是当前形状的初始参数，不是已校准硬件周期。

长延迟测试暴露并修复了一处调度死锁：后发指令可能先收齐少量寄存器，占住前条指令要用的入口。详细 collector 现在保持每个 warp 的 WGMMA 发出顺序，同时保留流水重叠。写回也按每warp动态序号完成，避免较早结果覆盖较晚结果。

## 自测结果与证据

以下是独立验收前的原自测记录。其中integration的屏障位置后来发现与TMA目标重叠，已由验证者修正并复跑；原integration结果不作为最终通过证据。合法测试、最终版本及独立结论以[验收记录](../../validation/stage-6/README.md)为准。

debug完整17项、release完整18项全部通过；debug另跑 `legacy8` 通过。两种构建的17项共同测试设备周期完全一致，pipeline与serial的原始BF16输出逐字节相同。全部使用最小合法 PTX，经实际 ptxas 编译、内嵌 fatbinary 注册；`ldd` 检查模拟器 libcudart，`PTX_SIM_MODE_FUNC=0`，日志确认 performance simulation。CPU 独立计算精确整数矩阵；人工16B块表准备64B/128B swizzle，测试不调用模拟器地址函数。每个测试连续两次 launch，逐项比较并保存原始输出。

| 检查 | 自测结果与证据 |
| --- | --- |
| 普通 shared 单次与初值 | `single64/128`、`initial` 精确比较全部8192项；`offset64/128`、`offset_low64/128` 跨八行组，含非零起点和相位。[独立地址表](../../../../workloads/moe/runs/hopper-stage-6/metadata/layout-addresses.json)。 |
| tuple64及完整写回 | 不连续寄存器与冻结模块16的 `%f7420…%f7357` 真实tuple逐项核对名字、顺序、逻辑目标及 bank 元数据；实际64目标每warp分4批写回，32bank时为8批，完成只在最后一批。普通store包含 j63。 |
| 协作与异步 | 四warp错开到达后才读取，一逻辑计算对应四次发射；同PC循环两轮不混用。`chain/groups` 检查2C与X/Y两组，长延迟时实际重叠、wait1提前返回；状态组件另控制最后warp完成，并确认两种copy队列仍未排空。 |
| 独立矩阵 IO | `ldmatrix` 用普通shared store准备四张矩阵，`stmatrix` 直接构造packed输入，分别检查每个元素；非仅往返检查。 |
| 额度 | 真实 ptxas 元数据168/thread、384线程；两个消费者先因空闲0等待，生产者dec40后各授予8192，最终40/232/232，总64512。高虚拟编号值及200个存活值保持，连续launch完成；组件检查邻CTA池不变。 |
| 阶段5首次联测 | `integration`：TMA→mbarrier→WGMMA→普通global store，全部8192项精确比较；输入交付先于矩阵读取。早期首次证据见[first9](../../../../workloads/moe/runs/hopper-stage-6/development/first9/run.json)，最终以delivery/release为准。 |
| 完整流水 | `pipeline/serial` 使用两组不同输入，经过BF16转换、stmatrix、TMA输出，逐项检查16384个BF16结果。wait后覆盖旧输入；bulk read wait后复用同一输出shared。流水与串行原始结果相同，长GMMA延迟时下一块输入交付落在当前GMMA执行期间。 |
| 受影响回归 | 阶段5 input32/mask3/output、阶段2 copy_mix/groups2及其状态组件；原8项WMMA load.c/store.d数值路径 `legacy8`。缺可运行Hopper SASS trace，未称SASS运行通过；未扩测无关旧指令。 |

[测试入口](../../../../workloads/moe/tests/hopper/stage-6/run.py) · [PTX生成](../../../../workloads/moe/tests/hopper/stage-6/generate_micro.py) · [独立CPU检查](../../../../workloads/moe/tests/hopper/stage-6/micro.cc) · [状态组件](../../../../workloads/moe/tests/hopper/stage-6/state_component.cc) · [事件核对](../../../../workloads/moe/tests/hopper/stage-6/summarize.py)。源快照、实际配置、命令、库散列、日志和原始bin均在对应运行目录；汇总见 [summary.json](../../../../workloads/moe/runs/hopper-stage-6/summary.json)。

### 周期与本机成本

设备周期包含既有5000周期launch费用；两次launch分别记录，不用第二次减第一次推算单条指令延迟。下表是第一次launch，涉及DRAM冷/热差异时另列第二次。

| 单参数变化/对照 | 实际观察 |
| --- | --- |
| GMMA latency64→256，init64 | chain 总周期40184→40375；首条发出31476，完成31736，第二条31617发出，证明有重叠。 |
| init64→128，latency固定256 | chain 发射间隔141→205，总40375→40439；pipeline 受输入供给限制，总周期不变。 |
| shared latency29→129 | integration 14603→15102；pipeline 9817→11209，均数值正确。此参数同时影响TMA与矩阵shared入口，不单独归因于一种指令。 |
| DRAM latency194→600 | integration 第二次14803→15077，pipeline 第二次10220→10504；首次DRAM读数为0、周期不变；第二次有384次DRAM读，未强行调参取得增长。 |
| register banks64→32 | 同一single64输入每warp写回4→8批，总周期40043→40049。 |
| pipeline 对照 serial | 首次9817对10656周期，第二次10220对11261；工作与结果相同，差异来自合法重叠，非硬件误差校准。 |

运行目录：[debug完整](../../../../workloads/moe/runs/hopper-stage-6/delivery/run.json)、[release完整](../../../../workloads/moe/runs/hopper-stage-6/release/run.json)、[延迟](../../../../workloads/moe/runs/hopper-stage-6/sensitivity-latency/run.json)、[发起间隔](../../../../workloads/moe/runs/hopper-stage-6/sensitivity-init/run.json)、[shared](../../../../workloads/moe/runs/hopper-stage-6/sensitivity-shared/run.json)、[DRAM](../../../../workloads/moe/runs/hopper-stage-6/sensitivity-dram/run.json)、[bank](../../../../workloads/moe/runs/hopper-stage-6/sensitivity-banks/run.json)、[阶段5回归](../../../../workloads/moe/runs/hopper-stage-6/regression-final-stage5/run.json)、[阶段2回归](../../../../workloads/moe/runs/hopper-stage-6/regression-final-stage2/run.json)。

下表为release本机实测；测试带事件记录，部分检查并行运行，属于本次观测。注册列为main至模块注册完成，模拟列为每次kernel提交至同步返回；进程整体时间还包含分配、拷贝等准备工作。

| 样本 | 注册秒 | 两次模拟秒 | 两次CPU比较秒 | 进程墙钟秒 | 峰值RSS KiB |
| --- | ---: | --- | --- | ---: | ---: |
| single64 | 0.775 | 7.613 / 10.515 | 0.0034 / 0.0033 | 18.96 | 297400 |
| integration | 0.762 | 2.709 / 2.807 | 0.0034 / 0.0033 | 6.34 | 314612 |
| pipeline | 0.765 | 1.510 / 1.606 | 0.0071 / 0.0069 | 3.95 | 329788 |
| quota | 0.801 | 1.910 / 2.006 | 0.0003 / 0.0002 | 4.77 | 301116 |

对应debug的single64进程43.02秒、pipeline进程7.85秒，峰值RSS分别301228/333628 KiB；完整初始化、模拟、比较与RSS逐项见汇总。没有真实专家计算，不能据此更新冻结九例成本预算或声称H100时间误差达到约20%。

## 复验与版本

自测交付时的外层基准为 `3fc74557dcad398bbdf65ce71fcd928cf53c01c4`，内层基准为 `5f74b472599c8234961a202cd11f12caa628e494`。开始时均干净，当时实现未提交，新增测试与内层头文件仅设intent-to-add便于审阅。原自测diff、源/库SHA见[versions.json](../../../../workloads/moe/runs/hopper-stage-6/metadata/versions.json)；每次运行另保留自身源快照，开发失败日志不作通过证据。后续修正测试及提交归档以[独立验收记录](../../validation/stage-6/README.md#提交归档)为准。

CUDA12.8/ptxas12.8.61、GCC13.4.0、clang-format16.0.6；[debug构建](../../../../workloads/moe/runs/hopper-stage-6/metadata/build-debug.log)、[release构建](../../../../workloads/moe/runs/hopper-stage-6/metadata/build-release.log)。重跑时从仓库根目录：

```bash
source workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-6/run.py --trace \
  --test integration,pipeline,serial \
  --run-dir workloads/moe/runs/hopper-stage-6/recheck-joint
python3 workloads/moe/tests/hopper/stage-6/run.py --trace \
  --run-dir workloads/moe/runs/hopper-stage-6/recheck-all
python3 workloads/moe/tests/hopper/stage-6/run.py --trace \
  --gmma-latency 256 --test chain,groups,pipeline \
  --run-dir workloads/moe/runs/hopper-stage-6/recheck-overlap
python3 workloads/moe/tests/hopper/stage-6/summarize.py \
  workloads/moe/runs/hopper-stage-6/recheck-all \
  workloads/moe/runs/hopper-stage-6/recheck-overlap \
  --output workloads/moe/runs/hopper-stage-6/recheck-summary.json
```

验证agent先看 `integration` 的全部数值与输入完成顺序，补齐阶段5首次联测，再检查 `pipeline/serial`、完整tuple写回、异步组及额度；本轮不填写或覆盖独立验收结论。当前形状以外的矩阵形式、真实专家组合、九例端到端和硬件时间校准留在各自阶段；必要自测结束后停止，不实施阶段7。
