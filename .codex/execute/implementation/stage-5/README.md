# 阶段5实际实现：tensor map 与 TMA

状态：**阶段5实现及自测完成；2026-11-09已补齐TMA→WGMMA联测，独立验收通过（限阶段5范围）**。现在可以通过正式CUDA查询接口取得编码函数，把128B描述符传入PTX、修改和发布，再搬入shared、向两个CTA组播或写回global。输入在数据和零填交付后才通知mbarrier；输出读完shared后可以复用源，kernel退出仍等待写确认。

当前独立结论见[验收记录](../../validation/stage-5/README.md)，两阶段共同联测已补验；下文保留原实现与自测交付。前置阶段3/4已独立验收。本轮从冻结 PTX 提取真实专家的地址指令，在微测内重放地址计算、TMA、屏障和普通shared读回；没有运行完整冻结专家。冻结 PTX 中的实际shared起点均对应swizzle phase 0，非零phase仍由人工地址测试覆盖。

[方案](../../../plans/stage-5.md) · [验收要求](../../../validation/stage-5.md) · [自测汇总](../../../../workloads/moe/runs/hopper-stage-5/summary.json) · [测试源码](../../../../workloads/moe/tests/hopper/stage-5/)

```text
主机编码 → 参数/shared副本 → replace → 128B发布 → 写确认 → acquire
输入：描述符快照 + global请求 → 全部返回 → shared服务/零填 → 每目标屏障
输出：描述符读取 → shared源读取 → read wait释放 → global请求 → 写确认
                                      ↓可复用源               ↓允许kernel退出
```

## 实际改动与方案对应

以下源码路径相对独立模拟器仓库；[完整差异](../../../../workloads/moe/runs/hopper-stage-5/metadata/inner.diff)。

| 对应plan | 文件、函数或状态 | 为什么改、行为变化 |
| --- | --- | --- |
| [128B对象](../../../plans/stage-5.md#descriptor) | 新增 `src/cuda-sim/tensor_map.h`、`libcuda/tensor_map.cc`；修改 `libcuda/Makefile` | 固定偏移格式，不依赖C++ padding或外部host表。`cudaGetDriverEntryPointByVersion` 按12000/default返回可调用编码函数；允许初始小dim/stride0占位对象。独立翻译单元使用本机CUDA12.8 tensor-map ABI，避免全面升级历史 `cuda_api.h`。 |
| [解析与replace](../../../plans/stage-5.md#descriptor) | `ptx.l`、`opcodes.def`、`ptx_ir.h`、`cuda-sim.cc`、`instructions.cc::tensor_impl`；`core_t::execute_warp_inst_t` | 识别下列11种mnemonic，登记地址、坐标元组、ordinal等输入依赖。复用已有texture元组语法展开 `[map,{x,y}]`。replace在shared/SYNCS服务点写回，warp等collector写回再继续；地址64位、dim实际值、stride字节，保留inactive字段。 |
| [发布/acquire](../../../plans/stage-5.md#publication) | `issue_warp/store_ack`、`ldst_unit::issue/service_ptx_tensor`；kernel的 `ptx_tensor_publications`、warp的 `m_ptx_tensor_acquire` | 一次warp发布分为32 lane各4B，生成真实128B合并写；shared读取服务结束捕获对象，最后写确认提交对象。acquire只等对应描述符地址的在途发布；跨线程仍用原同步。每次TMA保存描述符快照，后续更新不改变旧请求。 |
| [输入](../../../plans/stage-5.md#input) | `ptx_tensor_transfer`、`tensor_impl`、`ldst_unit::writeback/schedule_ptx_tensor/service_ptx_tensor` | 生成有效二维地址、2B数据及OOB零，按绝对shared地址作64B/128B swizzle。复用TMA合并、mask、sector、L2/DRAM及返回仲裁，128B描述符也产生读取。最后global返回后预留shared服务，先写数据/零填，再向正确对象/phase完成整个tile逻辑字节。 |
| [组播](../../../plans/stage-5.md#multicast) | 上述入口、阶段3的 `resolve_cluster_shared`、阶段4屏障表 | mask严格决定目标，发起者不默认收到通知。一次global读取、一份payload，每目标独立shared服务/完成；源warp在所有目标交付前保留在途状态，避免CTA/kernel/shared提前销毁。 |
| [输出和两种完成](../../../plans/stage-5.md#output) | LSU描述符请求队列、shared服务事件、`store_ack`；warp的 `m_ptx_bulk_groups/m_ptx_tensor_pending/m_ptx_tensor_stores_outstanding` | 描述符读取共用LSU请求端口，返回后读shared；读取完成捕获数据并完成本lane的bulk read组，之后发global写。最后ack提交捕获值，不重读已复用shared。复用阶段2序号计数类，另存逐lane bulk队列。普通release检查排除PTX TMA输出ack，避免隐式串行；kernel退出仍等所有ack。 |
| [async fence](../../../plans/stage-5.md#proxy) | `PTX_ASYNC_FENCE`、SYNCS/shared服务与warp pending | 本线程此前shared操作沿同一有序服务通路完成，fence写回后才允许该warp异步读取。跨线程仍需原barrier；无全局排空或shadow shared，阶段6直接复用此入口。 |
| 事件证据 | `trace_streams.tup::PTX_TMA`，发起/服务/完成及 `mbarrier_complete` | 默认关闭的trace记录请求数、逻辑字节、OOB、描述符基址、目标交付、真实wait、read/ack、分组和源复用；沿既有trace开关，无永久调试输出。 |

实际新增并执行的指令形式：

- `tensormap.replace.tile.global_address.shared::cta.b1024.b64`
- `tensormap.replace.tile.global_dim.shared::cta.b1024.b32`
- `tensormap.replace.tile.global_stride.shared::cta.b1024.b64`
- `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned`
- `fence.proxy.tensormap::generic.acquire.gpu`、`fence.proxy.async.shared::cta`
- `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`
- 同上带 `.multicast::cluster.L2::cache_hint`
- `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group`
- `cp.async.bulk.commit_group`、`cp.async.bulk.wait_group.read`（0/1）

同时回归阶段4的init/expect/arrive/arrive.expect/try_wait/fence和阶段3cluster同步。普通cp.async没有合入bulk队列。

## 复用、时序与方案差别

复用SASS/common的TMA地址合并、sector/mask、global缓存/DRAM/互连、返回仲裁、写确认、正常shared/SYNCS管线、collector/scoreboard及warp退出。PTX自行生成trace原已提供的地址、值、描述符和屏障目标。PTX绕过原SASS的issue时OOB扣数、默认更新发起者及ack兼任bulk read完成；SASS保留原组接口，没有套入PTX状态。

shared交付选择方案允许的**全tile统一交付**：全部global返回后按 `ceil(bytes/(32 banks × 4B))` 预留shared服务量，叠加配置中的29周期端点延迟；远端再加一次端点延迟，每目标独立排队。普通shared dispatch与这份服务预留相互排队。它近似shared端口和两CTA运输，没有复刻GPC网络或逐sector流式交付；不是固定等待后无条件放行屏障。

当前只读矩阵在功能发起时取值快照，但shared可见性严格等实际返回/服务。输入描述符和数据在同批请求中建模，不额外串行添加描述符延迟；输出独立发128B描述符读，返回后读shared。缓存沿CG/L2，精细L2 hint和描述符缓存政策暂缓。只保存必要payload，不每sector复制整tile。

## 自测与证据

debug构建通过，新增C++按clang-format16.0.6格式化。两SM H100副本只将132→2，保留两partition、1980MHz、容量/缓存参数；shared129另改29→129，dram1194另改DRAM延迟194→1194。各run.json保存命令、配置散列、工具链、ldd和库散列，确认加载debug模拟libcudart、`PTX_SIM_MODE_FUNC=0` 和 `mode=performance simulation`。

[汇总脚本](../../../../workloads/moe/tests/hopper/stage-5/summarize.py)对保存的数值结论和因果事件执行 **604条断言**。这是实现自测，不代替独立验收。

| 项目 | 实际结果与入口 |
| --- | --- |
| 正式编码、replace、发布 | 15项都经正式查询ABI，参数→shared→修改→global；逐字节检查64位高地址、stride0=320B、dims0/1=128/160、inactive dims=1及strides=160，原host对象不变。每次发布恰好4个32B请求。[final-debug/run.json](../../../../workloads/moe/runs/hopper-stage-5/final-debug/run.json)。 |
| 三种box × 完整/部分/全OOB | 32×64/64B、64×64/128B、64×128/128B；128×160域、320B行步长、shared先填0xabcd。两次launch换位数据，以独立8×8块表逐元素检查普通shared读回。部分有效512B，其余精确零；全OOB无矩阵读取，仅4个descriptor请求，仍完成全tile字节。[input128](../../../../workloads/moe/runs/hopper-stage-5/final-debug/input128.log)、[oob128](../../../../workloads/moe/runs/hopper-stage-5/final-debug/oob128.log)。 |
| 冻结专家地址采样 | [微测源码](../../../../workloads/moe/tests/hopper/stage-5/generate_micro.py)、[来源及行号](../../../../workloads/moe/runs/hopper-stage-5/expert-addresses/expert-address-source.json)、[运行记录](../../../../workloads/moe/runs/hopper-stage-5/expert-addresses/run.json)。对唯一含TMA的冻结PTX（SHA256 `118abe9e...19a651c5`）全部20个位置作静态分类：8条shared→global输出、8条64B输入、2条128B组播、2条128B普通输入；记录各组shared起始公式。微测实际重放一条64B输入和两种128B输入的地址/TMA指令，逐stage/rank检查目的地址、交付地址、barrier、descriptor、数据及guard的shared读回。13份dump均匹配独立16B块排列表，TMA交付数分别为1、12、12，成功wait分别为32、192、192条warp-lane记录。实际输入/输出起点phase均为0；输入目标在两块tile交付后成功wait。 |
| 非零shared相位 | base1280的64B/128B相位均为2，rotate还用9472/17664，保留16B块内顺序。冻结专家的真实地址片段已覆盖，但这些地址按实际计算均对齐到swizzle phase 0；因此非零phase仍由人工地址测试覆盖，不能把人工结果描述成真实专家地址。 |
| 在途描述符轮换 | 保留副本B，发布B并发搬运，原描述符再指向A/发布。旧B请求仍得到B、新A全999、副本B再次使用仍为B；第二次launch换为20000范围。首个B请求与第二次发布重叠。[rotate.log](../../../../workloads/moe/runs/hopper-stage-5/final-debug/rotate.log)。 |
| mask1/2/3、两cluster | 未选shared保持0xabcd且不等待不存在的字节；mask2仅rank1收到。源请求都为132个（4096B矩阵+128B描述符），mask3不翻倍；每目标各交付4096B，成功wait均晚于交付，另一cluster不混用。[mask2](../../../../workloads/moe/runs/hopper-stage-5/final-debug/mask2.log)、[mask3](../../../../workloads/moe/runs/hopper-stage-5/final-debug/mask3.log)。 |
| 独立输出与源复用 | CPU独立块表准备输入，经普通shared写和async fence输出。read wait0后覆写12345，再输出到另一global域；A/B全矩阵及域外0xeeee均正确，partial不写域外。首次读完28324、写确认28632，28338已覆盖shared，仍写回A。[output](../../../../workloads/moe/runs/hopper-stage-5/final-debug/output.log)、[output_edge](../../../../workloads/moe/runs/hopper-stage-5/final-debug/output_edge.log)。 |
| read wait1/0、最终退出 | 真实PTX：C=2时wait1截止1，pending1却不等待；wait0截止2则等待。正式warp状态组件固定read事件3/6，ack由20/40延至200/400，wait释放不变，`hardware_done`只在最后ack成立；另查空组、未commit、不同lane、普通cp.async和SASS组适配器。[组件](../../../../workloads/moe/runs/hopper-stage-5/final-debug/state_component.log)。人工延长ack不是GPU微测或SASS trace运行。 |
| 时序敏感性 | shared29→129：input32为11613/11813→19503/19638周期，mask2为11810→19895，交付/wait延后。DRAM194→1194：input32第二次11813→12808，查询总数21→50；首次未变，与缓存前序有关。独立输出read/ack本次均未变，不能声称只延长写确认；双完成隔离由正式状态组件及真实源覆盖证明。[shared129](../../../../workloads/moe/runs/hopper-stage-5/shared129/run.json)、[dram1194](../../../../workloads/moe/runs/hopper-stage-5/dram1194/run.json)。 |
| 相关回归 | 阶段4 local/remote/pending_tx及状态组件通过；阶段2 copy_mix/groups2及组件通过，覆盖普通shared/global和原异步等待。[stage4](../../../../workloads/moe/runs/hopper-stage-5/regression-stage4/run.json)、[stage2](../../../../workloads/moe/runs/hopper-stage-5/regression-stage2/run.json)。无可运行Hopper SASS trace，不称SASS运行通过。 |

输入全用真实expect/arrive/try_wait，没有注入TMA完成。input32两次共21次查询、2次成功；mask2两cluster两次共48次、4次成功。hint单次轮询近似沿阶段4，真实专家轮询成本尚未知。

### 本机成本与设备周期

debug微测包含既有5000周期launch费用，不是专家性能或九例预算。注册/模拟/比较为host时间戳，进程墙钟和RSS来自time-v；运行时有其它必要检查并行。

| 程序 | 两次设备周期 | 注册 | 两次launch/同步墙钟 | 每次比较 | 总墙钟 / RSS |
| --- | --- | --- | --- | --- | --- |
| input32 | 11613 / 11813 | 0.713s | 2.809 / 2.907s | 约0.48ms | 6.61s / 275.8MiB |
| input128 | 27068 / 27269 | 0.732s | 7.915 / 8.113s | 约0.82ms | 16.94s / 288.7MiB |
| rotate | 26649 / 26762 | 0.712s | 7.615 / 7.612s | 约0.65ms | 16.12s / 276.8MiB |
| mask3（两cluster） | 11950 / 11950 | 0.717s | 4.511 / 4.509s | 约0.89ms | 9.92s / 276.9MiB |
| output（两块） | 33971 / 47237 | 0.714s | 9.217 / 12.818s | 约29ms | 23.00s / 275.8MiB |

## 未完成与交接

1. 原待办首次TMA→WGMMA→普通store联测已于2026-11-09补验，报告在[阶段5验收记录](../../validation/stage-5/README.md)统一索引，阶段6引用同一份证据；当前未验的完整真实专家留阶段7。
2. 冻结样本的shared起点均为swizzle phase 0，没有实际非零起点可采。人工base1280的phase 2测试继续覆盖非零相位；若后续冻结 PTX 出现非零起点，再增加真实地址采样。真实地址片段、屏障及shared读回应由阶段5实现者负责，本轮已完成。
3. shared/远端交付量和顺序已建模，吞吐/端点延迟为上述近似，代表性专家具备后再判断主要耗时影响。未扩rank>2、其它dtype/interleave/swizzle、任意GPC、native编码；精细L2 hint暂缓。
4. 独立验收未复现实现错误；测试补强了128B编码独立预期及每轮输出哨兵，后续联测另修正了屏障位置、启动与输入。完整结论、版本及证据见[验收记录](../../validation/stage-5/README.md)，保留SASS trace未覆盖边界。

## 版本、命令与附件

自测/独立验收时的外层基准为 `3eb3261021dae8c3ae8fa3b98f6422b57089ece2`，内层基准为 `6e60cf3b1b2957a082ae6cad03613e8d43ef15c6`，当时实现未提交。内层14份文件差异；外层新增5份测试并更新阶段5文档。未改冻结程序/输入/容差、生产配置或其它阶段记录。提交归档及未完成验收项见[验收记录](../../validation/stage-5/README.md#提交归档)。

604条汇总所测diff SHA256：`1cd4fd24c5011c3f86a904238314ace68b9caf1b47b762f41638be034c392076`；库：`b72f702843347742430de54866e1ee9c5800090dd9753e0aa7758d660d88794a`。CUDA12.8/ptxas12.8.61、GCC13.4.0、clang-format16.0.6。[versions.json](../../../../workloads/moe/runs/hopper-stage-5/metadata/versions.json)保存源文件/测试散列，[构建日志](../../../../workloads/moe/runs/hopper-stage-5/development/build-final-debug.log)保存命令结果。随后仅为trace参数增加显式64位cast，清除本轮printf格式警告；重建及定向output/组件复跑见[final-format](../../../../workloads/moe/runs/hopper-stage-5/final-format/)，最终差异与散列保存在metadata的final文件，不改计算/调度。

从项目根目录复验，另建证据目录：

```bash
source workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-5/run.py --trace \
  --test rotate,input32,input64,input128,edge32,edge64,edge128,oob32,oob64,oob128,mask1,mask2,mask3,output,output_edge \
  --run-dir workloads/moe/runs/hopper-stage-5/recheck/final-debug
python3 workloads/moe/tests/hopper/stage-5/run.py --trace \
  --test expert64,expert128,expert128_64 \
  --run-dir workloads/moe/runs/hopper-stage-5/recheck/expert-addresses
python3 workloads/moe/tests/hopper/stage-5/run.py --trace --smem-latency 129 \
  --test input32,mask2 --run-dir workloads/moe/runs/hopper-stage-5/recheck/shared129
python3 workloads/moe/tests/hopper/stage-5/run.py --trace --dram-latency 1194 \
  --test input32,output --run-dir workloads/moe/runs/hopper-stage-5/recheck/dram1194
python3 workloads/moe/tests/hopper/stage-4/run.py --test local,remote,pending_tx \
  --run-dir workloads/moe/runs/hopper-stage-5/recheck/regression-stage4
python3 workloads/moe/tests/hopper/stage-2/run.py --test copy_mix,groups2 \
  --run-dir workloads/moe/runs/hopper-stage-5/recheck/regression-stage2
python3 workloads/moe/tests/hopper/stage-5/summarize.py \
  workloads/moe/runs/hopper-stage-5/recheck
```

开发期测试PTX曾有动态param地址、mask位宽、与entry重名参数导致ptxas异常，修正测试后重新检查。普通CTA测试发现不能从cluster表取本地shared，已改用当前线程对象，远端才调用阶段3解析。开发日志留在runs的development，不算通过证据。官方网页工具本次503，沿已确认方案、本机CUDA头文件、真实PTX和ptxas核对单位/ABI，未引入新指令变体。
