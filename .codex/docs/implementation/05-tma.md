# 阶段 5：TMA 与异步内存

## 当前审计结论（2026-10-13）

状态: 进行中

前置：阶段 4 新版验收通过。已有 logical descriptor 的 1D/2D load、OOB、LDST
completion、四 CTA mask=5 multicast 正例；固定 tile/barrier/绑定的旧测试不足以
覆盖并发与实际布局。原生 tensor-map、store/bulk 等不能据此声称完成。

### 本阶段如何补齐

1. 冻结本阶段 TMA 组件合同，至少保留现有 1D/2D load/OOB/multicast 支持；记录
   rank、stride、box、swizzle、barrier 和 shared 地址。需要真实 PTX/布局时在
   本阶段先做最小导出，不依赖阶段 10 完成；不猜 opaque descriptor 二进制布局。
2. 在 tma.h/instructions.cc 补合同中所需地址/布局和 logical descriptor 导入，
   逐元素对照独立 reference。校验生命周期、地址替换、越界及非法描述符。
3. completion record 按 kernel/CTA/操作独立，支持多 tile、多 barrier 在途、
   缓冲重用、OOB/tail 和 response 乱序；数据写入 shared 后再完成 transaction。
4. multicast 覆盖多 mask、慢目标和计数守恒；复用实际 LDST 路径检查 issue、
   response、数据可见、completion、wait 顺序，不用功能写入伪装详细完成。

### 本阶段可验收条件

- [ ] 冻结合同的所有 descriptor/layout 与 reference 一致，逻辑导入入口可复现。
- [ ] 并发多 tile 无丢失/串扰，字节只计一次，wait 不早于数据可见。
- [ ] local/detailed/mask=5 旧正例、多 mask/压力/非法输入回归通过，持久证据完整。

公开 tensor-map API 归阶段 9，Dense 新增 TMA 形式归 11/12，原版 FA3 特有的
高维/store/bulk/prefetch 等归阶段 15（如 Dense 提前需要，则由 11/12 补齐）。
这些项单列未完成及成功用例，不因本阶段组件验收自动变为支持；H100 成本归 14。

当前执行顺序见 [单卡 Dense 计划](../../plans/single-card-dense.md)。

<details>
<summary>历史实现、测试结果和旧计划（不作为当前验收状态或执行顺序）</summary>

当前结果：单 CTA detailed LDST 完成路径及四 CTA `mask=5` multicast 已验证，
目标写入/完成各 2 次；逻辑 descriptor 仍不是原生 CUtensorMap。
先读[解阻结果与最终证据](#解阻结果与最终证据)和[当前总览](README.md)。
下方“初版”记录及末尾原计划保留历史，不代表当前仍被阻塞。

## 前置条件与预期结果

阶段 4 旧范围通过；起点外层 `f5ccda3b7df7d74c3841a041fe63b0ed370647de`，内层实际 HEAD 为 `3bdfa94f236aee5f14b1b4f418e3da92010c27d6`（阶段 4 报告中的 SHA 存在转录错误，以 Git 为准）。本阶段先覆盖 tiled global-to-shared tensor load，验证 1D/2D 地址、越界零填充、shared 可见性、异步 completion 和 multicast 目标，并严格拒绝未知 descriptor 与未实现形式。

CUDA tensor map 的二进制对象是 opaque；文本 harness 使用显式注册的逻辑 descriptor 字段，不假造 CUDA 二进制布局。是否可以验收，要分别记录功能、同步、动态元数据、实际时序路径验证的结果。

本报告所在外层提交包含本阶段代码、harness、reference、失败 fixture 和验收证据；配对内层实现提交为 `a97a8639`。

## 目标

从 PTX descriptor 生成与 SASS trace 等价的 TMA 地址、字节计数、OOB 和 multicast
元数据，并完成 shared-memory 语义。

## 操作

- 解析 tensor-map descriptor。
- 计算 global-to-shared 地址展开和边界填充。
- 计算 byte count、OOB byte count 和 mbarrier 关联。
- 实现 shared 写侧、异步完成和 multicast CTA mask。
- 将结果填入 `warp_inst_t`，复用现有 LDST/mbarrier timing。

## 验收

- 一维、二维和越界 descriptor 与 reference 一致。
- TMA completion 发生后 mbarrier 才可满足等待。
- shared-memory 写入和 multicast 目标正确。
- 发起、等待、完成和访存统计可以区分。

## 初版实现与证据（历史，详细路径后续已补齐）

- 内层 `src/cuda-sim/ptx.l`、`ptx.y`、`ptx_parser.{h,cc}` 和 `opcodes.def` 严格识别 `cp.async.bulk.tensor.1d`--`.5d` 的 global-to-shared `mbarrier::complete_tx::bytes` 形式、multicast/cache hint 修饰符和 L2 prefetch；store、未知 rank、错误坐标元数和未注册 descriptor 均失败。
- `src/cuda-sim/tma.h` 定义显式逻辑 descriptor：rank、element bytes、dimensions、byte strides、box。地址展开检查 rank、溢出、负坐标和每元素边界；OOB 元素填零，完成事件在 shared 写入后发布。
- `instructions.cc` 按 PTX 发起线程提交 CTA 异步队列；harness 用 lane 0 的 predicate 限定一次发起。`try_wait` 推进队列，随后更新 mbarrier transaction。`cuda-sim.cc` 将实际地址、in-bounds/OOB byte count、mbarrier 地址和 mask 填入 `warp_inst_t` 的既有 TMA 字段。原功能版曾对详细模式严格报错；本次续做已接入实际 LDST issue/response/writeback，见后面的最终证据。
- `research/hopper-ptx/stage5/tma_reference.cc` 验证 1D、二维带 padding、负坐标、全 OOB 和 mask=5 的两个目标；`tma_harness.cc` 验证二维 tile `[13,0,23,0]`、16 bytes、8 OOB bytes 和 wait 顺序；`tma_frontend.cc` 验证四条 TMA 元数据。

## 初版命令与结果（历史）

```text
cmake --build /tmp/hopper-stage3-build --target cudart -j2
[100%] Built target cudart

research/hopper-ptx/stage5/run.sh /tmp/hopper-stage5-check
FRONTEND_RESULT status=PASS tma_instructions=4
TMA_RESULT status=PASS tile=13,0,23,0 bytes=16 oob=8 wait_before=0 wait_after=1
REFERENCE_RESULT status=PASS one_d=1 two_d=1 padded=1 negative_oob=1 all_oob=1 multicast_mask=5
EVENT_RESULT issue=1 global_reads=2 shared_writes=2 completion=2 wait_before=0 wait_after=1
NEGATIVE_RESULT status=PASS file=tma_bad_rank.ptx
NEGATIVE_RESULT status=PASS file=tma_bad_store.ptx
NEGATIVE_RESULT status=PASS file=tma_bad_rank_operands.ptx
NEGATIVE_RESULT status=PASS file=unregistered

research/hopper-ptx/stage4/run.sh /tmp/hopper-stage4-run8
SYNC_RESULT status=PASS n=32 first=10 last=10
FRONTEND_RESULT status=PASS mbarrier_instructions=9
STATE_RESULT status=PASS phase=2 pending=0 tx=0 expected=0
NEGATIVE_RESULT status=PASS file=mbarrier_bad_modifier.ptx
NEGATIVE_RESULT status=PASS file=mbarrier_bad_operands.ptx
NEGATIVE_RESULT status=PASS file=mbarrier_cluster_wait.ptx
```

原功能版证据目录为 `/tmp/hopper-stage5-final3`；最终续做证据为 `/tmp/hopper-stage5-final/`；源码证据集中在 `gpu-simulator/gpgpu-sim/src/cuda-sim/tma.h`、`instructions.cc`、`cuda-sim.cc` 和 `ptx_parser.cc`。官方 PTX 语法参考：<https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>。

## 解阻结果与最终证据

续做基线外层 `ad6b3774b8d22d56b746b490bd25622aad8034de`，内层 `24f8e5ab8cec1c24b36e84090c367d0af63dd6e8`；用户新增计划和既有 `experiments/`、内层 `.serena/` 在开始时已存在。基线命令、工作树状态和 functional PASS 保存于 `/tmp/hopper-stage5-baseline/` 与 `research/hopper-ptx/stage5/artifacts/baseline.log`。最终配对内层完整 SHA 为 `a97a8639179dcdc8c7d46303a3b6f090e50f8ba2`。通过 amend 原阶段提交保留两仓库各一个阶段5提交。

单 CTA detailed 2D 和非 multicast local 形式均通过；2D 事件顺序为 cycle 4286 issue/OOB 8 bytes、4287 两个 4-byte global request、4293 wait 失败、4532/4533 global response、4533 shared write 与 in-bounds 8-byte completion、4556 wait 成功。OOB 只计一次，每个 selected target 完成的总 transaction 为 16 bytes。全 OOB 在 issue 发布零 tile 后完成 16 bytes，不发 global request；其 issue 前 wait 为0，issue 后 wait 为1。

联调中遇到三类失败，均已修复：带输出 mbarrier 写回错误进入普通 load pending-register 分支；全 OOB 导致末尾预取取到空指令后解引用；阶段4旧 fixture 的 predicate/指令数断言与当前语义不一致。最终回归 PASS，失败日志留于 `/tmp/tma-single*.err`、`/tmp/tma-all_oob*.err` 和 `/tmp/stage5-stage4-regression*.log`。

目标身份由 harness 显式注册 kernel/CTA/cluster/rank，mbarrier.init 的实际 CTA shared writer 在 SYNCS writeback 时绑定到已有 cluster lookup，rank 地址使用现有的高8位/lower24位 offset 语义。四 CTA 各有独立 writer，不从 shader slot 推测身份。目标验证发生在 TMA issue 前；空mask、未知rank、重复target、已注册但未启动CTA、barrier offset不匹配、未注册descriptor、非法descriptor、aliasing strides、详细3D和prefetch均严格失败。

详细模式范围限定自然对齐、无alias的1D/2D tiled load、一个注册tile/mbarrier、显式CTA绑定；3D以上、详细prefetch、cluster-addressed mbarrier、真实runtime cluster调度/DSMEM仍严格不支持或留到后续阶段。这里验证实际PTX跨CTA写回语义，未宣称multicast互连带宽或硬件周期精度。

续做实现把动态元数据和功能写入分离：`tma_impl` 在 detailed 模式只解析 descriptor、建立 completion record 并验证所有目标；`warp_inst_t::generate_mem_accesses` 生成精确 global sector 队列；LDST issue 对 OOB bytes 只完成一次，global response 在 writeback 中填充 record，随后由每个 CTA writer 写入 shared tile 并完成对应 mbarrier。completion record 随 `warp_inst_t` 和 `mem_fetch` 传递，没有使用全局“最近一次 TMA”状态。

`research/hopper-ptx/stage5/tma_detailed.ptx` 运行单 CTA 2D、单 CTA 1D、全 OOB 和四 CTA multicast；`run_detailed.sh`、`run_multicast.sh` 固定 detailed 配置并保存完整日志。四 CTA 用 `mask=5`，rank 0 和 rank 2 的 shared tile 为 `[13,0,23,0]`，rank 1、3 保持 `0xcccccccc`；选中目标 phase=1、tx=0、pending=1，未选中目标 phase=0、tx=16、pending=0。

关键证据：

```text
cmake --build /tmp/hopper-stage3-build --target cudart -j2
[100%] Built target cudart

research/hopper-ptx/stage5/run.sh /tmp/hopper-stage5-final/functional
FRONTEND_RESULT status=PASS tma_instructions=4
TMA_RESULT status=PASS tile=13,0,23,0 bytes=16 oob=8 wait_before=0 wait_after=1
REFERENCE_RESULT status=PASS one_d=1 two_d=1 padded=1 negative_oob=1 all_oob=1 multicast_mask=5
PTX_EVENT_RESULT status=PASS issue=1 completion=1 global_reads=2 shared_writes=1 wait_failure=32 wait_success=32 metadata=PASS
NEGATIVE_RESULT status=PASS file=unregistered

research/hopper-ptx/stage5/run_detailed.sh /tmp/hopper-stage5-final/detailed
VISIBILITY_RESULT status=PASS before=cccccccc wait_before=0 wait_after=1
METADATA_RESULT status=PASS tile_bytes=16 in_bounds=4 oob=12 barrier=10 mask=1
DETAILED_RESULT status=PASS scenario=one_d mask=1 cycles=22467 issue=1 requests=1 responses=1 writes=1 completions=1 waits_failed=7 waits_success=1
METADATA_RESULT status=PASS tile_bytes=16 in_bounds=0 oob=16 barrier=10 mask=1
DETAILED_RESULT status=PASS scenario=all_oob mask=1 cycles=21667 issue=1 requests=0 responses=0 writes=1 completions=1 waits_failed=1 waits_success=2

research/hopper-ptx/stage5/run_multicast.sh /tmp/hopper-stage5-final/multicast
CTA_RESULT status=PASS rank=0 selected=1 phase=1 tx=0 pending=1
CTA_RESULT status=PASS rank=1 selected=0 phase=0 tx=16 pending=0
CTA_RESULT status=PASS rank=2 selected=1 phase=1 tx=0 pending=1
CTA_RESULT status=PASS rank=3 selected=0 phase=0 tx=16 pending=0
VISIBILITY_RESULT status=PASS before=cccccccc wait_before=0 wait_after=1
METADATA_RESULT status=PASS tile_bytes=16 in_bounds=8 oob=8 barrier=10 mask=5
DETAILED_RESULT status=PASS scenario=multicast mask=5 cycles=22467 issue=1 requests=2 responses=2 writes=2 completions=2 waits_failed=47 waits_success=2
NEGATIVE_RESULT status=PASS scenario=empty
NEGATIVE_RESULT status=PASS scenario=unknown
NEGATIVE_RESULT status=PASS scenario=not_started
NEGATIVE_RESULT status=PASS scenario=offset
NEGATIVE_RESULT status=PASS scenario=duplicate
NEGATIVE_RESULT status=PASS scenario=unregistered
NEGATIVE_RESULT status=PASS scenario=bad_descriptor
NEGATIVE_RESULT status=PASS scenario=overlap
NEGATIVE_RESULT status=PASS scenario=high_rank
NEGATIVE_RESULT status=PASS scenario=prefetch

research/hopper-ptx/stage4/run.sh /tmp/hopper-stage5-final/stage4-regression
SYNC_RESULT status=PASS n=32 first=10 last=10
FRONTEND_RESULT status=PASS mbarrier_instructions=9
STATE_RESULT status=PASS phase=2 pending=0 tx=0 expected=0
```

完整命令日志和配置在 `/tmp/hopper-stage5-final/`；精简证据和失败诊断在 `research/hopper-ptx/stage5/artifacts/`。阶段报告与实现配对提交：外层本报告所在提交，内层 `a97a8639`。

## 限制与下一阶段入口

- descriptor 是研究 harness 的显式注册对象，不是 CUDA `CUtensorMap` 二进制 ABI；真实 runtime 注册留给阶段9。
- 功能模式仍使用显式逻辑 descriptor 和 CTA-local queue；详细模式额外运行四个 CTA 的实际 mask=5 multicast。真实 CUtensorMap ABI、runtime module/launch API 和硬件校准仍不在本阶段范围。
- 详细模式已接入每条指令的 TMA completion record、精确 global sector access、LDST response/writeback、shared writer 和 mbarrier completion；未支持的详细形式继续严格失败。
- 原阻塞项已由本报告后面的“解阻计划（阶段 5 续）”完成：单 CTA detailed issue/response/writeback、实际四 CTA mask=5 multicast、shared 可见性、精确 transaction 计数和严格负例均通过。
- 没有真实 H100/H200 测量，不作周期误差或硬件准确度结论。

## 解阻计划（阶段 5 续）

执行结果：本节所有退出条件已经完成，三层验收与阶段4回归通过，状态从 `阻塞` 经 `待验收` 重新验收为 `旧范围通过`。以下保留用户原计划作为实施和验收依据。

本计划只解决阶段 5 当前的两个阻塞项：PTX TMA 的详细 LDST 时序，以及 PTX
跨 CTA multicast 的实际执行。阶段状态继续保持 `阻塞`；只有本节的所有退出条件
满足后，才可改为 `待验收`，并重新运行完整验收协议。阶段 6 在此之前不得开始。

### 1. 固定实现边界和基线

开始修改前，记录外层和内层仓库的实际 HEAD、工作树状态和构建目录。保留当前
显式注册的逻辑 descriptor，不在本阶段引入 `CUtensorMap` 二进制 ABI 或 runtime
module/launch API。继续严格拒绝未知 rank、未注册 descriptor、非法 mask 和不支持
的 TMA store 形式。

基线命令：

```bash
git rev-parse HEAD
git -C gpu-simulator/gpgpu-sim rev-parse HEAD
git status --short
git -C gpu-simulator/gpgpu-sim status --short
research/hopper-ptx/stage5/run.sh /tmp/hopper-stage5-baseline
```

基线证据保存到 `/tmp/hopper-stage5-baseline/`，包括命令输出、配置文件和失败日志。

### 2. 先完成单 CTA 详细 TMA 路径

#### 2.1 分离“动态元数据生成”和“功能写入”

调整 `src/cuda-sim/instructions.cc:tma_impl` 和
`src/cuda-sim/cuda-sim.cc:ptx_exec_inst` 的职责：

1. 两种模式都解析 descriptor、坐标、地址、in-bounds/OOB byte count、barrier 和
   multicast mask。
2. functional 模式继续使用 `ptx_tma_complete`，负责 reference 所需的 global read、
   shared write 和 barrier completion。
3. detailed 模式不直接执行 global read，不调用 functional `tma_submit`，也不把
   completion 提前到 PTX 指令执行时刻；只把 TMA 动态元数据写入 `warp_inst_t`，并
   生成供 LDST 使用的 access queue。
4. 详细模式遇到无法表示的 descriptor、地址或目标时立即失败，不能退化成近似的
   本地写入。

需要保持的字段包括：每个 in-bounds 元素地址、元素大小、总 tile bytes、OOB bytes、
barrier 地址、源 cluster rank 和 multicast mask。若 access queue 需要额外的 tile
    布局信息，应增加明确的 TMA side-band metadata，而不是从 sector 地址反推。

#### 2.2 把 TMA 访问接入现有 LDST issue

在 `shader.cc` 的 LDST issue 路径验证以下不变量：

- TMA 指令产生 global read access，并进入现有 interconnect/cache 路径；
- OOB bytes 在 issue 阶段只计入一次 transaction completion；
- in-bounds bytes 留给 global response/writeback；
- `inc_tma_load_req`、response queue 和 warp completion 计数保持一致；
- TMA 指令不会被普通 register-load 的输出依赖错误阻塞或错误释放。

如果当前 `mem_access_t` 无法携带 tile 写回布局，则扩展它或增加与 `warp_inst_t`
关联的 TMA completion record，至少包含目标 shared 地址、元素地址顺序、OOB zero-fill
信息和 barrier/mask。该 record 必须随 mem_fetch 到达 writeback，不能使用全局的
“最近一次 TMA”状态代替。

#### 2.3 在 writeback 完成 shared 写和 barrier completion

扩展 `shader.cc:ldst_unit::writeback` 及其辅助函数：

1. global response 返回后，先按 tile 顺序生成 shared tile；OOB 元素写入零。
2. 对源 CTA 写入本地 shared memory；写入成功后再完成源 CTA barrier 的 in-bounds
   transaction。
3. completion 的字节数使用精确的 in-bounds byte count，不能使用 sector-rounded
   access size。
4. 所有目标写入和对应 transaction completion 完成后，TMA load request 才减少，
   warp 才能继续通过 `try_wait`/依赖检查。
5. shared 写入或 barrier 查找失败时报告 CTA、cluster rank、barrier 地址和 mask，
   并终止该测试；不得静默丢弃数据。

单 CTA 阶段的最低验收序列是：issue → wait 失败 → global response → shared write
可见 → barrier completion → wait 成功。测试必须证明 `test_wait` 不会推进传输，只有
实际的 detailed response/writeback 才能使等待完成。

### 3. 实现实际跨 CTA multicast

#### 3.1 建立 cluster CTA 目标注册

在 PTX detailed harness 中启动至少 4 个 CTA，并为每个 CTA 注册：

- cluster id 和 cluster rank；
- 本地 shared-memory tile 起始地址；
- 对应 mbarrier 地址和 offset；
- TMA completion writer。

目标查找应复用现有 cluster/mbarrier lookup 的 rank 语义。不得只复用 barrier 指针
而忽略目标 CTA 的 shared memory；每个目标必须有独立的 writer。

#### 3.2 完成 multicast 数据路径

扩展 TMA completion record，使一个 global response 可以向 mask 选中的多个 CTA
写入同一 tile：

- mask 中的 rank 必须全部已注册，否则在任何 global read 或 shared write 前失败；
- 只向 mask 选中的 rank 写入；未选中的 CTA 保持原值；
- 每个选中 CTA 的 shared write 完成后，才完成其对应 barrier transaction；
- 源 rank 不得被重复 completion；
- OOB zero-fill 对所有目标一致。

先支持一维和二维 load、固定 tile、单个 mbarrier；三维以上 descriptor 和真实
runtime descriptor 仍留在原有边界内，不借本阶段扩展范围。

#### 3.3 multicast 负例

新增或扩展 fixture，至少覆盖：空 mask、未知 rank、重复 target、目标 CTA 未启动、
barrier offset 不匹配。每个负例都必须在写入前失败，并输出可定位的诊断信息。

### 4. 测试和证据分层

新增三个相互独立的检查层，避免 reference PASS 掩盖 execution-driven 缺口：

1. **功能层**：继续运行现有 `tma_reference` 和 functional harness，验证地址展开、
   padding、负坐标、全 OOB、零填充和 mask 选择。
2. **详细时序层**：使用 detailed 配置运行单 CTA 和多 CTA harness，记录 issue、
   global request、response、shared write、completion、wait 成功/失败以及每个
   CTA 的 rank。
3. **负例层**：运行所有非法 PTX、未知 descriptor、未注册 target 和不支持的模式，
   确认严格失败。

建议命令：

```bash
cmake --build /tmp/hopper-stage3-build --target cudart -j2
make -C research/hopper-ptx/stage5 clean all
research/hopper-ptx/stage5/run.sh /tmp/hopper-stage5-functional
research/hopper-ptx/stage5/run_detailed.sh /tmp/hopper-stage5-detailed
research/hopper-ptx/stage5/run_multicast.sh /tmp/hopper-stage5-multicast
```

若暂时不新增两个脚本，必须在阶段报告中写出等价的完整命令、配置和环境变量，不能
只记录 frontend 或 reference 命令。所有输出保存到 `/tmp/hopper-stage5-final/`，
并把关键日志路径写入本报告。

### 5. 验收断言

阶段 5 重新验收前，必须同时满足：

- 单 CTA detailed TMA 真正经过 LDST issue 和 global response/writeback；
- shared tile 在 barrier completion 之前不可被后续 PTX 观察到；
- OOB bytes 只 completion 一次，in-bounds bytes 在 writeback 精确 completion；
- `wait_before=0`、`wait_after=1`，且 after 状态由 detailed completion 触发；
- 至少一个非 1 的 multicast mask 在 PTX detailed 路径实际执行成功；
- mask 选中的 CTA shared 内容正确，未选中的 CTA 未被修改；
- 选中 CTA 的 barrier phase/count 正确，未选中的 barrier 不被错误推进；
- 非法 target、非法 descriptor 和未实现形式严格失败；
- 功能、同步和动态元数据日志彼此一致；
- 外层和内层阶段提交各只有一个阶段提交，报告、代码、脚本和证据在对应提交内；
- 报告记录外层 commit、内层 commit、实际命令、结果摘要、失败日志和证据目录。

### 6. 停止条件和退出路径

遇到以下任一情况，保持阶段 `阻塞`，不要开始阶段 6：

- detailed 模式仍在 `tma_impl` 直接抛出未实现错误；
- 测试只通过 functional queue 或 reference，无法证明 LDST response/writeback；
- multicast 只更新 barrier，没有实际 shared 写入；
- OOB 或 sector rounding 导致 transaction 计数不能归零；
- 目标 CTA、cluster rank 或 barrier lookup 依赖未验证的近似映射；
- 任一负例静默接受或在写入后才报错。

所有验收项通过后，将本节结果补入“实际实现与证据”和“命令与结果”，把状态改为
`待验收`，再由阶段验收把状态改为 `旧范围通过`。在此之前，阶段 6 的 WGMMA 工作只可
作为独立设计讨论，不能修改阶段 6 实现代码或启动其验收。

</details>
