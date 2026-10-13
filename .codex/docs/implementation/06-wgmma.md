# 阶段 6：WGMMA

## 当前审计结论（2026-10-13）

状态: 进行中

前置：阶段 5 新版验收通过。已有 FP16→FP32、N=128/176、SS/RS 的 16 组数值及
commit/wait 测试；无 swizzle 限制、多 warpgroup/CTA/SM 并发未测、shared/寄存器
成本不完整，使旧整体完成结论不足。后续完整 Dense 不作为本阶段验收前置。

### 本阶段如何补齐

1. 冻结矩阵组件的 shape/SS/RS/layout/transpose/swizzle/pipeline 清单，至少
   覆盖已承诺形式，增加针对性真实 PTX 片段；未支持类型明确列出，不假称通用。
2. 修正 fragment/descriptor 解码、组状态和生命周期；独立 reference 覆盖
   一般 FP16 正负/随机输入、scale、tail、多 K 累加，不只用可精确表示的特殊值。
3. 核查 shader/scoreboard/GMMA 的 warp/warpgroup 计费单位、RAW、寄存器所有权、
   fence/commit/wait；补多组、CTA、SM 竞争和等待顺序，禁止状态相互覆盖。
4. 明确 shared/bank/寄存器成本及统计；验证依赖延迟、吞吐、重叠和容量边界。
   可使用明确的抽象模型，但不能漏计/重复计费，准确度在阶段 14 实测。

### 本阶段可验收条件

- [ ] 冻结形状/布局数值通过，旧 16 组及新随机/边界用例回归通过。
- [ ] 多 warpgroup/CTA/SM 输出、资源与等待顺序通过，非法参与严格拒绝。
- [ ] 支持合同内矩阵路径已接详细时序，shared/寄存器成本有模型说明与证据。
- [ ] 持久记录实际命令、输出/误差、事件、统计与配对版本。

Dense 所有 GEMM 新形状/布局在阶段 10 盘点、11/12 补齐并集成；如使用 mma.sync，
它须单独有功能和 timing 正例，不能由 WGMMA 子集替代。完整 block 与硬件准确度
仍未完成，不从本阶段组件通过推导。

当前执行顺序见 [单卡 Dense 计划](../../plans/single-card-dense.md)。

<details>
<summary>历史实现、测试结果和旧计划（不作为当前验收状态或执行顺序）</summary>

## 目标

实现 128-thread warpgroup 的 WGMMA 功能语义，并连接现有 GMMA latency、II 和
commit/wait timing。本阶段覆盖冻结的 `m64n128k16` 与 `m64n176k16.f32.f16.f16`；
支持 register/shared 和 shared/shared 输入，以及无 swizzle 的 K-major/MN-major descriptor。

## 前置条件与提交

- 阶段 5 旧范围通过；两仓库均在 `research/hopper-ptx`。
- 外层起点：`dd9fb4bad043b371cea107399f6e25bc07136ec3`。
- 内层起点：`a97a8639179dcdc8c7d46303a3b6f090e50f8ba2`。
- 配对内层实现提交：`461bf037459153f101fdd1dcf8d1bbcf77989ca3`。
- 外层实现提交：本报告所在外层提交。该提交同时包含 harness、证据和报告，
  内外层各一个阶段提交；提交正文记录问题、实现、验证和限制。
- `experiments/` 和内层 `.serena/` 是既有用户内容，未纳入阶段提交。

## 实施前计划与实际修改

实施前预期：修正操作数校验，按 NVIDIA fragment 与 descriptor 布局计算矩阵，
要求 128 线程集体参与，复用 GMMA 单元和 commit 队列，在整个 warpgroup 上检查
wait 条件；分别验证数值、同步和周期。以下代码与证据已完成该计划。

源码依据为上述内层实现提交和本报告所在外层提交：

| 位置与符号 | 修改与作用 |
|---|---|
| 内层 `src/cuda-sim/ptx_parser.cc::validate_matrix_operands/add_instruction` | shared/shared 改为 8 个操作数，register/shared 保持 7 个；校验 tuple 长度、类型、重复寄存器、predicate/immediate scale-d、±1 scale、0/1 transpose、0..7 wait；填充 MMA/fence/commit/wait 元数据和共享 commit 标志 |
| 内层 `src/cuda-sim/ptx.l`、`ptx.y`、`ptx_ir.h` | 保存完整 64 位十进制/十六进制整数，溢出报告位置并失败；新增 `fence.proxy.async.shared::cta`，复用统一 shared 存储和已有 membar 排序路径 |
| 内层 `src/cuda-sim/instructions.cc::wgmma_impl/wgmma_shared` | 解码 descriptor、重建 register A，执行 FP16 输入、FP32 累加，更新每线程 64/88 个 accumulator |
| 内层 `src/cuda-sim/ptx_sim.cc::wgmma_threads/wgmma_arrive` | 按 CTA 内线性 tid/128 分组；128 个活线程到达同一 PC 后才执行集体操作并放行；缺线程、提前退出、不同 PC 严格失败 |
| 内层 `src/cuda-sim/ptx_sim.cc::wgmma_check_register` 与 `instructions.cc::get_reg/set_reg` | 维护未提交/已提交组的 accumulator 与 register A 所有权，阻止等待前读取/覆盖结果或覆盖输入；检查 fence 后对参与寄存器的普通写 |
| 内层 `src/cuda-sim/cuda-sim.cc::pre_decode/set_opcode_and_latency/functionalCoreSim::executeWarp` | 建立 descriptor、register A、scale predicate 的 RAW 依赖；MMA 路由到 `SPECIALIZED_UNIT_5_OP`，latency/II=N/2；功能模式等待集体 rendezvous |
| 内层 `src/gpgpu-sim/scoreboard.cc::checkCollision` | 通过 metadata 检查全部 accumulator 的 prior-writer 依赖，避免 8 个普通输出槽的容量限制；异步 accumulator 由 commit/wait 所有权管理 |
| 内层 `src/gpgpu-sim/shader.cc::issue_warp/ptx_wgmma_waiting/writeback/specialized_unit::issue` | 复用 GMMA outstanding/commit 队列，wait 检查同组四个 warp 的 committed-group 数；添加 issue、unit、complete、wait、release 事件及统计；缺少 GMMA 配置明确失败 |
| 外层 `research/hopper-ptx/stage6/` | PTX fixture generator、CPU reference harness、独立时序验证器、正/负向执行脚本及证据 |
| 外层 `research/hopper-ptx/stage3/frontend_wgmma.ptx` | 将早期错误的 5 操作数形式修正为 NVIDIA 的 shared/shared 语法，前端回归通过 |

布局来源：[NVIDIA PTX ISA 的 WGMMA fragment、shared layout 与 descriptor 定义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)。

## 功能、布局与同步逻辑

数据流为 `PTX -> ptx_matrix_metadata/operands -> 128-thread rendezvous -> matrix
reconstruction/FP32 accumulator -> warp_inst_t -> shared shader GMMA pipeline`。

对 warpgroup 内线程 `t` 和 accumulator 索引 `r`：

```text
row = 16*(t/32) + (t%32)/4 + 8*((r%4)/2)
col = 8*(r/4) + 2*(t%4) + r%2
```

register A 的每个 `.b32` 寄存器打包两个 FP16 元素；对元素 `e=0..7`，使用相同
row/col 映射重建 64×16 的 A。descriptor 的 start/LBO/SBO 分别解码低 14 位、
16..29 位、32..45 位，再乘 16。无 swizzle 的字节偏移为：

```text
K-major: 16*(mn%8) + SBO*(mn/8) + 2*(k%8) + LBO*(k/8)
MN-major: 2*(mn%8) + SBO*(mn/8) + 16*(k%8) + LBO*(k/8)
```

每个 descriptor tile 检查地址唯一性和声明的 shared 范围；整个组的 descriptor 与
scale-d 必须一致。scale-d 为 false 时清零旧 D，否则保留 D；scale-a/b 的 ±1
决定输入符号。predicate 按现有模拟器的反向 zero-flag 表示正确解码。

所有 WGMMA 操作在完整组上 rendezvous。MMA 在集体到达后进行功能计算，结果寄存器
仍由异步组持有；普通读写必须由对应 wait 覆盖。commit 收集上一次 commit 后的
操作，wait N 保留最新 N 组。功能模型保守要求最多 8 个未被 wait 覆盖的组；本阶段
验证 0、1、7 的 wait 边界和空 commit。详细模式额外等待四个 warp 的时序队列，
因此 CPU 提前算出数值不会让后续 PTX 提前访问。

## 运行命令

```bash
cmake --build /tmp/hopper-stage3-build --target cudart -j2
research/hopper-ptx/stage6/run.sh /tmp/hopper-stage6-acceptance
```

`run.sh` 在独立目录生成 PTX、复制 SM90_H100 配置，并加入现有 H100 trace 配置的
specialized unit 1..5。配置解析器要求 specialized units 连续启用，所以不能只
启用 unit 5。功能模式和详细模式的日志、生成 PTX 均留在该运行目录。

回归使用同一重建 runtime，实际命令包括：

```bash
make -C research/hopper-ptx/stage2 clean all CXX=/usr/bin/g++ SIM_RUNTIME=/tmp/hopper-stage3-build/libcudart.so
# 在 /tmp/hopper-stage6-regression2 配置目录直接运行 stage2_harness，LD_LIBRARY_PATH=/tmp/hopper-stage3-build
make -C research/hopper-ptx/stage3 clean all
# frontend_probe 的 Python 驱动依次检查 5 个已有正/负向 PTX fixture
make -C research/hopper-ptx/stage4 clean all
research/hopper-ptx/stage4/run.sh /tmp/hopper-stage6-regression4
make -C research/hopper-ptx/stage5 clean all
research/hopper-ptx/stage5/run.sh /tmp/hopper-stage6-regression5
research/hopper-ptx/stage5/run_detailed.sh /tmp/hopper-stage6-regression5-detailed
research/hopper-ptx/stage5/run_multicast.sh /tmp/hopper-stage6-regression5-multicast
git diff --check
git -C gpu-simulator/gpgpu-sim diff --check
```

## 实际结果与证据

构建成功；两仓库 whitespace 检查通过。数值输入随行、列和 K 改变，包含正负数、
零与 FP16 可精确表示的小数；reference 使用独立标量 GEMM，逐元素重建完整输出。

| 用例 | 功能与详细模式检查 |
|---|---|
| ss128 / ss128_hex | shared/shared，scale-d=0；分别以十进制/十六进制构造 64 位 descriptor；各 8,192 元素零误差 |
| ss176 | shared/shared、A/B transpose、双指令累加；11,264 元素零误差 |
| rs128 | register/shared、负 scale、双指令累加；8,192 元素零误差 |
| rs176 / rs176_zero | register/shared、B transpose，predicate true/false 的 scale-d；各 11,264 元素零误差 |
| empty | 空 commit、wait 7 后 wait 0；8,192 个初始 accumulator 保持 0.5 |
| timing128 / timing176 | 独立详细时序场景，三个 MMA 分别 commit，wait 1 后 wait 0；完整输出仍零误差 |

合计 **16 组矩阵检查通过，max_error=0；36 个负向用例通过**。13 个前端错误覆盖
长度、重复寄存器、类型、descriptor 宽度、modifier、scale、wait、predicate 和
64 位整数溢出；11 个功能错误分别在功能/详细模式运行，包括 64 线程、部分活跃
warp、组内提前退出、缺 fence、fence 后修改 D、提前覆盖 A、wait 1 后提前读 D、
无 wait、swizzle、越界 descriptor、组内不一致 descriptor。另验证缺 GMMA 单元
配置明确失败，超时/被 kill 不算负向测试成功。

时序验证器逐个关联 UID 的单元发射与 completion，检查每个 wait release 时四个
warp 的 outstanding 数不超过 N；每个时序场景包含 12 个 warp 指令 completion
和 8 个 wait release。

| N | 配置 latency / II | 单元发射到 writeback | 同一 warp 最小实际发射间隔 | 总 kernel cycles | blocked_checks |
|---|---|---|---|---|---|
| 128 | 64 / 64 | 66 | 64 | 88,923 | 1,494 |
| 176 | 88 / 88 | 90 | 88 | 112,168 | 2,070 |

单元到 writeback 的额外 2 周期来自现有 dispatch/writeback 路径。总周期还包含
shared 初始化、寄存器初始化和全部结果的 global store，不能作为纯 GMMA latency。
`mma/commit/wait` 统计按 128-thread 集体操作计数，`completion` 按 warp 指令计数；
`blocked_checks` 是调度器检查次数，不能当作 stall cycles。

证据已随本阶段提交：

- `research/hopper-ptx/stage6/evidence/acceptance.log`：全部验收结果和时序摘要。
- `research/hopper-ptx/stage6/evidence/timing128.log`、`timing176.log`：实际 UID、周期与等待事件。
- `research/hopper-ptx/stage6/evidence/failures.log`：36 个错误用例的实际诊断与位置。
- `research/hopper-ptx/stage6/evidence/regressions.log`：阶段 2–5 的相关回归结果。

完整原始运行日志在 `/tmp/hopper-stage6-acceptance/`；构建日志在
`/tmp/hopper-stage6-build.log`。阶段 2 回归最初受旧 runtime 与 conda linker 的
GLIBC 版本影响；改为上述系统编译器和明确的当前 runtime 后通过
`n=128 checksum=161408 first=1007 last=1515`。这是验证环境修正，未修改阶段 2 代码。

## 验收与限制

- 单指令、多指令累加与 reference 一致：通过。
- 非完整 warpgroup、错误 tuple 和 modifier 严格失败：通过。
- wait 不提前释放未完成 GMMA：通过，覆盖 wait 0/1、四个 warp 和 register 所有权。
- latency/II 随 N 变化，统计项可解释：通过。

范围仍限于上述 FP16 输入/FP32 累加的两种 shape。swizzle/base-offset/reserved bits
和其他 shape/type/modifier 默认严格失败；predicated WGMMA 暂不支持。尚未验证多个
warpgroup、多 CTA、多 SM 的 WGMMA 并发，也未新增矩阵 shared 读取的独立 LDST
请求或完整 accumulator 寄存器 bank 成本。时序复用既有每 warp GMMA 单元模型，
功能计算与寄存器保护在 128-thread 组上完成；没有 H100/H200 实测，不能宣称
硬件周期准确度或完整 FA3 kernel 已运行。

阶段 6 旧范围通过，可进入阶段 7 的 cluster/DSMEM；本次未开始后续阶段。

</details>
