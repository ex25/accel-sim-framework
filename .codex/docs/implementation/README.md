# 当前实现、结果与能力边界

核对日期：2026-10-13（重新审计状态）；2026-10-10 的复验另列为历史记录。
本文是阶段结果的阅读入口；各阶段报告首页为当前状态，折叠部分为历史实现与结果。
阶段 1–9 共九个实施阶段，另有阶段 0 的基线准备，共十份编号报告。

## 结论

目前完成的是 **Hopper PTX 的受限指令功能/时序模型、验证 harness，以及文本 PTX
Driver module/launch 子集**。**尚无完整 Dense block 执行证据**；普通 attention
也只有 functional 全输出及 detailed 单 CTA 诊断，不能当作全 grid 时序结果。

此前审计将阶段 **3–9 撤下整体验收，恢复“进行中”**，每份报告首页均补了缺口、实现步骤
和重新验收条件。阶段 **0–2 保留原窄范围验收**（基线、冻结输入、最小 harness）。
已有 PASS 原始证据不改写，但不再由它们推导整阶段或完整应用完成。

同日的阶段 3 实现已补真实执行并通过当前组件合同：**阶段 3 已验收，4–9 仍进行中**。
42 个逐 lane ldmatrix 数值 case、前端/参数/label/普通 kernel/WGMMA 回归全部通过。
[最终 gate](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen/gate.json)
和 [阶段 3 报告](03-frontend-foundation.md) 保存配对版本、完整输出和失败记录。
ldmatrix detailed timing、mma.sync arithmetic 与完整 Dense 仍未完成；本会话不实施阶段 4。

特别注意：阶段 8 的完整 FA3 **解析**成功，但多类指令**执行**仍明确报错；阶段 9
的文本 PTX kernel 已经通过公开 Driver API 加载和运行，但原生框架集成没有完成。

实现前静态审计的代码版本（历史）：

- 外层 `research/hopper-ptx`：`f9faa73a4f7907db88db450e54000a502ecf39d2`。
- 内层 `research/hopper-ptx`：`f4fe783ac7fc4f1cdaa9efa72a3575537d4fba36`。
- Python 使用 Conda 环境 **`in-switch`**，不是把项目目录当成 Python 环境；已核对
  `/home/wangjz/miniconda3/envs/in-switch/bin/python3`，Python 3.11.16、NumPy 2.4.6。
- 2026-10-13 的该次审计仅做源码/报告/文件检查，没有运行模拟器或采集硬件；旧数值来自
  已保存的旧结果，2026-10-10 的重新执行记录见后文，不是本轮重新执行。

当前阶段 3 新证据为同日重新构建/执行：外层是本页所在阶段提交，内层
`eca83b30959381ee14bb26384a0c4f01c3d35c01`，两仓库均为 `research/hopper-ptx`。
runtime 为 `/tmp/hopper-stage3-acceptance-build/libcudart.so`，SHA256
`7ce26b6fbec28ff680efcf1f18b846c35856cf82b1e66e4444d97614ffe3552c`。
测试时外层入口 HEAD `63d6e475` 加本次修改，内层无代码 dirty；提交后只保留外层
`experiments/` 和内层 `.serena/` 未跟踪。实际工具版本与 patch/hash 见阶段 3 报告。

## 当前阶段状态与缺口

| 阶段 | 当前状态 | 还缺什么 / 验收入口 |
| --- | --- | --- |
| 0–2 | 已验收（仅原窄范围） | 不包括 Hopper 完整执行；新的 Dense 规格由阶段 10 冻结 |
| [3](03-frontend-foundation.md) | 已验收（当前组件合同） | 12 种 ldmatrix 形式/42 数值 case 全 lane 零误差；详细时序与 Dense 新需求仍分别归 12、10/11 |
| [4](04-mbarrier-and-sync.md) | 进行中 | detailed 已有受限 TMA binding 路径；补多 barrier/phase 与真实 pipeline，修正旧“完全没有 timing”的描述 |
| [5](05-tma.md) | 进行中 | logical descriptor 子集有效；补实际布局/descriptor 入口、多 tile 在途和所需 store/bulk 等 |
| [6](06-wgmma.md) | 进行中 | 两种无 swizzle shape 有结果；补正式 GEMM 需求、多组/CTA/SM 并发和资源成本 |
| [7](07-cluster-and-dsmem.md) | 进行中 | setmaxnreg/occupancy、detailed 远程/cluster 和组件级 launch 元数据未完成 |
| [8](08-a800-and-model-validation.md) | 进行中 | 补普通 attention 全 grid 组件验证及持久证据；Dense/H100/原版 FA3 分别归 11–15 |
| [9](09-runtime-integration.md) | 进行中 | 补真实 Python 算子/短链、API、资源与 stream；完整 block 归 13 |
| [10](10-dense-contract.md) | 未开始 | 冻结真实 Dense 规格、kernel/PTX/launch、容差及能力清单 |
| [11](11-dense-functional.md) | 未开始 | 三个正式 shape 的所有算子与完整 block 功能执行 |
| [12](12-dense-timing.md) | 未开始 | 全 grid 详细时序、资源、并发与约一天预算 |
| [13](13-dense-python-runtime.md) | 未开始 | 真实 Python block 功能/详细运行，先 adapter 后所需 shim |
| [14](14-dense-h100-validation.md) | 未开始 | H100 校准、独立验证与完整单卡 baseline |

当前执行顺序为 [单卡 Dense 计划](../../plans/single-card-dense.md) 中的
**3→4→5→6→7→8→9→10→11→12→13→14**。阶段 3 已完成并报告阶段 4 入口，下一
实现会话从阶段 4 开始。组件与后续应用明确分工，不能跳过未验收阶段，也不能用尚未开始
的完整应用结果作为前序组件的循环验收条件。

## 2026-10-13 实现前审计证据与局限（历史）

- [审计 JSON](evidence/single-card-audit-20261013.json)：阶段 8 的 32 个文件、阶段 9
  的 20 个文件、阶段 1 的 Q/K/V 与冻结 PTX 共 **56/56 hash 一致**。这证明保存
  文件未损坏，不证明测试覆盖完整。
- `stage8/frozen-readiness.json` 为 `parser_only=true`；`run_model.sh` 明确对
  detailed 传入 rows=1/heads=1，结果每个 case 128 元素；functional S=128 是
  131072 元素。二者不是同一工作量。
- 当前 `instructions.cc` 的 `setmaxnreg_impl`、`elect_impl`、
  `prefetch_tensormap_impl`、`hopper_frontend_only_impl` 仍抛未实现；
  `cuda_driver_modern.cc` tensor-map/cluster occupancy 仍返回 NOT_SUPPORTED。
- 检查 `stage3` 及后续回归的 ldmatrix 调用发现现有 probe 只解析；报告引用的
  `stage2_kernel.ptx` 不包含 ldmatrix，不能作为矩阵加载数值验收。
- `research/hopper-ptx/` 未找到 norm/RoPE/SwiGLU 完整 Dense 实现与结果；新的
  10–14 是本次新增的实施计划。`in-switch` Python 为 3.11.16，torch/triton 未安装。

该次审计未重建 runtime、重跑历史 suite 或租用 H100；不把静态审计包装成动态验证。
后续阶段 3 的新动态证据单独列于首页与其报告，不改写上述审计文件。

## 各阶段实际交付

“负例通过”表示不支持/非法输入被正确拒绝，不表示该功能已实现。

| 阶段 / 报告 | 实际实现与结果 | 可直接查看的证据 | 不能据此声称 |
|---|---|---|---|
| [0 基线](00-baseline-and-branches.md) | 配对分支、文档入口、阶段门控 | 报告中的 Git 基线 | 已有 Hopper 执行能力 |
| [1 workload 冻结](01-workload-freeze.md) | FA3 PTX、Q/K/V、shape 和工具版本冻结；`sm_90a` PTX 由 ptxas 编译成功 | [输入清单](../../../research/hopper-ptx/stage1/artifacts/inputs/inputs.json)、[ptxas 输出](../../../research/hopper-ptx/stage1/artifacts/ptxas.log) | FA3 已被模拟器执行 |
| [2 PTX harness](02-ptx-harness.md) | 最小 kernel 解析→执行→写回；`n=128 checksum=161408 first=1007 last=1515` | 阶段报告命令；[后续回归证据](../../../research/hopper-ptx/stage6/evidence/regressions.log) | 原生 CUDA/PyTorch 运行时完整支持 |
| [3 前端](03-frontend-foundation.md) | `sm_90a`、矩阵 metadata/tuple/type；12 种 ldmatrix 实际执行，42 数值 case 全输出零误差 | [新 gate](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen/gate.json)、[证据校验](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/verification.json)、阶段报告 | ldmatrix timing、mma.sync 算术、所有类型或完整 Dense 已实现 |
| [4 mbarrier](04-mbarrier-and-sync.md) | CTA phase/arrival/transaction/token/parity 功能状态；32 线程输出 10，phase=2/pending=0/tx=0 | [阶段 9 保存的回归](../../../research/hopper-ptx/stage9/evidence/stage4-regression.log) | 全部 cluster-addressed barrier timing 完成 |
| [5 TMA](05-tma.md) | 逻辑 descriptor、地址/OOB、LDST completion/shared 写回；四 CTA `mask=5` 产生 2 次目标写和 completion | [detailed](../../../research/hopper-ptx/stage5/artifacts/detailed.log)、[multicast](../../../research/hopper-ptx/stage5/artifacts/multicast.log)、[负例](../../../research/hopper-ptx/stage5/artifacts/negative.log) | 原生 CUtensorMap ABI、TMA store 或完整硬件时序已实现 |
| [6 WGMMA](06-wgmma.md) | 128-thread 组、FP16→FP32、N=128/176、SS/RS、commit/wait；16 组矩阵检查零误差，36 个负例通过 | [验收](../../../research/hopper-ptx/stage6/evidence/acceptance.log)、[时序](../../../research/hopper-ptx/stage6/evidence/timing128.log) | swizzle、任意 shape/type、多 warpgroup 并发或硬件周期准确度 |
| [7 cluster/DSMEM](07-cluster-and-dsmem.md) | rank/address、functional 远程 shared/barrier/fence、detailed metadata；报告记录 15 个正向/25 个负向场景 | [阶段 9 保存的回归](../../../research/hopper-ptx/stage9/evidence/stage7-regression.log) | detailed 远程 DSMEM/barrier/fence 或 setmaxnreg 资源模型完成 |
| [8 A800/模型验证](08-a800-and-model-validation.md) | 8 个普通 attention 数值场景、真实 A800 trace/replay、7 项硬件 counter、原 FA3 parser readiness；聚合 PASS | [最终 results.json](../../../research/hopper-ptx/stage8/evidence/acceptance-20261010/results.json)、[验证日志](../../../research/hopper-ptx/stage8/evidence/acceptance-20261010/verification.log) | 实际跑通原 FA3，或验证 H100/H200 硬件精度 |
| [9 runtime 子集](09-runtime-integration.md) | 文本 PTX load→function→launch→sync→unload；functional/detailed 各 82 checks、128 数值通过 | [functional](../../../research/hopper-ptx/stage9/evidence/runtime-functional.log)、[detailed](../../../research/hopper-ptx/stage9/evidence/runtime-detailed.log) | PyTorch/Triton 原生执行、cubin/JIT、原生 cluster/TMA launch 已完成 |

## 最值得看的数值结果

### Attention 对照（阶段 8 保存的最终证据）

这是普通 CUDA-core attention 和对应手写文本 PTX，与 FA3 为相同数学问题；不是 FA3
本体。S=128 的完整 functional 输出有 131,072 个 FP16 元素，CPU FP64 reference
最大绝对误差为 `2.4430074510806143e-4`。A800 对应最大误差为
`2.4440458503072193e-4`；两者直接比较最大差异 `0.00048828125`，26 个元素存在
FP16 舍入差异。统一容差为 `abs_error <= 0.002 + 0.002*abs(reference)`。

| 序列长度 S | A800 完整 grid 中位 kernel 时间（µs） | PTX detailed 单 CTA 周期 |
|---|---:|---:|
| 64 | 41.984001 | 218905 |
| 128 | 141.311996 | 370567 |
| 256 | 556.031982 | 708229 |

两列工作量不同，只能分别看趋势；不能相除得到加速比或硬件校准倍率。

同一 A800 单 CTA trace：`134375` scalar instructions，SM80 模型回放也是
`134375` instructions、`32979` cycles。回放验证动态 trace 可重放，不提供数值
输出正确性或 A800 硬件周期误差证明。

管理员采集的 S=128 [counter 校验结果](../../../research/hopper-ptx/stage8/evidence/acceptance-20261010/profiler-csv-validation.json)：
DRAM read `803712 B`、DRAM write `0 B`、active warps `53.91%`、SM throughput
`7.8%`、DRAM throughput `0.23%`、registers/thread `34`。它们是该次 profile 的
测量值；DRAM write=0 不表示 kernel 没有输出，不能把缓存下的 DRAM 计数等同于逻辑
global store 字节。profile duration `183264 ns` 与上表的未插桩 CUDA event 时间
分开记录。原始报告包含采集来源、设备 UUID 与输入 hash 核对。

### 冻结 FA3 的真实进度

[frozen-readiness.json](../../../research/hopper-ptx/stage8/evidence/acceptance-20261010/frozen-readiness.json)
明确写着 `parser_only=true`：完整入口 `sm_90a`，3025 个包含 label 的前端条目，
退出码 0。它证明原文可解析/汇编，不是执行成功。

[frontend-contracts.json](../../../research/hopper-ptx/stage8/evidence/acceptance-20261010/frontend-contracts.json)
还明确验证了 24 次“functional/detailed 执行必须拒绝”的情况。因此下面的缺口是
已知、被测试保护的实现边界：

- `elect.sync`、`prefetch.tensormap`、init fence、`stmatrix`；
- TMA store、bulk commit/wait、grid dependency、`setmaxnreg`；
- detailed 远程 DSMEM、cluster barrier/proxy fence；
- 原生 opaque tensor-map、dynamic shared resource 和 cluster launch 的 runtime 接口。

源码佐证：内层 `f4fe783a` 的
[`instructions.cc`](../../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) 中
`elect_impl`、`prefetch_tensormap_impl`、`hopper_frontend_only_impl`、`setmaxnreg_impl`
仍抛未实现异常；
[`cuda_driver_modern.cc`](../../../gpu-simulator/gpgpu-sim/libcuda/cuda_driver_modern.cc)
中的 tensor-map/cluster occupancy 接口返回 `CUDA_ERROR_NOT_SUPPORTED`。

### Runtime 的真实进度

阶段 9 的两个同名 entry module 分别输出 `121..184` 与 `221..284`，覆盖模块隔离、
参数 ABI、默认 stream launch 与卸载句柄失效。82 checks 同时包含成功执行和错误
返回测试；不能理解成实现了 82 个 CUDA API。

PyTorch/Triton 调用清单来自固定版本源码的静态分析。对应原生应用没有运行验收；
context、cubin/fatbinary、JIT、非默认 stream、allocator/动态库替换等尚有缺口。
详见 [阶段 9 支持矩阵](09-runtime-integration.md#框架调用面和支持矩阵)。

## 2026-10-10 历史复验做了什么（非本轮执行）

1. 对照两仓库 HEAD、阶段报告、实际源码严格失败入口及保存的测试结果。
2. 校验阶段 8 最终 `sha256.json` 的 32 个文件，以及阶段 9 `SHA256SUMS` 的
   20 个文件：无缺失、无 hash 不匹配。hash 一致只证明证据未变，不自动证明测试充分。
3. 核对阶段 9 environment.json 中 probe、PTX、CUDA 头与当前 runtime hash：全部
   一致；当前 runtime SHA256 为
   `c70230dbe417d6f337ff1b7c57e080d7e7088202eeb84cf427d211ee1b73ff51`。
4. 用 `in-switch` Python 对保存的阶段 8 原始数值输出和日志重新运行聚合验证：
   `STAGE8_RESULT status=PASS numerical_cases=8 blockers=0`。本次没有重新运行
   A800 kernel 或采集 counters，也没有重建阶段 8 的全部模型结果。
5. 当前 runtime 上重新构建并运行阶段 9 functional/detailed：各
   `RUNTIME_RESULT status=PASS checks=82 failures=0 numeric_values=128`。

当时命令（外层仓库根目录）：

```bash
/home/wangjz/miniconda3/envs/in-switch/bin/python3 \
  research/hopper-ptx/stage8/verify_results.py \
  /tmp/hopper-stage8-readiness-20261010/model-final \
  /tmp/hopper-stage8-readiness-20261010/hardware \
  /tmp/hopper-status-audit-20261010-stage8.json
bash research/hopper-ptx/stage9/run.sh /tmp/hopper-status-audit-20261010-functional
bash research/hopper-ptx/stage9/run.sh /tmp/hopper-status-audit-20261010-detailed 0
```

新 runtime 日志为上述两个目录中的 `runtime.log`，阶段 8 聚合输出为上述 JSON。
`/tmp` 路径是本机复验记录；长期可读的原阶段结果优先使用本页表格中的仓库链接。

## 历史报告为何容易误读

- 阶段 5/8 同时保存计划、失败、解阻和范围有限的 PASS；现在 3–9 的当前缺口与计划
  统一移到首页，历史内容折叠，原 JSON 不修改。原始 PASS 不再决定当前整体验收。
- 阶段 3/4 等报告只描述当时提交，不会自动更新为阶段 9 的全部能力。阶段 3 中
  “WGMMA 未实现”不能用来推翻阶段 6 的受限 WGMMA 实现。
- 阶段 7 的资源/远程 timing、阶段 8 的完整 FA3 执行、阶段 9 的真实框架集成
  均未完成；之前将限定测试结果汇总为整阶段通过过于宽泛，现已撤下完成标记。
  不以阶段数量换算总目标完成百分比。
- 阶段 4 的内层 SHA 曾有转录错误，本次按 Git 修正；若需复现历史阶段，应使用
  配对提交与该阶段配置，不把现有 runtime 当作历史二进制。

## 已确认的后续目标（2026-10-13）

长期目标保留在 [H100 PTX → DGX/NVLS 路线](../../plans/h100-ptx-dgx-roadmap.md)。
本轮按用户最新要求先完成单卡，当前 10–14 均为 `未开始`：

1. 先完成 FP16、causal MHA 的完整 Dense prefill block，覆盖 norm、attention、
   SwiGLU FFN 与 residual；接入真实 Python 程序，先 PTX/Triton 适配器，后所需 Driver shim。
2. 校准单卡 H100，独立关键微基准逐项误差 ≤20%；完整 Dense 单独报告性能与约一天
   的详细仿真预算。达到此处停止，不自动开始原版 FA3 或多卡工作。
3. 阶段 15 原版 FA3 和阶段 16–19 DGX/NVLS/DeepEP/MoE 仅保留未来计划。

主开发环境为 Conda `in-switch`。权限阻塞时 AI 编写可审阅脚本，由用户/管理员
运行；H100 与 DGX 分别准备短租包，详见
[短租测试手册](../h100-dgx-rental-runbook.md)。以上后续目标来自实现前的计划整理；
随后仅实施并验收阶段 3，未开始阶段 4 或 10–14。
