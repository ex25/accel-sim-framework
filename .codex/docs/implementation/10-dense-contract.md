# 阶段 10：单卡 Dense 规格、工具链与验收集冻结

状态: 未开始

前置条件：阶段 3–9 已按各自组件退出条件验收。复用此前的最小规格/PTX 准备，
正式扩展到完整 Dense；本阶段不是当前入口，不能在阶段 3 未完成时提前推进。
当前依据：[单卡执行计划](../../plans/single-card-dense.md)。已有基线为外层
`f9faa73a4f7907db88db450e54000a502ecf39d2`、内层
`f4fe783ac7fc4f1cdaa9efa72a3575537d4fba36`；这不是本阶段实现提交。
实际命令/结果/证据：尚无。下列文件与脚本是待实现产物，不是可运行工具。

## 为什么先做这个

旧冻结 workload 是 noncausal FA3；阶段 8 是普通 attention；二者均不是已确认的
causal Dense block。现有框架清单也没有真实 Python 执行。需要先固定真实目标，
否则后续会继续以几个小 fixture 通过代替整体完成。

## 实施步骤与产物

1. 在 `research/hopper-ptx/stage10/` 建立版本化 `dense-spec.json`：三个正式 shape、
   dtype/layout、norm epsilon、RoPE base/位置/旋转方式、QKV packing、权重方向、
   bias、激活、residual/cast 位置、seed、初始化、softmax scale 和 causal mask。
   给默认值写出理由并冻结；禁止用隐含框架默认参数。
2. 建立独立 CPU/reference，明确数学式与 FP16 舍入点；RMSNorm FP32 reduction、
   GEMM FP32 accumulation、稳定 softmax、SwiGLU 的每项写入规格。准备可复现输入、
   权重及中间 tensor，按内存预算分块生成大参考结果，不能只生成小 case。
3. 冻结真实 Triton/CUTLASS/PyTorch 工具链和 kernel 源码。优先先解决 Dense 所需
   版本，不把未来 DeepEP 的工具链升级强加给当前目标。保存 PTX、entry、参数 ABI、
   grid/block、register/static/dynamic shared/local memory、编译 flags 及 hash。
4. 静态编译/导出可不依赖真 H100 时先做；需要目标设备时准备最小采集方案并记录
   阻塞。不能假设 A800 编译缓存或旧框架静态清单就是 H100 真实生成结果。
5. 建 `required-capabilities.json`，按每个算子列真实 PTX 指令/类型/layout/API/
   资源需求，关联阶段 3–9 的已有实现与待补测试。TMA/WGMMA/cluster 是否使用须
   来自导出结果；不用的功能明确延后，不为了跳过失败静默更换正式 kernel。
6. 冻结数值容差及校准/held-out 列表；包含 GEMM、norm、softmax、访存、同步、
   资源占用等正式路径。运行清单给每项 mode、预期输出、timeout、是否必需。
7. 记录环境 lock/重建说明、构建与 runtime 路径；为后续 runner 约定 JSON/CSV
   schema，正例失败、负例意外成功、missing/skip 都应导致相关 gate 不通过。

## 验收条件

- [ ] 三个正式 shape 的规格完整，可重建输入/权重/reference，hash 可复核。
- [ ] 所有正式算子真实 PTX/launch/resource manifest 已生成，来源版本固定。
- [ ] 每个能力缺口有所有者阶段、源码位置和计划正例，不再仅用 opcode 名称声称支持。
- [ ] atol/rtol、calibration/held-out、计时范围预先登记；未参与调参集可识别。
- [ ] 在 `in-switch` 或明确隔离的环境重跑导出/参考生成成功，保存实际命令和版本。

本阶段只验收规格和可复现输入，不要求模拟器已经执行完整 Dense。若真实 kernel
导出或参考生成仍失败，保留“进行中/阻塞”，不能只交设计文档就进入阶段 11。

## 实施后填写

外层/内层版本、环境/dirty hash、实际命令、manifest/reference 路径、用例与结果表、
失败与限制、验收结论。完成后只开启 [阶段 11](11-dense-functional.md)。
