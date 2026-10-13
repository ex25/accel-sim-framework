# 阶段 11：单卡完整 Dense 功能执行

状态: 未开始

前置 gate：[阶段 10](10-dense-contract.md) 已完成规格与真实 kernel 冻结。
实际命令/结果/证据：尚无；旧阶段 8 attention PASS 不是本阶段结果。

## 工作项与实现顺序

1. 在 `research/hopper-ptx/stage11/` 建立通用 tensor 分配、加载 PTX、参数打包、
   launch、输出回读和检查 harness。先使用内部 PTX 入口完成数值闭环，公开 Python
   入口在阶段 13 接入。动态 shared、寄存器/局部内存 metadata 也需显式导入，不能
   以 harness 绕过真实资源需求；必要扩展记录在阶段 9。
2. 单算子顺序为 RMSNorm → GEMM → RoPE → causal attention → activation/residual，
   然后按下表组合 block；每一步先定位第一个未支持语义，再做最小正例并回归。

| 子链 | 必需验证 |
| --- | --- |
| RMSNorm1/2 | 行归约、epsilon、FP32 累加与 FP16 cast、逐行独立，无跨 token 混入 |
| QKV + RoPE | GEMM 方向和 Q/K/V 分片、head reshape、位置和奇偶通道、S=128/512/2048 |
| causal MHA | mask 对角线/未来位置、scale、softmax 稳定性、所有 head/row；改变未来 token 不影响此前输出 |
| O projection + residual | heads 合并、GEMM、第一次 residual 的来源/精度 |
| SwiGLU FFN | gate/up 两路、SiLU 与逐元素乘、down projection；2816/11008 边界 tile |
| 最终 residual | 第二次 residual 来源及最终 `[B,S,hidden]` 所有元素 |

3. 依据真实 PTX 补 `ptx.l/ptx.y/ptx_parser.cc`、`instructions.cc`、`ptx_sim.cc`
   中被触发的语义；所需 ldmatrix/mma、shuffle、conversion、math 等分别做独立
   数值检查。不要假设现有 WGMMA N=128/176 就覆盖所有 GEMM。
4. 使用 TMA/WGMMA 的 kernel，按阶段 4–6 补 descriptor/layout/swizzle、共享
   内存与异步完成；使用 setmaxnreg 的 kernel，按阶段 7 补资源语义。未知形式仍
   严格失败。无需 cluster 的目标不强行开启 cluster 功能，也不宣称其完成。
5. 逐算子比较，再比较整链关键中间值与最终输出；测试多个固定 seed、零/正负/
   大小值、tail tile 和 causal 性质。reference 不通过模拟器同一布局函数生成。
6. 跑三个正式 shape 的全 grid；保存 kernel 序列、动态执行计数和实际输出，
   建立首个偏差定位。严禁把 CPU reference、输出缓存或零填充当成模拟计算。

## 验收条件

- [ ] 三个 shape 全部完整运行；每层、中间值及最终输出满足阶段 10 的逐项容差。
- [ ] 原始输入/权重/output/launch/PTX/hash 和逐算子检查结果持久保存，无缺项。
- [ ] 正式路径所有必需 opcode/API 真实执行成功；unsupported 的非法输入仍拒绝。
- [ ] 旧 2–9 中受影响的正例回归通过；新增能力不再仅由“预期失败”覆盖。
- [ ] 输出来自 simulator 内存，host 只做调度和 reference，无目标算子代算。

性能此时未验收。不要求冻结原版 FA3 执行；原版 FA3 的失败仍在阶段 15 登记。
若只 small 通过、large 超时/未跑，不可标完整 Dense 功能通过。

## 实施后填写

两仓库版本与 runtime hash、各算子支持矩阵、每 case 实际命令/误差/耗时、失败
记录及证据路径。然后进入 [阶段 12](12-dense-timing.md)。
