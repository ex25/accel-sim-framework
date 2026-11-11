# 一次 H100 数值取证

用于解释冻结 H100 参考输出与模拟器的局部差异，不修改模拟器、参考值或容差。
原始失败坐标的 CPU 重建位于 `runs/hopper-stage-7/numerical-diagnosis-20261110/`。

测试只运行独立 kernel：Qwen 的两个失败 token、Phi 的十八个、Mixtral 的四十六个。
收集这些 token 的全部 FC1、SwiGLU、FC2 输出及路由权重，覆盖原有 209 个失败坐标，
并可检查同 token 其它输出。不是完整 MoE 前向验收，也不运行周期模拟。

- 原始 SwiGLU 与逐步测量核：记录指数输入、指数、分母、除法、两次乘法、BF16 编码。
- 全部实际 gate/up 组合去重，并补充 `[-16,16]` 内全部 BF16 gate、up=1。
- 单独记录 `ex2.approx`、`rcp.approx`、`div.full`、`div.rn`、`div.approx`；
  包含实际分母及上下四个 FP32 相邻值、`[1,2)` 的尾数采样，供本地验证通用近似模型。
- 同一 PTX 分别由本地 CUDA 12.8 编成机器码、由 H100 驱动即时编译；保留驱动缓存。
- 原始 grouped GEMM 只保留受影响 token 的专家组，N/K、矩阵权重配方和 kernel 不变。
  FC2 同时使用 CPU 激活与原生激活，区分数学函数差异与矩阵累加差异。

先用 `prepare.py --run-dir RUN` 在本地生成输入和编译全部产物。该步骤依赖上述 CPU 诊断文件。
进入 `RUN/package`，在真实 CUDA 环境执行 `./batch ../a800-preflight a800`，
然后执行 `preflight.py RUN`。预检核对原始/测量核、之前有效 A800 输出、路由掩码、
权重归一化及 grouped GEMM 参数。A800 无法预检 H100 专用 WGMMA。

付费执行只使用一次 `modal_run.py`：设置 `H100_NUMERICS_PACKAGE=RUN/package`，
按 `docs/modal-h100.md` 使用 `moe-h100` profile，传 `--result-dir RUN/h100`。
固定 `H100!`、单容器、零重试、一次性容器；子进程限 60 秒，整个函数限 120 秒。
编译和输入生成在本地完成；即使 grouped GEMM 失败也先保留已经通过 A800 预检的数学证据，
失败会明确记录，不视为通过。返回后结束 GPU 任务，在本地执行 `analyze.py RUN`。

远端执行时间包含同步与复制，不能作为 kernel 性能数据。实际账单以 Modal 用量记录为准。

## 不再申请 H100 的修复回归

已下载的一批证据可反复离线使用。`numerics.cc`直接调用模拟器生产头文件，
`check_numerics.py`检查全部数学探针、公开 H100 验证向量、已采集专家的全部 FC2 输出，
再按原路由权重和独立 CPU 合并检查原容差。它不会创建 Modal 任务。

```bash
python workloads/moe/tests/hopper/stage-7/h100-numerics/check_numerics.py \
  workloads/moe/runs/hopper-stage-7/h100-numerics-20261110 \
  --run-dir workloads/moe/runs/hopper-stage-7/numerical-repair-20261110 \
  --literature workloads/moe/runs/hopper-stage-7/literature-20261110
```

`--literature`中的 `a_H100_bf16.txt`、`b_H100_bf16.txt`、`c_H100_fp32.txt`、
`d_H100_fp32.txt`来自作者模型仓库的 `model_validation/H100/bf16/`，固定版本
`79143bd29a4a71dc7c2ed47741b63572498b17f6`。
原文及来源见本轮实现记录和运行报告；未将公开模型作者的精度声明当作本项目的验收结果。

`small.py`把保存的 H100 数学观测和独立构造的 BF16 边界矩阵送入实际 PTX 模拟路径，
对比原库/新库的输出、周期和动态指令数。需要已激活模拟器构建环境，以及保存的原库：

```bash
python workloads/moe/tests/hopper/stage-7/h100-numerics/small.py \
  workloads/moe/runs/hopper-stage-7/h100-numerics-20261110 \
  --matrix-ptx workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/matrix-regression/micro.ptx \
  --baseline-library workloads/moe/runs/hopper-stage-7/numerical-repair-20261110/baseline-libcudart.so \
  --run-dir /tmp/stage7-numerics-small-recheck
```

矩阵 PTX 由既有 `stage-6/run.py` 生成；边界输入在新驱动内构造，不更改冻结 MoE PTX。
`sfu_probe.cu`用于本机 A800 补测非单位分子、FTZ 和指数边界，读取两个 FP32 一组的输入，
输出每组九个 FP32 观测；这项也不使用 Modal。当前报告区分此 A800 测试与已保存的 H100 实测。
