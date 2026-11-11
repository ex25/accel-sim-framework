# 阶段7测试入口与验收范围

已验收：单卡Qwen64/128、Phi/128、Mixtral/128，各一次无预热完整前向。
**六个大用例没有运行模拟：Qwen64/4096、Qwen64/8192、Phi/4096、Phi/8192、Mixtral/4096、Mixtral/8192。**
这些用例已有硬件基线，不代表模拟器已通过。多卡尚未验收。

[实际验收记录](../../../../../.codex/execute/validation/stage-7/README.md) ·
[完整kernel/端到端表](../../../evidence/hopper-stage7-128/README.md)

从项目根目录激活现有CUDA12.8/GCC13.4环境，并执行：

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source gpu-simulator/gpgpu-sim/setup_environment release
```

## 完整前向和独立检查

```bash
# 显式限制为已验收用例；run-dir须为新目录。
python3 workloads/moe/tests/hopper/stage-7/forward.py \
  --run-dir workloads/moe/runs/qwen128-recheck --cases qwen64-t128 --sim-events
python3 workloads/moe/tests/hopper/stage-7/acceptance.py \
  workloads/moe/runs/qwen128-recheck/qwen64-t128
```

其它两个用例替换为`phi-t128`或`mixtral-t128`，各用独立目录。
`forward.py`调用原冻结程序、原比较器和`summarize.py`；即使数值失败仍保留逐核摘要。
`acceptance.py`独立解码全部输出，并核对110个模块、22次调用的顺序/资源与数值结果。
**其退出码只反映执行及数值检查，性能字段仍须按主要内核与总和分别审查。**

`--sim-events`通过`event_timing.py`使用GDB读取运行库已有设备周期戳，需带调试符号的模拟库及支持Python的GDB。
它启用冻结启动器原有event分支；设备时间使用`event-timing.json`，不能使用运行库返回主机墙钟的`output.json/cuda_event_ms`。
单核执行、含启动的模拟区间和完整前向事件分别保存。显式DMA/真实主机间隙未完整建模，硬件首次/预热比值只作参照。

`batch_128.py`用于在已经准备好`batch.json`、环境和固定运行库副本的目录中并发调度三例；
实际运行参数见原验收记录。它的`passed`标签同样不能代替主要性能验收。

## 几分钟内可反馈的定向实验

```bash
python3 workloads/moe/tests/hopper/stage-7/expert.py \
  --run-dir workloads/moe/runs/expert-fc1-recheck \
  --test fc1 --real-shape --active-group 30 --trace
python3 workloads/moe/tests/hopper/stage-7/pipeline.py \
  workloads/moe/runs/expert-fc1-recheck
python3 workloads/moe/tests/hopper/stage-2/real.py \
  --run-dir workloads/moe/runs/gating-micro-recheck \
  --test qwen,phi,mixtral --full-config --single-cta --single-iteration
```

FC2使用`--test fc2`及新目录。局部专家保持真实N/K和原启动资源，只令一个分组非空；
独立数学参考和非活动/域外哨兵全量检查。它不是完整分组工作量或硬件性能证据。
`pipeline.py`检查输入交付、屏障、矩阵读取/写回、bulk组、源复用和最终确认，并报告未观察到的时序窗口。
`routing.py`复用真实前缀内核和独立路由参考；CPU-scores选项仅作定位，不能替代正式前向。

## H100数值取证与离线复算

[h100-numerics](h100-numerics/README.md)包含取证包准备、原生CUDA探针、已保存H100数据的生产算术回放，
以及旧库/新库的四个实际PTX短复现。取证和分析脚本依赖原运行目录中已生成的诊断数据，详见其README。
`check_numerics.py`与`small.py`不申请GPU；`modal_run.py`是单独的付费硬件取证入口，需要另行明确授权后使用。

完整日志、运行库、GPU取证数据和大张量留本地`runs/`，不随源码提交。小型结果表和版本散列在验收附件中归档。
