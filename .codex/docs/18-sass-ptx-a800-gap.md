# SASS 模式与 PTX 模式差距：A800 单卡实验证据

本实验的完整元数据、命令、固定版本、失败分类和逐例数据见
`.codex/experiments/2026-09-20-sass-ptx-a800-gap.md`。这里保留面向后续设计的结论。

## 结论（Observed）

在同一个 `sm_80` CUDA 程序、同一个 `SM80_A100/gpgpusim.config` 基线下，
SASS trace replay 与 PTX execution-driven replay 的周期通常接近，但退休指令数和
IPC 并不接近：

- GEMV 64/128/256 的 PTX 指令数比 SASS 少 36.5%–38.2%，IPC 少 39.5%–41.6%，
  但周期只多 3.5%–5.9%。
- GEMM 16/32/64 的 PTX 指令数少 20.0%–25.3%，IPC 少 22.7%–29.0%，周期多
  3.4%–5.2%。
- 全部 12 个样例中周期差范围为 −2.6% 到 +19.0%；说明不能用一个固定比例把
  PTX 结果换算成 SASS 结果。

这些数字是本地 A800 上的 Accel-Sim 相对证据，不是 A800 的绝对性能校准，也不是
A100/H100/NVLink/NVSwitch 结论（Needs hardware validation）。

## 设计含义

PTX 适合作为新 ISA、调度器和微架构实验的主线；SASS trace 仍然需要保留，用来检查
真实编译器形状、指令展开、访存行为和硬件 trace 一致性。GEMV/GEMM 结果尤其说明，
即使高层内存读写计数相同，前端产生的动态指令/IPC 仍会显著不同。

## 可复现性与限制

- 12 个 CUDA 原生样例、12 条 A800 trace、12 个 SASS replay、12 个 PTX replay
  均通过功能检查。
- CUDA 12.8 `-lineinfo` 触发 PTX `.loc ... inlined_at` 解析失败；去掉编译期
  `-lineinfo`，并设置 `TRACE_LINEINFO=0` 后通过。这是 parser compatibility
  baseline failure，不应误报成模式差距。
- PTX 直接执行 Slurm wrapper 时末尾 `mv` 缺少错误文件会返回 1，但模拟器已经
  输出正常退出和 `RESULT ... status=PASS`；这是 harness artifact。
- trace metadata 的 `total_insts` 与 `gpu_tot_sim_insn` 属于不同计数口径，不能直接
  相除得出“模式膨胀倍数”。
- 规模控制在 GEMV/GEMM 64 以内是为了避免模拟时间失控；没有多卡、cuBLAS、真实
  H100 或 NVSwitch 验证。
