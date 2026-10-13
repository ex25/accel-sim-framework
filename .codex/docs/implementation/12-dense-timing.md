# 阶段 12：完整 Dense 详细时序与资源

状态: 未开始

前置 gate：[阶段 11](11-dense-functional.md) 全部功能验收通过。
实际命令/结果/证据：尚无。旧 attention detailed 的 1 row/1 head 不算本阶段运行。

## 实施步骤

1. 用同一 PTX/输入/launch 切换详细模式，先逐算子，后完整 block。检查每条动态
   指令从 PTX 执行、scoreboard、调度、执行单元到 writeback 的对应；功能语义
   提前算出输出时，依赖指令仍须等待模拟完成，不能因此提前消费结果。
2. 对实际 GEMM 路径检查 Tensor Core 路由、latency/II、warp/warpgroup 发射与
   commit/wait 单位。现有每 warp GMMA 配置不能未经检查就当作 warpgroup 吞吐。
   对矩阵 shared 读、bank/端口、寄存器依赖、cache/HBM 流量建立可解释计费；如
   采用抽象模型而非每次独立 LDST，明确资源占用和校准方法，不能漏计或重复计费。
3. 阶段 4/5 补多 barrier、多 tile 在途、ring buffer 重用、乱序 global response、
   OOB/尾块、TMA 与计算重叠；保存 issue→response→shared 可见→transaction
   完成→wait release 顺序。脱离固定单 tile harness 绑定后仍须成立。
4. 阶段 7 补正式 kernel 的寄存器/static/dynamic shared/local memory 和 CTA
   residency；若使用 setmaxnreg，测 inc/dec 的等待/释放及守恒；不使用时保存证据。
   用资源临界点验证 occupancy，不以 API 返回 success 替代调度行为。
5. 从单 CTA 到多 CTA、多 warpgroup、多 SM 压力用例，验证资源竞争、队列满/恢复、
   同步不提前完成和无死锁；比较算术/访存/同步主导时的模型变化是否合理。
6. 完整运行三个 Dense case；保存逐算子和端到端模拟 cycles/GPU 时间、执行条数、
   字节/transaction/队列统计，以及宿主 wall time/RSS。规范 clock domain 转换。
7. 最大 case 超预算时分析模拟器宿主热点、日志开销和事件推进，做语义等价优化。
   正式输出不可改成只跑一个 CTA、缩小 shape 或按比例外推。

## 验收条件

- [ ] 三个完整 block 的详细输出与阶段 11/reference 一致，正式 grid 未裁剪。
- [ ] 必需动态指令有 timing/resource 模型；无未标注 functional-only、no-op 或旁路。
- [ ] 请求/字节/同步/资源守恒及竞争压力测试通过，逐算子统计能解释端到端时间。
- [ ] 最大正式 case 详细执行目标 ≤24 小时，记录机器 CPU/内存与实际 wall time；
      这是本路线“约一天”的执行预算，不是模拟 GPU 时间。
- [ ] 全部证据和失败保存，旧组件回归通过；尚未实测的 H100 误差明确为未验证。

## 实施后填写

配对代码/runtime/config hash、资源表、统计/输出、完整运行命令与实际成本。
完成后开启 [阶段 13](13-dense-python-runtime.md)，硬件准确度留阶段 14。
