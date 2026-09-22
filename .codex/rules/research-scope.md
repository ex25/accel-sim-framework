# Scope guard 与 phase gate

第一阶段是 reconnaissance 与 evidence capture：audit repository/history、读取
`.claude`、记录 A800 environment、build existing code、运行小型 real trace、
mapping 一条 instruction、audit Hopper/multi-GPU gap、研究 NCCL/NVLink/
NVSwitch/NVLS/BookSim，并记录 integration design。不得一开始就全面重写
simulator core、完整实现 NVSwitch/NVLS/NCCL，或把 copied 8-rank trace 当作
multi-GPU execution。

已确认方向是 PTX-first。A800 SASS baseline 之后，依次完成 SM90 PTX baseline、
minimal custom instruction、communication-event schema。SASS 用于 existing
hardware validation，不用于为硬件中不存在的 GPU/instruction 生成主要输入。

当前 phase 没有 exit record 时不得进入下一阶段。Exit record 至少包含 baseline
SHA/command、source evidence、known limitation、focused test plan 与 next-step
decision。最小实现顺序为：

1. PTX baseline 与 source call path；
2. 一个有 timing/dependency focused tests 的 custom/synthetic instruction；
3. communication-event schema 与 logical GPU contexts；
4. BookSim transport facade 与 clock/event coupling；
5. Ring collective；
6. NVSwitch topology 与 semantic state；
7. NVLS multicast/reduction/completion；
8. SASS/H100 validation 与 hardware calibration。

Real H100/NVSwitch calibration 是后期 gate，不能成为从 A800 或 synthetic PTX
workload 制造 hardware claim 的理由。
