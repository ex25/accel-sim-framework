# Research 文档入口

这里是面向人的长期 research record。文档按主题维护，不按每个 phase 无限新增
文件；时间线只写入 `changelog.md`。核心入口如下：

1. [roadmap.md](roadmap.md)：唯一计划、阶段状态、进入/退出条件和下一步。
2. [implementation.md](implementation.md)：已实现功能、代码/symbol、调用路径、
   实验、验证结果和限制。

## 主题文档

- [concepts.md](concepts.md)：GPU、PTX/SASS、trace、
  NCCL、NVLink、NVSwitch、BookSim 和事件调用路径的概念说明。
- [custom-isa.md](custom-isa.md)：
  multi-GPU、custom ISA、in-switch computing、输入边界和逐层验证方法。
- [gpu-execution.md](gpu-execution.md)：A800 trace pipeline、Hopper/SM90 source
  audit 和 SASS/PTX gap。
- [network-model.md](network-model.md)：BookSim、Accel-Sim、clock coupling 和
  NVSwitch/NVLS transport boundary。

## 按需阅读

- [decisions.md](decisions.md)：不会频繁变化的架构决定。
- [changelog.md](changelog.md)：按日期记录阶段 commit 和历史变化。

以后新增内容先归入上述主题文档、`roadmap.md`、`implementation.md` 或实验
metadata；只有出现长期独立主题时才新增文档，不能按日期或 phase 自动新建
Markdown。
