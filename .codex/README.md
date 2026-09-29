# Hopper PTX 研究资料

这套资料服务于 Accel-Sim 与 GPGPU-Sim 的配对研究分支：

- 外层仓库分支：`research/hopper-ptx`
- 内层仓库分支：`research/hopper-ptx`
- 外层起点：`dev`，`.gitignore` 提交后为 `907c0c8`
- 内层起点：`dev`，`91880c53`

## 阅读顺序

1. `rules/hopper-ptx.md`：适用于本研究的工作规则。
2. `plans/hopper-ptx.md`：完整目标、阶段顺序和门控条件。
3. `docs/evidence.md`：源码证据和已知缺口。
4. `docs/implementation/00-baseline-and-branches.md`：当前阶段记录。
5. 后续只读取当前阶段及其验收记录，不提前执行下一阶段。

## 状态约定

阶段状态只能是：`未开始`、`进行中`、`待验收`、`已验收`、`阻塞`。
状态为 `已验收` 前，必须写出实际命令、结果摘要、证据路径和代码 commit。

这不是 Hopper PTX 已经实现的声明。当前资料记录的是研究设计、源码事实和
后续实施顺序；未完成的结果必须明确写为“未验证”。
