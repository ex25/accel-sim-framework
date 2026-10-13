# 阶段 13：真实 Python Dense 入口与所需 Driver shim

状态: 未开始

前置 gate：[阶段 12](12-dense-timing.md) 全 grid 详细模型通过。
实际命令/结果/证据：尚无。阶段 9 的 C++ probe 和静态源码清单不计为 Python 验收。

## 实施步骤

1. 复用阶段 9 的 PTX/Triton adapter 并扩展到完整 block：真实 Python workload 编译/选择 kernel，导出
   并执行 PTX，管理 tensor 与 launch 依赖。支持 hardware/simulator 两种明确配置。
   不承诺 stock PyTorch/cuBLAS/cubin 二进制兼容，不能用该限制掩盖 Python 没有跑通。
2. 再按实际调用轨迹补阶段 9 Driver shim，包括所需 context/device attributes、
   内存/copy、module/function、资源属性、dynamic shared、stream/event、同步
   和错误路径。实际入口需要 `libcuda.so.1`/dlsym 时，验证加载路径而非仅检查导出符号。
3. kernel metadata 来源绑定 PTX/compiler hash；寄存器/共享内存属性影响阶段 12
   调度。tensor-map 通过声明的逻辑 descriptor 适配或正确受支持 API 创建，不猜测
   opaque 二进制布局，不返回未参与资源计费的假值。
4. 处理 compile/cache hit/miss、模块重载、内存生命周期、重复运行；测试所需非默认
   stream/event 的顺序、错误传播、卸载时仍在途任务。多 host thread 未用则声明排除。
5. 同一个 Python block 跑三个 case，在功能/详细模式分别比较阶段 11/12 的输出和
   kernel 序列。保存实际 API/launch 调用记录，确认没有 CPU 算子/硬件旁路替代。
6. 把环境安装、编译、启动和结果检查放进可重复脚本；主环境 `in-switch` 保持可追溯，
   需要隔离环境时记录名称、路径和版本。文档写清 adapter 的使用方法及覆盖矩阵。

## 验收条件

- [ ] 真实 Python 程序覆盖三个正式 case，functional/detailed 均完整通过。
- [ ] 目标算子均在模拟器执行；输入/输出、kernel/PTX/launch 与冻结规格一致。
- [ ] 所需 API/资源/stream 路径均有正例；非法/未实现路径给明确错误，无假成功。
- [ ] context/module/内存/事件生命周期和重复运行通过，无旧句柄复活或结果串扰。
- [ ] 文档分列 adapter 和 shim 验收；所有 required API 完成，不能只完成 adapter
      就标整个阶段或阶段 9 完成。

阶段 9 的前置验收仅为其冻结算子/短链；本阶段新 API/shape 需求必须补齐并追加
覆盖记录，不能沿用旧组件状态自动声称完整 block 通过。
完成后进入 [阶段 14](14-dense-h100-validation.md)。

## 实施后填写

配对提交、环境 lock、真实 Python 命令、API/PTX/launch 日志、逐 case 输出/时间、
资源属性与实际调度证据、错误/生命周期结果、支持/不支持范围和证据路径。
