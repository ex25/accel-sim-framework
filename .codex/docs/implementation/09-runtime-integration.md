# 阶段 9：运行时集成

## 当前审计结论（2026-10-13）

状态: 进行中

前置：阶段 8 新版验收通过。文本 PTX module/launch 各模式 82 checks 有效，但
context/stream/资源等接口不完整；in-switch 未安装 torch/triton，无真实 Python
结果。本阶段补算子/短链 runtime，三个完整 Dense block 的验收由阶段 13 承接。

### 本阶段如何补齐

1. 在本阶段冻结 Python/PyTorch/Triton/compiler 版本和可复现算子清单，包含真实
   生成的矩阵 kernel 与至少两个有数据依赖的 kernel 短链。不等阶段 10 先完成。
2. 先 PTX/Triton adapter，后所需 Driver shim：context、分配/copy、module、
   launch、资源属性、dynamic shared、stream/event、错误及同步。追踪真实 API，
   不用静态 inventory 代替；保留阶段 3–7 组件支持范围的接口/导入用例。
3. 接入阶段 5 descriptor 与阶段 7 cluster/occupancy 的公开接口或明确适配入口，
   在 harness 与 API 两侧验证一致性。opaque ABI 不猜布局；所承诺公开 API 不能
   只返回成功，register/shared/local memory 属性必须影响实际调度。
4. 验证 JIT/cache、动态库加载、重复运行、在途卸载/内存释放及错误传播；非默认
   stream/event 有真实顺序和并发正例，不只测试拒绝。

### 本阶段可验收条件

- [ ] 冻结 Python 算子/短链在 functional/detailed 真正执行，输出与 reference 一致。
- [ ] 所需 API、descriptor/cluster 导入及资源模型贯通；非法/未支持调用明确错误。
- [ ] adapter 与 shim 分别有证据；生命周期/cache/stream/资源测试及旧 probe 回归通过。
- [ ] 完整保存环境、API/PTX/launch、输出/统计、配对版本和可复现命令，无 CPU 代算。

任意 stock PyTorch/cuBLAS/cubin 兼容不在承诺范围。阶段 13 复用这里的能力，补
完整 Dense 的新增需求并跑三个正式 shape；本阶段通过不等于完整 Dense 原生运行。

当前执行顺序见 [单卡 Dense 计划](../../plans/single-card-dense.md)。

<details>
<summary>历史实现、测试结果和旧计划（不作为当前验收状态或执行顺序）</summary>

验收范围：**文本 PTX Driver module/launch 子集**。functional/detailed 各
`82 checks、128 numeric_values` 通过；PyTorch/Triton 原生应用与完整 FA3 执行
尚未完成。先看[支持矩阵](#框架调用面和支持矩阵)、[结果证据](#验收命令结果和证据)
及[实现总览](README.md)，不要把标题“运行时集成”理解为原生框架已全面可用。

## 目标、前置条件和执行计划

阶段 8 旧范围通过，两仓库均在 `research/hopper-ptx`。开始时外层 HEAD 为
`9ddf32f78f8c1ada328c076484e44bb437be79d5`，内层 HEAD 为
`d3494f34c75bea3edfbcace59dc1f31701597fb8`。

先列出固定版本 PyTorch/Triton 的官方源码调用点，再建立文本 PTX module/function
生命周期并接通实际 launch；独立评估 dynamic shared memory、cluster 和 tensor-map，
无法贯通资源/执行模型的 API 明确拒绝。预期公开 API 正例、参数错误、卸载失效、
输出保留与已有阶段回归均可测试。原始报告写在本文件，不新增独立 report。

## 实际实现和数据流

`cuInit → cuModuleLoad{,Data,DataEx} → 独立 symbol_table/parser → CUmodule →
cuModuleGetFunction → CUfunction → cuLaunchKernel → 既有 parameter ABI →
stream queue → functional/detailed execution → cuCtxSynchronize → cuModuleUnload`。

- 内层 `libcuda/cuda_api_object.h` 的 `cuda_runtime_api::driver_modules/driver_functions`
  保存 module/function 显式状态；`CUctx_st::find_kernel` 使用检查式查找。卸载等待
  在途默认 stream 工作完成，移除 launch 注册，保留 tombstone 防止旧句柄复活。
- `libcuda/cuda_runtime_api.cc` 实现文本 load/file/DataEx 无 options、按 module 查询
  entry、PTX/static-shared 属性、可读错误名、launch 参数校验及同步。保留既有
  host function 注册入口，失败不写伪造输出。公开事件是 `DRIVER_MODULE`、
  `DRIVER_FUNCTION`、`DRIVER_LAUNCH`。
- `src/cuda-sim/ptx_loader.cc::load_driver_ptx` 创建独立 parser，继承 simulator
  core_config，恢复原 parser/global table/source 状态；常规语法错误映射为
  `CUDA_ERROR_INVALID_PTX`。`ptx.l` 的 recoverable_errors 仅用于该加载路径。
- `libcuda/cuda_driver_modern.cc` 使用安装的 CUDA SDK 头定义现代 ABI，导出
  `cuLaunchKernelEx`、`cuOccupancyMaxActiveClusters`、`cuTensorMapEncodeTiled`、
  `cuTensorMapReplaceAddress`。无 attrs 的 Ex 复用基础 launch；cluster/cooperative
  attrs 和 tensor-map 请求显式拒绝，日志给出 API/attribute/descriptor 请求状态。
- CMake 和 Make 同步加入现代 API TU；静态链接 anchor 保证该文件的符号进入
  libcudart.so。所有现有 Driver warning-only 假成功 stub（含历史 ABI 条件块）
  改为 NOT_SUPPORTED；源码审计 warning_success_stubs=0，281 个定义显式拒绝。

## 框架调用面和支持矩阵

固定评估版本为 Triton `v3.3.0` / `819e9c8c29ad2ae96cbd93a1d3b8a3a0f4c8f09c`、
PyTorch `v2.7.0` / `134179474539648ba7dee1317959529fbd0e7f89`；官方源码 URL 和 SHA-256
见 `research/hopper-ptx/stage9/source-manifest.json`。源码调用清单是静态检查，
本机 in-switch Python 环境未安装 torch/triton，不是动态运行 Python 应用的证据。

| 调用点 | 实际 API | 当前状态 |
|---|---|---|
| Triton driver.c::loadBinary | cuCtxGetCurrent、cuDevicePrimaryCtxRetain、cuCtxSetCurrent、cuModuleLoadData、cuModuleGetFunction | 三个 context API 明确拒绝；文本 module/function 可用，cubin 拒绝 |
| 同一 loadBinary 的 metadata/shared 分支 | cuFuncGetAttribute NUM_REGS/LOCAL_SIZE_BYTES/SHARED_SIZE_BYTES、cuFuncSetAttribute、cuFuncSetCacheConfig | PTX version/static shared 可查；无法得到的 ptxas 资源和 dynamic shared 配置明确拒绝 |
| Triton driver.py 生成 _launch | cuLaunchKernel、动态 dlsym cuLaunchKernelEx | 默认 stream 基础 launch 实际完成；cluster/cooperative attrs 拒绝 |
| Triton driver.c::occupancyMaxActiveClusters | cuFuncSetAttribute、cuOccupancyMaxActiveClusters | 拒绝，保留 occupancy 输出 |
| Triton driver.c::fill1D/2DTMADescriptor | 动态 dlsym cuTensorMapEncodeTiled | 拒绝，128 字节 opaque descriptor 不变 |
| PyTorch native/cuda/jit_utils.cpp::jit_pwise_function/launch_jitted_pwise_function | cuCtxGetCurrent、cuModuleLoadData、cuModuleGetFunction、cuLaunchKernel；可选择 NVRTC PTX 或 cubin | 文本子集已验证；context、cubin、非默认 stream 拒绝，NVRTC/Python 未集成 |
| PyTorch c10/cuda/driver_api.h::C10_LIBCUDA_DRIVER_API | cuDeviceGetAttribute、cuMemAddressReserve/Map/Create/SetAccess 等 virtual-memory family、cuGetErrorString | 错误字符串已实现；virtual-memory allocator 不支持，部分符号仍缺失 |
| PyTorch CUDAContext.cpp::initDeviceProperty/canDeviceAccessPeer | cudaGetDeviceProperties、cudaDeviceCanAccessPeer | 既有 runtime API；无 Python 原生应用验收 |

接口依据为 NVIDIA CUDA 12.8 官方 [module API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__MODULE.html)、
[execution API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EXEC.html)
及本地 CUDA 12.8 cuda.h。Triton 与 PyTorch 的官方固定 tag 源码 URL 见 manifest；
API inventory 包含源码行号。完整 PyTorch/Triton 原生路径还需要 libcuda.so.1 动态
解析与 context、cubin/JIT、allocator、stream 兼容，不能因本 TU 导出 cu* 符号而宣称完成。

## 验收命令、结果和证据

构建环境为 CUDA 12.8.61、/usr/bin/g++、CMake Release，运行库为
`/tmp/hopper-stage3-build/libcudart.so`。

```bash
cmake --build /tmp/hopper-stage3-build --target cudart -j2
bash research/hopper-ptx/stage9/run.sh /tmp/hopper-stage9-verified-functional
bash research/hopper-ptx/stage9/run.sh /tmp/hopper-stage9-verified-detailed 0
bash research/hopper-ptx/stage8/run_param_regressions.sh /tmp/hopper-stage9-final-regressions
```

两种模式均为 `RUNTIME_RESULT status=PASS checks=82 failures=0 numeric_values=128`。
两个同时加载、同名 entry 的 module 使用不同的常量，分别得到 `121..184` 和
`221..284`；参数包括 u64 pointer、u8、align 16 的 16-byte record、u32 tail，
验证了 ABI offset 与实际数值结果。负例包括未初始化、空 image/output、missing file、
malformed syntax、静态 global/constant、cubin/fatbinary、JIT options、未知 function/module、失效句柄、
reload 不复活、非法 grid/block、缺少参数、null 参数、extra、dynamic shared、
非默认 stream、cluster、occupancy 和 TMA。所有被检查的错误返回值符合预期，
失败时句柄/metadata/occupancy/descriptor 输出保留。

阶段 3–7 回归通过，包含 frontend、TMA detailed/multicast、WGMMA、cluster 和
parameter layout：`PARAM_LAYOUT_RESULT status=PASS positive=4 negatives=5 copy_guard=PASS`、
`PARAM_REGRESSIONS status=PASS stages=3,4,5,6,7`。共享 parser 的关键回归已通过。

`research/hopper-ptx/stage9/evidence/` 保存功能/详细日志、回归日志、输入配置、
build 日志、失败开发日志、symbol exports 和 SHA256SUMS。失败开发日志中的 SIGSEGV
发生在新 parser 未继承 core_config 时；修正后两种模式的完整正例均已通过。

## 验收和限制

三项验收条件均满足：module load/function query/launch 参数与错误可测试；本阶段
处理的 API 错误不伪装 CUDA_SUCCESS；文本 PTX 与原生框架支持明确区分。

- 静态 module global/constant 分配、原生 cubin/fatbinary、JIT options、非默认 stream、dynamic shared resource 计费、
  cluster/cooperative launch attrs、NVIDIA opaque tensor-map ABI 尚不支持，明确拒绝。
  第 5/7 阶段的 fixture descriptor/cluster harness 不是这些原生 API 的证据。
- 未获得 ptxas resource metadata，寄存器分配/LOCAL_SIZE 属性返回 NOT_SUPPORTED；
  detailed 数值结果与执行链有效，不构成 occupancy/resource 或硬件周期准确度证明。
- module/launch 仅验证单 host thread 与 simulator 隐式 context；无原生 CUDA
  context stack、多线程或完整动态库替换保证。
- 句柄 tombstone、成功/失败解析的 IR/source storage 保留到进程结束，尚未做
  反复 load/unload 的完整内存回收。
- 某些 lexer/assembler assert/abort 的坏 PTX 仍可能进程退出；常规 malformed PTX
  已验证可恢复。阶段 8 的严格 Hopper 未实现指令仍不允许近似成功。
- 此处未验证完整 PyTorch/Triton 应用、FA3 原始 PTX 整体执行或 H100/H200 硬件时序。

外层 revision：本报告所在外层提交。
配对内层 revision：`f4fe783ac7fc4f1cdaa9efa72a3575537d4fba36`。

</details>
