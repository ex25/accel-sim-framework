# 阶段 3：两个 CTA 的调度与 shared 身份

状态：仅方案；本轮无实现、修复或运行验收。[对应验收](../validation/stage-3.md)。依赖阶段 0 的动态 shared 及阶段 1 的基础执行。

## 本阶段解决什么

专家 GEMM 的两个线程块必须同时有机会执行，并能找到同一 cluster 内对方的 shared。CTA 就是线程块，SM 是运行线程块的硬件单元。本阶段只建立身份、存储和共同推进，真正的数据搬运及完成通知由阶段 4/5 接上。

## 当前事实与直接影响

| 静态/硬件记录事实 | 为什么影响当前 workload |
| --- | --- |
| `cudaLaunchKernelExC` 尚未导出 | 冻结启动器不能通过所需入口启动带 cluster 参数的专家 GEMM |
| generic 地址范围仍按 96KiB shared、80 SM 划分 | 当前 shared 大于 96KiB、配置有 132 SM；这些地址不能继续按旧范围分类，否则可能认错空间/SM |
| 现有 PTX 线程初始化只连接本 CTA shared | 尚不能仅凭已有 cluster 字段访问同组另一个 CTA 的存储 |
| 每 CTA 384 线程、动态 shared 215040B；每 SM shared 233472B | 两 CTA 的动态部分就超过一个 SM，必须跨 SM 驻留；不能因为系统里有“cluster”类就把两个 CTA 塞进一个 SM |

H100 记录总 shared 分配 216064B = 动态 215040B + 驱动保留 1024B，PTX 静态部分为 0B。此内核每 SM 一个 CTA，不是所有 Hopper 程序都只能如此。[资源依据](../hopper-notes.md#resource-facts)

## 指令与接口清单

| 对象 | 实现动作 | 当前状态 |
| --- | --- | --- |
| `cudaLaunchKernelExC` 的 cluster dimension 属性 | 传递实际 grid/block/shared/stream 和 1×2×1 cluster | 待新增 |
| `%cluster_ctaid.x/y/z`、`%cluster_ctarank` | 根据本次 launch 和 CTA 坐标求值 | 待接通/验证 |
| `mapa.shared::cluster.u32` | 保留 shared 偏移、改变目标 rank，不搬数据 | 待新增 |
| `barrier.cluster.arrive.relaxed.aligned`、`barrier.cluster.wait.aligned` | 记录本轮参与并等待两个 CTA 的全部所需线程 | 待新增 |
| shared/local 的 generic 地址转换、`cvta` 相关已有指令 | 扩大合法范围并保持正反转换一致 | 待修正/回归 |

## 如何实现

1. **沿普通 launch 扩展。** ExC 读取当前唯一需要的 cluster 属性，复用阶段 0 参数绑定和入队；在 kernel 对象保留 cluster 维度。普通 launch 的单 CTA 行为继续使用原路径，不创建新的内核执行器。
2. **用确定的 CTA 身份定位存储。** grid 为 66×2×1、cluster 为 1×2×1 时，`(x,0,0)` 与 `(x,1,0)` 是同一组，rank 分别为 0/1。以 kernel 身份、cluster 坐标、rank 找 shared 实例，SM 只是当前放置位置。沿现有 `ClusterCTAIdentifier` 补 kernel 隔离的所有权，不用全局 offset 单独当键。
3. **统一地址编码和解码。** 扩展 generic 地址区间时同时检查 shared/local/global 的范围和反向转换。`mapa` 用独立的 cluster shared 地址编码保留 rank 与偏移，沿已有 SASS 编码能满足当前需求则复用；由同一处解码供远端屏障及 TMA 使用，不让各 opcode 自己猜位。不要把 SM id 或主机指针暴露为 PTX shared 地址。
4. **成组安排资源。** 在普通 CTA 调度中，找到能容纳当前两个 CTA 的 SM 后再接纳这一组；可先保留两边所需资源，再按正常启动延迟发射。不要先把所有组的 rank0 发完、等占满 SM 才找 rank1。复用实际 shared/线程/寄存器占用计算，不能仅凭 cluster 数量放行。当前 H100 配置的 `simt_core_cluster` 每组只有一核，不能直接当 CUDA 的两 CTA cluster。
5. **存储跟随当前 cluster 生命周期。** 创建本组 CTA shared 后供各线程引用；本组还有远端访问或完成回调时保留相应对象。用现有资源所有权加本组待完成状态即可，不建全局垃圾回收系统。允许各 CTA 正常执行退出处理，不能把数据尚未交付的对象提前分给下一组。
6. **屏障只管所需参与。** 用本组本轮的参与状态实现 arrive/wait，收齐合法线程后完成。relaxed arrive 不额外发布任意 shared 写；不要用一次 cluster barrier 把所有 TMA/WGMMA 都强制排空。远端通知通过阶段 4 的目标解析接入。

## SASS 性能模型复用

复用 cluster 元数据字段、普通 CTA 资源计算、调度入口、shared 存储和既有互连能力。trace 给出的 CTA/cluster 身份在 PTX 中需由 launch 自行计算。不能把现有结构字段误当成已实现跨 SM PTX 调度。

跨 SM 通知/交付沿已有模型中可用的排队、传输与完成路径接入；若缺少对应路径，先做当前两 CTA 所需的有限传输成本并标记近似，不实现任意 GPC 拓扑，也不假定跨 SM 零代价。具体数据字节数由阶段 5 提供。

## 本阶段要排除的实现错误与暂缓

两个 CTA 都使用偏移 4096 时，若只按偏移寻址会写到同一份存储；若只调度半组，已经驻留的 CTA 可能等不到同伴。这些是设计需防止的后果，不是当前运行复现。验收以不同标记值、延迟到达和多组驻留检查，不在生产代码中加“等太久就放行”。

暂缓任意 cluster 形状、多 GPU、完整 GPC 放置模型和 workload 未用的远端 load/store 指令。保留当前两 CTA 跨 SM、SM≥80、大 shared 和生命周期测试，因为它们直接覆盖实际配置。

## 源码入口与实际交付记录

[cuda_runtime_api.cc](../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc)；[abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) 的地址范围、`kernel_info_t`、`ClusterCTAIdentifier`；[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) 的线程/shared 初始化；[gpu-sim.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) 与 [shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) 的 CTA 分配/回收。

待填写：接口和指令改动、共同驻留/地址证据、提交与验收链接。当前：无。
