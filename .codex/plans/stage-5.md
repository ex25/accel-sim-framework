# 阶段 5：tensor map、TMA 输入、组播与输出

状态：仅方案；本轮无实现、修复或运行验收。[对应验收](../validation/stage-5.md)。搬运依赖阶段 3/4；主机编码和 descriptor 修改可先独立完成。

## 本阶段解决什么

tensor map 是描述矩阵地址、形状、步长和 shared 排列方式的 128B 对象。TMA 读取它，按 tile 搬运数据。当前专家循环会修改描述符以切换专家，所以不能只记住启动时的主机参数。输入搬完才能计算；输出读完 shared 源以后才允许覆盖该缓冲区。

## 当前事实与影响

| 当前证据 | 影响与处理 |
| --- | --- |
| 编码函数查询入口尚未导出，本机真实驱动探针无可用 GPU | 冻结程序在注册模块之前就无法完成 tensor map 准备；实现模拟器内部编码，不依赖真实驱动 |
| SASS 已有 TMA 地址合并、OOB、byte mask 和组播完成处理 | 可复用时间模型，但 PTX 仍要自己算地址并搬真实数据 |
| bulk wait 首次判断和后续更新方向不一致 | 不能把现有判断原样作为 PTX wait1/0，需按实际组定义统一 |
| TMA store outstanding 在 global ack 清除，没有独立源读完事件 | 若只等 ack，shared 会被占用过久；若只看发起，可能还没读完就覆盖。新增所需源读取完成点 |

后两项是静态代码行为及适配理由，尚无当前 PTX workload 的错误输出复现。[依据](../hopper-notes.md#existing-waits)

## 需要实现的接口与全部指令

下表全部待实现或接入，未完成验收。

| 对象 | 本阶段动作 |
| --- | --- |
| `cudaGetDriverEntryPointByVersion` → `cuTensorMapEncodeTiled` | 对当前请求返回模拟器编码函数，使用冻结启动器的 ABI |
| `tensormap.replace.tile.global_address.shared::cta.b1024.b64` | 替换 64 位 global 基址 |
| `tensormap.replace.tile.global_dim.shared::cta.b1024.b32` | 替换维度；实物 ord0..4 |
| `tensormap.replace.tile.global_stride.shared::cta.b1024.b64` | 替换字节步长；实物 ord0..3 |
| `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned` | warp 协作复制/发布 128B 描述符 |
| `fence.proxy.tensormap::generic.acquire.gpu` | 对该 128B 范围建立后续 descriptor 读取顺序 |
| `fence.proxy.async.shared::cta` | 普通 shared 写入与后续异步读之间排序，阶段 5/6 共用 |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` | 非组播输入，真实数据交付后通知屏障 |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint` | 当前两 CTA 组播 |
| `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` | 输出写回 |
| `cp.async.bulk.commit_group`、`cp.async.bulk.wait_group.read` | 输出分组和源读取等待，当前 wait1/0 |

## 如何实现

### 1. 一个能按字节复制的内部描述符

主机编码、设备 replace 和 TMA 解码使用同一个固定 128B 格式，无主机指针、容器或外部对象索引。以下是适配当前参数的拟布局；使用固定宽度字段/显式偏移，编译时检查总大小，不依赖编译器随意 padding。

| 偏移/B | 内容 | 大小/B |
| --- | --- | ---: |
| 0 | 模拟设备 global 基址，u64 | 8 |
| 8 | global strides[4]，u64，按字节保存 | 32 |
| 40 | global dims[5]，u32，对应当前 PTX replace 字段 | 20 |
| 60 | box dims[5]，u32 | 20 |
| 80 | element strides[5]，u32 | 20 |
| 100 | rank、dtype、interleave、swizzle、L2 promotion、OOB fill，各 u32 | 24 |
| 124 | 保留字节置零 | 4 |

编码入口将当前合法 API 参数写入这些字段，保留重定位后的真实模拟地址。实际 rank2 仍允许 PTX 写入 inactive ordinals，不能按 rank 拒绝 ord2..4 的 replace。复制一个描述符再修改副本必须互不影响；TMA 每次按发布后对象内容求值，不按原 host 地址查旁路表。不实现与 GPU 不透明编码互换。

`cp_fenceproxy` 是 warp 协作的一次 128B 复制，不能 32 lane 各产生一份相同写流量。把复制及发布接到现有 global 写请求/顺序通路；acquire 与后续 descriptor 读按同一对象范围建立顺序。优先利用现有 pending 请求，不做全内存版本管理。

### 2. 先完成非组播输入，并立即交叉测试

读取当前 descriptor 和二维坐标；按当前 UINT16/BF16 的 2B 元素、element stride=1 计算有效 global 地址。rank2 的核心关系是 `base + x*2 + y*global_stride[0]`，x/y 包括 tile 起始坐标及元素偏移；box 决定搬运大小。越界元素生成零，不发无效 global 请求。

根据当前 64B/128B swizzle 把逻辑元素映射到 shared 字节位置。swizzle 是为改变共享存储访问分布而采用的地址重排，不改变矩阵的逻辑数值。实现一个当前布局的纯地址映射函数，供搬入/搬出使用，并与阶段 6 的矩阵描述符解释核对；不能把 64 位 WGMMA 描述符与本节 128B tensor map 混为一物。具体 swizzle 按原指令文档引用的布局定义核对，不按普通行主序猜测。

将有效地址及 byte mask 接到已有 TMA 合并、L2/DRAM 队列。返回时写入目标 shared 并通知 mbarrier；每个逻辑有效字节完成一次，不能按整 sector 大小多计。OOB 的零填写完后计入该 tile 完成字节，不等不存在的 DRAM 返回。

**非组播输入和阶段 6 单次 WGMMA 分别通过后，马上执行[首次交叉测试](../validation/stage-5.md#early-integration)。** 不等本阶段剩余功能或阶段 6 全部结束。

### 3. 再完成两 CTA 组播

沿阶段 3 的目标身份解析 mask 选中的 rank。一次 global 源读取可以向多个目标交付；每个目标的 shared 数据和屏障完成各算一次。记录“global 有效读取字节”和“各目标 shared 交付字节”，不能因复制到两个 CTA 就重复发两份相同 DRAM 读取，也不能只通知发起 CTA。

复用现有 multicast 路由/回调。没有现成的目标 shared 写入时序时，只补当前两目标的交付和必要通信成本；不建任意拓扑模拟器。

### 4. 输出分开两个完成时刻

```text
发起 → descriptor及shared源读完 → global写入确认
                    ↑                 ↑
              read wait可放行      内核退出仍需完成
```

从 shared 读取输出时按当前 swizzle 逆向取逻辑元素，生成 global 写请求。请求持有已读取的数据，不能在以后 ack 时再从可能被复用的 shared 取值。源读取占用需要的 shared 服务时间；不能仅因 CPU 在发起时复制了数据就把模型中的源读取耗时置零。

bulk group 按发起线程归属、提交顺序维护。read wait N 等较早组的 descriptor/源读取完成，最近 N 组可以仍在读。global ack 另更新最终写回状态，保留内核退出等待。PTX opcode 明确选择 bulk 队列，不能被最近的 WGMMA/普通 cp.async commit 改变。

### 5. shared 发布共用一处实现

`fence.proxy.async.shared::cta` 由阶段 5 或 6 先行提供，另一边复用。沿已有普通 shared 操作完成信息排序后续异步读取，不把 fence 变成全 GPU 等空。若功能值已提前写入 shared，仍要保证相关时序请求完成后后续操作才使用；无需为当前合法消费者保存整份 shadow shared 副本。

## SASS 复用与暂缓

复用 TMA transaction 合并、byte mask、L2/DRAM、OOB 完成入口、multicast 以及 store ack。PTX 自己生成 descriptor、真实地址/数据和目标屏障；`.read` 的源完成点与正确分组需补，不把已有 ack 当作同一事件。共用修改回归受影响 SASS TMA/屏障路径。

暂缓 rank>2 的实际搬运、其它 dtype/interleave/swizzle、精细 L2 hint 策略和原生 descriptor 互换；保留当前 inactive ordinal 写入、全/部分 OOB、循环替换 descriptor、wait1/0，因为真实专家使用这些行为。不要新增未知格式静默回退到普通 memcpy。

## 源码入口与实际交付记录

[launcher.cu](../../workloads/moe/source/launcher.cu) 的 Encode ABI；[cuda_runtime_api.cc](../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) 的运行时查询入口；[instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) 的新语义；[abstract_hardware_model.cc](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc) 的 TMA 合并；[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) 的 OOB、返回、组播和 `store_ack`。

待填写：descriptor/非组播/首次联测/组播/输出分别实现了哪些指令、修复的完成条件、验收及提交。当前：无。
