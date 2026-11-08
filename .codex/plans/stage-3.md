# 阶段 3 实现方案：让同组两个线程块驻留并找到各自的shared

状态：待执行方案，尚无本阶段实现或运行验收。依赖阶段0的启动/动态shared和阶段1的基础执行；配套：[阶段3验收](../validation/stage-3.md)。

## 完成后应当能做什么

用原专家内核所需的cluster启动形式运行小程序：两个CTA能共同推进，知道自己在组内的位置，并能把一个shared偏移解释为“同组另一个CTA的这个位置”。CTA是线程块，SM是运行线程块的硬件单元，cluster是CUDA程序指定的一组协作CTA。DSMEM指这些CTA可互相访问的shared存储，不是另建一块所有CTA共用的数组。

```text
本次kernel：grid=66×2×1，cluster=1×2×1
  cluster(x,0,0)
    CTA(x,0,0)，rank0 → 一个SM → shared对象A
    CTA(x,1,0)，rank1 → 另一个SM → shared对象B
  mapa(偏移4096, rank1) → 对象B的4096，不复制数据
```

完成本阶段只证明启动、身份、地址和同步基础。TMA搬运在阶段5，mbarrier通知在阶段4；不能把尚未实现的异步数据错误当成本阶段已有故障。

## 指令、接口与当前证据

| 对象 | 当前静态事实/直接影响 | 本阶段交付 |
| --- | --- | --- |
| `cudaLaunchKernelExC` | 导出核查没有该入口，冻结启动器使用它提交cluster属性 | 读取本次配置，沿普通launch路径创建kernel |
| `%cluster_ctaid.x/y/z`、`%cluster_ctarank` | 既有cluster字段不等于PTX已能求值 | 从CTA坐标和本次cluster维度求出本地坐标/rank |
| `mapa.shared::cluster.u32` | 当前线程shared连接本CTA，缺所需PTX映射 | 生成包含目标rank的地址，由共用解析入口找对象 |
| `barrier.cluster.arrive.relaxed.aligned`、`barrier.cluster.wait.aligned` | 需要跨两个CTA记录到达；不能套用本SM的CTA屏障 | 每轮正确登记、等待、释放，不附带异步排空 |
| generic shared/local地址及已有`cvta` | 现有上限为96KiB/80SM，地址分界由这些常量计算 | 统一扩大地址布局并验证分类/正反转换 |
| shared和寄存器资源 | 当前专家384线程、动态shared215040B、物理寄存器64512/CTA | 两CTA跨SM接纳，按真实资源占用/归还 |

静态事实来自文末源码及已有硬件记录，尚未运行复现地址混淆或CTA死等。

<a id="launch"></a>

## 1. cluster启动和身份：沿现有普通入口增加属性

1. 在 `libcuda/cuda_runtime_api.cc` 导出与本机CUDA头文件一致的 `cudaLaunchKernelExC`。从 `cudaLaunchConfig_t` 读取grid、block、dynamicSmemBytes、stream和 `cudaLaunchAttributeClusterDimension`；当前只接1×2×1及普通1×1×1，不增加其它启动属性框架。
2. 复用阶段0的函数查找、实参绑定和入队方法，把cluster维度放在本次 `kernel_info_t`。不要把属性写到共用函数后影响下一次launch，也不要在ExC里复制另一套参数布局算法。
3. 对CTA坐标b、cluster维度c，逐维计算 `cluster_id=b/c`、`local_cta=b%c`；rank按x最快展开：`local.x + c.x*(local.y+c.y*local.z)`。当前 `(7,1,0)` 得到cluster `(7,0,0)`、本地 `(0,1,0)`、rank1。
4. 特殊寄存器从该线程所属CTA身份读取，不从SM号反推。共享表归本次kernel所有，再以cluster坐标和rank索引；`ClusterCTAIdentifier` 本身没有kernel字段，优先利用现有kernel级容器隔离，不重复制造全局身份表。

**SASS复用：**复用 `kernel_info_t` 和既有CUDA cluster元数据。trace中的身份已被记录；PTX必须自己从launch生成。`simt_core_cluster` 是模拟器内部调度/互连分组，当前H100配置每组一核，不能把它当作CUDA的两个CTA cluster。

<a id="addresses"></a>

## 2. 地址：分清generic指针和cluster shared地址

generic指针是普通PTX可携带的统一地址；模拟器依其数值区间判断是global、shared还是线程私有local。cluster shared地址则用于表示组内目标rank及shared偏移，两者不应混用。

### generic区间的具体改法

当前 `GLOBAL_HEAP_START=0xC0000000`，shared和local预留区间从这个3GiB分界向下排。每SM的local预留为2048线程×16KiB=32MiB；只把SM上限改成132，local总预留就达到4224MiB，超过原分界，减法会发生无符号下溢。这是静态算式可确认的限制，不是已运行故障。

采用最小的64位区间调整：

1. shared地址槽上限设为256KiB，覆盖H100的233472B；SM地址槽数至少132，当前范围采用132即可。这些是**地址容量**，实际每SM资源仍由配置233472B限制，不能因槽大就允许多分配。
2. 将动态global堆起点和下方预留区一起移到足够高的64位边界，例如 `0x200000000ULL`（8GiB）。保持原布局公式集中计算 `TOTAL_SHARED_MEM`、`TOTAL_LOCAL_MEM`、`SHARED_GENERIC_START`、`LOCAL_GENERIC_START`、`STATIC_ALLOC_LIMIT`，不要只改某个比较分支。旧静态global区仍在最低的global合法区。
3. `gpu_malloc/gpu_mallocarray` 的起点、`shared_to_generic/generic_to_shared`、local对应转换、`isspace_*` 和 `whichspace` 使用同一组常量。现有堆游标为64位，但本次调用链中的中间变量/日志/指针参数也须核对不能截成32位；冻结PTX为 `.address_size 64`。
4. 不重写分页内存实现；`memory_space_impl<16*1024>` 的16KiB是页大小，不是CTA容量。普通shared偏移仍由当前CTA上下文解释，不在每次load上反复做launch已保证的容量检查。

这是拟采用的区间方案，执行agent需记录最终常量与受影响调用；若已有分支提供更小且等价的64位布局，可复用并说明。不要借此实现通用虚拟地址管理器或32位旧PTX兼容层。

### mapa与目标对象

沿已有mbarrier地址的低24位offset约定，当前PTX可采用 `rank<<24 | offset` 的32位内部编码；rank仅0/1。普通本地shared地址通过显式“本地”入口携带当前rank，不能把本地偏移的高字节0自动当成远端rank0。

`mapa` 保留offset、换成操作数指定rank；共用解码入口接收本次kernel/cluster上下文，返回目标CTA对象和offset。它不搬数据、不申请内存、不按目标SM号编码。阶段4远端到达与阶段5组播都调用这个入口；SASS已有编码在其适配入口转换，不能假定外部trace编码与新PTX编码恰好一致。

**SASS复用：**复用已有地址分类、cluster对象和屏障offset索引；新增的是PTX身份到真实shared对象的连接。既有普通global/local/shared访问按原功能路径回归。

<a id="scheduling"></a>

## 3. 调度：有两份资源才接纳一组

如果把66个rank0依次启动、用光资源后再等rank1，已经运行的CTA可能等不到同伴。这里描述的是错误实现的后果，不是当前PTX已经复现的死锁。

1. 在kernel的CTA枚举与 `issue_block2core` 上方增加当前cluster接纳步骤：枚举下一组两个CTA，先找到两份满足线程、寄存器和shared限制的SM槽位，再登记该组的保留资源和两个放置位置。
2. 两份资源同时可用才消耗该组CTA编号；没有完整位置就保留在待发队列，不能占住半组等待另半组。接纳后仍走原每CTA初始化和启动延迟，不把两个CTA当成零延迟同时开始。
3. 当前专家的两个CTA必须放到不同SM：`2*215040 > 233472`。选择确定的可用SM配对即可，不实现完整GPC拓扑；普通小CTA仍按自身资源允许同SM多驻留，不能全局限制每SM一个CTA。
4. 复用阶段0统一的shared占用入口和现有寄存器/线程占用计算。硬件专家总shared为216064B，其中215040B动态、1024B驱动保留、0B PTX静态；先核对ptxas declared/system是否已包含保留量，缺少时只在有证据的H100资源来源处补齐一次，不能把1024B伪装成静态数组或重复相加。
5. 当前专家物理寄存器是384×168=64512，不取PTX数千个虚拟寄存器声明求和。采用该入口已有ptxas/cubin资源记录；若提取值不符，定位提取来源并保存证据，不在调度器按函数名字硬编码168。运行中组间配额调整属于阶段6，不改变本CTA已占的总资源。

**SASS复用：**沿用容量判定、占用/释放、启动延迟和SM执行；cluster共同接纳是PTX需补的调度连接，不复制SASS轨迹的CTA发射顺序来代替执行。

<a id="barrier"></a>

## 4. cluster屏障和shared生命周期

`arrive` 表示本线程到达本轮屏障但可以继续；`wait` 在本轮所需线程均已到达前暂停当前执行路径。`.aligned` 要求一个warp一致执行，不表示“一条warp指令就只算一个线程”。当前384线程×2CTA，一轮需要对应768个线程的到达。

以本次kernel/cluster保存轮次、各CTA的到达位图或warp计数和等待warp集合。全warp一致执行时可一次登记32个线程，不能逐lane和warp处理各加一次。arrive不自动阻塞；wait绑定本线程最近到达的轮次，完成后释放该轮等待者并为下一轮重新计数。不同cluster互不等待。

relaxed arrive没有普通内存release保证；wait不能凭空补出该保证。屏障本身不排空TMA/WGMMA，也不发布任意shared数据。初始化mbarrier的发布由阶段4的专用fence完成。语义依据：[官方cluster屏障](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)。

shared对象按CTA创建并由当前cluster拥有；线程只引用所属对象。两个CTA合法结束且本组没有持有对象的在途操作后再销毁。阶段3先接普通退出；阶段4/5的远端通知/数据返回复用同一所有权，保留必要的未完成计数，不另建全局垃圾回收或超时强制放行。

跨SM到达与后续交付沿现有互连/队列可用位置接入。若现有SASS仅直接更新计数、没有DSM传输延迟，就明确加当前两目标所需的排队/交付事件，并记录采用的延迟/吞吐依据；此时不能声称已精确复现GPC网络。数据字节及主要通信校准归阶段5，本阶段至少区分发出和目标可见，不能隐藏为零成本全局写。

**SASS复用：**复用kernel级cluster容器、现有warp暂停/释放及通信服务位置；本阶段补PTX线程到达计数、轮次和shared对象所有权。单SM的CTA屏障计数不能直接当跨SM cluster屏障。

## 实施顺序、暂缓和交付记录

先ExC/身份→64位generic地址与mapa→两CTA成组接纳→cluster屏障/对象生命周期。先用小shared身份程序，再用真实215040B和132SM配置验证边界；不以实际专家所有指令已经实现为前置。

暂缓任意cluster形状、多GPU、完整GPC拓扑和当前未用的远端load/store指令。测试可直接观察共用目标解析接口，远端实际PTX副作用由阶段4补验，不为演示mapa额外实现一整个远端指令族。

| 源码/证据 | 本阶段入口 |
| --- | --- |
| [cuda_runtime_api.cc](../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) | ExC、阶段0共用启动与参数绑定 |
| [abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h)、[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) | 地址常量、kernel/cluster身份、cvta转换和线程shared初始化 |
| [gpu-sim.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc)、[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) | CTA接纳、资源占用/退出和屏障等待 |
| [kernel-times.json](../../workloads/moe/evidence/kernel-times.json)、[专家PTX](../../workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx) | 物理资源记录、实际cluster/特殊寄存器/屏障形式 |

实际实现/修复的接口与指令、地址布局、放置结果及通信近似填写到[验收记录](../validation/stage-3.md#results)。当前均未验证。
