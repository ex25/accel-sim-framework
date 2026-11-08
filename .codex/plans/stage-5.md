# 阶段 5 实现方案：描述矩阵、搬入shared、组播和写回global

状态：待执行方案，尚无本阶段实现或运行验收。目标对象/通知依赖阶段3/4；主机编码和描述符修改可独立先做。配套：[阶段5验收](../validation/stage-5.md)。

## 完成后应当能做什么

TMA根据一个128B的tensor map搬矩阵小块。tensor map记录基址、形状、行步长和shared排列方式；它不是矩阵本身，也不是阶段6供WGMMA使用的64位描述符。

例如同一块专家计算代码先处理专家A、再处理专家B，PTX会改tensor map的基址/尺寸并发布新对象。只记启动时主机参数会继续搬A的数据。输入需要“shared写好再通知”，输出需要“shared源已读完才可覆盖”；这两条都是待实现保证，尚未在当前PTX运行复现错误。

```text
主机编码128B对象 → PTX复制到shared并replace → cp_fenceproxy发布到global
                                                      ↓ acquire
输入：tensor map + tile坐标 → global请求 → shared数据/零填 → mbarrier完成
输出：tensor map + shared源 → 源读取完成 → global写请求 → 最终写确认
                                  ↑read wait允许复用源       ↑kernel完成仍要等
```

## 指令、接口及已有事实

| 完整指令/入口 | 本阶段交付 |
| --- | --- |
| `cudaGetDriverEntryPointByVersion` → `cuTensorMapEncodeTiled` | 返回模拟器自己的编码函数，写出可复制的128B对象 |
| `tensormap.replace.tile.global_address.shared::cta.b1024.b64` | 修改64位global基址 |
| `tensormap.replace.tile.global_dim.shared::cta.b1024.b32` | 修改维度，保留实际ordinal0～4 |
| `tensormap.replace.tile.global_stride.shared::cta.b1024.b64` | 按字节修改步长，保留实际ordinal0～3 |
| `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned` | 一次warp协作复制/发布128B |
| `fence.proxy.tensormap::generic.acquire.gpu` | 对指定对象建立后续描述符读取顺序 |
| `fence.proxy.async.shared::cta` | 普通shared写与后续异步读排序，和阶段6共用 |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` | 非组播TMA输入，真实交付后通知 |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint` | 一份源读取向当前两个CTA交付 |
| `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` | 从shared读输出并写回global |
| `cp.async.bulk.commit_group`、`cp.async.bulk.wait_group.read` | 本线程输出分组、read wait1/0 |

静态证据：编码查询入口未导出，本机真实driver无GPU；现有SASS有TMA合并、byte mask、OOB和组播通知。`shader.cc` 的OOB分支目前直接扣屏障计数，没有PTX零数据交付；TMA store在global ack清除outstanding，没有独立源读完事件；bulk wait初次与后续判断口径不同。以上说明哪些位置可复用、哪些必须接通，不是已运行故障清单。

<a id="descriptor"></a>

## 1. 主机与PTX共用一份128B内部格式

本轮已决定不与真实GPU不透明编码互换。对象必须能被原程序按字节复制、传参和修改，所以不能只存host指针或外部表索引；复制副本后修改一份，另一份应保持原值。

采用以下固定偏移，所有读写用固定宽度字段/字节存取，不依赖C++结构体padding：

| 偏移/B | 字段 | 大小/B |
| --- | --- | ---: |
| 0 | 模拟设备global基址u64 | 8 |
| 8 | global strides[4]，u64字节步长 | 32 |
| 40 | global dims[5]，u32实际元素数 | 20 |
| 60 | box dims[5]，u32一次tile的元素数 | 20 |
| 80 | element strides[5]，u32 | 20 |
| 100 | rank、dtype、interleave、swizzle、L2 promotion、OOB fill，各u32 | 24 |
| 124 | 保留字节，编码时置零 | 4 |

1. 在模拟器运行时导出查询入口，按本机头文件ABI处理原launcher的名字 `cuTensorMapEncodeTiled`、version12000及 `cudaEnableDefault`，返回真实可调用的模拟器函数指针。当前函数参数类型与 [launcher的Encode声明](../../workloads/moe/source/launcher.cu)一致；不用加载真实GPU driver再回退。
2. 编码函数写入重定位后的模拟global地址、当前rank2参数和枚举语义。API维度输入是u64，本内部格式u32足够当前实物尺寸；转换边界在入口处理，不为尚无消费者的超大维度扩格式。
3. 初始对象是待设备修改的占位描述：当前有小dim和stride0，不能套上“已经可搬真实矩阵”的额外检查拒绝编码。实际TMA使用的是replace/发布后的值。只在真正使用的入口落实当前格式要求，不加每次访问重复验证或未知格式静默memcpy回退。
4. 三种replace按字段偏移写值：global_address为64位；global_dim为实际元素数，不减1；global_stride是字节数，不右移4，也不再乘一次BF16的2B。当前PTX已经把元素步长乘2。
5. rank2仍保存被写入的inactive dims2～4、strides1～3。它们不参与本次二维地址计算，不能因ordinal≥rank而拒绝原程序，也不能把这些位置当第三维搬运。

**SASS复用：**没有主机编码或设备replace的数值实现可直接搬用；复用分页内存及普通shared读写。SASS提供的已展开地址不能替代这份对象。字段单位依据[原指令说明](../../workloads/moe/docs/ptx-instructions.md)和[官方replace](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)。

<a id="publication"></a>

## 2. 复制和发布：一次128B操作，顺序对准对象

`cp_fenceproxy` 由warp协作，把shared中的128B对象发布到global，使TMA的描述符读取看到更新。不能让32个lane各自再复制128B，产生32份流量。

1. 收齐本次合法warp参与，保存源/目的及size128。功能上复制一次；时序可把128B分成32个lane各4B的地址交给既有shared服务和global合并，或沿已有等价批量路径，不新增DMA引擎。
2. 描述符复制也产生真实访存/完成状态。在复制/发布所需顺序完成前，不允许相应acquire后的TMA读取未发布对象。复用已有pending store和fence顺序，必要时仅记该128B范围的未完成发布，不为全部global内存建立版本系统。
3. acquire按原操作数的地址和128B范围排序后续tensor map读取；不把它当成矩阵数据已搬完的通知。若发布来自其它线程，保留原mbarrier/线程协作，不在acquire里凭空同步所有生产者。
4. 一次TMA发起时保存其已取得的描述符字段。以后同址描述符被合法更新，不能反向改变旧请求的矩阵地址或尺寸；不用在返回时重新读描述符，也不永久缓存启动时的一份host参数。

**SASS复用：**复用global写合并、缓存和顺序完成；PTX补128B真实复制与对象范围。缓存提示的精细策略暂缓，但实际描述符读取/复制不能被当无流量操作。

<a id="input"></a>

## 3. 非组播输入：先确定逻辑元素，再确定shared位置

本轮实际rank2、元素2B（UINT16/BF16位存储）、element stride1、无interleave；box为64×64、64×128、32×64，swizzle为64B/128B。按实物组合实现，不要求三种box和两种swizzle的任意笛卡尔积都合法。

### 地址与数据算法

1. 解码二维起点 `(x0,y0)` 和box `(bx,by)`；坐标是有符号元素坐标，不能把负数先转巨大无符号地址。对每个 `0<=u<bx,0<=v<by`，令x=x0+u、y=y0+v。
2. 若x/y在维度内，global地址为 `base+2*x+stride[0]*y`；否则产生两个零字节，不发该元素global请求。输入OOB的tile仍完整写shared、仍完成整块的逻辑字节数 `2*bx*by`。
3. 先生成当前无interleave的tile内线性字节偏移 `q=2*(v*bx+u)`，再进行swizzle重排。swizzle是改shared地址，不改逻辑数值；相邻行的global stride不能被误用为紧凑shared行宽。
4. 对重复图案起点正确对齐的地址，64B/128B的核心分别是把地址bits[8:7]/[9:7]异或到bits[5:4]/[6:4]，可写成 `swizzle(a)=a xor (((a>>7)&mask)<<4)`，mask分别3/7；16B组内字节位置不变。非零起始图案相位按实际shared基址计算，不一律用q从0开始异或。先以官方16B块表和真实base样本核对，矩阵端也使用同一位重排规则。[官方swizzle表](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#swizzling-modes)。
5. 建立一份操作记录：UID、描述符快照、源有效范围、目标CTA/shared、每个返回对应的逻辑元素/OOB、目标屏障及轮次。保留一次tile所需的有效数据，不能为每个sector重复复制整块矩阵。

### 请求和完成接线

把有效元素的地址、有效byte mask送入已有TMA请求合并、L2/DRAM和返回仲裁。缓存事务按sector传输，但屏障完成量按逻辑tile字节计，不按放大的sector大小扣数。

返回处理先把对应数据写入目标shared，再调用阶段4的complete接口。OOB部分由零填交付事件完成，不再沿当前SASS分支在issue时无条件直接扣数。全OOB时没有DRAM返回，仍需正常的零填/交付完成路径。

可以沿现有返回粒度分批交付，也可在一个tile所有有效返回及必要shared服务完成后统一交付；首版选择一种并记录其时间近似。无论选哪种，屏障通过之前所有应消费字节必须可见；不能以CPU已算好地址代替建模的返回/写shared完成。

**SASS复用：**复用TMA地址合并、sector/byte mask、缓存和完成位置；补PTX真实数据、shared swizzle及准确逻辑字节。现有模型主要计global侧，缺少shared交付服务的地方用已有shared bank/服务能力补当前tile所需成本，不把新旧费用重复计算。

非组播输入和阶段6单次WGMMA各自通过后，立即执行[首次交叉测试](../validation/stage-5.md#early-integration)，不等待下面的组播、输出或寄存器配额。

<a id="multicast"></a>

## 4. 两CTA组播：一份global源，各目标独立交付

组播mask第r位选择rank r，当前两个目标只需1、2、3。先经阶段3定位被选中的目标对象，目标数据和对应屏障都按相同shared offset解析；不把发起CTA默认加到目标中。

保留一次源global读取，返回数据向被选目标分别交付。为每目标记录还欠的shared字节/通知，目标数据可见之后才更新其屏障。两目标共享源快照，不各做一次相同DRAM读取；但两份shared写及远端交付仍有工作量，不能因为global合并就少算。

可借用现有按offset找组播屏障的结构，但要核对现有回调是否先无条件更新发起者。PTX应严格以mask决定目标，尤其测试只选rank1的mask2。复用既有互连队列/投递能力，缺少DSM段时按当前两目标增加有限服务事件和参数依据，不实现任意GPC拓扑；报告global读取量与每目标交付量。

<a id="output"></a>

## 5. 输出写回：read完成和最终ack是两个事件

输出从本CTA shared取数据，按tensor map写global。对有效目标元素应用上节的shared位置映射读取原值；global目标越界元素不写内存，不能像输入那样越界写零。shared中可能是阶段6转换/stmatrix后的BF16位串，TMA不再做数值转换。

1. 发起时捕获本线程bulk组、描述符和shared源身份，登记shared读取服务。可以为功能执行保存源快照，但模型的 `read_done` 必须等描述符/源读取服务完成；不能在issue就宣布读完。
2. 读取完成后，输出请求携带这批值，不在未来ack时从已被复用的shared重新取值。global数据按已有写完成通路提交；最迟最终确认时功能内存与对应写入一致。
3. 分开保存 `read_pending` 与 `write_pending`。`cp.async.bulk.commit_group` 只提交当前线程之前未提交的输出；空组合法。组序号从0开始，C表示已经提交的组数（也是下一组序号）；`wait_group.read N` 只等序号 `<max(0,C-N)` 的旧组read完成，当前N=0/1。例如已有组0和1时C=2，wait1等组0，wait0等两组。
4. 最近N组可以仍在读；旧组已经读完但global ack没到时，read wait应允许复用源。最终kernel完成/stream后续消费者则仍要等相关global写确认，不能只看线程ret或read队列空。
5. 显式bulk opcode只访问bulk队列；普通cp.async和WGMMA的commit不能改它的类别。可复用阶段2的简单序号规则，但不必抽象成一个通用异步框架。

**SASS复用：**保留TMA global写请求、cache/DRAM和 `store_ack`；新增read完成事件及正确的组截止。当前ack清outstanding的代码继续负责最终写完，不再一份状态兼任两种含义。

<a id="proxy"></a>

## 6. ordinary shared到async的发布只维护一处

`fence.proxy.async.shared::cta` 排序普通shared写与后续TMA/WGMMA异步读取。阶段5或6谁先实现就由谁提供共用入口，另一阶段直接调用；不要各保留一套“可读标志”。

对本CTA/执行线程此前相关普通shared操作，沿既有完成信息建立后续async读取的先后。跨线程写入仍需程序自己的CTA/warp同步；fence本身不等待整个GPU，也不保证异步计算已经结束。功能shared即使在CPU上早写了值，后续异步服务也要受这个时间条件约束；不复制整份shadow shared来模拟当前不需要的弱内存历史。

**SASS复用：**沿用相关shared服务完成与fence调度位置；PTX补普通写与异步读取的真实先后约束。SASS trace不提供这里的完整功能值，不能据trace已有fence名称就省略数据连接。

## 交付、暂缓和证据入口

按内部对象→复制/修改/发布→非组播输入→首次联测→组播→输出/read wait推进。保留rank2的真实inactive字段、尾块/全OOB、描述符轮换、read wait1/0；暂缓rank>2搬运、其它dtype/interleave、精细L2 hint、原生编码互换和任意组播拓扑。

| 位置 | 本阶段接入内容 |
| --- | --- |
| [launcher.cu](../../workloads/moe/source/launcher.cu)、[cuda_runtime_api.cc](../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) | 查询ABI、模拟编码函数与真实地址传递 |
| [instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc)、[ptx_ir.h](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h) | 当前指令解析后的求值、descriptor字段与坐标 |
| [abstract_hardware_model.cc](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc) | TMA地址/byte mask/请求关联 |
| [shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)、[shader.h](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h) | OOB、返回、组播通知、shared服务、bulk组与store ack |
| [原指令说明](../../workloads/moe/docs/ptx-instructions.md)、[专家PTX](../../workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx) | 各操作数单位、两种wait和实际描述符更新 |

实际改动、指令/接口、复用及自测填写到 [阶段 5 实际实现记录](../execute/implementation/stage-5/README.md)；独立运行结论填写到 [实际验收记录](../execute/validation/stage-5/README.md)。本文件保留实现方案，不重复维护实际状态。
