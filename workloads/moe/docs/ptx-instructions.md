# Megatron-TE 的逐条 PTX 指令参考

本清单与[执行要求](execution-requirements.md)共同给出当前九个单卡用例的实现范围。指令支持包括实际计算、同步和访存行为；这里的分类记录准备 workload 时的源码核查结果。

当前九个模型/尺寸用例共 110 个去重 PTX 内核模块，202 种完整操作码写法。这里按 opcode 与全部修饰符去重，`wait_group 0/4` 属于同一写法。静态出现不代表每次前向都会执行，断言失败路径单列。

| 查阅分类 | 写法数 |
| --- | ---: |
| [需要新增的指令](#missing) | 34 |
| [已有指令需要扩展的写法](#extend) | 12 |
| [已有执行路径，随 workload 回归](#existing-path-unverified) | 154 |
| [仅断言失败分支](#failure-path) | 2 |

## 共用记号

`d/p` 为输出数据/输出 predicate，`a/b/c` 为输入；后缀决定类型，寄存器名字不能代替类型检查。`b16/b32/b64` 为位串，`s/u` 为有符号/无符号整数，`f32/bf16` 为不同浮点格式。`.rn` 为最近偶数舍入，`.rm` 为向负无穷舍入。`0fXXXXXXXX` 为 FP32 原始位模式。

`[addr]` 为字节地址，花括号为寄存器元组；`@p/@!p` 为执行谓词，`_` 只丢弃结果。`.sync/.aligned` 的参与范围依具体指令定义，不表示所有异步操作已完成。普通 cp.async、bulk/TMA、WGMMA 各有独立异步组。

tensor map 是 128 字节搬运描述符，CUDA 对象要求 64 字节对齐；它与 WGMMA 的 64 位 shared 矩阵描述符不同。当前 TE cluster 为 **1×2×1、两个 CTA**，不是多卡通信。

每项包含参数、行为、影响、当前 PTX 示例与官方条目；点击展开。中文说明只覆盖当前发出的形式，边界与未定义输入参见官方。窄整数合法写入宽寄存器时遵循[PTX 尺寸规则](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#operand-size-exceeding-instruction-type-size)。

<a id="missing"></a>

## 需要新增的指令

<details>
<summary><code>barrier.cluster.arrive.relaxed.aligned</code> · 需新增</summary>

```ptx
barrier.cluster.arrive.relaxed.aligned;
```

**参数：**无显式操作数；协作对象或异步组由线程/warpgroup 上下文决定。

**行为：**登记线程到达 cluster 屏障，不等待其他 warp 全部到达。`.aligned` 要求同 warp 一致执行；`.relaxed` 不提供普通内存访问的 release 排序。

**影响：**为后续 cluster wait 建立到达状态，不能当成 CTA 屏障，也不能将它当作共享数据发布。

实际示例（[PTX 第 653 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:653)）：

```ptx
barrier.cluster.arrive.relaxed.aligned;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)。

</details>

<details>
<summary><code>barrier.cluster.wait.aligned</code> · 需新增</summary>

```ptx
barrier.cluster.wait.aligned;
```

**参数：**无显式操作数；协作对象或异步组由线程/warpgroup 上下文决定。

**行为：**等待 cluster 对应参与者完成到达，默认 `.acquire`；同 warp 一致执行。当前配对的 arrive 是 relaxed，不能从 wait 推导任意此前写入都已被 release 发布。

**影响：**保证协作块推进/共享存储生命周期；遗漏会提前使用或释放对方存储，错误参与集合可能死锁。

实际示例（[PTX 第 657 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:657)）：

```ptx
barrier.cluster.wait.aligned;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)。

</details>

<details>
<summary><code>cp.async.bulk.commit_group</code> · 需新增</summary>

```ptx
cp.async.bulk.commit_group;
```

无显式操作数。

**行为：**将本线程之前尚未提交的 bulk async 操作组成一组并提交，包括本程序的 TMA shared→global 写回。

**影响：**为 bulk wait 建立组边界；组提交本身不表示传输完成，也不提交普通 cp.async/WGMMA。

实际示例（[PTX 第 1117 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1117)）：

```ptx
cp.async.bulk.commit_group;
```

静态 14 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)。

</details>

<details>
<summary><code>cp.async.bulk.tensor.2d.global.shared::cta.bulk_group</code> · 需新增</summary>

```ptx
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [tensorMap, {x,y}], [src];
```

| 参数 | 含义 |
| --- | --- |
| `tensorMap` | 128 字节 tensor map 描述符的地址，描述全局目标 |
| `x, y` | 目标 tile 的二维元素坐标 |
| `src` | 当前 CTA 共享内存源 tile 地址 |

**行为：**按 tensor map 的形状、步长和 swizzle 将 shared tile 异步写回 global，操作归入本线程的 bulk async group。

**影响：**TE GEMM epilogue 的输出路径；需与 shared generic 写入通过 proxy fence 排序，并保留传输、提交及读取完成阶段。

实际示例（[PTX 第 3706 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3706)）：

```ptx
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd794, {%r139, %r138}], [%r932];
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)。

</details>

<details>
<summary><code>cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint</code> · 需新增</summary>

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [dst], [tensorMap, {x,y}], [bar], cache_policy;
```

| 参数 | 含义 |
| --- | --- |
| `dst` | cluster 共享地址空间中的目的 tile 地址 |
| `tensorMap` | 全局 tensor map 地址 |
| `x, y` | 二维元素坐标 |
| `bar` | 接收完成字节数的 mbarrier 地址 |
| `cache_policy` | 64 位 L2 缓存策略描述符；不改变拷贝数据 |

**行为：**按 tensor map 将 global tile 异步复制到 shared；实际完成后向指定 mbarrier 报告完成的字节数。

**影响：**是 GEMM 输入 TMA 搬运；需要正确的地址映射、swizzle、事务字节计数和传输延迟，L2 hint 不等于屏障。

实际示例（[PTX 第 5872 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:5872)）：

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r1470], [%rd87, {%r206, %r205}], [%r1475], %rd507;
```

静态 10 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)。

</details>

<details>
<summary><code>cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint</code> · 需新增</summary>

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [dst], [tensorMap, {x,y}], [bar], cta_mask, cache_policy;
```

| 参数 | 含义 |
| --- | --- |
| `dst` | cluster 共享地址空间中的目的 tile 地址 |
| `tensorMap` | 全局 tensor map 地址 |
| `x, y` | 二维元素坐标 |
| `bar` | 接收完成字节数的 mbarrier 地址 |
| `cta_mask` | 16 位 cluster rank 掩码；当前 cluster 有 2 个 CTA |
| `cache_policy` | 64 位 L2 缓存策略描述符；不改变拷贝数据 |

**行为：**按 tensor map 将 global tile 异步复制到 shared，组播到掩码选中的 CTA；实际完成后向指定 mbarrier 报告完成的字节数。

**影响：**是 GEMM 输入 TMA 搬运；需要正确的地址映射、swizzle、事务字节计数和传输延迟，L2 hint 不等于屏障。

实际示例（[PTX 第 6626 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:6626)）：

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r1673], [%rd99, {%r2016, %r230}], [%r1674], %rs25, %rd679;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)。

</details>

<details>
<summary><code>cp.async.bulk.wait_group.read</code> · 需新增</summary>

```ptx
cp.async.bulk.wait_group.read N;
```

| 参数 | 含义 |
| --- | --- |
| `N` | 立即数：允许仍未达到 read 完成条件的最近 bulk group 数量 |

**行为：**等待较早的 bulk group 完成读取 tensor map 和源数据；当前 N=0。`.read` 不要求全局目标写入及其可见性已经全部完成。

**影响：**等待后可复用 shared 源缓冲区；仍要在内核完成/目标消费前满足真正写回完成的要求，不能把 `.read` 简化成所有目标写入即时完成。

实际示例（[PTX 第 1120 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1120)）：

```ptx
cp.async.bulk.wait_group.read 0;
```

静态 15 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)。

</details>

<details>
<summary><code>cp.async.cg.shared.global</code> · 需新增</summary>

```ptx
cp.async.cg.shared.global [dst], [src], cp_size, src_size;
```

| 参数 | 含义 |
| --- | --- |
| `dst` | 共享内存目的地址 |
| `src` | 全局内存源地址 |
| `cp_size` | 当前为 16 字节 |
| `src_size` | 实际读取字节数，0..cp_size；缺省等于 cp_size |

**行为：**发起异步 global→shared 拷贝；不足 cp_size 的尾部零填充。`.cg` 提示仅在全局级别缓存；用 commit/wait 追踪完成。

**影响：**参与 Triton GEMM 双缓冲；发起时不能立即解除目的共享内存依赖，predicate 为假则不发起。

实际示例（[PTX 第 324 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4ebef89fddede208.ptx:324)）：

```ptx
@%p133 cp.async.cg.shared.global [ %r18 + 0 ], [ %rd38 + 0 ], 0x10, %r19;
```

静态 452 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。

</details>

<details>
<summary><code>cp.async.commit_group</code> · 需新增</summary>

```ptx
cp.async.commit_group;
```

**参数：**无显式操作数；协作对象或异步组由线程/warpgroup 上下文决定。

**行为：**将本线程此前尚未提交的 cp.async 组成一个组；不等待拷贝完成。组中各拷贝之间不因此获得额外先后顺序。

**影响：**决定 wait_all/wait_group 所等待的普通异步拷贝集合，与 WGMMA 的组相互独立。

实际示例（[PTX 第 372 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4ebef89fddede208.ptx:372)）：

```ptx
cp.async.commit_group;
```

静态 78 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group)。

</details>

<details>
<summary><code>cp.async.wait_group</code> · 需新增</summary>

```ptx
cp.async.wait_group N;
```

| 参数 | 含义 |
| --- | --- |
| `N` | 立即数：允许仍未完成的最近普通 async group 数量 |

**行为：**等待到仅最近 N 组可以未完成，更早已提交组必须完成；N=0 等待所有已提交普通 async group。只涉及本线程发起的 cp.async。

**影响：**保护 shared 消费和复用；普通 cp.async group 与 bulk/TMA、WGMMA group 是独立队列。

实际示例（[PTX 第 7002 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4ebef89fddede208.ptx:7002)）：

```ptx
cp.async.wait_group 0x4;
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)。

</details>

<details>
<summary><code>elect.sync</code> · 需新增</summary>

```ptx
elect.sync d|p, membermask;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位输出，返回选中线程的 lane 编号，可用 `_` 丢弃 |
| `p` | 输出 predicate，仅选中 leader 为真 |
| `membermask` | 32 位参与 mask，bit 对应 lane；执行线程必须合法参与 |

**行为：**在参与的 predicated-active 线程中选出 leader。相同合法参与条件下选举确定，但 ISA 不保证一定选最低编号。等待参与线程执行相应选举。

**影响：**决定谁发起 TMA/修改屏障/描述符；双目标与 mask 错误会重复发起、漏发或产生错误 lane 编号。

实际示例（[PTX 第 199 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:199)）：

```ptx
elect.sync %rx|%px, %r318;
```

静态 29 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)。

</details>

<details>
<summary><code>fence.mbarrier_init.release.cluster</code> · 需新增</summary>

```ptx
fence.mbarrier_init.release.cluster;
```

**参数：**无显式操作数；协作对象或异步组由线程/warpgroup 上下文决定。

**行为：**将本线程此前 mbarrier 初始化的结果按 cluster 范围发布。它只针对初始化发布，不是任意 shared 写入的通用 fence。

**影响：**保证其他协作线程/块不会使用尚未可见的屏障初始状态；不等待 TMA 完成。

实际示例（[PTX 第 335 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:335)）：

```ptx
fence.mbarrier_init.release.cluster;
```

静态 5 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)。

</details>

<details>
<summary><code>fence.proxy.async.shared::cta</code> · 需新增</summary>

```ptx
fence.proxy.async.shared::cta;
```

无显式操作数。

**行为：**在当前 CTA shared 空间中对 generic 与 async 两种访问代理建立内存顺序，使之前普通共享写入可供后续异步操作读取。

**影响：**保证 stmatrix/普通 shared 写入产生的 tile 在 TMA 写回时可见；不是等待所有 TMA 完成的指令。

实际示例（[PTX 第 3694 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3694)）：

```ptx
fence.proxy.async.shared::cta;
```

静态 14 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)。

</details>

<details>
<summary><code>fence.proxy.tensormap::generic.acquire.gpu</code> · 需新增</summary>

```ptx
fence.proxy.tensormap::generic.acquire.gpu [map], size;
```

| 参数 | 含义 |
| --- | --- |
| `[map]` | global 中要获取的 tensor map 字节范围起始地址 |
| `size` | 立即数字节数，唯一支持值为 128 |

**行为：**在 GPU 范围对给定范围建立 generic 写入与后续 tensor-map 读取的单向 acquire 排序；不复制描述符。

**影响：**让 TMA 看到此前正确发布的新描述符；忽略可能沿用旧专家地址/尺寸。

实际示例（[PTX 第 3454 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3454)）：

```ptx
fence.proxy.tensormap::generic.acquire.gpu [%rd794], 128;
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)。

</details>

<details>
<summary><code>ldmatrix.sync.aligned.x4.m8n8.shared.b16</code> · 需新增</summary>

```ptx
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {d0,d1,d2,d3}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 本 lane 提供的一行共享内存起始地址；16 字节对齐 |
| `{d0..d3}` | 四个 b32 输出寄存器，各装两个 b16 元素；每个寄存器对应一张矩阵 |

**行为：**协作加载四张 8×8、16-bit 元素矩阵，总计 512 字节。lane 0..7、8..15、16..23、24..31 分别提供四张矩阵的八行地址。每张矩阵的第 i 行由 lane 4i..4i+3 各持两个连续元素。全部 32 lane 需执行同一指令；当前没有 `.trans`。

**影响：**需按 warp 的矩阵片段布局收集/分发寄存器和共享内存访问；按每线程独立连续 load/store 会把结果放错位置。

实际示例（[PTX 第 3571 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3571)）：

```ptx
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r1986, %r1987, %r1988, %r1989}, [%r874];
```

静态 16 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix)。

</details>

<details>
<summary><code>mapa.shared::cluster.u32</code> · 需新增</summary>

```ptx
mapa.shared::cluster.u32 d, a, rank;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位目标 cluster shared 地址 |
| `a` | 当前块 shared 中的字节地址 |
| `rank` | 目标线程块在同一 cluster 中的编号；当前范围 0..3 |

**行为：**保留 shared offset，将地址映射到指定 rank 的共享存储；不搬数据。

**影响：**决定远端屏障/数据访问落在哪一块。若退化为当前块地址，就会通知或读取错误对象。

实际示例（[PTX 第 3106 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3106)）：

```ptx
mapa.shared::cluster.u32  remAddr32, %r827, %r32;
```

静态 23 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa)。

</details>

<details>
<summary><code>mbarrier.arrive.expect_tx.shared::cta.b64</code> · 需新增</summary>

```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 state, [addr], tx_bytes;
```

| 参数 | 含义 |
| --- | --- |
| `state` | 输出 64 位阶段 token，当前 `_` 丢弃它 |
| `[addr]` | 本块 shared 的 64 位屏障地址，8 字节对齐 |
| `tx_bytes` | 本次登记的预期完成字节数，不是元素数、线程数或已完成字节数 |

**行为：**先使 tx-count 增加 tx_bytes，再到达一次使 pending-arrivals 减 1；默认 `.release.cta`。阶段完成仍要求 arrival 和事务计数均为零。

**影响：**把 TMA 的真正完成连接到消费者；若只数 arrival，数据没搬完就会开始计算。内部 tx-count 必须支持正负，完成通知可先于 expect 登记而暂时为负。

官方事务计数范围为 −(2^20−1)..+(2^20−1)；不能用 unsigned 下溢或对所有负值报错来实现。

实际示例（[PTX 第 6613 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:6613)）：

```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1669], %r29;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)。

</details>

<details>
<summary><code>mbarrier.arrive.shared::cluster.b64</code> · 需新增</summary>

```ptx
mbarrier.arrive.shared::cluster.b64 state, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `state` | 当前为 `_`；操作远端块屏障时必须丢弃返回值，不可读取远端 state token |
| `[addr]` | cluster 空间中本组目标块的 shared 屏障字节地址，8 字节对齐 |

**行为：**当前省略 count，每次到达使 pending-arrivals 减 1。省略 `.sem/.scope` 时按默认 `.release.cta`；`.shared::cluster` 是地址空间限定，不能当作 `.scope=.cluster`。

**影响：**影响对应阶段何时能完成；只有 pending-arrivals 与 tx-count 同时归零才推进 phase，并重装下一阶段 arrival 计数。

实际示例（[PTX 第 3107 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3107)）：

```ptx
mbarrier.arrive.shared::cluster.b64  _, [remAddr32];
```

静态 23 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)。

</details>

<details>
<summary><code>mbarrier.arrive.shared::cta.b64</code> · 需新增</summary>

```ptx
mbarrier.arrive.shared::cta.b64 state, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `state` | 输出 64 位阶段状态 token；当前为 `_`，仅丢弃返回值，仍执行到达操作 |
| `[addr]` | 本线程块的 shared 屏障字节地址，8 字节对齐 |

**行为：**当前省略 count，每次到达使 pending-arrivals 减 1。省略 `.sem/.scope` 时按默认 `.release.cta`；`.shared::cluster` 是地址空间限定，不能当作 `.scope=.cluster`。

**影响：**影响对应阶段何时能完成；只有 pending-arrivals 与 tx-count 同时归零才推进 phase，并重装下一阶段 arrival 计数。

实际示例（[PTX 第 3385 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3385)）：

```ptx
mbarrier.arrive.shared::cta.b64 _, [%r100];
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)。

</details>

<details>
<summary><code>mbarrier.expect_tx.shared::cta.b64</code> · 需新增</summary>

```ptx
mbarrier.expect_tx.shared::cta.b64 [bar], tx_count;
```

| 参数 | 含义 |
| --- | --- |
| `bar` | 当前 CTA shared 中 8 字节对齐的 mbarrier 地址 |
| `tx_count` | 本阶段新增的预期完成事务字节数 |

**行为：**给当前 mbarrier phase 增加预期事务量；它只登记 expected transaction count，不执行 arrival。

**影响：**phase 必须同时等到到达计数归零和传输字节完成；缺失会导致提前开始计算，或永远等待。

实际示例（[PTX 第 5876 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:5876)）：

```ptx
mbarrier.expect_tx.shared::cta.b64 [%r1475], %r34;
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx)。

</details>

<details>
<summary><code>mbarrier.init.shared::cta.b64</code> · 需新增</summary>

```ptx
mbarrier.init.shared::cta.b64 [addr], count;
```

| 参数 | 含义 |
| --- | --- |
| `[addr]` | 本块 shared 中 64 位屏障对象的地址，8 字节对齐 |
| `count` | 初始预期到达数，范围 1..2^20−1 |

**行为：**设 phase=0、expected-arrivals=pending-arrivals=count、tx-count=0；初始化本身不等待其他参与者。

**影响：**定义每阶段完成条件；其他线程/块使用前还必须正确发布初始化。屏障状态不能只用一个布尔 ready 表示。

实际示例（[PTX 第 237 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:237)）：

```ptx
mbarrier.init.shared::cta.b64 [%r355], %r382;
```

静态 36 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init)。

</details>

<details>
<summary><code>mbarrier.try_wait.parity.shared::cta.b64</code> · 需新增</summary>

```ptx
mbarrier.try_wait.parity.shared::cta.b64 p, [addr], phase_parity{, suspend_hint};
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate，真表示所等待阶段已完成 |
| `[addr]` | 本块 shared 屏障地址，8 字节对齐 |
| `phase_parity` | u32 值 0 或 1，等待的阶段奇偶值，不能任意传累增 phase 编号 |
| `suspend_hint（可选）` | u32 纳秒提示；当前既有无 hint，也有寄存器 hint 的形式 |

**行为：**按阶段奇偶测试完成，默认 `.acquire.cta`。等待当前未完成阶段时可返回假；观察到紧邻已完成阶段时返回真。try_wait 可以挂起并在完成或超时后返回，hint 不是保证等待时长，也不是 GPU 周期数。

**影响：**决定何时消费 TMA 数据和轮询重试；不能总返回真、把 hint 当周期，或默认一次调用必定等到完成。它允许当前/紧邻完成阶段的受限使用，不能比较任意久远 phase。

实际示例（[PTX 第 1208 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1208)）：

```ptx
mbarrier.try_wait.parity.shared::cta.b64 P1, [%r99], %r1977, %r692;
```

静态 45 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)。

</details>

<details>
<summary><code>redux.sync.add.s32</code> · 需新增</summary>

```ptx
redux.sync.add.s32 d, a, membermask;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 每个参与线程的 32 位结果 |
| `a` | 本线程 s32 输入 |
| `membermask` | 参与 lane 的 32 位掩码 |

**行为：**等待 mask 中未退出线程到齐，将所有成员的 a 相加并截断为 32 位，每个成员收到同一结果。

**影响：**用于 warp 内计数归约；不是跨 warp/CTA 的全局归约，需要协作执行与结果广播。

实际示例（[PTX 第 104 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/c479f139e809aa8b.ptx:104)）：

```ptx
redux.sync.add.s32 %r5, %r62, %r63;
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync)。

</details>

<details>
<summary><code>setmaxnreg.dec.sync.aligned.u32</code> · 需新增</summary>

```ptx
setmaxnreg.dec.sync.aligned.u32 reg_count;
```

| 参数 | 含义 |
| --- | --- |
| `reg_count` | 每线程新的绝对寄存器配额，24..256 且是 8 的倍数 |

**行为：**降低配额并将余量归还本 CTA 的寄存器池。同一 warpgroup 必须一致执行；数值不是本次归还/新增的数量。

**影响：**对应 producer/consumer 寄存器交接，会影响可推进时间和资源约束；数值存储与物理配额不能混为一谈。

实际示例（[PTX 第 5113 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:5113)）：

```ptx
setmaxnreg.dec.sync.aligned.u32 40;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)。

</details>

<details>
<summary><code>setmaxnreg.inc.sync.aligned.u32</code> · 需新增</summary>

```ptx
setmaxnreg.inc.sync.aligned.u32 reg_count;
```

| 参数 | 含义 |
| --- | --- |
| `reg_count` | 每线程新的绝对寄存器配额，24..256 且是 8 的倍数 |

**行为：**从本 CTA 的池申请更高配额；不足时阻塞，新取得寄存器内容未定义。同一 warpgroup 必须一致执行；数值不是本次归还/新增的数量。

**影响：**对应 producer/consumer 寄存器交接，会影响可推进时间和资源约束；数值存储与物理配额不能混为一谈。

实际示例（[PTX 第 956 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:956)）：

```ptx
setmaxnreg.inc.sync.aligned.u32 232;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)。

</details>

<details>
<summary><code>stmatrix.sync.aligned.x4.m8n8.shared.b16</code> · 需新增</summary>

```ptx
stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr], {a0,a1,a2,a3};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 本 lane 提供的一行共享内存起始地址；16 字节对齐 |
| `{a0..a3}` | 四个 b32 输入寄存器，各装两个 b16 元素；每个寄存器对应一张矩阵 |

**行为：**协作存储四张 8×8、16-bit 元素矩阵，总计 512 字节。lane 0..7、8..15、16..23、24..31 分别提供四张矩阵的八行地址。每张矩阵的第 i 行由 lane 4i..4i+3 各持两个连续元素。全部 32 lane 需执行同一指令；当前没有 `.trans`。

**影响：**需按 warp 的矩阵片段布局收集/分发寄存器和共享内存访问；按每线程独立连续 load/store 会把结果放错位置。

实际示例（[PTX 第 3685 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3685)）：

```ptx
stmatrix.sync.aligned.x4.m8n8.shared.b16 [%r156], {%r892, %r893, %r894, %r895};
```

静态 16 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)。

</details>

<details>
<summary><code>tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned</code> · 需新增</summary>

```ptx
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
    [dst], [src], size;
```

| 参数 | 含义 |
| --- | --- |
| `[dst]` | global 中目的 tensor map 字节地址 |
| `[src]` | 本块 shared 中已修改的 tensor map 字节地址 |
| `size` | 立即数字节数，唯一支持值为 128 |

**行为：**将 128 字节描述符由 shared 复制到 global，并建立 generic→tensormap 的 GPU 范围 release 排序。`.sync.aligned` 要求整个 warp 一致协作执行，不是独立 lane 的普通 memcpy。

**影响：**把更换专家后的地址/尺寸/stride 正式发布；只有 replace 而没有相应发布，后续搬运仍可能看到旧内容。

实际示例（[PTX 第 1131 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1131)）：

```ptx
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd794], [%r642], 128;
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy)。

</details>

<details>
<summary><code>tensormap.replace.tile.global_address.shared::cta.b1024.b64</code> · 需新增</summary>

```ptx
tensormap.replace.tile.global_address.shared::cta.b1024.b64 [map], new_address;
```

| 参数 | 含义 |
| --- | --- |
| `[map]` | shared::cta 中 128 字节 opaque tensor map 对象的字节地址；描述符大小不等于对齐值 |
| `new_address` | 新矩阵 global 基址，64 位；当前 BF16 布局满足 TMA 基址对齐要求 |

**行为：**修改描述符的 global_address 字段；b1024 表示完整对象 1024 bit=128 字节，b64 是替换字段宽度。

**影响：**更换专家矩阵基址，后续 TMA 必须读新专家；修改本身不搬矩阵，也未完成向 tensor-map 访问的发布。

实际示例（[PTX 第 1061 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1061)）：

```ptx
tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%r627], %rd274;
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)。

</details>

<details>
<summary><code>tensormap.replace.tile.global_dim.shared::cta.b1024.b32</code> · 需新增</summary>

```ptx
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [map], ordinal, new_dim;
```

| 参数 | 含义 |
| --- | --- |
| `[map]` | shared::cta 中 128 字节 opaque tensor map 对象的字节地址；描述符大小不等于对齐值 |
| `ordinal` | 立即数待改维度序号；当前出现 0、1、2 |
| `new_dim` | 该维度实际元素数，32 位；当前不是减 1 后的编码 |

**行为：**修改指定 global 维度大小；描述符对象大小与字段宽度分别由 b1024、b32 指定。

**影响：**专家 token 数不同，尺寸决定边界和 OOB；旧尺寸可能读越界或漏数据。

实际示例（[PTX 第 1073 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1073)）：

```ptx
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%rd288], 0, %r94;
```

静态 40 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)。

</details>

<details>
<summary><code>tensormap.replace.tile.global_stride.shared::cta.b1024.b64</code> · 需新增</summary>

```ptx
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [map], ordinal, new_stride;
```

| 参数 | 含义 |
| --- | --- |
| `[map]` | shared::cta 中 128 字节 opaque tensor map 对象的字节地址；描述符大小不等于对齐值 |
| `ordinal` | 立即数显式 stride 的序号；0 对应逻辑维 1，1 对应逻辑维 2；最内层元素步长隐含 |
| `new_stride` | 当前 CUDA 12.8 分支传入的字节步长，64 位；BF16 元素 stride 已乘 2 |

**行为：**修改对应 global stride。不要将 CUTLASS 旧编译器分支的右移 4 位规则套到当前产物，也不能把 tensor-map 的不透明物理编码直接当作该参数单位。

**影响：**决定相邻行/更外层维度的地址；单位错 16 倍或少算 BF16 的 2 字节会搬错矩阵。

单位同时核对 [CUDA 12.8.1 tensor-map API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) 与 [冻结 CUTLASS 的替换分支](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/include/cute/arch/copy_sm90_desc.hpp#L324)。

实际示例（[PTX 第 1088 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1088)）：

```ptx
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%rd288], 0, %rd282;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)。

</details>

<details>
<summary><code>wgmma.commit_group.sync.aligned</code> · 需新增</summary>

```ptx
wgmma.commit_group.sync.aligned;
```

**参数：**无显式操作数；协作对象或异步组由线程/warpgroup 上下文决定。

**行为：**把此前发起但尚未提交的 WGMMA 纳入新异步组；没有待提交操作时也可提交空组。状态按 warpgroup 协作维护。 `.sync` 要求 warp 内到达，`.aligned` 要求整个 warpgroup 一致执行；不能把这些限定当作矩阵计算已完成。

**影响：**定义后续 wait_group 等待的组边界，不等于执行完成；漏组或过早完成会放行未算好的结果。

实际示例（[PTX 第 1828 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1828)）：

```ptx
wgmma.commit_group.sync.aligned;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-commit-group)。

</details>

<details>
<summary><code>wgmma.fence.sync.aligned</code> · 需新增</summary>

```ptx
wgmma.fence.sync.aligned;
```

**参数：**无显式操作数；协作对象或异步组由线程/warpgroup 上下文决定。

**行为：**排序本 warpgroup 此前对相关累加器/A 寄存器的普通访问，与后续 WGMMA 对这些寄存器的访问；第一次 WGMMA 前及相关寄存器被普通操作改写后需要该顺序。 `.sync` 要求 warp 内到达，`.aligned` 要求整个 warpgroup 一致执行；不能把这些限定当作矩阵计算已完成。

**影响：**影响异步矩阵计算是否读到正确寄存器值；不能用它替代 shared 数据发布或 TMA 完成等待。

实际示例（[PTX 第 1634 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1634)）：

```ptx
wgmma.fence.sync.aligned;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-fence)。

</details>

<details>
<summary><code>wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16</code> · 需新增</summary>

```ptx
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
    {D0,…,D63}, a_desc, b_desc, scale_d, scale_a, scale_b, trans_a, trans_b;
```

| 参数 | 含义 |
| --- | --- |
| `{D0,…,D63}` | 每线程 64 个 FP32 累加寄存器，既是输出，也在 scale_d 为真时提供旧 D |
| `a_desc` | 64 位共享内存矩阵 A 描述符：编码起始地址、布局/步长、swizzle；不是普通指针 |
| `b_desc` | 64 位共享内存矩阵 B 描述符，同样需要解码 |
| `scale_d` | predicate：假为 D=A×B；真为 D=A×B+D |
| `scale_a` | 立即数 +1 或 -1，控制 A 是否取负；当前为 1 |
| `scale_b` | 立即数 +1 或 -1，控制 B 是否取负；当前为 1 |
| `trans_a` | 立即数 0/1，控制 A 转置；当前 0 |
| `trans_b` | 立即数 0/1，控制 B 转置；当前 0 |

**行为：**四个连续 warp、128 个线程协作发起异步矩阵乘加。m64n128k16 对应 M=64、N=128、K=16；BF16×BF16，FP32 累加。`.sync` 是 warp 内到达要求，`.aligned` 要求整个 warpgroup 执行同一条指令，均不表示乘加已经完成。

**影响：**必须正确组合各 lane 持有的 64 项结果，解读 A/B 的 shared 布局，维护异步组依赖；一个协作运算不能因分属四个 warp 而重复计成四次完整计算。

官方对 BF16 WGMMA 规定至少单精度累加，但没有规定累加顺序、中间舍入和次正规输入处理，因此不能要求它与逐项 FP32 FMA 的标量实现逐位相同。验证不同后端时需识别中间 BF16 舍入边界；同一二进制的模拟器对比仍保留严格检查。

共享内存矩阵描述符的位布局见同一官方条目及其 Matrix Descriptor Format 链接。输入搬运的完成与共享内存可见性由 TMA/mbarrier/fence 等另行保证。

实际示例（[PTX 第 1659 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1659)）：

```ptx
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%f7420,   %f7419,   %f7418,   %f7417,   %f7416,   %f7415,   %f7414,   %f7413,    %f7412,   %f7411,   %f7410,  %f7409,  %f7408,  %f7407,  %f7406,  %f7405,   %f7404,  %f7403,  %f7402,  %f7401,  %f7400,  %f7399,  %f7398,  %f7397,   %f7396,  %f7395,  %f7394,  %f7393,  %f7392,  %f7391,  %f7390,  %f7389,   %f7388,  %f7387,  %f7386,  %f7385,  %f7384,  %f7383,  %f7382,  %f7381,   %f7380,  %f7379,  %f7378,  %f7377,  %f7376,  %f7375,  %f7374,  %f7373,   %f7372,  %f7371,  %f7370,  %f7369,  %f7368,  %f7367,  %f7366,  %f7365,   %f7364,  %f7363,  %f7362,  %f7361,  %f7360,  %f7359,  %f7358,  %f7357}, %rd302, %rd305, p,    1,  1,  0,  0;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async)。

</details>

<details>
<summary><code>wgmma.wait_group.sync.aligned</code> · 需新增</summary>

```ptx
wgmma.wait_group.sync.aligned N;
```

| 参数 | 含义 |
| --- | --- |
| `N` | 非负立即数，允许尚未完成的最近组数；当前 N=1 和 N=0 |

**行为：**等待较早组完成，直到最近仍在途的已提交组不超过 N。N=0 等待此前全部已提交组；N=1 允许最近一个组继续运行。访问某组累加结果前，必须已等待覆盖该组的完成。 `.sync` 要求 warp 内到达，`.aligned` 要求整个 warpgroup 一致执行；不能把这些限定当作矩阵计算已完成。

**影响：**决定计算流水线重叠与结果何时可被普通指令读取；把 wait 1 改成 wait 0 会增加不必要串行等待。

实际示例（[PTX 第 2839 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:2839)）：

```ptx
wgmma.wait_group.sync.aligned 1;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group)。

</details>

<a id="extend"></a>

## 已有指令需要扩展的写法

<details>
<summary><code>bar.warp.sync</code> · 需扩展</summary>

```ptx
bar.warp.sync membermask;
```

| 参数 | 含义 |
| --- | --- |
| `membermask` | 32 位参与 mask；当前 -1 表示 0xffffffff，即全部 32 个 lane |

**行为：**等待指定参与线程执行对应 warp barrier，并建立参与者之间的相关内存排序。与 shfl/elect 的寄存器交换/选举同步作用不同。

**影响：**影响同 warp 内 shared 数据的安全复用；lexer 将其当 NOP 不能自动视为语义正确，需要实现或证明当前执行模型对固定用法等价。

实际示例（[PTX 第 1018 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1018)）：

```ptx
bar.warp.sync 	-1;
```

静态 9 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync)。

</details>

<details>
<summary><code>bfind.shiftamt.u32</code> · 需扩展</summary>

```ptx
bfind.shiftamt.u32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位输出移位量 |
| `a` | u32 输入 |

**行为：**a≠0 时，d=31−最高置位位置，即左移至最高位所需的位数；a=0 时 d=0xffffffff。

**影响：**当前 bfind 实现未处理 `.shiftamt`；按普通 bfind 返回位置会算错，零输入循环也需核查。

实际示例（[PTX 第 821 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:821)）：

```ptx
bfind.shiftamt.u32 	%r553, %r552;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind)。

</details>

<details>
<summary><code>cvt.f32.bf16</code> · 需扩展</summary>

```ptx
cvt.f32.bf16 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数结果 |
| `a` | 16 位 BF16 浮点数输入 |

**行为：**将 BF16 数值精确扩宽为 FP32，不能按 FP16 解码。NaN 的具体 payload 未作保证。

**影响：**改变数值精度、掩码或地址偏移；需要区分浮点格式转换与整数截断/扩展。

实际示例（[PTX 第 71 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0246c7163d0f0932.ptx:71)）：

```ptx
cvt.f32.bf16 %r19, %rs9;
```

静态 732 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

</details>

<details>
<summary><code>cvt.rn.bf16.f32</code> · 需扩展</summary>

```ptx
cvt.rn.bf16.f32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 16 位 BF16 浮点数结果 |
| `a` | FP32 浮点数输入 |

**行为：**将 FP32 按最近偶数舍入为 BF16；可溢出为无穷，不能用取低 16 位代替。

**影响：**改变数值精度、掩码或地址偏移；需要区分浮点格式转换与整数截断/扩展。

实际示例（[PTX 第 272 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:272)）：

```ptx
cvt.rn.bf16.f32 %rs17, %r50;
```

静态 138 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

</details>

<details>
<summary><code>cvt.rn.bf16x2.f32</code> · 需扩展</summary>

```ptx
cvt.rn.bf16x2.f32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位打包结果 |
| `a, b` | 两个 FP32 输入 |

**行为：**分别按最近偶数舍入为 BF16；a 放 d 的高 16 位，b 放低 16 位。

**影响：**影响 epilogue 的成对 BF16 写回；上下半字交换会打乱矩阵。

实际示例（[PTX 第 3653 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3653)）：

```ptx
cvt.rn.bf16x2.f32 %r892, %f6909, %f6910;
```

静态 64 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

</details>

<details>
<summary><code>ld.global.L1::evict_first.b32</code> · 需扩展</summary>

```ptx
ld.global.L1::evict_first.b32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个原始位串32 位值 |

**行为：**从设备全局内存读取连续 4 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。`L1::evict_first/last` 分别提示优先/最后逐出缓存，不改变读取的值或同步语义。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 70 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/5ee83ea169c92aec.ptx:70)）：

```ptx
@%p1 ld.global.L1::evict_first.b32 { %r2 }, [ %rd1 + 0 ];
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.L1::evict_first.b8</code> · 需扩展</summary>

```ptx
ld.global.L1::evict_first.b8 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个原始位串8 位值 |

**行为：**从设备全局内存读取连续 1 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。`L1::evict_first/last` 分别提示优先/最后逐出缓存，不改变读取的值或同步语义。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 55 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:55)）：

```ptx
@%p1 ld.global.L1::evict_first.b8 { %rs1 }, [ %rd1 + 0 ];
```

静态 10 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.L1::evict_last.b32</code> · 需扩展</summary>

```ptx
ld.global.L1::evict_last.b32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个原始位串32 位值 |

**行为：**从设备全局内存读取连续 4 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。`L1::evict_first/last` 分别提示优先/最后逐出缓存，不改变读取的值或同步语义。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 83 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:83)）：

```ptx
@%p4 ld.global.L1::evict_last.b32 { %r7 }, [ %rd7 + 0 ];
```

静态 52 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>shfl.sync.bfly.b32</code> · 需扩展</summary>

```ptx
shfl.sync.bfly.b32 d|p, a, b, c, membermask;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位目标值 |
| `p` | 可选：源 lane 是否有效；当前实例未接收 |
| `a` | 本 lane 提供的 32 位输入 |
| `b` | lane 偏移，取低 5 位 |
| `c` | 低 5 位 clamp，高段含分段掩码 |
| `membermask` | 参与该指令的 32 位 lane 掩码 |

**行为：**`.bfly` 使用 laneid XOR b 选择源；边界按 c 的 clamp/分段规则判断，无效源保持本 lane 的 a。所有被 mask 指定且未退出的线程必须以相同 mask 执行，满足后才能继续。

**影响：**用于 warp 内归约/扫描；现有旧式 shuffle 不能忽略新增 membermask。它不提供共享内存屏障。

实际示例（[PTX 第 77 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:77)）：

```ptx
shfl.sync.bfly.b32	%r12, %r11, 16, 31, -1;
```

静态 332 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。

</details>

<details>
<summary><code>shfl.sync.idx.b32</code> · 需扩展</summary>

```ptx
shfl.sync.idx.b32 d|p, a, b, c, membermask;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位输出数据 |
| `p` | 输出 predicate，表示计算得到的源索引在 clamp/segment 范围内 |
| `a` | 各参与 lane 各自持有的 32 位源值 |
| `b` | 源 lane 索引，使用 b[4:0] |
| `c` | 打包参数：c[4:0] 为 clamp，c[12:8] 为 segment mask |
| `membermask` | 32 位参与 mask；只有合法参与、活动的源线程才能提供数据 |

**行为：**令 lane 为本线程编号，seg=c[12:8]，clamp=c[4:0]；minLane=lane&seg，maxLane=minLane|(clamp&~seg)，j=minLane|(b[4:0]&~seg)。若 j<=maxLane，则 p=true、d 读取 lane j 的 a；否则 p=false、d 取自己的 a。计算限于 5 位 lane 字段。

**影响：**影响广播的元数据、指针和 lane 选择；p 只反映索引范围，不检测源是否 active 或属于 membermask。读取非活动/非参与源未定义。此同步不是 shared/global 内存 fence。

实际示例（[PTX 第 180 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:180)）：

```ptx
shfl.sync.idx.b32 	%r338|%p13, %r21, %r320, %r337, %r318;
```

静态 29 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。

</details>

<details>
<summary><code>shfl.sync.up.b32</code> · 需扩展</summary>

```ptx
shfl.sync.up.b32 d|p, a, b, c, membermask;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位目标值 |
| `p` | 可选：源 lane 是否有效；当前实例未接收 |
| `a` | 本 lane 提供的 32 位输入 |
| `b` | lane 偏移，取低 5 位 |
| `c` | 低 5 位 clamp，高段含分段掩码 |
| `membermask` | 参与该指令的 32 位 lane 掩码 |

**行为：**`.up` 从 laneid−b 选择源；边界按 c 的 clamp/分段规则判断，无效源保持本 lane 的 a。所有被 mask 指定且未退出的线程必须以相同 mask 执行，满足后才能继续。

**影响：**用于 warp 内归约/扫描；现有旧式 shuffle 不能忽略新增 membermask。它不提供共享内存屏障。

实际示例（[PTX 第 63 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/23ba58950db8f4f0.ptx:63)）：

```ptx
shfl.sync.up.b32	%r10, %r9, 2, 0, -1;
```

静态 72 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。

</details>

<details>
<summary><code>vote.sync.ballot.b32</code> · 需扩展</summary>

```ptx
vote.sync.ballot.b32 d, p, membermask;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位投票结果 |
| `p` | 每个参与 lane 的 predicate |
| `membermask` | 参与 lane 掩码 |

**行为：**等待 mask 中未退出线程执行相同指令；结果 bit i 对应参与且 predicate 为真的 lane i。

**影响：**参与 token 排名和压缩；必须使用显式成员掩码，不能直接用当前 active mask 替代。

实际示例（[PTX 第 552 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:552)）：

```ptx
vote.sync.ballot.b32 	%r456, %p7, %r166;
```

静态 10 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync)。

</details>

<a id="existing-path-unverified"></a>

## 已有执行路径，随 workload 回归

<details>
<summary><code>abs.f32</code> · 已有路径，待回归</summary>

```ptx
abs.f32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `参数` | a 是 FP32 输入，d 是 FP32 输出。 |

**行为：**清除 FP32 输入的符号位，得到绝对值。

**影响：**影响数值、索引或执行掩码；使用现有对应功能路径后仍需回归结果与依赖。

实际示例（[PTX 第 814 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:814)）：

```ptx
abs.f32 	%f37, %f5;
```

静态 30 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-abs)。

</details>

<details>
<summary><code>add.f32</code> · 已有路径，待回归</summary>

```ptx
add.f32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数结果 |
| `a` | FP32 浮点数左操作数 |
| `b` | FP32 浮点数右操作数 |

**行为：**计算 `d = a + b`。默认最近偶数舍入；没有 `.ftz`，不能主动清零次正规数。

**影响：**影响数值或索引；消费者须等待 d 就绪，执行延迟进入相应运算管线。

实际示例（[PTX 第 198 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:198)）：

```ptx
add.f32 	%f43, %f3, 0f3F800000;
```

静态 68 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-add)。

</details>

<details>
<summary><code>add.s32</code> · 已有路径，待回归</summary>

```ptx
add.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位有符号整数结果 |
| `a` | 32 位有符号整数左操作数 |
| `b` | 32 位有符号整数右操作数 |

**行为：**计算 `d = a + b`。保留指令位宽内结果，当前没有饱和修饰。

**影响：**影响数值或索引；消费者须等待 d 就绪，执行延迟进入相应运算管线。

实际示例（[PTX 第 46 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:46)）：

```ptx
add.s32 	%r10, %r9, 4096;
```

静态 1407 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-add)。

</details>

<details>
<summary><code>add.s64</code> · 已有路径，待回归</summary>

```ptx
add.s64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位有符号整数结果 |
| `a` | 64 位有符号整数左操作数 |
| `b` | 64 位有符号整数右操作数 |

**行为：**计算 `d = a + b`。保留指令位宽内结果，当前没有饱和修饰。

**影响：**影响数值或索引；消费者须等待 d 就绪，执行延迟进入相应运算管线。

实际示例（[PTX 第 49 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:49)）：

```ptx
add.s64 	%rd1, %rd8, %rd10;
```

静态 1263 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-add)。

</details>

<details>
<summary><code>and.b16</code> · 已有路径，待回归</summary>

```ptx
and.b16 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 16 位位串结果 |
| `a` | 16 位位串输入 |
| `b` | 16 位位串输入 |

**行为：**逐位 AND；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 58 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:58)）：

```ptx
and.b16  	%rs3, %rs1, 255;
```

静态 41 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-and)。

</details>

<details>
<summary><code>and.b32</code> · 已有路径，待回归</summary>

```ptx
and.b32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位位串结果 |
| `a` | 32 位位串输入 |
| `b` | 32 位位串输入 |

**行为：**逐位 AND；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 40 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:40)）：

```ptx
and.b32  	%r6, %r5, 31;
```

静态 447 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-and)。

</details>

<details>
<summary><code>and.b64</code> · 已有路径，待回归</summary>

```ptx
and.b64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位位串结果 |
| `a` | 64 位位串输入 |
| `b` | 64 位位串输入 |

**行为：**逐位 AND；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 870 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:870)）：

```ptx
and.b64  	%rd55, %rd215, %rd216;
```

静态 14 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-and)。

</details>

<details>
<summary><code>and.pred</code> · 已有路径，待回归</summary>

```ptx
and.pred d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 布尔 predicate结果 |
| `a` | 布尔 predicate输入 |
| `b` | 布尔 predicate输入 |

**行为：**布尔 AND；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 70 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:70)）：

```ptx
and.pred  	%p12, %p1, %p10;
```

静态 93 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-and)。

</details>

<details>
<summary><code>atom.shared.add.u32</code> · 已有路径，待回归</summary>

```ptx
atom.shared.add.u32 d, [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 原来的 u32 值 |
| `addr` | 共享内存地址 |
| `a` | u32 加数 |

**行为：**原子地读取旧值、将旧值+a 按 32 位写回，并返回旧值。默认 relaxed 顺序，shared 原子作用于该共享地址。

**影响：**保证竞争计数不会丢更新；必须建模原子串行化，不能当成一次普通 load 加 store。

实际示例（[PTX 第 614 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:614)）：

```ptx
atom.shared.add.u32 	%r190, [%r189], %r460;
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)。

</details>

<details>
<summary><code>bar.sync</code> · 已有路径，待回归</summary>

```ptx
bar.sync barrier_id;
```

| 参数 | 含义 |
| --- | --- |
| `barrier_id` | 本块逻辑屏障编号 0..15；当前全部使用 0 |

**行为：**当前未显式给线程数，等待本块所有未退出参与线程到达；完成后可复用，并保证此前相关访存完成及可见性。

**影响：**影响 helper 的 shared 数据使用；忽略会读到尚未写好的值。与 mbarrier 内存对象不是同一套机制。

实际示例（[PTX 第 139 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:139)）：

```ptx
bar.sync 	0;
```

静态 194 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier)。

</details>

<details>
<summary><code>bfe.u32</code> · 已有路径，待回归</summary>

```ptx
bfe.u32 d, a, pos, len;
```

| 参数 | 含义 |
| --- | --- |
| `d` | u32 输出 |
| `a` | 32 位源值 |
| `pos, len` | 最低 8 位指定起始位与位数 |

**行为：**从 a 的 bit pos 起抽取 len 位，低位对齐并零扩展；超出源宽度的位为零。

**影响：**用于排序键和线程索引位域，错误会改变路由。

实际示例（[PTX 第 59 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4ebef89fddede208.ptx:59)）：

```ptx
bfe.u32 	%r225, %r2, 5, 2;
```

静态 46 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe)。

</details>

<details>
<summary><code>bfi.b32</code> · 已有路径，待回归</summary>

```ptx
bfi.b32 d, a, b, pos, len;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位输出 |
| `a` | 待插入的低位源 |
| `b` | 保留其他位的原值 |
| `pos, len` | 最低 8 位指定目标起始位与位数 |

**行为：**将 a 的低 len 位插入 b 从 bit pos 开始的范围，保留 b 的其他位；仅写 0..31 位。

**影响：**影响位打包和排序；不能交换两个源。

实际示例（[PTX 第 685 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:685)）：

```ptx
bfi.b32 %r468, %r461, %r468, %r3, %r214;
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi)。

</details>

<details>
<summary><code>bra</code> · 已有路径，待回归</summary>

```ptx
@p bra target;
```

| 参数 | 含义 |
| --- | --- |
| `target` | 当前函数内目标 label |
| `@p（指令前缀，可省略）` | predicate 为真才跳转；@!p 表示取反，省略则无条件 |

**行为：**满足 guard 时跳到 target，否则顺序执行。允许 warp 内分歧。

**影响：**决定动态指令流、活动 mask、循环次数和分歧处理。

实际示例（[PTX 第 130 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:130)）：

```ptx
@%p11 bra 	$L__BB0_2;
```

静态 400 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#control-flow-instructions-bra)。

</details>

<details>
<summary><code>bra.uni</code> · 已有路径，待回归</summary>

```ptx
@p bra.uni target;
```

| 参数 | 含义 |
| --- | --- |
| `target` | 当前函数内目标 label |
| `@p（指令前缀，可省略）` | predicate 为真才跳转；@!p 表示取反，省略则无条件 |

**行为：**满足 guard 时跳到 target，否则顺序执行。`.uni` 保证当前活动线程的 guard 与目标一致，无分歧。

**影响：**决定动态指令流、活动 mask、循环次数和分歧处理。

实际示例（[PTX 第 96 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:96)）：

```ptx
bra.uni 	$L__BB0_1;
```

静态 49 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#control-flow-instructions-bra)。

</details>

<details>
<summary><code>brev.b32</code> · 已有路径，待回归</summary>

```ptx
brev.b32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位输出 |
| `a` | 32 位输入 |

**行为：**把 a 的 32 个 bit 顺序完全反转：d[i]=a[31−i]。

**影响：**用于位序和索引，不是字节序转换。

实际示例（[PTX 第 820 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:820)）：

```ptx
brev.b32 	%r552, %r44;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev)。

</details>

<details>
<summary><code>cvt.s64.s32</code> · 已有路径，待回归</summary>

```ptx
cvt.s64.s32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位有符号整数结果 |
| `a` | 32 位有符号整数输入 |

**行为：**将 s32 符号扩展为 s64，保持负数的值。

**影响：**改变数值精度、掩码或地址偏移；需要区分浮点格式转换与整数截断/扩展。

实际示例（[PTX 第 50 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:50)）：

```ptx
cvt.s64.s32 	%rd11, %r10;
```

静态 116 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

</details>

<details>
<summary><code>cvt.u16.u32</code> · 已有路径，待回归</summary>

```ptx
cvt.u16.u32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 16 位无符号整数结果 |
| `a` | 32 位无符号整数输入 |

**行为：**无饱和地缩窄，保留低 16 位。

**影响：**改变数值精度、掩码或地址偏移；需要区分浮点格式转换与整数截断/扩展。

实际示例（[PTX 第 74 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/5ee83ea169c92aec.ptx:74)）：

```ptx
cvt.u16.u32 	%rs1, %r28;
```

静态 16 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

</details>

<details>
<summary><code>cvt.u32.u64</code> · 已有路径，待回归</summary>

```ptx
cvt.u32.u64 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位无符号整数结果 |
| `a` | 64 位无符号整数输入 |

**行为：**无饱和地缩窄，保留低 32 位。

**影响：**改变数值精度、掩码或地址偏移；需要区分浮点格式转换与整数截断/扩展。

实际示例（[PTX 第 88 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:88)）：

```ptx
cvt.u32.u64 	%r16, %rd19;
```

静态 179 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

</details>

<details>
<summary><code>cvt.u64.u32</code> · 已有路径，待回归</summary>

```ptx
cvt.u64.u32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位无符号整数结果 |
| `a` | 32 位无符号整数输入 |

**行为：**将 u32 零扩展为 u64，高位补 0。

**影响：**改变数值精度、掩码或地址偏移；需要区分浮点格式转换与整数截断/扩展。

实际示例（[PTX 第 80 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:80)）：

```ptx
cvt.u64.u32 	%rd15, %r12;
```

静态 365 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

</details>

<details>
<summary><code>cvta.global.u64</code> · 已有路径，待回归</summary>

```ptx
cvta.global.u64 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位generic 地址 |
| `a` | 64 位global 空间地址或符号 |

**行为：**将global 地址转换为 generic 地址。只转换表示，不搬数据。

**影响：**影响随后访存所落空间；内部地址表示必须一致。

实际示例（[PTX 第 172 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:172)）：

```ptx
cvta.global.u64 	%rd20, %rd19;
```

静态 42 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta)。

</details>

<details>
<summary><code>cvta.to.global.u64</code> · 已有路径，待回归</summary>

```ptx
cvta.to.global.u64 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位global 空间地址 |
| `a` | 64 位generic 地址 |

**行为：**将generic 地址转换为 global 地址，输入须属于该空间窗口。只转换表示，不搬数据。

**影响：**影响随后访存所落空间；内部地址表示必须一致。

实际示例（[PTX 第 734 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:734)）：

```ptx
cvta.to.global.u64 	%rd7, %rd22;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta)。

</details>

<details>
<summary><code>div.full.f32</code> · 已有路径，待回归</summary>

```ptx
div.full.f32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 商 |
| `a, b` | FP32 被除数、除数 |

**行为：**执行覆盖完整输入范围的 FP32 除法；`.full` 与 `div.approx.f32` 的输入范围/误差约定不同，也不等同于显式 `.rn` 的精确舍入约定。

**影响：**现有词法已接受 `.full`，仍要验证数值范围和依赖，不能据修饰符新增一套无关指令。

实际示例（[PTX 第 209 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:209)）：

```ptx
div.full.f32 %r26, %r27, %r28;
```

静态 84 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-div)。

</details>

<details>
<summary><code>div.u32</code> · 已有路径，待回归</summary>

```ptx
div.u32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位无符号整数商 |
| `a` | 32 位无符号整数被除数 |
| `b` | 32 位无符号整数除数 |

**行为：**按指令的有符号性做整数除法。官方规定除零结果为未指定的机器相关值，不能自行承诺除零得到 0。

**影响：**影响归一化数值或整数划分/索引；除法依赖和延迟需要独立建模。

实际示例（[PTX 第 643 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:643)）：

```ptx
div.u32 	%r99, %r2, %r98;
```

静态 11 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-div)。

</details>

<details>
<summary><code>div.u64</code> · 已有路径，待回归</summary>

```ptx
div.u64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位无符号整数商 |
| `a` | 64 位无符号整数被除数 |
| `b` | 64 位无符号整数除数 |

**行为：**按指令的有符号性做整数除法。官方规定除零结果为未指定的机器相关值，不能自行承诺除零得到 0。

**影响：**影响归一化数值或整数划分/索引；除法依赖和延迟需要独立建模。

实际示例（[PTX 第 880 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:880)）：

```ptx
div.u64 	%rd788, %rd56, %rd57;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-div)。

</details>

<details>
<summary><code>ex2.approx.f32</code> · 已有路径，待回归</summary>

```ptx
ex2.approx.f32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 输出 |
| `a` | FP32 指数 |

**行为：**近似计算 2 的 a 次方。当前形式没有显式 `.ftz`；次正规数等边界按官方 ex2 规则处理。

**影响：**参与 softmax/SwiGLU，近似误差和吞吐需对照真实 GPU。

实际示例（[PTX 第 166 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:166)）：

```ptx
ex2.approx.f32 %f1, %f2;
```

静态 101 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-ex2)。

</details>

<details>
<summary><code>fma.rn.f32</code> · 已有路径，待回归</summary>

```ptx
fma.rn.f32 d, a, b, c;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 结果 |
| `a` | FP32 乘数 |
| `b` | FP32 乘数 |
| `c` | FP32 加数 |

**行为：**融合计算 a×b+c，中间乘积不先舍入，最终按最近偶数舍入。

**影响：**影响路由、激活和输出数值；拆成先乘后加或更换舍入方式可能改变结果及路由。

实际示例（[PTX 第 142 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4d78f60e03b223a2.ptx:142)）：

```ptx
fma.rn.f32 	%f15, %f10, %f11, %f15;
```

静态 9100 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-fma)。

</details>

<details>
<summary><code>ld.f32</code> · 已有路径，待回归</summary>

```ptx
ld.f32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；generic 地址所指空间，需按地址映射确定 |

**行为：**从generic 地址所指空间，需按地址映射确定读取 4 字节；按总访问宽度自然对齐。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 3468 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3468)）：

```ptx
ld.f32 	%f7549, [%rd385];
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.b16</code> · 已有路径，待回归</summary>

```ptx
ld.global.b16 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个原始位串16 位值 |

**行为：**从设备全局内存读取连续 2 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 48 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/97544f6266334931.ptx:48)）：

```ptx
@%p1 ld.global.b16 { %rs2 }, [ %rd1 + 0 ];
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.b32</code> · 已有路径，待回归</summary>

```ptx
ld.global.b32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个原始位串32 位值 |

**行为：**从设备全局内存读取连续 4 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 73 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:73)）：

```ptx
@%p4 ld.global.b32 { %r6 }, [ %rd6 + 0 ];
```

静态 122 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.b64</code> · 已有路径，待回归</summary>

```ptx
ld.global.b64 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个原始位串64 位值 |

**行为：**从设备全局内存读取连续 8 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 65 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:65)）：

```ptx
@%p4 ld.global.b64 { %rd4 }, [ %rd5 + 0 ];
```

静态 15 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.b8</code> · 已有路径，待回归</summary>

```ptx
ld.global.b8 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个原始位串8 位值 |

**行为：**从设备全局内存读取连续 1 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 103 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cc04e1548eb39807.ptx:103)）：

```ptx
@%p1 ld.global.b8 { %rs1 }, [ %rd1 + 0 ];
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.f32</code> · 已有路径，待回归</summary>

```ptx
ld.global.f32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；设备 global 内存 |

**行为：**从设备 global 内存读取 4 字节；按总访问宽度自然对齐。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 748 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:748)）：

```ptx
ld.global.f32 	%f69, [%rd25];
```

静态 5 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.nc.f32</code> · 已有路径，待回归</summary>

```ptx
ld.global.nc.f32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `d` | 输出，1 个浮点数32 位值 |

**行为：**从设备全局内存读取连续 4 字节，按地址递增顺序写入对应寄存器。`.nc` 通过非一致的只读缓存路径加载；本次输入在该内核内不被修改。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 570 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:570)）：

```ptx
ld.global.nc.f32 %f11, [%rd13];
```

静态 5 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.s32</code> · 已有路径，待回归</summary>

```ptx
ld.global.s32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位有符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；设备 global 内存 |

**行为：**从设备 global 内存读取 4 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 130 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:130)）：

```ptx
ld.global.s32 	%rd166, [%rd12];
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.u32</code> · 已有路径，待回归</summary>

```ptx
ld.global.u32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；设备 global 内存 |

**行为：**从设备 global 内存读取 4 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 124 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:124)）：

```ptx
ld.global.u32 	%r1956, [%rd165];
```

静态 24 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.u64</code> · 已有路径，待回归</summary>

```ptx
ld.global.u64 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；设备 global 内存 |

**行为：**从设备 global 内存读取 8 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 763 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:763)）：

```ptx
ld.global.u64 	%rd79, [%rd28];
```

静态 21 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.v2.b32</code> · 已有路径，待回归</summary>

```ptx
ld.global.v2.b32 {d0, d1}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `{d0, d1}` | 输出，2 个原始位串32 位值 |

**行为：**从设备全局内存读取连续 8 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 78 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:78)）：

```ptx
@%p10 ld.global.v2.b32 { %r15, %r16 }, [ %rd8 + 0 ];
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.v2.b64</code> · 已有路径，待回归</summary>

```ptx
ld.global.v2.b64 {d0, d1}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `{d0, d1}` | 输出，2 个原始位串64 位值 |

**行为：**从设备全局内存读取连续 16 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 69 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:69)）：

```ptx
@%p10 ld.global.v2.b64 { %rd5, %rd6 }, [ %rd7 + 0 ];
```

静态 15 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.global.v4.b32</code> · 已有路径，待回归</summary>

```ptx
ld.global.v4.b32 {d0, d1, d2, d3}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `{d0, d1, d2, d3}` | 输出，4 个原始位串32 位值 |

**行为：**从设备全局内存读取连续 16 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 56 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0246c7163d0f0932.ptx:56)）：

```ptx
@%p1 ld.global.v4.b32 { %r7, %r8, %r9, %r10 }, [ %rd1 + 0 ];
```

静态 64 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.f32</code> · 已有路径，待回归</summary>

```ptx
ld.param.f32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 内核/函数参数空间中的字节地址 |
| `d` | 输出，1 个浮点数32 位值 |

**行为：**从内核/函数参数空间读取连续 4 字节，按地址递增顺序写入对应寄存器。

**影响：**决定内核/函数实参的布局和取值；不能把参数空间误当成共享内存。

实际示例（[PTX 第 3486 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3486)）：

```ptx
ld.param.f32 	%f7549, [%rd155+512];
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.s32</code> · 已有路径，待回归</summary>

```ptx
ld.param.s32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位有符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前入口的参数空间，包括参数结构的字节偏移 |

**行为：**从当前入口的参数空间，包括参数结构的字节偏移读取 4 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 116 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1fe01d4c3326bbed.ptx:116)）：

```ptx
ld.param.s32 	%rd3, [moe_kernel_1fe01d4c3326bbed_param_5];
```

静态 12 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.s8</code> · 已有路径，待回归</summary>

```ptx
ld.param.s8 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 内核/函数参数空间中的字节地址 |
| `d` | 输出，1 个有符号整数8 位值 |

**行为：**从内核/函数参数空间读取连续 1 字节，按地址递增顺序写入对应寄存器。目标寄存器更宽时，符号扩展。

**影响：**决定内核/函数实参的布局和取值；不能把参数空间误当成共享内存。

实际示例（[PTX 第 477 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:477)）：

```ptx
ld.param.s8 	%rs1, [moe_kernel_cd71bc5848a4af72_param_3];
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.u32</code> · 已有路径，待回归</summary>

```ptx
ld.param.u32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前入口的参数空间，包括参数结构的字节偏移 |

**行为：**从当前入口的参数空间，包括参数结构的字节偏移读取 4 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 39 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1fcdae4b81ca74a4.ptx:39)）：

```ptx
ld.param.u32 	%r27, [moe_kernel_1fcdae4b81ca74a4_param_2];
```

静态 98 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.u64</code> · 已有路径，待回归</summary>

```ptx
ld.param.u64 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前入口的参数空间，包括参数结构的字节偏移 |

**行为：**从当前入口的参数空间，包括参数结构的字节偏移读取 8 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 30 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:30)）：

```ptx
ld.param.u64 	%rd9, [moe_kernel_01074d70bb53382e_param_1];
```

静态 179 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.u8</code> · 已有路径，待回归</summary>

```ptx
ld.param.u8 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 8 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前入口的参数空间，包括参数结构的字节偏移 |

**行为：**从当前入口的参数空间，包括参数结构的字节偏移读取 1 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 85 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:85)）：

```ptx
ld.param.u8 	%rs1, [moe_kernel_df3e0c49e4650c1f_param_0+1216];
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.v2.u32</code> · 已有路径，待回归</summary>

```ptx
ld.param.v2.u32 {d0, d1}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `{d0, d1}` | 2 个 32 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前入口的参数空间，包括参数结构的字节偏移 |

**行为：**从当前入口的参数空间，包括参数结构的字节偏移读取 8 字节；向量分量对应连续元素；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 81 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:81)）：

```ptx
ld.param.v2.u32 	{%r305, %r306}, [moe_kernel_df3e0c49e4650c1f_param_0+1176];
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.param.v2.u64</code> · 已有路径，待回归</summary>

```ptx
ld.param.v2.u64 {d0, d1}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `{d0, d1}` | 2 个 64 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前入口的参数空间，包括参数结构的字节偏移 |

**行为：**从当前入口的参数空间，包括参数结构的字节偏移读取 16 字节；向量分量对应连续元素；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 82 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:82)）：

```ptx
ld.param.v2.u64 	{%rd157, %rd158}, [moe_kernel_df3e0c49e4650c1f_param_0+1184];
```

静态 35 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.b32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.b32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `d` | 输出，1 个原始位串32 位值 |

**行为：**从当前 CTA 的共享内存读取连续 4 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 195 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1fcdae4b81ca74a4.ptx:195)）：

```ptx
@%p20 ld.shared.b32 %r15, [ %r16 + 0 ];
```

静态 15 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.b64</code> · 已有路径，待回归</summary>

```ptx
ld.shared.b64 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `d` | 输出，1 个原始位串64 位值 |

**行为：**从当前 CTA 的共享内存读取连续 8 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 144 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:144)）：

```ptx
@%p6 ld.shared.b64 %rd4, [ %r3 + 0 ];
```

静态 15 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.f32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.f32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前线程块 shared 内存，默认 shared::cta |

**行为：**从当前线程块 shared 内存，默认 shared::cta读取 4 字节；按总访问宽度自然对齐。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 1147 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4ebef89fddede208.ptx:1147)）：

```ptx
ld.shared.f32 	%f97, [%r507+17936];
```

静态 4001 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.s32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.s32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `d` | 输出，1 个有符号整数32 位值 |

**行为：**从当前 CTA 的共享内存读取连续 4 字节，按地址递增顺序写入对应寄存器。目标寄存器更宽时，符号扩展。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 127 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/c479f139e809aa8b.ptx:127)）：

```ptx
ld.shared.s32 	%rd12, [global_smem];
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.u32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.u32 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位无符号整数目标寄存器 |
| `addr` | 字节地址，可含字节偏移；当前线程块 shared 内存，默认 shared::cta |

**行为：**从当前线程块 shared 内存，默认 shared::cta读取 4 字节；按总访问宽度自然对齐。窄整数装入更宽合法寄存器时，signed 符号扩展、unsigned 零扩展。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新寄存器并生成读请求，消费者等待数据。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 202 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:202)）：

```ptx
ld.shared.u32 	%r42, [%r55];
```

静态 43 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.u64</code> · 已有路径，待回归</summary>

```ptx
ld.shared.u64 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `d` | 输出，1 个无符号整数64 位值 |

**行为：**从当前 CTA 的共享内存读取连续 8 字节，按地址递增顺序写入对应寄存器。目标寄存器更宽时，零扩展。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 165 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:165)）：

```ptx
ld.shared.u64 	%rd6, [global_smem];
```

静态 87 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.u8</code> · 已有路径，待回归</summary>

```ptx
ld.shared.u8 d, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `d` | 输出，1 个无符号整数8 位值 |

**行为：**从当前 CTA 的共享内存读取连续 1 字节，按地址递增顺序写入对应寄存器。目标寄存器更宽时，零扩展。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 833 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:833)）：

```ptx
ld.shared.u8 	%rs30, [%r39];
```

静态 56 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.v2.f32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.v2.f32 {d0, d1}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{d0, d1}` | 输出，2 个浮点数32 位值 |

**行为：**从当前 CTA 的共享内存读取连续 8 字节，按地址递增顺序写入对应寄存器。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 680 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/b4d06c15353157e5.ptx:680)）：

```ptx
ld.shared.v2.f32 	{%f67, %f68}, [%r18+1016];
```

静态 12 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.v2.u32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.v2.u32 {d0, d1}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{d0, d1}` | 输出，2 个无符号整数32 位值 |

**行为：**从当前 CTA 的共享内存读取连续 8 字节，按地址递增顺序写入对应寄存器。目标寄存器更宽时，零扩展。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 480 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cc04e1548eb39807.ptx:480)）：

```ptx
ld.shared.v2.u32 	{%r271, %r272}, [global_smem+112];
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.v4.b16</code> · 已有路径，待回归</summary>

```ptx
ld.shared.v4.b16 {d0, d1, d2, d3}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{d0, d1, d2, d3}` | 输出，4 个原始位串16 位值 |

**行为：**从当前 CTA 的共享内存读取连续 8 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 68 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0246c7163d0f0932.ptx:68)）：

```ptx
ld.shared.v4.b16 	{%rs9, %rs10, %rs11, %rs12}, [%r38];
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.v4.b32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.v4.b32 {d0, d1, d2, d3}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{d0, d1, d2, d3}` | 输出，4 个原始位串32 位值 |

**行为：**从当前 CTA 的共享内存读取连续 16 字节，按地址递增顺序写入对应寄存器。这是位模式搬运；合法宽目标按 PTX 尺寸规则扩展，不做数值格式转换。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 322 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/b4d06c15353157e5.ptx:322)）：

```ptx
ld.shared.v4.b32 	{%r692, %r693, %r694, %r695}, [%r17+256];
```

静态 63 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.v4.f32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.v4.f32 {d0, d1, d2, d3}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{d0, d1, d2, d3}` | 输出，4 个浮点数32 位值 |

**行为：**从当前 CTA 的共享内存读取连续 16 字节，按地址递增顺序写入对应寄存器。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 682 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/b4d06c15353157e5.ptx:682)）：

```ptx
ld.shared.v4.f32 	{%f71, %f72, %f73, %f74}, [%r18+928];
```

静态 22 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>ld.shared.v4.u32</code> · 已有路径，待回归</summary>

```ptx
ld.shared.v4.u32 {d0, d1, d2, d3}, [addr];
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{d0, d1, d2, d3}` | 输出，4 个无符号整数32 位值 |

**行为：**从当前 CTA 的共享内存读取连续 16 字节，按地址递增顺序写入对应寄存器。目标寄存器更宽时，零扩展。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 7161 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4ebef89fddede208.ptx:7161)）：

```ptx
ld.shared.v4.u32 	{%r558, %r559, %r560, %r561}, [%r640];
```

静态 38 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)。

</details>

<details>
<summary><code>mad.lo.s32</code> · 已有路径，待回归</summary>

```ptx
mad.lo.s32 d, a, b, c;
```

| 参数 | 含义 |
| --- | --- |
| `d` | s32 结果 |
| `a` | s32 乘数 |
| `b` | s32 乘数 |
| `c` | s32 加数 |

**行为：**取 a×b 的低 32 位，再加 c，保留 32 位结果。

**影响：**影响索引和地址；这是整数乘加，不能按浮点 FMA 实现。

实际示例（[PTX 第 204 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/647c5fa6ac891eff.ptx:204)）：

```ptx
mad.lo.s32 	%r6, %r11, -24, %r58;
```

静态 37 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad)。

</details>

<details>
<summary><code>max.s32</code> · 已有路径，待回归</summary>

```ptx
max.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `参数` | a/b 是 s32 输入，d 是 s32 输出。 |

**行为：**按有符号 32 位整数比较，输出较大值。

**影响：**影响数值、索引或执行掩码；使用现有对应功能路径后仍需回归结果与依赖。

实际示例（[PTX 第 658 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:658)）：

```ptx
max.s32 	%r108, %r107, -3;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-max)。

</details>

<details>
<summary><code>min.s32</code> · 已有路径，待回归</summary>

```ptx
min.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | s32 结果 |
| `a` | s32 候选值 |
| `b` | s32 候选值 |

**行为：**按有符号比较选择较小值。

**影响：**影响边界裁剪和循环范围。

实际示例（[PTX 第 2350 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:2350)）：

```ptx
min.s32 	%r764, %r116, 1;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-min)。

</details>

<details>
<summary><code>mov.b32</code> · 已有路径，待回归</summary>

```ptx
mov.b32 d, a;
mov.b32 {lo, hi}, a;
mov.b32 d, {lo, hi};
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位输出，或拆成两个 16 位输出寄存器 |
| `a` | 32 位输入、同宽立即数/地址，或两个 16 位源寄存器的元组 |

**行为：**复制位模式，也支持将 32 位值拆成两个 16 位寄存器，或将二者合并；元组第一项是低半部，第二项是高半部。源为可寻址符号时取地址，不读取内存；不做浮点格式转换。

**影响：**建立寄存器值和依赖。

实际示例（[PTX 第 78 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:78)）：

```ptx
mov.b32 	%r13, 0;
```

静态 1780 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mov.b64</code> · 已有路径，待回归</summary>

```ptx
mov.b64 d, a;
mov.b64 {lo, hi}, a;
mov.b64 d, {lo, hi};
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位输出，或拆成两个 32 位输出寄存器 |
| `a` | 64 位输入、同宽立即数/地址，或两个 32 位源寄存器的元组 |

**行为：**复制位模式，也支持将 64 位值拆成两个 32 位寄存器，或将二者合并；元组第一项是低半部，第二项是高半部。源为可寻址符号时取地址，不读取内存；不做浮点格式转换。

**影响：**建立寄存器值和依赖。

实际示例（[PTX 第 87 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:87)）：

```ptx
mov.b64 {tmp, %r15}, %rd19;
```

静态 225 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mov.f32</code> · 已有路径，待回归</summary>

```ptx
mov.f32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数目标寄存器 |
| `a` | FP32 浮点数源寄存器或对应立即数；当前不取地址 |

**行为：**复制该类型的值或位模式，不作数值格式转换。

**影响：**建立寄存器值和依赖。

实际示例（[PTX 第 152 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:152)）：

```ptx
mov.f32 	%f33, 0f00000000;
```

静态 380 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mov.pred</code> · 已有路径，待回归</summary>

```ptx
mov.pred d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 布尔 predicate目标寄存器 |
| `a` | 布尔 predicate源寄存器或对应立即数；当前不取地址 |

**行为：**复制该类型的值或位模式，不作数值格式转换。

**影响：**建立寄存器值和依赖。

实际示例（[PTX 第 49 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0246c7163d0f0932.ptx:49)）：

```ptx
mov.pred 	%p1, -1;
```

静态 86 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mov.s32</code> · 已有路径，待回归</summary>

```ptx
mov.s32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位有符号整数目标寄存器 |
| `a` | 32 位有符号整数源寄存器或对应立即数；当前不取地址 |

**行为：**复制该类型的值或位模式，不作数值格式转换。

**影响：**建立寄存器值和依赖。

实际示例（[PTX 第 531 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:531)）：

```ptx
mov.s32 %r188, %laneid;
```

静态 61 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mov.u16</code> · 已有路径，待回归</summary>

```ptx
mov.u16 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 16 位无符号整数目标寄存器 |
| `a` | 16 位无符号整数源寄存器或对应立即数；当前不取地址 |

**行为：**复制该类型的值或位模式，不作数值格式转换。

**影响：**建立寄存器值和依赖。

实际示例（[PTX 第 56 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:56)）：

```ptx
@!%p1 mov.u16 %rs1, 0x0;
```

静态 48 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mov.u32</code> · 已有路径，待回归</summary>

```ptx
mov.u32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位无符号整数目标寄存器 |
| `a` | 同宽源值/立即数、特殊寄存器或可寻址符号的地址；以实际形式为准 |

**行为：**复制值或位模式，不作 cvt 式数值转换；源为可寻址符号时取其所属空间的地址，不读取该地址内容。当前还读取 `%cluster_ctaid.x/y/z`、`%cluster_ctarank` 等特殊寄存器。

**影响：**建立寄存器值和依赖。 mov 本身已有入口，但 cluster 特殊寄存器值的生成仍须接通。

实际示例（[PTX 第 134 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:134)）：

```ptx
mov.u32 	%r33, global_smem;
```

静态 809 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mov.u64</code> · 已有路径，待回归</summary>

```ptx
mov.u64 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位无符号整数目标寄存器 |
| `a` | 同宽源值/立即数、特殊寄存器或可寻址符号的地址；以实际形式为准 |

**行为：**复制值或位模式，不作 cvt 式数值转换；源为可寻址符号时取其所属空间的地址，不读取该地址内容。

**影响：**建立寄存器值和依赖。

实际示例（[PTX 第 173 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:173)）：

```ptx
mov.u64 	%rd21, assertFile_0;
```

静态 133 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov)。

</details>

<details>
<summary><code>mul.f32</code> · 已有路径，待回归</summary>

```ptx
mul.f32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数结果 |
| `a` | FP32 浮点数左操作数 |
| `b` | FP32 浮点数右操作数 |

**行为：**计算 `d = a × b`。默认最近偶数舍入；没有 `.ftz`，不能主动清零次正规数。

**影响：**影响数值或索引；消费者须等待 d 就绪，执行延迟进入相应运算管线。

实际示例（[PTX 第 168 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:168)）：

```ptx
mul.f32 	%f4, %f35, 0f3FB8AA3B;
```

静态 325 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-mul)。

</details>

<details>
<summary><code>mul.hi.u64</code> · 已有路径，待回归</summary>

```ptx
mul.hi.u64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位结果 |
| `a` | 64 位无符号整数乘数 |
| `b` | 64 位无符号整数乘数 |

**行为：**取 128 位乘积的高 64 位。按末尾类型解释有符号性。

**影响：**影响地址偏移和线性索引；宽度或高/低位选择错误会产生错误访存地址。

实际示例（[PTX 第 138 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:138)）：

```ptx
mul.hi.u64 	%rd777, %rd169, %rd7;
```

静态 6 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)。

</details>

<details>
<summary><code>mul.lo.s32</code> · 已有路径，待回归</summary>

```ptx
mul.lo.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位结果 |
| `a` | 32 位有符号整数乘数 |
| `b` | 32 位有符号整数乘数 |

**行为：**取 64 位乘积的低 32 位。按末尾类型解释有符号性。

**影响：**影响地址偏移和线性索引；宽度或高/低位选择错误会产生错误访存地址。

实际示例（[PTX 第 81 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4d78f60e03b223a2.ptx:81)）：

```ptx
mul.lo.s32 	%r3, %r5, %r14;
```

静态 42 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)。

</details>

<details>
<summary><code>mul.lo.s64</code> · 已有路径，待回归</summary>

```ptx
mul.lo.s64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位结果 |
| `a` | 64 位有符号整数乘数 |
| `b` | 64 位有符号整数乘数 |

**行为：**取 128 位乘积的低 64 位。按末尾类型解释有符号性。

**影响：**影响地址偏移和线性索引；宽度或高/低位选择错误会产生错误访存地址。

实际示例（[PTX 第 111 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:111)）：

```ptx
mul.lo.s64 	%rd162, %rd13, %rd161;
```

静态 12 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)。

</details>

<details>
<summary><code>mul.wide.s32</code> · 已有路径，待回归</summary>

```ptx
mul.wide.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位结果 |
| `a` | 32 位有符号整数乘数 |
| `b` | 32 位有符号整数乘数 |

**行为：**保存完整 64 位乘积，不能先截窄再扩宽。按末尾类型解释有符号性。

**影响：**影响地址偏移和线性索引；宽度或高/低位选择错误会产生错误访存地址。

实际示例（[PTX 第 43 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0d3fa1d540e73560.ptx:43)）：

```ptx
mul.wide.s32 	%rd7, %r16, 4;
```

静态 314 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)。

</details>

<details>
<summary><code>mul.wide.u16</code> · 已有路径，待回归</summary>

```ptx
mul.wide.u16 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `参数` | a/b 是 u16，d 是 32 位输出；不是低 16 位乘法。 |

**行为：**将两个 u16 输入相乘，保留完整 32 位乘积。

**影响：**影响数值、索引或执行掩码；使用现有对应功能路径后仍需回归结果与依赖。

实际示例（[PTX 第 989 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:989)）：

```ptx
mul.wide.u16 	%r617, %rs12, 128;
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)。

</details>

<details>
<summary><code>mul.wide.u32</code> · 已有路径，待回归</summary>

```ptx
mul.wide.u32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位结果 |
| `a` | 32 位无符号整数乘数 |
| `b` | 32 位无符号整数乘数 |

**行为：**保存完整 64 位乘积，不能先截窄再扩宽。按末尾类型解释有符号性。

**影响：**影响地址偏移和线性索引；宽度或高/低位选择错误会产生错误访存地址。

实际示例（[PTX 第 975 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/4ebef89fddede208.ptx:975)）：

```ptx
mul.wide.u32 	%rd298, %r441, 16;
```

静态 30 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)。

</details>

<details>
<summary><code>neg.s32</code> · 已有路径，待回归</summary>

```ptx
neg.s32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `参数` | a/d 是 s32；溢出按 32 位截断。 |

**行为：**计算 32 位二补码负值 d=−a。

**影响：**影响数值、索引或执行掩码；使用现有对应功能路径后仍需回归结果与依赖。

实际示例（[PTX 第 61 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/c479f139e809aa8b.ptx:61)）：

```ptx
neg.s32 	%r40, %r39;
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg)。

</details>

<details>
<summary><code>not.b32</code> · 已有路径，待回归</summary>

```ptx
not.b32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位位串结果 |
| `a` | 32 位位串输入 |

**行为：**反转每一位；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 657 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:657)）：

```ptx
not.b32 	%r107, %r323;
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-not)。

</details>

<details>
<summary><code>not.pred</code> · 已有路径，待回归</summary>

```ptx
not.pred d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 布尔 predicate结果 |
| `a` | 布尔 predicate输入 |

**行为：**布尔取反；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 159 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:159)）：

```ptx
not.pred 	%p14, %p13;
```

静态 13 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-not)。

</details>

<details>
<summary><code>or.b16</code> · 已有路径，待回归</summary>

```ptx
or.b16 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 16 位位串结果 |
| `a` | 16 位位串输入 |
| `b` | 16 位位串输入 |

**行为：**逐位 OR；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 63 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/b772016a040a6f6a.ptx:63)）：

```ptx
or.b16  	%rs1, %rs2, %rs4;
```

静态 10 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-or)。

</details>

<details>
<summary><code>or.b32</code> · 已有路径，待回归</summary>

```ptx
or.b32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位位串结果 |
| `a` | 32 位位串输入 |
| `b` | 32 位位串输入 |

**行为：**逐位 OR；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 45 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0246c7163d0f0932.ptx:45)）：

```ptx
or.b32  	%r34, %r27, %r32;
```

静态 359 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-or)。

</details>

<details>
<summary><code>or.b64</code> · 已有路径，待回归</summary>

```ptx
or.b64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位位串结果 |
| `a` | 64 位位串输入 |
| `b` | 64 位位串输入 |

**行为：**逐位 OR；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 83 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:83)）：

```ptx
or.b64  	%rd18, %rd15, %rd17;
```

静态 368 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-or)。

</details>

<details>
<summary><code>or.pred</code> · 已有路径，待回归</summary>

```ptx
or.pred d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 布尔 predicate结果 |
| `a` | 布尔 predicate输入 |
| `b` | 布尔 predicate输入 |

**行为：**布尔 OR；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 101 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cabbd061a6da98db.ptx:101)）：

```ptx
or.pred  	%p9, %p7, %p8;
```

静态 12 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-or)。

</details>

<details>
<summary><code>popc.b32</code> · 已有路径，待回归</summary>

```ptx
popc.b32 d, a;
```

| 参数 | 含义 |
| --- | --- |
| `d` | u32 计数，0..32 |
| `a` | 32 位输入掩码 |

**行为：**统计 a 中值为 1 的 bit 数。

**影响：**参与 lane 前缀计数与 token 重排。

实际示例（[PTX 第 569 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cc04e1548eb39807.ptx:569)）：

```ptx
popc.b32 	%r293, %r292;
```

静态 9 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-popc)。

</details>

<details>
<summary><code>prmt.b32</code> · 已有路径，待回归</summary>

```ptx
prmt.b32 d, a, b, c;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 4 字节输出 |
| `a, b` | 共 8 字节源，a 对应编号 0..3，b 对应 4..7 |
| `c` | 低 16 位的四个 nibble，分别控制一个输出字节 |

**行为：**每个控制 nibble 的低 3 位选取一个源字节；bit3=0 时复制字节，bit3=1 时复制所选字节的符号位到整字节。

**影响：**用于索引/数据字节重排，不能简化成普通移位或忽略控制高 bit。

实际示例（[PTX 第 76 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0d3fa1d540e73560.ptx:76)）：

```ptx
prmt.b32 	%r19, %r18, %r17, 0x3340U;
```

静态 21 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt)。

</details>

<details>
<summary><code>rem.s32</code> · 已有路径，待回归</summary>

```ptx
rem.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | s32 余数 |
| `a` | s32 被除数 |
| `b` | s32 除数 |

**行为：**求整数除法余数。官方说明负数行为依赖机器采用的除法取整方式，不能直接等同于 Python 的 `%`。

**影响：**影响 tile 内索引；负数边界需按目标行为核查。

实际示例（[PTX 第 164 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:164)）：

```ptx
rem.s32 	%r328, %r327, %r19;
```

静态 6 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem)。

</details>

<details>
<summary><code>rem.u32</code> · 已有路径，待回归</summary>

```ptx
rem.u32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `参数` | a 是被除数，b 是除数，d 是 u32 输出；当前有效路径 b≠0。 |

**行为：**输出无符号整数 a 除以 b 的余数。

**影响：**影响数值、索引或执行掩码；使用现有对应功能路径后仍需回归结果与依赖。

实际示例（[PTX 第 784 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:784)）：

```ptx
rem.u32 	%r231, %r230, %r41;
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem)。

</details>

<details>
<summary><code>ret</code> · 已有路径，待回归</summary>

```ptx
ret;
```

**参数：**无显式操作数；协作对象或异步组由线程/warpgroup 上下文决定。

**行为：**当前入口 kernel 中终止当前线程；普通设备函数中则返回调用者。

**影响：**影响线程/warp/块完成和资源释放；未完成请求和同步依赖仍需处理。

实际示例（[PTX 第 1545 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:1545)）：

```ptx
ret;
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#control-flow-instructions-ret)。

</details>

<details>
<summary><code>selp.b32</code> · 已有路径，待回归</summary>

```ptx
selp.b32 d, a, b, p;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位位串结果 |
| `a` | p 为真时选取的值 |
| `b` | p 为假时选取的值 |
| `p` | 控制选择的 predicate |

**行为：**选择 `p ? a : b`，不进行数值格式转换。

**影响：**影响专家候选、索引、mask 或指针选择；不能交换 a/b 顺序。

实际示例（[PTX 第 227 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1fcdae4b81ca74a4.ptx:227)）：

```ptx
selp.b32 	%r20, -1, %r72, %p31;
```

静态 178 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp)。

</details>

<details>
<summary><code>selp.b64</code> · 已有路径，待回归</summary>

```ptx
selp.b64 d, a, b, p;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位位串结果 |
| `a` | p 为真时选取的值 |
| `b` | p 为假时选取的值 |
| `p` | 控制选择的 predicate |

**行为：**选择 `p ? a : b`，不进行数值格式转换。

**影响：**影响专家候选、索引、mask 或指针选择；不能交换 a/b 顺序。

实际示例（[PTX 第 78 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/647c5fa6ac891eff.ptx:78)）：

```ptx
selp.b64 	%rd20, %rd19, 0, %p1;
```

静态 22 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp)。

</details>

<details>
<summary><code>selp.f32</code> · 已有路径，待回归</summary>

```ptx
selp.f32 d, a, b, p;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 与后缀类型一致的输出 |
| `a, b` | 候选输入 |
| `p` | 选择 predicate |

**行为：**p 为真时 d=a，否则 d=b。

**影响：**选择值或地址，不会同步线程；被选中的值参与后续路由或重排。

实际示例（[PTX 第 117 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:117)）：

```ptx
selp.f32 	%f11, %f8, %f10, %p9;
```

静态 31 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp)。

</details>

<details>
<summary><code>selp.u16</code> · 已有路径，待回归</summary>

```ptx
selp.u16 d, a, b, p;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 16 位无符号整数结果 |
| `a` | p 为真时选取的值 |
| `b` | p 为假时选取的值 |
| `p` | 控制选择的 predicate |

**行为：**选择 `p ? a : b`，不进行数值格式转换。

**影响：**影响专家候选、索引、mask 或指针选择；不能交换 a/b 顺序。

实际示例（[PTX 第 776 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:776)）：

```ptx
selp.u16 	%rs27, 1, 0, %p1;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp)。

</details>

<details>
<summary><code>selp.u32</code> · 已有路径，待回归</summary>

```ptx
selp.u32 d, a, b, p;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位无符号整数结果 |
| `a` | p 为真时选取的值 |
| `b` | p 为假时选取的值 |
| `p` | 控制选择的 predicate |

**行为：**选择 `p ? a : b`，不进行数值格式转换。

**影响：**影响专家候选、索引、mask 或指针选择；不能交换 a/b 顺序。

实际示例（[PTX 第 75 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0d3fa1d540e73560.ptx:75)）：

```ptx
selp.u32 	%r18, 1, 0, %p6;
```

静态 69 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp)。

</details>

<details>
<summary><code>selp.u64</code> · 已有路径，待回归</summary>

```ptx
selp.u64 d, a, b, p;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 与后缀类型一致的输出 |
| `a, b` | 候选输入 |
| `p` | 选择 predicate |

**行为：**p 为真时 d=a，否则 d=b。

**影响：**选择值或地址，不会同步线程；被选中的值参与后续路由或重排。

实际示例（[PTX 第 69 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:69)）：

```ptx
selp.u64 	%rd12, 1, 0, %p11;
```

静态 25 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp)。

</details>

<details>
<summary><code>setp.eq.b32</code> · 已有路径，待回归</summary>

```ptx
setp.eq.b32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位位串左操作数 |
| `b` | 32 位位串右操作数 |

**行为：**比较 a 是否等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 158 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:158)）：

```ptx
setp.eq.b32 	%p13, %r39, 1;
```

静态 19 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.eq.f32</code> · 已有路径，待回归</summary>

```ptx
setp.eq.f32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | FP32 浮点数左操作数 |
| `b` | FP32 浮点数右操作数 |

**行为：**比较 a 是否等于 b，结果写 p。有序浮点比较：任一输入为 NaN 时结果为假。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 3552 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:3552)）：

```ptx
setp.eq.f32 	%p121, %f7549, 0f00000000;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.eq.s16</code> · 已有路径，待回归</summary>

```ptx
setp.eq.s16 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 16 位有符号整数左操作数 |
| `b` | 16 位有符号整数右操作数 |

**行为：**比较 a 是否等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 830 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:830)）：

```ptx
setp.eq.s16 	%p22, %rs92, 0;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.eq.s32</code> · 已有路径，待回归</summary>

```ptx
setp.eq.s32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位有符号整数左操作数 |
| `b` | 32 位有符号整数右操作数 |

**行为：**比较 a 是否等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 169 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1fcdae4b81ca74a4.ptx:169)）：

```ptx
setp.eq.s32 	%p31, %r4, 0;
```

静态 184 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.eq.s64</code> · 已有路径，待回归</summary>

```ptx
setp.eq.s64 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 64 位有符号整数左操作数 |
| `b` | 64 位有符号整数右操作数 |

**行为：**比较 a 是否等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 106 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/23ba58950db8f4f0.ptx:106)）：

```ptx
setp.eq.s64 	%p10, %rd1, %rd12;
```

静态 17 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.ge.s32</code> · 已有路径，待回归</summary>

```ptx
setp.ge.s32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位有符号整数左操作数 |
| `b` | 32 位有符号整数右操作数 |

**行为：**比较 a 是否大于等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 117 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:117)）：

```ptx
setp.ge.s32 	%p9, %r12, %r7;
```

静态 9 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.ge.u32</code> · 已有路径，待回归</summary>

```ptx
setp.ge.u32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | u32 类型的两个输入 |

**行为：**按无符号整数比较 a≥b。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 638 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:638)）：

```ptx
setp.ge.u32 	%p5, %r96, %r84;
```

静态 15 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.gt.f32</code> · 已有路径，待回归</summary>

```ptx
setp.gt.f32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | FP32 浮点数左操作数 |
| `b` | FP32 浮点数右操作数 |

**行为：**比较 a 是否大于 b，结果写 p。有序浮点比较：任一输入为 NaN 时结果为假。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 119 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1d77cfc1699759be.ptx:119)）：

```ptx
setp.gt.f32 	%p16, %f15, %f13;
```

静态 19 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.gt.s32</code> · 已有路径，待回归</summary>

```ptx
setp.gt.s32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位有符号整数左操作数 |
| `b` | 32 位有符号整数右操作数 |

**行为：**比较 a 是否大于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 673 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:673)）：

```ptx
setp.gt.s32 	%p42, %r469, %r26;
```

静态 5 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.gt.u32</code> · 已有路径，待回归</summary>

```ptx
setp.gt.u32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位无符号整数左操作数 |
| `b` | 32 位无符号整数右操作数 |

**行为：**比较 a 是否大于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 64 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/23ba58950db8f4f0.ptx:64)）：

```ptx
setp.gt.u32 	%p7, %r4, 1;
```

静态 23 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.gt.u64</code> · 已有路径，待回归</summary>

```ptx
setp.gt.u64 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 64 位无符号整数左操作数 |
| `b` | 64 位无符号整数右操作数 |

**行为：**比较 a 是否大于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 813 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:813)）：

```ptx
setp.gt.u64 	%p62, %rd46, %rd16;
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.le.f32</code> · 已有路径，待回归</summary>

```ptx
setp.le.f32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | f32 类型的两个输入 |

**行为：**按操作数类型比较 a≤b。浮点有任一 NaN 时为假。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 815 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:815)）：

```ptx
setp.le.f32 	%p19, %f37, 0f7F800000;
```

静态 30 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.le.s32</code> · 已有路径，待回归</summary>

```ptx
setp.le.s32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | s32 类型的两个输入 |

**行为：**按操作数类型比较 a≤b。浮点有任一 NaN 时为假。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 105 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/756851cbac2894af.ptx:105)）：

```ptx
setp.le.s32 	%p8, %r26, %r29;
```

静态 16 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.le.u32</code> · 已有路径，待回归</summary>

```ptx
setp.le.u32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | u32 类型的两个输入 |

**行为：**按操作数类型比较 a≤b。浮点有任一 NaN 时为假。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 1180 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:1180)）：

```ptx
setp.le.u32 	%p102, %r480, %r120;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.le.u64</code> · 已有路径，待回归</summary>

```ptx
setp.le.u64 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 64 位无符号整数左操作数 |
| `b` | 64 位无符号整数右操作数 |

**行为：**比较 a 是否小于等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 660 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:660)）：

```ptx
setp.le.u64 	%p38, %rd10, %rd16;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.leu.f32</code> · 已有路径，待回归</summary>

```ptx
setp.leu.f32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | f32 类型的两个输入 |

**行为：**执行无序浮点比较：a≤b 或任一输入为 NaN 时为真。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 1371 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:1371)）：

```ptx
setp.leu.f32 	%p119, %f27, %f28;
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.lt.s32</code> · 已有路径，待回归</summary>

```ptx
setp.lt.s32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位有符号整数左操作数 |
| `b` | 32 位有符号整数右操作数 |

**行为：**比较 a 是否小于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 140 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:140)）：

```ptx
setp.lt.s32 	%p6, %r5, 2;
```

静态 159 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.lt.u32</code> · 已有路径，待回归</summary>

```ptx
setp.lt.u32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位无符号整数左操作数 |
| `b` | 32 位无符号整数右操作数 |

**行为：**比较 a 是否小于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 663 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:663)）：

```ptx
setp.lt.u32 	%p7, %r110, 3;
```

静态 30 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.lt.u64</code> · 已有路径，待回归</summary>

```ptx
setp.lt.u64 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 64 位无符号整数左操作数 |
| `b` | 64 位无符号整数右操作数 |

**行为：**比较 a 是否小于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 129 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:129)）：

```ptx
setp.lt.u64 	%p11, %rd22, 64;
```

静态 12 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.nan.f32</code> · 已有路径，待回归</summary>

```ptx
setp.nan.f32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | f32 类型的两个输入 |

**行为：**任一输入为 NaN 时为真。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 122 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1d77cfc1699759be.ptx:122)）：

```ptx
setp.nan.f32 	%p18, %f15, %f15;
```

静态 8 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.ne.b32</code> · 已有路径，待回归</summary>

```ptx
setp.ne.b32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位位串左操作数 |
| `b` | 32 位位串右操作数 |

**行为：**比较 a 是否不等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 1658 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:1658)）：

```ptx
setp.ne.b32 p, %r711, 0;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.ne.s16</code> · 已有路径，待回归</summary>

```ptx
setp.ne.s16 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 16 位有符号整数左操作数 |
| `b` | 16 位有符号整数右操作数 |

**行为：**比较 a 是否不等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 65 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:65)）：

```ptx
setp.ne.s16 	%p9, %rs3, 0;
```

静态 50 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.ne.s32</code> · 已有路径，待回归</summary>

```ptx
setp.ne.s32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a` | 32 位有符号整数左操作数 |
| `b` | 32 位有符号整数右操作数 |

**行为：**比较 a 是否不等于 b，结果写 p。按类型保留有符号性或位串语义。

**影响：**决定 Top-k、分支、循环退出或屏障轮询；predicate 也有就绪依赖。

实际示例（[PTX 第 66 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0d3fa1d540e73560.ptx:66)）：

```ptx
setp.ne.s32 	%p5, %r3, 0;
```

静态 72 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.ne.s64</code> · 已有路径，待回归</summary>

```ptx
setp.ne.s64 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | s64 类型的两个输入 |

**行为：**按操作数类型比较 a≠b。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 159 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1fe01d4c3326bbed.ptx:159)）：

```ptx
setp.ne.s64 	%p12, %rd33, 0;
```

静态 21 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.neu.f32</code> · 已有路径，待回归</summary>

```ptx
setp.neu.f32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | f32 类型的两个输入 |

**行为：**a≠b 或任一输入为 NaN 时为真。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 822 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:822)）：

```ptx
setp.neu.f32 	%p49, %f13, 0f00000000;
```

静态 2 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>setp.num.f32</code> · 已有路径，待回归</summary>

```ptx
setp.num.f32 p, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `p` | 输出 predicate |
| `a, b` | f32 类型的两个输入 |

**行为：**两个输入都不是 NaN 时为真。 将布尔结果写入 p。

**影响：**影响分支、排序和 top-k 的掩码；尤其要保留有序/无序浮点比较的差别。

实际示例（[PTX 第 575 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:575)）：

```ptx
setp.num.f32 	%p9, %f11, %f11;
```

静态 6 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)。

</details>

<details>
<summary><code>shl.b16</code> · 已有路径，待回归</summary>

```ptx
shl.b16 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 16 位结果 |
| `a` | 16 位位串待移位值 |
| `b` | 无符号 32 位移位量，与 a 的位宽无关 |

**行为：**左移，低位补 0。移位量大于位宽时按位宽处理，不能将移位量按 32/64 取模。

**影响：**影响索引、组播 mask 或描述符字段。

实际示例（[PTX 第 62 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/b772016a040a6f6a.ptx:62)）：

```ptx
shl.b16 	%rs4, %rs3, 8;
```

静态 10 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-shl)。

</details>

<details>
<summary><code>shl.b32</code> · 已有路径，待回归</summary>

```ptx
shl.b32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位结果 |
| `a` | 32 位位串待移位值 |
| `b` | 无符号 32 位移位量，与 a 的位宽无关 |

**行为：**左移，低位补 0。移位量大于位宽时按位宽处理，不能将移位量按 32/64 取模。

**影响：**影响索引、组播 mask 或描述符字段。

实际示例（[PTX 第 141 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:141)）：

```ptx
shl.b32 	%r34, %r5, 3;
```

静态 500 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-shl)。

</details>

<details>
<summary><code>shl.b64</code> · 已有路径，待回归</summary>

```ptx
shl.b64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位结果 |
| `a` | 64 位位串待移位值 |
| `b` | 无符号 32 位移位量，与 a 的位宽无关 |

**行为：**左移，低位补 0。移位量大于位宽时按位宽处理，不能将移位量按 32/64 取模。

**影响：**影响索引、组播 mask 或描述符字段。

实际示例（[PTX 第 82 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:82)）：

```ptx
shl.b64 	%rd17, %rd16, 32;
```

静态 285 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-shl)。

</details>

<details>
<summary><code>shr.s32</code> · 已有路径，待回归</summary>

```ptx
shr.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位结果 |
| `a` | 32 位有符号整数待移位值 |
| `b` | 无符号 32 位移位量，与 a 的位宽无关 |

**行为：**算术右移，高位补符号位。移位量大于位宽时按位宽处理，不能将移位量按 32/64 取模。

**影响：**影响索引、组播 mask 或描述符字段。

实际示例（[PTX 第 46 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:46)）：

```ptx
shr.s32 	%r70, %r68, 11;
```

静态 48 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-shr)。

</details>

<details>
<summary><code>shr.u32</code> · 已有路径，待回归</summary>

```ptx
shr.u32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位结果 |
| `a` | 32 位无符号整数待移位值 |
| `b` | 无符号 32 位移位量，与 a 的位宽无关 |

**行为：**逻辑右移，高位补 0。移位量大于位宽时按位宽处理，不能将移位量按 32/64 取模。

**影响：**影响索引、组播 mask 或描述符字段。

实际示例（[PTX 第 132 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:132)）：

```ptx
shr.u32 	%r31, %r5, 2;
```

静态 135 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-shr)。

</details>

<details>
<summary><code>shr.u64</code> · 已有路径，待回归</summary>

```ptx
shr.u64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位结果 |
| `a` | 64 位无符号整数待移位值 |
| `b` | 无符号 32 位移位量，与 a 的位宽无关 |

**行为：**逻辑右移，高位补 0。移位量大于位宽时按位宽处理，不能将移位量按 32/64 取模。

**影响：**影响索引、组播 mask 或描述符字段。

实际示例（[PTX 第 155 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:155)）：

```ptx
shr.u64 	%rd174, %rd778, %r307;
```

静态 23 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-shr)。

</details>

<details>
<summary><code>st.global.b16</code> · 已有路径，待回归</summary>

```ptx
st.global.b16 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `a` | 输入，1 个原始位串16 位值 |

**行为：**按元组顺序将源寄存器的低 16 位写入设备全局内存，总计连续 2 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 65 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/b772016a040a6f6a.ptx:65)）：

```ptx
@%p1 st.global.b16 [ %rd2 + 0 ], { %rs1 };
```

静态 3 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.global.b32</code> · 已有路径，待回归</summary>

```ptx
st.global.b32 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `a` | 输入，1 个原始位串32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入设备全局内存，总计连续 4 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 205 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:205)）：

```ptx
@%p10 st.global.b32 [ %rd31 + 0 ], { %r42 };
```

静态 81 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.global.b64</code> · 已有路径，待回归</summary>

```ptx
st.global.b64 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `a` | 输入，1 个原始位串64 位值 |

**行为：**按元组顺序将源寄存器的低 64 位写入设备全局内存，总计连续 8 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 174 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:174)）：

```ptx
@%p8 st.global.b64 [ %rd7 + 0 ], { %rd6 };
```

静态 15 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.global.b8</code> · 已有路径，待回归</summary>

```ptx
st.global.b8 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `a` | 输入，1 个原始位串8 位值 |

**行为：**按元组顺序将源寄存器的低 8 位写入设备全局内存，总计连续 1 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 58 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/6783f479ece78566.ptx:58)）：

```ptx
@%p1 st.global.b8 [ %rd2 + 0 ], { %rs1 };
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.global.f32</code> · 已有路径，待回归</summary>

```ptx
st.global.f32 [addr], b;
```

| 参数 | 含义 |
| --- | --- |
| `b` | FP32 浮点数源值 |
| `addr` | 字节地址，可含字节偏移；设备 global 内存 |

**行为：**向设备 global 内存写入 4 字节；按总访问宽度自然对齐。源寄存器更宽时按元素位宽截取，不能覆盖相邻字节。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新存储并生成写请求，影响数据可见性。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 1522 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:1522)）：

```ptx
st.global.f32 	[%rd69], %f67;
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.global.u64</code> · 已有路径，待回归</summary>

```ptx
st.global.u64 [addr], b;
```

| 参数 | 含义 |
| --- | --- |
| `b` | 64 位无符号整数源值 |
| `addr` | 字节地址，可含字节偏移；设备 global 内存 |

**行为：**向设备 global 内存写入 8 字节；按总访问宽度自然对齐。源寄存器更宽时按元素位宽截取，不能覆盖相邻字节。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新存储并生成写请求，影响数据可见性。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 1527 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:1527)）：

```ptx
st.global.u64 	[%rd72], %rd70;
```

静态 4 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.global.v2.b32</code> · 已有路径，待回归</summary>

```ptx
st.global.v2.b32 [addr], {a0, a1};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `{a0, a1}` | 输入，2 个原始位串32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入设备全局内存，总计连续 8 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 89 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0d3fa1d540e73560.ptx:89)）：

```ptx
@%p1 st.global.v2.b32 [ %rd3 + 0 ], { %r23, %r30 };
```

静态 5 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.global.v4.b32</code> · 已有路径，待回归</summary>

```ptx
st.global.v4.b32 [addr], {a0, a1, a2, a3};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 设备全局内存中的字节地址 |
| `{a0, a1, a2, a3}` | 输入，4 个原始位串32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入设备全局内存，总计连续 16 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 108 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0246c7163d0f0932.ptx:108)）：

```ptx
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r23, %r24, %r25, %r26 };
```

静态 70 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.param.b32</code> · 已有路径，待回归</summary>

```ptx
st.param.b32 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 内核/函数参数空间中的字节地址 |
| `a` | 输入，1 个原始位串32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入内核/函数参数空间，总计连续 4 字节。

**影响：**决定内核/函数实参的布局和取值；不能把参数空间误当成共享内存。

实际示例（[PTX 第 183 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:183)）：

```ptx
st.param.b32 	[param2], 30;
```

静态 14 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.param.b64</code> · 已有路径，待回归</summary>

```ptx
st.param.b64 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 内核/函数参数空间中的字节地址 |
| `a` | 输入，1 个原始位串64 位值 |

**行为：**按元组顺序将源寄存器的低 64 位写入内核/函数参数空间，总计连续 8 字节。

**影响：**决定内核/函数实参的布局和取值；不能把参数空间误当成共享内存。

实际示例（[PTX 第 179 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:179)）：

```ptx
st.param.b64 	[param0], %rd20;
```

静态 56 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.b32</code> · 已有路径，待回归</summary>

```ptx
st.shared.b32 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `a` | 输入，1 个原始位串32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入当前 CTA 的共享内存，总计连续 4 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 188 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/1fcdae4b81ca74a4.ptx:188)）：

```ptx
@%p19 st.shared.b32 [ %r13 + 0 ], %r14;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.b64</code> · 已有路径，待回归</summary>

```ptx
st.shared.b64 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `a` | 输入，1 个原始位串64 位值 |

**行为：**按元组顺序将源寄存器的低 64 位写入当前 CTA 的共享内存，总计连续 8 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 137 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/01074d70bb53382e.ptx:137)）：

```ptx
@%p5 st.shared.b64 [ %r2 + 0 ], %rd3;
```

静态 48 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.f32</code> · 已有路径，待回归</summary>

```ptx
st.shared.f32 [addr], b;
```

| 参数 | 含义 |
| --- | --- |
| `b` | FP32 浮点数源值 |
| `addr` | 字节地址，可含字节偏移；当前线程块 shared 内存，默认 shared::cta |

**行为：**向当前线程块 shared 内存，默认 shared::cta写入 4 字节；按总访问宽度自然对齐。源寄存器更宽时按元素位宽截取，不能覆盖相邻字节。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新存储并生成写请求，影响数据可见性。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 756 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:756)）：

```ptx
st.shared.f32 	[%r32], %f69;
```

静态 34 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.u32</code> · 已有路径，待回归</summary>

```ptx
st.shared.u32 [addr], b;
```

| 参数 | 含义 |
| --- | --- |
| `b` | 32 位无符号整数源值 |
| `addr` | 字节地址，可含字节偏移；当前线程块 shared 内存，默认 shared::cta |

**行为：**向当前线程块 shared 内存，默认 shared::cta写入 4 字节；按总访问宽度自然对齐。源寄存器更宽时按元素位宽截取，不能覆盖相邻字节。当前默认 weak，不自行建立 acquire/release 顺序。

**影响：**更新存储并生成写请求，影响数据可见性。空间、字节数、分量和线程地址影响合并、缓存或 shared bank 统计。

实际示例（[PTX 第 545 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/cd71bc5848a4af72.ptx:545)）：

```ptx
st.shared.u32 	[%r160], %r161;
```

静态 21 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.u64</code> · 已有路径，待回归</summary>

```ptx
st.shared.u64 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `a` | 输入，1 个无符号整数64 位值 |

**行为：**按元组顺序将源寄存器的低 64 位写入当前 CTA 的共享内存，总计连续 8 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 771 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:771)）：

```ptx
st.shared.u64 	[%r33], %rd79;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.u8</code> · 已有路径，待回归</summary>

```ptx
st.shared.u8 [addr], a;
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `a` | 输入，1 个无符号整数8 位值 |

**行为：**按元组顺序将源寄存器的低 8 位写入当前 CTA 的共享内存，总计连续 1 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 777 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/7bd6f3b7131e7873.ptx:777)）：

```ptx
st.shared.u8 	[%r34], %rs27;
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.v2.b32</code> · 已有路径，待回归</summary>

```ptx
st.shared.v2.b32 [addr], {a0, a1};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{a0, a1}` | 输入，2 个原始位串32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入当前 CTA 的共享内存，总计连续 8 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 197 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:197)）：

```ptx
@%p10 st.shared.v2.b32 [ %r39 + 0 ], { %r40, %r41 };
```

静态 31 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.v2.b64</code> · 已有路径，待回归</summary>

```ptx
st.shared.v2.b64 [addr], {a0, a1};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{a0, a1}` | 输入，2 个原始位串64 位值 |

**行为：**按元组顺序将源寄存器的低 64 位写入当前 CTA 的共享内存，总计连续 16 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 186 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/13feaf4f46a3800d.ptx:186)）：

```ptx
@%p10 st.shared.v2.b64 [ %r38 + 0 ], { %rd29, %rd30 };
```

静态 20 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.v2.u64</code> · 已有路径，待回归</summary>

```ptx
st.shared.v2.u64 [addr], {a0, a1};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{a0, a1}` | 输入，2 个无符号整数64 位值 |

**行为：**按元组顺序将源寄存器的低 64 位写入当前 CTA 的共享内存，总计连续 16 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 992 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:992)）：

```ptx
st.shared.v2.u64 	[%r619+214400], {%rd241, %rd242};
```

静态 32 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.v4.b32</code> · 已有路径，待回归</summary>

```ptx
st.shared.v4.b32 [addr], {a0, a1, a2, a3};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{a0, a1, a2, a3}` | 输入，4 个原始位串32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入当前 CTA 的共享内存，总计连续 16 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 63 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0246c7163d0f0932.ptx:63)）：

```ptx
@%p1 st.shared.v4.b32 [ %r6 + 0 ], { %r7, %r8, %r9, %r10 };
```

静态 16 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>st.shared.v4.u32</code> · 已有路径，待回归</summary>

```ptx
st.shared.v4.u32 [addr], {a0, a1, a2, a3};
```

| 参数 | 含义 |
| --- | --- |
| `addr` | 当前 CTA 的共享内存中的字节地址 |
| `{a0, a1, a2, a3}` | 输入，4 个无符号整数32 位值 |

**行为：**按元组顺序将源寄存器的低 32 位写入当前 CTA 的共享内存，总计连续 16 字节。

**影响：**数据宽度、向量顺序和地址空间决定计算值与访存请求；共享内存还影响 bank 冲突，全局内存影响合并访问及缓存耗时。指令本身不提供跨线程屏障。

实际示例（[PTX 第 5428 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:5428)）：

```ptx
st.shared.v4.u32 	[%r1880+213792], {%r2039, %r2040, %r2041, %r2042};
```

静态 1 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)。

</details>

<details>
<summary><code>sub.f32</code> · 已有路径，待回归</summary>

```ptx
sub.f32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | FP32 浮点数结果 |
| `a` | FP32 浮点数左操作数 |
| `b` | FP32 浮点数右操作数 |

**行为：**计算 `d = a - b`。默认最近偶数舍入；没有 `.ftz`，不能主动清零次正规数。

**影响：**影响数值或索引；消费者须等待 d 就绪，执行延迟进入相应运算管线。

实际示例（[PTX 第 156 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:156)）：

```ptx
sub.f32 	%f35, %f33, %f18;
```

静态 71 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#floating-point-instructions-sub)。

</details>

<details>
<summary><code>sub.s32</code> · 已有路径，待回归</summary>

```ptx
sub.s32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位有符号整数结果 |
| `a` | 32 位有符号整数左操作数 |
| `b` | 32 位有符号整数右操作数 |

**行为：**计算 `d = a - b`。保留指令位宽内结果，当前没有饱和修饰。

**影响：**影响数值或索引；消费者须等待 d 就绪，执行延迟进入相应运算管线。

实际示例（[PTX 第 50 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/080eab9246a9e166.ptx:50)）：

```ptx
sub.s32 	%r73, %r66, %r72;
```

静态 183 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-sub)。

</details>

<details>
<summary><code>sub.s64</code> · 已有路径，待回归</summary>

```ptx
sub.s64 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 64 位有符号整数结果 |
| `a` | 64 位有符号整数左操作数 |
| `b` | 64 位有符号整数右操作数 |

**行为：**计算 `d = a - b`。保留指令位宽内结果，当前没有饱和修饰。

**影响：**影响数值或索引；消费者须等待 d 就绪，执行延迟进入相应运算管线。

实际示例（[PTX 第 818 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx:818)）：

```ptx
sub.s64 	%rd209, %rd779, %rd784;
```

静态 6 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-sub)。

</details>

<details>
<summary><code>xor.b32</code> · 已有路径，待回归</summary>

```ptx
xor.b32 d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `d` | 32 位位串结果 |
| `a` | 32 位位串输入 |
| `b` | 32 位位串输入 |

**行为：**逐位 XOR；不作数值格式转换。

**影响：**影响 predicate、掩码、地址或描述符位域，进而改变控制流或访存。

实际示例（[PTX 第 76 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/756851cbac2894af.ptx:76)）：

```ptx
xor.b32  	%r22, %r20, 1;
```

静态 45 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-xor)。

</details>

<details>
<summary><code>xor.pred</code> · 已有路径，待回归</summary>

```ptx
xor.pred d, a, b;
```

| 参数 | 含义 |
| --- | --- |
| `参数` | a/b 是输入 predicate；d 在两者不同时为真。 |

**行为：**对两个 predicate 求逻辑异或。

**影响：**影响数值、索引或执行掩码；使用现有对应功能路径后仍需回归结果与依赖。

实际示例（[PTX 第 106 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/756851cbac2894af.ptx:106)）：

```ptx
xor.pred  	%p9, %p8, %p7;
```

静态 24 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#logic-and-shift-instructions-xor)。

</details>

<a id="failure-path"></a>

## 仅断言失败分支

<details>
<summary><code>call.uni</code> · 仅断言失败路径</summary>

```ptx
call.uni __assertfail, (message, file, line, function, charSize);
```

| 参数 | 含义 |
| --- | --- |
| `目标` | 本次仅 __assertfail，外部无返回断言函数 |
| `参数列表` | 在 .param 中准备：消息地址、文件名地址、行号、函数名地址、字符宽度 |

**行为：**执行一致分支的函数调用；`.uni` 声明同 warp 的调用目标/控制流一致。这里仅在框架检查失败时调用断言终止。

**影响：**加载时须接受外部函数声明和 `.noreturn`；正常前向不要求执行断言，但失败时不能当成功返回。

实际示例（[PTX 第 188 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:188)）：

```ptx
call.uni
	__assertfail,
	(
	param0,
	param1,
	param2,
	param3,
	param4
	);
```

静态 14 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#control-flow-instructions-call)。

</details>

<details>
<summary><code>trap</code> · 仅断言失败路径</summary>

```ptx
trap;
```

无显式操作数。

**行为：**触发线程异常，终止内核执行。当前位于断言失败分支。

**影响：**已有 opcode 的实现仍是占位；必须能清楚报告失败，不能把它算成正常计算已支持。

实际示例（[PTX 第 198 行](/home/wangjz/in-switch/accel-sim/workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx:198)）：

```ptx
trap;
```

静态 9 处；[NVIDIA PTX 8.7 官方条目](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#miscellaneous-instructions-trap)。

</details>

## 特殊寄存器与复核数据

`%tid/%ntid/%ctaid/%nctaid` 给出线程、CTA 与网格尺寸；`%laneid` 为 warp 内编号。新增的 `%lanemask_le` 返回 lane 0..当前 lane（含自身）的 32 位掩码，lane 31 为 0xffffffff；当前模拟器虽能解析该名字，取值实现仍会报未实现。`%cluster_ctaid.x/y/z` 给出 CTA 在 1×2×1 cluster 中的坐标。详见[官方特殊寄存器](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#special-registers)和[执行要求](execution-requirements.md)。

完整设备代码位于 [artifacts/ptx](../artifacts/ptx/)。[模块用途表](ptx-modules.md)解释全部 110 个模块；[workloads.json](../artifacts/workloads.json)保存用例的调用顺序与框架来源。
