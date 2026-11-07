# 阶段 1：基础数值与 warp 协作

状态：仅方案；本轮无实现、修复或运行验收。[对应验收](../validation/stage-1.md)。依赖阶段 0 的加载与单模块入口。

## 本阶段解决什么

先运行转换、选专家、计数和重排等真实内核，尽早发现会改变后续专家输入的数值或索引错误。warp 是 32 个线程组成的执行组；这里的协作指令必须使用它们显式指定的参与线程，不能把所有线程都算进去。

## 指令、实现动作与现状

| 指令/配置 | 拟实现或扩展 | 当前状态及影响 |
| --- | --- | --- |
| `cvt.f32.bf16`、`cvt.rn.bf16.f32`、`cvt.rn.bf16x2.f32` | BF16 位表示、最近偶数舍入、双值打包顺序 | 待扩展；错误会直接改变路由或矩阵输入 |
| `shfl.sync.bfly.b32`、`shfl.sync.idx.b32`、`shfl.sync.up.b32`、`vote.sync.ballot.b32` | 解析 member mask，保留索引/分段/predicate 语义 | 现有旧形式待扩展；参与集合不对会得到错误索引/归约结果 |
| `bar.warp.sync` | 只等待参与 mask 内线程，并保证所需 shared 顺序 | 待扩展；不能替换成整个 CTA 的屏障 |
| `redux.sync.add.s32`、`elect.sync` | 参与集合求和；一致选一个 leader，支持双输出或 `_` | 待新增；计数错误或重复 leader 会让后续搬运出错 |
| `bfind.shiftamt.u32`、`%lanemask_le` | 补 shiftamt 变体与 lane 掩码取值 | 前者待扩展；后者只有词法入口，尚无取值实现 |
| `ld.global.L1::evict_first.b32`、`ld.global.L1::evict_first.b8`、`ld.global.L1::evict_last.b32` | 接受实际修饰，正常生成读取地址和缓存请求 | 待扩展；不得因 hint 丢掉真实访存 |
| integer latency/initiation 的 SHFL 第六项 | 按解析端的六项格式补齐 H100 PTX 配置 | 当前仅五项，shuffle 时间参数不完整 |
| 普通算术、分支、load/store、shared atomic、`bar.sync` | 通过真实消费者回归 | 已有路径，不代表 154 种写法均已运行通过 |

表内“错误会怎样”是实现不正确的后果，不是本轮复现结果。静态缺口依据见[调查笔记](../hopper-notes.md#current-gaps)。

## 如何实现

1. **数值转换沿旧入口扩展。** BF16→FP32 将 16 位放在 FP32 高半部；FP32→BF16 对截掉的低位做最近偶数舍入，不能直接截断。打包指令按 PTX 规定的两个操作数次序放高/低半部，并以两个不同值验证。复用现有类型/寄存器值存取；特殊浮点处理能沿现有实现正确保留就保留，不另建浮点库。
2. **集合操作先收输入，再写输出。** 把 member mask 作为独立操作数保存，不与 shuffle 的 clamp/索引混淆。沿现有线程执行/warp 协作入口收齐本次参与者的输入，再计算输出，防止先执行的 lane 覆盖后执行者要读的值。只在现有 SIMT 路径不足以表达实际合法汇合时补最小等待状态，不预先实现完整独立线程调度。
3. **屏障不扩大范围。** `bar.warp.sync` 按本 warp、本次指令及参与 mask 等待；完成时发布参与者所需 shared 写入，不等待整个 CTA。普通 CTA barrier 继续走原路径。
4. **复用简单运算。** redux 在合法参与集合累加 s32 并广播；elect 可确定性选参与集合中的最低 lane，ISA 不要求硬件也选该 lane，验收检查唯一性与一致性。`%lanemask_le` 用 lane 编号构造低位全 1，lane31 的实现避免 32 位移位越界。`bfind.shiftamt` 复用已有位查找并补 PTX 所需返回值。
5. **时间随功能走。** shuffle/集合指令沿对应执行单元和依赖路径计时；填齐 SHFL 配置后用依赖链核查参数生效。cache hint 保留普通访问和已有 L1/L2 行为；只记录细微替换策略近似，不为 hint 单独造缓存。

先交付转换和路由/计数所需部分；只用于专家的 elect、打包转换等最迟在其消费者前完成，不把它们人为放到普通内核之前。

## SASS 可复用与不能照搬的部分

复用已有 SIMT 调度、算术/整数/SHFL 流水、scoreboard、shared bank 和缓存模型。SASS trace 的 active mask 是已有执行结果，不能代替 PTX 自己按 predicate/member mask 求参与集合。BF16 数值、redux、elect 和结果分发必须由 PTX 功能执行产生；时序分类可以借鉴 trace opcode 对应关系。

## 真实消费者与暂缓范围

用模块 0/70 转换、3/4 top-k/排序、10/14 映射、82 SwiGLU 等代表检查输入→输出；不必逐一单测 110 个模块。首个真实普通内核就记录初始化、模拟、比较墙钟和 RSS，不能用它的速度直接预测 WGMMA。

| 暂缓项 | 为何可以暂缓；何时重开 |
| --- | --- |
| 未使用的 BF16 算术、其它 redux/vote 变体 | 冻结指令中无消费者；新增实际指令再扩展 |
| 非法 mask、未定义 shuffle 源的恢复与穷举 | 不属于合法 workload；不添加补零或强制放行 |
| 全 IEEE 特殊值组合的大规模测试 | 当前验收聚焦真实转换与舍入；不允许破坏已有特殊值处理，实际结果出现非有限值必须定位 |
| eviction hint 的精细替换优先级 | 先保留正确数据与缓存通路；主要 L2/DRAM/时间差额指向它时再做 |

## 源码入口与实际交付记录

[instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) 的 `cvt_impl`、`shfl_impl`、`vote_impl`、`bar_impl`、`bfind_impl`；[ptx_sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_sim.cc) 的特殊寄存器取值；[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) 的 decode/时序分类；[H100 配置](../../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config)。

待填写：具体已实现/修复的指令、提交、真实 entry、验收链接和暂缓项。当前：无。
