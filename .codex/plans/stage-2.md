# 阶段 2 实现方案：让普通异步拷贝支撑真实 gating GEMM

本文件是待执行方案，不是完成报告。当前没有本阶段代码改动或运行验收。依赖阶段0的启动/动态shared，以及阶段1中gating所需的转换、普通计算和访存；配套：[阶段 2 验收](../validation/stage-2.md)。

## 完成后应当能做什么

gating GEMM 是把每个token的输入向量乘以路由权重矩阵，产生“给各专家的分数”。当前九例使用普通 `cp.async` 把数据从global（设备全局内存）搬到shared（本CTA共享内存），再用FP32乘加计算。这里不需要TMA、tensor map或WGMMA。

异步的含义是：发出搬运后，线程可以先做与这批数据无关的工作，稍后再等待它完成。为了同时算对数值和时间，模拟器必须记录**搬哪一段、属于哪个线程的哪一组、什么时候能读到**，不能只增加一个延迟，也不能发起时立即复制到可消费的shared。

```text
cp.async：发起global读取 ──→ 现有L2/DRAM请求和返回 ──→ 写入shared ──→ 标记完成
             │                                                       │
             └→ commit：给本线程此前拷贝划组 → 独立计算 → wait：检查所需旧组
                                                               ↓
                                                    必要的bar.sync → 读shared计算
```

`commit` 只是提交分组，不等待；`wait` 等搬运组，不代替不同线程之间的CTA屏障。把每次cp.async都改成“搬完才继续”会丢掉原程序的预取重叠，时间也会失真。

## 工作范围和具体交付

| 指令或能力 | 本阶段必须做到 | 验收 |
| --- | --- | --- |
| `cp.async.cg.shared.global` | 新增PTX语义：求两种地址、按predicate决定是否执行、搬16B或零填16B，并接入真实访存计时 | [单组拷贝](../validation/stage-2.md#copy) |
| `cp.async.commit_group` | 新增本线程分组，不等待；没有拷贝时仍提交合法空组 | [组边界](../validation/stage-2.md#groups) |
| `cp.async.wait_group` | 新增按本线程提交顺序等待；支持当前实际出现的N=0、2、4、6 | [组边界](../validation/stage-2.md#groups) |
| 动态请求身份与shared交付 | 同一PC反复执行仍匹配自己的返回；数据交付先于对应wait放行 | [复用与顺序](../validation/stage-2.md#reuse) |
| 普通shared、`bar.sync`、FP32 FMA和BF16转换 | 复用已有/阶段1路径，执行原gating并比较全输出 | [真实gating](../validation/stage-2.md#gating) |
| 主要时间行为 | 请求经过缓存/内存模型，等待不冻结其它warp，搬运和独立计算可重叠 | [时间](../validation/stage-2.md#timing) |

## 冻结程序实际用了什么

以下来自九份gating PTX的静态扫描，不是运行覆盖结果。每条普通cp.async的目的大小均为 `0x10`，即16B；源长度寄存器由常量16或 `selp` 的0/16二选一产生。`selp` 是按predicate选择两个值中的一个。

| 模块 | 实际wait常量 | 说明 |
| --- | --- | --- |
| 2、20、31、41、54、86 | 0、6 | 多组预取，先等旧组，最后排空 |
| 61 | 0、2 | 需要同一个按N计算的等待算法 |
| 71 | 0 | Qwen64/128首个代表 |
| 99 | 0、4 | 不能把wait写死为0/6分支 |

因此修正此前“只覆盖6/0”的描述。当前最重要的两种复制是“完整16B读取”和“源长度0、目的16B全部写零”；未发现1～15B部分源消费者，不能把它们说成当前必须解决的运行边界。

## 现有代码能做什么，还缺什么

| 静态代码事实 | 直接影响 | 实施位置 |
| --- | --- | --- |
| PTX opcode/功能执行缺普通cp.async三条入口 | 不能从PTX生成真实源地址、shared目的和组状态 | 解析与发起 |
| SASS LDGSTS已按global load生成请求，LSU有按动态指令UID统计未完成返回的路径 | 可以复用带宽/延迟；但trace不执行shared数据复制，必须补数据交付 | 请求与完成 |
| `issue_warp` 现有LDGSTS分组保存的是`next_inst`；`unset_depbar` 用PC和lane0地址匹配组中条目 | 不能原样作为PTX逐lane数据/循环身份。PC只是代码位置，同一位置和地址可以出现多次 | 身份与组 |
| SASS DEPBAR根据最近提交组类型选择LDGSTS/TMA/GMMA等待 | PTX有独立opcode，应直接选择普通拷贝组，不能靠上一条commit猜 | 等待 |

上述事实不等于当前PTX已复现“旧tile”“组误唤醒”或“卡死”。这些是本阶段实现必须用测试排除的后果，不是另设的前置修复工程。

<a id="copy"></a>

## 1. 从PTX建立一次真实拷贝

### 操作数与状态

实际形式例如：

```ptx
@p cp.async.cg.shared.global [shared_dst], [global_src], 16, src_size;
```

这里 shared_dst 是当前CTA内的偏移，global_src 是全局地址。`p=false` 表示整条指令不执行、目的保持原值；`p=true && src_size=0` 表示执行一次16B零填。这两种情况不能合并，否则矩阵边缘数据会残留。

1. 在 `ptx.l/ptx.y`、`opcodes.def` 和指令表示中接入copy、commit、wait三种opcode/子操作，保留 `.cg`、shared/global两种空间和全部源。decode把copy标为global读取、`m_is_ldgsts=true`、目的宽16B、`cache_op=CACHE_GLOBAL`；它没有普通寄存器结果。
2. shared目的虽在语法上列第一项，其**地址寄存器是输入依赖**，不能当输出寄存器。global地址和可变src_size也是输入；commit没有数据源/目的，wait的N是立即数。核对地址计算所依赖的寄存器已就绪再发起。
3. 只为predicate为真的实际lane建立拷贝记录。记录放在本次动态指令/未完成请求状态中，不写入共用静态 `ptx_instruction`。最少需要：本次UID、CTA/shared对象身份、warp/lane、当前线程组序号、目的偏移、有效源长度及待交付16B数据。
4. 本阶段可在发起时从功能global内存读取有效源字节，存入这份16B暂存，剩余字节置零；**此时不写shared**。当前gating的输入矩阵在kernel内只读，因此该快照满足当前需求。沿用既有内存读取方法，不复制整个矩阵或每组整块shared；不新增并发修改源数据的通用一致性模型。
5. `src_size=0` 不读取global，也不生成虚假DRAM请求；predicate=false连零填都不做。当前实现可自然使用“复制src_size字节、清零余下字节”的短代码，不必为0/16写两套算法。若额外部分源需要复杂请求拆分，可先记为未验收，不能阻塞当前九例。

**SASS复用：**LDGSTS表示global直接搬到shared，其访存类别、合并、队列和返回是可复用基础。新增记录仅提供PTX缺失的目的和数据；不要制造一个假寄存器load再额外发普通store来代替这条指令，那会增加不属于它的寄存器依赖和计时。

<a id="completion"></a>

## 2. 数据何时交付：接到真正的内存完成事件

LSU是执行load/store并处理返回的单元；UID是一次动态指令的唯一身份。循环每执行一次同一条PTX，都产生新的UID，不能只用代码地址PC区分。

### 具体接法

1. `exec_shader_core_ctx::func_exec_inst` 完成功能求值后，通过原 `generate_mem_accesses` 生成有效lane的global请求。`.cg`沿现有全局缓存策略绕过L1数据缓存、进入L2/DRAM。当前0/16范围只需排除零源lane并保留完整源lane的16B访问；不让目的零填长度扩大global流量。
2. 复用 `ldst_unit::m_pending_ldgsts[warp][uid]` 的请求/返回计数。在指令被流水寄存器复制、请求被合并或分拆时，保持UID和待交付记录关联；一份数据状态存一次，不随每个 `mem_fetch` 复制32份lane数据。不要将尚未执行求值的 `next_inst` 当成已有动态地址/有效mask的实例。
3. 在现有LDGSTS“该UID的有效请求全部完成”入口，调用一次PTX交付函数：把该动态指令所有实际拷贝写到捕获的CTA shared对象，之后再减少各lane所属组的未完成拷贝数，最后检查wait是否可解除。
4. 本阶段沿用现有**warp指令的最后一个请求返回时统一交付**的粒度；它可能晚于其中某个lane最早返回，但不会在数据未完成时提前唤醒。记录这个时间模型假设，不声称已逐lane建模shared写端口。若gating主要时间偏差指向这一粒度再细化，不预先增加每字节完成网络。
5. 全零源的动态指令没有内存返回，要在正常发射/完成处理内交付零填并完成记录；不能建立“等一个返回”的计数，也不能由于访问队列为空而漏写零。混合有效源和零源时，允许随同一动态指令最后返回统一交付。
6. 核对已有完成分支的缓存返回、旁路返回和无请求结束路径，把PTX交付挂到共同的完成动作，保证只执行一次。已有SASS指令没有PTX数据记录时沿原计时路径，不重复扣计数或重复 `unset_depbar`。

**必须保持的顺序：**请求发起→有效请求完成→shared交付→拷贝/组标记完成→解除对应wait→合法消费者读取。计时模型里global sector/事务可能比有效数据大，不能要求DRAM字节数恰等于16×lane数；但不允许为纯零填生成一次读取。

**SASS复用与近似边界：**复用L2/DRAM延迟、带宽、请求合并、返回仲裁及LDGSTS完成位置。shared消费者继续走已有bank/端口模型。本阶段不另建异步写shared的精细bank冲突模型，也不重复加一条普通store的费用；在报告中记录这种沿用SASS模型的近似，主要流量和gating时间检查不可省。

<a id="groups"></a>

## 3. 每线程分组与wait：明确等的是哪几组

“每线程”指lane0的拷贝由lane0自己的commit归组；lane1即使执行同一个commit位置，也有自己的组。不必为32个线程各写一套类，可以在现有warp状态里保存32项紧凑记录。

### 最少状态和操作

| 状态 | 用途 |
| --- | --- |
| 每lane当前组序号、未提交组未完成拷贝数 | 发起copy时归属本轮，完成事件可以早于commit到达 |
| 已提交组的序号和未完成计数 | 按顺序判断旧组是否完成；空组计数为0 |
| 本次wait的参与lane及各lane截止序号 | 等待边界固定在这次wait，不被后续其它活动修改 |
| UID→本次拷贝记录 | 完成回调能定位组及shared目的，不靠PC或lane0地址猜测 |

实现步骤：

1. 每lane从组序号0开始，copy加入该lane的当前未提交组；增加未完成数，再允许copy完成事件扣减。谓词为假的lane不加入copy，但之后若执行commit，仍可能产生空组。
2. commit把当前组记为已提交，再让当前序号加1；不发global请求、不等待未完成数归零。空组也前进序号，并视为已完成。
3. 设某lane执行wait时共提交了C组，编号为0～C-1。`wait N`只要求编号 `< max(0,C-N)` 的组完成，最近N个已提交组允许未完成；尚未提交组不在这个等待范围内。例如C=8、N=6，必须完成0、1，不能只检查“目前未完成组总数≤6”。
4. wait进入等待时冻结这个截止序号；每次相关copy完成后检查所有参与lane是否满足各自条件，满足后才让本次指令之后的代码继续。只根据普通copy队列判断，不访问TMA/WGMMA队列或 `m_last_depbar_group_type`。
5. 旧组全部完成且不再需要保存其拷贝内容后释放数据记录。保留单调提交序号，用序号求截止位置，不能按删减后的队列长度重新计算C，否则空组或先完成组会改变wait含义。寄存器/线程槽位复用前清空本次kernel的组状态。

组/等待语义依据：[官方commit](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group)、[官方wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)。这里只新建当前普通拷贝所需状态，不抽象成三类异步引擎的通用框架。

### 等待接在哪里

借鉴 `shd_warp_t` 的等待标志和 `waiting()` 调度检查：功能执行登记本次wait条件后，阻止后续指令越过该点；访存返回和其它可调度warp继续运行。当前同一SIMT发射路径可以在warp层停发，但条件须逐lane计算，不能把没有执行wait的lane当成额外等待者，也不能等待另一个warp的组。

不依赖回退重执行PC来“试到完成”，避免反复创建组或重复发copy。若真实控制流要求部分lane继续运行，沿阶段1确认的SIMT路径处理，不在本阶段预先做一般独立线程调度。

**SASS复用：**保留已有LDGDEPBAR/DEPBAR的发射、等待和完成挂接位置。PTX的组归属、显式N和UID关联使用以上确定规则；不强迫SASS trace也重新生成PTX数值状态。若局部共用完成代码发生变化，仅回归受影响LDGSTS路径。

<a id="reuse"></a>

## 4. shared复用与其它线程读取

wait完成只确认执行该wait的线程所提交的相关copy。若lane0负责搬一段、lane1负责读取它，程序还需要原本的线程间同步；同样，旧数据还被其它线程消费时，不能仅凭“下一次拷贝可以发起”就覆盖同一个shared槽位。

实现时保留原gating的 `bar.sync` 及其位置，不自动在每个copy后加全CTA屏障，也不省掉原来的屏障。shared对象属于发起CTA，完成回调不能按“当前占据这个硬件线程槽的CTA”重新查目的。完成记录在该copy完成前保持有效，CTA释放与线程槽位重用不得丢掉已有未完成操作；先复用现有未完成指令/退出约束，仅在所测路径证明缺失时局部接通。

验证用原程序顺序的多轮缓冲轮换，检查数据与事件。读到旧tile或循环组混淆是本阶段要排除的实现错误，尚未在当前workload上运行复现；不把它们列成“先修完才能开始cp.async”的独立任务。

<a id="gating"></a>

## 5. 真实gating与实施顺序

1. 先接通完整16B拷贝、commit、wait0及单组输出；同时让请求走原LSU计时，不先交付一个同步memcpy版本作为“完成”。
2. 补零源/predicate、空组和N的截止算法；用0/2/4/6及循环复用测试组边界。完成回调按UID稳定工作后，才跑长循环。
3. 首先运行模块71（Qwen64/128，wait0），再运行模块2（Mixtral/128，wait6/0）。使用单模块入口和原合法launch，CPU独立生成输入与矩阵参考，不要求冻结完整启动器先通过。
4. 这两个代表验证gating主路径；模块61/99的2/4由微测先覆盖，真实调用留到九例推进。若两代表暴露未覆盖的数据类型/布局差异，就按具体差异补当前消费者，不能宣称两例等于九例全部通过。
5. 每个代表记录初始化、模拟、比较墙钟和RSS。性能比较使用冻结尺寸、launch和可比缓存前序；人工小值只验证数值，不用它与真实ncu时间直接计算20%误差。

## 暂缓范围、证据入口和实际交付

| 暂缓项 | 理由/重开条件 |
| --- | --- |
| `.ca`、其它copy目的长度、`wait_all`、未使用的cache hint | 当前普通拷贝只有 `.cg` 16B及commit/wait；实际出现再加 |
| 1～15B部分源的复杂拆分与穷举 | 九个gating源长度静态为0/16；简单代码自然支持可保留，但不为此增加复杂通路和验收矩阵 |
| 数据竞争、非法重叠目的或并发改源的恢复 | 不属于合法当前程序；不做补零容错、重试或自动插入屏障 |
| 精细异步shared写bank建模、逐lane最早返回交付 | 先沿用LDGSTS时间粒度，避免新建搬运引擎；主要时间偏差有证据再重开 |

| 源码/材料 | 查看和修改内容 |
| --- | --- |
| [PTX词法](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l)、[语法](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y)、[opcode表](../../gpu-simulator/gpgpu-sim/src/cuda-sim/opcodes.def)、[instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc)、[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) | 三条指令入口、地址/长度求值、真实predicate、源依赖及无寄存器输出的分类 |
| [abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h)、[.cc](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc) | 动态warp指令UID、有效访问mask、global请求合并及附属记录关联 |
| [shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)、[shader.h](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h) | `issue_warp`、`m_pending_ldgsts`、`writeback_complete`及其它LDGSTS完成分支、`unset_depbar`、`waiting()` |
| [trace_driven.cc](../../gpu-simulator/trace-driven/trace_driven.cc) | `OP_LDGSTS/OP_LDGDEPBAR/OP_DEPBAR`分类和复用边界 |
| [Qwen64/128 PTX](../../workloads/moe/artifacts/ptx/b4d06c15353157e5.ptx)、[Mixtral/128 PTX](../../workloads/moe/artifacts/ptx/cf685bb5cf3652b7.ptx)、[全部模块](../../workloads/moe/artifacts/workloads.json) | 实际copy长度、src_size来源、wait常量、循环/barrier与矩阵地址计算 |

执行后列清三条指令实际实现/修复内容、完成回调粒度、真实gating数值/时序、受影响SASS回归和暂缓项，链接[验收报告](../validation/stage-2.md#results)。当前没有实现或运行结果。
