# 阶段 4 实现方案：让mbarrier同时等待线程到达和搬运字节

状态：待执行方案，尚无本阶段实现或运行验收。远端操作依赖阶段3的CTA身份和shared目标解析，本地状态可先做；配套：[阶段4验收](../validation/stage-4.md)。

## 完成后应当能做什么

PTX可以在shared地址初始化屏障、登记参与者和待搬运字节，并查询某轮是否完成。mbarrier是一种有状态的屏障：普通到达数和搬运完成字节是两个独立条件，只有两者都满足才能进入下一轮。

例如初始化期望2次到达，登记64B搬运：两次到达都发生但只搬好32B时仍不能计算；64B都到而还差一次到达时也不能计算。phase是内部轮次，parity是轮次奇偶值0/1，供PTX用同一地址反复等待。

```text
init(count=2) → 本轮 pending arrivals=2，tx=0，phase=0
expect(64)   → tx=64
arrive两次   → pending=0，但仍等64B
complete(64) → tx=0，phase变1，下一轮pending重装为2
```

完成本阶段不代表TMA已搬过数据。非零字节先通过正式状态接口做组件测试，阶段5必须用真正的TMA返回再次证明通知在数据之后。

## 全部指令、现有事实与处理

| 指令 | 当前需要的行为 | 本阶段动作 |
| --- | --- | --- |
| `mbarrier.init.shared::cta.b64` | 在本CTA地址建立到达数和初始轮次 | 新增PTX入口，复用已有状态对象 |
| `mbarrier.arrive.shared::cta.b64` | 本地减到达数，当前结果为`_` | 新增求值/副作用，不丢弃操作 |
| `mbarrier.arrive.shared::cluster.b64` | 对mapa指定目标减到达数 | 接阶段3目标解析和远端交付 |
| `mbarrier.expect_tx.shared::cta.b64` | 加期望字节，不减到达数 | 新增入口，状态更新后检查完成 |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | 一次操作登记字节并到达 | 原子更新两个计数，再检查完成 |
| `mbarrier.try_wait.parity.shared::cta.b64` | 输入地址/parity/可选hint，输出真假 | 新增PTX求值和有界等待适配 |
| `fence.mbarrier_init.release.cluster` | 发布此前屏障初始化 | 接初始化与跨CTA后续访问的顺序 |

静态核查：已有 `mbarrier_t` 保存到达数、`int32_t tx_count` 和phase；`arrive_on/complete_on`检查轮次转换，`expect_on`目前只加数。现有 `mbarrier_waiting` 把输入右移31位，这是SASS记录的编码；PTX直接传0/1，不能照搬。SASS `check_trywait_ready` 在发射前取trace操作数，并含 `|| true` 的持续重试路径；PTX值尚未求出时不能使用它。

这些是可定位的静态差异，尚未在当前PTX运行复现挂死或提前唤醒。

<a id="state"></a>

## 1. 一份状态，两个计数，一个完成入口

1. 继续使用 `mbarrier_t` 和kernel下的 `ClusterMbarriersLookupTable`。key由本kernel、cluster、目标rank、shared offset组成；本地地址先绑定本CTA，远端地址经阶段3解码。两个CTA相同offset不是同一个屏障。
2. init按实际count设置 `expected_arrivals=count`、`pending_arrivals=count`、`tx=0`、`phase=0`。shared中的b64地址是对象身份，不需要与真实GPU的8B不透明屏障编码互换；当前程序通过mbarrier指令访问状态，不新增普通load解码屏障内部的旁路。
3. 各变更集中调用一次 `try_phase_transition`：仅当pending=0且tx=0，phase加1并把pending重装为expected。phase内部保留轮次整数，PTX查询取低位；不能只用永久ready标志。
4. 单独expect修改tx后也检查转换，覆盖“完成先到、到达已齐、最后expect配平”的路径。组合 `arrive.expect_tx` 必须把 `tx+=bytes` 和 `pending-=arrival_count` 作为一次状态变更后检查，不能调用一个会中途翻转的expect再做arrive。
5. tx保持有符号数。当前专家有先发TMA、后expect的合法顺序：若64B先完成，tx为-64；稍后expect64才回0。不能负值报错、钳零或忽略这次完成。合法程序仍必须保持正确的到达/轮次顺序，不为任意错误通知增加恢复逻辑。
6. TMA请求捕获目标屏障对象和发起轮次，完成时扣实际逻辑字节。阶段5负责保证数据已交付、每字节只通知一次；本阶段不再用“最近使用的屏障”查目标，也不在错误回调上自动切到当前phase。

**SASS复用：**保留现有状态类和查表/完成函数，SASS入口负责把自己的地址和phase编码转换后再调用。已有针对其它trace内核的 `init_as_one` 或首次trywait修正不能默认套到PTX init上。

<a id="execution"></a>

## 2. 求值、状态修改和predicate写回分别放在哪里

PTX功能函数与性能模型都会经过一条指令；若两边各减一次到达数，屏障就会提前完成。采用一个明确分工：

1. parser新增实际修饰和操作数；decode把地址、count/bytes、parity、hint登记为输入，try_wait的predicate登记为输出。`_`只代表没有返回寄存器，到达副作用仍存在。
2. `instructions.cc` 的入口从当前线程寄存器求值，填入**本次动态指令**的 `syncs_operand` 或等价已有记录，不在此先修改一次屏障。不要把运行时操作数写进共用静态PTX对象。
3. 在LSU现有屏障处理/服务位置应用一次状态变更；远端到达在通知到达目标时修改。plain arrive无显式count时减1；有显式count的实际形式按操作数减，不能一概减warp宽32。全warp分别执行32次arrive与一个leader执行一次arrive要区分。
4. try_wait取得结果后沿正常目的寄存器/scoreboard写回，采用模拟器既有predicate真/假编码。若结果需要等待，保存目标线程和目的寄存器直到完成，不能先写假再让后面的分支越过尚未结束的指令。
5. init和到达各自仍占用既有屏障/LSU服务资源。不是把一切副作用提前放到功能阶段、只在旁边累计一个互不相关的周期数。只改当前PTX入口和必要共用变更，不重构全部SASS barrier分派。

**SASS复用：**已有LSU屏障服务、动态指令记录和完成位置可复用；需要补的是PTX运行时求值和predicate写回，不能使用trace预先给定的值代替。

<a id="polling"></a>

## 3. parity与hint：让原PTX轮询真实地得到真假

正常等待刚刚发起的一轮时，`current_phase&1`仍等于输入parity，结果为假；该轮完成、phase翻转后结果为真。PTX直接使用parity0/1，SASS的bit31提取只留在SASS入口。

当前PTX同时有三操作数和四操作数try_wait：后者多一个纳秒hint，表示允许尝试等待的时间提示，不是必须固定睡满的周期数。

- 已完成：按普通执行延迟写真，不再额外等待hint。
- 未完成、无hint：首版采用正常计时的观察后返回假，由原PTX循环重试。它是明确的轮询近似，不走现有无限阻塞的trace前置检查。
- 未完成、有hint：优先复用现有warp暂停/唤醒容器，保存本次地址、parity、目的和截止时刻；`deadline=now+ceil(hint_ns*core_MHz/1000)`。屏障完成可提前唤醒并写真，期限结束重新观察后写真/假，不强制一直等到真。
- 暂停只影响调用warp，LSU返回、其它warp和同伴CTA继续运行；同一条等待只登记一次，不能周期性重发到达或复制指令。重复轮询来自PTX原分支，不是生产路径自动恢复机制。

首版若有hint也采用允许立即返回假的实现，必须在交付记录明确这一选择，并测真实专家的轮询指令量/等待周期。它不能造成忙轮询主导性能后仍被当成低影响近似；出现这种证据时补有界暂停。语义依据：[官方try_wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-try-wait)。

**SASS复用：**借用warp等待集合、条件检查和唤醒调度；PTX parity转换、有界截止及返回假需明确实现，不直接沿用现有无限重试条件。

<a id="ordering"></a>

## 4. 初始化发布与远端使用

屏障init由一个线程执行，并不等于另一CTA已经可以立刻使用。`fence.mbarrier_init.release.cluster` 在已有初始化服务完成后，建立对本cluster后续使用的发布顺序；实际专家随后还执行cluster arrive/wait。保留这两步，不把一个换成另一个。

默认arrive的release和成功wait的acquire按实际指令作用域接到已有相关访存完成状态。成功等待TMA字节的acquire必须晚于数据交付；初始化fence只发布屏障初始化，不替代普通shared→async的proxy fence，也不替代TMA完成等待。

远端arrive捕获发起者和目标身份，经阶段3的通知服务到达目标后扣计数。即使结果为`_`也要发通知。复用同一个目标解析入口，按不同cluster相同offset验证隔离。对象所有权随cluster，不在CTA退出时遗留指向已复用存储的回调。

**SASS复用：**复用cluster查表、通知完成和可用的fence/访存等待状态；新增PTX初始化发布与真实目标对象的关联，不建立全局内存版本系统。

## 实施顺序、暂缓和证据入口

先本地init/arrive/无hint轮询→组合expect与状态组件→远端通知/初始化fence→hint及三轮复用。阶段5实际TMA返回完成后重验非零字节，不等待TMA才编写本阶段的基础测试。

暂缓arrive_drop、未使用的token消费/test_wait、任意久远phase查询、非法重初始化及错误通知恢复。当前负tx、两个条件独立、重复轮次和远端到达都不能暂缓。

| 位置 | 实现内容 |
| --- | --- |
| [abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) | `mbarrier_t`、计数更新/翻转、kernel所有的屏障查表 |
| [shader.h](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h)、[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) | `mbarrier_waiting`、`check_trywait_ready`、LSU服务/通知和暂停唤醒 |
| [instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc)、[cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) | 新语义求值、输入/输出依赖、动态操作数接线 |
| [专家PTX](../../workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx) | 初始化发布、远端到达、hint及先TMA后expect的实际消费者 |

实际六种mbarrier及fence的实现/修复、hint模型与[验收记录](../validation/stage-4.md#results)分别填写。当前没有运行证据。
