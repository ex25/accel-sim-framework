# 阶段 4 验收：检查屏障的两个条件、真假返回和重复轮次

状态：未构建、未运行。配套：[实现方案](../plans/stage-4.md)。复用阶段0入口，拟在 `workloads/moe/tests/hopper/stage-4/` 保存测试。所有非零字节组件结果必须与后续TMA运行结果分开标注。

## 要证明什么

六种mbarrier及初始化fence能作用于正确对象；到达和字节同时齐备才翻转一次；原PTX轮询能取得正确predicate，且等待不阻止生产者或返回事件推进。

<a id="local"></a>

## 1. 本地零事务程序：能实际得到假和真

一个CTA至少两个warp，shared中选8B对齐offset=1024。由一个线程init(count=2)，执行初始化fence并用合法CTA同步让使用者开始。安排两个leader各arrive一次，其中一个通过有限计算晚到；等待线程以parity0查询并将predicate转成整数输出。

先用组件检查固定“未到齐时查询”为假；周期运行通过事件日志确认实际查询先后，不靠不同warp一定按某个顺序执行。若运行中首次查询已经完成，不强求它返回假，应调整合法延迟取得至少一次未完成观察。

预期状态：初始化 `(pending=2,tx=0,phase=0)`；第一次到达为 `(1,0,0)`；第二次后为 `(2,0,1)`，其中2是下一轮重装值。等待该轮parity0结果为真。检查每个arrive只扣一次，`_`没有返回值但副作用存在。

<a id="counts"></a>

## 2. 非零事务组件：固定状态表直接给出预期

通过正式状态函数和未来TMA完成入口测试，不往生产PTX增加测试opcode。每个用例重新合法初始化，以下“完成”是人工调用正式完成接口，不能写成真实DRAM搬运。

| 初始状态/操作序列 | 关键预期 |
| --- | --- |
| init2→expect64→arrive1→arrive1 | pending0、tx64、phase0，仍未完成 |
| 上行继续complete32→complete32 | 第一次仍phase0/tx32；第二次phase1/pending2/tx0 |
| init2→expect64→complete64→arrive1→arrive1 | 字节先齐也不翻转；最后到达才变phase1 |
| init1→complete64→arrive1→expect64 | tx先为-64；arrive后pending0/tx-64仍phase0；最后expect配平触发phase1 |
| init1→arrive.expect_tx(64) | 原子结果pending0/tx64/phase0；不在两个子操作之间误翻转；complete64后phase1 |

最后一项专门避免“先arrive触发零事务完成，再给下一轮加64B”的错误。第四项说明负数是当前合法顺序需要，不要求为无所属事务的重复通知提供容错。

每行检查最终值及中间值；预期写死为上表，不能调用模拟器的完成判断函数产生参考。phase翻转次数也要精确，不能只比较最后parity。

<a id="remote"></a>

## 3. 真实远端arrive和初始化发布

采用阶段3的两个cluster，每组两个CTA。rank1在offset1024初始化count1，执行 `fence.mbarrier_init.release.cluster`，所有线程按原模式完成cluster arrive/wait。rank0将自己的同offset映射到rank1，只由一个leader执行 `mbarrier.arrive.shared::cluster.b64 _,[mapped]`；rank1本地轮询parity0并输出完成标记。

第二个cluster使用不同延迟，记录通知发起、目标交付、目标phase转换和predicate写回。预期只更新所指cluster/rank1的屏障；rank0自己的同offset屏障和另一cluster不会被这次通知扣数。等待在通知实际到达前不应通过。

先分别初始化需要观察的本地对象，不能把“查不到对象”当正常路径后自动创建。该测试为真实PTX远端副作用，补足阶段3仅验证mapa目标解析的边界。

<a id="phase"></a>

## 4. 同一对象循环三轮

用count2的屏障合法执行三轮，输入parity依次0、1、0。每轮所有消费者确认完成后，再开始下一轮；输出轮次标记分别11、22、33，不查询任意久远轮次。

保存每轮pending/tx清零、phase变化与重装，以及wait的输入parity。预期phase为1、2、3，不能在第二轮直接沿用第一轮的ready结果。非零字节版本在组件先测，阶段5实际TMA流水再重验。

<a id="hint"></a>

## 5. hint、时间和predicate依赖

覆盖真实PTX两种try_wait形式：无hint；hint由寄存器提供。选一个可换算的小hint，例如100ns，若core为1980MHz对应198周期上限，具体运行使用实际频率计算。

- 条件已满足时应返回真，不为hint强制再等满100ns。
- 条件未满足且在期限内变为满足时，有界等待版本提前写真；若期限结束仍未满足，返回假，之后原PTX循环能继续等待成功。
- 等待期间另一个warp/CTA执行最后到达，LSU仍能处理完成；不能把最后到达者一起挂起。
- 分支紧接try_wait读取predicate，必须等本次结果写回；不能使用前一轮残留predicate。

如果首版选择非挂起轮询，记录此选择，上面“必须提前唤醒”的要求改为检查原循环最终正确，并报告轮询次数/周期；不把它冒称有界等待已实现。真实专家出现明显轮询开销时，这项近似需重开。

无hint和hint路径都不能进入仅依赖trace操作数的发射前无限重试。SASS bit31编码在SASS入口回归，PTX0/1不右移31位。

## 证据、回归和推进条件

每项保存kernel/cluster/rank/offset、操作身份、pending、tx、phase前后值及释放周期；只对关键事件记录，正式长运行关闭详细日志。解释“还差一次到达”或“还差32B”，不只打印屏障编号。

共用状态函数改动回归一次SASS本地/组播完成适配，检查没有同一事件在功能和时序两处更新；无trace则注明未覆盖。初始化fence只证明屏障发布，不能写成普通shared数据或TMA已经可见。

零字节PTX本地/远端、非零组件和三轮状态通过即可推进阶段5。阶段5使用真实搬运返回前，非零TMA运行状态仍是未验证。

<a id="results"></a>

## 实际结果

| 项目 | 当前状态 | 后续填写 |
| --- | --- | --- |
| init/本地arrive/try_wait | 未实现/未运行 | 指令、predicate和实际状态序列 |
| expect、arrive.expect_tx及完成组件 | 待接通/未运行 | 固定序列、中间计数、转换次数 |
| 远端arrive/初始化fence | 未实现/未运行 | 目标隔离、通知与可见顺序 |
| 三轮phase/hint | 未验证 | 采用模型、轮询/暂停和结果依赖 |
| SASS适配/真实TMA完成 | 未验证 | 前者回归，后者留阶段5复验 |
