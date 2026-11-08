# 阶段 5 验收：检查描述符、TMA数据交付与两个完成时刻

状态：未构建、未运行。配套：[实现方案](../plans/stage-5.md)。沿阶段0入口，拟在 `workloads/moe/tests/hopper/stage-5/` 保存测试。按非组播→首次联测→组播→输出推进，不等所有项目齐备才接矩阵计算。

## 当前数据范围与覆盖

只读解码九份forward.bin得到相同的三种参数组合（每例有四个描述符，其中一种重复）：

| 类型 | rank/box | swizzle | 单tile逻辑字节 |
| --- | --- | --- | ---: |
| BF16 | 2，32×64 | 64B | 4096 |
| UINT16位存储 | 2，64×64 | 128B | 8192 |
| UINT16位存储 | 2，64×128 | 128B | 16384 |

均为element stride1、无interleave、零OOB填充、L2 promotion128B。编码/replace、发布、输入两形式、输出、bulk commit/read wait及共用async fence均需有对应记录；不扩其它参数组合。

<a id="descriptor"></a>

## 1. 128B对象与字段修改

经正式driver-entry查询取得编码函数指针，再调用编码，不能直接从测试调用私有辅助函数冒充ABI通过。准备两个不同模拟global基址、非紧凑行步长的矩阵；保存128B输出，按计划的明确偏移检查base/dim/stride/box等字段。

将对象复制为两份，沿真实路径经过参数内存、global和shared。只修改副本B的base、dim0/1、stride0；副本A按字节保持不变。再按实际PTX写inactive dim2～4=1及stride1～3的合法当前值，rank仍为2，后续二维地址不受这些未用维度影响。

定向检查单位：把stride0从256B改为320B，逻辑相邻行地址应只增加320B，不能变成20B或640B；dim写64后应保存64而不是63。base保留完整64位模拟地址，不能丢阶段3的新高位。

通过条件包括：原ABI调用成功、对象可逐字节复制、所有被替换字段准确、两份不串改。初始stride0占位对象可编码，不要求它在replace前就用于有效矩阵搬运。

<a id="publication"></a>

## 2. replace→复制发布→acquire→读取新对象

warp按实际形式执行replace、`tensormap.cp_fenceproxy ...128`，随后消费者按原同步顺序执行acquire和TMA。第一次指向矩阵A（标记范围1000起），第二次合法更新为矩阵B（标记范围20000起）。

检查每次TMA捕获的对象字段和实际数据均对应本次发布；已发起的A操作不因B发布而换源。副本A再用一次，证明未被修改。记录一次128B复制的有效字节与合并请求，不能出现32×128B重复功能复制/流量。

测试对发布及后续使用加已有合法同步，不用删除fence制造竞争后要求必然读旧值。单测fence的顺序事件与数据一起检查，不能仅“最终值碰巧正确”。

<a id="input"></a>

## 3. 非组播输入、swizzle与真实mbarrier完成

对上表三种组合各准备一个合法非紧凑global矩阵，逻辑元素位模式用 `value(x,y)=x+128*y+1`，尺寸取128×160，行步长320B。这里检查2B位数据，BF16组合也不把这些位当算术值计算。

每次在shared先填非零标记，初始化屏障并登记该tile完整逻辑字节，然后发TMA、arrive、轮询，成功后通过普通shared读取输出。按以下三类边界核对：

- 完整tile：坐标(0,0)，所有逻辑元素与CPU二维索引逐项相同。
- 部分OOB：例如(112,144)，仅与128×160范围相交部分有效，其余每个2B元素为零。
- 全OOB：例如(256,0)，tile全零且没有global有效读取；屏障仍能完成，不能等不存在的返回。

通过公开的16B块排列独立检查shared物理位置，不用模拟器swizzle函数生成预期。对对齐图案，64B模式的128B行内块序列依次包含 `0,1,2,3,4,5,6,7` 和 `1,0,3,2,5,4,7,6`；128B模式扩展到8种行相位。测试参考按官方表枚举块，再保留块内16B顺序，而实现采用位运算，两者独立。

对冻结专家实际计算的shared起点采样，保存原指令、基址和手工块偏移表，并从真实shared地址读回，不能仅解码TMA记录中的逻辑元素。静态审计唯一含TMA的冻结PTX后，全部20条指令的起点都对齐到1KiB：64B输入偏移为196608/200704/204800/208896；128B输入为`32768*stage + 8192*rank`（第二块再加16384），或`98304 + 32768*stage`（第二块再加16384）；输出偏移为196608/200704/204800/208896。当前shared符号基址为0，因此实际起点都对应phase 0，没有可采的真实非零起点。若冻结专家中存在非零swizzle phase，至少覆盖一个；若实际起点均为phase 0，应保存这些地址和phase 0，并用人工非零起点单独覆盖swizzle相位，不得把人工结果描述成真实专家采样。

用阶段4的正式expect/arrive/try_wait替换组件注入。每个目标总完成量精确等于表中的tile字节，有效和OOB分别计数；global sector可能更大，不得把sector字节全扣给屏障。检查最后数据/零填交付先于phase翻转和成功wait。合法安排先TMA后expect，再观察同一计数通路，不要求真实返回必然早于expect；先完成的确定次序已由阶段4组件测试覆盖。

<a id="early-integration"></a>

## 4. 首次交叉测试：输入搬运接一次WGMMA

入口条件仅为：非组播TMA和实际mbarrier完成通过；阶段6普通shared准备的单次WGMMA通过；阶段1必需基础指令可用。不等待组播、TMA输出、ldmatrix/stmatrix或setmaxnreg。

沿合法cluster启动，选一个CTA的128线程warpgroup执行矩阵，另一个CTA按所需cluster同步正常推进。输入用上表既有box，给A/B保留足够填充，不能为缩小测试新造一个未实现的TMA box。

逻辑A为64×16、B为16×128，选择：`A[m,k]=(m%7)-3+(k%3)`、`B[k,n]=(n%11)-5-(k%2)`，先转换成准确BF16。CPU用数学坐标直接计算8192项，不调用模拟器shared/寄存器映射函数。

```text
编码/发布A、B描述符 → TMA输入 → mbarrier成功等待
  → wgmma.fence → 单次WGMMA(scale_d=0) → commit → wait0
  → 128线程各用普通store写出自己的64项
```

A/B的K方向按阶段6描述符布局放入TMA box；未用填充不参与乘法。等待TMA输入的acquire提供相应完成保证；普通shared写输入时才需要对应proxy发布，不把三种fence当可互换指令。

比较所有128×64项与CPU参考，检查输入交付→屏障通过→矩阵读取/发起→写回完成→普通输出的事件。再换一组不同输入合法重复，排除只沿用首轮tile/描述符。

本测试报告在阶段5实际验收 README 统一索引，阶段6引用同一报告。记录两条开发线实际提交。通过仅表示首次接通，后续连续组、输出和配额仍分别验收。

<a id="multicast"></a>

## 5. 组播：mask与每目标通知

用两个cluster，组内同offset存不同初值。分别运行mask1、2、3：只给被选目标登记相应expect和等待，未选目标用最终本地观察确认未被该操作改变，不能让未选者等待不存在的字节。

| mask | 目标 | 预期 |
| --- | --- | --- |
| 1 | rank0 | 仅rank0数据/屏障更新 |
| 2 | rank1 | 仅rank1更新，发起rank0不能被默认通知 |
| 3 | rank0和rank1 | 两份shared数据相同，各自收到一次完整tile完成量 |

global有效源读取量在相同tile下不因mask3翻倍；shared交付总量为目标数×tile字节，远端交付有时间记录。另一cluster的同offset不改变。增加远端服务延迟时，相应目标通知应延后；不能所有目标在源返回时无条件同时完成。

<a id="output"></a>

## 6. 输出：读完源后复用，最终写回仍完整

按实际32×64/BF16/64B swizzle准备shared源A，普通写后按程序要求同步并执行async proxy fence。发TMA输出→bulk commit→read wait0；wait满足后把同一shared改为B，global输出A仍必须保持A。随后发B到另一global区域并最终等待kernel完成，比较两份输出。

输出参考按逻辑坐标生成；shared的写入地址按独立块表准备，不通过被测TMA逆函数准备。再取一个合法部分边界tile，global域外不写；预先放置的其它有效矩阵元素保持原值，不做非法目标地址恢复测试。

两组read wait1/0测试：向两个不同shared槽发起两组，在组件事件中故意让新组的read晚完成，旧组read已完成时wait1可返回，wait0仍等待新组。真实PTX再验证同一接线；不在wait1后消费或覆盖新组尚未读完的源。

记录 `source_read_done` 与每个global写ack。微测仅延长写确认相关路径时，read wait仍应在源读完后释放；最终kernel完成不得早于所有必要写ack。确认返回携带的是A快照，不在ack时重新读取已变成B的shared。

## 回归与完成边界

共用TMA/屏障/ack修改运行对应SASS最小回归，普通shared服务修改回归受影响load/store；缺少可运行trace就写明未覆盖。不要重复阶段2全部普通copy测试。

功能、OOB/组播字节、发布顺序、read/ack双完成点及首次联测通过后进入完整流水/专家集成。其它rank、dtype、swizzle组合、任意mask和原生编码互换暂缓；数值错、提前通知、丢最终写回不可暂缓。

<a id="results"></a>

## 实际结果

实际状态与结果表已迁至 [阶段 5 实际验收记录](../execute/validation/stage-5/README.md)；改动说明见 [实际实现记录](../execute/implementation/stage-5/README.md)。这里保留验收要求，不重复填写结果。
