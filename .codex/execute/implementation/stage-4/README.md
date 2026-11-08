# 阶段 4 实际实现：mbarrier

状态：**已完成（本阶段实现与自测；hint 采用方案允许的单次轮询）**。PTX 可以初始化本地屏障、登记字节、逐线程到达、向另一 CTA 通知，并取得每次轮询的真假结果。同一屏障只有到达数与字节数同时归零才进入下一轮。独立验收已通过，范围与证据见[验收记录](../../validation/stage-4/README.md)；下文保留实现时的自测记录。

[实现方案](../../../plans/stage-4.md) · [验收要求](../../../validation/stage-4.md) · [独立验收记录](../../validation/stage-4/README.md)

## 当前范围与前置

本轮只交付阶段 4。阶段 3 已实现并独立验收，提供两个 CTA 的身份、shared 目标解析、成组驻留和通知队列。开始时外层/内层工作区均干净，基准分别为 `89d06f0` / `a4c35975`；下文版本与差异记录实现和验收时的未提交状态，归档提交见文末。

```text
线程寄存器中的地址、计数、parity
  → 本次动态指令 → LSU 服务
       本地：更新计数 / 查询结果
       远端：捕获目标对象与轮次 → 通知队列 → 交付时扣数
  → 寄存器写回完成 → 调用 warp 可继续
```

现在可独立检查零字节 PTX 屏障和非零字节状态接口，满足进入阶段 5 的实现前置。独立验收另补64B登记后返回未完成的真实PTX检查；**尚无真实 TMA 搬运及其非零字节完成证据**。冻结专家还需要阶段 5/6，不在本轮启动或声称已经支持。

## 实际改动与计划对应

源码链接均指向独立内层仓库。

| 对应 plan | 文件与入口 | 为什么修改、实际行为 |
| --- | --- | --- |
| [一份状态、两个计数](../../../plans/stage-4.md#state) | [abstract_hardware_model.h](../../../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h)：`mbarrier_t::expect_on`、`arrive_expect_on`、`waiting_parity` | 单独 expect 配平字节时也检查轮次转换；组合操作先一起更新到达数/字节再检查，避免中途翻转；保留有符号 tx。复用原对象、pending/expected/tx/phase 和 kernel 所有的屏障表。 |
| [求值与依赖](../../../plans/stage-4.md#execution) | [ptx.l](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l)、[opcodes.def](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/opcodes.def)；[cuda-sim.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc)：`set_opcode_and_latency`、`pre_decode`、`ptx_exec_inst`；[cuda-sim.h](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.h) | 识别下述七种 mnemonic，分类为共用 SYNCS/LSU 指令；登记方括号内地址寄存器以及 count、bytes、parity、hint 的输入依赖、predicate 的输出依赖。`_` 沿既有非架构寄存器处理，保留到达副作用。 |
| 同上 | [abstract_hardware_model.cc](../../../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc)：`core_t::execute_warp_inst_t`；[instructions.cc](../../../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc)：`mbarrier_impl`、`mbarrier_complete` | 在逐线程 predicate 已求出后读取活动 lane 的操作数，只填本次动态 `syncs_operand`；LSU 服务时才更新一次屏障。普通 offset 绑定当前 CTA rank，避免 rank1 的本地地址指向 rank0。try_wait 在服务时计算结果，使用既有 predicate 编码（低位0为真）。 |
| [轮询与写回](../../../plans/stage-4.md#polling) | [shader.cc](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)、[shader.h](../../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h)：`issue_warp`、`scheduler_unit::cycle`、`ldst_unit::writeback/writeback_complete`、`shd_warp_t::waiting/reset/init` | PTX 绕过发射前仅适用于 trace 的无限重试。调用 warp 从发起到正常 collector/scoreboard 写回完成保持 pending，其它 warp/CTA 和 LSU 返回继续推进。predicate 使用者不能越过本次写回；连续 kernel 会清理 pending。 |
| [初始化与远端顺序](../../../plans/stage-4.md#ordering) | 同上调度与服务；`kernel_info_t::cluster_mbarrier_arrive`、`service_cluster_arrivals` | arrive release 和初始化 fence 等待本 warp 既有寄存器依赖与 store 完成。init 经 LSU 完成后，本 warp 才能发 fence，fence 写回后才能进入程序原有 cluster arrive/wait；不新增全局同步。远端使用阶段3目标解析，捕获真实屏障指针和轮次，到通知实际交付才扣数。最后一个 CTA 退出时沿阶段3检查队列已排空。 |
| 共用状态适配 | `ldst_unit::mbarrier_arrive/arrive_drop`、`mbarrier_waiting` | SASS 组合到达也改用一次计数更新，避免新增 expect 翻转检查造成中途切轮；SASS 仍在入口提取 bit31，PTX 直接使用0/1。未改 SASS 调度重试策略、首次 try_wait 修正或 `init_as_one`；这些 trace 特例没有套入 PTX。 |

本轮新增的实际指令形式：

- `mbarrier.init.shared::cta.b64 [addr], count`
- `mbarrier.arrive.shared::cta.b64 _, [addr]`
- `mbarrier.arrive.shared::cluster.b64 _, [mapped]`
- `mbarrier.expect_tx.shared::cta.b64 [addr], bytes`
- `mbarrier.arrive.expect_tx.shared::cta.b64 _, [addr], bytes`
- `mbarrier.try_wait.parity.shared::cta.b64 pred, [addr], parity`，以及末尾带寄存器 hint 的形式
- `fence.mbarrier_init.release.cluster`

当前冻结消费者的 arrive 结果均为 `_`。未验收 token 的后续消费、arrive_drop PTX、test_wait 或任意旧轮查询；不因代码能接收额外操作数而宣称这些形式已支持。

## 复用、时序与方案差别

复用 SASS/common 的 `mbarrier_t`、kernel 屏障表、SYNCS 的 LSU 服务管线、collector 写回与 scoreboard，以及阶段3的 shared 身份和通知 collector。PTX 自己生成地址、计数和 predicate，不能使用 trace 预先记录的操作数代替。PTX 在同一 LSU writeback 服务点分派到自己的求值结果，共用状态只更新一次。

远端通知与 cluster barrier 共用每 cluster 的队列：最早在源 LSU 服务周期加 `smem_latency` 后交付，collector 每周期至多接收一条通知；实际是否完成仍由两个计数决定。H100 默认29周期只近似端点运输，尚未用真实专家校准。129周期副本同时改变普通 shared/SYNCS 管线延迟，不能把总周期差全部归为远端运输。

按方案允许的首版选择，**无 hint 和有 hint 均在 LSU 服务时观察一次，未完成就返回假**；没有实现按100ns暂停并提前唤醒的有界等待。hint 寄存器仍参与输入依赖。100ns在当前1980MHz下对应198周期，但本实现没有强制睡满198周期。自测中每 CTA 三轮查询45/37/37次；增加生产者计算后为89/82/82次。真实专家尚不可运行，因此没有专家轮询开销结论；阶段5/6首次联测应记录该项，若忙轮询明显影响主要耗时，再补有界暂停。此选择沿[原方案](../../../plans/stage-4.md#polling)，没有放宽数值或屏障完成条件。

初始化 fence 的作用由原 init 的实际完成、调用 warp 的顺序和 fence 的 LSU 服务建立。它只发布屏障初始化，不证明普通 shared 数据已发布到 async proxy，也不替代 TMA 完成或程序原有的 cluster barrier。

## 构建与自测

CUDA12.8/ptxas12.8.61、GCC13.4.0，debug 构建 `source setup_environment debug; make -j8` 通过；新增 C++ 按 clang-format16.0.6 格式化。使用生产 H100 配置副本，仅将132个SM改为2，保留两 chiplet partition、1980MHz及资源/缓存参数；通知敏感性副本另改 shared latency29→129。各 run.json 保存配置/程序/PTX/库散列和 ldd，确认加载模拟器 debug libcudart，使用 `PTX_SIM_MODE_FUNC=0` / performance simulation。

[summary.json](../../../../workloads/moe/runs/hopper-stage-4/summary.json)由[summarize.py](../../../../workloads/moe/tests/hopper/stage-4/summarize.py)对保存的数值、事件、时序证据执行 **3840条断言**。这是实现自测，独立验收表保持未运行。

| 自测 | 实际结果与证据 |
| --- | --- |
| `local` / `hint` | 两个普通 CTA，各64线程、offset1024、count2；由两个 warp 的leader到达。真实查询同时得到假/真，同一对象三轮 phase准确为1/2/3，输出11/22/33。每项连续两次launch，均正确。[run.json](../../../../workloads/moe/runs/hopper-stage-4/final-debug/run.json)、[local事件](../../../../workloads/moe/runs/hopper-stage-4/events-local/events.log)、[hint事件](../../../../workloads/moe/runs/hopper-stage-4/events-hint/events.log)。每种事件运行有4次init、24次到达、476次查询，其中12次真；每次query的collector写回均晚于查询服务。 |
| `lanes` | 64个线程分别执行arrive，count64，仅翻转一次；不是每个warp减1或32再叠加功能更新。紧随try_wait的分支检查predicate，结果99正确。 |
| `zero_tx` | 实际 PTX 执行expect0和arrive.expect_tx0，随后predicate/分支输出99，覆盖两个入口的解析、依赖及服务。非零字节见下一项。 |
| 非零字节组件 | [state_component.cc](../../../../workloads/moe/tests/hopper/stage-4/state_component.cc)对正式状态API使用验收表的固定预期：到达先齐、字节先齐、完成先于expect导致tx=-64、组合登记及三轮；所有中间pending/tx/phase精确正确。[日志](../../../../workloads/moe/runs/hopper-stage-4/final-debug/state_component.log)。完成是人工调用`complete_on`，不是DRAM/TMA。另查SASS bit31适配关系和本地/同offset peer的独立完成；未运行SASS trace或完整LSU组播适配器。 |
| `remote` | grid2×2、cluster1×2、每CTA64线程，所有对象先合法初始化并fence/cluster同步。rank0通过mapa向本组rank1到达，rank1轮询成功，rank0本地同offset仍为假。两组不同计算延迟，连续两次launch正确。[事件](../../../../workloads/moe/runs/hopper-stage-4/events-remote/events.log)：8次init、4次通知/交付/翻转、282次query。成功查询均晚于所指通知交付，另一cluster和rank0不被扣数。 |
| 主要时序变化 | 生产者依赖计算100→200次，local每次kernel从10408/10407变为15232/15231周期，remote从8752变为11956周期。[delay200](../../../../workloads/moe/runs/hopper-stage-4/delay200/run.json)。通知端点29→129，4次实际交付均满足更长延迟，首次launch两组翻转由6820/8421变为7320/8921；成功wait仍不能先于交付。[129事件](../../../../workloads/moe/runs/hopper-stage-4/events-transport129/events.log)。 |
| 受影响回归 | 阶段3 `cluster_sync`（两组、三轮、两次launch）和`ordinary`通过：[run.json](../../../../workloads/moe/runs/hopper-stage-4/regression-stage3/run.json)。阶段2 `copy_mix`和`groups2`通过，覆盖逐线程predicate、原异步返回及等待：[run.json](../../../../workloads/moe/runs/hopper-stage-4/regression-stage2/run.json)。未扩大到无关数值指令或长gating。 |

### 本机成本与设备周期

微测不能外推九例，也不用于宣称硬件性能误差约20%。下表为debug正常运行，设备周期含既有5000周期launch费用；注册是main至注册结束，kernel墙钟是提交至同步返回（含既有host轮询），比较为CPU检查循环。详细 `/usr/bin/time -v` 保存在各 `.time`。

| 程序 | 两次kernel设备周期 | 注册墙钟 | 两次kernel墙钟 | 比较循环 | 进程总墙钟 / RSS |
| --- | --- | --- | --- | --- | --- |
| local | 10408 / 10407 | 0.664s | 3.218 / 3.208s | 0.168 / 0.090ms | 7.12s / 273.0MiB |
| hint | 10408 / 10407 | 0.657s | 3.218 / 3.308s | 0.186 / 0.100ms | 7.21s / 273.3MiB |
| 64线程到达 | 5471 / 5471 | 0.678s | 1.007 / 1.005s | 0.131 / 0.087ms | 2.73s / 273.0MiB |
| 组合零字节 | 5505 / 5505 | 0.681s | 1.006 / 1.005s | 0.162 / 0.095ms | 2.73s / 272.9MiB |
| 远端 | 8752 / 8752 | 0.688s | 2.510 / 2.507s | 0.153 / 0.109ms | 5.74s / 273.3MiB |

## 剩余限制与阶段5交接

- 非零字节尚未接真实TMA。阶段5在发起时捕获本kernel/cluster/目标rank/offset对应的对象与phase，数据实际写入目标shared之后，用实际逻辑字节调用共用`complete_on`，逐次验证没有跨轮或重复通知。不能使用最近访问的屏障、在错误轮次自动回退，或用本阶段组件结果代替搬运运行证据。
- hint首版是上述单次轮询；首次真实专家/提前联测必须记录轮询量和等待周期，不能默认其性能影响很小。普通mbarrier服务也暂按每warp完成后再继续的顺序模型运行，尚未据专家校准吞吐。
- 初始化发布已通过真实远端使用检查；普通shared→async的proxy fence不属于本阶段。TMA在途生命周期与数据可见性由阶段5补齐。
- 没有可运行Hopper SASS trace；只交付共用状态/表/编码组件与相关PTX回归，未覆盖SASS运行和完整组播LSU服务适配器。此边界不能写成SASS回归通过。
- 不支持任意cluster形状、非法重初始化、任意久远phase、错误通知恢复；本轮没有扩展这些范围。

开发期第一次测试未加原微测惯用的PTX首行注释，cuobjdump提取后丢了version，加载失败；已在测试生成器补齐头部并全量重跑，未改加载器。GDB首次在sandbox中被ptrace限制，后获准只读运行；观察器一次把`core_t*`当作派生指针读SM字段而中止，修正观察器后四份最终日志均正常退出，无Python异常。开发记录留在[development](../../../../workloads/moe/runs/hopper-stage-4/development/)，不作通过证据。

## 版本、准确命令与复验入口

外层基准 `89d06f0bf34ef01ecaf0ddb67acc85f593cd1c48`；内层基准 `a4c359759c097f4362200e3bbadf3ff63a373a3b` 加9个源码文件的未提交差异。内层diff SHA256：`eb82541423298bb51d9413dec7e8da4ad738cdcd592514181107a27405885a1c`。最终正常运行与GDB使用同一debug库、同一差异；[versions.json](../../../../workloads/moe/runs/hopper-stage-4/metadata/versions.json)记录代码/测试散列，[inner.diff](../../../../workloads/moe/runs/hopper-stage-4/metadata/inner.diff)保存差异，[构建日志](../../../../workloads/moe/runs/hopper-stage-4/development/build-final-debug.log)保存构建结果。外层仅新增六份stage-4测试和更新本文，未改独立验收、冻结程序、输入、容差或生产配置。

从项目根目录复跑，使用新目录，避免覆盖现有证据：

```bash
source workloads/moe/runs/hopper-stage-0/independent-20261108/environment.sh
make -C gpu-simulator/gpgpu-sim -j8
python3 workloads/moe/tests/hopper/stage-4/run.py \
  --run-dir workloads/moe/runs/hopper-stage-4/recheck/final-debug
python3 workloads/moe/tests/hopper/stage-4/run.py --test local,remote --delay 200 \
  --run-dir workloads/moe/runs/hopper-stage-4/recheck/delay200
python3 workloads/moe/tests/hopper/stage-3/run.py --test cluster_sync,ordinary \
  --run-dir workloads/moe/runs/hopper-stage-4/recheck/regression-stage3
python3 workloads/moe/tests/hopper/stage-2/run.py --test copy_mix,groups2 \
  --run-dir workloads/moe/runs/hopper-stage-4/recheck/regression-stage2
```

事件采集的准确cwd、参数、配置散列和日志位置见[event-commands.json](../../../../workloads/moe/runs/hopper-stage-4/event-commands.json)。在新证据根目录建立相同四个events目录，从final-debug复制micro/config/icnt；只有events-transport129将shared latency29改129。source环境后须再次cd到对应运行目录（环境脚本会cd到项目根），再执行记录中的GDB命令。观察器需要允许跟踪本次子进程；只读状态，不注入到达、不修改内存。

保留上述子目录名后运行：

```bash
python3 workloads/moe/tests/hopper/stage-4/summarize.py \
  workloads/moe/runs/hopper-stage-4/recheck
```

验证 agent 建议先跑 `run.py` 的本地/组件，再检查远端和四份事件断言；单独确认hint轮询近似与SASS/TMA未覆盖边界。独立结论填写validation README；本轮停止于阶段4，不开始阶段5或6。

## 提交归档

内层实现提交 `6e60cf3b1b2957a082ae6cad03613e8d43ef15c6`，内容与独立验收通过的九文件diff完全一致。外层配套提交保存最终六份测试源码及两份实际记录；独立验收补强了重复启动的输出初始化、寄存器操作数和64B登记检查，详见[验收记录](../../validation/stage-4/README.md)。上文未提交版本、自测次数与散列保留为实现时的历史证据，最终测试版本以验收记录为准。

本次仅归档已通过的内容，未改功能或模型参数，未重复构建/模拟，未推送，也未开始阶段5。原始日志、二进制和运行产物留在本地 runs，不纳入提交。
