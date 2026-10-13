# 阶段 4：mbarrier 与同步

## 当前审计结论（2026-10-13）

状态: 进行中

前置：阶段 3 新版验收通过。已有 CTA 功能状态测试，阶段 5 也补了受限 detailed
完成路径；旧“完全没有 timing”不代表当前状态。当前 `mbarrier_impl` detailed
仍依赖 `get_ptx_tma_cta` 的预注册 binding/固定 barrier offset，通用同步证据不足。

阶段 3 前置已由同日 [冻结 gate](../../../research/hopper-ptx/stage3/evidence/acceptance-20261013/frozen/gate.json)
满足；配对内层 `eca83b30`，外层为阶段 3 报告所在提交。本次会话未实施本阶段，
下一会话从以下组件合同开始，不能把既有 stage4/5 PASS 当作本阶段重新验收。

### 本阶段如何补齐

1. 固定 CTA 级 barrier 支持清单与测试，按 kernel/CTA/地址隔离状态，解除单个
   fixture barrier 的绑定假设。修改 ptx_sim、instructions.cc 和 SYNCS writeback。
2. 统一 issue/writeback/transaction completion 的状态推进，避免功能与详细路径
   重复记账。补多 barrier、多 phase、多缓冲重用、延迟 producer、乱序完成测试。
3. 验证 expect/complete、wait/token/parity、drop/invalidate、非法 transaction、
   状态销毁与跨 launch 隔离；复用现有 TMA 组件检查数据可见后才释放等待。

### 本阶段可验收条件

- [ ] CTA 支持清单的功能与 detailed 顺序一致，arrival/transaction 未齐不释放。
- [ ] 多 barrier/phase 压力和非法状态用例通过，完成恰好一次，不跨 kernel/CTA 串扰。
- [ ] 阶段 4 功能及现有阶段 5 TMA 回归通过，有事件、计数、输出和配对版本。

cluster-addressed 同步由阶段 7 承接，完整 Dense pipeline 由阶段 12 集成。
它们仍未完成，不作为阶段 4 的后向依赖；本阶段不能宣称全 cluster/Dense 同步通过。

当前执行顺序见 [单卡 Dense 计划](../../plans/single-card-dense.md)。

<details>
<summary>历史实现、测试结果和旧计划（不作为当前验收状态或执行顺序）</summary>

## 前置条件与配对提交

- 外层分支：`research/hopper-ptx`；本报告所在外层阶段提交记录本报告。
- 内层分支：`research/hopper-ptx`；配对实现提交：`3bdfa94f236aee5f14b1b4f418e3da92010c27d6`。
- 阶段 3 旧范围通过：外层 `0c929bbea1ad292b15170c2bd2b61058eeae029f`，内层 `16f592795466f13d82463daecbbc8ae2ba89828a`。
- 外层 `experiments/` 与内层 `.serena/` 为用户已有未跟踪内容，未纳入本阶段。

## 目标

把 PTX 的 barrier 初始化、到达、transaction 和等待语义接入现有 mbarrier 状态机。

## 操作

- `ptx.l`、`ptx.y`、`opcodes.def` 增加 `mbarrier.*` 入口与 `MBARRIER_OP`。
- `ptx_parser.cc` 严格识别 init、inval、arrive、arrive.expect_tx、arrive_drop、expect_tx、complete_tx、test_wait、try_wait、pending_count，要求 shared `.b64` 并校验操作数；同步元数据映射到已有 `SYNCS_*` 枚举。
- `mbarrier_t` 增加 pending/expected/transaction 不变量、phase、token/parity `test_wait`。
- `ptx_cta_info` 为每个 CTA 维护 barrier 地址表，实现初始化、到达、drop、transaction、pending count、wait 和 invalidate；phase 只在 pending 与 transaction 同时归零时推进。
- `mbarrier_impl` 把 PTX 动态操作数转换为 CTA 状态更新，并写回 arrive token、wait predicate 和 pending count。

数据流：PTX 文本 -> lexer/parser -> `ptx_mbarrier_metadata` 与 `SYNCS_*` -> `mbarrier_impl` -> CTA `mbarrier_t` -> token/predicate/global store。

## 命令与结果

```text
cmake --build /tmp/hopper-stage3-build --target cudart -j2
[100%] Built target cudart

./research/hopper-ptx/stage4/run.sh /tmp/hopper-stage4-run
SYNC_RESULT status=PASS n=32 first=10 last=10
FRONTEND_RESULT status=PASS mbarrier_instructions=8
STATE_RESULT status=PASS phase=2 pending=0 tx=0 expected=0
NEGATIVE_RESULT status=PASS file=mbarrier_bad_modifier.ptx
NEGATIVE_RESULT status=PASS file=mbarrier_bad_operands.ptx
NEGATIVE_RESULT status=PASS file=mbarrier_cluster_wait.ptx
```

`mbarrier_kernel.ptx` 使用 32-thread CTA：transaction 未完成时第一次 `test_wait` 失败，`complete_tx` 后 parity wait 成功，所有线程写回 10。`state_probe` 对 `expect -> arrive -> wait -> complete -> drop/arrive` 的 phase/count/transaction 做断言。`mbarrier_frontend.ptx` 覆盖 8 种 mbarrier 指令；三个非法 fixture 均按预期拒绝，诊断日志位于 `/tmp/hopper-stage4-run/mbarrier_*.ptx.log`。

## 验收

- 单 CTA phase 转换确定：通过 `SYNC_RESULT` 与 `STATE_RESULT`。
- transaction 完成前不能释放等待者：第一次 wait predicate 失败，完成 transaction 后 parity wait 成功。
- try-wait 成功和失败路径均有断言。
- PTX 序列与现有 SASS `mbarrier_t` 的 pending/expected/tx/phase 规则一致。

外层阶段提交已生成，内层配对提交为 `3bdfa94f236aee5f14b1b4f418e3da92010c27d6`。

## 限制

- 本阶段提供 CTA 功能语义和同步元数据；`timing_supported` 仍为 false，PTX 动态地址尚未自动灌入详细时序的 LDST cluster lookup。
- cluster wait 明确拒绝；跨 CTA DSMEM、TMA 产生的 barrier 事件留给后续阶段。
- 没有真实 H100/H200 测量，不作周期误差或硬件准确度结论。

</details>
