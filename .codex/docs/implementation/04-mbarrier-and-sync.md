# 阶段 4：mbarrier 与同步

状态: 已验收

## 前置条件与配对提交

- 外层分支：`research/hopper-ptx`；本报告所在外层阶段提交记录本报告。
- 内层分支：`research/hopper-ptx`；配对实现提交：`3bdfa94f236a5e14b1b4f418e3da92010c27d6`。
- 阶段 3 已验收：外层 `0c929bbea1ad292b15170c2bd2b61058eeae029f`，内层 `16f592795466f13d82463daecbbc8ae2ba89828a`。
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

外层阶段提交已生成，内层配对提交为 `3bdfa94f236a5e14b1b4f418e3da92010c27d6`。

## 限制

- 本阶段提供 CTA 功能语义和同步元数据；`timing_supported` 仍为 false，PTX 动态地址尚未自动灌入详细时序的 LDST cluster lookup。
- cluster wait 明确拒绝；跨 CTA DSMEM、TMA 产生的 barrier 事件留给后续阶段。
- 没有真实 H100/H200 测量，不作周期误差或硬件准确度结论。
