# 阶段 3：PTX 前端基础

状态: 未开始

## 目标

为后续 Hopper 指令提供可靠的 lexer、parser、类型、寄存器 tuple 和 `warp_inst_t`
入口；区分旧 WMMA、普通 MMA 和 WGMMA。

## 操作

- 扩展 opcode/token 和 modifier grammar。
- 增加 workload 所需 BF16、TF32、FP8 等类型。
- 扩展 vector operand 长度和寄存器 tuple 校验。
- 增加 `ldmatrix` 和必要的 shared layout 语义。
- 验证 `sm_90a` 在 loader/ptxas 路径中完整保留。

## 代码逻辑

```text
PTX token -> ptx_instruction -> operand_info/type metadata
           -> functional implementation
           -> warp_inst_t arch-op and Hopper metadata
```

## 验收

- 每条新增 opcode 有正例、错误 modifier 和错误 operand 测试。
- reference 数值结果通过。
- 尚未有 timing 实现的指令显式标记为“功能通过、时序未通过”。
- 未支持指令严格失败，不静默映射成普通 ALU 或旧 WMMA。
