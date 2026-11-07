# 阶段 2：普通 cp.async 与 gating GEMM

状态：仅方案；本轮无实现、修复或运行验收。[对应验收](../validation/stage-2.md)。依赖阶段 0/1 的普通访存和计算。

## 本阶段解决什么

gating GEMM 用矩阵乘法计算每个 token 的专家分数。它采用普通 global→shared 异步拷贝和 FP32 FMA，不需要 TMA/WGMMA。当前有 SASS LDGSTS 的请求/返回跟踪，但尚缺 PTX 的真实拷贝语义；只接一个延迟或直接 memcpy 都不足以同时得到正确流水与时间。

## 指令范围

| 指令 | 拟实现 | 当前状态 |
| --- | --- | --- |
| `cp.async.cg.shared.global` | 当前 16B 目的、实际源字节数、谓词和尾部零填，生成访存请求 | 待新增 |
| `cp.async.commit_group` | 提交本线程尚未提交的拷贝，不等待 | 待新增 |
| `cp.async.wait_group` | 当前 6/0；只等待所属线程的旧组 | 待新增 |
| 普通 shared 读写、`bar.sync`、FMA、BF16 转换 | 让 gating 的搬运与计算正确协作 | 现有/阶段 1 路径待组合回归 |

实际常量以冻结 PTX 为准，不能只照原指令说明中的 wait0。源大小、谓词及多级流水来自当前 gating，不是可删掉的边界测试。

## 如何实现

1. **解析和求值。** 在 opcode/语法中区分普通 cp.async 与 TMA；保存目的 shared、源 global、拷贝长度和可选实际源长度。执行时只为 predicate 为真的 lane 求请求，捕获目标 CTA/shared 身份及组归属。
2. **复用真实访存通路。** 将有效源字节交给已有 LDGSTS 合并、L2/DRAM 请求和返回处理；`.cg` 沿全局缓存策略。给待完成拷贝附上 PTX 所需目的偏移和有效字节信息。只读取有效源，不为零填部分访问 global。
3. **数据交付与完成绑定。** 返回时把该请求对应数据写到正确 shared，再减少未完成计数；同一拷贝的零填部分也必须交付后才完成。发起时可准备必要数据状态，但不能立即把共享目的变为可消费。源长度 0 不应等待不存在的 DRAM 请求。
4. **每线程提交与等待。** 保留未提交组和已提交组顺序；实现可用 warp 内数组紧凑存储，但 lane 的操作和组边界不得混合。commit 包含当前空组的合法语义；wait N 允许最近 N 个已提交组未完成，必须等更早组完成，不等未提交的未来操作。用动态组序号区分循环中相同 PC 的操作。
5. **只暂停需要等待的执行。** wait 阻止相应线程越过等待点；其它可调度 warp、请求返回和发起后的独立算术仍可推进。PTX 的 wait 根据自身 opcode 选普通拷贝组，不按“上一种 commit 指令”猜类型。

## SASS 复用边界及容易混淆之处

| 现有代码 | 可复用 | PTX 必须补的部分 |
| --- | --- | --- |
| LDGSTS 请求合并、访存队列、返回 | 内存延迟、带宽和有效访问 | shared 目标、真实数据、部分源/零填 |
| LDGDEPBAR/DEPBAR 的组与完成跟踪 | 状态保存位置和完成回调位置 | 按本线程实际参与归组，明确 wait6/0，不能把 trace 状态直接当 PTX 输入 |
| 普通 shared/FMA 流水 | bank、端口、依赖、计算延迟 | 保证数据完成后才进入合法消费 |

后续实现若在发起时就放行，可能读取上一块数据且低估时间；若每次拷贝后都等完，则抹掉预取重叠。这是本阶段需排除的实现错误，尚未在当前 PTX workload 复现。

## 交付与暂缓

先交付单组和多组最小程序，再运行模块 2（Mixtral/128 gating）和 71（Qwen64/128 gating）。CPU 独立计算预期。人工输入用于数值；性能比较使用冻结尺寸与可比缓存前序，检查 kernel 时间和 DRAM/L2，不拿缩小微测与真实时间硬比。

记录实际模拟墙钟/RSS，区分加载和计算。只为当前 gating 补普通拷贝；`.ca`、其它长度/变体、未使用的 wait_all 和不合法重叠写入不扩入本阶段。若共享 LDGSTS 完成代码有改动，做对应 SASS 最小回归，不扩大成所有访存模型回归。

## 源码入口与实际交付记录

[shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) 的 `issue_warp`、LDGSTS 组和 LSU 返回；[abstract_hardware_model.cc](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc) 的访问生成；[trace_driven.cc](../../gpu-simulator/trace-driven/trace_driven.cc) 的 LDGSTS 分类；[instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) 和 PTX parser 新入口。

待填写：三条指令实际实现/修复情况、gating 数值与时序结果、SASS 回归、提交与报告。当前：无。
