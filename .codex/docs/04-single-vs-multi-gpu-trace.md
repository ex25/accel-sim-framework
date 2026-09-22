# Single-GPU 与 multi-GPU trace

## 初步结论

`Verified source boundary / needs workload validation`：只有当研究问题明确
允许各 logical rank 使用相同 compute workload 时，single-GPU compute SASS
trace 才能作为 workload template 复用；它本身不是 multi-GPU execution record。

简单复制 trace 不会产生 rank/device identity、peer address、NCCL algorithm、
collective launch/completion、message partitioning、synchronization、topology、
routing、contention、switch reduction、multicast 或 compute/communication
overlap。除非已有 trace path 经验证能提供这些信息，否则它们必须由独立的
communication events 和 orchestrator 表达。当前 tracer 的 multi-context
能力以 `CUcontext` 为单位，包括 `ctx_kernelid`、context-specific files 和
context lifecycle callbacks；trace record 中有 CTA/cluster identity，但没有
明确的 NCCL rank、CUDA device ID、peer endpoint 或 fabric event。因此，
release notes 中的 “multi-GPU trace” 不能直接当作 multi-GPU simulation 证据。

已评估的方案包括：(A) 扩展 SASS trace；(B) PTX workload + communication
event trace；(C) high-level event orchestrator；(D) PTX-primary hybrid，并用
SASS validation。当前选择 D：PTX 用于 proposed ISA/microarchitecture 与新
GPU behaviour；显式 communication semantics 放在普通 instruction record
之外；SASS 保留给现有 hardware validation。
