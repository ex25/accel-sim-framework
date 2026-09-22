# Testing 与 evidence policy

每个 core change 都必须有 focused test 与最小相关 regression。Documentation-only
change 也要运行 structural check（适用时检查 link/file existence）和
`git diff --check`。

## 先记录 baseline

改变 behaviour 前记录 baseline command、SHA、configuration、exit code、必要时的
wall time 和短 output summary。同一 command 在修改前就失败时标为
`BASELINE FAILURE`；区分 environment failure（缺 CUDA/NVBit/GPU/dependency）与
真实 branch regression。没有 test suite 不能静默当作 pass。

Simulation 必须记录 workload/trace、GPU model 或 synthetic topology、config、
random seed、main/nested SHA、output directory 与 evidence level。大型 output
放在 Git 外，只提交 metadata 和必要的小 fixture。

## Validation level

使用明确 label：`unit`、`replay smoke`、`analytical`、`synthetic network`、
`public-data comparison`、`real hardware`。BookSim packet-delivery test 只能
证明 transport correctness，不能证明 NVSwitch fidelity；A800 replay 只能证明
software-pipeline compatibility，不能证明 Hopper SASS 或 NVLS behaviour。

## Test report

Implementation note 与 commit body 必须写明 command、expected result、actual
result、failure classification 和 limitation。Timing change 要检查 bytes/flits、
latency/completion ordering 与 counters；trace change 要记录 trace version、
kernel count 与 parser/replay result。
