# 阶段 7：Cluster、DSMEM 与寄存器配置

状态: 已验收

本阶段完成了文本 PTX execution-driven 路径的 cluster 元数据、CTA rank、功能级远程 shared/DSMEM、cluster barrier/fence，以及资源指令的严格失败语义。真实 runtime launch API 与 detailed DSMEM/cluster barrier 时序仍属于后续 runtime/model 阶段。

## 目标和前置条件

- 外层和内层均位于配对的 `research/hopper-ptx` 分支；阶段 6 已验收。
- 用户已有的外层 `experiments/` 和内层 `.serena/` 未加入本阶段提交。
- 参考 [NVIDIA PTX ISA cluster introduction](https://docs.nvidia.com/cuda/parallel-thread-execution/) 定义 cluster dimensions、`%clusterid`/`%nclusterid`、`%cluster_ctaid`/`%cluster_nctaid`、`%cluster_ctarank`/`%cluster_nctarank`、`%is_explicit_cluster`，并使用 `mapa` 与 `ld/st.shared::cluster` 访问远程 shared。

## 实现

内层 `src/cuda-sim/` 和 `src/abstract_hardware_model.*` 完成以下数据流：

- `gpgpu_t::register_ptx_cluster_launch` 保存显式 launch dimensions；未注册 kernel 使用隐式 `1x1x1`。线程初始化计算 cluster id、rank、rank 数量和 `is_explicit_cluster`，并拒绝不能整除 grid 的 launch。
- `ptx_thread_info::get_builtin` 增加全部 cluster special registers。三维 CTA 坐标展开为唯一 rank。
- `gpgpu_t` 按 `(kernel, cluster id, rank)` 保存 shared memory。`mapa.shared::cluster` 返回带 rank 标签的地址，`decode_space` 将地址重定向到目标 CTA shared。地址、rank、对齐、容量及普通 `ld.shared`/`st.shared` 误用严格失败。
- functional driver 在 cluster launch 时同时初始化所有 CTA，按 warp step 交替推进；`barrier.cluster.arrive.*` 记录 arrivals，`wait.*` 只有在完整 generation 后才放行。统计 `mapa`、远程读写、arrival、wait、proxy fence 和 wait checks。
- `fence.proxy.async.shared::cluster` 复用现有 memory barrier 路径并记录事件。`setmaxnreg.inc/dec.sync.aligned.u32` 完成前端检查，但寄存器池重分配、occupancy 和 warpgroup 资源更新尚未存在，执行时严格报告 `setmaxnreg register-pool redistribution/occupancy is not implemented`。
- detailed path 对远程 DSMEM、cluster barrier 和 cluster proxy fence 明确失败；前两者分别报告 `PTX DSMEM remote timing is not implemented` / `PTX cluster barrier timing is not implemented`，不伪造周期语义。

## Harness 和命令

阶段目录为 `research/hopper-ptx/stage7/`：`generate.py` 生成 fixture，`cluster_harness.cc` 验证 rank、cluster/grid 坐标、远程 shared 写读、`mapa`、`getctarank` 的 u32/u64 地址形式、barrier/fence，`run.sh` 运行正向矩阵和负向诊断。

```bash
cmake --build /tmp/hopper-stage3-build --target cudart -j2
bash research/hopper-ptx/stage7/run.sh /tmp/hopper-stage7-acceptance7 > /tmp/hopper-stage7-acceptance7.log 2>&1
git diff --check
git -C gpu-simulator/gpgpu-sim diff --check
```

## 实际结果

完整运行目录为 `/tmp/hopper-stage7-acceptance7`，摘要为 `/tmp/hopper-stage7-acceptance7.log`。

| 场景 | 结果 |
|---|---|
| 隐式/显式 `1x1x1` | metadata PASS |
| 双 CTA `2x1x1` | rank/address/remote read-write PASS；`wait_checks=180` |
| 四 CTA `4x1x1` | rank/address PASS；`wait_checks=540` |
| 二维 `2x2x1`、三维 `2x2x2` | CTA rank/cluster id PASS；`wait_checks=540/1260` |
| 多 cluster `4x2x1`、cluster `2x2x1` | cluster 隔离和地址 PASS |
| detailed metadata | 1/2/8 CTA 与 2D/3D PASS；17484–17493 cycles，不宣称硬件准确度 |

延迟最后 CTA 的 functional barrier 在 arrival 不完整时产生 wait checks，随后才释放，验证没有提前完成。正向输出中 1/2/4/8 CTA 的 barrier 统计分别为 96/192/384/768。

15 个正向场景、25 个负向场景全部通过。负向 fixture 全部以非零状态结束且匹配诊断：非法 modifier、寄存器宽度、operand 数量、谓词 barrier、非法 rank/address/alignment/capacity、普通 shared 映射地址、缺失/重复 arrival、functional deadlock、detailed DSMEM/barrier/proxy-fence timing、非法 launch/grid 和 `setmaxnreg`。没有把 timeout 或 signal 当作负向通过。

阶段 2 回归使用当前 runtime 通过：`RESULT status=PASS n=128 checksum=161408 first=1007 last=1515`。

## 提交和证据

- 外层：本报告所在外层提交（`research/hopper-ptx`）；基线 `268ce5ef562f5a5920d95aa6be068621aacc735a`。
- 配对内层：`a7d9951175826c586f6d9f7d67380beac5468ec2`（`research/hopper-ptx`）；基线 `461bf037459153f101fdd1dcf8d1bbcf77989ca3`。
- 阶段证据：`/tmp/hopper-stage7-acceptance7/functional/*.log`、`/tmp/hopper-stage7-acceptance7/detailed/*.log`、`/tmp/hopper-stage7-acceptance7/ptx/` 和 `/tmp/hopper-stage7-acceptance7.log`；构建日志为 `/tmp/hopper-stage7-build-final.log`。

## 限制和下一阶段入口

- 本阶段只承诺 functional DSMEM/barrier 的确定性语义；detailed 远程 DSMEM、cluster barrier 和 cluster proxy fence 的 issue/response/occupancy 周期留给后续模型验证和 runtime integration。
- `setmaxnreg` 的 register-pool redistribution、occupancy 变化和 warpgroup resource accounting 尚未实现，使用它的 kernel 会严格失败。
- multicast target table 仍由阶段 5 TMA descriptor 语义管理，本阶段没有扩展 CUDA runtime cluster launch API。
