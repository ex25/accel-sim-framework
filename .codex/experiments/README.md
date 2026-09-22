# 实验 metadata 索引

这里保存实验 specification 和 compact metadata。Raw trace、log、checkpoint
与大型 result table 必须放在 gitignored run directory 中。每个 metadata file
都要固定 Accel-Sim/BookSim SHA、configuration、workload、rank/GPU count、
collective/algorithm、message size、seed、command、output path、date 和
hardware/model evidence level。

已完成实验：

- `2026-09-19-a800-nvbit-booksim-nvlink2.md`：真实 A800 vectoradd trace、
  post-processing/replay 与 standalone BookSim two-endpoint smoke/sweep。
- `2026-09-20-sass-ptx-a800-gap.md`：single-A800 dual-code-object matrix，
  对比 SASS trace replay 与 PTX execution-driven replay，包含有界 GEMV/GEMM
  和 CUDA 12.8 lineinfo parser failure。
- `2026-09-20-sm90-ptx-baseline.md`：CUDA 12.8 `compute_90` PTX-only kernel
  被未修改的 `H100-PTX`/`SM90_H100` execution-driven path 接受。
- `2026-09-21-cais-add-ptx.md`：最小 single-GPU `cais.add.u32` 的 parser、
  functional、dependency、timing、completion 与 SM90 regression evidence。
