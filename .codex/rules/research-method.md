# Research method 与 evidence label

除非存在已记录 blocker，否则按以下顺序工作：

```text
understand → inspect history/source → reproduce baseline → identify gap
→ design alternatives → smallest implementation → focused test
→ regression → document → synthetic/analytical calibration
→ hardware validation
```

提出 architecture claim 前，先阅读 checkout source、Git history、build file、
README/release notes，并观察 executable behaviour。README、comment、issue summary
与 paper 都是需要对照 code 验证的 claim。外部 hardware/software fact 优先引用
NVIDIA official documentation、PTX/NVBit/CUDA/NCCL source/documentation、
BookSim source 与 primary paper，并记录 URL/version/date。

所有非平凡结论必须标注：

- `Verified`：已在当前 checkout 或指定 hardware 上复现。
- `Observed`：来自已有 artifact，但本阶段未独立复跑。
- `Inferred`：根据 source 推断，没有直接 runtime proof。
- `Hypothesis`：proposed design 或 expected behaviour。
- `Unknown`：已搜索但无法确认。
- `Needs hardware validation`：只有目标 H100/NVSwitch system 才能确认。

必须区分 software trace-pipeline validation、synthetic network/model sanity、
public-data trend comparison 与 real H100/NVSwitch correlation。Command、SHA、
config、output summary、assumption 与 limitation 必须写入 repository；chat 不是
evidence storage。
