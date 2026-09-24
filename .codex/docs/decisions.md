# Design decisions

## D001 — 将 research memory 保存在 `.codex/`

- Date：2026-09-18
- Context：未来 session 必须在没有 chat history 时恢复项目。
- Decision：rules、reusable workflows、compact facts、full research 和
  experiment metadata 统一放在 `.codex/`；`AGENTS.md` 只做 thin loader。
- Why：提供单一 persistent source of truth，并支持 progressive disclosure。
- Validation：每个 milestone 执行 context-loss reload check。

## D002 — 通过 `.agents/skills` 暴露 canonical `.codex/skills`

- Date：2026-09-18
- Context：项目要求使用 `.codex/skills`，而 Codex discovery 会扫描
  `.agents/skills`。
- Decision：skill 只在 `.codex/skills` 编写，并在 `.agents/skills` 下建立
  symlink。
- Trade-off：Git 记录的是 link，部分工具可能不跟随；禁止复制 skill，避免
  两份内容漂移。
- Evidence：2026-09-18 查阅的 Codex skills documentation 说明 repository
  discovery 扫描 `.agents/skills`，且每个 skill 需要带 `name`/`description` 的
  `SKILL.md`。

## D003 — 优先复用已有 BookSim-derived `intersim2` boundary

- Date：2026-09-18
- Context：`booksim2` 是 `28f4329` 的独立 checkout，而 nested GPGPU-Sim 已
  有 `src/intersim2` 与 GPU adapter。
- Decision：不 vendor/copy reference checkout；先理解并复用现有
  `InterconnectInterface`/`GPUTrafficManager` boundary，再决定 logical
  GPU/NVSwitch layer 是否位于其上。
- Why：保留 Accel-Sim build/history，避免第二套 generic network implementation。
  现有 path 是 memory traffic，不是 NVLS。
- Validation：API source audit 已完成；provenance 与 standalone smoke-build
  comparison 仍开放。

## D004 — 将 compute trace 与 communication event 分离

- Date：2026-09-18
- Context：单条 SASS trace 没有 collective rank、topology 或 transport state。
- Decision：初期保留现有 compute trace format；future communication 使用独立
  event model 或 orchestrated layer，除非 source evidence 证明必须扩展现有格式。
- Validation：实现前必须完成 trace autopsy 与 multi-GPU capability audit。

## D005 — 先用 external BookSim 做 two-GPU transport prototype

- Date：2026-09-19
- Context：在修改 Accel-Sim 或 nested `intersim2` 前，需要证明 BookSim 能表示
  两个 logical GPU endpoint。
- Options：vendor BookSim、立即扩展 `intersim2`，或使用 pinned external
  BookSim executable 加 reproducible topology/config wrapper。
- Decision：第一版 AnyNet smoke/sweep 使用 `28f4329` external checkout；Accel-Sim
  只保存 config、wrapper、metadata 与结论。
- Why：source intrusion 小、provenance 清晰，并可分离 generic transport timing
  与未来 NVLink/NVSwitch/NVLS semantics。
- Trade-off：没有 embedded injection API 或 Accel-Sim clock coupling；synthetic
  BookSim traffic 不是 communication event trace。
- Validation：topology/delivery/latency tests 通过；hardware bandwidth/protocol
  calibration 需要未来 physical multi-GPU system。

## D006 — 用 timestamped endpoint adapter 嵌入未来 BookSim

- Date：2026-09-19
- Context：standalone BookSim prototype 无法在任意 Accel-Sim cycle 注入并完成
  event；`TrafficManager::Run` 将 `_Step` 与 packet queue 隐藏在 protected method
  后面。
- Options：等同 GPU/network cycle、batch standalone executable、patch narrow
  BookSim facade，或使用显式 clock conversion 的 event-driven adapter。
- Decision：BookSim 作为 transport backend，置于 endpoint adapter 后；接口提供
  `create/inject/advance_to/poll_completed/stats`，使用 rational timestamp
  conversion 和 integer picosecond global clock。只有 transport API 确定后才添加
  tracked BookSim facade。
- Why：保留 unequal clock domain 下的 event ordering，并让 transport、NVSwitch
  semantics 与 NVLS reduction 可独立测试。
- Validation：Ring/NVLS 前必须验证 packetization、completion ordering、clock-ratio
  conversion 与 byte/flit accounting。

## D007 — PTX 作为 design input，SASS 作为 validation input

- Date：2026-09-20
- Context：研究目标包含 A800/H100 hardware 不存在的 ISA/microarchitecture
  feature，而 SASS replay 必须依赖真实 GPU 产生对应 instruction。
- Decision：PTX execution-driven mode 主导新 ISA、GPU timing、scheduler 和
  in-switch-computing semantics；真实 SASS trace 继续验证已有 A800/H100
  behaviour、compiler effect 与 hardware measurement。
- Why：真实 GPU 不可能产生不存在的 instruction trace；PTX 可以定义 functional
  semantics/timing，同时不把 synthetic trace 伪装成 hardware evidence。SASS
  保留 compiler/machine-level fidelity。
- Trade-off：PTX 不会自动模拟真实 SASS register allocation 或 machine scheduling；
  Hopper validation 仍需 H100。multi-GPU rank/collective/NVSwitch semantics 与
  两条 compute path 分离。
- Validation：先运行标准 SM90 PTX baseline，再运行 minimal custom PTX instruction
  test；A800/H100 SASS replay 作为独立 evidence track。

## D008 — `.codex` 面向人的正文默认使用中文

- Date：2026-09-21
- Context：历史文档中中英混排过多，影响长期阅读和恢复。
- Decision：README、docs、experiments、knowledge、rules 与 skills 的解释性
  正文使用中文；PTX、SASS、opcode、pipeline、BookSim、NVLink、NVSwitch、
  NVLS、API、symbol、path、metric、evidence label 等专有名词保留英文。
- Why：中文保证阅读连贯，英文术语保证与 source、command 和论文一致。
- Validation：后续每次 document-change 检查语言一致性，并避免翻译 code/path。
