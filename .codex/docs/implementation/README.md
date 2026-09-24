# Implementation records

这里按 milestone 保存实现记录。每份记录都应包含问题、设计、修改文件/符号、
call path、配置、before/after、测试、evidence、限制和下一步。

- `007-nvbit-a800-booksim-nvlink2.md`：A800/NVBit 与 BookSim transport scaffold。
- `008-trace-autopsy-booksim-clock.md`：trace autopsy 与 endpoint/clock contract。
- `009-sm90-ptx-baseline.md`：SM90 PTX execution-driven baseline。
- `010-cais-add-ptx.md`：最小 single-GPU `cais.add.u32` custom PTX instruction。
- `011-ptx-event-schema.md`：PTX-to-communication event schema 与 P0 reference
  orchestrator。
- `012-ptx-sync-event-producer.md`：P1 GPU-local PTX synchronization event
  producer/completion bridge。
- `013-point-to-point-event-bridge.md`：P2 point-to-point event lifecycle 与
  reference transport adapter。
- `014-ring-collective.md`：P3 NCCL-style Ring all-reduce schedule。
- `015-nvswitch-topology.md`：P4 configurable NVSwitch topology/route facade。
- `016-nvls-reference.md`：P5 NVLS multicast/reduction completion reference。

Raw trace、binary、build tree 和大 log 不放在本目录；只保存可读的 research
record 与 reproducibility metadata。
