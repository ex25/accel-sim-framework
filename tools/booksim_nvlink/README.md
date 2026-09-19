# BookSim two-GPU NVLink transport prototype

This directory contains the first deliberately small network experiment for
the project. BookSim treats node 0 and node 1 as logical GPU endpoints. The
AnyNet topology has one router per endpoint and one bidirectional router link.
The `bitcomp` traffic pattern on two nodes deterministically sends GPU 0 to GPU
1 and GPU 1 to GPU 0, so the run is suitable for a packet-delivery sanity
check and a first latency/throughput trend.

This is a transport model, not a calibrated NVIDIA NVLink implementation. The
link latency, packet size, router delays, flit width, and injection rate are
BookSim parameters. NVLink protocol overhead, link training, credit protocol,
GPU memory semantics, NCCL, NVSwitch, and NVLS are not modelled.

## Reproduce

The BookSim checkout remains an external repository and is pinned in the
project state. Build it there, then run from this repository root:

```bash
make -C /home/wangjz/in-switch/booksim2/src -j8
BOOKSIM_ROOT=/home/wangjz/in-switch/booksim2 \
  tools/booksim_nvlink/run_nvlink2.sh
```

The script accepts `BOOKSIM_ROOT` and `BOOKSIM_BIN` and writes raw output to a
temporary directory unless `OUT_DIR` is supplied. It checks that BookSim
reports two nodes and that the configured `bitcomp` pattern produces traffic
in both directions. The reported latency is BookSim network latency in cycles,
not measured A800 or H100 wall-clock latency.

For a first parameter-sensitivity check:

```bash
BOOKSIM_ROOT=/home/wangjz/in-switch/booksim2 \
  tools/booksim_nvlink/sweep_nvlink2.sh
```

The sweep varies router-link latency 1/4/8 cycles and packet size 1/4/16
flits. It emits a CSV containing final latency, accepted packet rate, and
simulation cycles. Run the analytical accounting check against that CSV:

```bash
python3 tools/booksim_nvlink/check_nvlink2_analytical.py \
  /tmp/accelsim-booksim-nvlink2-sweep/nvlink2_sweep.csv \
  --flit-bytes 32 --link-flits-per-cycle 1
```

The checker verifies `message_bytes = packet_flits * flit_bytes`, a
serialization lower bound, and monotonic latency trends as packet size or
configured link latency increases. These are model sanity checks, not
calibration against physical NVLink bandwidth.

Before embedding BookSim, run the dependency-free endpoint contract tests:

```bash
cd tools/booksim_nvlink
python3 -m unittest test_transport_contract.py
```

They validate packetization, exact rational clock conversion, and completion
ordering. `transport_contract.py` is a reference specification only; its fixed
latency is not a BookSim or hardware result.

## Interpretation

The first correctness checks are topology connectivity, non-zero delivered
packets, bidirectional packet count, and finite latency. Performance checks
compare latency and throughput while sweeping only one explicit parameter at a
time. Any comparison to a physical NVLink link requires later H100/A800
hardware measurements and a documented frequency/clock conversion.
