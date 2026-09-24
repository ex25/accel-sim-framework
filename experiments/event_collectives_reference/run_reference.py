#!/usr/bin/env python3
"""Run the P2/P3/P4/P5 deterministic collective reference checks."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/booksim_nvlink"))

from nvswitch_transport import NVSwitchTopology, NVSwitchTransportBridge  # noqa: E402
from nvls_collective import NVLSReducePlan, NVLSReferenceEngine  # noqa: E402
from ring_collective import RingAllReducePlan  # noqa: E402
from transport_contract import ClockDomain  # noqa: E402


def main() -> int:
    topology = NVSwitchTopology()
    clock = ClockDomain.from_frequency_mhz(1000)

    ring_bridge = NVSwitchTransportBridge(topology, clock)
    ring = RingAllReducePlan("ring-8", tuple(range(8)), 1024)
    ring.submit_to(ring_bridge)
    for now_ps in range(0, 80_000, 1_000):
        ring_bridge.pump(now_ps)
        if ring.is_complete(ring_bridge):
            break
    if not ring.is_complete(ring_bridge):
        print("REFERENCE_COLLECTIVES_FAIL ring did not complete")
        return 1

    nvls_engine = NVLSReferenceEngine(topology, clock)
    nvls = NVLSReducePlan("nvls-8", tuple(range(8)), 1024)
    nvls_engine.submit(nvls)
    for now_ps in range(0, 50_000, 1_000):
        nvls_engine.pump(now_ps)
        if nvls_engine.complete(nvls.collective_id):
            break
    if not nvls_engine.complete(nvls.collective_id):
        print("REFERENCE_COLLECTIVES_FAIL nvls did not complete")
        return 1

    expected_ring_bytes = 14 * 1024
    expected_nvls_bytes = 8 * 1024
    if ring_bridge.transport.total_bytes != expected_ring_bytes:
        print(f"REFERENCE_COLLECTIVES_FAIL ring_bytes={ring_bridge.transport.total_bytes}")
        return 1
    if nvls_engine.transport.total_bytes != expected_nvls_bytes:
        print(f"REFERENCE_COLLECTIVES_FAIL nvls_bytes={nvls_engine.transport.total_bytes}")
        return 1
    print(
        "REFERENCE_COLLECTIVES_PASS "
        f"ring_bytes={expected_ring_bytes} ring_flits={ring_bridge.transport.total_flits} "
        f"nvls_ingress_bytes={expected_nvls_bytes} "
        f"nvls_ingress_flits={nvls_engine.transport.total_flits} "
        "topology=8gpu-4nvswitch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
