#!/usr/bin/env python3
"""Focused tests for P5 NVLS multicast/reduction completion."""

import unittest

from communication_events import EventKind, EventState
from nvswitch_transport import NVSwitchTopology
from nvls_collective import NVLSReducePlan, NVLSReferenceEngine
from ring_collective import RingAllReducePlan
from transport_contract import ClockDomain


class NVLSCollectiveTest(unittest.TestCase):
    def test_eight_rank_ingress_and_switch_reduction(self) -> None:
        topology = NVSwitchTopology()
        engine = NVLSReferenceEngine(topology, ClockDomain.from_frequency_mhz(1000))
        plan = NVLSReducePlan("nvls-8", tuple(range(8)), 1024)
        engine.submit(plan)
        events = plan.events(topology)
        self.assertEqual(len(events), 9)
        self.assertTrue(all(event.multicast_group == tuple(range(8)) for event in events))
        self.assertEqual(
            engine.orchestrator.state(plan.reduction_id), EventState.WAITING
        )
        engine.pump(0)
        self.assertEqual(engine.orchestrator.state(plan.reduction_id), EventState.WAITING)
        for time_ps in range(1_000, 50_000, 1_000):
            engine.pump(time_ps)
            if engine.complete(plan.collective_id):
                break
        self.assertTrue(engine.complete(plan.collective_id))
        self.assertEqual(engine.transport.total_bytes, 8 * 1024)
        self.assertEqual(
            engine.orchestrator.state(plan.reduction_id), EventState.COMPLETED
        )

    def test_nvls_has_one_ingress_per_source_not_ring_fanout(self) -> None:
        topology = NVSwitchTopology()
        plan = NVLSReducePlan("nvls-count", tuple(range(8)), 1 << 20)
        ingress = plan.events(topology)[:-1]
        ring = RingAllReducePlan("ring-count", tuple(range(8)), 1 << 20)
        self.assertEqual(sum(event.bytes for event in ingress), 8 * (1 << 20))
        self.assertEqual(sum(event.bytes for event in ring.events()), 14 * (1 << 20))

    def test_unattached_rank_rejects_plan(self) -> None:
        topology = NVSwitchTopology(
            {0: (0,), 1: (0,), 2: (1,), 3: (1,)}, gpu_count=4, switch_count=2
        )
        plan = NVLSReducePlan("nvls-bad", (0, 1, 2), 96, switch_id=0)
        with self.assertRaises(ValueError):
            plan.events(topology)


if __name__ == "__main__":
    unittest.main()
