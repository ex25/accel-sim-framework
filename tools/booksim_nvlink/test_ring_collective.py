#!/usr/bin/env python3
"""Focused tests for P3 Ring all-reduce communication dependencies."""

import unittest

from communication_events import EventState
from point_to_point_bridge import PointToPointEventBridge
from ring_collective import RingAllReducePlan
from transport_contract import ClockDomain


class RingCollectiveTest(unittest.TestCase):
    def test_four_rank_schedule_and_completion(self) -> None:
        plan = RingAllReducePlan("ring-4", (0, 1, 2, 3), 256)
        events = plan.events()
        self.assertEqual(len(events), 24)
        self.assertTrue(all(event.bytes == 64 for event in events))
        self.assertEqual(sum(event.bytes for event in events), 1536)
        bridge = PointToPointEventBridge(ClockDomain.from_frequency_mhz(1000))
        plan.submit_to(bridge)

        # At time zero only the first reduce-scatter sends are ready.
        self.assertEqual(len(bridge.orchestrator.ready(0)), 4)
        self.assertEqual(bridge.pump(0), ())
        self.assertEqual(
            bridge.orchestrator.state("ring-4:reduce_scatter:1:rank0"),
            EventState.WAITING,
        )
        for time_ps in range(1_000, 40_000, 1_000):
            bridge.pump(time_ps)
            if plan.is_complete(bridge):
                break
        self.assertTrue(plan.is_complete(bridge))
        self.assertEqual(bridge.transport.total_bytes, 1536)
        self.assertEqual(bridge.transport.total_flits, 48)

    def test_chunk_and_predecessor_dependencies(self) -> None:
        plan = RingAllReducePlan("ring-3", (2, 4, 7), 300)
        events = {event.event_id: event for event in plan.events()}
        self.assertEqual(events["ring-3:reduce_scatter:0:rank2"].chunk_id, 0)
        self.assertEqual(
            events["ring-3:reduce_scatter:1:rank2"].dependency_ids,
            ("ring-3:reduce_scatter:0:rank7",),
        )
        self.assertEqual(
            events["ring-3:all_gather:0:rank2"].dependency_ids,
            ("ring-3:reduce_scatter:1:rank7",),
        )

    def test_invalid_sizes(self) -> None:
        with self.assertRaises(ValueError):
            RingAllReducePlan("bad", (0, 1, 2), 100)
        with self.assertRaises(ValueError):
            RingAllReducePlan("bad", (0, 0), 128)

    def test_eight_rank_message_accounting(self) -> None:
        plan = RingAllReducePlan("ring-8", tuple(range(8)), 1 << 20)
        events = plan.events()
        self.assertEqual(len(events), 2 * 7 * 8)
        self.assertEqual(plan.chunk_bytes, 1 << 17)
        self.assertEqual(sum(event.bytes for event in events), 14 * (1 << 20))


if __name__ == "__main__":
    unittest.main()
