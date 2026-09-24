#!/usr/bin/env python3
"""Focused tests for P4 NVSwitch topology/routing semantics."""

import unittest

from communication_events import CommunicationEvent, EventKind
from nvswitch_transport import NVSwitchTopology, NVSwitchTransportBridge
from ring_collective import RingAllReducePlan
from transport_contract import ClockDomain


class NVSwitchTransportTest(unittest.TestCase):
    def test_default_dgx_shape_is_explicit_and_connected(self) -> None:
        topology = NVSwitchTopology()
        self.assertEqual((topology.gpu_count, topology.switch_count), (8, 4))
        self.assertEqual(topology.route(0, 7, 0).path, ("gpu:0", "nvswitch:0", "gpu:7"))
        self.assertEqual(topology.route(0, 7, 5).switch, 1)
        self.assertEqual(topology.route(0, 7).link_count, 2)

    def test_custom_disconnected_topology_rejects_route(self) -> None:
        topology = NVSwitchTopology(
            {0: (0,), 1: (0,), 2: (1,), 3: (1,)}, gpu_count=4, switch_count=2
        )
        with self.assertRaises(ValueError):
            topology.route(0, 2)

    def test_ring_uses_switch_routes_and_completes(self) -> None:
        topology = NVSwitchTopology()
        bridge = NVSwitchTransportBridge(topology, ClockDomain.from_frequency_mhz(1000))
        plan = RingAllReducePlan("switch-ring", tuple(range(8)), 1024)
        plan.submit_to(bridge)
        for time_ps in range(0, 80_000, 1_000):
            bridge.pump(time_ps)
            if plan.is_complete(bridge):
                break
        self.assertTrue(plan.is_complete(bridge))
        self.assertEqual(len(bridge._routes), 112)
        self.assertTrue(all(route.link_count == 2 for route in bridge._routes.values()))
        self.assertEqual(bridge.transport.total_bytes, 14 * 1024)

    def test_route_checked_before_event_registration(self) -> None:
        topology = NVSwitchTopology({0: (0,), 1: (1,)}, gpu_count=2, switch_count=2)
        bridge = NVSwitchTransportBridge(topology, ClockDomain.from_frequency_mhz(1000))
        event = CommunicationEvent(
            event_id="disconnected",
            device_id=0,
            rank=0,
            stream_id="s",
            kernel_id="k",
            kind=EventKind.COMM_SUBMIT,
            src=(0,),
            dst=(1,),
            bytes=32,
        )
        with self.assertRaises(ValueError):
            bridge.submit(event)
        self.assertEqual(bridge.orchestrator.snapshot(), {})


if __name__ == "__main__":
    unittest.main()
