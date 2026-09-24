#!/usr/bin/env python3
"""Focused tests for the P2 point-to-point event bridge."""

from fractions import Fraction
import unittest

from communication_events import CommunicationEvent, EventKind, EventState
from point_to_point_bridge import PointToPointEventBridge
from transport_contract import ClockDomain


def comm_event(event_id: str, ready_time_ps: int = 0) -> CommunicationEvent:
    return CommunicationEvent(
        event_id=event_id,
        device_id=0,
        rank=0,
        stream_id="stream-0",
        kernel_id="kernel-0",
        kind=EventKind.COMM_SUBMIT,
        scope="gpu",
        operation="send",
        src=(0,),
        dst=(1,),
        bytes=64,
        ready_time_ps=ready_time_ps,
    )


class PointToPointBridgeTest(unittest.TestCase):
    def test_submit_inject_and_complete(self) -> None:
        bridge = PointToPointEventBridge(
            ClockDomain.from_frequency_mhz(1000), flit_bytes=32, fixed_latency_cycles=2
        )
        self.assertEqual(bridge.submit(comm_event("send-0")), EventState.READY)
        self.assertEqual(bridge.pump(Fraction(2_999)), ())
        completed = bridge.pump(Fraction(4_000))
        self.assertEqual([event.event_id for event in completed], ["send-0"])
        self.assertEqual(bridge.orchestrator.state("send-0"), EventState.COMPLETED)
        completion = bridge.orchestrator.completion("send-0")
        self.assertIsNotNone(completion)
        assert completion is not None
        self.assertEqual(dict(completion.metadata), {"flits": "2", "transport": "reference"})

    def test_ready_time_delays_injection(self) -> None:
        bridge = PointToPointEventBridge(ClockDomain.from_frequency_mhz(1000))
        event = comm_event("send-1", ready_time_ps=1_000)
        self.assertEqual(bridge.submit(event), EventState.READY)
        self.assertEqual(bridge.pump(999), ())
        self.assertEqual(bridge.orchestrator.state("send-1"), EventState.READY)
        self.assertEqual([item.event_id for item in bridge.pump(3_000)], ["send-1"])

    def test_rejects_invalid_scope_and_sim_cycle(self) -> None:
        bridge = PointToPointEventBridge(ClockDomain.from_frequency_mhz(1000))
        with self.assertRaises(ValueError):
            bridge.submit(
                CommunicationEvent(
                    **{**comm_event("bad").to_dict(), "src": [0, 1]}
                )
            )
        with self.assertRaises(ValueError):
            bridge.submit(
                CommunicationEvent.from_dict(
                    {**comm_event("cycle").to_dict(), "metadata": {"time_domain": "sim_cycle"}}
                )
            )


if __name__ == "__main__":
    unittest.main()
