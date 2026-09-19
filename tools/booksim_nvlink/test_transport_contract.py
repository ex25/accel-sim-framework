#!/usr/bin/env python3
"""Unit tests for the pre-BookSim endpoint contract."""

from fractions import Fraction
import unittest

from transport_contract import ClockDomain, ReferenceEndpointAdapter, packet_count


class TransportContractTest(unittest.TestCase):
    def test_packet_accounting(self) -> None:
        self.assertEqual(packet_count(1, 32), 1)
        self.assertEqual(packet_count(32, 32), 1)
        self.assertEqual(packet_count(33, 32), 2)
        self.assertEqual(packet_count(64, 32, 1), 3)

    def test_rational_clock_conversion(self) -> None:
        clock = ClockDomain.from_frequency_mhz(1250)
        self.assertEqual(clock.period_ps, Fraction(800, 1))
        self.assertEqual(clock.cycles_through(799), 0)
        self.assertEqual(clock.cycles_through(800), 1)
        self.assertEqual(clock.cycles_through(Fraction(2400)), 3)

    def test_completion_order_and_timestamp(self) -> None:
        clock = ClockDomain.from_frequency_mhz(1000)
        adapter = ReferenceEndpointAdapter(clock, flit_bytes=32, fixed_latency_cycles=2)
        adapter.inject_message(0, 1, 64, "large", 0)
        adapter.inject_message(1, 0, 32, "small", 0)
        self.assertEqual(adapter.total_flits, 3)
        adapter.advance_to(2_999)
        self.assertEqual(adapter.poll_completed(), [])
        adapter.advance_to(3_000)
        completed = adapter.poll_completed()
        self.assertEqual([item.tag for item in completed], ["small"])
        adapter.advance_to(4_000)
        self.assertEqual([item.tag for item in adapter.poll_completed()], ["large"])
        self.assertEqual(adapter.total_bytes, 96)


if __name__ == "__main__":
    unittest.main()
