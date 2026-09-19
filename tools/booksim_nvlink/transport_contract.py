"""Reference contract for a future embedded BookSim endpoint adapter.

This module intentionally contains no BookSim dependency.  It is a compact
executable specification for packet accounting and rational clock conversion;
the eventual backend must preserve these semantics while using BookSim for
transport timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from heapq import heappop, heappush
from math import ceil


@dataclass(frozen=True)
class ClockDomain:
    """Periodic clock represented exactly as picoseconds per cycle."""

    period_ps: Fraction

    @classmethod
    def from_frequency_mhz(cls, frequency_mhz: int) -> "ClockDomain":
        if frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive")
        return cls(Fraction(1_000_000, frequency_mhz))

    def cycles_through(self, time_ps: Fraction | int) -> int:
        """Number of clock edges at or before ``time_ps`` (edge zero included)."""
        if time_ps < 0:
            raise ValueError("time_ps must be non-negative")
        return int(Fraction(time_ps) // self.period_ps)


def packet_count(message_bytes: int, flit_bytes: int, protocol_overhead_bytes: int = 0) -> int:
    if message_bytes <= 0 or flit_bytes <= 0 or protocol_overhead_bytes < 0:
        raise ValueError("message, flit, and overhead values are invalid")
    return ceil((message_bytes + protocol_overhead_bytes) / flit_bytes)


@dataclass(frozen=True)
class Completion:
    tag: str
    src: int
    dst: int
    bytes: int
    completion_time_ps: Fraction


@dataclass(order=True)
class _Pending:
    completion_time_ps: Fraction
    sequence: int
    tag: str = field(compare=False)
    src: int = field(compare=False)
    dst: int = field(compare=False)
    bytes: int = field(compare=False)


class ReferenceEndpointAdapter:
    """Deterministic contract model used before a BookSim facade exists.

    The completion delay is a transport-only placeholder: fixed network cycles
    plus one serialization cycle per flit. It is not a physical NVLink model.
    """

    def __init__(self, network_clock: ClockDomain, flit_bytes: int = 32,
                 protocol_overhead_bytes: int = 0, fixed_latency_cycles: int = 0):
        if fixed_latency_cycles < 0:
            raise ValueError("fixed_latency_cycles must be non-negative")
        self.network_clock = network_clock
        self.flit_bytes = flit_bytes
        self.protocol_overhead_bytes = protocol_overhead_bytes
        self.fixed_latency_cycles = fixed_latency_cycles
        self._sequence = 0
        self._pending: list[_Pending] = []
        self._completed: list[Completion] = []
        self.total_bytes = 0
        self.total_flits = 0

    def inject_message(self, src: int, dst: int, bytes: int, tag: str,
                       ready_time_ps: Fraction | int) -> int:
        if src == dst or bytes <= 0:
            raise ValueError("source/destination or byte count is invalid")
        if ready_time_ps < 0:
            raise ValueError("ready_time_ps must be non-negative")
        flits = packet_count(bytes, self.flit_bytes, self.protocol_overhead_bytes)
        ready_cycle = self.network_clock.cycles_through(ready_time_ps)
        completion_cycle = ready_cycle + self.fixed_latency_cycles + flits
        completion_time = completion_cycle * self.network_clock.period_ps
        heappush(self._pending, _Pending(completion_time, self._sequence, tag, src, dst, bytes))
        self._sequence += 1
        self.total_bytes += bytes
        self.total_flits += flits
        return flits

    def advance_to(self, time_ps: Fraction | int) -> None:
        if time_ps < 0:
            raise ValueError("time_ps must be non-negative")
        target = Fraction(time_ps)
        while self._pending and self._pending[0].completion_time_ps <= target:
            item = heappop(self._pending)
            self._completed.append(Completion(item.tag, item.src, item.dst, item.bytes,
                                               item.completion_time_ps))

    def poll_completed(self) -> list[Completion]:
        completed, self._completed = self._completed, []
        return completed
