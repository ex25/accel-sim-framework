#!/usr/bin/env python3
"""Focused P0 tests for the PTX-to-communication event contract."""

from fractions import Fraction
import unittest

from communication_events import (
    CommunicationEvent,
    CommunicationEventOrchestrator,
    EventCompletion,
    EventKind,
    EventState,
    LogicalGPUContext,
)


def event(event_id: str, kind: EventKind, *, deps=(), ready=0, rank=0, device=0, bytes=0):
    is_communication = kind in (EventKind.COMM_SUBMIT, EventKind.COMM_INJECT)
    return CommunicationEvent(
        event_id=event_id,
        device_id=device,
        rank=rank,
        stream_id="stream-0",
        kernel_id="kernel-0",
        kind=kind,
        operation="allreduce" if is_communication else None,
        collective_id="collective-0" if is_communication else None,
        src=(0,) if is_communication else (),
        dst=(1,) if is_communication else (),
        bytes=bytes,
        ready_time_ps=ready,
        dependency_ids=tuple(deps),
    )


class CommunicationEventTest(unittest.TestCase):
    def test_round_trip_preserves_fraction_and_metadata(self) -> None:
        original = CommunicationEvent(
            event_id="send-0",
            device_id=0,
            rank=0,
            stream_id="s0",
            kernel_id="k0",
            kind=EventKind.COMM_SUBMIT,
            scope="gpu",
            operation="allreduce",
            collective_id="c0",
            src=(0,),
            dst=(1, 2),
            multicast_group=(0, 1, 2),
            bytes=4096,
            chunk_id=3,
            channel_id=1,
            pc=0x80,
            instruction_uid=17,
            token="mbar-0",
            ready_time_ps=Fraction(5, 2),
            dependency_ids=("arrive-0",),
            metadata=(("dtype", "bf16"),),
        )
        self.assertEqual(CommunicationEvent.from_dict(original.to_dict()), original)

    def test_dependency_release_is_deterministic(self) -> None:
        orchestrator = CommunicationEventOrchestrator()
        orchestrator.register_context(LogicalGPUContext(0, 0, "gpu0"))
        first = event("arrive-0", EventKind.LOCAL_BARRIER_ARRIVE)
        second = event("send-0", EventKind.COMM_SUBMIT, deps=(first.event_id,), ready=2, bytes=64)
        self.assertEqual(orchestrator.submit(first), EventState.READY)
        self.assertEqual(orchestrator.submit(second), EventState.WAITING)
        self.assertEqual(orchestrator.ready(0), (first,))
        orchestrator.mark_injected(first.event_id)
        orchestrator.complete(EventCompletion(first.event_id, Fraction(3, 2), "phase-1"))
        self.assertEqual(orchestrator.state(second.event_id), EventState.READY)
        self.assertEqual(orchestrator.ready(1), ())
        self.assertEqual(orchestrator.ready(2), (second,))

    def test_completion_requires_ready_or_injected(self) -> None:
        orchestrator = CommunicationEventOrchestrator()
        pending = event("send-0", EventKind.COMM_SUBMIT, ready=10, bytes=32)
        orchestrator.submit(pending)
        with self.assertRaisesRegex(ValueError, "cannot complete"):
            orchestrator.complete(EventCompletion(pending.event_id, 10))
        self.assertEqual(orchestrator.ready(10), (pending,))
        orchestrator.mark_injected(pending.event_id)
        completed = orchestrator.complete(EventCompletion(pending.event_id, 12))
        self.assertEqual(completed, pending)
        self.assertEqual(orchestrator.state(pending.event_id), EventState.COMPLETED)

    def test_duplicate_rank_context_is_rejected(self) -> None:
        orchestrator = CommunicationEventOrchestrator()
        orchestrator.register_context(LogicalGPUContext(0, 0))
        with self.assertRaisesRegex(ValueError, "different context"):
            orchestrator.register_context(LogicalGPUContext(1, 0))


if __name__ == "__main__":
    unittest.main()
