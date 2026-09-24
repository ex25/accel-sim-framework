"""P2 point-to-point event bridge over the deterministic transport contract.

This is still a reference adapter, not an embedded BookSim or NVLink model. It
connects the P0 event lifecycle to the existing transport accounting model so a
future backend can replace only the transport object.
"""

from __future__ import annotations

from fractions import Fraction

from communication_events import (
    CommunicationEvent,
    CommunicationEventOrchestrator,
    EventCompletion,
    EventKind,
    EventState,
)
from transport_contract import ClockDomain, ReferenceEndpointAdapter


class PointToPointEventBridge:
    """Submit ``COMM_SUBMIT`` events and route transport completions back."""

    def __init__(
        self,
        network_clock: ClockDomain,
        *,
        flit_bytes: int = 32,
        protocol_overhead_bytes: int = 0,
        fixed_latency_cycles: int = 0,
        orchestrator: CommunicationEventOrchestrator | None = None,
    ) -> None:
        self.orchestrator = orchestrator or CommunicationEventOrchestrator()
        self.transport = ReferenceEndpointAdapter(
            network_clock,
            flit_bytes=flit_bytes,
            protocol_overhead_bytes=protocol_overhead_bytes,
            fixed_latency_cycles=fixed_latency_cycles,
        )
        self._flits: dict[str, int] = {}
        self._events: dict[str, CommunicationEvent] = {}

    def submit(self, event: CommunicationEvent) -> EventState:
        """Register one point-to-point event with deterministic validation."""

        if event.kind is not EventKind.COMM_SUBMIT:
            raise ValueError("point-to-point bridge accepts only comm_submit events")
        if len(event.src) != 1 or len(event.dst) != 1:
            raise ValueError("point-to-point events require one source and one destination")
        if event.src[0] == event.dst[0]:
            raise ValueError("point-to-point source and destination must differ")
        if event.bytes <= 0:
            raise ValueError("point-to-point event bytes must be positive")
        metadata = dict(event.metadata)
        if metadata.get("time_domain") == "sim_cycle":
            raise ValueError("sim_cycle timestamps must be converted before transport injection")
        state = self.orchestrator.submit(event)
        self._events[event.event_id] = event
        return state

    def pump(self, now_ps: Fraction | int) -> tuple[CommunicationEvent, ...]:
        """Inject ready events and return events completed by ``now_ps``."""

        if now_ps < 0:
            raise ValueError("now_ps must be non-negative")
        for event in self.orchestrator.ready(now_ps):
            self.orchestrator.mark_injected(event.event_id)
            self._flits[event.event_id] = self.transport.inject_message(
                event.src[0], event.dst[0], event.bytes, event.event_id, event.ready_time_ps
            )

        self.transport.advance_to(now_ps)
        completed: list[CommunicationEvent] = []
        for completion in self.transport.poll_completed():
            event = self._events[completion.tag]
            event = self.orchestrator.complete(
                EventCompletion(
                    completion.tag,
                    completion.completion_time_ps,
                    token=event.token,
                    metadata=(
                        ("transport", "reference"),
                        ("flits", str(self._flits.pop(completion.tag))),
                    ),
                )
            )
            completed.append(event)
        return tuple(completed)
