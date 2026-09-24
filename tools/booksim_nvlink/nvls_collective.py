"""P5 NVLS-style multicast/reduction completion reference."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from communication_events import (
    CommunicationEvent,
    CommunicationEventOrchestrator,
    EventCompletion,
    EventKind,
    EventState,
)
from nvswitch_transport import NVSwitchTopology
from transport_contract import ClockDomain, ReferenceEndpointAdapter


@dataclass(frozen=True)
class NVLSReducePlan:
    collective_id: str
    ranks: tuple[int, ...]
    bytes_per_rank: int
    root: int = 0
    switch_id: int = 0
    stream_id: str = "stream-0"
    kernel_id: str = "nvls-reduce"
    ready_time_ps: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if not self.collective_id:
            raise ValueError("collective_id must not be empty")
        if len(self.ranks) < 2 or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("NVLS reduction requires at least two unique ranks")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("ranks must be non-negative")
        if self.bytes_per_rank <= 0:
            raise ValueError("bytes_per_rank must be positive")
        if self.root not in self.ranks or self.switch_id < 0 or self.ready_time_ps < 0:
            raise ValueError("root, switch_id, and ready_time_ps are invalid")

    def ingress_id(self, rank: int) -> str:
        return f"{self.collective_id}:multicast:rank{rank}"

    @property
    def reduction_id(self) -> str:
        return f"{self.collective_id}:reduce:complete"

    def events(self, topology: NVSwitchTopology) -> tuple[CommunicationEvent, ...]:
        """Create one source-to-switch multicast ingress plus one reduction event."""

        for rank in self.ranks:
            if self.switch_id not in topology.gpu_to_switches[rank]:
                raise ValueError(f"GPU {rank} is not attached to NVSwitch {self.switch_id}")
        switch_endpoint = topology.gpu_count + self.switch_id
        group = tuple(self.ranks)
        ingress = tuple(
            CommunicationEvent(
                event_id=self.ingress_id(rank),
                device_id=rank,
                rank=rank,
                stream_id=self.stream_id,
                kernel_id=self.kernel_id,
                kind=EventKind.COMM_SUBMIT,
                scope="nvswitch",
                operation="multicast",
                collective_id=self.collective_id,
                src=(rank,),
                dst=(switch_endpoint,),
                multicast_group=group,
                bytes=self.bytes_per_rank,
                ready_time_ps=self.ready_time_ps,
                metadata=(
                    ("phase", "multicast_ingress"),
                    ("destination_kind", "nvswitch"),
                    ("switch_id", str(self.switch_id)),
                ),
            )
            for rank in self.ranks
        )
        reduction = CommunicationEvent(
            event_id=self.reduction_id,
            device_id=self.root,
            rank=self.root,
            stream_id=self.stream_id,
            kernel_id=self.kernel_id,
            kind=EventKind.COLLECTIVE_COMPLETE,
            scope="nvswitch",
            operation="reduce",
            collective_id=self.collective_id,
            src=group,
            dst=(self.root,),
            multicast_group=group,
            bytes=self.bytes_per_rank,
            ready_time_ps=self.ready_time_ps,
            dependency_ids=tuple(event.event_id for event in ingress),
            metadata=(
                ("phase", "switch_reduce"),
                ("reduction", "abstract"),
                ("switch_id", str(self.switch_id)),
            ),
        )
        return ingress + (reduction,)


class NVLSReferenceEngine:
    """Complete multicast ingress and abstract switch reduction deterministically."""

    def __init__(
        self,
        topology: NVSwitchTopology,
        network_clock: ClockDomain,
        *,
        fixed_latency_cycles: int = 0,
        flit_bytes: int = 32,
    ) -> None:
        self.topology = topology
        self.orchestrator = CommunicationEventOrchestrator()
        self.transport = ReferenceEndpointAdapter(
            network_clock, flit_bytes=flit_bytes, fixed_latency_cycles=fixed_latency_cycles
        )
        self._events: dict[str, CommunicationEvent] = {}
        self._plans: dict[str, NVLSReducePlan] = {}
        self._flits: dict[str, int] = {}

    def submit(self, plan: NVLSReducePlan) -> None:
        events = plan.events(self.topology)
        if plan.collective_id in self._plans:
            raise ValueError(f"duplicate collective_id: {plan.collective_id}")
        for event in events:
            self.orchestrator.submit(event)
            self._events[event.event_id] = event
        self._plans[plan.collective_id] = plan

    def pump(self, now_ps: Fraction | int) -> tuple[CommunicationEvent, ...]:
        if now_ps < 0:
            raise ValueError("now_ps must be non-negative")
        for event in self.orchestrator.ready(now_ps):
            if event.kind is not EventKind.COMM_SUBMIT:
                continue
            self.orchestrator.mark_injected(event.event_id)
            self._flits[event.event_id] = self.transport.inject_message(
                event.src[0], event.dst[0], event.bytes, event.event_id, event.ready_time_ps
            )

        self.transport.advance_to(now_ps)
        completed: list[CommunicationEvent] = []
        for transport_completion in self.transport.poll_completed():
            event = self._events[transport_completion.tag]
            self.orchestrator.complete(
                EventCompletion(
                    transport_completion.tag,
                    transport_completion.completion_time_ps,
                    metadata=(
                        ("transport", "nvls_ingress_reference"),
                        ("flits", str(self._flits.pop(transport_completion.tag))),
                    ),
                )
            )
            completed.append(event)

        for event in self.orchestrator.ready(now_ps):
            if event.kind is not EventKind.COLLECTIVE_COMPLETE:
                continue
            self.orchestrator.mark_injected(event.event_id)
            self.orchestrator.complete(
                EventCompletion(
                    event.event_id,
                    now_ps,
                    metadata=(("transport", "nvls_switch_reduce_reference"),),
                )
            )
            completed.append(event)
        return tuple(completed)

    def complete(self, collective_id: str) -> bool:
        plan = self._plans[collective_id]
        return self.orchestrator.state(plan.reduction_id) == EventState.COMPLETED

