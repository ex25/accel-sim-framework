"""P3 NCCL-style Ring all-reduce communication schedule (reference model).

This models rank/chunk/step dependencies and byte movement. It does not
execute reduction arithmetic or claim to reproduce NCCL kernel scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from communication_events import CommunicationEvent, EventKind, EventState
from point_to_point_bridge import PointToPointEventBridge


@dataclass(frozen=True)
class RingAllReducePlan:
    collective_id: str
    ranks: tuple[int, ...]
    bytes_per_rank: int
    stream_id: str = "stream-0"
    kernel_id: str = "ring-allreduce"
    ready_time_ps: Fraction = Fraction(0)
    channel_id: int = 0

    def __post_init__(self) -> None:
        if not self.collective_id:
            raise ValueError("collective_id must not be empty")
        if len(self.ranks) < 2 or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("ring requires at least two unique ranks")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("ranks must be non-negative")
        if self.bytes_per_rank <= 0 or self.bytes_per_rank % len(self.ranks):
            raise ValueError("bytes_per_rank must be positive and divisible by rank count")
        if self.channel_id < 0 or self.ready_time_ps < 0:
            raise ValueError("channel_id and ready_time_ps must be non-negative")

    @property
    def chunk_bytes(self) -> int:
        return self.bytes_per_rank // len(self.ranks)

    @property
    def steps_per_phase(self) -> int:
        return len(self.ranks) - 1

    def event_id(self, phase: str, step: int, rank_position: int) -> str:
        return f"{self.collective_id}:{phase}:{step}:rank{self.ranks[rank_position]}"

    def events(self) -> tuple[CommunicationEvent, ...]:
        """Generate reduce-scatter then all-gather directed send events.

        For each rank, step ``s`` sends to its successor only after its
        predecessor's step ``s-1`` send has completed. All-gather step zero
        depends on the last reduce-scatter receive.
        """

        count = len(self.ranks)
        result: list[CommunicationEvent] = []
        for phase in ("reduce_scatter", "all_gather"):
            for step in range(count - 1):
                for position, rank in enumerate(self.ranks):
                    predecessor = (position - 1) % count
                    successor = self.ranks[(position + 1) % count]
                    chunk = (
                        (position - step) % count
                        if phase == "reduce_scatter"
                        else (position + 1 - step) % count
                    )
                    dependencies: tuple[str, ...] = ()
                    if step > 0:
                        dependencies = (self.event_id(phase, step - 1, predecessor),)
                    elif phase == "all_gather":
                        dependencies = (
                            self.event_id("reduce_scatter", count - 2, predecessor),
                        )
                    result.append(
                        CommunicationEvent(
                            event_id=self.event_id(phase, step, position),
                            device_id=rank,
                            rank=rank,
                            stream_id=self.stream_id,
                            kernel_id=self.kernel_id,
                            kind=EventKind.COMM_SUBMIT,
                            scope="gpu",
                            operation="all_reduce",
                            collective_id=self.collective_id,
                            src=(rank,),
                            dst=(successor,),
                            bytes=self.chunk_bytes,
                            chunk_id=chunk,
                            channel_id=self.channel_id,
                            ready_time_ps=self.ready_time_ps,
                            dependency_ids=dependencies,
                            metadata=(("phase", phase), ("step", str(step))),
                        )
                    )
        return tuple(result)

    def submit_to(self, bridge: PointToPointEventBridge) -> None:
        for event in self.events():
            bridge.submit(event)

    def is_complete(self, bridge: PointToPointEventBridge) -> bool:
        return all(
            bridge.orchestrator.state(event.event_id) == EventState.COMPLETED
            for event in self.events()
        )
