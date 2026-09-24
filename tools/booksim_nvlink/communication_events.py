"""PTX-to-communication event contract and deterministic orchestrator.

This module is intentionally independent of GPGPU-Sim and BookSim.  It is the
P0 reference contract: a GPU-side producer creates an event, a future NCCL /
fabric backend consumes it, and a completion is routed back to the producer.
The event schema is transport-agnostic and does not claim to model NVLink,
NVSwitch, or NVLS timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Iterable, Mapping


def _as_fraction(value: Fraction | int) -> Fraction:
    result = Fraction(value)
    if result < 0:
        raise ValueError("time values must be non-negative")
    return result


def _as_tuple(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    result = tuple(values)
    if any(value < 0 for value in result):
        raise ValueError("rank/device identifiers must be non-negative")
    return result


class EventKind(str, Enum):
    """Observable boundary between GPU execution and communication layers."""

    LOCAL_BARRIER_ARRIVE = "local_barrier_arrive"
    LOCAL_BARRIER_WAIT = "local_barrier_wait"
    MEMORY_RELEASE = "memory_release"
    MEMORY_ACQUIRE = "memory_acquire"
    COMM_SUBMIT = "comm_submit"
    COMM_INJECT = "comm_inject"
    COMM_COMPLETE = "comm_complete"
    BARRIER_WAKEUP = "barrier_wakeup"
    COLLECTIVE_COMPLETE = "collective_complete"


class EventState(str, Enum):
    """State visible to the event orchestrator."""

    WAITING = "waiting"
    READY = "ready"
    INJECTED = "injected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CommunicationEvent:
    """A serializable event produced by a logical GPU context.

    ``bytes`` may be zero for local ordering/barrier events.  Communication
    backends should require a positive value for ``COMM_SUBMIT`` and
    ``COMM_INJECT``.  ``ready_time_ps`` is in the shared global time domain;
    callers must not assume that one GPU cycle equals one network cycle.
    """

    event_id: str
    device_id: int
    rank: int
    stream_id: str
    kernel_id: str
    kind: EventKind
    scope: str = "gpu"
    operation: str | None = None
    collective_id: str | None = None
    src: tuple[int, ...] = ()
    dst: tuple[int, ...] = ()
    multicast_group: tuple[int, ...] = ()
    bytes: int = 0
    chunk_id: int | None = None
    channel_id: int | None = None
    pc: int | None = None
    instruction_uid: int | None = None
    barrier_addr: int | None = None
    phase: int | None = None
    token: str | None = None
    ready_time_ps: Fraction = Fraction(0)
    dependency_ids: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.device_id < 0 or self.rank < 0:
            raise ValueError("device_id and rank must be non-negative")
        if not self.stream_id or not self.kernel_id:
            raise ValueError("stream_id and kernel_id must not be empty")
        if self.bytes < 0:
            raise ValueError("bytes must be non-negative")
        if self.chunk_id is not None and self.chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        if self.channel_id is not None and self.channel_id < 0:
            raise ValueError("channel_id must be non-negative")
        if self.phase is not None and self.phase < 0:
            raise ValueError("phase must be non-negative")
        object.__setattr__(self, "ready_time_ps", _as_fraction(self.ready_time_ps))
        object.__setattr__(self, "src", _as_tuple(self.src))
        object.__setattr__(self, "dst", _as_tuple(self.dst))
        object.__setattr__(self, "multicast_group", _as_tuple(self.multicast_group))
        object.__setattr__(self, "dependency_ids", tuple(self.dependency_ids))
        object.__setattr__(self, "metadata", tuple(self.metadata))

    @property
    def is_communication(self) -> bool:
        return self.kind in (EventKind.COMM_SUBMIT, EventKind.COMM_INJECT)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data without losing rational timestamps."""

        return {
            "event_id": self.event_id,
            "device_id": self.device_id,
            "rank": self.rank,
            "stream_id": self.stream_id,
            "kernel_id": self.kernel_id,
            "kind": self.kind.value,
            "scope": self.scope,
            "operation": self.operation,
            "collective_id": self.collective_id,
            "src": list(self.src),
            "dst": list(self.dst),
            "multicast_group": list(self.multicast_group),
            "bytes": self.bytes,
            "chunk_id": self.chunk_id,
            "channel_id": self.channel_id,
            "pc": self.pc,
            "instruction_uid": self.instruction_uid,
            "barrier_addr": self.barrier_addr,
            "phase": self.phase,
            "token": self.token,
            "ready_time_ps": str(self.ready_time_ps),
            "dependency_ids": list(self.dependency_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CommunicationEvent":
        """Construct an event from the stable dictionary representation."""

        raw_kind = data["kind"]
        kind = raw_kind if isinstance(raw_kind, EventKind) else EventKind(str(raw_kind))
        raw_metadata = data.get("metadata", {})
        if isinstance(raw_metadata, Mapping):
            metadata = tuple(sorted((str(k), str(v)) for k, v in raw_metadata.items()))
        else:
            metadata = tuple(tuple(item) for item in raw_metadata)  # type: ignore[arg-type]
        return cls(
            event_id=str(data["event_id"]),
            device_id=int(data["device_id"]),
            rank=int(data["rank"]),
            stream_id=str(data["stream_id"]),
            kernel_id=str(data["kernel_id"]),
            kind=kind,
            scope=str(data.get("scope", "gpu")),
            operation=None if data.get("operation") is None else str(data["operation"]),
            collective_id=None
            if data.get("collective_id") is None
            else str(data["collective_id"]),
            src=tuple(int(v) for v in data.get("src", ())),
            dst=tuple(int(v) for v in data.get("dst", ())),
            multicast_group=tuple(int(v) for v in data.get("multicast_group", ())),
            bytes=int(data.get("bytes", 0)),
            chunk_id=None if data.get("chunk_id") is None else int(data["chunk_id"]),
            channel_id=None
            if data.get("channel_id") is None
            else int(data["channel_id"]),
            pc=None if data.get("pc") is None else int(data["pc"]),
            instruction_uid=None
            if data.get("instruction_uid") is None
            else int(data["instruction_uid"]),
            barrier_addr=None
            if data.get("barrier_addr") is None
            else int(data["barrier_addr"]),
            phase=None if data.get("phase") is None else int(data["phase"]),
            token=None if data.get("token") is None else str(data["token"]),
            ready_time_ps=Fraction(str(data.get("ready_time_ps", "0"))),
            dependency_ids=tuple(str(v) for v in data.get("dependency_ids", ())),
            metadata=metadata,
        )


@dataclass(frozen=True)
class EventCompletion:
    event_id: str
    completion_time_ps: Fraction
    token: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        object.__setattr__(self, "completion_time_ps", _as_fraction(self.completion_time_ps))
        object.__setattr__(self, "metadata", tuple(self.metadata))


@dataclass
class _EventRecord:
    event: CommunicationEvent
    state: EventState
    sequence: int
    completion: EventCompletion | None = None


@dataclass(frozen=True)
class LogicalGPUContext:
    """Identity used by the event layer, not a simulated SM."""

    device_id: int
    rank: int
    name: str | None = None

    def __post_init__(self) -> None:
        if self.device_id < 0 or self.rank < 0:
            raise ValueError("device_id and rank must be non-negative")


class CommunicationEventOrchestrator:
    """Deterministic ready/inject/complete state machine.

    The orchestrator owns event lifecycle and dependency ordering.  It does
    not perform packetization or network timing; those belong to the future
    NCCL/fabric/BookSim layers.
    """

    def __init__(self) -> None:
        self._records: dict[str, _EventRecord] = {}
        self._sequence = 0
        self._contexts: dict[int, LogicalGPUContext] = {}

    def register_context(self, context: LogicalGPUContext) -> None:
        existing = self._contexts.get(context.rank)
        if existing is not None and existing != context:
            raise ValueError(f"rank {context.rank} already has a different context")
        self._contexts[context.rank] = context

    def submit(self, event: CommunicationEvent) -> EventState:
        if event.event_id in self._records:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        if event.rank not in self._contexts:
            self.register_context(LogicalGPUContext(event.device_id, event.rank))
        missing = [
            dependency
            for dependency in event.dependency_ids
            if dependency not in self._records
            or self._records[dependency].state != EventState.COMPLETED
        ]
        state = EventState.WAITING if missing else EventState.READY
        self._records[event.event_id] = _EventRecord(event, state, self._sequence)
        self._sequence += 1
        return state

    def ready(self, now_ps: Fraction | int) -> tuple[CommunicationEvent, ...]:
        now = _as_fraction(now_ps)
        return tuple(
            record.event
            for record in sorted(self._records.values(), key=lambda item: item.sequence)
            if record.state == EventState.READY and record.event.ready_time_ps <= now
        )

    def mark_injected(self, event_id: str) -> None:
        record = self._get(event_id)
        if record.state != EventState.READY:
            raise ValueError(f"event {event_id} is not ready: {record.state.value}")
        record.state = EventState.INJECTED

    def complete(self, completion: EventCompletion) -> CommunicationEvent:
        record = self._get(completion.event_id)
        if record.state != EventState.INJECTED:
            raise ValueError(
                f"event {completion.event_id} cannot complete from {record.state.value}"
            )
        record.state = EventState.COMPLETED
        record.completion = completion
        self._release_dependents()
        return record.event

    def cancel(self, event_id: str) -> None:
        record = self._get(event_id)
        if record.state == EventState.COMPLETED:
            raise ValueError(f"event {event_id} is already complete")
        record.state = EventState.CANCELLED

    def state(self, event_id: str) -> EventState:
        return self._get(event_id).state

    def completion(self, event_id: str) -> EventCompletion | None:
        return self._get(event_id).completion

    def snapshot(self) -> dict[str, str]:
        return {
            event_id: record.state.value
            for event_id, record in sorted(self._records.items())
        }

    def _get(self, event_id: str) -> _EventRecord:
        try:
            return self._records[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown event_id: {event_id}") from exc

    def _release_dependents(self) -> None:
        for record in sorted(self._records.values(), key=lambda item: item.sequence):
            if record.state != EventState.WAITING:
                continue
            if all(
                dependency in self._records
                and self._records[dependency].state == EventState.COMPLETED
                for dependency in record.event.dependency_ids
            ):
                record.state = EventState.READY
