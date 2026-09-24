"""P4 configurable NVSwitch topology and reference transport wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from communication_events import CommunicationEvent, EventState
from point_to_point_bridge import PointToPointEventBridge
from transport_contract import ClockDomain


@dataclass(frozen=True)
class NVSwitchRoute:
    src: int
    switch: int
    dst: int

    @property
    def path(self) -> tuple[str, ...]:
        return (f"gpu:{self.src}", f"nvswitch:{self.switch}", f"gpu:{self.dst}")

    @property
    def link_count(self) -> int:
        return len(self.path) - 1


class NVSwitchTopology:
    """A deterministic, explicitly configured GPU-to-switch fabric graph."""

    def __init__(
        self,
        gpu_to_switches: dict[int, tuple[int, ...]] | None = None,
        *,
        gpu_count: int = 8,
        switch_count: int = 4,
    ) -> None:
        if gpu_count < 2 or switch_count < 1:
            raise ValueError("topology must contain at least two GPUs and one switch")
        if gpu_to_switches is None:
            gpu_to_switches = {
                gpu: tuple(range(switch_count)) for gpu in range(gpu_count)
            }
        expected_gpus = set(range(gpu_count))
        if set(gpu_to_switches) != expected_gpus:
            raise ValueError("gpu_to_switches must define every GPU exactly once")
        for gpu, switches in gpu_to_switches.items():
            if not switches or any(switch < 0 or switch >= switch_count for switch in switches):
                raise ValueError(f"GPU {gpu} has an invalid switch attachment")
            if len(set(switches)) != len(switches):
                raise ValueError(f"GPU {gpu} has duplicate switch attachments")
        self.gpu_to_switches = {
            gpu: tuple(sorted(switches)) for gpu, switches in gpu_to_switches.items()
        }
        self.gpu_count = gpu_count
        self.switch_count = switch_count

    def route(self, src: int, dst: int, channel_id: int = 0) -> NVSwitchRoute:
        if src == dst or src not in self.gpu_to_switches or dst not in self.gpu_to_switches:
            raise ValueError("route requires two distinct GPUs in the topology")
        if channel_id < 0:
            raise ValueError("channel_id must be non-negative")
        common = sorted(set(self.gpu_to_switches[src]) & set(self.gpu_to_switches[dst]))
        if not common:
            raise ValueError(f"GPUs {src} and {dst} have no common NVSwitch")
        return NVSwitchRoute(src, common[channel_id % len(common)], dst)


class NVSwitchTransportBridge:
    """Route-checked wrapper around the P2 reference bridge."""

    def __init__(
        self,
        topology: NVSwitchTopology,
        network_clock: ClockDomain,
        *,
        flit_bytes: int = 32,
        protocol_overhead_bytes: int = 0,
        fixed_latency_cycles: int = 0,
    ) -> None:
        self.topology = topology
        self.point_to_point = PointToPointEventBridge(
            network_clock,
            flit_bytes=flit_bytes,
            protocol_overhead_bytes=protocol_overhead_bytes,
            fixed_latency_cycles=fixed_latency_cycles,
        )
        self.orchestrator = self.point_to_point.orchestrator
        self.transport = self.point_to_point.transport
        self._routes: dict[str, NVSwitchRoute] = {}

    def submit(self, event: CommunicationEvent) -> EventState:
        route = self.topology.route(event.src[0], event.dst[0], event.channel_id or 0)
        state = self.point_to_point.submit(event)
        self._routes[event.event_id] = route
        return state

    def pump(self, now_ps: Fraction | int) -> tuple[CommunicationEvent, ...]:
        return self.point_to_point.pump(now_ps)

    def route_for(self, event_id: str) -> NVSwitchRoute:
        try:
            return self._routes[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown route for event {event_id}") from exc

