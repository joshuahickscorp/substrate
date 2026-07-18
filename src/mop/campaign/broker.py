"""Unified campaign engine: the one global resource broker.

Every node's :class:`~mop.campaign.specs.ResourceRequest` is admitted (or throttled) by ONE broker, so
independent beds do not each spin up an uncoordinated maximum pool. The broker reuses the reviewed
``dynamic_worker_controller`` for the CPU worker ceiling and Hawking behavior (the measured 20-worker
seeded-hash optimum, shed-to-reserve under Hawking, the priority lever) and adds workload-specific
multi-class accounting on top: CPU-hash-heavy, CPU-light, native-threaded, memory-heavy, IO-heavy, and
exclusive nodes each consume the budget differently.

The live General Run and horizon successor chain are represented as external consumers: their CPU is
already reflected in the controller's short live CPU sample, so the recommended worker count this broker
grants is the campaign's OWN slice only. When Hawking appears the controller sheds the count and lowers
priority automatically, so this campaign yields its own resources and never touches Hawking.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mop.studio.dynamic_worker_controller import (
    DEFAULT_POLICY,
    HAWKING_COMMAND_MARKERS,
    WorkerPolicy,
    sample_host_state,
)

from .specs import ResourceClass, ResourceRequest


@dataclass(frozen=True, slots=True)
class Grant:
    """A broker admission: the slots and priority this node runs under."""

    node_id: str
    cpu_slots: int
    nice_level: int
    yield_to_hawking: bool

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cpu_slots": self.cpu_slots,
            "nice_level": self.nice_level,
            "yield_to_hawking": self.yield_to_hawking,
        }


@dataclass
class BrokerSnapshot:
    """The broker's current view of the machine and its own budget, for the operator status view."""

    cpu_budget: int
    cpu_in_use: int
    mem_available_gb: float
    mem_in_use_gb: float
    hawking_active: bool
    nice_level: int
    external_consumers: int
    recommended_workers: int
    binding_constraint: str
    created_at: float

    def payload(self) -> dict[str, Any]:
        return {
            "cpu_budget": self.cpu_budget,
            "cpu_in_use": self.cpu_in_use,
            "cpu_free": max(0, self.cpu_budget - self.cpu_in_use),
            "mem_available_gb": round(self.mem_available_gb, 3),
            "mem_in_use_gb": round(self.mem_in_use_gb, 3),
            "hawking_active": self.hawking_active,
            "nice_level": self.nice_level,
            "external_consumers": self.external_consumers,
            "recommended_workers": self.recommended_workers,
            "binding_constraint": self.binding_constraint,
            "mode": "hawking_yield" if self.hawking_active else "greedy",
        }


def _cpu_cost(request: ResourceRequest) -> int:
    """Slots a request consumes from the CPU budget. Native-threaded work reserves its thread fan-out."""

    if request.resource_class is ResourceClass.NATIVE_THREADED:
        return max(request.cpu_slots, request.native_threads)
    return request.cpu_slots


class ResourceBroker:
    """One global broker. Sample the host, then admit nodes against the campaign's own budget."""

    def __init__(
        self,
        policy: WorkerPolicy = DEFAULT_POLICY,
        external_labels: Sequence[str] = (),
        sampler: Callable[..., Any] = sample_host_state,
    ) -> None:
        self.policy = policy
        self.external_labels = tuple(external_labels)
        self._sampler = sampler
        self._snapshot: BrokerSnapshot | None = None
        self._own_pids: set[int] = set()

    def set_own_pids(self, pids: set[int]) -> None:
        """PIDs of this campaign's own workers, excluded from the host free-core/Hawking scan."""

        self._own_pids = set(pids)

    def sample(self, current_workers: int = 0) -> BrokerSnapshot:
        sample = self._sampler(
            current_workers=current_workers, policy=self.policy, exclude_pids=self._own_pids
        )
        external = self._count_external_consumers()
        snap = BrokerSnapshot(
            cpu_budget=int(sample.recommended_workers),
            cpu_in_use=0,
            mem_available_gb=float(sample.telemetry["mem_available_gb"]),
            mem_in_use_gb=0.0,
            hawking_active=bool(sample.state.hawking_active),
            nice_level=int(sample.priority.nice_level),
            external_consumers=external,
            recommended_workers=int(sample.recommended_workers),
            binding_constraint=str(sample.bounds["binding_constraint"]),
            created_at=time.time(),
        )
        self._snapshot = snap
        return snap

    def _count_external_consumers(self) -> int:
        """Best-effort count of live non-owned MOP campaign workers, so the operator view shows what this
        broker is coexisting with. The controller's CPU sample already prices their load into the budget."""

        try:
            import psutil  # type: ignore
        except Exception:
            return 0
        markers = tuple(self.external_labels) + (
            "mop-final-mechanic",
            "mop-g1-",
            "general-run",
            "mop-supervisor",
        )
        count = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] in self._own_pids:
                    continue
                hay = " ".join([proc.info.get("name") or "", " ".join(proc.info.get("cmdline") or [])])
                if any(m in hay for m in markers) and not any(m in hay for m in HAWKING_COMMAND_MARKERS):
                    count += 1
            except Exception:
                continue
        return count

    def admit(
        self, request: ResourceRequest, node_id: str, running: Sequence[ResourceRequest]
    ) -> Grant | None:
        """Admit ``request`` given what is already running, or return None with the budget unchanged."""

        if self._snapshot is None:
            self.sample()
        snap = self._snapshot
        assert snap is not None

        if any(r.exclusive for r in running):
            return None  # an exclusive node holds the machine
        if request.exclusive:
            # an exclusive node takes the machine among campaign nodes; admit only on a clear campaign
            # fleet. It self-limits its own internal widths, so it is not gated on the shared cpu budget.
            if running:
                return None
            return Grant(node_id=node_id, cpu_slots=_cpu_cost(request), nice_level=snap.nice_level,
                         yield_to_hawking=snap.hawking_active)

        cpu_used = sum(_cpu_cost(r) for r in running)
        mem_used = sum(r.mem_gb for r in running)
        need = _cpu_cost(request)
        # always admit at least one node even when the coexistence budget has shed to a tiny reserve, so a
        # heavily loaded host makes serial progress rather than deadlocking; extra nodes wait for the budget.
        if running and cpu_used + need > snap.cpu_budget:
            return None
        if mem_used + request.mem_gb > snap.mem_available_gb:
            return None
        return Grant(
            node_id=node_id, cpu_slots=need, nice_level=snap.nice_level, yield_to_hawking=snap.hawking_active
        )

    def snapshot(self) -> BrokerSnapshot | None:
        return self._snapshot
