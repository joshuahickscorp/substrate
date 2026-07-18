"""Unified campaign engine: the declarative node and campaign specifications.

This is the append-only replacement for serial bed-by-bed orchestration. Work is described as a durable
DAG of typed nodes, not a hardcoded stage list. A node becomes runnable when its declared dependencies and
authorities are satisfied; the scheduler launches the whole runnable frontier under one global resource
broker (see :mod:`mop.campaign.broker`).

The spec objects the mandate names are all here: ``CampaignSpec``, ``ResearchQuestionSpec``, ``BedSpec``,
``ArmSpec``, ``ReproductionSpec``, ``VerificationSpec``, ``ResourceRequest``, ``Dependency``, and
``DecisionRule``. They are pure data (frozen dataclasses) with canonical serialization and digests, so a
campaign manifest is byte-reproducible and a node's identity is stable across restarts.

Nodes are genuinely executable: each carries an ``entrypoint`` of the form ``"module:function"`` resolved
by :mod:`mop.campaign.runners`, so the scheduler runs real code in a worker process and seals a real
artifact. Nothing here reads a wall clock or an unsealed sibling result.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mop.substrate.events import canonical_sha256

CAMPAIGN_SPEC_SCHEMA = "mop-campaign-spec/v1"


class SpecError(ValueError):
    """Raised when a campaign specification violates the engine contract."""


# ---------------------------------------------------------------------------
# Resource classes and requests (consumed by the global broker).
# ---------------------------------------------------------------------------


class ResourceClass(StrEnum):
    """Workload-specific resource classes; the broker weights concurrency per class from measured
    throughput, not a single decorative worker count."""

    CPU_HASH_HEAVY = "cpu_hash_heavy"  # the existing seeded-hash mechanics workload (measured optimum ~16-20)
    CPU_LIGHT = "cpu_light"  # short deterministic analysis/verification nodes
    NATIVE_THREADED = "native_threaded"  # a node that internally uses BLAS/OMP threads
    MEMORY_HEAVY = "memory_heavy"  # large cache or large-array nodes
    IO_HEAVY = "io_heavy"  # intake / disk-bound nodes
    EXCLUSIVE = "exclusive"  # must not overlap any other campaign node


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What one node needs from the global broker. All fields are advisory maxima the broker may throttle."""

    resource_class: ResourceClass = ResourceClass.CPU_LIGHT
    cpu_slots: int = 1
    native_threads: int = 1
    mem_gb: float = 0.5
    disk_write_gb: float = 0.1
    io_weight: float = 0.0
    exclusive: bool = False
    est_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.cpu_slots < 1:
            raise SpecError("cpu_slots must be at least 1")
        if self.native_threads < 1:
            raise SpecError("native_threads must be at least 1")
        if self.mem_gb <= 0 or self.disk_write_gb < 0:
            raise SpecError("mem_gb must be positive and disk_write_gb nonnegative")

    def payload(self) -> dict[str, Any]:
        return {
            "resource_class": self.resource_class.value,
            "cpu_slots": self.cpu_slots,
            "native_threads": self.native_threads,
            "mem_gb": round(float(self.mem_gb), 6),
            "disk_write_gb": round(float(self.disk_write_gb), 6),
            "io_weight": round(float(self.io_weight), 6),
            "exclusive": bool(self.exclusive),
            "est_seconds": round(float(self.est_seconds), 3),
        }


# ---------------------------------------------------------------------------
# Dependencies and authorities.
# ---------------------------------------------------------------------------


class DependencyKind(StrEnum):
    COMPLETION = "completion"  # the other node reached a terminal (sealed or null-sealed) state
    SEAL = "seal"  # the other node produced a byte-sealed artifact this node reads
    AUTHORITY = "authority"  # a named sealed authority (may be a node id or an external boundary)
    EXTERNAL_BOUNDARY = "external_boundary"  # an external live process/authority must clear first


@dataclass(frozen=True, slots=True)
class Dependency:
    """A real dependency, preserved only where a later decision truly needs the earlier sealed result."""

    node_id: str
    kind: DependencyKind = DependencyKind.COMPLETION

    def payload(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "kind": self.kind.value}


# ---------------------------------------------------------------------------
# Precommitted decision rules (conditional branch queuing).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionRule:
    """A precommitted rule: after this node seals, if ``when`` matches its result, enqueue ``enqueue``.

    ``when`` is a small declarative predicate over the sealed artifact evaluated by
    :func:`mop.campaign.decisions.evaluate`, so criteria are fixed before results are read. It never mutates
    the criteria; it only selects which already-declared child nodes become eligible.
    """

    name: str
    when: dict[str, Any]  # e.g. {"field": "verdict", "equals": "survives"}
    enqueue: tuple[str, ...]  # node ids to make eligible
    note: str = ""

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "when": self.when, "enqueue": list(self.enqueue), "note": self.note}


# ---------------------------------------------------------------------------
# Coverage tags (the coverage-aware scheduler reads these).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Coverage:
    """The breadth coordinates a node advances, so scheduling can actively improve coverage."""

    form_family: str = "none"  # e.g. vision, native_audio, audiovisual, symbolic, action, memory_episode
    phenomenon: str = "none"  # e.g. event_formation, persistence, prediction, plasticity, value_of_compute
    mechanism_family: str = "none"  # e.g. value_of_computation, structured_state, replay, routing
    unit_class: str = "none"  # independent experimental unit: room, session, world, partner, seed, clip
    evidence_level: str = "M0"  # M0..M7 replication level this node can reach at most

    def payload(self) -> dict[str, Any]:
        return {
            "form_family": self.form_family,
            "phenomenon": self.phenomenon,
            "mechanism_family": self.mechanism_family,
            "unit_class": self.unit_class,
            "evidence_level": self.evidence_level,
        }


# ---------------------------------------------------------------------------
# The node: the unit the scheduler runs.
# ---------------------------------------------------------------------------


class NodeKind(StrEnum):
    QUESTION = "research_question"
    BED = "bed"
    ARM = "arm"
    REPRODUCTION = "reproduction"
    VERIFICATION = "verification"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    READINESS = "readiness"
    INTAKE = "intake"
    INVARIANCE = "invariance"


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """One runnable (or contracted-blocked) node in the campaign DAG."""

    node_id: str
    kind: NodeKind
    title: str
    entrypoint: str  # "module:function"; resolved by mop.campaign.runners
    params: dict[str, Any] = field(default_factory=dict)
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    dependencies: tuple[Dependency, ...] = ()
    authorities: tuple[str, ...] = ()  # named sealed authorities that must be satisfied
    verifier: str | None = None  # node id of an independent verifier node (structurally separate)
    decision_rules: tuple[DecisionRule, ...] = ()
    coverage: Coverage = field(default_factory=Coverage)
    null_safe: bool = True  # a null seals the node; it is never re-run to chase a positive
    blocked_reason: str = ""  # nonempty => contracted-but-blocked (e.g. named external data input)
    priority: int = 100  # lower runs earlier within the eligible frontier before coverage adjustment

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise SpecError("node_id must be a nonempty string")
        if ":" not in self.entrypoint:
            raise SpecError(f"entrypoint must be 'module:function', got {self.entrypoint!r}")

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_reason.strip())

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "title": self.title,
            "entrypoint": self.entrypoint,
            "params": self.params,
            "resources": self.resources.payload(),
            "dependencies": [d.payload() for d in self.dependencies],
            "authorities": list(self.authorities),
            "verifier": self.verifier,
            "decision_rules": [r.payload() for r in self.decision_rules],
            "coverage": self.coverage.payload(),
            "null_safe": self.null_safe,
            "blocked_reason": self.blocked_reason,
            "priority": self.priority,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Named specialized node builders (thin, so a question is a declarative spec).
# ---------------------------------------------------------------------------


def _node(kind: NodeKind, **kw: Any) -> NodeSpec:
    return NodeSpec(kind=kind, **kw)


def ResearchQuestionSpec(**kw: Any) -> NodeSpec:  # noqa: N802 (spec-factory names mirror the mandate)
    """A research question: an estimand that changes what is being asked. See mandate file 01."""

    return _node(NodeKind.QUESTION, **kw)


def BedSpec(**kw: Any) -> NodeSpec:  # noqa: N802
    return _node(NodeKind.BED, **kw)


def ArmSpec(**kw: Any) -> NodeSpec:  # noqa: N802
    return _node(NodeKind.ARM, **kw)


def ReproductionSpec(**kw: Any) -> NodeSpec:  # noqa: N802
    return _node(NodeKind.REPRODUCTION, **kw)


def VerificationSpec(**kw: Any) -> NodeSpec:  # noqa: N802
    return _node(NodeKind.VERIFICATION, **kw)


# ---------------------------------------------------------------------------
# External dependencies (the live campaign, adopted, not disturbed).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExternalDependency:
    """A live process/authority this campaign observes but does not own. Represents the running General Run
    and the horizon successor chain as external resource consumers and as authority boundaries."""

    name: str
    kind: str  # "live_process" | "sealed_authority"
    match_label: str = ""  # process label substring (observe-only) if a live process
    authority_path: str = ""  # a sealed terminal artifact path if a sealed authority
    est_cpu_workers: int = 0  # measured/observed CPU consumption to subtract from the broker budget
    note: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "match_label": self.match_label,
            "authority_path": self.authority_path,
            "est_cpu_workers": self.est_cpu_workers,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# The campaign.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """A durable campaign manifest: many questions, beds, arms, reproductions, verifiers, and follow-ups in
    one DAG, plus the external live dependencies it adopts and the coverage targets it drives toward."""

    campaign_id: str
    title: str
    nodes: tuple[NodeSpec, ...]
    external_dependencies: tuple[ExternalDependency, ...] = ()
    coverage_targets: dict[str, int] = field(
        default_factory=dict
    )  # e.g. {"form_families": 6, "phenomena": 10}
    schema: str = CAMPAIGN_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if not self.campaign_id.strip():
            raise SpecError("campaign_id must be a nonempty string")
        seen: set[str] = set()
        ids = {n.node_id for n in self.nodes}
        for node in self.nodes:
            if node.node_id in seen:
                raise SpecError(f"duplicate node_id {node.node_id!r}")
            seen.add(node.node_id)
        # every internal dependency and verifier and enqueue must resolve to a declared node
        for node in self.nodes:
            for dep in node.dependencies:
                if dep.kind in (DependencyKind.COMPLETION, DependencyKind.SEAL) and dep.node_id not in ids:
                    raise SpecError(f"node {node.node_id} depends on unknown node {dep.node_id}")
            if node.verifier is not None and node.verifier not in ids:
                raise SpecError(f"node {node.node_id} names unknown verifier {node.verifier}")
            for rule in node.decision_rules:
                for target in rule.enqueue:
                    if target not in ids:
                        raise SpecError(f"node {node.node_id} rule {rule.name} enqueues unknown {target}")

    def node(self, node_id: str) -> NodeSpec:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise SpecError(f"no node {node_id!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "title": self.title,
            "n_nodes": len(self.nodes),
            "nodes": [n.payload() for n in self.nodes],
            "external_dependencies": [e.payload() for e in self.external_dependencies],
            "coverage_targets": dict(self.coverage_targets),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())
