
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from typing import Any, Self, cast

from mop.substrate.events import (
    BranchRef,
    EventRef,
    FrozenJSON,
    canonical_bytes,
    canonical_sha256,
)

from .accounting import FACTUAL_BRANCH, WorkVector
from .events import (
    CommitmentEvent,
    ConsequenceEvent,
    EpistemicStatus,
    ESCSEvent,
    EvidenceClass,
    HypothesisEvent,
    ObservationEvent,
    state_version_for_parents,
)
from .ledger import EventLedger

FACTOR_FRONTIER_CAPS_SCHEMA = "mop-escs-factor-frontier-caps/v1"
FACTOR_NODE_SCHEMA = "mop-escs-shadow-factor-node/v1"
FACTOR_FRONTIER_SCHEMA = "mop-escs-shadow-factor-frontier/v1"
FACTOR_PROJECTION_RECEIPT_SCHEMA = "mop-escs-factor-frontier-projection-receipt/v1"
FACTOR_QUERY_SCHEMA = "mop-escs-factor-frontier-query/v1"
FACTOR_QUERY_RECEIPT_SCHEMA = "mop-escs-factor-frontier-query-receipt/v1"
FACTOR_INVALIDATION_PLAN_SCHEMA = "mop-escs-factor-frontier-invalidation-plan/v1"

ACTIVATION_ENABLED = False
RUNTIME_CONSUMABLE = False
FACTUAL_WRITE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

MAX_HARD_EVENTS = 2_048
MAX_HARD_NODES = 1_024
MAX_HARD_EDGES = 65_536
MAX_HARD_KEYS_PER_NODE = 64
MAX_HARD_INDEX_POSTINGS = 131_072
MAX_HARD_QUERY_RESULTS = 1_024
MAX_HARD_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_HARD_SOURCE_LEDGER_BYTES = 64 * 1024 * 1024
MAX_HARD_EVENT_ENCODED_BYTES = 2 * 1024 * 1024
MAX_HARD_OPAQUE_PAYLOAD_BYTES = 1024 * 1024
MAX_HARD_WORK_UNITS = 16 * 1024 * 1024
MAX_HARD_TEXT_CHARS = 1_024

IndexRows = tuple[tuple[str, tuple[str, ...]], ...]

_PROJECTION_RECEIPT_ISSUANCE_TOKEN = object()
_QUERY_RECEIPT_ISSUANCE_TOKEN = object()
_INVALIDATION_PLAN_ISSUANCE_TOKEN = object()


class FactorFrontierError(ValueError):
    pass


class FactorFrontierCapExceeded(FactorFrontierError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FactorFrontierError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    actual = set(value)
    _require(
        actual == expected,
        f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FactorFrontierError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise FactorFrontierError(f"{label} must be positive")
    return result


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FactorFrontierError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise FactorFrontierError(f"{label} must be nonempty text")
    if len(value) > MAX_HARD_TEXT_CHARS:
        raise FactorFrontierCapExceeded(f"{label} exceeds the hard text bound")
    return value


def _require_false(value: object, label: str) -> None:
    if not isinstance(value, bool) or value:
        raise FactorFrontierError(f"{label} must be the boolean false")


def _require_true(value: object, label: str) -> None:
    if not isinstance(value, bool) or not value:
        raise FactorFrontierError(f"{label} must be the boolean true")


def _canonical_texts(
    values: object,
    label: str,
    *,
    refs: type[EventRef] | type[BranchRef] | None = None,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise FactorFrontierError(f"{label} must be an immutable tuple")
    for value in values:
        _text(value, label)
        if refs is not None:
            refs(value)
    if len(set(values)) != len(values):
        raise FactorFrontierError(f"{label} must contain unique values")
    if values != tuple(sorted(values)):
        raise FactorFrontierError(f"{label} must use canonical sorted order")
    return values


def _bounded_sequence(
    value: object,
    label: str,
    *,
    max_items: int,
) -> tuple[Any, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, tuple | list):
        raise FactorFrontierError(f"{label} must be a finite tuple or list")
    if len(value) > max_items:
        raise FactorFrontierCapExceeded(f"{label} exceeds its pre-scan item cap")
    return tuple(value)


def _string_list(
    value: object,
    label: str,
    *,
    max_items: int = MAX_HARD_EDGES,
) -> tuple[str, ...]:
    bounded = _bounded_sequence(value, label, max_items=max_items)
    if not all(isinstance(row, str) for row in bounded):
        raise FactorFrontierError(f"{label} must be a string list")
    return tuple(row for row in bounded if isinstance(row, str))


def _index_payload(rows: IndexRows) -> list[dict[str, Any]]:
    return [{"key": key, "factor_ids": list(factor_ids)} for key, factor_ids in rows]


def _index_from_payload(value: object, label: str) -> IndexRows:
    if not isinstance(value, list):
        raise FactorFrontierError(f"{label} must be a list")
    if len(value) > MAX_HARD_INDEX_POSTINGS:
        raise FactorFrontierCapExceeded(f"{label} row count exceeds the hard bound")
    rows: list[tuple[str, tuple[str, ...]]] = []
    postings = 0
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise FactorFrontierError(f"{label}[{index}] must be a mapping")
        _exact_keys(row, {"key", "factor_ids"}, f"{label}[{index}]")
        key = _text(row["key"], f"{label}[{index}].key")
        factor_ids = _string_list(
            row["factor_ids"],
            f"{label}[{index}].factor_ids",
            max_items=MAX_HARD_NODES,
        )
        _canonical_texts(factor_ids, f"{label}[{index}].factor_ids")
        for factor_id in factor_ids:
            _digest(factor_id, f"{label}[{index}].factor_id")
        postings += len(factor_ids)
        if postings > MAX_HARD_INDEX_POSTINGS:
            raise FactorFrontierCapExceeded(f"{label} postings exceed the hard bound")
        rows.append((key, factor_ids))
    result = tuple(rows)
    keys = tuple(key for key, _ in result)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise FactorFrontierError(f"{label} keys must be unique and canonically sorted")
    return result


@dataclass(frozen=True, slots=True)
class FactorFrontierCaps:

    max_events: int = 512
    max_nodes: int = 512
    max_edges: int = 16_384
    max_referents_per_node: int = 64
    max_scopes_per_node: int = 64
    max_index_postings: int = 32_768
    max_query_results: int = 128
    max_snapshot_bytes: int = 8 * 1024 * 1024
    max_source_ledger_bytes: int = 16 * 1024 * 1024
    max_event_encoded_bytes: int = 1024 * 1024
    max_opaque_payload_bytes: int = 512 * 1024
    max_work_units: int = 4 * 1024 * 1024
    schema: str = FACTOR_FRONTIER_CAPS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FACTOR_FRONTIER_CAPS_SCHEMA:
            raise FactorFrontierError(f"unsupported factor-frontier caps schema {self.schema!r}")
        hard = {
            "max_events": MAX_HARD_EVENTS,
            "max_nodes": MAX_HARD_NODES,
            "max_edges": MAX_HARD_EDGES,
            "max_referents_per_node": MAX_HARD_KEYS_PER_NODE,
            "max_scopes_per_node": MAX_HARD_KEYS_PER_NODE,
            "max_index_postings": MAX_HARD_INDEX_POSTINGS,
            "max_query_results": MAX_HARD_QUERY_RESULTS,
            "max_snapshot_bytes": MAX_HARD_SNAPSHOT_BYTES,
            "max_source_ledger_bytes": MAX_HARD_SOURCE_LEDGER_BYTES,
            "max_event_encoded_bytes": MAX_HARD_EVENT_ENCODED_BYTES,
            "max_opaque_payload_bytes": MAX_HARD_OPAQUE_PAYLOAD_BYTES,
            "max_work_units": MAX_HARD_WORK_UNITS,
        }
        for name, ceiling in hard.items():
            value = _positive_int(getattr(self, name), f"FactorFrontierCaps.{name}")
            if value > ceiling:
                raise FactorFrontierCapExceeded(f"FactorFrontierCaps.{name} exceeds its hard ceiling")
        if self.max_query_results > self.max_nodes:
            raise FactorFrontierError("query-result cap cannot exceed the node cap")
        if self.max_opaque_payload_bytes > self.max_event_encoded_bytes:
            raise FactorFrontierError("opaque-payload cap cannot exceed the encoded-event cap")
        if self.max_event_encoded_bytes > self.max_source_ledger_bytes:
            raise FactorFrontierError("encoded-event cap cannot exceed the source-ledger cap")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        expected = {
            "schema",
            "max_events",
            "max_nodes",
            "max_edges",
            "max_referents_per_node",
            "max_scopes_per_node",
            "max_index_postings",
            "max_query_results",
            "max_snapshot_bytes",
            "max_source_ledger_bytes",
            "max_event_encoded_bytes",
            "max_opaque_payload_bytes",
            "max_work_units",
        }
        _exact_keys(payload, expected, "FactorFrontierCaps")
        return cls(**dict(payload))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "max_events": self.max_events,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_referents_per_node": self.max_referents_per_node,
            "max_scopes_per_node": self.max_scopes_per_node,
            "max_index_postings": self.max_index_postings,
            "max_query_results": self.max_query_results,
            "max_snapshot_bytes": self.max_snapshot_bytes,
            "max_source_ledger_bytes": self.max_source_ledger_bytes,
            "max_event_encoded_bytes": self.max_event_encoded_bytes,
            "max_opaque_payload_bytes": self.max_opaque_payload_bytes,
            "max_work_units": self.max_work_units,
        }


@dataclass(frozen=True, slots=True)
class ShadowFactorNode:

    factor_id: str
    source_hypothesis_event_id: str
    branch_id: str
    causal_event_ids: tuple[str, ...]
    parent_factor_ids: tuple[str, ...]
    supersedes_factor_ids: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    referent_hypotheses: tuple[str, ...]
    factor_scopes: tuple[str, ...]
    epistemic_status: EpistemicStatus
    evidence_class: EvidenceClass
    source_payload_digest: str
    producer_state_version: str
    clock_start_tick: int
    clock_end_tick: int
    clock_uncertainty: int
    schema: str = FACTOR_NODE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FACTOR_NODE_SCHEMA:
            raise FactorFrontierError(f"unsupported factor-node schema {self.schema!r}")
        _digest(self.factor_id, "factor_id")
        _text(self.source_hypothesis_event_id, "source_hypothesis_event_id")
        EventRef(self.source_hypothesis_event_id)
        _text(self.branch_id, "branch_id")
        BranchRef(self.branch_id)
        edge_sequences = (
            self.causal_event_ids,
            self.parent_factor_ids,
            self.supersedes_factor_ids,
            self.supporting_event_ids,
        )
        if (
            any(len(values) > MAX_HARD_EDGES for values in edge_sequences)
            or sum(len(values) for values in edge_sequences) > MAX_HARD_EDGES
        ):
            raise FactorFrontierCapExceeded("factor-node edges exceed the hard pre-scan cap")
        if len(self.referent_hypotheses) > MAX_HARD_KEYS_PER_NODE:
            raise FactorFrontierCapExceeded("factor-node referents exceed the hard per-node cap")
        if len(self.factor_scopes) > MAX_HARD_KEYS_PER_NODE:
            raise FactorFrontierCapExceeded("factor-node scopes exceed the hard per-node cap")
        _canonical_texts(self.causal_event_ids, "causal_event_ids", refs=EventRef)
        _canonical_texts(self.parent_factor_ids, "parent_factor_ids")
        _canonical_texts(self.supersedes_factor_ids, "supersedes_factor_ids")
        _canonical_texts(self.supporting_event_ids, "supporting_event_ids", refs=EventRef)
        _canonical_texts(self.referent_hypotheses, "referent_hypotheses")
        _canonical_texts(self.factor_scopes, "factor_scopes")
        for label, values in (
            ("parent_factor_ids", self.parent_factor_ids),
            ("supersedes_factor_ids", self.supersedes_factor_ids),
        ):
            for value in values:
                _digest(value, label)
        if not set(self.supporting_event_ids) <= set(self.causal_event_ids):
            raise FactorFrontierError("supporting events must be direct causal events")
        if not set(self.supersedes_factor_ids) <= set(self.parent_factor_ids):
            raise FactorFrontierError("superseded factors must also be factor parents")
        if self.factor_id in set(self.parent_factor_ids) | set(self.supersedes_factor_ids):
            raise FactorFrontierError("a factor node cannot depend on or supersede itself")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise FactorFrontierError("factor epistemic_status must be typed")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise FactorFrontierError("factor evidence_class must be typed")
        factual = str(FACTUAL_BRANCH)
        if self.epistemic_status is EpistemicStatus.SIMULATED:
            if self.branch_id == factual:
                raise FactorFrontierError("simulated factor nodes require a counterfactual branch")
        elif self.branch_id != factual:
            raise FactorFrontierError("non-simulated factor nodes must remain factual")
        _digest(self.source_payload_digest, "source_payload_digest")
        _digest(self.producer_state_version, "producer_state_version")
        expected_state_version = state_version_for_parents(
            tuple(EventRef(event_id) for event_id in self.causal_event_ids)
        )
        if self.producer_state_version != expected_state_version:
            raise FactorFrontierError("factor producer state does not bind its causal event IDs")
        _nonnegative_int(self.clock_start_tick, "clock_start_tick")
        _nonnegative_int(self.clock_end_tick, "clock_end_tick")
        _nonnegative_int(self.clock_uncertainty, "clock_uncertainty")
        if self.clock_end_tick < self.clock_start_tick:
            raise FactorFrontierError("factor-node clock interval is invalid")
        if canonical_sha256(self.payload(include_factor_id=False)) != self.factor_id:
            raise FactorFrontierError("factor-node self-hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        source_hypothesis_event_id: str,
        branch_id: str,
        causal_event_ids: Sequence[str],
        parent_factor_ids: Sequence[str],
        supersedes_factor_ids: Sequence[str],
        supporting_event_ids: Sequence[str],
        referent_hypotheses: Sequence[str],
        factor_scopes: Sequence[str],
        epistemic_status: EpistemicStatus,
        evidence_class: EvidenceClass,
        source_payload_digest: str,
        producer_state_version: str,
        clock_start_tick: int,
        clock_end_tick: int,
        clock_uncertainty: int,
    ) -> Self:
        causal = _string_list(causal_event_ids, "causal_event_ids")
        parents = _string_list(parent_factor_ids, "parent_factor_ids")
        supersedes = _string_list(supersedes_factor_ids, "supersedes_factor_ids")
        supporting = _string_list(supporting_event_ids, "supporting_event_ids")
        if sum(len(values) for values in (causal, parents, supersedes, supporting)) > MAX_HARD_EDGES:
            raise FactorFrontierCapExceeded("factor-node edges exceed the hard pre-sort cap")
        referents = _string_list(
            referent_hypotheses,
            "referent_hypotheses",
            max_items=MAX_HARD_KEYS_PER_NODE,
        )
        scopes = _string_list(
            factor_scopes,
            "factor_scopes",
            max_items=MAX_HARD_KEYS_PER_NODE,
        )
        core: dict[str, Any] = {
            "schema": FACTOR_NODE_SCHEMA,
            "source_hypothesis_event_id": source_hypothesis_event_id,
            "branch_id": branch_id,
            "causal_event_ids": sorted(causal),
            "parent_factor_ids": sorted(parents),
            "supersedes_factor_ids": sorted(supersedes),
            "supporting_event_ids": sorted(supporting),
            "referent_hypotheses": sorted(referents),
            "factor_scopes": sorted(scopes),
            "epistemic_status": epistemic_status.value,
            "evidence_class": evidence_class.value,
            "source_payload_digest": source_payload_digest,
            "producer_state_version": producer_state_version,
            "clock_start_tick": clock_start_tick,
            "clock_end_tick": clock_end_tick,
            "clock_uncertainty": clock_uncertainty,
        }
        return cls(
            factor_id=canonical_sha256(core),
            source_hypothesis_event_id=source_hypothesis_event_id,
            branch_id=branch_id,
            causal_event_ids=tuple(sorted(causal)),
            parent_factor_ids=tuple(sorted(parents)),
            supersedes_factor_ids=tuple(sorted(supersedes)),
            supporting_event_ids=tuple(sorted(supporting)),
            referent_hypotheses=tuple(sorted(referents)),
            factor_scopes=tuple(sorted(scopes)),
            epistemic_status=epistemic_status,
            evidence_class=evidence_class,
            source_payload_digest=source_payload_digest,
            producer_state_version=producer_state_version,
            clock_start_tick=clock_start_tick,
            clock_end_tick=clock_end_tick,
            clock_uncertainty=clock_uncertainty,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        expected = {
            "schema",
            "factor_id",
            "source_hypothesis_event_id",
            "branch_id",
            "causal_event_ids",
            "parent_factor_ids",
            "supersedes_factor_ids",
            "supporting_event_ids",
            "referent_hypotheses",
            "factor_scopes",
            "epistemic_status",
            "evidence_class",
            "source_payload_digest",
            "producer_state_version",
            "clock_start_tick",
            "clock_end_tick",
            "clock_uncertainty",
        }
        _exact_keys(payload, expected, "ShadowFactorNode")
        return cls(
            schema=payload["schema"],
            factor_id=payload["factor_id"],
            source_hypothesis_event_id=payload["source_hypothesis_event_id"],
            branch_id=payload["branch_id"],
            causal_event_ids=_string_list(payload["causal_event_ids"], "causal_event_ids"),
            parent_factor_ids=_string_list(payload["parent_factor_ids"], "parent_factor_ids"),
            supersedes_factor_ids=_string_list(payload["supersedes_factor_ids"], "supersedes_factor_ids"),
            supporting_event_ids=_string_list(payload["supporting_event_ids"], "supporting_event_ids"),
            referent_hypotheses=_string_list(
                payload["referent_hypotheses"],
                "referent_hypotheses",
                max_items=MAX_HARD_KEYS_PER_NODE,
            ),
            factor_scopes=_string_list(
                payload["factor_scopes"],
                "factor_scopes",
                max_items=MAX_HARD_KEYS_PER_NODE,
            ),
            epistemic_status=EpistemicStatus(payload["epistemic_status"]),
            evidence_class=EvidenceClass(payload["evidence_class"]),
            source_payload_digest=payload["source_payload_digest"],
            producer_state_version=payload["producer_state_version"],
            clock_start_tick=payload["clock_start_tick"],
            clock_end_tick=payload["clock_end_tick"],
            clock_uncertainty=payload["clock_uncertainty"],
        )

    def payload(self, *, include_factor_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "source_hypothesis_event_id": self.source_hypothesis_event_id,
            "branch_id": self.branch_id,
            "causal_event_ids": list(self.causal_event_ids),
            "parent_factor_ids": list(self.parent_factor_ids),
            "supersedes_factor_ids": list(self.supersedes_factor_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "referent_hypotheses": list(self.referent_hypotheses),
            "factor_scopes": list(self.factor_scopes),
            "epistemic_status": self.epistemic_status.value,
            "evidence_class": self.evidence_class.value,
            "source_payload_digest": self.source_payload_digest,
            "producer_state_version": self.producer_state_version,
            "clock_start_tick": self.clock_start_tick,
            "clock_end_tick": self.clock_end_tick,
            "clock_uncertainty": self.clock_uncertainty,
        }
        if include_factor_id:
            result["factor_id"] = self.factor_id
        return result


def _build_index(nodes: Sequence[ShadowFactorNode], attribute: str) -> IndexRows:
    postings: dict[str, set[str]] = {}
    for node in nodes:
        values = getattr(node, attribute)
        for value in values:
            postings.setdefault(value, set()).add(node.factor_id)
    return tuple((key, tuple(sorted(factor_ids))) for key, factor_ids in sorted(postings.items()))


def _edge_count(nodes: Sequence[ShadowFactorNode]) -> int:
    return sum(
        len(node.causal_event_ids)
        + len(node.parent_factor_ids)
        + len(node.supersedes_factor_ids)
        + len(node.supporting_event_ids)
        for node in nodes
    )


def _source_event_edge_count(ledger: EventLedger) -> int:
    return sum(len(event.envelope.causal_parent_ids) for event in ledger.events)


def _index_posting_count(rows: IndexRows) -> int:
    return sum(len(factor_ids) for _, factor_ids in rows)


def _validate_factor_dag(nodes: Sequence[ShadowFactorNode]) -> None:

    by_id = {node.factor_id: node for node in nodes}
    indegree = {node.factor_id: len(node.parent_factor_ids) for node in nodes}
    children: dict[str, list[str]] = {node.factor_id: [] for node in nodes}
    for node in nodes:
        for parent_id in node.parent_factor_ids:
            if parent_id not in by_id:
                raise FactorFrontierError(f"factor {node.factor_id} has an unknown factor parent")
            children[parent_id].append(node.factor_id)
            parent = by_id[parent_id]
            if parent.branch_id != node.branch_id and (
                node.branch_id == str(FACTUAL_BRANCH) or parent.branch_id != str(FACTUAL_BRANCH)
            ):
                raise FactorFrontierError("factor-parent edge crosses branches illegally")
            if node.evidence_class.taint_rank < parent.evidence_class.taint_rank:
                raise FactorFrontierError("factor node downgraded parent evidence taint")
            if parent.clock_end_tick > node.clock_start_tick:
                raise FactorFrontierError("factor child begins before its parent factor ends")
        for superseded_id in node.supersedes_factor_ids:
            target = by_id.get(superseded_id)
            if target is None or target.branch_id != node.branch_id:
                raise FactorFrontierError("factor supersession crosses a branch or names unknown state")

    ready = deque(sorted(factor_id for factor_id, degree in indegree.items() if degree == 0))
    visited = 0
    while ready:
        factor_id = ready.popleft()
        visited += 1
        for child_id in sorted(children[factor_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
    if visited != len(nodes):
        raise FactorFrontierError("factor-parent graph contains a cycle")


def _snapshot_fixed_point(base: Mapping[str, Any]) -> tuple[int, str]:
    retained_state_bytes = 0
    for _ in range(32):
        core = {**base, "retained_state_bytes": retained_state_bytes}
        digest = canonical_sha256(core)
        measured = len(canonical_bytes({**core, "snapshot_sha256": digest}))
        if measured == retained_state_bytes:
            return retained_state_bytes, digest
        retained_state_bytes = measured
    raise FactorFrontierError("factor-frontier retained-byte fixed point did not converge")


@dataclass(frozen=True, slots=True)
class ShadowCausalFactorFrontier:

    source_ledger_sha256: str
    source_ledger_head_sha256: str | None
    through_sequence: int
    nodes: tuple[ShadowFactorNode, ...]
    active_factor_ids: tuple[str, ...]
    superseded_factor_ids: tuple[str, ...]
    referent_index: IndexRows
    factor_scope_index: IndexRows
    retained_state_bytes: int
    snapshot_sha256: str
    activation_enabled: bool = ACTIVATION_ENABLED
    runtime_consumable: bool = RUNTIME_CONSUMABLE
    factual_write_authorized: bool = FACTUAL_WRITE_AUTHORIZED
    scientific_promotion_allowed: bool = SCIENTIFIC_PROMOTION_ALLOWED
    schema: str = FACTOR_FRONTIER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FACTOR_FRONTIER_SCHEMA:
            raise FactorFrontierError(f"unsupported factor-frontier schema {self.schema!r}")
        _digest(self.source_ledger_sha256, "source_ledger_sha256")
        _nonnegative_int(self.through_sequence, "through_sequence")
        if self.through_sequence > MAX_HARD_EVENTS:
            raise FactorFrontierCapExceeded("frontier source sequence exceeds the hard event cap")
        if self.source_ledger_head_sha256 is None:
            if self.through_sequence != 0:
                raise FactorFrontierError("nonempty source ledger requires a head digest")
        else:
            _digest(self.source_ledger_head_sha256, "source_ledger_head_sha256")
            if self.through_sequence == 0:
                raise FactorFrontierError("empty source ledger cannot have a head digest")
        if not isinstance(self.nodes, tuple) or not all(
            type(node) is ShadowFactorNode for node in self.nodes
        ):
            raise FactorFrontierError("frontier nodes must be exact immutable ShadowFactorNode records")
        if len(self.nodes) > MAX_HARD_NODES:
            raise FactorFrontierCapExceeded("frontier node count exceeds the hard bound")
        if len(self.nodes) > self.through_sequence:
            raise FactorFrontierError("frontier cannot contain more factor nodes than source events")
        node_ids = tuple(node.factor_id for node in self.nodes)
        if node_ids != tuple(sorted(node_ids)) or len(set(node_ids)) != len(node_ids):
            raise FactorFrontierError("frontier nodes must be unique and canonically sorted by factor ID")
        source_ids = tuple(node.source_hypothesis_event_id for node in self.nodes)
        if len(set(source_ids)) != len(source_ids):
            raise FactorFrontierError("one hypothesis event cannot produce multiple v1 factor nodes")
        if len(self.active_factor_ids) > MAX_HARD_NODES or len(self.superseded_factor_ids) > MAX_HARD_NODES:
            raise FactorFrontierCapExceeded("frontier factor-ID sets exceed the hard pre-scan cap")
        _canonical_texts(self.active_factor_ids, "active_factor_ids")
        _canonical_texts(self.superseded_factor_ids, "superseded_factor_ids")
        for label, values in (
            ("active_factor_ids", self.active_factor_ids),
            ("superseded_factor_ids", self.superseded_factor_ids),
        ):
            for value in values:
                _digest(value, label)
        all_ids = set(node_ids)
        expected_superseded = {factor_id for node in self.nodes for factor_id in node.supersedes_factor_ids}
        expected_active = all_ids - expected_superseded
        if set(self.superseded_factor_ids) != expected_superseded:
            raise FactorFrontierError("superseded-factor set does not match explicit node revisions")
        if set(self.active_factor_ids) != expected_active:
            raise FactorFrontierError("active-factor set is not exactly revision-derived")
        if set(self.active_factor_ids) & set(self.superseded_factor_ids):
            raise FactorFrontierError("active and superseded factor sets overlap")
        if _edge_count(self.nodes) > MAX_HARD_EDGES:
            raise FactorFrontierCapExceeded("frontier edge count exceeds the hard bound")
        _validate_factor_dag(self.nodes)
        factor_by_source_event = {node.source_hypothesis_event_id: node.factor_id for node in self.nodes}
        for node in self.nodes:
            expected_factor_parents = tuple(
                sorted(
                    factor_by_source_event[event_id]
                    for event_id in node.causal_event_ids
                    if event_id in factor_by_source_event
                )
            )
            if node.parent_factor_ids != expected_factor_parents:
                raise FactorFrontierError(
                    "factor parents do not exactly match causal hypothesis-event parents"
                )
        expected_referent = _build_index(self.nodes, "referent_hypotheses")
        expected_scope = _build_index(self.nodes, "factor_scopes")
        if self.referent_index != expected_referent:
            raise FactorFrontierError("referent index is not exactly node-derived")
        if self.factor_scope_index != expected_scope:
            raise FactorFrontierError("factor-scope index is not exactly node-derived")
        postings = _index_posting_count(self.referent_index) + _index_posting_count(self.factor_scope_index)
        if postings > MAX_HARD_INDEX_POSTINGS:
            raise FactorFrontierCapExceeded("frontier index postings exceed the hard bound")
        _nonnegative_int(self.retained_state_bytes, "retained_state_bytes")
        _require_false(self.activation_enabled, "activation_enabled")
        _require_false(self.runtime_consumable, "runtime_consumable")
        _require_false(self.factual_write_authorized, "factual_write_authorized")
        _require_false(self.scientific_promotion_allowed, "scientific_promotion_allowed")
        _digest(self.snapshot_sha256, "snapshot_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.snapshot_sha256:
            raise FactorFrontierError("factor-frontier snapshot self-hash mismatch")
        measured = len(canonical_bytes(self.payload()))
        if measured != self.retained_state_bytes:
            raise FactorFrontierError("factor-frontier retained bytes do not match canonical snapshot bytes")
        if measured > MAX_HARD_SNAPSHOT_BYTES:
            raise FactorFrontierCapExceeded("factor-frontier snapshot exceeds the hard byte bound")

    @classmethod
    def create(
        cls,
        *,
        source_ledger_sha256: str,
        source_ledger_head_sha256: str | None,
        through_sequence: int,
        nodes: Sequence[ShadowFactorNode],
    ) -> Self:
        _nonnegative_int(through_sequence, "through_sequence")
        if through_sequence > MAX_HARD_EVENTS:
            raise FactorFrontierCapExceeded("frontier source sequence exceeds the hard event cap")
        bounded_nodes = _bounded_sequence(nodes, "frontier nodes", max_items=MAX_HARD_NODES)
        if len(bounded_nodes) > through_sequence:
            raise FactorFrontierError("frontier cannot contain more factor nodes than source events")
        if not all(type(node) is ShadowFactorNode for node in bounded_nodes):
            raise FactorFrontierError("frontier nodes must be exact ShadowFactorNode records")
        typed_nodes = tuple(cast(ShadowFactorNode, node) for node in bounded_nodes)
        canonical_nodes = tuple(sorted(typed_nodes, key=lambda node: node.factor_id))
        superseded = tuple(
            sorted({factor_id for node in canonical_nodes for factor_id in node.supersedes_factor_ids})
        )
        active = tuple(sorted({node.factor_id for node in canonical_nodes} - set(superseded)))
        referent_index = _build_index(canonical_nodes, "referent_hypotheses")
        factor_scope_index = _build_index(canonical_nodes, "factor_scopes")
        base: dict[str, Any] = {
            "schema": FACTOR_FRONTIER_SCHEMA,
            "source_ledger_sha256": source_ledger_sha256,
            "source_ledger_head_sha256": source_ledger_head_sha256,
            "through_sequence": through_sequence,
            "nodes": [node.payload() for node in canonical_nodes],
            "active_factor_ids": list(active),
            "superseded_factor_ids": list(superseded),
            "referent_index": _index_payload(referent_index),
            "factor_scope_index": _index_payload(factor_scope_index),
            "activation_enabled": False,
            "runtime_consumable": False,
            "factual_write_authorized": False,
            "scientific_promotion_allowed": False,
        }
        retained_state_bytes, snapshot_sha256 = _snapshot_fixed_point(base)
        return cls(
            source_ledger_sha256=source_ledger_sha256,
            source_ledger_head_sha256=source_ledger_head_sha256,
            through_sequence=through_sequence,
            nodes=canonical_nodes,
            active_factor_ids=active,
            superseded_factor_ids=superseded,
            referent_index=referent_index,
            factor_scope_index=factor_scope_index,
            retained_state_bytes=retained_state_bytes,
            snapshot_sha256=snapshot_sha256,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        expected = {
            "schema",
            "source_ledger_sha256",
            "source_ledger_head_sha256",
            "through_sequence",
            "nodes",
            "active_factor_ids",
            "superseded_factor_ids",
            "referent_index",
            "factor_scope_index",
            "retained_state_bytes",
            "snapshot_sha256",
            "activation_enabled",
            "runtime_consumable",
            "factual_write_authorized",
            "scientific_promotion_allowed",
        }
        _exact_keys(payload, expected, "ShadowCausalFactorFrontier")
        raw_nodes = payload["nodes"]
        if not isinstance(raw_nodes, list):
            raise FactorFrontierError("frontier nodes must be a list")
        if len(raw_nodes) > MAX_HARD_NODES:
            raise FactorFrontierCapExceeded("frontier node count exceeds the hard bound")
        nodes: list[ShadowFactorNode] = []
        for index, row in enumerate(raw_nodes):
            if not isinstance(row, Mapping):
                raise FactorFrontierError(f"frontier node {index} must be a mapping")
            nodes.append(ShadowFactorNode.from_payload(row))
        return cls(
            schema=payload["schema"],
            source_ledger_sha256=payload["source_ledger_sha256"],
            source_ledger_head_sha256=payload["source_ledger_head_sha256"],
            through_sequence=payload["through_sequence"],
            nodes=tuple(nodes),
            active_factor_ids=_string_list(
                payload["active_factor_ids"],
                "active_factor_ids",
                max_items=MAX_HARD_NODES,
            ),
            superseded_factor_ids=_string_list(
                payload["superseded_factor_ids"],
                "superseded_factor_ids",
                max_items=MAX_HARD_NODES,
            ),
            referent_index=_index_from_payload(payload["referent_index"], "referent_index"),
            factor_scope_index=_index_from_payload(payload["factor_scope_index"], "factor_scope_index"),
            retained_state_bytes=payload["retained_state_bytes"],
            snapshot_sha256=payload["snapshot_sha256"],
            activation_enabled=payload["activation_enabled"],
            runtime_consumable=payload["runtime_consumable"],
            factual_write_authorized=payload["factual_write_authorized"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "source_ledger_sha256": self.source_ledger_sha256,
            "source_ledger_head_sha256": self.source_ledger_head_sha256,
            "through_sequence": self.through_sequence,
            "nodes": [node.payload() for node in self.nodes],
            "active_factor_ids": list(self.active_factor_ids),
            "superseded_factor_ids": list(self.superseded_factor_ids),
            "referent_index": _index_payload(self.referent_index),
            "factor_scope_index": _index_payload(self.factor_scope_index),
            "retained_state_bytes": self.retained_state_bytes,
            "activation_enabled": self.activation_enabled,
            "runtime_consumable": self.runtime_consumable,
            "factual_write_authorized": self.factual_write_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["snapshot_sha256"] = self.snapshot_sha256
        return result


def _opaque_size(value: FrozenJSON, label: str, caps: FactorFrontierCaps) -> None:
    if len(value.canonical) > caps.max_opaque_payload_bytes:
        raise FactorFrontierCapExceeded(f"{label} exceeds the opaque-payload cap")


def _precheck_event_shape(event: ESCSEvent, caps: FactorFrontierCaps) -> None:
    envelope = event.envelope
    if len(envelope.causal_parent_ids) + len(envelope.supersedes_event_ids) > caps.max_edges:
        raise FactorFrontierCapExceeded("one event exceeds the declared edge cap")
    _opaque_size(envelope.source_and_provenance, "event source/provenance", caps)
    if isinstance(event, ObservationEvent):
        if len(event.raw_packet_or_delta_refs) > caps.max_edges:
            raise FactorFrontierCapExceeded("observation references exceed the declared edge cap")
        for observation_text in (*event.raw_packet_or_delta_refs, event.adapter_version):
            _text(observation_text, "observation text")
        _opaque_size(event.sensor_scope, "observation sensor scope", caps)
    elif isinstance(event, HypothesisEvent):
        if len(event.supporting_event_ids) > caps.max_edges:
            raise FactorFrontierCapExceeded("hypothesis support exceeds the declared edge cap")
        for label, opaque_value in (
            ("referent hypotheses", event.referent_hypotheses),
            ("factor changes", event.factor_change_distribution),
            ("decision relevance", event.decision_relevance_distribution),
            ("reducibility", event.reducibility_distribution),
        ):
            _opaque_size(opaque_value, label, caps)
    elif isinstance(event, CommitmentEvent):
        for label, opaque_value in (
            ("committed payload", event.committed_payload),
            ("decision distribution", event.decision_distribution),
            ("predicted utility", event.predicted_utility_vector),
        ):
            _opaque_size(opaque_value, label, caps)
    elif isinstance(event, ConsequenceEvent):
        _opaque_size(event.observed_outcome, "observed outcome", caps)
        _opaque_size(event.realized_utility_vector, "realized utility", caps)


def _authoritative_replay(ledger: EventLedger, caps: FactorFrontierCaps) -> EventLedger:
    if type(ledger) is not EventLedger:
        raise FactorFrontierError("factor-frontier source must be an exact EventLedger")
    with ledger._lock:  # noqa: SLF001
        if ledger.entry_count > caps.max_events:
            raise FactorFrontierCapExceeded("source ledger exceeds the declared event cap")
        captured_entries = ledger.entries
        encoded_event_total = 0
        source_event_edges = 0
        for entry in captured_entries:
            _precheck_event_shape(entry.event, caps)
            source_event_edges += len(entry.event.envelope.causal_parent_ids)
            if source_event_edges > caps.max_edges:
                raise FactorFrontierCapExceeded("source-ledger causal edges exceed the declared edge cap")
            try:
                event_size = len(canonical_bytes(entry.event.payload()))
            except (TypeError, ValueError, RecursionError) as exc:
                raise FactorFrontierError(f"source event encoding failed closed: {exc}") from exc
            if event_size > caps.max_event_encoded_bytes:
                raise FactorFrontierCapExceeded("source event exceeds the encoded-event cap")
            encoded_event_total += event_size
            if encoded_event_total > caps.max_source_ledger_bytes:
                raise FactorFrontierCapExceeded("source ledger exceeds the declared byte cap")

        try:
            payload = ledger.payload()
            source_bytes = len(canonical_bytes(payload))
        except (TypeError, ValueError, RecursionError) as exc:
            raise FactorFrontierError(f"source ledger encoding failed closed: {exc}") from exc
        if source_bytes > caps.max_source_ledger_bytes:
            raise FactorFrontierCapExceeded("source ledger envelope exceeds the declared byte cap")
        captured_sha256 = canonical_sha256(payload)
    try:
        replay = EventLedger.replay(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        raise FactorFrontierError(f"source ledger exact replay failed: {exc}") from exc
    if replay.payload() != payload or replay.sha256 != captured_sha256:
        raise FactorFrontierError("source ledger replay authority drifted")
    if problems := replay.verify():
        raise FactorFrontierError("source ledger verification failed: " + "; ".join(problems))
    return replay


def _outer_mapping_keys(
    value: FrozenJSON,
    *,
    label: str,
    limit: int,
    caps: FactorFrontierCaps,
) -> tuple[str, ...]:
    _opaque_size(value, label, caps)
    try:
        decoded = value.value()
    except (ValueError, RecursionError) as exc:
        raise FactorFrontierError(f"{label} is not bounded strict JSON") from exc
    if not isinstance(decoded, dict):
        return ()
    if len(decoded) > limit:
        raise FactorFrontierCapExceeded(f"{label} outer keys exceed the declared cap")
    keys = tuple(sorted(decoded))
    for key in keys:
        _text(key, f"{label} key")
    return keys


def _build_snapshot(replay: EventLedger, caps: FactorFrontierCaps) -> ShadowCausalFactorFrontier:
    nodes_in_sequence: list[ShadowFactorNode] = []
    factor_by_event: dict[str, str] = {}
    for entry in replay.entries:
        event = entry.event
        if not isinstance(event, HypothesisEvent):
            continue
        if len(nodes_in_sequence) >= caps.max_nodes:
            raise FactorFrontierCapExceeded("projected factor nodes exceed the declared cap")
        referents = _outer_mapping_keys(
            event.referent_hypotheses,
            label="referent hypotheses",
            limit=caps.max_referents_per_node,
            caps=caps,
        )
        scopes = _outer_mapping_keys(
            event.factor_change_distribution,
            label="factor-change distribution",
            limit=caps.max_scopes_per_node,
            caps=caps,
        )
        parent_factors = tuple(
            sorted(
                factor_by_event[str(parent_id)]
                for parent_id in event.envelope.causal_parent_ids
                if str(parent_id) in factor_by_event
            )
        )
        superseded_factors: list[str] = []
        for superseded_id in event.envelope.supersedes_event_ids:
            factor_id = factor_by_event.get(str(superseded_id))
            if factor_id is None:
                raise FactorFrontierError("hypothesis supersession lacks an earlier factor node")
            superseded_factors.append(factor_id)
        node = ShadowFactorNode.create(
            source_hypothesis_event_id=str(event.event_id),
            branch_id=str(event.branch_id),
            causal_event_ids=tuple(str(value) for value in event.envelope.causal_parent_ids),
            parent_factor_ids=parent_factors,
            supersedes_factor_ids=tuple(superseded_factors),
            supporting_event_ids=tuple(str(value) for value in event.supporting_event_ids),
            referent_hypotheses=referents,
            factor_scopes=scopes,
            epistemic_status=event.epistemic_status,
            evidence_class=event.evidence_class,
            source_payload_digest=event.envelope.payload_digest,
            producer_state_version=event.envelope.producer_state_version,
            clock_start_tick=event.envelope.clock_start_tick,
            clock_end_tick=event.envelope.clock_end_tick,
            clock_uncertainty=event.envelope.clock_uncertainty,
        )
        nodes_in_sequence.append(node)
        factor_by_event[str(event.event_id)] = node.factor_id

    if _edge_count(nodes_in_sequence) > caps.max_edges:
        raise FactorFrontierCapExceeded("projected factor edges exceed the declared cap")
    snapshot = ShadowCausalFactorFrontier.create(
        source_ledger_sha256=replay.sha256,
        source_ledger_head_sha256=replay.head_sha256,
        through_sequence=replay.entry_count,
        nodes=nodes_in_sequence,
    )
    postings = _index_posting_count(snapshot.referent_index) + _index_posting_count(
        snapshot.factor_scope_index
    )
    if postings > caps.max_index_postings:
        raise FactorFrontierCapExceeded("projected factor indexes exceed the declared posting cap")
    if snapshot.retained_state_bytes > caps.max_snapshot_bytes:
        raise FactorFrontierCapExceeded("projected factor snapshot exceeds the declared byte cap")
    return snapshot


def _projection_work(
    *,
    events: int,
    nodes: int,
    source_event_edges: int,
    factor_edges: int,
    index_postings: int,
) -> WorkVector:
    return WorkVector(
        indexing_and_graph_maintenance=(
            1 + events + nodes + source_event_edges + factor_edges + index_postings
        )
    )


@dataclass(frozen=True, slots=True)
class FrontierProjectionReceipt:
    snapshot: ShadowCausalFactorFrontier
    caps: FactorFrontierCaps
    previous_snapshot_sha256: str | None
    events_examined: int
    source_event_edges_examined: int
    nodes_materialized: int
    edges_materialized: int
    index_postings_materialized: int
    work: WorkVector
    accounting_applied: bool
    activation_enabled: bool
    runtime_consumable: bool
    factual_write_authorized: bool
    scientific_promotion_allowed: bool
    receipt_sha256: str
    schema: str = FACTOR_PROJECTION_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PROJECTION_RECEIPT_ISSUANCE_TOKEN:
            raise FactorFrontierError(
                "projection receipts can be issued only after exact source-ledger replay"
            )
        if self.schema != FACTOR_PROJECTION_RECEIPT_SCHEMA:
            raise FactorFrontierError(f"unsupported projection receipt schema {self.schema!r}")
        if type(self.snapshot) is not ShadowCausalFactorFrontier:
            raise FactorFrontierError("projection receipt requires a shadow frontier snapshot")
        if type(self.caps) is not FactorFrontierCaps:
            raise FactorFrontierError("projection receipt requires exact FactorFrontierCaps")
        _check_snapshot_caps(self.snapshot, self.caps)
        if self.previous_snapshot_sha256 is not None:
            _digest(self.previous_snapshot_sha256, "previous_snapshot_sha256")
        for label, value in (
            ("events_examined", self.events_examined),
            ("source_event_edges_examined", self.source_event_edges_examined),
            ("nodes_materialized", self.nodes_materialized),
            ("edges_materialized", self.edges_materialized),
            ("index_postings_materialized", self.index_postings_materialized),
        ):
            _nonnegative_int(value, label)
        if self.events_examined != self.snapshot.through_sequence:
            raise FactorFrontierError("projection event count drifted from the snapshot sequence")
        if self.events_examined > self.caps.max_events:
            raise FactorFrontierCapExceeded("projection receipt exceeds the declared event cap")
        if self.source_event_edges_examined > self.caps.max_edges:
            raise FactorFrontierCapExceeded("projection receipt exceeds the declared source-event edge cap")
        if self.nodes_materialized != len(self.snapshot.nodes):
            raise FactorFrontierError("projection node count drifted from the snapshot")
        if self.edges_materialized != _edge_count(self.snapshot.nodes):
            raise FactorFrontierError("projection edge count drifted from the snapshot")
        expected_postings = _index_posting_count(self.snapshot.referent_index) + _index_posting_count(
            self.snapshot.factor_scope_index
        )
        if self.index_postings_materialized != expected_postings:
            raise FactorFrontierError("projection posting count drifted from the snapshot")
        expected_work = _projection_work(
            events=self.events_examined,
            nodes=self.nodes_materialized,
            source_event_edges=self.source_event_edges_examined,
            factor_edges=self.edges_materialized,
            index_postings=self.index_postings_materialized,
        )
        if self.work != expected_work:
            raise FactorFrontierError("projection work is not exactly count-derived")
        if self.work.total_work > self.caps.max_work_units:
            raise FactorFrontierCapExceeded("projection receipt exceeds the declared work cap")
        _require_false(self.accounting_applied, "accounting_applied")
        _require_false(self.activation_enabled, "activation_enabled")
        _require_false(self.runtime_consumable, "runtime_consumable")
        _require_false(self.factual_write_authorized, "factual_write_authorized")
        _require_false(self.scientific_promotion_allowed, "scientific_promotion_allowed")
        _digest(self.receipt_sha256, "receipt_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.receipt_sha256:
            raise FactorFrontierError("projection receipt self-hash mismatch")

    @classmethod
    def _issue(
        cls,
        *,
        snapshot: ShadowCausalFactorFrontier,
        caps: FactorFrontierCaps,
        previous_snapshot_sha256: str | None,
        events_examined: int,
        source_event_edges_examined: int,
        _issuance_token: object,
    ) -> Self:
        if _issuance_token is not _PROJECTION_RECEIPT_ISSUANCE_TOKEN:
            raise FactorFrontierError("projection receipt issuance token is invalid")
        edges = _edge_count(snapshot.nodes)
        postings = _index_posting_count(snapshot.referent_index) + _index_posting_count(
            snapshot.factor_scope_index
        )
        work = _projection_work(
            events=events_examined,
            nodes=len(snapshot.nodes),
            source_event_edges=source_event_edges_examined,
            factor_edges=edges,
            index_postings=postings,
        )
        core = {
            "schema": FACTOR_PROJECTION_RECEIPT_SCHEMA,
            "snapshot": snapshot.payload(),
            "caps": caps.payload(),
            "previous_snapshot_sha256": previous_snapshot_sha256,
            "events_examined": events_examined,
            "source_event_edges_examined": source_event_edges_examined,
            "nodes_materialized": len(snapshot.nodes),
            "edges_materialized": edges,
            "index_postings_materialized": postings,
            "work": work.payload(),
            "accounting_applied": False,
            "activation_enabled": False,
            "runtime_consumable": False,
            "factual_write_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            snapshot=snapshot,
            caps=caps,
            previous_snapshot_sha256=previous_snapshot_sha256,
            events_examined=events_examined,
            source_event_edges_examined=source_event_edges_examined,
            nodes_materialized=len(snapshot.nodes),
            edges_materialized=edges,
            index_postings_materialized=postings,
            work=work,
            accounting_applied=False,
            activation_enabled=False,
            runtime_consumable=False,
            factual_write_authorized=False,
            scientific_promotion_allowed=False,
            receipt_sha256=canonical_sha256(core),
            _issuance_token=_issuance_token,
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "snapshot": self.snapshot.payload(),
            "caps": self.caps.payload(),
            "previous_snapshot_sha256": self.previous_snapshot_sha256,
            "events_examined": self.events_examined,
            "source_event_edges_examined": self.source_event_edges_examined,
            "nodes_materialized": self.nodes_materialized,
            "edges_materialized": self.edges_materialized,
            "index_postings_materialized": self.index_postings_materialized,
            "work": self.work.payload(),
            "accounting_applied": self.accounting_applied,
            "activation_enabled": self.activation_enabled,
            "runtime_consumable": self.runtime_consumable,
            "factual_write_authorized": self.factual_write_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def _ledger_prefix(replay: EventLedger, through_sequence: int) -> EventLedger:
    if through_sequence > replay.entry_count:
        raise FactorFrontierError("previous snapshot extends beyond the current source ledger")
    prefix = EventLedger()
    for source_entry in replay.entries[:through_sequence]:
        regenerated = prefix.append(source_entry.event)
        if regenerated.payload() != source_entry.payload():
            raise FactorFrontierError("source-ledger prefix authority drifted")
    return prefix


def _check_snapshot_caps(snapshot: ShadowCausalFactorFrontier, caps: FactorFrontierCaps) -> None:
    if snapshot.through_sequence > caps.max_events:
        raise FactorFrontierCapExceeded("frontier source sequence exceeds the event cap")
    if len(snapshot.nodes) > caps.max_nodes:
        raise FactorFrontierCapExceeded("frontier node count exceeds the declared cap")
    if _edge_count(snapshot.nodes) > caps.max_edges:
        raise FactorFrontierCapExceeded("frontier edge count exceeds the declared cap")
    for node in snapshot.nodes:
        if len(node.referent_hypotheses) > caps.max_referents_per_node:
            raise FactorFrontierCapExceeded("frontier referents exceed the per-node cap")
        if len(node.factor_scopes) > caps.max_scopes_per_node:
            raise FactorFrontierCapExceeded("frontier scopes exceed the per-node cap")
    postings = _index_posting_count(snapshot.referent_index) + _index_posting_count(
        snapshot.factor_scope_index
    )
    if postings > caps.max_index_postings:
        raise FactorFrontierCapExceeded("frontier index postings exceed the declared cap")
    if snapshot.retained_state_bytes > caps.max_snapshot_bytes:
        raise FactorFrontierCapExceeded("frontier snapshot exceeds the declared byte cap")


def project_shadow_factor_frontier(
    ledger: EventLedger,
    *,
    caps: FactorFrontierCaps,
    previous: ShadowCausalFactorFrontier | None = None,
) -> FrontierProjectionReceipt:

    if type(caps) is not FactorFrontierCaps:
        raise FactorFrontierError("projection requires typed FactorFrontierCaps")
    if type(ledger) is not EventLedger:
        raise FactorFrontierError("projection source must be an exact EventLedger")
    if ledger.entry_count > caps.max_events:
        raise FactorFrontierCapExceeded("source ledger exceeds the event cap")
    replay = _authoritative_replay(ledger, caps)
    previous_sha256: str | None = None
    if previous is not None:
        if type(previous) is not ShadowCausalFactorFrontier:
            raise FactorFrontierError("previous frontier must be a typed shadow snapshot")
        prefix = _ledger_prefix(replay, previous.through_sequence)
        problems = verify_shadow_factor_frontier(previous, ledger=prefix, caps=caps)
        if problems:
            raise FactorFrontierError("previous frontier authority failed: " + "; ".join(problems))
        previous_sha256 = previous.snapshot_sha256
    snapshot = _build_snapshot(replay, caps)
    receipt = FrontierProjectionReceipt._issue(
        snapshot=snapshot,
        caps=caps,
        previous_snapshot_sha256=previous_sha256,
        events_examined=replay.entry_count,
        source_event_edges_examined=_source_event_edge_count(replay),
        _issuance_token=_PROJECTION_RECEIPT_ISSUANCE_TOKEN,
    )
    if receipt.work.total_work > caps.max_work_units:
        raise FactorFrontierCapExceeded("projection work exceeds the declared cap")
    return receipt


def verify_shadow_factor_frontier(
    snapshot: ShadowCausalFactorFrontier,
    *,
    ledger: EventLedger,
    caps: FactorFrontierCaps,
) -> tuple[str, ...]:

    problems: list[str] = []
    try:
        if type(snapshot) is not ShadowCausalFactorFrontier:
            raise FactorFrontierError("snapshot must be a typed shadow frontier")
        if type(caps) is not FactorFrontierCaps:
            raise FactorFrontierError("verification requires exact FactorFrontierCaps")
        if type(ledger) is not EventLedger:
            raise FactorFrontierError("verification source must be an exact EventLedger")
        _check_snapshot_caps(snapshot, caps)
        replay = _authoritative_replay(ledger, caps)
        expected = _build_snapshot(replay, caps)
        if expected.payload() != snapshot.payload():
            problems.append("snapshot differs from exact ledger projection")
    except (FactorFrontierError, TypeError, ValueError) as exc:
        problems.append(str(exc))
    return tuple(problems)


@dataclass(frozen=True, slots=True)
class FactorFrontierQuery:
    branch_id: str
    referent_any: tuple[str, ...]
    factor_scope_any: tuple[str, ...]
    max_results: int
    include_superseded: bool = False
    schema: str = FACTOR_QUERY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FACTOR_QUERY_SCHEMA:
            raise FactorFrontierError(f"unsupported factor-query schema {self.schema!r}")
        _text(self.branch_id, "query branch_id")
        BranchRef(self.branch_id)
        if (
            len(self.referent_any) > MAX_HARD_KEYS_PER_NODE
            or len(self.factor_scope_any) > MAX_HARD_KEYS_PER_NODE
        ):
            raise FactorFrontierCapExceeded("query key count exceeds the hard pre-scan bound")
        _canonical_texts(self.referent_any, "query referent_any")
        _canonical_texts(self.factor_scope_any, "query factor_scope_any")
        if _positive_int(self.max_results, "query max_results") > MAX_HARD_QUERY_RESULTS:
            raise FactorFrontierCapExceeded("query max_results exceeds the hard bound")
        if not isinstance(self.include_superseded, bool):
            raise FactorFrontierError("include_superseded must be a boolean")

    @property
    def query_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "branch_id": self.branch_id,
            "referent_any": list(self.referent_any),
            "factor_scope_any": list(self.factor_scope_any),
            "max_results": self.max_results,
            "include_superseded": self.include_superseded,
        }


def _query_work(
    *,
    authority_events: int,
    authority_nodes: int,
    authority_source_event_edges: int,
    authority_factor_edges: int,
    authority_postings: int,
    index_keys_probed: int,
    postings: int,
    candidates: int,
    returned: int,
) -> WorkVector:
    authority = _projection_work(
        events=authority_events,
        nodes=authority_nodes,
        source_event_edges=authority_source_event_edges,
        factor_edges=authority_factor_edges,
        index_postings=authority_postings,
    )
    return authority + WorkVector(
        dispatch_and_exploration=(1 + index_keys_probed + postings + candidates + returned)
    )


@dataclass(frozen=True, slots=True)
class FrontierQueryReceipt:
    snapshot_sha256: str
    source_ledger_sha256: str
    caps: FactorFrontierCaps
    query_sha256: str
    matched_factor_ids: tuple[str, ...]
    total_match_count: int
    saturated: bool
    authority_events_examined: int
    authority_nodes_materialized: int
    authority_source_event_edges_examined: int
    authority_edges_materialized: int
    authority_index_postings_materialized: int
    index_keys_probed: int
    index_postings_touched: int
    candidates_considered: int
    work: WorkVector
    source_replay_verified: bool
    accounting_applied: bool
    activation_enabled: bool
    runtime_consumable: bool
    factual_write_authorized: bool
    scientific_promotion_allowed: bool
    receipt_sha256: str
    schema: str = FACTOR_QUERY_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _QUERY_RECEIPT_ISSUANCE_TOKEN:
            raise FactorFrontierError("query receipts can be issued only after exact source-ledger replay")
        if self.schema != FACTOR_QUERY_RECEIPT_SCHEMA:
            raise FactorFrontierError(f"unsupported query-receipt schema {self.schema!r}")
        _digest(self.snapshot_sha256, "snapshot_sha256")
        _digest(self.source_ledger_sha256, "source_ledger_sha256")
        if type(self.caps) is not FactorFrontierCaps:
            raise FactorFrontierError("query receipt requires exact FactorFrontierCaps")
        _digest(self.query_sha256, "query_sha256")
        if len(self.matched_factor_ids) > MAX_HARD_QUERY_RESULTS:
            raise FactorFrontierCapExceeded("query receipt matches exceed the hard pre-scan cap")
        _canonical_texts(self.matched_factor_ids, "matched_factor_ids")
        for factor_id in self.matched_factor_ids:
            _digest(factor_id, "matched_factor_id")
        _nonnegative_int(self.total_match_count, "total_match_count")
        _nonnegative_int(self.authority_events_examined, "authority_events_examined")
        _nonnegative_int(self.authority_nodes_materialized, "authority_nodes_materialized")
        _nonnegative_int(
            self.authority_source_event_edges_examined,
            "authority_source_event_edges_examined",
        )
        _nonnegative_int(self.authority_edges_materialized, "authority_edges_materialized")
        _nonnegative_int(
            self.authority_index_postings_materialized,
            "authority_index_postings_materialized",
        )
        _nonnegative_int(self.index_keys_probed, "index_keys_probed")
        _nonnegative_int(self.index_postings_touched, "index_postings_touched")
        _nonnegative_int(self.candidates_considered, "candidates_considered")
        if self.total_match_count < len(self.matched_factor_ids):
            raise FactorFrontierError("query receipt returned more factors than it matched")
        if self.total_match_count > self.authority_nodes_materialized:
            raise FactorFrontierError("query match count exceeds its replayed factor authority")
        if self.candidates_considered > self.authority_nodes_materialized:
            raise FactorFrontierError("query candidate count exceeds its replayed factor authority")
        if self.authority_events_examined > self.caps.max_events:
            raise FactorFrontierCapExceeded("query receipt exceeds the declared event cap")
        if self.authority_nodes_materialized > self.caps.max_nodes:
            raise FactorFrontierCapExceeded("query receipt exceeds the declared node cap")
        if self.authority_source_event_edges_examined > self.caps.max_edges:
            raise FactorFrontierCapExceeded("query receipt exceeds the declared source-event edge cap")
        if self.authority_edges_materialized > self.caps.max_edges:
            raise FactorFrontierCapExceeded("query receipt exceeds the declared factor-edge cap")
        if self.authority_index_postings_materialized > self.caps.max_index_postings:
            raise FactorFrontierCapExceeded("query receipt exceeds the declared index-posting cap")
        if self.index_postings_touched > self.caps.max_index_postings:
            raise FactorFrontierCapExceeded("query receipt touches excess declared index postings")
        if self.index_keys_probed > (self.caps.max_referents_per_node + self.caps.max_scopes_per_node):
            raise FactorFrontierCapExceeded("query receipt exceeds the declared key-probe cap")
        if len(self.matched_factor_ids) > self.caps.max_query_results:
            raise FactorFrontierCapExceeded("query receipt exceeds the declared result cap")
        if self.saturated != (self.total_match_count > len(self.matched_factor_ids)):
            raise FactorFrontierError("query saturation flag is not count-derived")
        expected_work = _query_work(
            authority_events=self.authority_events_examined,
            authority_nodes=self.authority_nodes_materialized,
            authority_source_event_edges=self.authority_source_event_edges_examined,
            authority_factor_edges=self.authority_edges_materialized,
            authority_postings=self.authority_index_postings_materialized,
            index_keys_probed=self.index_keys_probed,
            postings=self.index_postings_touched,
            candidates=self.candidates_considered,
            returned=len(self.matched_factor_ids),
        )
        if self.work != expected_work:
            raise FactorFrontierError("query work is not exactly count-derived")
        if self.work.total_work > self.caps.max_work_units:
            raise FactorFrontierCapExceeded("query receipt exceeds the declared work cap")
        _require_true(self.source_replay_verified, "source_replay_verified")
        _require_false(self.accounting_applied, "accounting_applied")
        _require_false(self.activation_enabled, "activation_enabled")
        _require_false(self.runtime_consumable, "runtime_consumable")
        _require_false(self.factual_write_authorized, "factual_write_authorized")
        _require_false(self.scientific_promotion_allowed, "scientific_promotion_allowed")
        _digest(self.receipt_sha256, "receipt_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.receipt_sha256:
            raise FactorFrontierError("query receipt self-hash mismatch")

    @classmethod
    def _issue(
        cls,
        *,
        snapshot_sha256: str,
        source_ledger_sha256: str,
        caps: FactorFrontierCaps,
        query_sha256: str,
        matched_factor_ids: Sequence[str],
        total_match_count: int,
        authority_events_examined: int,
        authority_nodes_materialized: int,
        authority_source_event_edges_examined: int,
        authority_edges_materialized: int,
        authority_index_postings_materialized: int,
        index_keys_probed: int,
        index_postings_touched: int,
        candidates_considered: int,
        _issuance_token: object,
    ) -> Self:
        if _issuance_token is not _QUERY_RECEIPT_ISSUANCE_TOKEN:
            raise FactorFrontierError("query receipt issuance token is invalid")
        raw_matched = _string_list(
            matched_factor_ids,
            "matched_factor_ids",
            max_items=MAX_HARD_QUERY_RESULTS,
        )
        matched = tuple(sorted(raw_matched))
        work = _query_work(
            authority_events=authority_events_examined,
            authority_nodes=authority_nodes_materialized,
            authority_source_event_edges=authority_source_event_edges_examined,
            authority_factor_edges=authority_edges_materialized,
            authority_postings=authority_index_postings_materialized,
            index_keys_probed=index_keys_probed,
            postings=index_postings_touched,
            candidates=candidates_considered,
            returned=len(matched),
        )
        core = {
            "schema": FACTOR_QUERY_RECEIPT_SCHEMA,
            "snapshot_sha256": snapshot_sha256,
            "source_ledger_sha256": source_ledger_sha256,
            "caps": caps.payload(),
            "query_sha256": query_sha256,
            "matched_factor_ids": list(matched),
            "total_match_count": total_match_count,
            "saturated": total_match_count > len(matched),
            "authority_events_examined": authority_events_examined,
            "authority_nodes_materialized": authority_nodes_materialized,
            "authority_source_event_edges_examined": authority_source_event_edges_examined,
            "authority_edges_materialized": authority_edges_materialized,
            "authority_index_postings_materialized": authority_index_postings_materialized,
            "index_keys_probed": index_keys_probed,
            "index_postings_touched": index_postings_touched,
            "candidates_considered": candidates_considered,
            "work": work.payload(),
            "source_replay_verified": True,
            "accounting_applied": False,
            "activation_enabled": False,
            "runtime_consumable": False,
            "factual_write_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            snapshot_sha256=snapshot_sha256,
            source_ledger_sha256=source_ledger_sha256,
            caps=caps,
            query_sha256=query_sha256,
            matched_factor_ids=matched,
            total_match_count=total_match_count,
            saturated=total_match_count > len(matched),
            authority_events_examined=authority_events_examined,
            authority_nodes_materialized=authority_nodes_materialized,
            authority_source_event_edges_examined=authority_source_event_edges_examined,
            authority_edges_materialized=authority_edges_materialized,
            authority_index_postings_materialized=authority_index_postings_materialized,
            index_keys_probed=index_keys_probed,
            index_postings_touched=index_postings_touched,
            candidates_considered=candidates_considered,
            work=work,
            source_replay_verified=True,
            accounting_applied=False,
            activation_enabled=False,
            runtime_consumable=False,
            factual_write_authorized=False,
            scientific_promotion_allowed=False,
            receipt_sha256=canonical_sha256(core),
            _issuance_token=_issuance_token,
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "snapshot_sha256": self.snapshot_sha256,
            "source_ledger_sha256": self.source_ledger_sha256,
            "caps": self.caps.payload(),
            "query_sha256": self.query_sha256,
            "matched_factor_ids": list(self.matched_factor_ids),
            "total_match_count": self.total_match_count,
            "saturated": self.saturated,
            "authority_events_examined": self.authority_events_examined,
            "authority_nodes_materialized": self.authority_nodes_materialized,
            "authority_source_event_edges_examined": self.authority_source_event_edges_examined,
            "authority_edges_materialized": self.authority_edges_materialized,
            "authority_index_postings_materialized": self.authority_index_postings_materialized,
            "index_keys_probed": self.index_keys_probed,
            "index_postings_touched": self.index_postings_touched,
            "candidates_considered": self.candidates_considered,
            "work": self.work.payload(),
            "source_replay_verified": self.source_replay_verified,
            "accounting_applied": self.accounting_applied,
            "activation_enabled": self.activation_enabled,
            "runtime_consumable": self.runtime_consumable,
            "factual_write_authorized": self.factual_write_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def query_shadow_factor_frontier(
    snapshot: ShadowCausalFactorFrontier,
    query: FactorFrontierQuery,
    *,
    ledger: EventLedger,
    caps: FactorFrontierCaps,
) -> FrontierQueryReceipt:

    if type(snapshot) is not ShadowCausalFactorFrontier:
        raise FactorFrontierError("query requires a typed shadow frontier")
    if type(query) is not FactorFrontierQuery:
        raise FactorFrontierError("query requires a typed FactorFrontierQuery")
    if type(caps) is not FactorFrontierCaps:
        raise FactorFrontierError("query requires typed FactorFrontierCaps")
    if type(ledger) is not EventLedger:
        raise FactorFrontierError("query source must be an exact EventLedger")
    if query.max_results > caps.max_query_results:
        raise FactorFrontierCapExceeded("query result limit exceeds the declared cap")
    if (
        len(query.referent_any) > caps.max_referents_per_node
        or len(query.factor_scope_any) > caps.max_scopes_per_node
    ):
        raise FactorFrontierCapExceeded("query key count exceeds the declared cap")
    _check_snapshot_caps(snapshot, caps)
    replay = _authoritative_replay(ledger, caps)
    authoritative_snapshot = _build_snapshot(replay, caps)
    if authoritative_snapshot.payload() != snapshot.payload():
        raise FactorFrontierError("query snapshot differs from exact source-ledger projection")
    authority_edges = _edge_count(authoritative_snapshot.nodes)
    authority_source_event_edges = _source_event_edge_count(replay)
    authority_postings = _index_posting_count(authoritative_snapshot.referent_index) + _index_posting_count(
        authoritative_snapshot.factor_scope_index
    )

    referent_map = dict(snapshot.referent_index)
    scope_map = dict(snapshot.factor_scope_index)
    postings_touched = 0
    referent_ids: set[str] | None = None
    if query.referent_any:
        referent_ids = set()
        for key in query.referent_any:
            postings = referent_map.get(key, ())
            postings_touched += len(postings)
            referent_ids.update(postings)
    scope_ids: set[str] | None = None
    if query.factor_scope_any:
        scope_ids = set()
        for key in query.factor_scope_any:
            postings = scope_map.get(key, ())
            postings_touched += len(postings)
            scope_ids.update(postings)

    if referent_ids is None and scope_ids is None:
        candidate_ids = {node.factor_id for node in snapshot.nodes}
    elif referent_ids is None:
        candidate_ids = set(scope_ids or ())
    elif scope_ids is None:
        candidate_ids = set(referent_ids)
    else:
        candidate_ids = referent_ids & scope_ids
    candidates_considered = len(candidate_ids)
    node_by_id = {node.factor_id: node for node in snapshot.nodes}
    active = set(snapshot.active_factor_ids)
    matches = tuple(
        sorted(
            factor_id
            for factor_id in candidate_ids
            if node_by_id[factor_id].branch_id == query.branch_id
            and (query.include_superseded or factor_id in active)
        )
    )
    returned = matches[: query.max_results]
    receipt = FrontierQueryReceipt._issue(
        snapshot_sha256=snapshot.snapshot_sha256,
        source_ledger_sha256=replay.sha256,
        caps=caps,
        query_sha256=query.query_sha256,
        matched_factor_ids=returned,
        total_match_count=len(matches),
        authority_events_examined=replay.entry_count,
        authority_nodes_materialized=len(authoritative_snapshot.nodes),
        authority_source_event_edges_examined=authority_source_event_edges,
        authority_edges_materialized=authority_edges,
        authority_index_postings_materialized=authority_postings,
        index_keys_probed=len(query.referent_any) + len(query.factor_scope_any),
        index_postings_touched=postings_touched,
        candidates_considered=candidates_considered,
        _issuance_token=_QUERY_RECEIPT_ISSUANCE_TOKEN,
    )
    if receipt.work.total_work > caps.max_work_units:
        raise FactorFrontierCapExceeded("query work exceeds the declared cap")
    return receipt


def _invalidation_work(
    *, events: int, event_edges: int, affected_events: int, affected_factors: int
) -> WorkVector:
    return WorkVector(
        indexing_and_graph_maintenance=(1 + events + event_edges + affected_events + affected_factors)
    )


@dataclass(frozen=True, slots=True)
class FactorInvalidationPlan:
    snapshot_sha256: str
    source_ledger_sha256: str
    caps: FactorFrontierCaps
    erased_event_ids: tuple[str, ...]
    affected_event_ids: tuple[str, ...]
    affected_factor_ids: tuple[str, ...]
    invalidated_active_factor_ids: tuple[str, ...]
    events_examined: int
    event_edges_examined: int
    work: WorkVector
    accounting_applied: bool
    archive_deletion_verified: bool
    application_authorized: bool
    activation_enabled: bool
    runtime_consumable: bool
    factual_write_authorized: bool
    scientific_promotion_allowed: bool
    plan_sha256: str
    schema: str = FACTOR_INVALIDATION_PLAN_SCHEMA
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _INVALIDATION_PLAN_ISSUANCE_TOKEN:
            raise FactorFrontierError(
                "invalidation plans can be issued only after exact source-ledger replay"
            )
        if self.schema != FACTOR_INVALIDATION_PLAN_SCHEMA:
            raise FactorFrontierError(f"unsupported invalidation-plan schema {self.schema!r}")
        _digest(self.snapshot_sha256, "snapshot_sha256")
        _digest(self.source_ledger_sha256, "source_ledger_sha256")
        if type(self.caps) is not FactorFrontierCaps:
            raise FactorFrontierError("invalidation plan requires exact FactorFrontierCaps")
        if len(self.erased_event_ids) > MAX_HARD_EVENTS or len(self.affected_event_ids) > MAX_HARD_EVENTS:
            raise FactorFrontierCapExceeded("invalidation event IDs exceed the hard pre-scan cap")
        if (
            len(self.affected_factor_ids) > MAX_HARD_NODES
            or len(self.invalidated_active_factor_ids) > MAX_HARD_NODES
        ):
            raise FactorFrontierCapExceeded("invalidation factor IDs exceed the hard pre-scan cap")
        for label, values in (
            ("erased_event_ids", self.erased_event_ids),
            ("affected_event_ids", self.affected_event_ids),
        ):
            _canonical_texts(values, label, refs=EventRef)
        if not self.erased_event_ids:
            raise FactorFrontierError("invalidation planning requires at least one erased event ID")
        for label, values in (
            ("affected_factor_ids", self.affected_factor_ids),
            ("invalidated_active_factor_ids", self.invalidated_active_factor_ids),
        ):
            _canonical_texts(values, label)
            for factor_id in values:
                _digest(factor_id, label)
        if not set(self.erased_event_ids) <= set(self.affected_event_ids):
            raise FactorFrontierError("erased events must be included in the affected-event closure")
        if not set(self.invalidated_active_factor_ids) <= set(self.affected_factor_ids):
            raise FactorFrontierError("active invalidation must be a subset of affected factors")
        _nonnegative_int(self.events_examined, "events_examined")
        _nonnegative_int(self.event_edges_examined, "event_edges_examined")
        if self.events_examined > self.caps.max_events:
            raise FactorFrontierCapExceeded("invalidation plan exceeds the declared event cap")
        if self.event_edges_examined > self.caps.max_edges:
            raise FactorFrontierCapExceeded("invalidation plan exceeds the declared edge cap")
        if len(self.affected_event_ids) > self.caps.max_events:
            raise FactorFrontierCapExceeded("invalidation plan exceeds the declared affected-event cap")
        if (
            len(self.affected_factor_ids) > self.caps.max_nodes
            or len(self.invalidated_active_factor_ids) > self.caps.max_nodes
        ):
            raise FactorFrontierCapExceeded("invalidation plan exceeds the declared affected-factor cap")
        expected = _invalidation_work(
            events=self.events_examined,
            event_edges=self.event_edges_examined,
            affected_events=len(self.affected_event_ids),
            affected_factors=len(self.affected_factor_ids),
        )
        if self.work != expected:
            raise FactorFrontierError("invalidation work is not exactly count-derived")
        if self.work.total_work > self.caps.max_work_units:
            raise FactorFrontierCapExceeded("invalidation plan exceeds the declared work cap")
        _require_false(self.accounting_applied, "accounting_applied")
        _require_false(self.archive_deletion_verified, "archive_deletion_verified")
        _require_false(self.application_authorized, "application_authorized")
        _require_false(self.activation_enabled, "activation_enabled")
        _require_false(self.runtime_consumable, "runtime_consumable")
        _require_false(self.factual_write_authorized, "factual_write_authorized")
        _require_false(self.scientific_promotion_allowed, "scientific_promotion_allowed")
        _digest(self.plan_sha256, "plan_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.plan_sha256:
            raise FactorFrontierError("invalidation plan self-hash mismatch")

    @classmethod
    def _issue(
        cls,
        *,
        snapshot: ShadowCausalFactorFrontier,
        caps: FactorFrontierCaps,
        erased_event_ids: Sequence[str],
        affected_event_ids: Sequence[str],
        affected_factor_ids: Sequence[str],
        invalidated_active_factor_ids: Sequence[str],
        events_examined: int,
        event_edges_examined: int,
        _issuance_token: object,
    ) -> Self:
        if _issuance_token is not _INVALIDATION_PLAN_ISSUANCE_TOKEN:
            raise FactorFrontierError("invalidation-plan issuance token is invalid")
        erased = tuple(
            sorted(
                _string_list(
                    erased_event_ids,
                    "erased_event_ids",
                    max_items=MAX_HARD_EVENTS,
                )
            )
        )
        affected_events = tuple(
            sorted(
                _string_list(
                    affected_event_ids,
                    "affected_event_ids",
                    max_items=MAX_HARD_EVENTS,
                )
            )
        )
        affected_factors = tuple(
            sorted(
                _string_list(
                    affected_factor_ids,
                    "affected_factor_ids",
                    max_items=MAX_HARD_NODES,
                )
            )
        )
        invalidated = tuple(
            sorted(
                _string_list(
                    invalidated_active_factor_ids,
                    "invalidated_active_factor_ids",
                    max_items=MAX_HARD_NODES,
                )
            )
        )
        work = _invalidation_work(
            events=events_examined,
            event_edges=event_edges_examined,
            affected_events=len(affected_events),
            affected_factors=len(affected_factors),
        )
        core = {
            "schema": FACTOR_INVALIDATION_PLAN_SCHEMA,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "source_ledger_sha256": snapshot.source_ledger_sha256,
            "caps": caps.payload(),
            "erased_event_ids": list(erased),
            "affected_event_ids": list(affected_events),
            "affected_factor_ids": list(affected_factors),
            "invalidated_active_factor_ids": list(invalidated),
            "events_examined": events_examined,
            "event_edges_examined": event_edges_examined,
            "work": work.payload(),
            "accounting_applied": False,
            "archive_deletion_verified": False,
            "application_authorized": False,
            "activation_enabled": False,
            "runtime_consumable": False,
            "factual_write_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            snapshot_sha256=snapshot.snapshot_sha256,
            source_ledger_sha256=snapshot.source_ledger_sha256,
            caps=caps,
            erased_event_ids=erased,
            affected_event_ids=affected_events,
            affected_factor_ids=affected_factors,
            invalidated_active_factor_ids=invalidated,
            events_examined=events_examined,
            event_edges_examined=event_edges_examined,
            work=work,
            accounting_applied=False,
            archive_deletion_verified=False,
            application_authorized=False,
            activation_enabled=False,
            runtime_consumable=False,
            factual_write_authorized=False,
            scientific_promotion_allowed=False,
            plan_sha256=canonical_sha256(core),
            _issuance_token=_issuance_token,
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "snapshot_sha256": self.snapshot_sha256,
            "source_ledger_sha256": self.source_ledger_sha256,
            "caps": self.caps.payload(),
            "erased_event_ids": list(self.erased_event_ids),
            "affected_event_ids": list(self.affected_event_ids),
            "affected_factor_ids": list(self.affected_factor_ids),
            "invalidated_active_factor_ids": list(self.invalidated_active_factor_ids),
            "events_examined": self.events_examined,
            "event_edges_examined": self.event_edges_examined,
            "work": self.work.payload(),
            "accounting_applied": self.accounting_applied,
            "archive_deletion_verified": self.archive_deletion_verified,
            "application_authorized": self.application_authorized,
            "activation_enabled": self.activation_enabled,
            "runtime_consumable": self.runtime_consumable,
            "factual_write_authorized": self.factual_write_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["plan_sha256"] = self.plan_sha256
        return result


def plan_shadow_invalidation(
    snapshot: ShadowCausalFactorFrontier,
    *,
    erased_event_ids: Sequence[str],
    ledger: EventLedger,
    caps: FactorFrontierCaps,
) -> FactorInvalidationPlan:

    if type(snapshot) is not ShadowCausalFactorFrontier:
        raise FactorFrontierError("invalidation requires an exact shadow frontier")
    if type(caps) is not FactorFrontierCaps:
        raise FactorFrontierError("invalidation requires exact FactorFrontierCaps")
    if type(ledger) is not EventLedger:
        raise FactorFrontierError("invalidation source must be an exact EventLedger")
    erased = tuple(
        sorted(
            _string_list(
                erased_event_ids,
                "erased_event_ids",
                max_items=MAX_HARD_EVENTS,
            )
        )
    )
    _canonical_texts(erased, "erased_event_ids", refs=EventRef)
    if not erased:
        raise FactorFrontierError("invalidation planning requires erased event IDs")
    if ledger.entry_count > caps.max_events:
        raise FactorFrontierCapExceeded("source ledger exceeds the event cap")
    replay = _authoritative_replay(ledger, caps)
    problems = verify_shadow_factor_frontier(snapshot, ledger=replay, caps=caps)
    if problems:
        raise FactorFrontierError("invalidation snapshot authority failed: " + "; ".join(problems))
    known = set(replay.event_ids)
    if not set(erased) <= known:
        raise FactorFrontierError("invalidation names an event outside the source ledger")

    children: dict[str, list[str]] = {}
    event_edges = 0
    for event in replay.events:
        child_id = str(event.event_id)
        for parent_id in event.envelope.causal_parent_ids:
            children.setdefault(str(parent_id), []).append(child_id)
            event_edges += 1
            if event_edges > caps.max_edges:
                raise FactorFrontierCapExceeded("invalidation event edges exceed the declared cap")
    affected = set(erased)
    queue = deque(erased)
    while queue:
        event_id = queue.popleft()
        for child_id in sorted(children.get(event_id, ())):
            if child_id in affected:
                continue
            affected.add(child_id)
            queue.append(child_id)
    affected_factors = tuple(
        sorted(node.factor_id for node in snapshot.nodes if node.source_hypothesis_event_id in affected)
    )
    active = set(snapshot.active_factor_ids)
    invalidated = tuple(sorted(active & set(affected_factors)))
    plan = FactorInvalidationPlan._issue(
        snapshot=snapshot,
        caps=caps,
        erased_event_ids=erased,
        affected_event_ids=tuple(affected),
        affected_factor_ids=affected_factors,
        invalidated_active_factor_ids=invalidated,
        events_examined=replay.entry_count,
        event_edges_examined=event_edges,
        _issuance_token=_INVALIDATION_PLAN_ISSUANCE_TOKEN,
    )
    if plan.work.total_work > caps.max_work_units:
        raise FactorFrontierCapExceeded("invalidation work exceeds the declared cap")
    return plan


__all__ = [
    "ACTIVATION_ENABLED",
    "FACTOR_FRONTIER_CAPS_SCHEMA",
    "FACTOR_FRONTIER_SCHEMA",
    "FACTOR_INVALIDATION_PLAN_SCHEMA",
    "FACTOR_NODE_SCHEMA",
    "FACTOR_PROJECTION_RECEIPT_SCHEMA",
    "FACTOR_QUERY_RECEIPT_SCHEMA",
    "FACTOR_QUERY_SCHEMA",
    "FACTUAL_WRITE_AUTHORIZED",
    "RUNTIME_CONSUMABLE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "FactorFrontierCapExceeded",
    "FactorFrontierCaps",
    "FactorFrontierError",
    "FactorFrontierQuery",
    "FactorInvalidationPlan",
    "FrontierProjectionReceipt",
    "FrontierQueryReceipt",
    "ShadowCausalFactorFrontier",
    "ShadowFactorNode",
    "plan_shadow_invalidation",
    "project_shadow_factor_frontier",
    "query_shadow_factor_frontier",
    "verify_shadow_factor_frontier",
]
