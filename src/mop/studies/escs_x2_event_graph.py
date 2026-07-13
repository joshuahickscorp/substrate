"""X2 event-graph memory, causal retrieval, and action-value experiment scaffold.

The generated fixtures in this module exercise ESCS event, branch, archive, revision, deletion,
poison-rejection, and lifecycle-accounting mechanics.  They are not a prediction benchmark and do
not support an intelligence, capability, efficiency, or scientific-promotion claim.  The official
configuration is activation-disabled and cannot run until its prerequisite proofs exist.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import stat
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.escs.accounting import FACTUAL_BRANCH, LifecycleLedger, WorkVector
from mop.escs.archive import BoundedArchive, PayloadErasedError, ReplayAuthority
from mop.escs.events import (
    CommitmentEvent,
    CommitmentKind,
    ConsequenceEvent,
    EpistemicStatus,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
    event_from_payload,
)
from mop.escs.ledger import EventLedger
from mop.substrate.events import BranchRef, canonical_bytes, canonical_sha256

ENVELOPE_SCHEMA = "mop-escs-x2-envelope/v1"
CONFIG_SCHEMA = "mop-escs-x2-config/v1"
AUTHORITY_SCHEMA = "mop-escs-x2-config-authority/v1"
IMPLEMENTATION_AUTHORITY_SCHEMA = "mop-escs-x2-implementation-authority/v1"
CHECKPOINT_SCHEMA = "mop-escs-x2-checkpoint/v1"
RECEIPT_SCHEMA = "mop-escs-x2-receipt/v1"
VERIFICATION_SCHEMA = "mop-escs-x2-verification/v1"
ROW_SCHEMA = "mop-escs-x2-seed-row/v1"
FIXTURE_SCHEMA = "mop-escs-x2-generated-fixture/v1"
CLAIM_SCOPE = "deterministic-generated-event-graph-mechanics-only"
OFFICIAL_CONTRACT_ID = "escs-x2-v1-2026-07-12"
OFFICIAL_CONFIG_AUTHORITY_SHA256 = "b7b8e840204a68f87cf2e2f5c392bcdff9ec4b7b9ab52d6e912f73d6c63c45a8"
OFFICIAL_IMPLEMENTATION_REVIEW_STATUS = "preregistered-scaffold-unexecuted"

ARM_NAMES = (
    "escs_event_graph",
    "fixed_recurrent",
    "bounded_raw_history",
    "episodic_kv_cache",
    "archive_only",
    "exact_global_history",
    "periodic_summary",
    "reactive_lower_bound",
    "action_blind",
    "referent_shuffled",
    "random_graph",
    "shuffled_graph",
    "oracle_state_nonpromotable",
)
ORACLE_ARMS = frozenset({"oracle_state_nonpromotable"})
LEARNED_COMPARATORS = (
    "fixed_recurrent",
    "bounded_raw_history",
    "episodic_kv_cache",
    "archive_only",
    "exact_global_history",
)
WORK_COMPONENTS = (
    "raw_transport_and_adapters",
    "event_formation",
    "indexing_and_graph_maintenance",
    "dispatch_and_exploration",
    "actor_execution",
    "messages",
    "counterfactual_credit",
    "learning",
    "archival_and_erasure",
    "retained_byte_time",
    "idle_floor",
    "serialization_and_receipts",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiment/escs_x2_event_graph.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X2_EVENT_GRAPH.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "proof/ESCS_X2_EVENT_GRAPH.checkpoint.json"
DEFAULT_VERIFICATION_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X2_EVENT_GRAPH.verification.json"
DEFAULT_VERIFICATION_CHECKPOINT_PATH = REPO_ROOT / "proof/ESCS_X2_EVENT_GRAPH.verification.checkpoint.json"
DEFAULT_IMPLEMENTATION_AUTHORITY_PATH = REPO_ROOT / "proof/ESCS_X2_EVENT_GRAPH.implementation-authority.json"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_SCOPED_FILE_BYTES = 64 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stable_int(*parts: Any, modulus: int = 2**63 - 1) -> int:
    return int.from_bytes(hashlib.sha256(canonical_bytes(list(parts))).digest()[:8], "big") % modulus


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_regular(path: Path, *, max_bytes: int, label: str) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        _require(before.st_size <= max_bytes, f"{label} exceeds byte envelope")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    _require(identity_before == identity_after, f"{label} changed during read")
    raw = b"".join(chunks)
    _require(len(raw) == before.st_size, f"{label} size changed during read")
    return raw


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = _read_regular(path, max_bytes=MAX_ARTIFACT_BYTES, label=label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _sealed(payload: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    document = dict(payload)
    document[digest_field] = canonical_sha256(document)
    return document


def _verify_self_hash(document: Mapping[str, Any], digest_field: str, label: str) -> None:
    body = dict(document)
    claimed = body.pop(digest_field, None)
    _require(isinstance(claimed, str), f"{label} is missing {digest_field}")
    _require(claimed == canonical_sha256(body), f"{label} self-hash mismatch")


def _path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_receipt(path: Path) -> dict[str, Any]:
    raw = _read_regular(path, max_bytes=MAX_SCOPED_FILE_BYTES, label=f"scoped file {path}")
    return {"path": _path_label(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    record_id: str
    session_id: str
    schema_id: str
    referent_id: str
    action: str
    utility_milli: int
    tick: int
    factual: bool
    poisoned: bool
    supersedes_record_id: str | None
    deletion_target_id: str | None
    factor_slot: str
    factor_value_milli: int

    def payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "schema_id": self.schema_id,
            "referent_id": self.referent_id,
            "action": self.action,
            "utility_milli": self.utility_milli,
            "tick": self.tick,
            "factual": self.factual,
            "poisoned": self.poisoned,
            "supersedes_record_id": self.supersedes_record_id,
            "deletion_target_id": self.deletion_target_id,
            "factor_slot": self.factor_slot,
            "factor_value_milli": self.factor_value_milli,
        }


@dataclass(frozen=True, slots=True)
class ActionQuery:
    query_id: str
    session_id: str
    schema_id: str
    referent_id: str
    factor_slot: str
    factor_value_milli: int
    action_values_milli: tuple[tuple[str, int], ...]
    ambiguous: bool

    @property
    def best_actions(self) -> frozenset[str]:
        best = max(value for _, value in self.action_values_milli)
        return frozenset(action for action, value in self.action_values_milli if value == best)

    def payload(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "schema_id": self.schema_id,
            "referent_id": self.referent_id,
            "factor_slot": self.factor_slot,
            "factor_value_milli": self.factor_value_milli,
            "action_values_milli": [[action, value] for action, value in self.action_values_milli],
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True, slots=True)
class GeneratedFixture:
    seed: int
    split: str
    actions: tuple[str, ...]
    records: tuple[MemoryRecord, ...]
    queries: tuple[ActionQuery, ...]
    schema_families: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        body = {
            "schema": FIXTURE_SCHEMA,
            "seed": self.seed,
            "split": self.split,
            "actions": list(self.actions),
            "records": [record.payload() for record in self.records],
            "queries": [query.payload() for query in self.queries],
            "schema_families": list(self.schema_families),
        }
        return {**body, "payload_sha256": canonical_sha256(body)}


def _action_values(
    actions: tuple[str, ...], *, seed: int, referent_index: int, ambiguous: bool
) -> dict[str, int]:
    winner = (seed + referent_index * 3) % len(actions)
    values = {action: 100 + 170 * ((winner - index) % len(actions)) for index, action in enumerate(actions)}
    values[actions[winner]] = 1000
    if ambiguous:
        second = max(
            (action for action in actions if action != actions[winner]),
            key=lambda action: values[action],
        )
        values[second] = 1000
    return values


def generate_fixture(config: Mapping[str, Any], *, seed: int, split: str) -> GeneratedFixture:
    """Generate session-disjoint mechanics with revisions, attacks, branches, and schema changes."""

    mechanics = config["mechanics"]
    actions = tuple(str(value) for value in mechanics["actions"])
    schema_families = tuple(str(value) for value in config["splits"][split]["schema_families"])
    prefix = str(config["splits"][split]["session_prefix"])
    referent_count = int(mechanics["referents_per_world"])
    session_count = int(mechanics["session_count"])
    rng = random.Random(_stable_int("x2-fixture", seed, split))
    records: list[MemoryRecord] = []
    tick = 0

    # Training sessions are distinct from the query session.  Stable referent identities are the
    # only cross-session link; schema-local slot names are never translated by an oracle map.
    for session_index in range(session_count - 1):
        session_id = f"session:{prefix}-{seed}-{session_index}"
        schema = "canonical" if session_index == 0 else schema_families[session_index % len(schema_families)]
        for referent_index in range(referent_count):
            referent = f"referent:role-{referent_index}"
            values = _action_values(actions, seed=seed, referent_index=referent_index, ambiguous=False)
            for action_index, action in enumerate(actions):
                record_id = f"record:{seed}/{session_index}/{referent_index}/{action_index}/base"
                utility = values[action]
                # One stale value is explicitly corrected by a later immutable revision.
                stale = action_index == (referent_index + 1) % len(actions)
                if stale:
                    utility = max(0, utility - 430)
                records.append(
                    MemoryRecord(
                        record_id=record_id,
                        session_id=session_id,
                        schema_id=schema,
                        referent_id=referent,
                        action=action,
                        utility_milli=utility,
                        tick=tick,
                        factual=True,
                        poisoned=False,
                        supersedes_record_id=None,
                        deletion_target_id=None,
                        factor_slot=f"slot:{schema}-{(referent_index + action_index) % 7}",
                        factor_value_milli=rng.randrange(1000),
                    )
                )
                tick += 1
                if stale:
                    records.append(
                        MemoryRecord(
                            record_id=f"{record_id}/revision",
                            session_id=session_id,
                            schema_id=schema,
                            referent_id=referent,
                            action=action,
                            utility_milli=values[action],
                            tick=tick,
                            factual=True,
                            poisoned=False,
                            supersedes_record_id=record_id,
                            deletion_target_id=None,
                            factor_slot=f"slot:{schema}-{(referent_index + action_index) % 7}",
                            factor_value_milli=rng.randrange(1000),
                        )
                    )
                    tick += 1

            # Counterfactual and poisoned records would reverse the action ranking if allowed to
            # contaminate factual memory.
            winner_action = max(actions, key=lambda candidate: values[candidate])
            wrong_action = actions[(actions.index(winner_action) + 2) % len(actions)]
            for branch_index in range(int(mechanics["branch_count_per_referent"])):
                records.append(
                    MemoryRecord(
                        record_id=f"record:{seed}/{session_index}/{referent_index}/branch/{branch_index}",
                        session_id=session_id,
                        schema_id=schema,
                        referent_id=referent,
                        action=wrong_action,
                        utility_milli=5000,
                        tick=tick,
                        factual=False,
                        poisoned=False,
                        supersedes_record_id=None,
                        deletion_target_id=None,
                        factor_slot=f"slot:{schema}-branch",
                        factor_value_milli=999,
                    )
                )
                tick += 1
            records.append(
                MemoryRecord(
                    record_id=f"record:{seed}/{session_index}/{referent_index}/poison",
                    session_id=session_id,
                    schema_id=schema,
                    referent_id=referent,
                    action=wrong_action,
                    utility_milli=9000,
                    tick=tick,
                    factual=True,
                    poisoned=True,
                    supersedes_record_id=None,
                    deletion_target_id=None,
                    factor_slot=f"slot:{schema}-poison",
                    factor_value_milli=999,
                )
            )
            tick += 1

    distractor_count = int(mechanics["history_distractors"])
    for index in range(distractor_count):
        record_id = f"record:{seed}/distractor/{index}"
        records.append(
            MemoryRecord(
                record_id=record_id,
                session_id=f"session:{prefix}-{seed}-distractor",
                schema_id="irrelevant",
                referent_id=f"referent:distractor-{index}",
                action=actions[index % len(actions)],
                utility_milli=rng.randrange(1000),
                tick=tick,
                factual=True,
                poisoned=False,
                supersedes_record_id=None,
                deletion_target_id=None,
                factor_slot=f"slot:irrelevant-{index}",
                factor_value_milli=rng.randrange(1000),
            )
        )
        tick += 1
    deletion_target = records[-1].record_id
    records.append(
        MemoryRecord(
            record_id=f"record:{seed}/deletion-marker",
            session_id=f"session:{prefix}-{seed}-deletion",
            schema_id="control",
            referent_id="referent:deletion-control",
            action=actions[0],
            utility_milli=0,
            tick=tick,
            factual=True,
            poisoned=False,
            supersedes_record_id=None,
            deletion_target_id=deletion_target,
            factor_slot="slot:deletion",
            factor_value_milli=0,
        )
    )

    query_count = int(mechanics["queries_per_seed"])
    queries: list[ActionQuery] = []
    query_session = f"session:{prefix}-{seed}-heldout"
    for index in range(query_count):
        referent_index = index % referent_count
        ambiguous = index % 7 == 0
        values = _action_values(actions, seed=seed, referent_index=referent_index, ambiguous=ambiguous)
        schema = schema_families[index % len(schema_families)]
        queries.append(
            ActionQuery(
                query_id=f"query:{seed}/{index}",
                session_id=query_session,
                schema_id=schema,
                referent_id=f"referent:role-{referent_index}",
                factor_slot=f"slot:{schema}-{(referent_index * 2 + index) % 11}",
                factor_value_milli=-1 if ambiguous else rng.randrange(1000),
                action_values_milli=tuple((action, values[action]) for action in actions),
                ambiguous=ambiguous,
            )
        )
    return GeneratedFixture(seed, split, actions, tuple(records), tuple(queries), schema_families)


def _event_source(seed: int, label: str) -> dict[str, Any]:
    return {"producer": "escs-x2-generated-fixture", "seed": seed, "label": label}


def build_escs_mechanics_fixture(seed: int) -> dict[str, Any]:
    """Exercise the repository's authoritative ESCS identity, branch, archive, and charge dialect."""

    ledger = EventLedger()
    charges = LifecycleLedger()
    source = _event_source(seed, "factual")
    observation = ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:x2-{seed}-0",),
        adapter_version="x2-generated-adapter/v1",
        sensor_scope={"sensor": "generated-async", "session": f"session:x2-{seed}"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=2),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(event_formation=1),
    )
    ledger.append(observation)
    hypothesis = HypothesisEvent.create(
        origin=HypothesisOrigin.ACTOR,
        epistemic_status=EpistemicStatus.INFERRED,
        referent_hypotheses={"referent:role-0": 1.0},
        factor_change_distribution={"slot:unknown": 1.0},
        decision_relevance_distribution={"advance": 0.6, "wait": 0.4},
        reducibility_distribution={"action_consequence": 1.0},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.6,
        abstention_reason=None,
        predicted_value_of_further_computation=0.1,
        causal_parent_ids=(observation.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(indexing_and_graph_maintenance=2),
    )
    ledger.append(hypothesis)
    commitment = CommitmentEvent.create(
        coalition_id="coalition:x2-fixture",
        commitment_kind=CommitmentKind.EXTERNAL_ACTION,
        committed_payload={"action": "advance"},
        decision_distribution={"advance": 0.6, "wait": 0.4},
        deadline_tick=4,
        predicted_utility_vector={"action_value_milli": 600},
        predicted_full_cost=WorkVector(actor_execution=1),
        causal_parent_ids=(hypothesis.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(messages=1),
    )
    ledger.append(commitment)
    consequence = ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome={"action_value_milli": 400, "revision": 0},
        realized_utility_vector={"action_value_milli": 400},
        delayed_or_partial=True,
        observation_uncertainty=0.2,
        realized_full_cost=WorkVector(actor_execution=1),
        causal_parent_ids=(commitment.event_id,),
        clock_start_tick=3,
        clock_end_tick=3,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(event_formation=1),
    )
    ledger.append(consequence)
    revised = ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome={"action_value_milli": 900, "revision": 1},
        realized_utility_vector={"action_value_milli": 900},
        delayed_or_partial=False,
        observation_uncertainty=0.0,
        realized_full_cost=WorkVector(actor_execution=1),
        causal_parent_ids=(commitment.event_id, consequence.event_id),
        clock_start_tick=4,
        clock_end_tick=4,
        source_and_provenance=_event_source(seed, "revision"),
        measured_creation_cost=WorkVector(indexing_and_graph_maintenance=2),
        supersedes_event_ids=(consequence.event_id,),
    )
    ledger.append(revised)

    branch = BranchRef(f"branch:x2-{seed}-counterfactual")
    simulated = HypothesisEvent.create(
        origin=HypothesisOrigin.ACTOR,
        epistemic_status=EpistemicStatus.SIMULATED,
        referent_hypotheses={"referent:role-0": 1.0},
        factor_change_distribution={"intervention:divert": 1.0},
        decision_relevance_distribution={"divert": 1.0},
        reducibility_distribution={"counterfactual": 1.0},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.5,
        abstention_reason=None,
        predicted_value_of_further_computation=0.2,
        causal_parent_ids=(observation.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance=_event_source(seed, "counterfactual"),
        measured_creation_cost=WorkVector(counterfactual_credit=1),
    )
    ledger.append(simulated)
    branch_commitment = CommitmentEvent.create(
        coalition_id="coalition:x2-fixture",
        commitment_kind=CommitmentKind.EXTERNAL_ACTION,
        committed_payload={"action": "divert"},
        decision_distribution={"divert": 1.0},
        deadline_tick=4,
        predicted_utility_vector={"action_value_milli": 999},
        predicted_full_cost=WorkVector(counterfactual_credit=1),
        causal_parent_ids=(simulated.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance=_event_source(seed, "counterfactual"),
    )
    ledger.append(branch_commitment)
    branch_consequence = ConsequenceEvent.create(
        commitment_event_id=branch_commitment.event_id,
        observed_outcome={"action_value_milli": 999, "simulated": True},
        realized_utility_vector={"action_value_milli": 999},
        delayed_or_partial=False,
        observation_uncertainty=0.0,
        realized_full_cost=WorkVector(counterfactual_credit=1),
        causal_parent_ids=(branch_commitment.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=3,
        clock_end_tick=3,
        source_and_provenance=_event_source(seed, "counterfactual"),
    )
    ledger.append(branch_consequence)

    charges.charge(
        owner="x2:event-intake",
        reason="raw-observation-and-event-formation",
        work=WorkVector(raw_transport_and_adapters=2, event_formation=1),
        start_tick=0,
        end_tick=0,
        causal_event_ids=(observation.event_id,),
    )
    charges.charge(
        owner="x2:graph",
        reason="causal-index-and-revision",
        work=WorkVector(indexing_and_graph_maintenance=5, actor_execution=2, messages=1),
        start_tick=1,
        end_tick=4,
        causal_event_ids=(hypothesis.event_id, revised.event_id),
    )
    charges.charge(
        owner="x2:counterfactual",
        reason="isolated-branch-mechanics",
        work=WorkVector(counterfactual_credit=3),
        start_tick=1,
        end_tick=3,
        branch_id=branch,
        causal_event_ids=(simulated.event_id, branch_commitment.event_id, branch_consequence.event_id),
    )
    charges.charge_idle(
        owner="x2:boundary",
        reason="idle-adapter-floor",
        idle_work=1,
        start_tick=5,
        end_tick=6,
    )

    archive = BoundedArchive(max_hot_bytes=512, max_hot_age_ticks=2, max_cache_bytes=128)
    archived: list[str] = []
    for tick, event in enumerate(ledger.events):
        payload = canonical_bytes(event.body_payload())
        archive.append(event.envelope.payload(), payload, admitted_tick=tick)
        archived.append(str(event.event_id))
    archive.compact(len(archived), force=True)
    retrieved = archive.retrieve(archived[0], current_tick=len(archived) + 1)
    _require(retrieved == canonical_bytes(observation.body_payload()), "archive retrieval drift")
    erased_id = archived[-1]
    archive.erase_payload(erased_id, deletion_tick=len(archived) + 2, reason_code="x2-test-deletion")
    deletion_rejected = False
    try:
        archive.retrieve(erased_id)
    except PayloadErasedError:
        deletion_rejected = True

    poisoned_payload = copy.deepcopy(revised.payload())
    poisoned_payload["body"]["observed_outcome"]["value"]["action_value_milli"] = 9999
    poison_rejected = False
    try:
        event_from_payload(poisoned_payload)
    except ValueError:
        poison_rejected = True

    replay = EventLedger.replay(ledger.payload())
    replay_charges = LifecycleLedger.replay(charges.payload())
    checks = {
        "event_ledger_replay": replay.sha256 == ledger.sha256 and not ledger.verify(),
        "lifecycle_ledger_replay": replay_charges.sha256 == charges.sha256
        and not charges.verify(event_ids=set(ledger.event_ids)),
        "revision_supersession": revised.envelope.supersedes_event_ids == (consequence.event_id,),
        "counterfactual_branch_isolation": all(
            event.branch_id == branch for event in ledger.events_on_branch(branch)
        )
        and not any(event.branch_id == branch for event in ledger.events_on_branch(FACTUAL_BRANCH)),
        "archive_integrity": not archive.audit(),
        "cold_retrieval": bool(retrieved),
        "deletion_nonretrievability": deletion_rejected
        and erased_id not in archive.payload_index_event_ids
        and erased_id not in archive.cached_event_ids,
        "replay_authority_loss_declared": archive.replay_authority is ReplayAuthority.DISABLED_AFTER_ERASURE,
        "poison_digest_rejected": poison_rejected,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "event_ledger_sha256": ledger.sha256,
        "event_count": ledger.entry_count,
        "factual_event_count": len(ledger.events_on_branch(FACTUAL_BRANCH)),
        "counterfactual_event_count": len(ledger.events_on_branch(branch)),
        "lifecycle_ledger_sha256": charges.sha256,
        "lifecycle_work": charges.total.payload(),
        "archive_accounting": archive.accounting_snapshot.payload(),
        "archive_retained_bytes": archive.retained_bytes,
        "archive_replay_authority": archive.replay_authority.value,
        "erased_event_id": erased_id,
    }


def _active_records(records: Sequence[MemoryRecord], *, strict: bool) -> list[MemoryRecord]:
    deleted = {row.deletion_target_id for row in records if row.deletion_target_id is not None}
    superseded = {row.supersedes_record_id for row in records if row.supersedes_record_id is not None}
    output = []
    for row in records:
        if row.deletion_target_id is not None:
            continue
        if row.record_id in deleted or row.record_id in superseded:
            continue
        if strict and (row.poisoned or not row.factual):
            continue
        output.append(row)
    return output


def _mean_scores(records: Sequence[MemoryRecord], actions: Sequence[str]) -> dict[str, float]:
    values: dict[str, list[int]] = defaultdict(list)
    for row in records:
        values[row.action].append(row.utility_milli)
    return {action: statistics.fmean(values[action]) if values[action] else 0.0 for action in actions}


def _referent_scores(
    records: Sequence[MemoryRecord], referent_id: str, actions: Sequence[str]
) -> dict[str, float]:
    return _mean_scores([row for row in records if row.referent_id == referent_id], actions)


def _choose(scores: Mapping[str, float], actions: tuple[str, ...]) -> tuple[str, float, bool]:
    ordered = sorted(actions, key=lambda action: (-scores[action], actions.index(action)))
    first, second = ordered[:2]
    scale = max(1.0, abs(scores[first]) + abs(scores[second]))
    margin = (scores[first] - scores[second]) / scale
    confidence = max(0.0, min(1.0, 0.5 + margin))
    return first, confidence, abs(scores[first] - scores[second]) <= 1e-9


def _arm_state_and_scorer(
    arm: str,
    fixture: GeneratedFixture,
    config: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, int], int]:
    """Return retained state, query scorer, operation counters, and explicit state bytes."""

    actions = fixture.actions
    all_records = list(fixture.records)
    strict_records = _active_records(all_records, strict=True)
    controls = config["controls"]
    operations = {
        "raw_transport_and_adapters": len(all_records) + len(fixture.queries),
        "event_formation": len(all_records),
        "indexing_and_graph_maintenance": 0,
        "dispatch_and_exploration": 0,
        "actor_execution": len(fixture.queries),
        "messages": 0,
        "counterfactual_credit": sum(not row.factual for row in all_records),
        "learning": 0,
        "archival_and_erasure": 0,
        "retained_byte_time": 0,
        "idle_floor": max(1, int(config["mechanics"]["session_count"])),
        "serialization_and_receipts": len(fixture.queries),
    }
    state: Any
    scorer: Any

    if arm == "escs_event_graph":
        state = {}
        for row in strict_records:
            state[(row.referent_id, row.action)] = row
        operations["indexing_and_graph_maintenance"] = 2 * len(all_records)
        operations["messages"] = len(strict_records)
        operations["archival_and_erasure"] = len(all_records) + 3

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(actions) + 1
            scores = {
                action: float(state.get((query.referent_id, action), _ZERO_RECORD).utility_milli)
                for action in actions
            }
            if query.factor_value_milli < 0:
                ranked = sorted(actions, key=lambda action: scores[action], reverse=True)
                scores[ranked[1]] = scores[ranked[0]]
            return scores

    elif arm == "archive_only":
        state = tuple(strict_records)
        operations["archival_and_erasure"] = 2 * len(all_records)

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(state)
            operations["archival_and_erasure"] += len(state)
            return _referent_scores(state, query.referent_id, actions)

    elif arm == "exact_global_history":
        state = tuple(strict_records)

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(state)
            return _referent_scores(state, query.referent_id, actions)

    elif arm == "bounded_raw_history":
        state = tuple(all_records[-int(controls["bounded_history_records"]) :])

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(state)
            return _referent_scores(state, query.referent_id, actions)

    elif arm == "episodic_kv_cache":
        state = {}
        for row in all_records:
            state[(row.schema_id, row.referent_id, row.action)] = row
            if len(state) > int(controls["kv_capacity"]):
                del state[next(iter(state))]
        operations["indexing_and_graph_maintenance"] = len(all_records)

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(actions)
            return {
                action: float(
                    state.get((query.schema_id, query.referent_id, action), _ZERO_RECORD).utility_milli
                )
                for action in actions
            }

    elif arm == "fixed_recurrent":
        slots = int(controls["fixed_recurrent_slots"])
        state = [[0.0 for _ in actions] for _ in range(slots)]
        counts = [[0 for _ in actions] for _ in range(slots)]
        for row in all_records:
            slot = _stable_int("recurrent-slot", row.referent_id, modulus=slots)
            action_index = actions.index(row.action)
            state[slot][action_index] = 0.7 * state[slot][action_index] + 0.3 * row.utility_milli
            counts[slot][action_index] += 1
        operations["learning"] = len(all_records) * 4

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(actions)
            slot = _stable_int("recurrent-slot", query.referent_id, modulus=slots)
            return {action: state[slot][index] for index, action in enumerate(actions)}

    elif arm == "periodic_summary":
        interval = int(controls["periodic_summary_interval"])
        selected = all_records[::interval]
        state = _mean_scores(selected, actions)
        operations["learning"] = len(selected)

        def scorer(query: ActionQuery) -> dict[str, float]:
            del query
            operations["dispatch_and_exploration"] += len(actions)
            return dict(state)

    elif arm == "reactive_lower_bound":
        state = {"rule": "current-factor-modulo"}

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += 1
            chosen = query.factor_value_milli % len(actions)
            return {action: float(index == chosen) for index, action in enumerate(actions)}

    elif arm == "action_blind":
        state = {"constant": 0.0}

        def scorer(query: ActionQuery) -> dict[str, float]:
            del query
            operations["dispatch_and_exploration"] += 1
            return {action: 0.0 for action in actions}

    elif arm == "referent_shuffled":
        state = tuple(strict_records)
        referents = tuple(
            f"referent:role-{index}" for index in range(int(config["mechanics"]["referents_per_world"]))
        )

        def scorer(query: ActionQuery) -> dict[str, float]:
            index = referents.index(query.referent_id)
            shuffled = referents[(index + 1) % len(referents)]
            operations["dispatch_and_exploration"] += len(state)
            return _referent_scores(state, shuffled, actions)

    elif arm in {"random_graph", "shuffled_graph"}:
        state = tuple(strict_records)

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(actions)
            if arm == "random_graph":
                return {
                    action: float(
                        _stable_int("random-graph", fixture.seed, query.query_id, action, modulus=1000)
                    )
                    for action in actions
                }
            base = _referent_scores(state, query.referent_id, actions)
            return {action: base[actions[(index + 1) % len(actions)]] for index, action in enumerate(actions)}

    elif arm == "oracle_state_nonpromotable":
        state = {"oracle": True}

        def scorer(query: ActionQuery) -> dict[str, float]:
            operations["dispatch_and_exploration"] += len(actions)
            return {action: float(value) for action, value in query.action_values_milli}

    else:
        raise ValueError(f"unsupported X2 arm {arm!r}")

    state_bytes = len(canonical_bytes(state_payload(state)))
    operations["retained_byte_time"] = state_bytes * int(config["mechanics"]["session_count"])
    return state, scorer, operations, state_bytes


_ZERO_RECORD = MemoryRecord(
    "record:zero",
    "session:zero",
    "schema:zero",
    "referent:zero",
    "wait",
    0,
    0,
    True,
    False,
    None,
    None,
    "slot:zero",
    0,
)


def state_payload(value: Any) -> Any:
    if isinstance(value, MemoryRecord):
        return value.payload()
    if isinstance(value, dict):
        return {str(key): state_payload(nested) for key, nested in value.items()}
    if isinstance(value, tuple | list):
        return [state_payload(nested) for nested in value]
    return value


def evaluate_arm(
    arm: str,
    fixture: GeneratedFixture,
    config: Mapping[str, Any],
    mechanics_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    state, scorer, work, state_bytes = _arm_state_and_scorer(arm, fixture, config)
    del state
    correct = 0
    utility_total = 0.0
    brier_total = 0.0
    ambiguity_correct = 0
    ambiguous_count = 0
    decisions: list[dict[str, Any]] = []
    for query in fixture.queries:
        scores = scorer(query)
        predicted, confidence, tied = _choose(scores, fixture.actions)
        is_correct = predicted in query.best_actions
        correct += int(is_correct)
        oracle_values = dict(query.action_values_milli)
        best_value = max(oracle_values.values())
        utility_total += oracle_values[predicted] / max(1, best_value)
        brier_total += (confidence - float(is_correct)) ** 2
        if query.ambiguous:
            ambiguous_count += 1
            ambiguity_correct += int(tied or confidence <= 0.55)
        decisions.append(
            {
                "query_id": query.query_id,
                "predicted_action": predicted,
                "best_actions": sorted(query.best_actions),
                "correct": is_correct,
                "confidence": confidence,
                "abstention_signaled": tied or confidence <= 0.55,
                "realized_action_value_milli": oracle_values[predicted],
                "scores_sha256": canonical_sha256(scores),
            }
        )
    count = len(fixture.queries)
    serialized_bytes = len(canonical_bytes(decisions)) + len(canonical_bytes(state_bytes))
    work["serialization_and_receipts"] += max(1, serialized_bytes // 64)
    operation_work = sum(
        value for component, value in work.items() if component not in {"retained_byte_time"}
    )

    strict = _active_records(fixture.records, strict=True)
    active_ids = {record.record_id for record in strict}
    superseded_ids = {
        record.supersedes_record_id for record in fixture.records if record.supersedes_record_id is not None
    }
    deleted_ids = {
        record.deletion_target_id for record in fixture.records if record.deletion_target_id is not None
    }
    graph_integrity = arm == "escs_event_graph"
    integrity = {
        "identity_revision_correctness": graph_integrity
        and not bool(active_ids & superseded_ids)
        and bool(superseded_ids),
        "stale_memory_excluded": graph_integrity and not bool(active_ids & superseded_ids),
        "poisoning_recovery": graph_integrity and not any(record.poisoned for record in strict),
        "deletion_completeness": graph_integrity
        and not bool(active_ids & deleted_ids)
        and bool(mechanics_receipt["checks"]["deletion_nonretrievability"]),
        "branch_counterfactual_isolation": graph_integrity
        and not any(not record.factual for record in strict)
        and bool(mechanics_receipt["checks"]["counterfactual_branch_isolation"]),
        "compaction_parity": graph_integrity and bool(mechanics_receipt["checks"]["archive_integrity"]),
        "provenance_complete": graph_integrity and bool(mechanics_receipt["checks"]["event_ledger_replay"]),
        "schema_transfer_without_oracle_map": graph_integrity
        and all(query.schema_id != "canonical" for query in fixture.queries),
        "replay_authority_status_explicit": graph_integrity
        and bool(mechanics_receipt["checks"]["replay_authority_loss_declared"]),
        "calibrated_abstention_fixture": graph_integrity
        and (ambiguity_correct == ambiguous_count)
        and ambiguous_count > 0,
    }
    result: dict[str, Any] = {
        "arm": arm,
        "evidence_standing": (
            "oracle_nonpromotable"
            if arm in ORACLE_ARMS
            else "candidate_unverified"
            if arm == "escs_event_graph"
            else "control_only"
        ),
        "activation_enabled": False,
        "scientific_promotion": False,
        "prediction_only_claim": False,
        "query_count": count,
        "heldout_intervention_ranking_accuracy": correct / count,
        "heldout_realized_action_value": utility_total / count,
        "calibration_brier": brier_total / count,
        "ambiguous_query_count": ambiguous_count,
        "abstention_accuracy": ambiguity_correct / max(1, ambiguous_count),
        "work_components": work,
        "abstract_operation_work": operation_work,
        "retained_state_bytes": state_bytes,
        "retained_byte_time": work["retained_byte_time"],
        "serialized_bytes": serialized_bytes,
        "work_per_correct_decision": operation_work / max(1, correct),
        "bytes_per_correct_decision": (state_bytes + serialized_bytes) / max(1, correct),
        "integrity_gates": integrity,
        "all_integrity_gates_passed": all(integrity.values()),
        "decision_digest": canonical_sha256(decisions),
    }
    _require(tuple(result["work_components"]) == WORK_COMPONENTS, "work component order drift")
    return result


def difficulty_gate(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    arms = row["arms"]
    criteria = config["difficulty_gate"]
    oracle = arms["oracle_state_nonpromotable"]
    reactive = arms["reactive_lower_bound"]
    fixture = row["fixture_summary"]
    checks = {
        "oracle_accuracy": oracle["heldout_intervention_ranking_accuracy"]
        >= float(criteria["min_oracle_accuracy"]),
        "oracle_utility": oracle["heldout_realized_action_value"] >= float(criteria["min_oracle_utility"]),
        "reactive_not_saturated": reactive["heldout_intervention_ranking_accuracy"]
        <= float(criteria["max_reactive_accuracy"]),
        "all_actions_represented": (not bool(criteria["require_all_actions_represented"]))
        or fixture["all_actions_represented"],
        "attack_cases_present": (not bool(criteria["require_attack_cases"]))
        or fixture["attack_cases_present"],
        "session_disjoint": fixture["session_disjoint"],
        "schema_changed": fixture["schema_changed"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "failure_interpretation": "invalid_bed_not_mechanism_null",
    }


def run_seed(config: Mapping[str, Any], *, seed: int, split: str) -> dict[str, Any]:
    fixture = generate_fixture(config, seed=seed, split=split)
    mechanics = build_escs_mechanics_fixture(seed)
    arms = {arm: evaluate_arm(arm, fixture, config, mechanics) for arm in ARM_NAMES}
    training_sessions = {record.session_id for record in fixture.records}
    query_sessions = {query.session_id for query in fixture.queries}
    best_action_union = set().union(*(query.best_actions for query in fixture.queries))
    fixture_summary = {
        "payload_sha256": fixture.payload()["payload_sha256"],
        "record_count": len(fixture.records),
        "query_count": len(fixture.queries),
        "factual_record_count": sum(record.factual for record in fixture.records),
        "counterfactual_record_count": sum(not record.factual for record in fixture.records),
        "poison_record_count": sum(record.poisoned for record in fixture.records),
        "revision_record_count": sum(record.supersedes_record_id is not None for record in fixture.records),
        "deletion_marker_count": sum(record.deletion_target_id is not None for record in fixture.records),
        "all_actions_represented": best_action_union == set(fixture.actions),
        "attack_cases_present": all(
            any(predicate(record) for record in fixture.records)
            for predicate in (
                lambda record: record.poisoned,
                lambda record: not record.factual,
                lambda record: record.supersedes_record_id is not None,
                lambda record: record.deletion_target_id is not None,
            )
        ),
        "session_disjoint": not bool(training_sessions & query_sessions),
        "schema_changed": all(query.schema_id != "canonical" for query in fixture.queries),
        "schema_families": list(fixture.schema_families),
    }
    body: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "seed": seed,
        "split": split,
        "fixture_summary": fixture_summary,
        "escs_mechanics": mechanics,
        "arms": arms,
        "activation_enabled": False,
        "scientific_promotion": False,
    }
    body["difficulty_gate"] = difficulty_gate(body, config)
    body["row_sha256"] = canonical_sha256(body)
    return body


def _paired_lower_bound(values: Sequence[float]) -> float:
    _require(bool(values), "paired confidence bound requires observations")
    if len(values) == 1:
        return values[0]
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return mean - 1.96 * standard_error


def aggregate_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    _require(bool(rows), "X2 aggregation requires at least one row")
    arm_summary: dict[str, dict[str, float]] = {}
    for arm in ARM_NAMES:
        arm_summary[arm] = {
            metric: statistics.fmean(float(row["arms"][arm][metric]) for row in rows)
            for metric in (
                "heldout_intervention_ranking_accuracy",
                "heldout_realized_action_value",
                "calibration_brier",
                "abstention_accuracy",
                "abstract_operation_work",
                "retained_state_bytes",
                "retained_byte_time",
                "serialized_bytes",
            )
        }
    graph = arm_summary["escs_event_graph"]
    best_accuracy = max(
        arm_summary[arm]["heldout_intervention_ranking_accuracy"] for arm in LEARNED_COMPARATORS
    )
    quality_eligible = [
        arm
        for arm in LEARNED_COMPARATORS
        if arm_summary[arm]["heldout_intervention_ranking_accuracy"] >= best_accuracy - 1e-12
    ]
    strongest = min(
        quality_eligible,
        key=lambda arm: (
            arm_summary[arm]["abstract_operation_work"],
            arm_summary[arm]["retained_state_bytes"],
        ),
    )
    control = arm_summary[strongest]
    paired_accuracy_differences = [
        float(row["arms"]["escs_event_graph"]["heldout_intervention_ranking_accuracy"])
        - float(row["arms"][strongest]["heldout_intervention_ranking_accuracy"])
        for row in rows
    ]
    accuracy_gain = (
        graph["heldout_intervention_ranking_accuracy"] - control["heldout_intervention_ranking_accuracy"]
    )
    work_saving = 1.0 - graph["abstract_operation_work"] / max(1.0, control["abstract_operation_work"])
    state_saving = 1.0 - graph["retained_state_bytes"] / max(1.0, control["retained_state_bytes"])
    criteria = config["criteria"]
    all_integrity = all(row["arms"]["escs_event_graph"]["all_integrity_gates_passed"] for row in rows)
    valid_bed = all(row["difficulty_gate"]["passed"] for row in rows)
    mechanics_valid = all(row["escs_mechanics"]["all_checks_passed"] for row in rows)
    enough_paired_seeds = len(rows) >= int(criteria["min_paired_seeds"])
    gain_route = (
        enough_paired_seeds
        and accuracy_gain >= float(criteria["min_intervention_ranking_gain"])
        and graph["abstract_operation_work"] <= control["abstract_operation_work"]
        and (
            not bool(criteria["require_no_greater_state_for_gain_route"])
            or graph["retained_state_bytes"] <= control["retained_state_bytes"]
        )
        and (
            not bool(criteria["require_positive_paired_confidence_bound"])
            or _paired_lower_bound(paired_accuracy_differences) > 0.0
        )
    )
    noninferiority_route = (
        enough_paired_seeds
        and accuracy_gain >= -float(criteria["max_noninferiority_accuracy_loss"])
        and work_saving >= float(criteria["min_saving_fraction"])
        and graph["retained_state_bytes"] <= control["retained_state_bytes"]
    )
    if not mechanics_valid:
        route = "failed"
    elif not valid_bed:
        route = "invalid_bed"
    elif all_integrity and (gain_route or noninferiority_route):
        route = "positive_candidate"
    else:
        route = "controlled_null"
    return {
        "seed_count": len(rows),
        "arm_summary": arm_summary,
        "strongest_quality_control": strongest,
        "paired_accuracy_difference_mean": statistics.fmean(paired_accuracy_differences),
        "paired_accuracy_difference_lower_95": _paired_lower_bound(paired_accuracy_differences),
        "accuracy_gain": accuracy_gain,
        "work_saving_fraction": work_saving,
        "state_saving_fraction": state_saving,
        "all_integrity_gates_passed": all_integrity,
        "all_mechanics_checks_passed": mechanics_valid,
        "valid_experimental_bed": valid_bed,
        "minimum_paired_seed_gate_passed": enough_paired_seeds,
        "gain_route_passed": gain_route,
        "noninferiority_savings_route_passed": noninferiority_route,
        "terminal_route": route,
        "route_instruction": config["routing"][route],
        "prediction_only_claim": False,
        "scientific_promotion": False,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == CONFIG_SCHEMA, "unsupported X2 config schema")
    _require(config.get("claim_scope") == CLAIM_SCOPE, "X2 claim scope drift")
    _require(tuple(config.get("arms", ())) == ARM_NAMES, "X2 arm set or order drift")
    activation = config["activation"]
    if not isinstance(activation, Mapping):
        raise ValueError("X2 activation contract missing")
    _require(activation.get("enabled") is False, "X2 scaffold activation must remain disabled")
    _require(
        activation.get("scientific_promotion_allowed") is False,
        "X2 scientific promotion must remain disabled",
    )
    seeds = config["seeds"]
    fresh = config["fresh_verifier_seeds"]
    if not isinstance(seeds, list) or len(seeds) < 5:
        raise ValueError("X2 requires at least five producer seeds")
    if not isinstance(fresh, list) or len(fresh) < 5:
        raise ValueError("X2 requires at least five fresh verifier seeds")
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in [*seeds, *fresh]),
        "X2 seeds must be integers",
    )
    _require(len(set(seeds)) == len(seeds), "X2 producer seeds must be unique")
    _require(len(set(fresh)) == len(fresh), "X2 fresh seeds must be unique")
    _require(set(seeds).isdisjoint(fresh), "X2 producer and fresh seeds must be disjoint")
    mechanics = config["mechanics"]
    if not isinstance(mechanics, Mapping):
        raise ValueError("X2 mechanics contract missing")
    actions = mechanics["actions"]
    if not isinstance(actions, list) or len(actions) < 3:
        raise ValueError("X2 requires at least three actions")
    _require(len(set(actions)) == len(actions), "X2 actions must be unique")
    _require(int(mechanics.get("queries_per_seed", 0)) > 0, "X2 query count must be positive")
    criteria = config["criteria"]
    if not isinstance(criteria, Mapping):
        raise ValueError("X2 criteria missing")
    _require(
        float(criteria.get("min_intervention_ranking_gain", -1)) == 0.05,
        "X2 gain threshold must remain 0.05",
    )
    _require(
        float(criteria.get("min_saving_fraction", -1)) == 0.30,
        "X2 savings threshold must remain 0.30",
    )
    _require(
        config["verdict"]["prediction_only_claim_forbidden"] is True,
        "X2 prediction-only claim guard missing",
    )
    _require(
        config["verdict"]["scientific_promotion"] == "blocked",
        "X2 scientific promotion verdict must remain blocked",
    )
    measurement = config["measurement"]
    if not isinstance(measurement, Mapping):
        raise ValueError("X2 measurement contract missing")
    _require(tuple(measurement.get("work_components", ())) == WORK_COMPONENTS, "X2 work boundary drift")


def load_config(path: Path = DEFAULT_CONFIG_PATH, *, exploratory: bool = False) -> dict[str, Any]:
    envelope = _load_json(path, label="X2 config")
    _require(envelope.get("schema") == ENVELOPE_SCHEMA, "unsupported X2 config envelope")
    authority = envelope.get("authority")
    payload = envelope.get("payload")
    if not isinstance(authority, Mapping):
        raise ValueError("X2 config authority missing")
    if not isinstance(payload, dict):
        raise ValueError("X2 config payload missing")
    _require(authority.get("schema") == AUTHORITY_SCHEMA, "unsupported X2 config authority")
    _require(authority.get("payload_sha256") == canonical_sha256(payload), "X2 config authority mismatch")
    if not exploratory:
        _require(authority.get("mode") == "official", "official X2 run requires official config")
        _require(authority.get("contract_id") == OFFICIAL_CONTRACT_ID, "official X2 contract drift")
        _require(
            authority.get("payload_sha256") == OFFICIAL_CONFIG_AUTHORITY_SHA256,
            "official X2 config does not match frozen authority",
        )
    _validate_config(payload)
    return payload


def config_authority(path: Path) -> dict[str, Any]:
    envelope = _load_json(path, label="X2 config")
    authority = envelope.get("authority")
    if not isinstance(authority, dict):
        raise ValueError("X2 config authority missing")
    return authority


def _implementation_scoped_paths() -> tuple[Path, ...]:
    return (
        REPO_ROOT / "src/mop/studies/escs_x2_event_graph.py",
        REPO_ROOT / "scripts/run_escs_x2_event_graph.py",
        REPO_ROOT / "configs/experiment/escs_x2_event_graph.json",
        REPO_ROOT / "docs/audits/escs_x2_event_graph.md",
    )


def build_implementation_authority(
    *,
    config_authority_sha256: str,
    mode: str,
    review_status: str,
) -> dict[str, Any]:
    _require(mode in {"official", "exploratory"}, "unsupported implementation authority mode")
    body = {
        "schema": IMPLEMENTATION_AUTHORITY_SCHEMA,
        "contract_id": OFFICIAL_CONTRACT_ID if mode == "official" else "exploratory-x2",
        "mode": mode,
        "claim_scope": CLAIM_SCOPE,
        "review_status": review_status,
        "config_authority_sha256": config_authority_sha256,
        "activation_enabled": False,
        "scientific_promotion_allowed": False,
        "prediction_only_claim_allowed": False,
        "scoped_files": [_file_receipt(path) for path in _implementation_scoped_paths()],
    }
    return _sealed(body, "manifest_sha256")


def verify_implementation_authority(
    path: Path,
    *,
    expected_config_authority_sha256: str,
    expected_manifest_sha256: str | None,
    exploratory: bool,
) -> dict[str, Any]:
    manifest = _load_json(path, label="X2 implementation authority")
    _verify_self_hash(manifest, "manifest_sha256", "X2 implementation authority")
    _require(manifest.get("schema") == IMPLEMENTATION_AUTHORITY_SCHEMA, "unsupported X2 authority")
    _require(manifest.get("claim_scope") == CLAIM_SCOPE, "X2 authority claim-scope drift")
    _require(
        manifest.get("config_authority_sha256") == expected_config_authority_sha256,
        "X2 authority is bound to another config",
    )
    _require(manifest.get("activation_enabled") is False, "X2 authority must keep activation disabled")
    _require(
        manifest.get("scientific_promotion_allowed") is False,
        "X2 authority must block scientific promotion",
    )
    _require(
        manifest.get("prediction_only_claim_allowed") is False,
        "X2 authority must forbid prediction-only claims",
    )
    if expected_manifest_sha256 is not None:
        _require(
            manifest.get("manifest_sha256") == expected_manifest_sha256,
            "X2 implementation authority SHA mismatch",
        )
    if not exploratory:
        _require(manifest.get("mode") == "official", "official X2 run requires official authority")
        _require(manifest.get("contract_id") == OFFICIAL_CONTRACT_ID, "X2 authority contract drift")
        _require(
            manifest.get("review_status") == OFFICIAL_IMPLEMENTATION_REVIEW_STATUS,
            "X2 implementation review status drift",
        )
    scoped = manifest.get("scoped_files")
    _require(isinstance(scoped, list), "X2 authority scoped files missing")
    current = [_file_receipt(path) for path in _implementation_scoped_paths()]
    _require(scoped == current, "X2 implementation files differ from authority")
    return manifest


def prerequisite_receipts(config: Mapping[str, Any], *, exploratory: bool) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for row in config["prerequisites"]:
        path = REPO_ROOT / str(row["path"])
        exists = path.is_file()
        if row["required"] and not exists and not exploratory:
            raise ValueError(f"X2 blocked by missing prerequisite {row['path']}")
        receipt: dict[str, Any] = {
            "path": str(row["path"]),
            "required": bool(row["required"]),
            "exists": exists,
            "evidence_ceiling": str(row["evidence_ceiling"]),
        }
        if exists:
            receipt.update(_file_receipt(path))
        receipts.append(receipt)
    return receipts


def _checkpoint_body(
    *,
    phase: str,
    config_sha256: str,
    implementation_sha256: str,
    seeds: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "phase": phase,
        "config_authority_sha256": config_sha256,
        "implementation_authority_sha256": implementation_sha256,
        "seed_order": list(seeds),
        "completed_rows": [dict(row) for row in rows],
        "complete": complete,
        "activation_enabled": False,
        "scientific_promotion": False,
    }


def _load_checkpoint(
    path: Path,
    *,
    phase: str,
    config_sha256: str,
    implementation_sha256: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    document = _load_json(path, label=f"X2 {phase} checkpoint")
    _verify_self_hash(document, "checkpoint_sha256", f"X2 {phase} checkpoint")
    _require(document.get("schema") == CHECKPOINT_SCHEMA, "unsupported X2 checkpoint")
    _require(document.get("phase") == phase, "X2 checkpoint phase mismatch")
    _require(document.get("config_authority_sha256") == config_sha256, "X2 checkpoint config drift")
    _require(
        document.get("implementation_authority_sha256") == implementation_sha256,
        "X2 checkpoint implementation drift",
    )
    _require(document.get("seed_order") == list(seeds), "X2 checkpoint seed order drift")
    rows = document.get("completed_rows")
    if not isinstance(rows, list):
        raise ValueError("X2 checkpoint rows missing")
    _require(len(rows) <= len(seeds), "X2 checkpoint contains excess rows")
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), "X2 checkpoint row must be an object")
        _verify_self_hash(row, "row_sha256", "X2 checkpoint seed row")
        _require(row.get("seed") == seeds[index], "X2 checkpoint seed prefix drift")
        _require(row.get("split") == phase, "X2 checkpoint row phase drift")
    return rows


def _write_checkpoint(
    path: Path,
    *,
    phase: str,
    config_sha256: str,
    implementation_sha256: str,
    seeds: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    document = _sealed(
        _checkpoint_body(
            phase=phase,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            seeds=seeds,
            rows=rows,
            complete=complete,
        ),
        "checkpoint_sha256",
    )
    _atomic_json(path, document)
    return document


def _validate_receipt_bindings(
    receipt: Mapping[str, Any], *, config_sha256: str, implementation_sha256: str
) -> None:
    _verify_self_hash(receipt, "receipt_sha256", "X2 producer receipt")
    _require(receipt.get("schema") == RECEIPT_SCHEMA, "unsupported X2 receipt schema")
    _require(receipt.get("execution_status") == "complete", "X2 receipt is not complete")
    _require(receipt.get("config_authority_sha256") == config_sha256, "X2 receipt config drift")
    _require(
        receipt.get("implementation_authority_sha256") == implementation_sha256,
        "X2 receipt implementation drift",
    )
    _require(receipt.get("activation_enabled") is False, "X2 receipt activation drift")
    _require(receipt.get("scientific_promotion") is False, "X2 receipt promotion drift")


def run_from_config(
    config_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    implementation_authority_path: Path,
    *,
    implementation_authority_sha256: str | None = None,
    max_new_seeds: int | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    if max_new_seeds is not None:
        _require(
            isinstance(max_new_seeds, int) and not isinstance(max_new_seeds, bool) and max_new_seeds >= 0,
            "max_new_seeds must be a nonnegative integer",
        )
    config = load_config(config_path, exploratory=exploratory)
    authority = config_authority(config_path)
    config_sha = str(authority["payload_sha256"])
    manifest = verify_implementation_authority(
        implementation_authority_path,
        expected_config_authority_sha256=config_sha,
        expected_manifest_sha256=implementation_authority_sha256,
        exploratory=exploratory,
    )
    implementation_sha = str(manifest["manifest_sha256"])
    if not exploratory:
        raise ValueError(
            "official X2 activation is disabled; freeze satisfied prerequisites and reseal "
            "an enabled campaign revision"
        )
    prerequisites = prerequisite_receipts(config, exploratory=exploratory)
    if output_path.exists():
        existing = _load_json(output_path, label="X2 producer receipt")
        _validate_receipt_bindings(
            existing, config_sha256=config_sha, implementation_sha256=implementation_sha
        )
        return existing
    seeds = [int(seed) for seed in config["seeds"]]
    rows = _load_checkpoint(
        checkpoint_path,
        phase="producer",
        config_sha256=config_sha,
        implementation_sha256=implementation_sha,
        seeds=seeds,
    )
    remaining = len(seeds) - len(rows)
    budget = remaining if max_new_seeds is None else min(remaining, max_new_seeds)
    for seed in seeds[len(rows) : len(rows) + budget]:
        rows.append(run_seed(config, seed=seed, split="producer"))
        _write_checkpoint(
            checkpoint_path,
            phase="producer",
            config_sha256=config_sha,
            implementation_sha256=implementation_sha,
            seeds=seeds,
            rows=rows,
            complete=len(rows) == len(seeds),
        )
    if len(rows) != len(seeds):
        checkpoint = _write_checkpoint(
            checkpoint_path,
            phase="producer",
            config_sha256=config_sha,
            implementation_sha256=implementation_sha,
            seeds=seeds,
            rows=rows,
            complete=False,
        )
        return {
            "schema": RECEIPT_SCHEMA,
            "execution_status": "partial",
            "resumable": True,
            "completed_seed_count": len(rows),
            "remaining_seed_count": len(seeds) - len(rows),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "activation_enabled": False,
            "scientific_promotion": False,
        }
    aggregate = aggregate_rows(rows, config)
    receipt = _sealed(
        {
            "schema": RECEIPT_SCHEMA,
            "claim_scope": CLAIM_SCOPE,
            "execution_mode": "exploratory" if exploratory else "official",
            "execution_status": "complete",
            "resumable": False,
            "config_path": _path_label(config_path),
            "config_authority_sha256": config_sha,
            "implementation_authority_path": _path_label(implementation_authority_path),
            "implementation_authority_sha256": implementation_sha,
            "prerequisite_receipts": prerequisites,
            "seed_order": seeds,
            "producer_rows": rows,
            "aggregate": aggregate,
            "fresh_verifier_status": "required",
            "activation_enabled": False,
            "prediction_only_claim": False,
            "scientific_promotion": False,
        },
        "receipt_sha256",
    )
    _atomic_json(output_path, receipt)
    _write_checkpoint(
        checkpoint_path,
        phase="producer",
        config_sha256=config_sha,
        implementation_sha256=implementation_sha,
        seeds=seeds,
        rows=rows,
        complete=True,
    )
    return receipt


def verify_receipt(
    receipt_path: Path,
    config_path: Path,
    implementation_authority_path: Path,
    verification_output_path: Path,
    verification_checkpoint_path: Path,
    *,
    implementation_authority_sha256: str | None = None,
    max_new_seeds: int | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    if max_new_seeds is not None:
        _require(
            isinstance(max_new_seeds, int) and not isinstance(max_new_seeds, bool) and max_new_seeds >= 0,
            "max_new_seeds must be a nonnegative integer",
        )
    config = load_config(config_path, exploratory=exploratory)
    authority = config_authority(config_path)
    config_sha = str(authority["payload_sha256"])
    manifest = verify_implementation_authority(
        implementation_authority_path,
        expected_config_authority_sha256=config_sha,
        expected_manifest_sha256=implementation_authority_sha256,
        exploratory=exploratory,
    )
    implementation_sha = str(manifest["manifest_sha256"])
    receipt = _load_json(receipt_path, label="X2 producer receipt")
    _validate_receipt_bindings(receipt, config_sha256=config_sha, implementation_sha256=implementation_sha)
    producer_rows = receipt.get("producer_rows")
    if not isinstance(producer_rows, list):
        raise ValueError("X2 producer rows missing")
    seeds = [int(seed) for seed in config["seeds"]]
    _require(receipt.get("seed_order") == seeds, "X2 producer seed order drift")
    _require(len(producer_rows) == len(seeds), "X2 producer row count drift")
    regenerated_rows = [run_seed(config, seed=seed, split="producer") for seed in seeds]
    producer_regeneration_match = canonical_bytes(regenerated_rows) == canonical_bytes(producer_rows)
    _require(producer_regeneration_match, "X2 producer regeneration mismatch")

    if verification_output_path.exists():
        existing = _load_json(verification_output_path, label="X2 verification receipt")
        _verify_self_hash(existing, "verification_sha256", "X2 verification receipt")
        _require(
            existing.get("producer_receipt_sha256") == receipt["receipt_sha256"],
            "X2 verification is bound to another producer receipt",
        )
        return existing

    fresh_seeds = [int(seed) for seed in config["fresh_verifier_seeds"]]
    fresh_rows = _load_checkpoint(
        verification_checkpoint_path,
        phase="fresh",
        config_sha256=config_sha,
        implementation_sha256=implementation_sha,
        seeds=fresh_seeds,
    )
    remaining = len(fresh_seeds) - len(fresh_rows)
    budget = remaining if max_new_seeds is None else min(remaining, max_new_seeds)
    for seed in fresh_seeds[len(fresh_rows) : len(fresh_rows) + budget]:
        fresh_rows.append(run_seed(config, seed=seed, split="fresh"))
        _write_checkpoint(
            verification_checkpoint_path,
            phase="fresh",
            config_sha256=config_sha,
            implementation_sha256=implementation_sha,
            seeds=fresh_seeds,
            rows=fresh_rows,
            complete=len(fresh_rows) == len(fresh_seeds),
        )
    if len(fresh_rows) != len(fresh_seeds):
        checkpoint = _write_checkpoint(
            verification_checkpoint_path,
            phase="fresh",
            config_sha256=config_sha,
            implementation_sha256=implementation_sha,
            seeds=fresh_seeds,
            rows=fresh_rows,
            complete=False,
        )
        return {
            "schema": VERIFICATION_SCHEMA,
            "verification_status": "partial",
            "resumable": True,
            "completed_fresh_seed_count": len(fresh_rows),
            "remaining_fresh_seed_count": len(fresh_seeds) - len(fresh_rows),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "activation_enabled": False,
            "scientific_promotion": False,
        }

    producer_aggregate = receipt["aggregate"]
    fresh_aggregate = aggregate_rows(fresh_rows, config)
    producer_route = str(producer_aggregate["terminal_route"])
    fresh_route = str(fresh_aggregate["terminal_route"])
    if "failed" in {producer_route, fresh_route}:
        terminal_route = "failed"
    elif "invalid_bed" in {producer_route, fresh_route}:
        terminal_route = "invalid_bed"
    elif producer_route == fresh_route == "positive_candidate":
        terminal_route = "positive_candidate"
    else:
        terminal_route = "controlled_null"
    verification = _sealed(
        {
            "schema": VERIFICATION_SCHEMA,
            "claim_scope": CLAIM_SCOPE,
            "verification_status": "complete",
            "resumable": False,
            "producer_receipt_path": _path_label(receipt_path),
            "producer_receipt_sha256": receipt["receipt_sha256"],
            "config_authority_sha256": config_sha,
            "implementation_authority_sha256": implementation_sha,
            "producer_regeneration_match": producer_regeneration_match,
            "producer_seed_ids": seeds,
            "fresh_seed_ids": fresh_seeds,
            "seed_sets_disjoint": set(seeds).isdisjoint(fresh_seeds),
            "fresh_session_prefix_disjoint": config["splits"]["producer"]["session_prefix"]
            != config["splits"]["fresh"]["session_prefix"],
            "fresh_rows": fresh_rows,
            "producer_aggregate": producer_aggregate,
            "fresh_aggregate": fresh_aggregate,
            "terminal_route": terminal_route,
            "route_instruction": config["routing"][terminal_route],
            "verification_accepts_terminal_result": terminal_route not in {"failed", "invalid_bed"},
            "activation_enabled": False,
            "prediction_only_claim": False,
            "scientific_promotion": False,
        },
        "verification_sha256",
    )
    _atomic_json(verification_output_path, verification)
    _write_checkpoint(
        verification_checkpoint_path,
        phase="fresh",
        config_sha256=config_sha,
        implementation_sha256=implementation_sha,
        seeds=fresh_seeds,
        rows=fresh_rows,
        complete=True,
    )
    return verification


def preflight(config_path: Path, implementation_authority_path: Path, *, exploratory: bool) -> dict[str, Any]:
    config = load_config(config_path, exploratory=exploratory)
    authority = config_authority(config_path)
    manifest = verify_implementation_authority(
        implementation_authority_path,
        expected_config_authority_sha256=str(authority["payload_sha256"]),
        expected_manifest_sha256=None,
        exploratory=exploratory,
    )
    prerequisites = prerequisite_receipts(config, exploratory=True)
    return {
        "schema": "mop-escs-x2-preflight/v1",
        "claim_scope": CLAIM_SCOPE,
        "config_authority_sha256": authority["payload_sha256"],
        "implementation_authority_sha256": manifest["manifest_sha256"],
        "activation_enabled": False,
        "official_execution_blocked": True,
        "missing_prerequisites": [
            row["path"] for row in prerequisites if row["required"] and not row["exists"]
        ],
        "prerequisites": prerequisites,
        "scientific_promotion": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    common.add_argument(
        "--implementation-authority", type=Path, default=DEFAULT_IMPLEMENTATION_AUTHORITY_PATH
    )
    common.add_argument("--implementation-authority-sha256")
    common.add_argument("--exploratory", action="store_true")

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    run.add_argument("--max-new-seeds", type=int)

    verify = subparsers.add_parser("verify", parents=[common])
    verify.add_argument("--receipt", type=Path, default=DEFAULT_OUTPUT_PATH)
    verify.add_argument("--out", type=Path, default=DEFAULT_VERIFICATION_OUTPUT_PATH)
    verify.add_argument("--checkpoint", type=Path, default=DEFAULT_VERIFICATION_CHECKPOINT_PATH)
    verify.add_argument("--max-new-seeds", type=int)

    subparsers.add_parser("preflight", parents=[common])

    authority = subparsers.add_parser("build-authority")
    authority.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    authority.add_argument("--out", type=Path, default=DEFAULT_IMPLEMENTATION_AUTHORITY_PATH)
    authority.add_argument("--mode", choices=("official", "exploratory"), default="official")
    authority.add_argument("--review-status", default=OFFICIAL_IMPLEMENTATION_REVIEW_STATUS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-authority":
        authority = config_authority(args.config.resolve())
        manifest = build_implementation_authority(
            config_authority_sha256=str(authority["payload_sha256"]),
            mode=str(args.mode),
            review_status=str(args.review_status),
        )
        _atomic_json(args.out.resolve(), manifest)
        print(f"wrote {args.out}: {manifest['manifest_sha256']}")
        return 0
    if args.command == "preflight":
        result = preflight(
            args.config.resolve(),
            args.implementation_authority.resolve(),
            exploratory=bool(args.exploratory),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        result = run_from_config(
            args.config.resolve(),
            args.out.resolve(),
            args.checkpoint.resolve(),
            args.implementation_authority.resolve(),
            implementation_authority_sha256=args.implementation_authority_sha256,
            max_new_seeds=args.max_new_seeds,
            exploratory=bool(args.exploratory),
        )
        print(
            f"X2 {result['execution_status']}: "
            f"{result.get('aggregate', {}).get('terminal_route', 'resumable')}"
        )
        return 0
    result = verify_receipt(
        args.receipt.resolve(),
        args.config.resolve(),
        args.implementation_authority.resolve(),
        args.out.resolve(),
        args.checkpoint.resolve(),
        implementation_authority_sha256=args.implementation_authority_sha256,
        max_new_seeds=args.max_new_seeds,
        exploratory=bool(args.exploratory),
    )
    print(f"X2 verifier {result['verification_status']}: {result.get('terminal_route', 'resumable')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
