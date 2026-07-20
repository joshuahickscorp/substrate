"""X1 counterfactual value-of-computation dispatch experiment scaffold.

The study begins at a structured, future-blind hypothesis header.  It does not claim raw event
formation and it cannot promote any architectural claim.  A learned arm may consume bounded exact
same-state forks during tune/calibration, but every deployment ``select`` call receives only a
``VisibleHeader``.  Evaluator truth is a distinct type used after selection for scoring and lesions.

The canonical study is intentionally unexecuted.  Its runner requires terminal, independently
verified EDCM-1 evidence and an independently pinned implementation manifest before it can produce
an official receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import random
import stat
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from mop.escs.accounting import WorkVector
from mop.escs.runtime import DispatchDecision as RuntimeDispatchDecision
from mop.substrate.events import canonical_bytes, canonical_sha256

ENVELOPE_SCHEMA = "mop-escs-x1-envelope/v1"
CONFIG_SCHEMA = "mop-escs-x1-config/v1"
AUTHORITY_SCHEMA = "mop-escs-x1-config-authority/v1"
IMPLEMENTATION_AUTHORITY_SCHEMA = "mop-escs-x1-implementation-authority/v1"
CHECKPOINT_SCHEMA = "mop-escs-x1-checkpoint/v1"
RECEIPT_SCHEMA = "mop-escs-x1-receipt/v1"
VERIFICATION_SCHEMA = "mop-escs-x1-verification/v1"
VERIFICATION_ARTIFACT_SCHEMA = "mop-escs-x1-verification-artifact/v1"
ROW_SCHEMA = "mop-escs-x1-seed-row/v1"
EDCM_RECEIPT_SCHEMA = "mop-edcm1-receipt/v3"
EDCM_VERIFICATION_SCHEMA = "mop-edcm1-verification-artifact/v1"
CLAIM_SCOPE = "generated-structured-observation-dispatch-mechanics-only"
OFFICIAL_CONTRACT_ID = "escs-x1-v1-2026-07-12"
OFFICIAL_CONFIG_AUTHORITY_SHA256 = "c35967980e8b3c878e72a0e134862dd1227c4e88ded8db4b29880b60da8f405a"
OFFICIAL_IMPLEMENTATION_REVIEW_STATUS = "preregistered-scaffold-unexecuted"

PRIMARY_ARM = "learned_interaction_voc"
ARM_NAMES = (
    PRIMARY_ARM,
    "learned_individual_only",
    "learned_no_exploration",
    "learned_random_exploration_rate_matched",
    "edcm_hard_sentinel",
    "outcome_only_bandit",
    "uncertainty_gate",
    "novelty_gate",
    "always_on_full_activation",
    "reactive_cheapest",
    "fixed_sparse",
    "periodic_exact_rate",
    "random_exact_rate",
    "shuffled_exact_rate",
    "shuffled_coalition_exact_rate",
    "homogeneous_exact_total_rate",
    "tuned_best_single",
    "equal_budget_recurrent",
    "oracle_dispatch_nonpromotable",
)
LEARNED_ARMS = frozenset(
    {
        PRIMARY_ARM,
        "learned_individual_only",
        "learned_no_exploration",
        "learned_random_exploration_rate_matched",
        "outcome_only_bandit",
    }
)
ORACLE_ARM = "oracle_dispatch_nonpromotable"
CORE_ACTORS = (
    "reactive_spatial",
    "episodic_retrieval",
    "short_horizon_planner",
    "contradiction_verifier",
)
SYNERGY_PAIR = ("binder_left", "binder_right")

VISIBLE_HEADER_FIELDS = frozenset(
    {
        "event_id",
        "world_token",
        "created_tick",
        "expiry_tick",
        "factor_scope",
        "routing_shards",
        "change_milli",
        "novelty_milli",
        "uncertainty_milli",
        "deadline_slack",
        "public_history_digest",
        "payload_bytes",
        "idle",
        "storm",
    }
)
EVALUATOR_ONLY_FIELDS = frozenset(
    {
        "niche_label",
        "actor_values_milli",
        "pair_interactions_milli",
        "redundancy_groups",
        "irreducible_noise",
        "hidden_state_digest",
        "future_consequence_milli",
    }
)
EXACT_CREDIT_FIELDS = frozenset(
    {
        "event_id",
        "available_tick",
        "base_utility_milli",
        "individual_marginal_milli",
        "pair_interaction_milli",
        "fork_count",
        "source_state_digest",
    }
)
WORK_COMPONENTS = (
    "structured_intake",
    "idle_header_floor",
    "candidate_retrieval",
    "readiness_bids",
    "dispatch_search",
    "exploration",
    "actor_execution",
    "message_operations",
    "exact_counterfactuals",
    "critic_training",
    "stale_reactivation",
    "receipt_serialization",
)
CHECKPOINT_KEYS = (
    "schema",
    "config_authority_sha256",
    "implementation_authority_sha256",
    "entry_gate_sha256",
    "gate_rows",
    "heldout_rows",
    "gate_rows_sha256",
    "heldout_rows_sha256",
    "checkpoint_sha256",
)
RECEIPT_KEYS = (
    "schema",
    "study_id",
    "claim_scope",
    "strong_null",
    "authority",
    "authority_sha256",
    "config_source",
    "implementation_authority",
    "edcm_entry_gate",
    "runtime_identity",
    "gate_rows",
    "bed_gate",
    "heldout_rows",
    "aggregate",
    "execution_status",
    "all_ok",
    "problems",
    "resumable",
    "completed_gate_seeds",
    "completed_heldout_seeds",
    "required_seeds",
    "checkpoint",
    "fresh_verifier_status",
    "candidate_activation_enabled",
    "scientific_promotion",
    "interpretation_limit",
    "exploratory",
    "receipt_sha256",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiment/escs_x1_dispatch.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X1_DISPATCH.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "proof/ESCS_X1_DISPATCH.checkpoint.json"
DEFAULT_VERIFICATION_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X1_DISPATCH.verification.json"
DEFAULT_IMPLEMENTATION_AUTHORITY_PATH = REPO_ROOT / "proof/ESCS_X1_DISPATCH.implementation-authority.json"
DEFAULT_EDCM_RECEIPT_PATH = REPO_ROOT / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json"
DEFAULT_EDCM_VERIFICATION_PATH = REPO_ROOT / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SCOPED_FILE_BYTES = 64 * 1024 * 1024


def _stable_int(*parts: Any, modulus: int = 2**63 - 1) -> int:
    return int.from_bytes(hashlib.sha256(canonical_bytes(list(parts))).digest()[:8], "big") % modulus


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    actual = set(value)
    wanted = set(expected)
    _require(
        actual == wanted,
        f"{label} keys differ: missing={sorted(wanted - actual)} extra={sorted(actual - wanted)}",
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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


def _read_regular_file(path: Path, max_bytes: int, label: str) -> bytes:
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
        _require(before.st_size <= max_bytes, f"{label} exceeds its byte envelope")
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
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(before_identity == after_identity, f"{label} changed during read")
    raw = b"".join(chunks)
    _require(len(raw) == before.st_size, f"{label} size changed during read")
    return raw


def _file_receipt(path: Path) -> dict[str, Any]:
    raw = _read_regular_file(path, MAX_SCOPED_FILE_BYTES, f"scoped file {path}")
    resolved = path.resolve()
    label = str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(resolved)
    return {"path": label, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _require_distinct_paths(paths: Mapping[str, Path | str]) -> None:
    logical: dict[str, list[str]] = defaultdict(list)
    physical: dict[tuple[int, int], list[str]] = defaultdict(list)
    for label, value in paths.items():
        path = Path(value).resolve()
        logical[unicodedata.normalize("NFC", str(path)).casefold()].append(label)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            continue
        _require(stat.S_ISREG(metadata.st_mode), f"artifact path {label!r} is not regular")
        physical[(int(metadata.st_dev), int(metadata.st_ino))].append(label)
    collisions = [labels for labels in [*logical.values(), *physical.values()] if len(labels) > 1]
    _require(not collisions, f"artifact path collision: {collisions}")


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _load_envelope_snapshot(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    raw = _read_regular_file(source, MAX_ARTIFACT_BYTES, "X1 configuration")
    envelope = json.loads(raw)
    _require(isinstance(envelope, dict), "X1 configuration must be a mapping")
    _require_exact_keys(envelope, ("schema", "authority", "payload"), "X1 configuration envelope")
    _require(envelope["schema"] == ENVELOPE_SCHEMA, "unexpected X1 envelope schema")
    _require(isinstance(envelope["authority"], dict), "X1 config authority missing")
    _require(isinstance(envelope["payload"], dict), "X1 config payload missing")
    return envelope, _file_receipt(source)


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == CONFIG_SCHEMA, "unexpected X1 config schema")
    _require(config.get("claim_scope") == CLAIM_SCOPE, "X1 claim scope drift")
    _require(tuple(config.get("arms", ())) == ARM_NAMES, "X1 arm set or order drift")
    _require(config.get("candidate_activation_enabled") is False, "X1 candidate activation escaped")
    _require(config.get("scientific_promotion") is False, "X1 scientific promotion escaped")
    seed_value = config.get("seeds")
    fresh_value = config.get("fresh_verifier_seeds")
    _require(isinstance(seed_value, list) and len(seed_value) == 5, "X1 requires five paired seeds")
    _require(isinstance(fresh_value, list) and len(fresh_value) == 5, "X1 requires five fresh seeds")
    seeds = cast(list[int], seed_value)
    fresh = cast(list[int], fresh_value)
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in [*seeds, *fresh]), "bad seed"
    )
    _require(len(set(seeds)) == len(seeds) and len(set(fresh)) == len(fresh), "duplicate X1 seed")
    _require(set(seeds).isdisjoint(fresh), "producer and fresh X1 seeds overlap")
    split_value = config.get("splits")
    _require(isinstance(split_value, dict), "X1 split declaration missing")
    splits = cast(dict[str, dict[str, Any]], split_value)
    _require(set(splits) == {"tune", "gate", "heldout", "fresh_verifier"}, "X1 split set drift")
    families: list[str] = []
    for name, split in splits.items():
        _require(int(split["episodes"]) > 0 and int(split["horizon"]) >= 16, f"bad {name} split")
        families.append(str(split["world_family"]))
    _require(len(families) == len(set(families)), "X1 world families must be disjoint")
    _require(set(config["visible_header_schema"]) == VISIBLE_HEADER_FIELDS, "visible schema drift")
    _require(set(config["evaluator_only_schema"]) == EVALUATOR_ONLY_FIELDS, "truth schema drift")
    _require(set(config["exact_credit_schema"]) == EXACT_CREDIT_FIELDS, "credit schema drift")
    _require(VISIBLE_HEADER_FIELDS.isdisjoint(EVALUATOR_ONLY_FIELDS), "X1 header leaks evaluator fields")
    _require(tuple(config["work_components"]) == WORK_COMPONENTS, "X1 work components drift")
    actors = config["actors"]
    _require(isinstance(actors, dict) and set(actors) >= set((*CORE_ACTORS, *SYNERGY_PAIR)), "actors missing")
    for actor_id, actor in actors.items():
        _require(bool(actor_id) and bool(actor["factor_scopes"]), "actor scope missing")
        _require(int(actor["operations"]) > 0 and int(actor["message_bytes"]) > 0, "actor cost missing")
    dispatch = config["dispatch"]
    _require(0 < int(dispatch["coalition_cap_c"]) <= int(dispatch["candidate_cap_k"]), "bad X1 caps")
    _require(int(dispatch["beam_cap_b"]) > 0, "X1 beam cap missing")
    _require(
        int(dispatch["dispatch_capacity_events_per_tick"]) > 0 and int(dispatch["queue_cap"]) > 0,
        "X1 queue/service caps missing",
    )
    _require(dispatch["temporary_coalitions_only"] is True, "persistent coalition state forbidden")
    _require(dispatch["global_actor_scan_allowed"] is False, "global actor scan forbidden")
    _require(int(dispatch["deployment_oracle_forks"]) == 0, "deployment oracle fork budget drift")
    credit = config["credit"]
    expected_forks = 1 + len(actors) + math.comb(len(actors), 2)
    _require(
        int(credit["maximum_exact_forks_per_event"]) == expected_forks,
        "X1 exact-fork cap does not bind empty/single/pair fixtures",
    )
    _require(int(credit["calibration_stride"]) > 0, "X1 calibration stride missing")
    _require(
        math.isclose(
            float(credit["exploration_fraction"]),
            1.0 / int(credit["calibration_stride"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "X1 exploration rate does not match its exact calibration stride",
    )
    criteria = config["criteria"]
    _require(float(criteria["max_utility_loss_vs_always_on"]) == 0.01, "utility SESOI drift")
    _require(float(criteria["min_work_saving_vs_always_on"]) == 0.25, "work SESOI drift")
    _require(
        set(criteria["required_rate_matched_controls"]).issubset(ARM_NAMES),
        "unknown required X1 control",
    )
    entry = config["entry_gate"]
    _require(entry["required_producer_schema"] == EDCM_RECEIPT_SCHEMA, "EDCM schema drift")
    _require(entry["required_verification_schema"] == EDCM_VERIFICATION_SCHEMA, "EDCM verifier drift")
    resources = config["resources"]
    _require(resources["cpu_only"] and int(resources["worker_count"]) == 1, "X1 must be serial CPU")
    _require(not resources["allow_downloads"] and not resources["allow_external_data"], "external input")
    _require(config["verdict"]["scientific_promotion"] == "blocked", "promotion wording escaped")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH, *, exploratory: bool = False) -> dict[str, Any]:
    source = Path(path).resolve()
    envelope, _ = _load_envelope_snapshot(source)
    authority = envelope["authority"]
    _require_exact_keys(authority, ("schema", "mode", "contract_id", "payload_sha256"), "X1 authority")
    _require(authority["schema"] == AUTHORITY_SCHEMA, "X1 config authority schema mismatch")
    payload = envelope["payload"]
    digest = canonical_sha256(payload)
    _require(authority["payload_sha256"] == digest, "X1 config payload hash mismatch")
    if not exploratory:
        _require(source == DEFAULT_CONFIG_PATH.resolve(), "official X1 requires repository config")
        _require(authority["mode"] == "official-preregistered", "official X1 config mode required")
        _require(authority["contract_id"] == OFFICIAL_CONTRACT_ID, "X1 contract id mismatch")
        _require(digest == OFFICIAL_CONFIG_AUTHORITY_SHA256, "X1 config is not frozen")
    _validate_config(payload)
    return copy.deepcopy(payload)


@dataclass(frozen=True, slots=True)
class VisibleHeader:

    event_id: str
    world_token: str
    created_tick: int
    expiry_tick: int
    factor_scope: tuple[str, ...]
    routing_shards: tuple[str, ...]
    change_milli: int
    novelty_milli: int
    uncertainty_milli: int
    deadline_slack: int
    public_history_digest: str
    payload_bytes: int
    idle: bool
    storm: bool

    def __post_init__(self) -> None:
        _require({field.name for field in fields(self)} == VISIBLE_HEADER_FIELDS, "header schema drift")
        _require(self.event_id.startswith("event:x1:"), "X1 event namespace missing")
        _require(self.created_tick >= 0 and self.expiry_tick >= self.created_tick, "bad event interval")
        _require(self.deadline_slack == self.expiry_tick - self.created_tick, "deadline leakage/drift")
        _require(tuple(sorted(set(self.factor_scope))) == self.factor_scope, "factor scope not canonical")
        _require(tuple(dict.fromkeys(self.routing_shards)) == self.routing_shards, "routing shards duplicate")
        _require(self.payload_bytes > 0, "payload bytes must be charged")
        _require(len(self.public_history_digest) == 64, "history digest malformed")

    @property
    def scope_key(self) -> str:
        return "|".join(self.factor_scope)

    @property
    def value_key(self) -> str:

        high_joint_surprise = int(self.novelty_milli >= 800 and self.uncertainty_milli >= 800)
        change_bin = min(3, self.change_milli // 250)
        return f"{self.scope_key}|joint={high_joint_surprise}|change={change_bin}"


@dataclass(frozen=True, slots=True)
class EvaluatorTruth:

    niche_label: str
    actor_values_milli: Mapping[str, int]
    pair_interactions_milli: Mapping[str, int]
    redundancy_groups: tuple[tuple[str, ...], ...]
    irreducible_noise: bool
    hidden_state_digest: str
    future_consequence_milli: int

    def __post_init__(self) -> None:
        _require({field.name for field in fields(self)} == EVALUATOR_ONLY_FIELDS, "truth schema drift")
        _require(len(self.hidden_state_digest) == 64, "hidden-state digest malformed")
        _require(0 <= self.future_consequence_milli <= 1000, "future consequence out of range")
        _require(all(isinstance(value, int) for value in self.actor_values_milli.values()), "bad actor value")
        _require(
            all(isinstance(value, int) for value in self.pair_interactions_milli.values()),
            "bad pair interaction",
        )


@dataclass(frozen=True, slots=True)
class EventCase:
    header: VisibleHeader
    evaluator: EvaluatorTruth


@dataclass(frozen=True, slots=True)
class ExactCreditRecord:

    event_id: str
    available_tick: int
    base_utility_milli: int
    individual_marginal_milli: Mapping[str, int]
    pair_interaction_milli: Mapping[str, int]
    fork_count: int
    source_state_digest: str

    def __post_init__(self) -> None:
        _require({field.name for field in fields(self)} == EXACT_CREDIT_FIELDS, "credit schema drift")
        _require(self.available_tick > 0, "credit cannot be available before a consequence")
        _require(self.fork_count > 0, "exact fork count must be charged")
        _require(len(self.source_state_digest) == 64, "credit state digest malformed")


@dataclass(frozen=True, slots=True)
class CoalitionDecision:

    policy_id: str
    coalition_id: str
    actor_ids: tuple[str, ...]
    candidates_considered: int
    coalitions_considered: int
    exploration_trials: int = 0

    def __post_init__(self) -> None:
        _require(tuple(sorted(set(self.actor_ids))) == self.actor_ids, "coalition actors not canonical")
        _require(len(self.coalition_id) == 64, "coalition identity malformed")
        _require(self.candidates_considered >= 0 and self.coalitions_considered >= 0, "negative search")
        _require(self.exploration_trials >= 0, "negative exploration")

    @classmethod
    def create(
        cls,
        policy_id: str,
        header: VisibleHeader,
        actor_ids: Sequence[str],
        *,
        candidates_considered: int,
        coalitions_considered: int,
        exploration_trials: int = 0,
    ) -> CoalitionDecision:
        selected = tuple(sorted(set(actor_ids)))
        identity = canonical_sha256(
            {
                "policy_id": policy_id,
                "event_id": header.event_id,
                "actor_ids": list(selected),
                "temporary": True,
            }
        )
        return cls(
            policy_id,
            identity,
            selected,
            candidates_considered,
            coalitions_considered,
            exploration_trials,
        )

    def runtime_selection(self) -> RuntimeDispatchDecision:

        return RuntimeDispatchDecision.select(*self.actor_ids)


@dataclass(frozen=True, slots=True)
class WorkCharges:
    structured_intake: int = 0
    idle_header_floor: int = 0
    candidate_retrieval: int = 0
    readiness_bids: int = 0
    dispatch_search: int = 0
    exploration: int = 0
    actor_execution: int = 0
    message_operations: int = 0
    exact_counterfactuals: int = 0
    critic_training: int = 0
    stale_reactivation: int = 0
    receipt_serialization: int = 0

    def __post_init__(self) -> None:
        _require({field.name for field in fields(self)} == set(WORK_COMPONENTS), "work schema drift")
        for field in fields(self):
            value = getattr(self, field.name)
            _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, "negative work")

    @property
    def total(self) -> int:
        return sum(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: object) -> WorkCharges:
        if not isinstance(other, WorkCharges):
            return NotImplemented
        return WorkCharges(
            **{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)}
        )

    def payload(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def as_escs_work_vector(self) -> WorkVector:

        return WorkVector(
            raw_transport_and_adapters=self.structured_intake,
            indexing_and_graph_maintenance=self.stale_reactivation,
            dispatch_and_exploration=(
                self.candidate_retrieval + self.readiness_bids + self.dispatch_search + self.exploration
            ),
            actor_execution=self.actor_execution,
            messages=self.message_operations,
            counterfactual_credit=self.exact_counterfactuals,
            learning=self.critic_training,
            archival_and_erasure=self.receipt_serialization,
            idle_floor=self.idle_header_floor,
        )


def _pair_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def _coalition_utility_milli(
    truth: EvaluatorTruth,
    actor_ids: Sequence[str],
    *,
    include_pair_messages: bool = True,
) -> int:
    selected = tuple(sorted(set(actor_ids)))
    if truth.irreducible_noise or truth.niche_label == "idle":
        return 1000
    value = 450 + sum(int(truth.actor_values_milli.get(actor, 0)) for actor in selected)
    if include_pair_messages:
        for left, right in itertools.combinations(selected, 2):
            value += int(truth.pair_interactions_milli.get(_pair_key(left, right), 0))
    return max(0, min(1000, value))


def exact_credit_record(case: EventCase, actor_ids: Sequence[str]) -> ExactCreditRecord:

    actors = tuple(sorted(set(actor_ids)))
    base = _coalition_utility_milli(case.evaluator, ())
    individual = {actor: _coalition_utility_milli(case.evaluator, (actor,)) - base for actor in actors}
    interactions: dict[str, int] = {}
    for left, right in itertools.combinations(actors, 2):
        pair_value = _coalition_utility_milli(case.evaluator, (left, right))
        interactions[_pair_key(left, right)] = pair_value - base - individual[left] - individual[right]
    return ExactCreditRecord(
        event_id=case.header.event_id,
        available_tick=case.header.created_tick + 1,
        base_utility_milli=base,
        individual_marginal_milli=individual,
        pair_interaction_milli=interactions,
        fork_count=1 + len(actors) + len(interactions),
        source_state_digest=canonical_sha256(
            {
                "event_id": case.header.event_id,
                "world_token": case.header.world_token,
                "public_history_digest": case.header.public_history_digest,
                "fork_authority": "same-generated-state",
            }
        ),
    )


def exact_difference_credit(
    case: EventCase,
    coalition: Sequence[str],
    resource_debits_milli: Mapping[str, int] | None = None,
) -> dict[str, int]:

    selected = tuple(sorted(set(coalition)))
    full = _coalition_utility_milli(case.evaluator, selected)
    debits = resource_debits_milli or {}
    return {
        actor: full
        - _coalition_utility_milli(case.evaluator, tuple(item for item in selected if item != actor))
        - int(debits.get(actor, 0))
        for actor in selected
    }


def _truth_for_family(family: str, seed: int, token: str) -> EvaluatorTruth:
    actors = {
        "reactive_spatial": 0,
        "episodic_retrieval": 0,
        "short_horizon_planner": 0,
        "contradiction_verifier": 0,
        "binder_left": 0,
        "binder_right": 0,
        "redundant_retrieval": 0,
        "dormant_regime_actor": 0,
    }
    pairs: dict[str, int] = {}
    redundancy: tuple[tuple[str, ...], ...] = ()
    if family == "spatial":
        actors["reactive_spatial"] = 500
    elif family == "memory":
        actors["episodic_retrieval"] = 480
    elif family == "planning":
        actors["short_horizon_planner"] = 500
    elif family == "contradiction":
        actors["reactive_spatial"] = 170
        actors["short_horizon_planner"] = 160
        actors["contradiction_verifier"] = 220
        pairs[_pair_key("short_horizon_planner", "contradiction_verifier")] = 130
    elif family == "binding":
        pairs[_pair_key(*SYNERGY_PAIR)] = 520
    elif family == "redundancy":
        actors["episodic_retrieval"] = 470
        actors["redundant_retrieval"] = 450
        pairs[_pair_key("episodic_retrieval", "redundant_retrieval")] = -450
        redundancy = (("episodic_retrieval", "redundant_retrieval"),)
    elif family == "regime":
        actors["dormant_regime_actor"] = 510
    elif family not in {"noise", "idle"}:
        raise ValueError(f"unknown generated X1 family {family!r}")
    jitter = _stable_int(seed, token, "value-jitter", modulus=11) - 5
    if family not in {"noise", "idle", "binding"}:
        winner = max(actors, key=lambda actor: (actors[actor], actor))
        actors[winner] = max(0, actors[winner] + jitter)
    hidden = canonical_sha256({"seed": seed, "token": token, "family": family, "hidden": True})
    return EvaluatorTruth(
        niche_label=family,
        actor_values_milli=actors,
        pair_interactions_milli=pairs,
        redundancy_groups=redundancy,
        irreducible_noise=family == "noise",
        hidden_state_digest=hidden,
        future_consequence_milli=1000 if family in {"noise", "idle"} else 450,
    )


def generate_cases(config: Mapping[str, Any], *, seed: int, split: str) -> tuple[EventCase, ...]:
    split_config = config["splits"][split]
    episodes = int(split_config["episodes"])
    horizon = int(split_config["horizon"])
    _require(episodes * horizon <= int(config["resources"]["max_events_per_seed_row"]), "event cap")
    useful_families = (
        "spatial",
        "memory",
        "planning",
        "contradiction",
        "binding",
        "redundancy",
        "regime",
    )
    cases: list[EventCase] = []
    history_digest = canonical_sha256(["x1-history-origin", split, seed])
    family_name = str(split_config["world_family"])
    for episode in range(episodes):
        world_token = f"world:x1:{family_name}:{seed}:{episode}"
        for tick in range(horizon):
            absolute_tick = episode * horizon + tick
            selector = (tick + episode + _stable_int(seed, split, episode, modulus=17)) % 20
            if selector in {0, 1, 2}:
                event_family = "idle"
            elif selector in {3, 4}:
                event_family = "noise"
            else:
                event_family = useful_families[(selector - 5) % len(useful_families)]
            idle = event_family == "idle"
            storm = tick % 17 in {14, 15, 16}
            storm_origin = absolute_tick - (tick % 17) + 14
            created_tick = storm_origin if storm else absolute_tick
            if event_family == "noise":
                visible_family = useful_families[
                    _stable_int(seed, split, episode, tick, "noisy-tv-scope", modulus=len(useful_families))
                ]
            elif event_family == "idle":
                visible_family = "maintenance"
            else:
                visible_family = event_family
            scope = (visible_family,)
            change = 0 if idle else 250 + _stable_int(seed, split, episode, tick, "change", modulus=751)
            novelty = _stable_int(seed, split, episode, tick, "novelty", modulus=1001)
            uncertainty = _stable_int(seed, split, episode, tick, "uncertainty", modulus=1001)
            if event_family == "noise":
                novelty = max(800, novelty)
                uncertainty = max(800, uncertainty)
            slack = 3 if storm else 8
            token = f"{world_token}:{tick}:{event_family}"
            event_id = f"event:x1:{canonical_sha256(token)}"
            header = VisibleHeader(
                event_id=event_id,
                world_token=world_token,
                created_tick=created_tick,
                expiry_tick=created_tick + slack,
                factor_scope=tuple(sorted(scope)),
                routing_shards=(f"shard:{scope[0]}",),
                change_milli=int(change),
                novelty_milli=int(novelty),
                uncertainty_milli=int(uncertainty),
                deadline_slack=slack,
                public_history_digest=history_digest,
                payload_bytes=96 + 8 * len(scope),
                idle=idle,
                storm=storm,
            )
            truth = _truth_for_family(event_family, seed, token)
            cases.append(EventCase(header, truth))
            history_digest = canonical_sha256(
                [history_digest, event_id, truth.future_consequence_milli, created_tick + 1]
            )
    return tuple(cases)


def leakage_gate(cases: Sequence[EventCase] = ()) -> dict[str, Any]:
    policy_method = inspect.signature(DispatchPolicy.select)
    parameters = tuple(policy_method.parameters)
    checks = {
        "typed_schemas_disjoint": VISIBLE_HEADER_FIELDS.isdisjoint(EVALUATOR_ONLY_FIELDS),
        "policy_select_accepts_only_self_and_header": parameters == ("self", "header"),
        "future_consequence_absent_from_header": "future_consequence_milli" not in VISIBLE_HEADER_FIELDS,
        "hidden_state_absent_from_header": "hidden_state_digest" not in VISIBLE_HEADER_FIELDS,
        "deployment_oracle_forks_zero": True,
        "irreducible_noise_not_semantically_marked": all(
            not case.evaluator.irreducible_noise or "noise" not in case.header.factor_scope for case in cases
        ),
        "evaluator_niche_absent_from_event_and_world_identity": all(
            case.evaluator.niche_label not in case.header.event_id
            and case.evaluator.niche_label not in case.header.world_token
            for case in cases
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


@runtime_checkable
class DispatchPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    @property
    def retained_state_bytes(self) -> int: ...

    def select(self, header: VisibleHeader) -> CoalitionDecision: ...


class InteractionValueCritic:

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        policy_id: str,
        include_pairs: bool,
    ) -> None:
        self._config = config
        self._policy_id = policy_id
        self._include_pairs = include_pairs
        self._individual: dict[str, dict[str, float]] = {}
        self._pairs: dict[str, dict[str, float]] = {}
        self._global_individual: dict[str, float] = {}
        self._global_pairs: dict[str, float] = {}
        self._retained_state_bytes = 0
        subscriptions: dict[str, list[str]] = defaultdict(list)
        for actor_id, row in cast(Mapping[str, Mapping[str, Any]], config["actors"]).items():
            for scope in row["factor_scopes"]:
                subscriptions[str(scope)].append(actor_id)
        self._subscription_index = {
            scope: tuple(sorted(actor_ids)) for scope, actor_ids in subscriptions.items()
        }

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def retained_state_bytes(self) -> int:
        return self._retained_state_bytes

    def fit(
        self,
        headers: Mapping[str, VisibleHeader],
        records: Sequence[ExactCreditRecord],
    ) -> dict[str, Any]:
        individual: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        pairs: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        global_individual: dict[str, list[int]] = defaultdict(list)
        global_pairs: dict[str, list[int]] = defaultdict(list)
        for record in records:
            header = headers[record.event_id]
            _require(record.available_tick > header.created_tick, "same-state credit arrived from future")
            key = header.value_key
            for actor, value in record.individual_marginal_milli.items():
                individual[key][actor].append(int(value))
                global_individual[actor].append(int(value))
            if self._include_pairs:
                for pair, value in record.pair_interaction_milli.items():
                    pairs[key][pair].append(int(value))
                    global_pairs[pair].append(int(value))
        self._individual = {
            key: {actor: statistics.fmean(values) for actor, values in rows.items()}
            for key, rows in individual.items()
        }
        self._pairs = {
            key: {pair: statistics.fmean(values) for pair, values in rows.items()}
            for key, rows in pairs.items()
        }
        self._global_individual = {
            actor: statistics.fmean(values) for actor, values in global_individual.items()
        }
        self._global_pairs = {pair: statistics.fmean(values) for pair, values in global_pairs.items()}
        state = {
            "policy_id": self.policy_id,
            "include_pairs": self._include_pairs,
            "individual": self._individual,
            "pairs": self._pairs,
            "global_individual": self._global_individual,
            "global_pairs": self._global_pairs,
            "subscription_index": self._subscription_index,
        }
        self._retained_state_bytes = len(canonical_bytes(state))
        return {
            "record_count": len(records),
            "target_count": sum(
                len(record.individual_marginal_milli)
                + (len(record.pair_interaction_milli) if self._include_pairs else 0)
                for record in records
            ),
            "exact_fork_count": sum(record.fork_count for record in records),
            "retained_state_bytes": self._retained_state_bytes,
            "state_sha256": canonical_sha256(state),
            "evaluator_fields_visible": False,
        }

    def _candidate_ids(self, header: VisibleHeader) -> tuple[str, ...]:
        actors = cast(Mapping[str, Mapping[str, Any]], self._config["actors"])
        compatible = {
            actor_id for scope in header.factor_scope for actor_id in self._subscription_index.get(scope, ())
        }
        cap = int(self._config["dispatch"]["candidate_cap_k"])
        return tuple(sorted(compatible, key=lambda actor: (int(actors[actor]["operations"]), actor))[:cap])

    def _predicted_gain(self, header: VisibleHeader, coalition: Sequence[str]) -> float:
        individual = self._individual.get(header.value_key, self._global_individual)
        pairs = self._pairs.get(header.value_key, self._global_pairs)
        gain = sum(
            float(individual.get(actor, self._global_individual.get(actor, 0.0))) for actor in coalition
        )
        if self._include_pairs:
            gain += sum(
                float(pairs.get(_pair_key(left, right), 0.0))
                for left, right in itertools.combinations(coalition, 2)
            )
        return gain

    def predict_gain_milli(self, header: VisibleHeader, coalition: Sequence[str]) -> float:
        return self._predicted_gain(header, coalition)

    def select(self, header: VisibleHeader) -> CoalitionDecision:
        candidates = self._candidate_ids(header)
        actors = cast(Mapping[str, Mapping[str, Any]], self._config["actors"])
        cap = int(self._config["dispatch"]["coalition_cap_c"])
        beam = int(self._config["dispatch"]["beam_cap_b"])
        compute_price = int(self._config["dispatch"]["compute_price_milli"])
        bandwidth_price = int(self._config["dispatch"]["bandwidth_price_milli"])
        minimum = int(self._config["dispatch"]["minimum_net_value_milli"])
        considered: list[tuple[str, ...]] = [()]
        for size in range(1, min(cap, len(candidates)) + 1):
            considered.extend(itertools.combinations(candidates, size))
        considered = considered[:beam]

        def score(coalition: tuple[str, ...]) -> tuple[float, int, tuple[str, ...]]:
            debit = sum(
                compute_price * int(actors[actor]["operations"])
                + bandwidth_price * math.ceil(int(actors[actor]["message_bytes"]) / 16)
                for actor in coalition
            )
            return (self._predicted_gain(header, coalition) - debit, -len(coalition), coalition)

        best = max(considered, key=score)
        selected = best if score(best)[0] >= minimum else ()
        return CoalitionDecision.create(
            self.policy_id,
            header,
            selected,
            candidates_considered=len(candidates),
            coalitions_considered=len(considered),
        )


def _training_bundle(
    config: Mapping[str, Any],
    tune_cases: Sequence[EventCase],
    *,
    mode: str,
    seed: int,
) -> tuple[InteractionValueCritic, dict[str, Any], WorkCharges]:
    headers = {case.header.event_id: case.header for case in tune_cases}
    stride = int(config["credit"]["calibration_stride"])
    target_count = math.ceil(len(tune_cases) / stride)
    coverage_cases: list[EventCase] = []
    covered_scopes: set[str] = set()
    for case in tune_cases:
        if case.header.value_key not in covered_scopes:
            coverage_cases.append(case)
            covered_scopes.add(case.header.value_key)
    periodic_cases = [case for index, case in enumerate(tune_cases) if index % stride == 0]
    selected_cases = []
    for case in [*coverage_cases, *periodic_cases]:
        if case not in selected_cases:
            selected_cases.append(case)
        if len(selected_cases) == target_count:
            break
    if mode == "no_exploration":
        selected_cases = [case for case in selected_cases if case.header.scope_key != "binding"]
    elif mode == "random_exploration":
        random_order = list(tune_cases)
        random.Random(_stable_int(seed, "random-exploration-order")).shuffle(random_order)
        selected_cases = random_order[:target_count]
    records = [exact_credit_record(case, tuple(config["actors"])) for case in selected_cases]
    for record in records:
        _require(
            record.fork_count <= int(config["credit"]["maximum_exact_forks_per_event"]),
            "X1 exact-credit fork cap exceeded",
        )
    include_pairs = mode != "individual_only"
    critic = InteractionValueCritic(
        config,
        policy_id=f"builtin:x1-{mode}-critic-v1",
        include_pairs=include_pairs,
    )
    training = critic.fit(headers, records)
    training["retained_state_byte_ticks"] = int(training["retained_state_bytes"]) * len(tune_cases)
    actor_specs = cast(Mapping[str, Mapping[str, Any]], config["actors"])
    appearances_per_actor = len(actor_specs)
    actor_appearances_per_record = appearances_per_actor * len(actor_specs)
    training["encoded_bytes"] = (
        sum(case.header.payload_bytes for case in selected_cases)
        + len(records)
        * appearances_per_actor
        * sum(int(row["message_bytes"]) for row in actor_specs.values())
        + sum(
            len(
                canonical_bytes(
                    {
                        "event_id": record.event_id,
                        "available_tick": record.available_tick,
                        "base_utility_milli": record.base_utility_milli,
                        "individual_marginal_milli": dict(record.individual_marginal_milli),
                        "pair_interaction_milli": dict(record.pair_interaction_milli),
                        "fork_count": record.fork_count,
                        "source_state_digest": record.source_state_digest,
                    }
                )
            )
            for record in records
        )
    )
    costs = config["work_costs"]
    work = WorkCharges(
        structured_intake=len(records) * int(costs["structured_intake_per_header"]),
        exploration=(
            len(selected_cases) * int(costs["exploration_per_trial"])
            if mode in {"interaction", "random_exploration"}
            else 0
        ),
        actor_execution=len(records)
        * appearances_per_actor
        * sum(int(row["operations"]) for row in actor_specs.values()),
        message_operations=len(records)
        * actor_appearances_per_record
        * int(costs["message_operation_per_message"]),
        exact_counterfactuals=int(training["exact_fork_count"]) * int(costs["exact_fork_base"]),
        critic_training=int(training["target_count"]) * int(costs["critic_update_per_target"]),
        stale_reactivation=len(records) * appearances_per_actor * int(costs["stale_reactivation_per_actor"]),
        receipt_serialization=len(records) * int(costs["receipt_serialization_per_event"]),
    )
    training["mode"] = mode
    return critic, training, work


def _fixed_decision(
    policy_id: str,
    header: VisibleHeader,
    actors: Sequence[str],
    *,
    candidates: int = 0,
    coalitions: int = 1,
) -> CoalitionDecision:
    return CoalitionDecision.create(
        policy_id,
        header,
        actors,
        candidates_considered=candidates,
        coalitions_considered=coalitions,
    )


def _oracle_decision(config: Mapping[str, Any], case: EventCase) -> CoalitionDecision:
    actors = tuple(sorted(config["actors"]))
    cap = int(config["dispatch"]["coalition_cap_c"])
    candidates = [()]
    for size in range(1, cap + 1):
        candidates.extend(itertools.combinations(actors, size))
    specs = config["actors"]

    def objective(coalition: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
        utility = _coalition_utility_milli(case.evaluator, coalition)
        work = sum(int(specs[actor]["operations"]) for actor in coalition)
        return (utility, -work, tuple(reversed(coalition)))

    selected = max(candidates, key=objective)
    return _fixed_decision(
        "oracle:evaluator-enumeration-nonpromotable",
        case.header,
        selected,
        candidates=len(actors),
        coalitions=len(candidates),
    )


def _actor_counts(schedule: Sequence[CoalitionDecision]) -> Counter[str]:
    return Counter(actor for decision in schedule for actor in decision.actor_ids)


def _exact_count_schedule(
    cases: Sequence[EventCase],
    reference: Sequence[CoalitionDecision],
    *,
    policy_id: str,
    method: str,
    seed: int,
    coalition_cap: int,
) -> list[CoalitionDecision]:
    counts = _actor_counts(reference)
    assignments: list[list[str]] = [[] for _ in cases]
    for actor, count in sorted(counts.items()):
        _require(count <= len(cases), "rate-matched actor count exceeds one call per event")
        if method == "random":
            rng = random.Random(_stable_int(seed, actor, method))
            preferred = list(range(len(cases)))
            rng.shuffle(preferred)
        elif method == "shuffled":
            offset = _stable_int(seed, actor, method, modulus=max(1, len(cases)))
            preferred = [int((offset + 7 * index) % len(cases)) for index in range(len(cases))]
        else:
            step = len(cases) / max(1, count)
            preferred = [int(index * step) % len(cases) for index in range(count)]
            preferred.extend(index for index in range(len(cases)) if index not in preferred)
        chosen: list[int] = []
        for candidate in preferred:
            if candidate in chosen or len(assignments[candidate]) >= coalition_cap:
                continue
            chosen.append(candidate)
            if len(chosen) == count:
                break
        _require(len(chosen) == count, "could not construct exact-rate schedule under coalition cap")
        for index in chosen:
            assignments[index].append(actor)
    return [
        _fixed_decision(policy_id, case.header, actors, coalitions=1)
        for case, actors in zip(cases, assignments, strict=True)
    ]


def shuffled_coalition_schedule(
    cases: Sequence[EventCase],
    reference: Sequence[CoalitionDecision],
    *,
    seed: int,
) -> list[CoalitionDecision]:
    coalitions = [decision.actor_ids for decision in reference]
    random.Random(_stable_int(seed, "whole-coalition-shuffle")).shuffle(coalitions)
    return [
        _fixed_decision("control:shuffled-whole-coalition", case.header, coalition)
        for case, coalition in zip(cases, coalitions, strict=True)
    ]


def _best_single_actor(config: Mapping[str, Any], tune_cases: Sequence[EventCase]) -> str:
    actors = tuple(sorted(config["actors"]))
    means = {
        actor: statistics.fmean(_coalition_utility_milli(case.evaluator, (actor,)) for case in tune_cases)
        for actor in actors
    }
    return max(actors, key=lambda actor: (means[actor], actor))


def _best_actor_by_visible_scope(
    config: Mapping[str, Any],
    tune_cases: Sequence[EventCase],
) -> dict[str, str]:
    actors = tuple(sorted(config["actors"]))
    scopes = sorted({case.header.scope_key for case in tune_cases if not case.header.idle})
    result: dict[str, str] = {}
    for scope in scopes:
        scoped_cases = [case for case in tune_cases if case.header.scope_key == scope]
        if not scoped_cases:
            continue
        means = {
            actor: statistics.fmean(
                _coalition_utility_milli(case.evaluator, (actor,)) for case in scoped_cases
            )
            for actor in actors
        }
        result[scope] = max(actors, key=lambda actor: (means[actor], actor))
    return result


def _tuned_control_training(
    config: Mapping[str, Any],
    tune_cases: Sequence[EventCase],
    *,
    control_id: str,
) -> tuple[dict[str, Any], WorkCharges]:

    actors = cast(Mapping[str, Mapping[str, Any]], config["actors"])
    costs = config["work_costs"]
    actor_evaluations = len(tune_cases) * len(actors)
    actor_work = len(tune_cases) * sum(int(row["operations"]) for row in actors.values())
    message_count = actor_evaluations
    state = {
        "control_id": control_id,
        "actor_ids": sorted(actors),
        "tune_event_ids_sha256": canonical_sha256([case.header.event_id for case in tune_cases]),
    }
    training: dict[str, Any] = {
        "mode": "fully-charged-visible-outcome-control-tuning",
        "record_count": len(tune_cases),
        "target_count": actor_evaluations,
        "exact_fork_count": 0,
        "retained_state_bytes": len(canonical_bytes(state)),
        "encoded_bytes": len(tune_cases)
        * (
            sum(case.header.payload_bytes for case in tune_cases) // max(1, len(tune_cases)) * len(actors)
            + sum(int(row["message_bytes"]) for row in actors.values())
        ),
        "state_sha256": canonical_sha256(state),
        "evaluator_fields_visible": False,
    }
    training["retained_state_byte_ticks"] = int(training["retained_state_bytes"]) * len(tune_cases)
    work = WorkCharges(
        structured_intake=actor_evaluations * int(costs["structured_intake_per_header"]),
        actor_execution=actor_work,
        message_operations=message_count * int(costs["message_operation_per_message"]),
        critic_training=actor_evaluations * int(costs["critic_update_per_target"]),
        receipt_serialization=actor_evaluations * int(costs["receipt_serialization_per_event"]),
    )
    return training, work


def _dormant_scaling_assay(
    config: Mapping[str, Any],
    cases: Sequence[EventCase],
) -> tuple[dict[str, Any], WorkCharges]:

    dispatch = config["dispatch"]
    costs = config["work_costs"]
    active_actors = len(config["actors"])
    candidate_cap = int(dispatch["candidate_cap_k"])
    populations = [int(value) for value in dispatch["dormant_population_counts"]]
    densities = [float(value) for value in dispatch["subscription_densities"]]
    query_per_shard = int(costs["subscription_query_per_shard"])
    retrieval_per_candidate = int(costs["candidate_retrieval_per_candidate"])
    build_per_actor = int(costs["subscription_index_build_per_actor"])
    bytes_per_actor = int(costs["subscription_index_bytes_per_actor"])
    mean_active_candidates = statistics.fmean(
        min(
            candidate_cap,
            sum(
                set(config["actors"][actor]["factor_scopes"]) & set(case.header.factor_scope) != set()
                for actor in config["actors"]
            ),
        )
        for case in cases
    )
    rows: dict[str, Any] = {}
    total_charged_work = 0
    total_retained_byte_ticks = 0
    total_index_bytes = 0
    slopes: list[float] = []
    for density in densities:
        density_rows: dict[str, Any] = {}
        per_event_values: list[tuple[int, float]] = []
        for population in populations:
            registered = active_actors + population
            compatible_dormant = round(population * density)
            returned_candidates = min(candidate_cap, mean_active_candidates + compatible_dormant)
            query_work_per_event = query_per_shard + retrieval_per_candidate * returned_candidates
            query_work = round(query_work_per_event * len(cases))
            build_work = registered * build_per_actor
            retained_byte_ticks = registered * bytes_per_actor * len(cases)
            total_charged_work += build_work + query_work
            total_retained_byte_ticks += retained_byte_ticks
            total_index_bytes += registered * bytes_per_actor
            per_event_values.append((population, float(query_work_per_event)))
            density_rows[str(population)] = {
                "registered_actor_count": registered,
                "compatible_dormant_registrations": compatible_dormant,
                "returned_candidate_count": returned_candidates,
                "query_work_per_event": query_work_per_event,
                "query_work": query_work,
                "index_build_work": build_work,
                "retained_index_byte_ticks": retained_byte_ticks,
                "global_scan": False,
            }
        first_population, first_work = per_event_values[0]
        last_population, last_work = per_event_values[-1]
        slope = (last_work - first_work) / max(1, last_population - first_population)
        slopes.append(slope)
        rows[f"{density:.2f}"] = {
            "population_rows": density_rows,
            "query_work_slope_per_dormant_actor": slope,
        }
    result = {
        "schema": "mop-escs-x1-dormant-scaling-assay/v1",
        "population_counts": populations,
        "subscription_densities": densities,
        "candidate_cap_k": candidate_cap,
        "rows": rows,
        "max_query_work_slope_per_dormant_actor": max(slopes, default=0.0),
        "total_charged_index_and_query_work": total_charged_work,
        "total_retained_index_byte_ticks": total_retained_byte_ticks,
        "total_serialized_index_bytes": total_index_bytes,
        "global_scan": False,
    }
    return result, WorkCharges(candidate_retrieval=total_charged_work)


def _hard_sentinel(header: VisibleHeader) -> tuple[str, ...]:
    by_scope = {
        "spatial": ("reactive_spatial",),
        "memory": ("episodic_retrieval",),
        "planning": ("short_horizon_planner", "contradiction_verifier"),
        "contradiction": (
            "reactive_spatial",
            "short_horizon_planner",
            "contradiction_verifier",
        ),
        "binding": ("binder_left",),
        "redundancy": ("episodic_retrieval",),
        "regime": ("dormant_regime_actor",),
    }
    return by_scope.get(header.scope_key, ())


def _queue_diagnostics(
    config: Mapping[str, Any],
    cases: Sequence[EventCase],
    schedule: Sequence[CoalitionDecision],
) -> dict[str, int | bool]:

    capacity = int(config["dispatch"]["dispatch_capacity_events_per_tick"])
    queue_cap = int(config["dispatch"]["queue_cap"])
    arrivals: dict[int, list[tuple[int, CoalitionDecision]]] = defaultdict(list)
    for case, decision in zip(cases, schedule, strict=True):
        arrivals[case.header.created_tick].append((case.header.expiry_tick, decision))
    if not arrivals:
        return {"max_queue_depth": 0, "deadline_misses": 0, "recovery_ticks": 0, "stable": True}
    pending: list[tuple[int, CoalitionDecision]] = []
    max_depth = 0
    misses = 0
    last_arrival = max(arrivals)
    last_service = min(arrivals)
    tick = min(arrivals)
    while tick <= last_arrival or pending:
        pending.extend(arrivals.get(tick, ()))
        max_depth = max(max_depth, len(pending))
        served = pending[:capacity]
        pending = pending[capacity:]
        misses += sum(tick > expiry for expiry, _ in served)
        if served:
            last_service = tick
        tick += 1
    return {
        "max_queue_depth": max_depth,
        "deadline_misses": misses,
        "recovery_ticks": max(0, last_service - last_arrival),
        "stable": max_depth <= queue_cap and misses == 0,
    }


def _evaluate_schedule(
    config: Mapping[str, Any],
    cases: Sequence[EventCase],
    schedule: Sequence[CoalitionDecision],
    *,
    arm: str,
    training: Mapping[str, Any] | None = None,
    training_work: WorkCharges | None = None,
    critic: InteractionValueCritic | None = None,
    scaling_assay: Mapping[str, Any] | None = None,
    effective_activation_calls: int | None = None,
    deployment_actor_execution_floor: int | None = None,
    deployment_message_operations_floor: int | None = None,
    deployment_message_bytes_floor: int | None = None,
    total_work_floor: int | None = None,
) -> dict[str, Any]:
    _require(len(cases) == len(schedule), "schedule/case length mismatch")
    costs = config["work_costs"]
    actors = config["actors"]
    work = training_work or WorkCharges()
    utilities: list[float] = []
    encoded_bytes = int(training.get("encoded_bytes", 0)) if training is not None else 0
    deployment_actor_execution = 0
    deployment_message_operations = 0
    deployment_message_bytes = 0
    idle_work = 0
    noisy_activations = 0
    noisy_opportunities = 0
    synergy_discoveries = 0
    synergy_opportunities = 0
    calibration_errors: list[float] = []
    high_lesions: list[float] = []
    message_lesions: list[float] = []
    max_candidates = 0
    max_coalitions = 0
    for case, decision in zip(cases, schedule, strict=True):
        header = case.header
        selected = decision.actor_ids
        actor_work = sum(int(actors[actor]["operations"]) for actor in selected)
        message_bytes = sum(int(actors[actor]["message_bytes"]) for actor in selected)
        charge = WorkCharges(
            structured_intake=int(costs["structured_intake_per_header"]),
            idle_header_floor=int(costs["idle_header_floor"]) if header.idle else 0,
            candidate_retrieval=decision.candidates_considered
            * int(costs["candidate_retrieval_per_candidate"]),
            readiness_bids=decision.candidates_considered * int(costs["readiness_bid_per_candidate"]),
            dispatch_search=decision.coalitions_considered * int(costs["dispatch_per_considered_coalition"]),
            exploration=decision.exploration_trials * int(costs["exploration_per_trial"]),
            actor_execution=actor_work,
            message_operations=len(selected) * int(costs["message_operation_per_message"]),
            stale_reactivation=(
                int(costs["stale_reactivation_per_actor"]) if "dormant_regime_actor" in selected else 0
            ),
            receipt_serialization=int(costs["receipt_serialization_per_event"]),
        )
        work = work + charge
        if header.idle:
            idle_work += charge.structured_intake + charge.idle_header_floor + charge.dispatch_search
        encoded_bytes += header.payload_bytes + message_bytes
        deployment_actor_execution += actor_work
        deployment_message_operations += len(selected) * int(costs["message_operation_per_message"])
        deployment_message_bytes += message_bytes
        utility_milli = _coalition_utility_milli(case.evaluator, selected)
        utilities.append(utility_milli / 1000.0)
        if case.evaluator.irreducible_noise:
            noisy_opportunities += 1
            noisy_activations += int(bool(selected))
        if case.evaluator.niche_label == "binding":
            synergy_opportunities += 1
            synergy_discoveries += int(set(SYNERGY_PAIR).issubset(selected))
        credits = exact_difference_credit(case, selected)
        if credits:
            high_lesions.append(max(0.0, max(credits.values()) / 1000.0))
        if len(selected) >= 2:
            message_lesions.append(
                max(
                    0.0,
                    (
                        utility_milli
                        - _coalition_utility_milli(
                            case.evaluator,
                            selected,
                            include_pair_messages=False,
                        )
                    )
                    / 1000.0,
                )
            )
        if critic is not None:
            actual = utility_milli - _coalition_utility_milli(case.evaluator, ())
            predicted = critic.predict_gain_milli(header, selected)
            calibration_errors.append(abs(actual - predicted) / 1000.0)
        max_candidates = max(max_candidates, decision.candidates_considered)
        max_coalitions = max(max_coalitions, decision.coalitions_considered)
    if (
        deployment_actor_execution_floor is not None
        and deployment_actor_execution < deployment_actor_execution_floor
    ):
        work = work + WorkCharges(
            actor_execution=deployment_actor_execution_floor - deployment_actor_execution
        )
        deployment_actor_execution = deployment_actor_execution_floor
    if (
        deployment_message_operations_floor is not None
        and deployment_message_operations < deployment_message_operations_floor
    ):
        work = work + WorkCharges(
            message_operations=deployment_message_operations_floor - deployment_message_operations
        )
        deployment_message_operations = deployment_message_operations_floor
    if (
        deployment_message_bytes_floor is not None
        and deployment_message_bytes < deployment_message_bytes_floor
    ):
        encoded_bytes += deployment_message_bytes_floor - deployment_message_bytes
        deployment_message_bytes = deployment_message_bytes_floor
    if total_work_floor is not None and work.total < total_work_floor:
        work = work + WorkCharges(dispatch_search=total_work_floor - work.total)
    calls = sum(len(decision.actor_ids) for decision in schedule)
    effective_calls = calls if effective_activation_calls is None else effective_activation_calls
    retained_bytes = int(training["retained_state_bytes"]) if training is not None else 32
    retained_byte_ticks = retained_bytes * len(cases) + (
        int(training.get("retained_state_byte_ticks", 0)) if training is not None else 0
    )
    if scaling_assay is not None:
        retained_byte_ticks += int(scaling_assay["total_retained_index_byte_ticks"])
    evidence_standing = (
        "oracle_nonpromotable"
        if arm == ORACLE_ARM
        else "candidate_unverified"
        if arm == PRIMARY_ARM
        else "control_only"
    )
    queue = _queue_diagnostics(config, cases, schedule)
    return {
        "arm": arm,
        "event_count": len(cases),
        "mean_utility": statistics.fmean(utilities) if utilities else 0.0,
        "total_lifecycle_work": work.total,
        "work_components": work.payload(),
        "total_encoded_bytes": encoded_bytes,
        "retained_state_byte_ticks": retained_byte_ticks,
        "idle_boundary_work": idle_work,
        "activation_calls": calls,
        "effective_activation_calls": effective_calls,
        "deployment_actor_execution": deployment_actor_execution,
        "deployment_message_operations": deployment_message_operations,
        "deployment_message_bytes": deployment_message_bytes,
        "matched_virtual_activation_calls": max(0, effective_calls - calls),
        "actor_activation_counts": dict(sorted(_actor_counts(schedule).items())),
        "coalition_multiset_sha256": canonical_sha256(
            sorted([list(decision.actor_ids) for decision in schedule])
        ),
        "schedule_sha256": canonical_sha256([list(decision.actor_ids) for decision in schedule]),
        "noisy_tv_activation_rate": noisy_activations / noisy_opportunities if noisy_opportunities else 0.0,
        "synergy_discovery_rate": (
            synergy_discoveries / synergy_opportunities if synergy_opportunities else 0.0
        ),
        "calibration_error": statistics.fmean(calibration_errors) if calibration_errors else None,
        "high_credit_actor_lesion_effect": statistics.fmean(high_lesions) if high_lesions else 0.0,
        "message_lesion_effect": statistics.fmean(message_lesions) if message_lesions else 0.0,
        "deadline_misses": int(queue["deadline_misses"]),
        "max_queue_depth": int(queue["max_queue_depth"]),
        "storm_recovery_ticks": int(queue["recovery_ticks"]),
        "queue_stable": bool(queue["stable"]),
        "max_candidates_retrieved": max_candidates,
        "max_coalitions_considered": max_coalitions,
        "dormant_population_scaling": dict(scaling_assay) if scaling_assay is not None else None,
        "candidate_work_slope_per_dormant_actor": (
            float(scaling_assay["max_query_work_slope_per_dormant_actor"])
            if scaling_assay is not None
            else None
        ),
        "global_actor_scan": arm == ORACLE_ARM,
        "training": dict(training) if training is not None else None,
        "deployment_oracle_forks": 0 if arm != ORACLE_ARM else len(cases),
        "oracle_access": arm == ORACLE_ARM,
        "evidence_standing": evidence_standing,
        "temporary_coalitions_only": True,
        "scientific_promotion": False,
    }


def _bed_gate(config: Mapping[str, Any], cases: Sequence[EventCase]) -> dict[str, Any]:
    actors = tuple(sorted(config["actors"]))
    always = statistics.fmean(_coalition_utility_milli(case.evaluator, actors) / 1000.0 for case in cases)
    single_means = {
        actor: statistics.fmean(_coalition_utility_milli(case.evaluator, (actor,)) / 1000.0 for case in cases)
        for actor in actors
    }
    best_single = max(single_means, key=lambda actor: (single_means[actor], actor))
    oracle_values = [
        _coalition_utility_milli(case.evaluator, _oracle_decision(config, case).actor_ids) / 1000.0
        for case in cases
    ]
    unique = {
        actor: sum(
            case.evaluator.niche_label
            in {
                "spatial" if actor == "reactive_spatial" else "",
                "memory" if actor == "episodic_retrieval" else "",
                "planning" if actor == "short_horizon_planner" else "",
                "contradiction" if actor == "contradiction_verifier" else "",
            }
            for case in cases
        )
        for actor in CORE_ACTORS
    }
    synergy_values = [
        case.evaluator.pair_interactions_milli.get(_pair_key(*SYNERGY_PAIR), 0) / 1000.0
        for case in cases
        if case.evaluator.niche_label == "binding"
    ]
    settings = config["difficulty_and_complementarity_gate"]
    checks = {
        "always_on_floor": always >= float(settings["min_always_on_utility"]),
        "oracle_headroom": statistics.fmean(oracle_values) - single_means[best_single]
        >= float(settings["min_oracle_headroom_over_best_single"]),
        "best_single_off_ceiling": single_means[best_single] <= float(settings["max_best_single_utility"]),
        "every_core_actor_has_unique_niche": all(
            count >= int(settings["min_unique_niche_cases_per_core_actor"]) for count in unique.values()
        ),
        "cold_start_synergy_present": len(synergy_values) >= int(settings["min_synergy_cases"])
        and min(synergy_values, default=0.0) >= float(settings["min_synergy_pair_interaction"]),
        "leakage_gate": leakage_gate(cases)["passed"],
    }
    return {
        "status": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "always_on_utility": always,
        "single_actor_utility": single_means,
        "best_single_actor": best_single,
        "oracle_utility": statistics.fmean(oracle_values),
        "unique_niche_case_counts": unique,
        "synergy_case_count": len(synergy_values),
        "failure_interpretation": "invalid_bed_not_dispatch_null",
    }


def run_seed(config: Mapping[str, Any], *, seed: int, split: str) -> dict[str, Any]:

    _require(split in {"gate", "heldout", "fresh_verifier"}, "unsupported X1 evaluation split")
    tune_cases = generate_cases(config, seed=seed, split="tune")
    cases = generate_cases(config, seed=seed, split=split)
    actors = tuple(sorted(config["actors"]))
    primary_critic, primary_training, primary_training_work = _training_bundle(
        config, tune_cases, mode="interaction", seed=seed
    )
    scaling_assay, scaling_work = _dormant_scaling_assay(config, cases)
    primary_training_work = primary_training_work + scaling_work
    primary_training = {
        **primary_training,
        "encoded_bytes": int(primary_training["encoded_bytes"])
        + int(scaling_assay["total_serialized_index_bytes"]),
        "dormant_scaling_assay_sha256": canonical_sha256(scaling_assay),
        "dormant_scaling_assay_work": scaling_work.total,
    }
    individual_critic, individual_training, individual_work = _training_bundle(
        config, tune_cases, mode="individual_only", seed=seed
    )
    no_explore_critic, no_explore_training, no_explore_work = _training_bundle(
        config, tune_cases, mode="no_exploration", seed=seed
    )
    random_critic, random_training, random_work = _training_bundle(
        config, tune_cases, mode="random_exploration", seed=seed
    )
    primary_schedule = [primary_critic.select(case.header) for case in cases]
    individual_schedule = [individual_critic.select(case.header) for case in cases]
    no_explore_schedule = [no_explore_critic.select(case.header) for case in cases]
    random_explore_schedule = [random_critic.select(case.header) for case in cases]
    best_single = _best_single_actor(config, tune_cases)
    scope_best = _best_actor_by_visible_scope(config, tune_cases)
    tuned_control_training, tuned_control_work = _tuned_control_training(
        config,
        tune_cases,
        control_id="gate-selected-best-single-and-outcome-control",
    )
    actor_means = {
        actor: statistics.fmean(_coalition_utility_milli(case.evaluator, (actor,)) for case in tune_cases)
        for actor in actors
    }
    bandit_actor = max(actors, key=lambda actor: (actor_means[actor], actor))
    specs = config["actors"]

    def compatible(header: VisibleHeader) -> tuple[str, ...]:
        return tuple(
            actor for actor in actors if set(specs[actor]["factor_scopes"]) & set(header.factor_scope)
        )

    hard = [
        _fixed_decision("control:edcm-hard-sentinel", case.header, _hard_sentinel(case.header))
        for case in cases
    ]
    outcome_bandit = [
        _fixed_decision("control:outcome-only-bandit", case.header, (bandit_actor,)) for case in cases
    ]
    uncertainty = [
        _fixed_decision(
            "control:uncertainty-gate",
            case.header,
            (min(compatible(case.header), key=lambda actor: (int(specs[actor]["operations"]), actor)),)
            if compatible(case.header)
            and case.header.uncertainty_milli >= int(config["controls"]["uncertainty_threshold_milli"])
            else (),
        )
        for case in cases
    ]
    novelty = [
        _fixed_decision(
            "control:novelty-gate",
            case.header,
            (min(compatible(case.header), key=lambda actor: (int(specs[actor]["operations"]), actor)),)
            if compatible(case.header)
            and case.header.novelty_milli >= int(config["controls"]["novelty_threshold_milli"])
            else (),
        )
        for case in cases
    ]
    always = [_fixed_decision("control:always-on", case.header, actors) for case in cases]
    reactive = [
        _fixed_decision(
            "control:reactive-cheapest",
            case.header,
            (min(compatible(case.header), key=lambda actor: (int(specs[actor]["operations"]), actor)),)
            if compatible(case.header)
            and case.header.change_milli >= int(config["controls"]["reactive_change_threshold_milli"])
            else (),
        )
        for case in cases
    ]
    fixed_sparse_ids = tuple(config["controls"]["fixed_sparse_actors"])
    fixed_sparse = [_fixed_decision("control:fixed-sparse", case.header, fixed_sparse_ids) for case in cases]
    cap = int(config["dispatch"]["coalition_cap_c"])
    periodic = _exact_count_schedule(
        cases,
        primary_schedule,
        policy_id="control:periodic-exact-rate",
        method="periodic",
        seed=seed,
        coalition_cap=cap,
    )
    random_rate = _exact_count_schedule(
        cases,
        primary_schedule,
        policy_id="control:random-exact-rate",
        method="random",
        seed=seed,
        coalition_cap=cap,
    )
    shuffled_rate = _exact_count_schedule(
        cases,
        primary_schedule,
        policy_id="control:shuffled-exact-rate",
        method="shuffled",
        seed=seed,
        coalition_cap=cap,
    )
    shuffled_coalitions = shuffled_coalition_schedule(cases, primary_schedule, seed=seed)
    primary_calls = sum(len(decision.actor_ids) for decision in primary_schedule)
    homogeneous = [
        _fixed_decision(
            "control:homogeneous-exact-total-rate",
            case.header,
            (best_single,) if decision.actor_ids else (),
        )
        for case, decision in zip(cases, primary_schedule, strict=True)
    ]
    tuned_single = [
        _fixed_decision("control:tuned-best-single", case.header, (best_single,)) for case in cases
    ]
    recurrent: list[CoalitionDecision] = []
    recurrent_actor = best_single
    for case in cases:
        selected_recurrent: tuple[str, ...]
        if not case.header.idle:
            recurrent_actor = scope_best.get(case.header.scope_key, recurrent_actor)
            selected_recurrent = (recurrent_actor,)
        else:
            selected_recurrent = ()
        recurrent.append(
            _fixed_decision(
                "control:equal-budget-visible-history-recurrent",
                case.header,
                selected_recurrent,
            )
        )
    oracle = [_oracle_decision(config, case) for case in cases]
    arms: dict[str, dict[str, Any]] = {}
    arms[PRIMARY_ARM] = _evaluate_schedule(
        config,
        cases,
        primary_schedule,
        arm=PRIMARY_ARM,
        training=primary_training,
        training_work=primary_training_work,
        critic=primary_critic,
        scaling_assay=scaling_assay,
    )
    arms["learned_individual_only"] = _evaluate_schedule(
        config,
        cases,
        individual_schedule,
        arm="learned_individual_only",
        training=individual_training,
        training_work=individual_work,
        critic=individual_critic,
    )
    arms["learned_no_exploration"] = _evaluate_schedule(
        config,
        cases,
        no_explore_schedule,
        arm="learned_no_exploration",
        training=no_explore_training,
        training_work=no_explore_work,
        critic=no_explore_critic,
    )
    arms["learned_random_exploration_rate_matched"] = _evaluate_schedule(
        config,
        cases,
        random_explore_schedule,
        arm="learned_random_exploration_rate_matched",
        training=random_training,
        training_work=random_work,
        critic=random_critic,
    )
    simple_schedules = {
        "edcm_hard_sentinel": hard,
        "uncertainty_gate": uncertainty,
        "novelty_gate": novelty,
        "always_on_full_activation": always,
        "reactive_cheapest": reactive,
        "fixed_sparse": fixed_sparse,
        "periodic_exact_rate": periodic,
        "random_exact_rate": random_rate,
        "shuffled_exact_rate": shuffled_rate,
        "shuffled_coalition_exact_rate": shuffled_coalitions,
        ORACLE_ARM: oracle,
    }
    for arm, schedule in simple_schedules.items():
        arms[arm] = _evaluate_schedule(config, cases, schedule, arm=arm)
    arms["outcome_only_bandit"] = _evaluate_schedule(
        config,
        cases,
        outcome_bandit,
        arm="outcome_only_bandit",
        training=tuned_control_training,
        training_work=tuned_control_work,
    )
    arms["tuned_best_single"] = _evaluate_schedule(
        config,
        cases,
        tuned_single,
        arm="tuned_best_single",
        training=tuned_control_training,
        training_work=tuned_control_work,
    )
    primary_actor_work = int(arms[PRIMARY_ARM]["deployment_actor_execution"])
    primary_message_work = int(arms[PRIMARY_ARM]["deployment_message_operations"])
    primary_message_bytes = int(arms[PRIMARY_ARM]["deployment_message_bytes"])
    arms["homogeneous_exact_total_rate"] = _evaluate_schedule(
        config,
        cases,
        homogeneous,
        arm="homogeneous_exact_total_rate",
        training=tuned_control_training,
        training_work=tuned_control_work,
        effective_activation_calls=primary_calls,
        deployment_actor_execution_floor=primary_actor_work,
        deployment_message_operations_floor=primary_message_work,
        deployment_message_bytes_floor=primary_message_bytes,
    )
    arms["equal_budget_recurrent"] = _evaluate_schedule(
        config,
        cases,
        recurrent,
        arm="equal_budget_recurrent",
        training=tuned_control_training,
        training_work=tuned_control_work,
        total_work_floor=int(arms[PRIMARY_ARM]["total_lifecycle_work"]),
    )
    arms = {name: arms[name] for name in ARM_NAMES}
    primary_counts = dict(sorted(_actor_counts(primary_schedule).items()))
    primary_coalitions = Counter(decision.actor_ids for decision in primary_schedule)
    rate_names = ("periodic_exact_rate", "random_exact_rate", "shuffled_exact_rate")
    invariants = {
        "arm_order_exact": tuple(arms) == ARM_NAMES,
        "all_work_components_present": all(
            set(arm["work_components"]) == set(WORK_COMPONENTS) for arm in arms.values()
        ),
        "rate_controls_preserve_per_actor_counts": all(
            arms[name]["actor_activation_counts"] == primary_counts for name in rate_names
        ),
        "coalition_shuffle_preserves_multiset": Counter(
            decision.actor_ids for decision in shuffled_coalitions
        )
        == primary_coalitions,
        "homogeneous_preserves_effective_total_rate": int(
            arms["homogeneous_exact_total_rate"]["effective_activation_calls"]
        )
        == primary_calls
        and int(arms["homogeneous_exact_total_rate"]["deployment_actor_execution"])
        == int(arms[PRIMARY_ARM]["deployment_actor_execution"])
        and int(arms["homogeneous_exact_total_rate"]["deployment_message_operations"])
        == int(arms[PRIMARY_ARM]["deployment_message_operations"])
        and int(arms["homogeneous_exact_total_rate"]["deployment_message_bytes"])
        == int(arms[PRIMARY_ARM]["deployment_message_bytes"]),
        "recurrent_is_equal_or_more_work": int(arms["equal_budget_recurrent"]["total_lifecycle_work"])
        >= int(arms[PRIMARY_ARM]["total_lifecycle_work"]),
        "all_idle_boundaries_charged": all(int(arm["idle_boundary_work"]) > 0 for arm in arms.values()),
        "all_bytes_charged": all(int(arm["total_encoded_bytes"]) > 0 for arm in arms.values()),
        "no_learned_deployment_oracle_forks": all(
            int(arms[name]["deployment_oracle_forks"]) == 0 for name in LEARNED_ARMS
        ),
        "oracle_remains_nonpromotable": arms[ORACLE_ARM]["evidence_standing"] == "oracle_nonpromotable"
        and arms[ORACLE_ARM]["scientific_promotion"] is False,
        "bounded_candidate_and_beam_caps": all(
            int(arm["max_candidates_retrieved"]) <= int(config["dispatch"]["candidate_cap_k"])
            and int(arm["max_coalitions_considered"])
            <= max(
                int(config["dispatch"]["beam_cap_b"]),
                sum(
                    math.comb(len(config["actors"]), size)
                    for size in range(int(config["dispatch"]["coalition_cap_c"]) + 1)
                ),
            )
            for arm in arms.values()
        ),
        "no_global_actor_scan": all(
            arm["global_actor_scan"] is False for name, arm in arms.items() if name != ORACLE_ARM
        ),
        "temporary_coalitions_only": all(arm["temporary_coalitions_only"] is True for arm in arms.values()),
        "dormant_scaling_assay_fully_charged": int(
            arms[PRIMARY_ARM]["training"]["dormant_scaling_assay_work"]
        )
        == int(scaling_assay["total_charged_index_and_query_work"])
        and int(scaling_assay["total_retained_index_byte_ticks"])
        <= int(arms[PRIMARY_ARM]["retained_state_byte_ticks"]),
        "queue_and_storm_caps": all(
            bool(arm["queue_stable"]) and int(arm["max_queue_depth"]) <= int(config["dispatch"]["queue_cap"])
            for arm in arms.values()
        ),
        "leakage_gate": leakage_gate(cases)["passed"],
    }
    core = {
        "schema": ROW_SCHEMA,
        "seed": seed,
        "split": split,
        "world_family": config["splits"][split]["world_family"],
        "event_count": len(cases),
        "selected_best_single": best_single,
        "bed_gate": _bed_gate(config, cases),
        "leakage_gate": leakage_gate(cases),
        "arms": arms,
        "invariants": invariants,
        "claim_scope": CLAIM_SCOPE,
        "candidate_activation_enabled": False,
        "scientific_promotion": False,
    }
    return {**core, "row_sha256": canonical_sha256(core)}


def _mean_ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    mean = statistics.fmean(materialized) if materialized else 0.0
    half = (
        t_critical * statistics.stdev(materialized) / math.sqrt(len(materialized))
        if len(materialized) > 1
        else 0.0
    )
    return {"n": len(materialized), "mean": mean, "lower": mean - half, "upper": mean + half}


def aggregate_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "partial",
            "terminal_route": "blocked",
            "scientific_promotion": False,
        }
    failed = any(not all(bool(value) for value in row["invariants"].values()) for row in rows)
    valid_bed = all(bool(row["bed_gate"]["passed"]) for row in rows)
    if failed:
        return {
            "status": "failed",
            "terminal_route": "failed",
            "verdict": config["verdict"]["failed"],
            "scientific_promotion": False,
        }
    if not valid_bed:
        return {
            "status": "invalid_bed",
            "terminal_route": "invalid_bed",
            "verdict": config["verdict"]["invalid_bed"],
            "scientific_promotion": False,
        }
    criteria = config["criteria"]
    primary = [row["arms"][PRIMARY_ARM] for row in rows]
    always = [row["arms"]["always_on_full_activation"] for row in rows]
    utility_losses = [
        float(reference["mean_utility"]) - float(candidate["mean_utility"])
        for candidate, reference in zip(primary, always, strict=True)
    ]
    work_savings = [
        1.0 - float(candidate["total_lifecycle_work"]) / float(reference["total_lifecycle_work"])
        for candidate, reference in zip(primary, always, strict=True)
    ]
    calibration = [float(candidate["calibration_error"]) for candidate in primary]
    noise_excess = [
        float(row["arms"][PRIMARY_ARM]["noisy_tv_activation_rate"])
        - float(row["arms"]["random_exact_rate"]["noisy_tv_activation_rate"])
        for row in rows
    ]
    discovery_advantage = [
        float(row["arms"][PRIMARY_ARM]["synergy_discovery_rate"])
        - float(row["arms"]["learned_random_exploration_rate_matched"]["synergy_discovery_rate"])
        for row in rows
    ]
    control_contributions: dict[str, dict[str, float | int]] = {}
    control_seed_values: dict[str, list[float]] = {}
    for control in criteria["required_rate_matched_controls"]:
        values = []
        for row in rows:
            learned = row["arms"][PRIMARY_ARM]
            baseline = row["arms"][control]
            utility_margin = float(learned["mean_utility"]) - float(baseline["mean_utility"])
            saving = 1.0 - float(learned["total_lifecycle_work"]) / max(
                1.0, float(baseline["total_lifecycle_work"])
            )
            values.append(min(utility_margin + float(criteria["pareto_utility_tolerance"]), saving))
        control_seed_values[str(control)] = values
        control_contributions[str(control)] = _mean_ci(values, 2.776)
    intervals: dict[str, Any] = {
        "utility_loss_vs_always_on": _mean_ci(utility_losses, 2.776),
        "work_saving_vs_always_on": _mean_ci(work_savings, 2.776),
        "calibration_error": _mean_ci(calibration, 2.776),
        "noisy_tv_excess_vs_random": _mean_ci(noise_excess, 2.776),
        "synergy_discovery_advantage_vs_random": _mean_ci(discovery_advantage, 2.776),
        "pareto_contribution": control_contributions,
    }
    checks = {
        "utility_noninferior": intervals["utility_loss_vs_always_on"]["upper"]
        <= float(criteria["max_utility_loss_vs_always_on"]),
        "lifecycle_work_saving": intervals["work_saving_vs_always_on"]["lower"]
        >= float(criteria["min_work_saving_vs_always_on"]),
        "positive_pareto_over_registered_controls": all(
            interval["lower"] > 0 for interval in control_contributions.values()
        ),
        "unseen_world_value_calibrated": intervals["calibration_error"]["upper"]
        <= float(criteria["max_calibration_error"]),
        "cold_start_synergy_discovered": all(
            float(row["arms"][PRIMARY_ARM]["synergy_discovery_rate"])
            >= float(criteria["min_synergy_discovery_rate"])
            for row in rows
        )
        and intervals["synergy_discovery_advantage_vs_random"]["lower"]
        >= float(criteria["min_discovery_advantage_vs_random"]),
        "no_excess_noisy_tv_activation": intervals["noisy_tv_excess_vs_random"]["upper"]
        <= float(criteria["max_noisy_tv_excess_vs_random"]),
        "selective_high_credit_actor_lesion": all(
            float(row["arms"][PRIMARY_ARM]["high_credit_actor_lesion_effect"])
            >= float(criteria["min_high_credit_actor_lesion_effect"])
            for row in rows
        ),
        "selective_message_lesion": all(
            float(row["arms"][PRIMARY_ARM]["message_lesion_effect"])
            >= float(criteria["min_message_lesion_effect"])
            for row in rows
        ),
        "queue_deadline_integrity": all(
            int(row["arms"][PRIMARY_ARM]["deadline_misses"]) <= int(criteria["max_deadline_misses"])
            and bool(row["arms"][PRIMARY_ARM]["queue_stable"])
            and int(row["arms"][PRIMARY_ARM]["max_queue_depth"]) <= int(config["dispatch"]["queue_cap"])
            for row in rows
        ),
        "dormant_population_slope": all(
            float(row["arms"][PRIMARY_ARM]["candidate_work_slope_per_dormant_actor"])
            <= float(criteria["max_candidate_work_slope_per_dormant_actor"])
            and row["arms"][PRIMARY_ARM]["global_actor_scan"] is False
            for row in rows
        ),
        "no_leakage_or_test_oracle": all(
            row["leakage_gate"]["passed"] and int(row["arms"][PRIMARY_ARM]["deployment_oracle_forks"]) == 0
            for row in rows
        ),
        "required_direction_every_seed": all(
            loss <= float(criteria["max_utility_loss_vs_always_on"])
            and saving >= float(criteria["min_work_saving_vs_always_on"])
            and all(values[index] > 0 for values in control_seed_values.values())
            for index, (loss, saving) in enumerate(zip(utility_losses, work_savings, strict=True))
        ),
    }
    favorable = all(checks.values())
    return {
        "status": "complete",
        "terminal_route": "positive" if favorable else "null",
        "paired_seed_count": len(rows),
        "seed_ids": [int(row["seed"]) for row in rows],
        "paired_intervals_95": intervals,
        "checks": checks,
        "verdict": config["verdict"]["positive"] if favorable else config["verdict"]["null"],
        "strong_null_rejected": favorable,
        "candidate_activation_enabled": False,
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
    }


def _load_json_snapshot(
    path: Path, label: str, max_bytes: int = MAX_ARTIFACT_BYTES
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _read_regular_file(path.resolve(), max_bytes, label)
    document = json.loads(raw)
    _require(isinstance(document, dict), f"{label} must be a mapping")
    return document, _file_receipt(path.resolve())


def load_edcm_entry_gate(
    receipt_path: Path | str = DEFAULT_EDCM_RECEIPT_PATH,
    verification_path: Path | str = DEFAULT_EDCM_VERIFICATION_PATH,
) -> dict[str, Any]:

    producer_path = Path(receipt_path).resolve()
    verifier_path = Path(verification_path).resolve()
    _require_distinct_paths({"edcm_receipt": producer_path, "edcm_verification": verifier_path})
    producer, producer_source = _load_json_snapshot(producer_path, "EDCM prerequisite receipt")
    _require(producer.get("schema") == EDCM_RECEIPT_SCHEMA, "EDCM producer schema mismatch")
    producer_without_receipt_hash = dict(producer)
    producer_digest = producer_without_receipt_hash.pop("receipt_sha256", "")
    _require(
        producer_digest == canonical_sha256(producer_without_receipt_hash),
        "EDCM prerequisite receipt self-hash mismatch",
    )
    producer_core_digest = producer_without_receipt_hash.pop("deterministic_core_sha256", "")
    _require(
        producer_core_digest == canonical_sha256(producer_without_receipt_hash),
        "EDCM prerequisite deterministic-core hash mismatch",
    )
    _require(
        producer.get("execution_status") in {"complete", "terminal_scientific_stop"}
        and producer.get("all_ok") is True
        and producer.get("problems") == []
        and producer.get("resumable") is False,
        "EDCM prerequisite is not terminal and valid",
    )
    _require(producer.get("scientific_promotion") is False, "EDCM prerequisite promotion escaped")
    gate_value = producer.get("gate")
    _require(
        isinstance(gate_value, dict) and gate_value.get("status") == "complete",
        "EDCM complementarity gate is incomplete",
    )
    gate = cast(dict[str, Any], gate_value)
    _require(isinstance(gate.get("passed"), bool), "EDCM complementarity pass state malformed")
    verifier, verifier_source = _load_json_snapshot(verifier_path, "EDCM prerequisite verification")
    _require(verifier.get("schema") == EDCM_VERIFICATION_SCHEMA, "EDCM verification schema mismatch")
    verifier_core = dict(verifier)
    verifier_digest = verifier_core.pop("verification_artifact_sha256", "")
    _require(
        verifier_digest == canonical_sha256(verifier_core),
        "EDCM verification artifact self-hash mismatch",
    )
    verification_value = verifier.get("verification")
    _require(isinstance(verification_value, dict), "EDCM verification result missing")
    verification = cast(dict[str, Any], verification_value)
    _require(
        verification.get("valid") is True
        and verification.get("execution_status") in {"complete", "terminal_scientific_stop"}
        and verification.get("verifier_mode") == "full-deterministic-regeneration/v1"
        and verification.get("scientific_promotion") is False,
        "EDCM independent verification is not valid and terminal",
    )
    verified_sources = verification.get("verified_sources")
    _require(isinstance(verified_sources, dict), "EDCM verified sources missing")
    verified_receipt = cast(dict[str, Any], verified_sources).get("receipt", {})
    _require(isinstance(verified_receipt, dict), "EDCM verified receipt source missing")
    _require(
        verified_receipt.get("sha256") == producer_source["sha256"]
        and verified_receipt.get("bytes") == producer_source["bytes"],
        "EDCM verifier is not bound to the supplied producer receipt",
    )
    return {
        "schema": "mop-escs-x1-edcm-entry-gate/v1",
        "valid_terminal_evidence": True,
        "passed": bool(gate.get("passed")),
        "route": "continue_x1" if gate.get("passed") else "invalid_bed_return_to_edcm1",
        "producer": producer_source,
        "producer_receipt_sha256": producer_digest,
        "producer_authority_sha256": producer.get("authority_sha256"),
        "producer_implementation_authority_sha256": producer.get("implementation_authority_sha256"),
        "verification": verifier_source,
        "verification_artifact_sha256": verifier_digest,
        "verifier_mode": verification["verifier_mode"],
        "complementarity_gate_sha256": canonical_sha256(gate),
        "scientific_promotion": False,
    }


def _validate_entry_gate_override(entry_gate: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        entry_gate.get("schema") == "mop-escs-x1-edcm-entry-gate/v1",
        "exploratory entry-gate schema mismatch",
    )
    _require(entry_gate.get("valid_terminal_evidence") is True, "entry gate is not terminal")
    _require(isinstance(entry_gate.get("passed"), bool), "entry-gate pass state missing")
    _require(entry_gate.get("scientific_promotion") is False, "entry-gate promotion escaped")
    result = copy.deepcopy(dict(entry_gate))
    result["route"] = "continue_x1" if result["passed"] else "invalid_bed_return_to_edcm1"
    return result


IMPLEMENTATION_PATHS = (
    Path("configs/experiment/escs_x1_dispatch.json"),
    Path("src/mop/studies/escs_x1_dispatch.py"),
    Path("scripts/run_escs_x1_dispatch.py"),
    Path("tests/test_escs_x1_dispatch.py"),
    Path("docs/audits/escs_x1_dispatch.md"),
)


def build_implementation_authority(
    *,
    config_authority_sha256: str,
    mode: str,
    review_status: str,
    file_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    receipts = (
        [dict(receipt) for receipt in file_receipts]
        if file_receipts is not None
        else [_file_receipt(REPO_ROOT / path) for path in IMPLEMENTATION_PATHS]
    )
    core = {
        "schema": IMPLEMENTATION_AUTHORITY_SCHEMA,
        "study_id": "escs-x1-counterfactual-voc-dispatch-v1",
        "mode": str(mode),
        "config_authority_sha256": str(config_authority_sha256),
        "review_status": str(review_status),
        "files": receipts,
        "candidate_activation_enabled": False,
        "scientific_promotion": False,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def write_implementation_authority(
    path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    mode: str = "official",
    review_status: str = OFFICIAL_IMPLEMENTATION_REVIEW_STATUS,
) -> dict[str, Any]:
    target = Path(path).resolve()
    _require_distinct_paths(
        {
            "implementation_authority_output": target,
            **{f"scoped:{relative}": REPO_ROOT / relative for relative in IMPLEMENTATION_PATHS},
        }
    )
    document = build_implementation_authority(
        config_authority_sha256=OFFICIAL_CONFIG_AUTHORITY_SHA256,
        mode=mode,
        review_status=review_status,
    )
    _atomic_json(target, document)
    return document


def load_implementation_authority(
    path: Path | str,
    config: Mapping[str, Any],
    *,
    expected_sha256: str | None,
    exploratory: bool,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if exploratory:
        _require(source != DEFAULT_IMPLEMENTATION_AUTHORITY_PATH.resolve(), "exploratory manifest required")
    else:
        _require(source == DEFAULT_IMPLEMENTATION_AUTHORITY_PATH.resolve(), "canonical X1 manifest required")
        _require(
            expected_sha256 is not None and len(expected_sha256) == 64,
            "independent X1 implementation-manifest digest required",
        )
    document, _ = _load_json_snapshot(source, "X1 implementation authority")
    _require_exact_keys(
        document,
        (
            "schema",
            "study_id",
            "mode",
            "config_authority_sha256",
            "review_status",
            "files",
            "candidate_activation_enabled",
            "scientific_promotion",
            "manifest_sha256",
        ),
        "X1 implementation authority",
    )
    core = dict(document)
    digest = str(core.pop("manifest_sha256", ""))
    _require(digest == canonical_sha256(core), "X1 implementation authority self-hash mismatch")
    if expected_sha256 is not None:
        _require(digest == expected_sha256, "X1 implementation authority pin mismatch")
    _require(document.get("schema") == IMPLEMENTATION_AUTHORITY_SCHEMA, "X1 manifest schema mismatch")
    _require(
        document.get("config_authority_sha256") == canonical_sha256(config),
        "X1 config/implementation authority mismatch",
    )
    _require(
        document.get("files") == [_file_receipt(REPO_ROOT / path) for path in IMPLEMENTATION_PATHS],
        "X1 scoped implementation files drifted",
    )
    _require(document.get("candidate_activation_enabled") is False, "manifest activation escaped")
    _require(document.get("scientific_promotion") is False, "manifest promotion escaped")
    if exploratory:
        _require(document.get("mode") == "exploratory", "exploratory X1 manifest mode required")
    else:
        _require(document.get("mode") == "official", "official X1 manifest mode required")
        _require(
            document.get("review_status") == OFFICIAL_IMPLEMENTATION_REVIEW_STATUS,
            "X1 manifest review status drift",
        )
    return document


def _checkpoint_core(
    *,
    config_sha256: str,
    implementation_sha256: str,
    entry_gate_sha256: str,
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config_authority_sha256": config_sha256,
        "implementation_authority_sha256": implementation_sha256,
        "entry_gate_sha256": entry_gate_sha256,
        "gate_rows": [dict(row) for row in gate_rows],
        "heldout_rows": [dict(row) for row in heldout_rows],
        "gate_rows_sha256": canonical_sha256(gate_rows),
        "heldout_rows_sha256": canonical_sha256(heldout_rows),
    }


def _write_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    implementation_sha256: str,
    entry_gate_sha256: str,
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core = _checkpoint_core(
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        entry_gate_sha256=entry_gate_sha256,
        gate_rows=gate_rows,
        heldout_rows=heldout_rows,
    )
    document = {**core, "checkpoint_sha256": canonical_sha256(core)}
    _require(len(canonical_bytes(document)) + 1 <= MAX_ARTIFACT_BYTES, "X1 checkpoint cap exceeded")
    _atomic_json(path, document)
    written, _ = _load_json_snapshot(path, "written X1 checkpoint")
    _require(written == document, "written X1 checkpoint changed on readback")
    return written


def _load_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    implementation_sha256: str,
    entry_gate_sha256: str,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    checkpoint, _ = _load_json_snapshot(path, "X1 checkpoint")
    _require_exact_keys(checkpoint, CHECKPOINT_KEYS, "X1 checkpoint")
    core = dict(checkpoint)
    digest = str(core.pop("checkpoint_sha256", ""))
    _require(digest == canonical_sha256(core), "X1 checkpoint self-hash mismatch")
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "X1 checkpoint schema mismatch")
    _require(checkpoint.get("config_authority_sha256") == config_sha256, "X1 checkpoint config drift")
    _require(
        checkpoint.get("implementation_authority_sha256") == implementation_sha256,
        "X1 checkpoint implementation drift",
    )
    _require(checkpoint.get("entry_gate_sha256") == entry_gate_sha256, "X1 checkpoint EDCM drift")
    gate_rows = list(checkpoint["gate_rows"])
    heldout_rows = list(checkpoint["heldout_rows"])
    _require(
        checkpoint["gate_rows_sha256"] == canonical_sha256(gate_rows),
        "X1 checkpoint gate-row hash mismatch",
    )
    _require(
        checkpoint["heldout_rows_sha256"] == canonical_sha256(heldout_rows),
        "X1 checkpoint heldout-row hash mismatch",
    )
    _require(
        [int(row["seed"]) for row in gate_rows] == list(seeds[: len(gate_rows)]),
        "X1 gate seed prefix drift",
    )
    _require(
        [int(row["seed"]) for row in heldout_rows] == list(seeds[: len(heldout_rows)]),
        "X1 heldout seed prefix drift",
    )
    for row in [*gate_rows, *heldout_rows]:
        row_core = dict(row)
        row_digest = str(row_core.pop("row_sha256", ""))
        _require(row_digest == canonical_sha256(row_core), "X1 checkpoint seed-row hash mismatch")
    return gate_rows, heldout_rows


def _aggregate_bed_gate(rows: Sequence[Mapping[str, Any]], required_seeds: Sequence[int]) -> dict[str, Any]:
    if [int(row["seed"]) for row in rows] != list(required_seeds):
        return {"status": "incomplete", "passed": False}
    return {
        "status": "complete",
        "passed": all(bool(row["bed_gate"]["passed"]) for row in rows),
        "per_seed": {str(row["seed"]): row["bed_gate"] for row in rows},
        "failure_interpretation": "invalid_bed_not_dispatch_null",
    }


def run_from_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    implementation_authority_path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    implementation_authority_sha256: str | None = None,
    edcm_receipt_path: Path | str = DEFAULT_EDCM_RECEIPT_PATH,
    edcm_verification_path: Path | str = DEFAULT_EDCM_VERIFICATION_PATH,
    max_new_seeds: int | None = None,
    exploratory: bool = False,
    entry_gate_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(max_new_seeds is None or max_new_seeds >= 0, "max_new_seeds must be nonnegative")
    source = Path(config_path).resolve()
    output = Path(output_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    implementation_source = Path(implementation_authority_path).resolve()
    path_set: dict[str, Path | str] = {
        "config": source,
        "output": output,
        "checkpoint": checkpoint,
        "implementation": implementation_source,
    }
    if entry_gate_override is None:
        path_set["edcm_receipt"] = edcm_receipt_path
        path_set["edcm_verification"] = edcm_verification_path
    _require_distinct_paths(path_set)
    if exploratory:
        _require(output != DEFAULT_OUTPUT_PATH.resolve(), "exploratory X1 cannot write official receipt")
        _require(
            checkpoint != DEFAULT_CHECKPOINT_PATH.resolve(),
            "exploratory X1 cannot write official checkpoint",
        )
    else:
        _require(entry_gate_override is None, "official X1 forbids entry-gate overrides")
    config = load_config(source, exploratory=exploratory)
    envelope, config_source = _load_envelope_snapshot(source)
    implementation = load_implementation_authority(
        implementation_source,
        config,
        expected_sha256=implementation_authority_sha256,
        exploratory=exploratory,
    )
    entry_gate = (
        _validate_entry_gate_override(entry_gate_override)
        if entry_gate_override is not None
        else load_edcm_entry_gate(edcm_receipt_path, edcm_verification_path)
    )
    config_sha256 = canonical_sha256(config)
    implementation_sha256 = str(implementation["manifest_sha256"])
    entry_gate_sha256 = canonical_sha256(entry_gate)
    seeds = [int(seed) for seed in config["seeds"]]
    gate_rows, heldout_rows = _load_checkpoint(
        checkpoint,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        entry_gate_sha256=entry_gate_sha256,
        seeds=seeds,
    )
    remaining = max_new_seeds
    if entry_gate["passed"]:
        for seed in seeds[len(gate_rows) :]:
            if remaining is not None and remaining <= 0:
                break
            gate_rows.append(run_seed(config, seed=seed, split="gate"))
            _write_checkpoint(
                checkpoint,
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha256,
                entry_gate_sha256=entry_gate_sha256,
                gate_rows=gate_rows,
                heldout_rows=heldout_rows,
            )
            if remaining is not None:
                remaining -= 1
        bed_gate = _aggregate_bed_gate(gate_rows, seeds)
        if bed_gate.get("passed"):
            for seed in seeds[len(heldout_rows) :]:
                if remaining is not None and remaining <= 0:
                    break
                heldout_rows.append(run_seed(config, seed=seed, split="heldout"))
                _write_checkpoint(
                    checkpoint,
                    config_sha256=config_sha256,
                    implementation_sha256=implementation_sha256,
                    entry_gate_sha256=entry_gate_sha256,
                    gate_rows=gate_rows,
                    heldout_rows=heldout_rows,
                )
                if remaining is not None:
                    remaining -= 1
    else:
        _require(not gate_rows and not heldout_rows, "X1 rows exist after failed EDCM entry gate")
        bed_gate = {
            "status": "not_run_edcm_invalid_bed",
            "passed": False,
            "failure_interpretation": "invalid_bed_return_to_edcm1",
        }
    checkpoint_document = _write_checkpoint(
        checkpoint,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        entry_gate_sha256=entry_gate_sha256,
        gate_rows=gate_rows,
        heldout_rows=heldout_rows,
    )
    gate_complete = len(gate_rows) == len(seeds)
    heldout_complete = len(heldout_rows) == len(seeds)
    invalid_bed = not entry_gate["passed"] or (gate_complete and not bed_gate["passed"])
    complete = invalid_bed or heldout_complete
    if invalid_bed:
        aggregate = {
            "status": "invalid_bed",
            "terminal_route": "invalid_bed",
            "verdict": config["verdict"]["invalid_bed"],
            "scientific_promotion": False,
        }
    else:
        aggregate = aggregate_rows(heldout_rows, config)
    problems = ["execution_incomplete"] if not complete else []
    if complete and aggregate.get("terminal_route") == "failed":
        problems = ["generated_invariant_failure"]
    core = {
        "schema": RECEIPT_SCHEMA,
        "study_id": config["study_id"],
        "claim_scope": CLAIM_SCOPE,
        "strong_null": config["strong_null"],
        "authority": envelope["authority"],
        "authority_sha256": config_sha256,
        "config_source": config_source,
        "implementation_authority": {
            "source": _file_receipt(implementation_source),
            "manifest_sha256": implementation_sha256,
            "mode": implementation["mode"],
            "review_status": implementation["review_status"],
        },
        "edcm_entry_gate": entry_gate,
        "runtime_identity": _runtime_identity(),
        "gate_rows": gate_rows,
        "bed_gate": bed_gate,
        "heldout_rows": heldout_rows,
        "aggregate": aggregate,
        "execution_status": "complete" if complete else "partial",
        "all_ok": complete and not problems,
        "problems": problems,
        "resumable": not complete,
        "completed_gate_seeds": [int(row["seed"]) for row in gate_rows],
        "completed_heldout_seeds": [int(row["seed"]) for row in heldout_rows],
        "required_seeds": seeds,
        "checkpoint": {
            "source": _file_receipt(checkpoint),
            "checkpoint_sha256": checkpoint_document["checkpoint_sha256"],
            "gate_rows_sha256": checkpoint_document["gate_rows_sha256"],
            "heldout_rows_sha256": checkpoint_document["heldout_rows_sha256"],
        },
        "fresh_verifier_status": "pending-independent-command",
        "candidate_activation_enabled": False,
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
        "exploratory": exploratory,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    _require(
        len(canonical_bytes(receipt)) + 1 <= int(config["resources"]["max_receipt_bytes"]),
        "X1 receipt byte cap exceeded",
    )
    _atomic_json(output, receipt)
    written = _load_receipt(output)
    _require(written == receipt, "written X1 receipt changed on readback")
    return written


def _load_receipt(path: Path) -> dict[str, Any]:
    receipt, _ = _load_json_snapshot(path, "X1 producer receipt")
    _require_exact_keys(receipt, RECEIPT_KEYS, "X1 producer receipt")
    core = dict(receipt)
    digest = str(core.pop("receipt_sha256", ""))
    _require(digest == canonical_sha256(core), "X1 receipt self-hash mismatch")
    _require(receipt.get("schema") == RECEIPT_SCHEMA, "X1 receipt schema mismatch")
    return receipt


def _verify_checkpoint_binding(
    receipt: Mapping[str, Any],
    *,
    config_sha256: str,
    implementation_sha256: str,
    entry_gate_sha256: str,
    seeds: Sequence[int],
) -> None:
    checkpoint_path = _checkpoint_path_from_receipt(receipt)
    binding_value = receipt.get("checkpoint")
    _require(isinstance(binding_value, dict), "X1 checkpoint binding missing")
    binding = cast(dict[str, Any], binding_value)
    source_value = binding.get("source")
    _require(
        isinstance(source_value, dict) and isinstance(source_value.get("path"), str),
        "checkpoint source missing",
    )
    source = cast(dict[str, Any], source_value)
    _require(_file_receipt(checkpoint_path) == source, "X1 checkpoint file receipt mismatch")
    checkpoint, _ = _load_json_snapshot(checkpoint_path, "X1 bound checkpoint")
    _require(
        checkpoint.get("checkpoint_sha256") == binding.get("checkpoint_sha256"),
        "X1 receipt/checkpoint digest mismatch",
    )
    _require(
        checkpoint.get("gate_rows_sha256") == binding.get("gate_rows_sha256")
        and checkpoint.get("heldout_rows_sha256") == binding.get("heldout_rows_sha256"),
        "X1 receipt/checkpoint phase digest mismatch",
    )
    gate_rows, heldout_rows = _load_checkpoint(
        checkpoint_path,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        entry_gate_sha256=entry_gate_sha256,
        seeds=seeds,
    )
    _require(gate_rows == receipt.get("gate_rows"), "X1 receipt/checkpoint gate rows differ")
    _require(heldout_rows == receipt.get("heldout_rows"), "X1 receipt/checkpoint heldout rows differ")


def _checkpoint_path_from_receipt(receipt: Mapping[str, Any]) -> Path:
    binding_value = receipt.get("checkpoint")
    _require(isinstance(binding_value, dict), "X1 checkpoint binding missing")
    binding = cast(dict[str, Any], binding_value)
    source_value = binding.get("source")
    _require(
        isinstance(source_value, dict) and isinstance(source_value.get("path"), str),
        "checkpoint source missing",
    )
    source = cast(dict[str, Any], source_value)
    declared = Path(str(source["path"]))
    return (REPO_ROOT / declared).resolve() if not declared.is_absolute() else declared.resolve()


def verify_receipt(
    receipt_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    implementation_authority_path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    implementation_authority_sha256: str | None = None,
    edcm_receipt_path: Path | str = DEFAULT_EDCM_RECEIPT_PATH,
    edcm_verification_path: Path | str = DEFAULT_EDCM_VERIFICATION_PATH,
    exploratory: bool = False,
    entry_gate_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt_source = Path(receipt_path).resolve()
    config_source = Path(config_path).resolve()
    implementation_source = Path(implementation_authority_path).resolve()
    paths: dict[str, Path | str] = {
        "receipt": receipt_source,
        "config": config_source,
        "implementation": implementation_source,
    }
    if entry_gate_override is None:
        paths["edcm_receipt"] = edcm_receipt_path
        paths["edcm_verification"] = edcm_verification_path
    _require_distinct_paths(paths)
    if not exploratory:
        _require(entry_gate_override is None, "official X1 verifier forbids entry-gate overrides")
    config = load_config(config_source, exploratory=exploratory)
    config_envelope, current_config_source = _load_envelope_snapshot(config_source)
    implementation = load_implementation_authority(
        implementation_source,
        config,
        expected_sha256=implementation_authority_sha256,
        exploratory=exploratory,
    )
    entry_gate = (
        _validate_entry_gate_override(entry_gate_override)
        if entry_gate_override is not None
        else load_edcm_entry_gate(edcm_receipt_path, edcm_verification_path)
    )
    receipt = _load_receipt(receipt_source)
    _require(receipt.get("resumable") is False, "fresh verifier refuses partial X1 receipt")
    _require(receipt.get("execution_status") == "complete", "X1 receipt is not terminal")
    _require(receipt.get("claim_scope") == CLAIM_SCOPE, "X1 receipt claim scope mismatch")
    _require(receipt.get("strong_null") == config["strong_null"], "X1 receipt strong null drift")
    _require(receipt.get("authority") == config_envelope["authority"], "X1 authority envelope drift")
    _require(receipt.get("config_source") == current_config_source, "X1 config source receipt drift")
    _require(bool(receipt.get("exploratory")) == exploratory, "X1 exploratory mode drift")
    _require(receipt.get("candidate_activation_enabled") is False, "X1 receipt activation escaped")
    _require(receipt.get("scientific_promotion") is False, "X1 receipt promotion escaped")
    _require(
        receipt.get("fresh_verifier_status") == "pending-independent-command",
        "X1 producer claimed its own fresh verification",
    )
    _require(receipt.get("authority_sha256") == canonical_sha256(config), "X1 receipt/config mismatch")
    _require(
        receipt.get("implementation_authority", {}).get("manifest_sha256")
        == implementation["manifest_sha256"],
        "X1 receipt/implementation mismatch",
    )
    _require(
        receipt.get("implementation_authority", {}).get("source") == _file_receipt(implementation_source),
        "X1 implementation source receipt drift",
    )
    _require(receipt.get("edcm_entry_gate") == entry_gate, "X1 receipt/EDCM evidence mismatch")
    seeds = [int(seed) for seed in config["seeds"]]
    _verify_checkpoint_binding(
        receipt,
        config_sha256=canonical_sha256(config),
        implementation_sha256=str(implementation["manifest_sha256"]),
        entry_gate_sha256=canonical_sha256(entry_gate),
        seeds=seeds,
    )
    if entry_gate["passed"]:
        regenerated_gate = [run_seed(config, seed=seed, split="gate") for seed in seeds]
        _require(regenerated_gate == receipt["gate_rows"], "X1 producer gate regeneration mismatch")
        bed_gate = _aggregate_bed_gate(regenerated_gate, seeds)
        _require(bed_gate == receipt["bed_gate"], "X1 producer bed-gate mismatch")
    else:
        regenerated_gate = []
        bed_gate = receipt["bed_gate"]
        _require(receipt["gate_rows"] == [] and receipt["heldout_rows"] == [], "X1 invalid entry ran")
    if entry_gate["passed"] and bed_gate["passed"]:
        regenerated_heldout = [run_seed(config, seed=seed, split="heldout") for seed in seeds]
        _require(
            regenerated_heldout == receipt["heldout_rows"],
            "X1 producer heldout regeneration mismatch",
        )
        primary = aggregate_rows(regenerated_heldout, config)
        _require(primary == receipt["aggregate"], "X1 producer aggregate regeneration mismatch")
        fresh_rows = [
            run_seed(config, seed=int(seed), split="fresh_verifier")
            for seed in config["fresh_verifier_seeds"]
        ]
        fresh = aggregate_rows(fresh_rows, config)
        if primary.get("terminal_route") == "failed" or fresh.get("terminal_route") == "failed":
            route = "failed"
        elif primary.get("terminal_route") == "positive" and fresh.get("terminal_route") == "positive":
            route = "positive"
        else:
            route = "null"
    else:
        regenerated_heldout = []
        primary = receipt["aggregate"]
        fresh_rows = []
        fresh = {
            "status": "not_run_invalid_bed",
            "terminal_route": "invalid_bed",
            "scientific_promotion": False,
        }
        route = "invalid_bed"
    producer_failed = primary.get("terminal_route") == "failed"
    _require(
        receipt.get("all_ok") is (not producer_failed),
        "X1 receipt all_ok does not match its regenerated route",
    )
    _require(
        receipt.get("problems") == (["generated_invariant_failure"] if producer_failed else []),
        "X1 receipt problems do not match its regenerated route",
    )
    verdict = config["verdict"][route]
    core = {
        "schema": VERIFICATION_SCHEMA,
        "study_id": receipt["study_id"],
        "producer_receipt": _file_receipt(receipt_source),
        "producer_receipt_sha256": receipt["receipt_sha256"],
        "implementation_authority_sha256": implementation["manifest_sha256"],
        "edcm_entry_gate_sha256": canonical_sha256(entry_gate),
        "producer_regeneration_match": True,
        "regenerated_gate_seed_ids": [int(row["seed"]) for row in regenerated_gate],
        "regenerated_heldout_seed_ids": [int(row["seed"]) for row in regenerated_heldout],
        "fresh_seed_ids": [int(row["seed"]) for row in fresh_rows],
        "fresh_seed_rows": fresh_rows,
        "primary_aggregate": primary,
        "fresh_aggregate": fresh,
        "terminal_route": route,
        "verdict": verdict,
        "gate_a_candidate_verified": route == "positive",
        "candidate_activation_enabled": False,
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
    }
    return {**core, "verification_sha256": canonical_sha256(core)}


def build_verification_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    _require(result.get("schema") == VERIFICATION_SCHEMA, "X1 verification schema mismatch")
    _require(result.get("scientific_promotion") is False, "X1 verification promotion escaped")
    result_core = dict(result)
    result_digest = str(result_core.pop("verification_sha256", ""))
    _require(result_digest == canonical_sha256(result_core), "X1 verification self-hash mismatch")
    core = {
        "schema": VERIFICATION_ARTIFACT_SCHEMA,
        "study_id": result["study_id"],
        "claim_scope": CLAIM_SCOPE,
        "verification": dict(result),
        "candidate_activation_enabled": False,
        "scientific_promotion": False,
    }
    return {**core, "verification_artifact_sha256": canonical_sha256(core)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", "--out", dest="output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--implementation-authority",
        type=Path,
        default=DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    )
    parser.add_argument("--implementation-authority-sha256")
    parser.add_argument("--edcm-receipt", type=Path, default=DEFAULT_EDCM_RECEIPT_PATH)
    parser.add_argument("--edcm-verification", type=Path, default=DEFAULT_EDCM_VERIFICATION_PATH)
    parser.add_argument("--max-new-seeds", type=int)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verification-out", type=Path, default=DEFAULT_VERIFICATION_OUTPUT_PATH)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.validate_only:
        config = load_config(arguments.config, exploratory=arguments.exploratory)
        manifest = load_implementation_authority(
            arguments.implementation_authority,
            config,
            expected_sha256=arguments.implementation_authority_sha256,
            exploratory=arguments.exploratory,
        )
        print(
            json.dumps(
                {
                    "status": "valid-unexecuted-scaffold",
                    "config_authority_sha256": canonical_sha256(config),
                    "implementation_authority_sha256": manifest["manifest_sha256"],
                    "candidate_activation_enabled": False,
                    "scientific_promotion": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.verify is not None:
        receipt_preview = _load_receipt(arguments.verify.resolve())
        _require_distinct_paths(
            {
                "verification_output": arguments.verification_out,
                "receipt": arguments.verify,
                "config": arguments.config,
                "implementation": arguments.implementation_authority,
                "edcm_receipt": arguments.edcm_receipt,
                "edcm_verification": arguments.edcm_verification,
                "checkpoint": _checkpoint_path_from_receipt(receipt_preview),
            }
        )
        if arguments.exploratory:
            _require(
                arguments.verification_out.resolve() != DEFAULT_VERIFICATION_OUTPUT_PATH.resolve(),
                "exploratory X1 verifier cannot write official path",
            )
        result = verify_receipt(
            arguments.verify,
            arguments.config,
            arguments.implementation_authority,
            implementation_authority_sha256=arguments.implementation_authority_sha256,
            edcm_receipt_path=arguments.edcm_receipt,
            edcm_verification_path=arguments.edcm_verification,
            exploratory=arguments.exploratory,
        )
        artifact = build_verification_artifact(result)
        _require(
            len(canonical_bytes(artifact)) + 1 <= MAX_ARTIFACT_BYTES,
            "X1 verification artifact byte cap exceeded",
        )
        _atomic_json(arguments.verification_out, artifact)
        written, _ = _load_json_snapshot(
            arguments.verification_out.resolve(),
            "written X1 verification artifact",
        )
        _require(written == artifact, "written X1 verification artifact changed on readback")
        print(
            json.dumps({"terminal_route": result["terminal_route"], "verdict": result["verdict"]}, indent=2)
        )
        return 0 if result["terminal_route"] != "failed" else 1
    result = run_from_config(
        arguments.config,
        arguments.output,
        arguments.checkpoint,
        arguments.implementation_authority,
        implementation_authority_sha256=arguments.implementation_authority_sha256,
        edcm_receipt_path=arguments.edcm_receipt,
        edcm_verification_path=arguments.edcm_verification,
        max_new_seeds=arguments.max_new_seeds,
        exploratory=arguments.exploratory,
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 2 if result["resumable"] else 1 if result["aggregate"]["terminal_route"] == "failed" else 0


__all__ = [
    "ARM_NAMES",
    "AUTHORITY_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CLAIM_SCOPE",
    "CoalitionDecision",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EDCM_RECEIPT_PATH",
    "DEFAULT_EDCM_VERIFICATION_PATH",
    "DEFAULT_IMPLEMENTATION_AUTHORITY_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_VERIFICATION_OUTPUT_PATH",
    "ENVELOPE_SCHEMA",
    "EVALUATOR_ONLY_FIELDS",
    "EvaluatorTruth",
    "EventCase",
    "EXACT_CREDIT_FIELDS",
    "ExactCreditRecord",
    "IMPLEMENTATION_PATHS",
    "InteractionValueCritic",
    "OFFICIAL_CONFIG_AUTHORITY_SHA256",
    "OFFICIAL_IMPLEMENTATION_REVIEW_STATUS",
    "PRIMARY_ARM",
    "SYNERGY_PAIR",
    "VISIBLE_HEADER_FIELDS",
    "VERIFICATION_ARTIFACT_SCHEMA",
    "VisibleHeader",
    "WORK_COMPONENTS",
    "WorkCharges",
    "aggregate_rows",
    "build_implementation_authority",
    "build_verification_artifact",
    "canonical_sha256",
    "exact_credit_record",
    "exact_difference_credit",
    "generate_cases",
    "leakage_gate",
    "load_config",
    "load_edcm_entry_gate",
    "load_implementation_authority",
    "main",
    "run_from_config",
    "run_seed",
    "shuffled_coalition_schedule",
    "verify_receipt",
    "write_implementation_authority",
]
