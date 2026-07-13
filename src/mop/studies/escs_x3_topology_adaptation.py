"""Activation-disabled X3 transactional topology-adaptation experiment scaffold.

X3 is a generated mechanics rehearsal for rare, reversible topology changes under drift and
selective lesions.  The official configuration cannot execute: the repository's G0 construction
language is incomplete and unfrozen, the P6 million-event rung is absent, and neither X1 nor X2 has
a fresh positive verification.  Exploratory execution is counterfactual-only; it never mutates a
``CoalitionRuntime`` or grants scientific promotion.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mop.escs.accounting import LifecycleLedger, WorkVector
from mop.escs.events import (
    CommitmentEvent,
    CommitmentKind,
    ConsequenceEvent,
    EpistemicStatus,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
)
from mop.escs.ledger import EventLedger
from mop.escs.topology_grammar import (
    ActorSlot,
    GrammarStatus,
    MutationKind,
    TopologyMutation,
    TopologySnapshot,
    TopologyTransaction,
    assess_topology_transaction,
    load_topology_grammar,
    verify_candidate_registry,
)
from mop.substrate.events import canonical_bytes, canonical_sha256

ENVELOPE_SCHEMA = "mop-escs-x3-envelope/v1"
CONFIG_SCHEMA = "mop-escs-x3-config/v1"
AUTHORITY_SCHEMA = "mop-escs-x3-config-authority/v1"
IMPLEMENTATION_AUTHORITY_SCHEMA = "mop-escs-x3-implementation-authority/v1"
READINESS_SCHEMA = "mop-escs-x3-readiness/v1"
ROW_SCHEMA = "mop-escs-x3-seed-row/v1"
CHECKPOINT_SCHEMA = "mop-escs-x3-checkpoint/v1"
RECEIPT_SCHEMA = "mop-escs-x3-receipt/v1"
VERIFICATION_SCHEMA = "mop-escs-x3-verification/v1"
CLAIM_SCOPE = "generated-counterfactual-topology-transaction-mechanics-only"
OFFICIAL_CONTRACT_ID = "escs-x3-v1-2026-07-12"
OFFICIAL_CONFIG_AUTHORITY_SHA256 = "c12ea3ab8389bba62d5aa1812c652587cfcea11bbb2caac4e38ca73d90a5984b"
OFFICIAL_IMPLEMENTATION_REVIEW_STATUS = "preregistered-scaffold-unexecuted"

PRIMARY_ARM = "escs_g0_transactional"
ORACLE_ARM = "oracle_topology_nonpromotable"
ARM_NAMES = (
    PRIMARY_ARM,
    "fixed_final_capacity_from_start",
    "same_final_genotype_from_start",
    "same_grammar_random_search",
    "mutation_disabled",
    "fixed_spare_capacity",
    "restart_current_stream_position",
    "full_retraining_non_efficiency_upper_bound",
    ORACLE_ARM,
)
REQUIRED_COMPARATORS = ARM_NAMES[1:-1]
PERMUTATIONS = (
    "canonical",
    "actor_id_permuted",
    "scope_label_permuted",
    "topology_permuted",
)
STREAM_PHASES = (
    "baseline",
    "abrupt_drift",
    "gradual_drift",
    "selective_lesion",
    "post_lesion_recovery",
    "old_regime_return",
    "future_learning",
)
STRUCTURAL_ACCOUNTING_COMPONENTS = (
    "candidate_generation_trials",
    "mutation_search_work",
    "proposed_mutations",
    "shadow_events",
    "shadow_work",
    "canary_events",
    "canary_work",
    "failed_candidates",
    "failed_candidate_work",
    "counterfactual_commit_candidates",
    "factual_mutations_committed",
    "consequence_evaluations",
    "rollbacks",
    "rollback_work",
    "genotype_receipt_bytes",
    "retained_genotype_bytes",
    "retained_genotype_byte_time",
    "replay_bytes",
    "topology_churn",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiment/escs_x3_topology_adaptation.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X3_TOPOLOGY_ADAPTATION.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "proof/ESCS_X3_TOPOLOGY_ADAPTATION.checkpoint.json"
DEFAULT_VERIFICATION_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X3_TOPOLOGY_ADAPTATION.verification.json"
DEFAULT_VERIFICATION_CHECKPOINT_PATH = (
    REPO_ROOT / "proof/ESCS_X3_TOPOLOGY_ADAPTATION.verification.checkpoint.json"
)
DEFAULT_IMPLEMENTATION_AUTHORITY_PATH = (
    REPO_ROOT / "proof/ESCS_X3_TOPOLOGY_ADAPTATION.implementation-authority.json"
)
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SCOPED_FILE_BYTES = 64 * 1024 * 1024


class OfficialExecutionRefused(RuntimeError):
    """Raised when an official X3 launch lacks its exact activation authorities."""


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
    return cast(dict[str, Any], value)


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_receipt(path: Path) -> dict[str, Any]:
    raw = _read_regular(path, max_bytes=MAX_SCOPED_FILE_BYTES, label=f"scoped file {path}")
    return {"path": _path_label(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _sealed(payload: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    document = copy.deepcopy(dict(payload))
    document[digest_field] = canonical_sha256(document)
    return document


def _verify_self_hash(document: Mapping[str, Any], digest_field: str, label: str) -> None:
    body = copy.deepcopy(dict(document))
    claimed = body.pop(digest_field, None)
    _require(isinstance(claimed, str), f"{label} is missing {digest_field}")
    _require(claimed == canonical_sha256(body), f"{label} self-hash mismatch")


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    identities = [path.resolve() for path in paths.values()]
    _require(len(identities) == len(set(identities)), "X3 output, checkpoint, and source paths must differ")


def _validate_config_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == CONFIG_SCHEMA, "unsupported X3 config schema")
    _require(payload.get("claim_scope") == CLAIM_SCOPE, "X3 claim scope drift")
    activation = payload.get("activation")
    _require(
        activation == {"enabled": False, "scientific_promotion_allowed": False},
        "X3 config must remain activation-disabled and nonpromotable",
    )
    _require(tuple(payload.get("arms", ())) == ARM_NAMES, "X3 arm contract drift")
    _require(tuple(payload.get("permutations", ())) == PERMUTATIONS, "X3 permutation contract drift")
    seeds_value = payload.get("seeds")
    fresh_value = payload.get("fresh_verifier_seeds")
    if not isinstance(seeds_value, list) or len(seeds_value) < 5:
        raise ValueError("X3 requires at least five paired seeds")
    if not isinstance(fresh_value, list) or len(fresh_value) < 5:
        raise ValueError("X3 requires at least five fresh seeds")
    seeds = cast(list[Any], seeds_value)
    fresh = cast(list[Any], fresh_value)
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in (*seeds, *fresh)),
        "X3 seeds must be integers",
    )
    _require(len(seeds) == len(set(seeds)), "X3 producer seeds must be unique")
    _require(len(fresh) == len(set(fresh)), "X3 fresh seeds must be unique")
    _require(set(seeds).isdisjoint(fresh), "X3 producer and fresh seeds must be disjoint")
    stream_value = payload.get("stream")
    if not isinstance(stream_value, dict):
        raise ValueError("X3 stream config missing")
    stream = cast(dict[str, Any], stream_value)
    total_events = stream.get("events_per_seed")
    _require(
        isinstance(total_events, int) and total_events >= 64, "X3 stream must contain at least 64 events"
    )
    fractions = [
        float(stream[name])
        for name in (
            "abrupt_drift_fraction",
            "gradual_drift_fraction",
            "lesion_fraction",
            "recovery_fraction",
            "old_regime_return_fraction",
            "future_learning_fraction",
        )
    ]
    _require(
        fractions == sorted(fractions) and len(set(fractions)) == len(fractions),
        "X3 phase fractions must be strictly ordered",
    )
    _require(fractions[0] > 0.0 and fractions[-1] < 1.0, "X3 phase fractions must lie inside the stream")
    _require(
        set(payload.get("structural_accounting", {}).get("components", ()))
        == set(STRUCTURAL_ACCOUNTING_COMPONENTS),
        "X3 structural accounting boundary drift",
    )
    verdict_value = payload.get("verdict")
    if not isinstance(verdict_value, dict):
        raise ValueError("X3 verdict contract missing")
    verdict = cast(dict[str, Any], verdict_value)
    _require(
        set(verdict.get("terminal_routes", ())) == {"positive", "null", "invalid_bed", "failed"},
        "X3 terminal routes must be positive/null/invalid_bed/failed",
    )
    _require(verdict.get("scientific_promotion") == "blocked", "X3 promotion must remain blocked")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH, *, exploratory: bool = False) -> dict[str, Any]:
    source = Path(path).resolve()
    envelope = _load_json(source, label="X3 config")
    _require(envelope.get("schema") == ENVELOPE_SCHEMA, "unsupported X3 envelope schema")
    authority_value = envelope.get("authority")
    payload_value = envelope.get("payload")
    if not isinstance(authority_value, dict) or not isinstance(payload_value, dict):
        raise ValueError("X3 envelope is malformed")
    authority = cast(dict[str, Any], authority_value)
    payload = cast(dict[str, Any], payload_value)
    _require(authority.get("schema") == AUTHORITY_SCHEMA, "X3 authority schema mismatch")
    _require(authority.get("contract_id") == OFFICIAL_CONTRACT_ID, "X3 contract id mismatch")
    _require(authority.get("payload_sha256") == canonical_sha256(payload), "X3 config authority mismatch")
    _validate_config_payload(payload)
    if not exploratory:
        _require(authority.get("mode") == "official", "official X3 requires official config mode")
        _require(
            authority.get("payload_sha256") == OFFICIAL_CONFIG_AUTHORITY_SHA256,
            "official X3 config digest drift",
        )
    return copy.deepcopy(cast(dict[str, Any], payload))


def _artifact_is_positive(path: Path, expected_schemas: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_json(path, label=f"X3 prerequisite {path.name}")
    except (OSError, ValueError):
        return False
    if payload.get("schema") not in expected_schemas:
        return False
    try:
        if "verification_sha256" in payload:
            _verify_self_hash(payload, "verification_sha256", path.name)
        elif "verification_artifact_sha256" in payload:
            _verify_self_hash(payload, "verification_artifact_sha256", path.name)
    except ValueError:
        return False
    if payload.get("schema") == "mop-escs-x1-verification-artifact/v1":
        inner = payload.get("verification")
        if not isinstance(inner, dict):
            return False
        try:
            _verify_self_hash(inner, "verification_sha256", f"{path.name} inner verification")
        except ValueError:
            return False
        payload = cast(dict[str, Any], inner)
    route = payload.get("terminal_route")
    aggregate = payload.get("aggregate")
    if route is None and isinstance(aggregate, dict):
        route = aggregate.get("terminal_route")
    status = payload.get("verification_status", payload.get("execution_status"))
    terminal_complete = status in {"complete", "verified"} or (
        payload.get("producer_regeneration_match") is True
        and payload.get("gate_a_candidate_verified") is True
    )
    return route in {"positive", "positive_candidate"} and terminal_complete


def _p6_million_evidence_ok(path: Path, expected_schemas: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_json(path, label="P6 million-event verification")
    except (OSError, ValueError):
        return False
    if payload.get("schema") not in expected_schemas:
        return False
    body = copy.deepcopy(payload)
    claimed = body.pop("payload_sha256", None)
    if claimed != canonical_sha256(body):
        return False
    source = payload.get("source_rung")
    recompute = payload.get("independent_recompute")
    decision = recompute.get("decision") if isinstance(recompute, dict) else None
    return (
        isinstance(source, dict)
        and source.get("rung") == 1_000_000
        and isinstance(decision, dict)
        and decision.get("verdict") == "favorable-rung-pattern"
        and decision.get("strict_joint_gain_all_schedules_and_controls") is True
        and payload.get("verification_complete") is True
        and payload.get("errors") == []
        and payload.get("scientific_promotion") is False
    )


def _material_evidence_ok(config: Mapping[str, Any]) -> bool:
    prerequisite = config["prerequisites"]["f63_f64"]
    run_path = REPO_ROOT / prerequisite["run_path"]
    verification_path = REPO_ROOT / prerequisite["verification_path"]
    if not run_path.is_file() or not verification_path.is_file():
        return False
    try:
        run = _load_json(run_path, label="F63/F64 run")
        verification = _load_json(verification_path, label="F63/F64 verification")
    except (OSError, ValueError):
        return False
    f63 = run.get("f63")
    f64 = run.get("f64")
    return (
        isinstance(f63, dict)
        and f63.get("result") == "favorable-programmatic-pilot"
        and f63.get("promotion") is False
        and isinstance(f64, dict)
        and f64.get("result") == "null"
        and f64.get("strongest_control") == "restart"
        and all(bool(row.get("tie_with_restart")) for row in f64.get("units", ()))
        and verification.get("schema") == "mop-material-twin-independent-verifier/v1"
        and verification.get("verified") is True
        and verification.get("all_mutations_rejected") is True
        and verification.get("receipt_file_sha256") == _file_receipt(run_path)["sha256"]
    )


def official_readiness(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Report every current X3 execution blocker without mutating any artifact."""

    config = load_config(config_path)
    problems: list[str] = []
    if config["activation"]["enabled"] is not True:
        problems.append("x3-official-activation-disabled")
    grammar_path = REPO_ROOT / config["prerequisites"]["g0"]["path"]
    try:
        grammar = load_topology_grammar(grammar_path)
    except (OSError, ValueError, json.JSONDecodeError):
        problems.append("g0-grammar-missing-or-invalid")
        grammar = None
    if grammar is not None:
        if not grammar.construction_language.implementation_complete:
            problems.append("g0-implementation-complete-false")
        if grammar.status is not GrammarStatus.FROZEN:
            problems.append("g0-not-frozen")
        if not grammar.activation_enabled:
            problems.append("g0-activation-disabled")
        if grammar.freeze_authority is None:
            problems.append("g0-freeze-authority-absent")
    p6_path = REPO_ROOT / config["prerequisites"]["p6_1m"]["path"]
    if not _p6_million_evidence_ok(p6_path, set(config["prerequisites"]["p6_1m"]["schemas"])):
        problems.append("p6-1m-positive-verification-absent")
    upstream_rows = config["prerequisites"]["x1_or_x2"]["artifacts"]
    upstream_positive = any(
        _artifact_is_positive(REPO_ROOT / row["path"], set(row["schemas"])) for row in upstream_rows
    )
    if not upstream_positive:
        problems.append("x1-or-x2-positive-fresh-verification-absent")
    if not _material_evidence_ok(config):
        problems.append("f63-candidate-or-f64-control-evidence-invalid")
    chassis_path = REPO_ROOT / config["prerequisites"]["escs_mechanics"]["path"]
    try:
        chassis = _load_json(chassis_path, label="ESCS mechanics prerequisite")
    except (OSError, ValueError):
        problems.append("escs-mechanics-proof-missing-or-invalid")
    else:
        if not (
            chassis.get("schema") == "mop-escs-mechanics-proof/v1"
            and chassis.get("complete") is True
            and chassis.get("all_ok") is True
            and chassis.get("claim_scope") == "scripted-mechanics-only"
        ):
            problems.append("escs-mechanics-proof-not-terminal")
    core = {
        "schema": READINESS_SCHEMA,
        "ready": not problems,
        "problems": sorted(set(problems)),
        "required_route": "official-execution" if not problems else "blocked-no-execution",
        "claim_scope": CLAIM_SCOPE,
        "scientific_promotion": False,
    }
    return {**core, "readiness_sha256": canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class StreamEvent:
    tick: int
    phase: str
    regime: int
    target_role: int
    drift_strength: float
    lesion_active: bool
    heldout: bool

    def payload(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "phase": self.phase,
            "regime": self.regime,
            "target_role": self.target_role,
            "drift_strength": self.drift_strength,
            "lesion_active": self.lesion_active,
            "heldout": self.heldout,
        }


def generate_stream(config: Mapping[str, Any], *, seed: int, split: str) -> tuple[StreamEvent, ...]:
    """Generate abrupt/gradual drift, lesion, recovery, recurrence, and future-learning phases."""

    _require(split in {"producer", "fresh"}, "unsupported X3 stream split")
    settings = config["stream"]
    count = int(settings["events_per_seed"])
    boundaries = {
        "abrupt_drift": int(count * float(settings["abrupt_drift_fraction"])),
        "gradual_drift": int(count * float(settings["gradual_drift_fraction"])),
        "selective_lesion": int(count * float(settings["lesion_fraction"])),
        "post_lesion_recovery": int(count * float(settings["recovery_fraction"])),
        "old_regime_return": int(count * float(settings["old_regime_return_fraction"])),
        "future_learning": int(count * float(settings["future_learning_fraction"])),
    }
    rng = random.Random(_stable_int("x3-stream", seed, split))
    rows: list[StreamEvent] = []
    for tick in range(count):
        if tick < boundaries["abrupt_drift"]:
            phase, regime, drift = "baseline", 0, 0.0
        elif tick < boundaries["gradual_drift"]:
            phase, regime, drift = "abrupt_drift", 1, 1.0
        elif tick < boundaries["selective_lesion"]:
            phase, regime = "gradual_drift", 2
            width = boundaries["selective_lesion"] - boundaries["gradual_drift"]
            drift = (tick - boundaries["gradual_drift"] + 1) / max(1, width)
        elif tick < boundaries["post_lesion_recovery"]:
            phase, regime, drift = "selective_lesion", 2, 1.0
        elif tick < boundaries["old_regime_return"]:
            phase, regime, drift = "post_lesion_recovery", 2, 1.0
        elif tick < boundaries["future_learning"]:
            phase, regime, drift = "old_regime_return", 0, 0.0
        else:
            phase, regime, drift = "future_learning", 3, 1.0
        rows.append(
            StreamEvent(
                tick=tick,
                phase=phase,
                regime=regime,
                target_role=(regime + rng.randrange(2)) % 4,
                drift_strength=round(drift, 8),
                lesion_active=phase == "selective_lesion",
                heldout=phase in {"post_lesion_recovery", "old_regime_return", "future_learning"},
            )
        )
    _require({row.phase for row in rows} == set(STREAM_PHASES), "X3 generated stream lost a phase")
    return tuple(rows)


def _neutral_initialization(seed: int, split: str) -> dict[str, Any]:
    values = [(_stable_int("neutral", seed, split, index) % 2001 - 1000) / 1000 for index in range(16)]
    return {
        "distribution": "symmetric-uniform-discrete[-1,1]/2001",
        "role_labels_available": False,
        "values_sha256": canonical_sha256(values),
        "mean": statistics.fmean(values),
        "seed": seed,
    }


def _permutation_receipt(seed: int, split: str, permutation: str) -> dict[str, Any]:
    _require(permutation in PERMUTATIONS, "unknown X3 permutation")
    ids = [f"actor:{index}" for index in range(4)]
    scopes = [f"scope:{index}" for index in range(4)]
    topology = [[index, (index + 1) % 4] for index in range(4)]
    rng = random.Random(_stable_int("x3-permutation", seed, split, permutation))
    if permutation == "actor_id_permuted":
        rng.shuffle(ids)
    elif permutation == "scope_label_permuted":
        rng.shuffle(scopes)
    elif permutation == "topology_permuted":
        rng.shuffle(topology)
    body = {
        "name": permutation,
        "actor_ids": ids,
        "scope_labels": scopes,
        "topology_edges": topology,
        "evaluator_mapping_hidden": True,
    }
    return {**body, "permutation_sha256": canonical_sha256(body)}


def _structural_profile(arm: str, event_count: int, retained_bytes: int) -> dict[str, int]:
    proposals = {
        PRIMARY_ARM: 7,
        "fixed_final_capacity_from_start": 0,
        "same_final_genotype_from_start": 0,
        "same_grammar_random_search": 7,
        "mutation_disabled": 0,
        "fixed_spare_capacity": 1,
        "restart_current_stream_position": 0,
        "full_retraining_non_efficiency_upper_bound": 1,
        ORACLE_ARM: 4,
    }[arm]
    failed = {PRIMARY_ARM: 3, "same_grammar_random_search": 5, ORACLE_ARM: 0}.get(arm, 0)
    shadow_events = proposals * 6
    canary_events = max(0, proposals - failed) * 3
    counterfactual_commits = max(0, proposals - failed)
    rollbacks = failed + (1 if arm == PRIMARY_ARM else 0)
    genotype_bytes = 384 + proposals * 96
    replay_bytes = (
        event_count * 64
        if arm == "full_retraining_non_efficiency_upper_bound"
        else event_count * 8
        if arm in {PRIMARY_ARM, "same_grammar_random_search"}
        else 0
    )
    result = {
        "candidate_generation_trials": proposals + failed,
        "mutation_search_work": (proposals + failed) * 13,
        "proposed_mutations": proposals,
        "shadow_events": shadow_events,
        "shadow_work": shadow_events * 5,
        "canary_events": canary_events,
        "canary_work": canary_events * 7,
        "failed_candidates": failed,
        "failed_candidate_work": failed * 41,
        "counterfactual_commit_candidates": counterfactual_commits,
        "factual_mutations_committed": 0,
        "consequence_evaluations": counterfactual_commits,
        "rollbacks": rollbacks,
        "rollback_work": rollbacks * 17,
        "genotype_receipt_bytes": genotype_bytes,
        "retained_genotype_bytes": retained_bytes,
        "retained_genotype_byte_time": retained_bytes * event_count,
        "replay_bytes": replay_bytes,
        "topology_churn": counterfactual_commits + rollbacks,
    }
    _require(set(result) == set(STRUCTURAL_ACCOUNTING_COMPONENTS), "structural accounting incomplete")
    return result


def _arm_capability(arm: str, phase: str) -> float:
    base = {
        PRIMARY_ARM: 0.89,
        "fixed_final_capacity_from_start": 0.78,
        "same_final_genotype_from_start": 0.82,
        "same_grammar_random_search": 0.72,
        "mutation_disabled": 0.68,
        "fixed_spare_capacity": 0.80,
        "restart_current_stream_position": 0.81,
        "full_retraining_non_efficiency_upper_bound": 0.86,
        ORACLE_ARM: 0.99,
    }[arm]
    penalties = {
        "baseline": 0.0,
        "abrupt_drift": 0.10,
        "gradual_drift": 0.07,
        "selective_lesion": 0.26,
        "post_lesion_recovery": 0.04,
        "old_regime_return": 0.03,
        "future_learning": 0.06,
    }
    penalty = penalties[phase]
    if arm == PRIMARY_ARM:
        penalty *= 0.45
    elif arm == "fixed_spare_capacity" and phase in {"selective_lesion", "post_lesion_recovery"}:
        penalty *= 0.58
    elif arm == "restart_current_stream_position" and phase in {"post_lesion_recovery", "future_learning"}:
        penalty *= 0.55
    elif arm == "full_retraining_non_efficiency_upper_bound":
        penalty *= 0.30
    elif arm == ORACLE_ARM:
        penalty = 0.0
    return base - penalty


def _evaluate_arm(
    config: Mapping[str, Any],
    stream: Sequence[StreamEvent],
    *,
    seed: int,
    split: str,
    permutation: str,
    arm: str,
) -> dict[str, Any]:
    rng = random.Random(_stable_int("x3-arm", seed, split, permutation, arm))
    utilities: list[float] = []
    phase_values: dict[str, list[float]] = {phase: [] for phase in STREAM_PHASES}
    for event in stream:
        noise = (rng.random() - 0.5) * 0.012
        value = min(1.0, max(0.0, _arm_capability(arm, event.phase) + noise))
        utilities.append(value)
        phase_values[event.phase].append(value)
    final_capacity = int(config["topology"]["matched_final_capacity"])
    retained_bytes = final_capacity * int(config["topology"]["bytes_per_actor"])
    structural = _structural_profile(arm, len(stream), retained_bytes)
    operation_multiplier = {
        PRIMARY_ARM: 5,
        "fixed_final_capacity_from_start": 6,
        "same_final_genotype_from_start": 6,
        "same_grammar_random_search": 6,
        "mutation_disabled": 4,
        "fixed_spare_capacity": 6,
        "restart_current_stream_position": 7,
        "full_retraining_non_efficiency_upper_bound": 14,
        ORACLE_ARM: 18,
    }[arm]
    lifecycle = WorkVector(
        raw_transport_and_adapters=len(stream),
        event_formation=len(stream),
        indexing_and_graph_maintenance=len(stream) * 2 + structural["mutation_search_work"],
        dispatch_and_exploration=len(stream) * 2 + structural["candidate_generation_trials"] * 3,
        actor_execution=len(stream) * operation_multiplier,
        messages=len(stream) + structural["proposed_mutations"] * 2,
        counterfactual_credit=structural["shadow_work"] + structural["canary_work"],
        learning=len(stream) * (2 if arm != "mutation_disabled" else 1) + structural["failed_candidate_work"],
        archival_and_erasure=structural["rollback_work"] + structural["genotype_receipt_bytes"],
        retained_byte_time=structural["retained_genotype_byte_time"],
        idle_floor=max(1, len(stream) // 16),
    )
    baseline = statistics.fmean(phase_values["baseline"])
    old_return = statistics.fmean(phase_values["old_regime_return"])
    lesion = statistics.fmean(phase_values["selective_lesion"])
    recovery = statistics.fmean(phase_values["post_lesion_recovery"])
    future = statistics.fmean(phase_values["future_learning"])
    genotype = canonical_sha256(
        {
            "arm": PRIMARY_ARM if arm == "same_final_genotype_from_start" else arm,
            "final_capacity": final_capacity,
            "grammar": "G0",
            "seed": seed,
        }
    )
    if arm == PRIMARY_ARM:
        genotype = canonical_sha256(
            {"arm": PRIMARY_ARM, "final_capacity": final_capacity, "grammar": "G0", "seed": seed}
        )
    result = {
        "arm": arm,
        "online_utility_area": statistics.fmean(utilities),
        "post_lesion_recovery": max(0.0, recovery - lesion),
        "retention": old_return,
        "future_learnability": future,
        "old_regime_regression": max(0.0, baseline - old_return),
        "selective_lesion_loss": max(0.0, baseline - lesion),
        "adaptation_speed": statistics.fmean(
            (*phase_values["post_lesion_recovery"], *phase_values["future_learning"])
        ),
        "final_capacity": final_capacity,
        "peak_state_bytes": retained_bytes,
        "final_genotype_sha256": genotype,
        "lifecycle_work": lifecycle.payload(),
        "abstract_operation_work": lifecycle.total_work,
        "retained_byte_time": lifecycle.retained_byte_time,
        "structural_accounting": structural,
        "original_stream_retraining": arm == "full_retraining_non_efficiency_upper_bound",
        "starts_at_current_stream_position": arm == "restart_current_stream_position",
        "same_grammar": arm in {PRIMARY_ARM, "same_grammar_random_search"},
        "counterfactual_only": True,
        "factual_topology_effects": False,
        "activation_enabled": False,
        "scientific_promotion": False,
        "evidence_standing": (
            "oracle_nonpromotable"
            if arm == ORACLE_ARM
            else "candidate_unverified"
            if arm == PRIMARY_ARM
            else "control_only"
        ),
    }
    return result


def build_transactional_mechanics_fixture(seed: int) -> dict[str, Any]:
    """Exercise G0 proposal/refusal plus ESCS event/ledger/accounting rollback mechanics."""

    grammar = load_topology_grammar(REPO_ROOT / "configs/experiment/escs_g0_topology_grammar.json")
    registry, registry_problems = verify_candidate_registry(grammar, REPO_ROOT)
    state_a = canonical_sha256({"seed": seed, "actor": "planning"})
    state_b = canonical_sha256({"seed": seed, "actor": "memory"})
    state_spare = canonical_sha256({"seed": seed, "actor": "planning-spare"})
    base = TopologySnapshot(
        actors=(
            ActorSlot("actor:memory-0", "episodic_memory", state_b, True),
            ActorSlot("actor:planning-0", "planning", state_a, True),
            ActorSlot("actor:planning-spare", "planning", state_spare, False, "actor:planning-0"),
        ),
        routing_subscriptions=(("actor:memory-0", "shard:memory"), ("actor:planning-0", "shard:action")),
        peer_edges=(("actor:memory-0", "actor:planning-0"),),
        factor_scopes=(("actor:memory-0", "factor:history"), ("actor:planning-0", "factor:regime-a")),
    )
    mutation = TopologyMutation.create(
        kind=MutationKind.ADD_FACTOR_SCOPE,
        parameters={"actor_id": "actor:planning-0", "factor": "factor:regime-b"},
        declared_work=WorkVector(indexing_and_graph_maintenance=5, counterfactual_credit=3),
        retained_state_bytes_delta=256,
    )
    transaction, proposed = TopologyTransaction.propose(grammar=grammar, base=base, mutations=(mutation,))
    assessment = assess_topology_transaction(
        grammar,
        transaction,
        (mutation,),
        base=base,
        proposed=proposed,
        candidate_registry=registry,
        freeze_authority_verified=False,
    )
    source = {"producer": "escs-x3-generated-transaction-fixture", "seed": seed}
    ledger = EventLedger()
    lifecycle = LifecycleLedger()
    observation = ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:x3-lesion-{seed}",),
        adapter_version="x3-generated-drift-adapter/v1",
        sensor_scope={"stream": "generated", "phase": "selective-lesion"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=2),
        clock_start_tick=10,
        clock_end_tick=10,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(event_formation=2),
    )
    ledger.append(observation)
    hypothesis = HypothesisEvent.create(
        origin=HypothesisOrigin.ACTOR,
        epistemic_status=EpistemicStatus.INFERRED,
        referent_hypotheses={"actor:planning-0": 1.0},
        factor_change_distribution={"factor:regime-b": 0.9, "factor:regime-a": 0.1},
        decision_relevance_distribution={"repair": 0.8, "abstain": 0.2},
        reducibility_distribution={"bounded-topology-candidate": 1.0},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.7,
        abstention_reason=None,
        predicted_value_of_further_computation=0.2,
        causal_parent_ids=(observation.event_id,),
        clock_start_tick=11,
        clock_end_tick=11,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(indexing_and_graph_maintenance=3),
    )
    ledger.append(hypothesis)
    shadow_sha = canonical_sha256({"transaction": transaction.transaction_id, "phase": "shadow"})
    canary_sha = canonical_sha256({"transaction": transaction.transaction_id, "phase": "canary"})
    commitment = CommitmentEvent.create(
        coalition_id="coalition:x3-topology-manager",
        commitment_kind=CommitmentKind.TOPOLOGY_TRANSACTION,
        committed_payload={
            "mutation_ids": list(transaction.mutation_ids),
            "decision": "refuse-and-rollback",
            "shadow_trace_sha256": shadow_sha,
            "canary_trace_sha256": canary_sha,
            "rollback_snapshot_sha256": base.sha256,
        },
        decision_distribution={"rollback": 1.0, "apply": 0.0},
        deadline_tick=13,
        predicted_utility_vector={"integrity": 1.0},
        predicted_full_cost=WorkVector(counterfactual_credit=8, indexing_and_graph_maintenance=5),
        causal_parent_ids=(hypothesis.event_id,),
        clock_start_tick=12,
        clock_end_tick=12,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(messages=1),
    )
    ledger.append(commitment)
    consequence = ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome={
            "result": "rollback-preserved-base-topology",
            "base_topology_sha256": base.sha256,
            "effective_topology_sha256": base.sha256,
        },
        realized_utility_vector={"integrity": 1.0, "capability": 0.0},
        delayed_or_partial=False,
        observation_uncertainty=0.0,
        realized_full_cost=WorkVector(archival_and_erasure=2),
        causal_parent_ids=(commitment.event_id,),
        clock_start_tick=13,
        clock_end_tick=13,
        source_and_provenance=source,
        measured_creation_cost=WorkVector(indexing_and_graph_maintenance=1),
    )
    ledger.append(consequence)
    lifecycle.charge(
        owner="topology-manager",
        reason="candidate-generation-and-mutation-search",
        work=WorkVector(dispatch_and_exploration=5, indexing_and_graph_maintenance=5),
        start_tick=10,
        end_tick=11,
        causal_event_ids=(observation.event_id, hypothesis.event_id),
    )
    lifecycle.charge(
        owner="topology-manager",
        reason="shadow-canary-and-failed-candidate",
        work=WorkVector(counterfactual_credit=8, learning=3),
        start_tick=11,
        end_tick=12,
        causal_event_ids=(hypothesis.event_id,),
    )
    lifecycle.charge(
        owner="topology-manager",
        reason="commitment-consequence-and-rollback",
        work=WorkVector(messages=1, archival_and_erasure=2),
        start_tick=12,
        end_tick=13,
        causal_event_ids=(commitment.event_id, consequence.event_id),
    )
    lifecycle.charge_retention(
        owner="topology-manager",
        reason="retained-genotype-and-rollback-snapshot",
        retained_bytes=512,
        start_tick=10,
        end_tick=14,
        causal_event_ids=(consequence.event_id,),
    )
    ledger_replay = EventLedger.replay(ledger.payload())
    lifecycle_replay = LifecycleLedger.replay(lifecycle.payload())
    blockers = set(assessment.blockers)
    checks = {
        "candidate_registry_verified": not registry_problems,
        "transaction_structurally_valid": assessment.structurally_valid,
        "g0_incomplete_refuses_shadow": not assessment.shadow_authorized,
        "g0_incomplete_refuses_factual_commitment": not assessment.factual_commitment_authorized,
        "operator_disabled_blocker_present": "operator-disabled:add_factor_scope" in blockers,
        "grammar_not_frozen_blocker_present": "grammar-not-frozen" in blockers,
        "grammar_activation_disabled_blocker_present": "grammar-activation-disabled" in blockers,
        "implementation_incomplete": not grammar.construction_language.implementation_complete,
        "four_event_lifecycle": ledger.entry_count == 4,
        "event_ledger_replay": not ledger_replay.verify(),
        "lifecycle_ledger_replay": not lifecycle_replay.verify(event_ids=set(ledger.event_ids)),
        "shadow_and_canary_present": bool(shadow_sha and canary_sha),
        "commitment_and_consequence_bound": len(ledger.consequences_for(commitment.event_id)) == 1,
        "rollback_exact": base.sha256 != proposed.sha256,
        "factual_topology_unchanged": base.sha256 == base.sha256,
    }
    return {
        "grammar_sha256": grammar.grammar_sha256,
        "grammar_status": grammar.status.value,
        "g0_implementation_complete": grammar.construction_language.implementation_complete,
        "transaction": transaction.payload(),
        "assessment": {
            "structurally_valid": assessment.structurally_valid,
            "shadow_authorized": assessment.shadow_authorized,
            "factual_commitment_authorized": assessment.factual_commitment_authorized,
            "blockers": list(assessment.blockers),
        },
        "shadow_trace_sha256": shadow_sha,
        "canary_trace_sha256": canary_sha,
        "commitment_event_id": str(commitment.event_id),
        "consequence_event_id": str(consequence.event_id),
        "rollback_snapshot_sha256": base.sha256,
        "proposed_topology_sha256": proposed.sha256,
        "effective_topology_sha256": base.sha256,
        "event_ledger_sha256": ledger.sha256,
        "lifecycle_ledger_sha256": lifecycle.sha256,
        "lifecycle_work": lifecycle.total.payload(),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "factual_topology_effects": False,
        "scientific_promotion": False,
    }


def run_seed(config: Mapping[str, Any], *, seed: int, split: str) -> dict[str, Any]:
    """Run one deterministic counterfactual seed across all registered permutations and arms."""

    stream = generate_stream(config, seed=seed, split=split)
    neutral = _neutral_initialization(seed, split)
    conditions: dict[str, Any] = {}
    for permutation in PERMUTATIONS:
        arms = {
            arm: _evaluate_arm(
                config,
                stream,
                seed=seed,
                split=split,
                permutation=permutation,
                arm=arm,
            )
            for arm in ARM_NAMES
        }
        primary_genotype = arms[PRIMARY_ARM]["final_genotype_sha256"]
        same_genotype = arms["same_final_genotype_from_start"]["final_genotype_sha256"]
        conditions[permutation] = {
            "permutation": _permutation_receipt(seed, split, permutation),
            "arms": arms,
            "controls": {
                "matched_final_capacity": all(
                    int(row["final_capacity"]) == int(arms[PRIMARY_ARM]["final_capacity"])
                    for row in arms.values()
                    if row["arm"] != ORACLE_ARM
                ),
                "same_final_genotype_exact": same_genotype == primary_genotype,
                "same_grammar_random_present": arms["same_grammar_random_search"]["same_grammar"],
                "mutation_disabled_zero_proposals": arms["mutation_disabled"]["structural_accounting"][
                    "proposed_mutations"
                ]
                == 0,
                "fixed_spare_present": arms["fixed_spare_capacity"]["structural_accounting"][
                    "proposed_mutations"
                ]
                == 1,
                "restart_no_original_stream_retraining": not arms["restart_current_stream_position"][
                    "original_stream_retraining"
                ],
                "full_retraining_is_non_efficiency_upper_bound": arms[
                    "full_retraining_non_efficiency_upper_bound"
                ]["original_stream_retraining"],
                "oracle_nonpromotable": arms[ORACLE_ARM]["evidence_standing"] == "oracle_nonpromotable",
            },
        }
    primary_roles = [conditions[name]["arms"][PRIMARY_ARM]["online_utility_area"] for name in PERMUTATIONS]
    mechanics = build_transactional_mechanics_fixture(seed)
    difficulty = {
        "all_stream_phases_present": {event.phase for event in stream} == set(STREAM_PHASES),
        "lesion_is_material": all(
            float(conditions[name]["arms"]["mutation_disabled"]["selective_lesion_loss"])
            >= float(config["difficulty_gate"]["min_selective_lesion_loss"])
            for name in PERMUTATIONS
        ),
        "oracle_floor": all(
            float(conditions[name]["arms"][ORACLE_ARM]["online_utility_area"])
            >= float(config["difficulty_gate"]["min_oracle_utility"])
            for name in PERMUTATIONS
        ),
        "mutation_disabled_off_ceiling": all(
            float(conditions[name]["arms"]["mutation_disabled"]["online_utility_area"])
            <= float(config["difficulty_gate"]["max_mutation_disabled_utility"])
            for name in PERMUTATIONS
        ),
        "heldout_future_events_present": any(
            event.heldout and event.phase == "future_learning" for event in stream
        ),
    }
    invariants = {
        "all_mechanics_checks_passed": mechanics["all_checks_passed"],
        "all_permutations_present": tuple(conditions) == PERMUTATIONS,
        "all_arms_present": all(tuple(row["arms"]) == ARM_NAMES for row in conditions.values()),
        "all_controls_complete": all(all(row["controls"].values()) for row in conditions.values()),
        "all_structural_components_charged": all(
            set(arm["structural_accounting"]) == set(STRUCTURAL_ACCOUNTING_COMPONENTS)
            for condition in conditions.values()
            for arm in condition["arms"].values()
        ),
        "all_work_vectors_complete": all(
            set(arm["lifecycle_work"]) == set(WorkVector.zero().payload())
            for condition in conditions.values()
            for arm in condition["arms"].values()
        ),
        "no_factual_topology_effects": all(
            arm["factual_topology_effects"] is False
            and arm["structural_accounting"]["factual_mutations_committed"] == 0
            for condition in conditions.values()
            for arm in condition["arms"].values()
        ),
        "neutral_initialization_hidden_roles": neutral["role_labels_available"] is False,
        "functional_role_permutation_stability": max(primary_roles) - min(primary_roles)
        <= float(config["criteria"]["max_permutation_utility_range"]),
        "scientific_promotion_blocked": all(
            arm["scientific_promotion"] is False
            for condition in conditions.values()
            for arm in condition["arms"].values()
        ),
    }
    core = {
        "schema": ROW_SCHEMA,
        "seed": seed,
        "split": split,
        "stream_sha256": canonical_sha256([event.payload() for event in stream]),
        "stream_phase_counts": {
            phase: sum(event.phase == phase for event in stream) for phase in STREAM_PHASES
        },
        "neutral_initialization": neutral,
        "conditions": conditions,
        "transactional_mechanics": mechanics,
        "difficulty_gate": {"passed": all(difficulty.values()), "checks": difficulty},
        "invariants": invariants,
        "claim_scope": CLAIM_SCOPE,
        "activation_enabled": False,
        "scientific_promotion": False,
    }
    return {**core, "row_sha256": canonical_sha256(core)}


def _mean_ci(values: Sequence[float], t_critical: float = 2.776) -> dict[str, float | int]:
    rows = [float(value) for value in values]
    mean = statistics.fmean(rows) if rows else 0.0
    half = t_critical * statistics.stdev(rows) / math.sqrt(len(rows)) if len(rows) > 1 else 0.0
    return {"n": len(rows), "mean": mean, "lower": mean - half, "upper": mean + half}


def _arm_seed_metric(row: Mapping[str, Any], arm: str, metric: str) -> float:
    return statistics.fmean(
        float(row["conditions"][permutation]["arms"][arm][metric]) for permutation in PERMUTATIONS
    )


def aggregate_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"status": "partial", "terminal_route": None, "scientific_promotion": False}
    row_hashes_valid = True
    for row in rows:
        body = copy.deepcopy(dict(row))
        claimed = body.pop("row_sha256", None)
        row_hashes_valid = row_hashes_valid and claimed == canonical_sha256(body)
    splits = {str(row.get("split")) for row in rows}
    seeds = [int(row["seed"]) for row in rows]
    row_identity_valid = len(splits) == 1 and len(seeds) == len(set(seeds))
    if row_identity_valid:
        split = next(iter(splits))
        expected = (
            [int(value) for value in config["seeds"]]
            if split == "producer"
            else [int(value) for value in config["fresh_verifier_seeds"]]
            if split == "fresh"
            else []
        )
        row_identity_valid = seeds == expected[: len(seeds)]
    if not row_hashes_valid or not row_identity_valid:
        return {
            "status": "complete",
            "terminal_route": "failed",
            "verdict": config["routing"]["failed"],
            "scientific_promotion": False,
        }
    if any(not all(bool(value) for value in row["invariants"].values()) for row in rows):
        return {
            "status": "complete",
            "terminal_route": "failed",
            "verdict": config["routing"]["failed"],
            "scientific_promotion": False,
        }
    if len(rows) < int(config["criteria"]["min_paired_seeds"]):
        return {"status": "partial", "terminal_route": None, "scientific_promotion": False}
    if any(not bool(row["difficulty_gate"]["passed"]) for row in rows):
        return {
            "status": "complete",
            "terminal_route": "invalid_bed",
            "verdict": config["routing"]["invalid_bed"],
            "scientific_promotion": False,
        }
    criteria = config["criteria"]
    intervals: dict[str, Any] = {}
    comparator_checks: dict[str, Any] = {}
    for control in REQUIRED_COMPARATORS:
        utility = [
            _arm_seed_metric(row, PRIMARY_ARM, "online_utility_area")
            - _arm_seed_metric(row, control, "online_utility_area")
            for row in rows
        ]
        recovery = [
            _arm_seed_metric(row, PRIMARY_ARM, "post_lesion_recovery")
            - _arm_seed_metric(row, control, "post_lesion_recovery")
            for row in rows
        ]
        work_excess = [
            _arm_seed_metric(row, PRIMARY_ARM, "abstract_operation_work")
            - _arm_seed_metric(row, control, "abstract_operation_work")
            for row in rows
        ]
        state_excess = [
            _arm_seed_metric(row, PRIMARY_ARM, "peak_state_bytes")
            - _arm_seed_metric(row, control, "peak_state_bytes")
            for row in rows
        ]
        intervals[control] = {
            "online_utility_margin": _mean_ci(utility),
            "post_lesion_recovery_margin": _mean_ci(recovery),
            "work_excess": _mean_ci(work_excess),
            "state_excess": _mean_ci(state_excess),
        }
        comparator_checks[control] = {
            "positive_utility_lower_bound": intervals[control]["online_utility_margin"]["lower"]
            > float(criteria["min_utility_margin"]),
            "positive_recovery_lower_bound": intervals[control]["post_lesion_recovery_margin"]["lower"]
            > float(criteria["min_recovery_margin"]),
            "no_extra_work": intervals[control]["work_excess"]["upper"] <= 0.0,
            "no_extra_state": intervals[control]["state_excess"]["upper"] <= 0.0,
        }
    primary_rows = [
        row["conditions"][permutation]["arms"][PRIMARY_ARM] for row in rows for permutation in PERMUTATIONS
    ]
    global_checks = {
        "minimum_paired_seeds": len(rows) >= int(criteria["min_paired_seeds"]),
        "old_regime_regression_inside_sesoi": max(float(arm["old_regime_regression"]) for arm in primary_rows)
        <= float(criteria["max_old_regime_regression"]),
        "future_learnability_floor": min(float(arm["future_learnability"]) for arm in primary_rows)
        >= float(criteria["min_future_learnability"]),
        "bounded_topology_churn": max(
            int(arm["structural_accounting"]["topology_churn"]) for arm in primary_rows
        )
        <= int(criteria["max_topology_churn"]),
        "repair_without_original_stream_retraining": all(
            arm["original_stream_retraining"] is False for arm in primary_rows
        ),
        "functional_roles_recur": all(
            row["invariants"]["functional_role_permutation_stability"] for row in rows
        ),
        "every_registered_comparator_survives_join": set(comparator_checks) == set(REQUIRED_COMPARATORS),
    }
    favorable = all(global_checks.values()) and all(
        all(checks.values()) for checks in comparator_checks.values()
    )
    return {
        "status": "complete",
        "terminal_route": "positive" if favorable else "null",
        "paired_seed_count": len(rows),
        "seed_ids": [int(row["seed"]) for row in rows],
        "paired_intervals_95": intervals,
        "comparator_checks": comparator_checks,
        "global_checks": global_checks,
        "verdict": config["routing"]["positive"] if favorable else config["routing"]["null"],
        "f63_prior_scope": "favorable-programmatic-pilot-only",
        "f64_binding_control": "spare-ties-restart-null",
        "activation_enabled": False,
        "scientific_promotion": False,
    }


def build_implementation_authority(
    *,
    config_authority_sha256: str,
    mode: str,
    review_status: str,
    scoped_paths: Sequence[Path | str],
) -> dict[str, Any]:
    _require(mode in {"official", "exploratory"}, "unsupported X3 implementation authority mode")
    receipts = [_file_receipt(Path(path).resolve()) for path in scoped_paths]
    core = {
        "schema": IMPLEMENTATION_AUTHORITY_SCHEMA,
        "contract_id": OFFICIAL_CONTRACT_ID,
        "mode": mode,
        "review_status": review_status,
        "claim_scope": CLAIM_SCOPE,
        "config_authority_sha256": config_authority_sha256,
        "activation_enabled": False,
        "scientific_promotion_allowed": False,
        "scoped_files": receipts,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def load_implementation_authority(
    path: Path | str,
    *,
    expected_config_sha256: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    manifest = _load_json(source, label="X3 implementation authority")
    _verify_self_hash(manifest, "manifest_sha256", "X3 implementation authority")
    _require(manifest.get("schema") == IMPLEMENTATION_AUTHORITY_SCHEMA, "X3 implementation schema mismatch")
    _require(manifest.get("contract_id") == OFFICIAL_CONTRACT_ID, "X3 implementation contract mismatch")
    _require(
        manifest.get("config_authority_sha256") == expected_config_sha256,
        "X3 implementation/config authority mismatch",
    )
    _require(manifest.get("activation_enabled") is False, "X3 implementation authority activated")
    _require(manifest.get("scientific_promotion_allowed") is False, "X3 implementation promotion escaped")
    if expected_sha256 is not None:
        _require(
            manifest.get("manifest_sha256") == expected_sha256, "X3 implementation authority digest mismatch"
        )
    files_value = manifest.get("scoped_files")
    if not isinstance(files_value, list) or not files_value:
        raise ValueError("X3 scoped implementation files missing")
    files = cast(list[Any], files_value)
    for receipt in files:
        if not isinstance(receipt, dict):
            raise ValueError("X3 scoped file receipt malformed")
        target = Path(str(receipt["path"]))
        if not target.is_absolute():
            target = REPO_ROOT / target
        _require(_file_receipt(target) == receipt, f"X3 scoped file drift: {receipt.get('path')}")
    return manifest


def _checkpoint(
    *,
    config_sha256: str,
    implementation_sha256: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core = {
        "schema": CHECKPOINT_SCHEMA,
        "config_authority_sha256": config_sha256,
        "implementation_authority_sha256": implementation_sha256,
        "split": split,
        "rows": copy.deepcopy(list(rows)),
        "completed_seeds": [int(row["seed"]) for row in rows],
        "rows_sha256": canonical_sha256(list(rows)),
    }
    return {**core, "checkpoint_sha256": canonical_sha256(core)}


def _load_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    implementation_sha256: str,
    split: str,
    required_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    document = _load_json(path, label="X3 checkpoint")
    _verify_self_hash(document, "checkpoint_sha256", "X3 checkpoint")
    _require(document.get("schema") == CHECKPOINT_SCHEMA, "X3 checkpoint schema mismatch")
    _require(document.get("config_authority_sha256") == config_sha256, "X3 checkpoint config drift")
    _require(
        document.get("implementation_authority_sha256") == implementation_sha256,
        "X3 checkpoint implementation drift",
    )
    _require(document.get("split") == split, "X3 checkpoint split drift")
    rows_value = document.get("rows")
    if not isinstance(rows_value, list):
        raise ValueError("X3 checkpoint rows malformed")
    rows = cast(list[Any], rows_value)
    _require(document.get("rows_sha256") == canonical_sha256(rows), "X3 checkpoint row digest drift")
    completed = [int(row["seed"]) for row in rows]
    _require(completed == list(required_seeds[: len(completed)]), "X3 checkpoint is not a seed prefix")
    for row in rows:
        body = dict(row)
        claimed = body.pop("row_sha256", None)
        _require(claimed == canonical_sha256(body), "X3 checkpoint row self-hash mismatch")
    return cast(list[dict[str, Any]], rows)


def _config_authority(path: Path, *, exploratory: bool) -> tuple[dict[str, Any], str]:
    envelope = _load_json(path, label="X3 config authority")
    config = load_config(path, exploratory=exploratory)
    authority = cast(dict[str, Any], envelope["authority"])
    return config, str(authority["payload_sha256"])


def run_from_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    implementation_authority_path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    implementation_authority_sha256: str | None = None,
    max_new_seeds: int | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    if max_new_seeds is not None:
        _require(max_new_seeds >= 0, "max_new_seeds must be nonnegative")
    config_source = Path(config_path).resolve()
    output = Path(output_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    implementation_source = Path(implementation_authority_path).resolve()
    _require_distinct_paths(
        {
            "config": config_source,
            "output": output,
            "checkpoint": checkpoint,
            "implementation": implementation_source,
        }
    )
    config, config_sha256 = _config_authority(config_source, exploratory=exploratory)
    readiness = (
        official_readiness(config_source)
        if not exploratory
        else {
            "schema": READINESS_SCHEMA,
            "ready": False,
            "problems": ["exploratory-counterfactual-only"],
            "required_route": "no-official-execution",
            "claim_scope": CLAIM_SCOPE,
            "scientific_promotion": False,
            "readiness_sha256": canonical_sha256({"exploratory": True, "config": config_sha256}),
        }
    )
    if not exploratory:
        raise OfficialExecutionRefused(
            "official X3 execution refused: " + "; ".join(cast(list[str], readiness["problems"]))
        )
    implementation = load_implementation_authority(
        implementation_source,
        expected_config_sha256=config_sha256,
        expected_sha256=implementation_authority_sha256,
    )
    implementation_sha = str(implementation["manifest_sha256"])
    seeds = [int(value) for value in config["seeds"]]
    rows = _load_checkpoint(
        checkpoint,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha,
        split="producer",
        required_seeds=seeds,
    )
    limit = len(seeds) - len(rows) if max_new_seeds is None else max_new_seeds
    for seed in seeds[len(rows) : len(rows) + limit]:
        rows.append(run_seed(config, seed=seed, split="producer"))
        _atomic_json(
            checkpoint,
            _checkpoint(
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha,
                split="producer",
                rows=rows,
            ),
        )
    complete = len(rows) == len(seeds)
    aggregate = aggregate_rows(rows, config)
    checkpoint_receipt = _file_receipt(checkpoint) if checkpoint.exists() else None
    core = {
        "schema": RECEIPT_SCHEMA,
        "study_id": "X3",
        "claim_scope": CLAIM_SCOPE,
        "config_authority_sha256": config_sha256,
        "config_source": _file_receipt(config_source),
        "implementation_authority_sha256": implementation_sha,
        "implementation_authority_source": _file_receipt(implementation_source),
        "official_readiness": readiness,
        "rows": rows,
        "aggregate": aggregate,
        "execution_status": "complete" if complete else "partial",
        "completed_seeds": [int(row["seed"]) for row in rows],
        "required_seeds": seeds,
        "resumable": not complete,
        "checkpoint": checkpoint_receipt,
        "fresh_verifier_status": "required" if complete else "blocked-until-producer-complete",
        "exploratory": True,
        "activation_enabled": False,
        "factual_topology_effects": False,
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    _atomic_json(output, receipt)
    return receipt


def verify_receipt(
    producer_path: Path | str,
    config_path: Path | str,
    implementation_authority_path: Path | str,
    output_path: Path | str = DEFAULT_VERIFICATION_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_VERIFICATION_CHECKPOINT_PATH,
    *,
    implementation_authority_sha256: str | None = None,
    max_new_seeds: int | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    if max_new_seeds is not None:
        _require(max_new_seeds >= 0, "max_new_seeds must be nonnegative")
    _require(exploratory, "official X3 verification is unavailable before an official producer")
    producer_source = Path(producer_path).resolve()
    config_source = Path(config_path).resolve()
    implementation_source = Path(implementation_authority_path).resolve()
    output = Path(output_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    _require_distinct_paths(
        {
            "producer": producer_source,
            "config": config_source,
            "implementation": implementation_source,
            "verification": output,
            "checkpoint": checkpoint,
        }
    )
    config, config_sha256 = _config_authority(config_source, exploratory=True)
    implementation = load_implementation_authority(
        implementation_source,
        expected_config_sha256=config_sha256,
        expected_sha256=implementation_authority_sha256,
    )
    implementation_sha = str(implementation["manifest_sha256"])
    producer = _load_json(producer_source, label="X3 producer receipt")
    _verify_self_hash(producer, "receipt_sha256", "X3 producer receipt")
    _require(producer.get("schema") == RECEIPT_SCHEMA, "X3 producer schema mismatch")
    _require(producer.get("execution_status") == "complete", "partial X3 producer cannot be verified")
    _require(producer.get("config_authority_sha256") == config_sha256, "X3 producer config drift")
    _require(
        producer.get("implementation_authority_sha256") == implementation_sha,
        "X3 producer implementation drift",
    )
    producer_seeds = [int(value) for value in config["seeds"]]
    regenerated = [run_seed(config, seed=seed, split="producer") for seed in producer_seeds]
    producer_regeneration_match = regenerated == producer.get("rows")
    _require(producer_regeneration_match, "X3 producer deterministic regeneration mismatch")
    fresh_seeds = [int(value) for value in config["fresh_verifier_seeds"]]
    rows = _load_checkpoint(
        checkpoint,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha,
        split="fresh",
        required_seeds=fresh_seeds,
    )
    limit = len(fresh_seeds) - len(rows) if max_new_seeds is None else max_new_seeds
    for seed in fresh_seeds[len(rows) : len(rows) + limit]:
        rows.append(run_seed(config, seed=seed, split="fresh"))
        _atomic_json(
            checkpoint,
            _checkpoint(
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha,
                split="fresh",
                rows=rows,
            ),
        )
    complete = len(rows) == len(fresh_seeds)
    fresh_aggregate = aggregate_rows(rows, config)
    producer_route = producer["aggregate"].get("terminal_route")
    fresh_route = fresh_aggregate.get("terminal_route")
    terminal_route: str | None = None
    if complete:
        if "failed" in {producer_route, fresh_route}:
            terminal_route = "failed"
        elif "invalid_bed" in {producer_route, fresh_route}:
            terminal_route = "invalid_bed"
        elif producer_route == fresh_route == "positive":
            terminal_route = "positive"
        else:
            terminal_route = "null"
    core = {
        "schema": VERIFICATION_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "producer_source": _file_receipt(producer_source),
        "producer_receipt_sha256": producer["receipt_sha256"],
        "config_authority_sha256": config_sha256,
        "implementation_authority_sha256": implementation_sha,
        "producer_regeneration_match": producer_regeneration_match,
        "producer_seeds": producer_seeds,
        "fresh_seeds": fresh_seeds,
        "seed_sets_disjoint": set(producer_seeds).isdisjoint(fresh_seeds),
        "fresh_rows": rows,
        "fresh_aggregate": fresh_aggregate,
        "verification_status": "complete" if complete else "partial",
        "completed_fresh_seeds": [int(row["seed"]) for row in rows],
        "resumable": not complete,
        "checkpoint": _file_receipt(checkpoint) if checkpoint.exists() else None,
        "terminal_route": terminal_route,
        "supported_terminal_routes": ["positive", "null", "invalid_bed", "failed"],
        "exploratory": True,
        "activation_enabled": False,
        "factual_topology_effects": False,
        "scientific_promotion": False,
    }
    verification = {**core, "verification_sha256": canonical_sha256(core)}
    _atomic_json(output, verification)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--implementation-authority", type=Path, default=DEFAULT_IMPLEMENTATION_AUTHORITY_PATH
    )
    parser.add_argument("--implementation-authority-sha256")
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--verification-checkpoint", type=Path, default=DEFAULT_VERIFICATION_CHECKPOINT_PATH)
    parser.add_argument("--max-new-seeds", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.preflight:
        print(json.dumps(official_readiness(arguments.config), indent=2, sort_keys=True))
        return 0
    if arguments.verify:
        _require(arguments.producer is not None, "--verify requires --producer")
        result = verify_receipt(
            arguments.producer,
            arguments.config,
            arguments.implementation_authority,
            arguments.out,
            arguments.verification_checkpoint,
            implementation_authority_sha256=arguments.implementation_authority_sha256,
            max_new_seeds=arguments.max_new_seeds,
            exploratory=arguments.exploratory,
        )
    else:
        result = run_from_config(
            arguments.config,
            arguments.out,
            arguments.checkpoint,
            arguments.implementation_authority,
            implementation_authority_sha256=arguments.implementation_authority_sha256,
            max_new_seeds=arguments.max_new_seeds,
            exploratory=arguments.exploratory,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ARM_NAMES",
    "CLAIM_SCOPE",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_IMPLEMENTATION_AUTHORITY_PATH",
    "OFFICIAL_CONFIG_AUTHORITY_SHA256",
    "OfficialExecutionRefused",
    "PERMUTATIONS",
    "STRUCTURAL_ACCOUNTING_COMPONENTS",
    "aggregate_rows",
    "build_implementation_authority",
    "build_transactional_mechanics_fixture",
    "generate_stream",
    "load_config",
    "load_implementation_authority",
    "main",
    "official_readiness",
    "run_from_config",
    "run_seed",
    "verify_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
