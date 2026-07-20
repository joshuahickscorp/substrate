
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from mop.config import REPO_ROOT
from mop.studies import edcm1_event_triggered_coalition as edcm
from mop.substrate.events import EventRef, canonical_bytes, canonical_sha256

from .accounting import FACTUAL_BRANCH, LifecycleCharge, LifecycleLedger, WorkVector
from .actors import ActionIntent
from .events import (
    CommitmentEvent,
    CommitmentKind,
    ConsequenceEvent,
    EpistemicStatus,
    EvidenceClass,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
)
from .messages import ClaimMessage, ClaimSchema

ADAPTER_SCHEMA = "mop-escs-edcm-adapter/v1"
ACCOUNTING_SCHEMA = "mop-escs-edcm-adapter-accounting/v1"
AUTHORITY_SCHEMA = "mop-escs-edcm-verified-authority/v1"
TRANSLATION_SCHEMA = "mop-escs-edcm-translation/v1"
ASSESSMENT_SCHEMA = "mop-escs-edcm-activation-assessment/v1"
ADAPTER_ID = "edcm1-v3-to-escs-v1"
EDCM_RECEIPT_SCHEMA = "mop-edcm1-receipt/v3"
EDCM_VERIFICATION_ARTIFACT_SCHEMA = "mop-edcm1-verification-artifact/v1"
EDCM_PROPOSAL_SCHEMA = "mop-edcm1-proposal/v3"
EDCM_VERIFICATION_SCHEMA = "mop-edcm1-verification/v3"
ADAPTER_ACTIVATION_ENABLED = False
SCIENTIFIC_PROMOTION_ALLOWED = False
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TARGET_BYTES = 4 * 1024 * 1024
MAX_EDCM_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_EDCM_VERIFICATION_BYTES = 2 * 1024 * 1024
MAX_INITIAL_SPECIALISTS = len(edcm.PROPOSER_ORDER)
MAX_REASONING_ROUNDS = 2
FACTOR_SCOPE = (
    "edcm:local-blocked",
    "edcm:novelty-channels",
    "edcm:previous-action",
    "edcm:previous-reward",
    "edcm:relative-goal",
)
ROUTING_SHARDS = ("edcm:structured-observation",)

DEFAULT_WORK_WEIGHTS = (
    ("bytes_hashed", 1),
    ("bytes_serialized", 1),
    ("comparisons", 1),
    ("nonlinearities", 4),
    ("scalar_ops", 1),
    ("table_reads", 3),
    ("table_writes", 4),
)

PROPOSAL_CLAIM_SCHEMA = ClaimSchema(
    schema_id="mop.edcm1.proposal",
    version=3,
    claim_types=frozenset({"action_proposal"}),
    payload_forms=frozenset({"edcm1-canonical-json"}),
    epistemic_statuses=frozenset({EpistemicStatus.INFERRED}),
    max_payload_bytes=256 * 1024,
)
VERIFICATION_CLAIM_SCHEMA = ClaimSchema(
    schema_id="mop.edcm1.verification",
    version=3,
    claim_types=frozenset({"proposal_verification"}),
    payload_forms=frozenset({"edcm1-canonical-json"}),
    epistemic_statuses=frozenset({EpistemicStatus.INFERRED}),
    max_payload_bytes=256 * 1024,
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset(
    {
        "action_rotation",
        "canonical_referent_truth",
        "evaluator",
        "evaluator_label",
        "future_consequence",
        "future_outcome",
        "ground_truth",
        "ground_truth_label",
        "hidden_change",
        "hidden_change_point",
        "hidden_state_digest",
        "irreducible_noise",
        "niche_label",
        "noise_label",
        "oracle_coalition",
        "oracle_label",
        "physical_action",
    }
)
_FORBIDDEN_TEXT = (
    "action_rotation",
    "evaluator_label",
    "future_outcome",
    "ground_truth",
    "hidden_change",
    "niche_label",
    "noise_label",
    "oracle_label",
    "physical_action",
)
_WORK_BUCKETS = frozenset(
    {
        "raw_transport_and_adapters",
        "event_formation",
        "indexing_and_graph_maintenance",
        "dispatch_and_exploration",
        "actor_execution",
        "messages",
        "counterfactual_credit",
        "learning",
        "archival_and_erasure",
        "idle_floor",
    }
)
_AUTHORITY_VALIDATION_TOKEN = object()


class AdapterContractError(ValueError):
    pass


class AdapterActivationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterContractError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{label} must be a mapping")
    actual = set(value)
    forbidden = sorted(actual & _FORBIDDEN_KEYS)
    if forbidden:
        raise AdapterContractError(f"{label} contains future/evaluator-only fields: {forbidden}")
    if actual != expected:
        raise AdapterContractError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key) for key in value}
        for nested in value.values():
            result.update(_walk_keys(nested))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for nested in value:
            result.update(_walk_keys(nested))
        return result
    return set()


def _reject_forbidden(value: Any, label: str) -> None:
    keys = sorted(_walk_keys(value) & _FORBIDDEN_KEYS)
    if keys:
        raise AdapterContractError(f"{label} contains future/evaluator-only fields: {keys}")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise AdapterContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdapterContractError(f"{label} must be a nonnegative integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise AdapterContractError(f"{label} must be finite")
    return float(value)


def _canonical_limited(value: Any, limit: int, label: str) -> bytes:
    try:
        payload = canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise AdapterContractError(f"{label} is not strict canonical JSON") from exc
    if len(payload) > limit:
        raise AdapterContractError(f"{label} exceeds its byte cap")
    return payload


def _source_work_payload(work: edcm.AbstractWork) -> dict[str, int]:
    if type(work) is not edcm.AbstractWork:
        raise AdapterContractError("source work must be an exact EDCM AbstractWork")
    payload = dataclasses.asdict(work)
    expected = {field.name for field in fields(edcm.AbstractWork)}
    _require(set(payload) == expected, "EDCM abstract-work schema drift")
    return {name: _nonnegative_int(value, f"source work {name}") for name, value in payload.items()}


def _add_source_work(left: edcm.AbstractWork, right: edcm.AbstractWork) -> edcm.AbstractWork:
    result = edcm.AbstractWork(**_source_work_payload(left))
    return result.add(edcm.AbstractWork(**_source_work_payload(right)))


def _work_vector_with(bucket_values: Mapping[str, int]) -> WorkVector:
    unknown = set(bucket_values) - _WORK_BUCKETS
    if unknown:
        raise AdapterContractError(f"unknown ESCS work buckets: {sorted(unknown)}")
    values = {field.name: 0 for field in fields(WorkVector)}
    for name, value in bucket_values.items():
        values[name] = _nonnegative_int(value, f"work bucket {name}")
    return WorkVector(**values)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or path.is_symlink():
        raise AdapterContractError(f"authority path must be a regular non-symlink file: {path}")
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise AdapterContractError(f"authority file changed while reading: {resolved}")
    return {
        "path": _display_path(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_json(path: Path, *, max_bytes: int, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file() or path.is_symlink():
        raise AdapterContractError(f"{label} must be a regular non-symlink file")
    before = resolved.stat()
    size = before.st_size
    if size > max_bytes:
        raise AdapterContractError(f"{label} exceeds its byte cap")
    raw = resolved.read_bytes()
    after = resolved.stat()
    if len(raw) != size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise AdapterContractError(f"{label} changed while reading")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, nested in pairs:
            if key in result:
                raise AdapterContractError(f"{label} contains duplicate JSON field {key!r}")
            result[key] = nested
        return result

    def reject_nonfinite_constant(constant: str) -> None:
        raise AdapterContractError(f"{label} contains nonfinite JSON constant {constant!r}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise AdapterContractError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise AdapterContractError(f"{label} must be an object")
    return value, {
        "path": _display_path(resolved),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    adapter_id: str = ADAPTER_ID
    source_observation_schema: str = "mop-edcm1-visible-observation/v3"
    source_proposal_schema: str = EDCM_PROPOSAL_SCHEMA
    source_verification_schema: str = EDCM_VERIFICATION_SCHEMA
    max_initial_specialists: int = MAX_INITIAL_SPECIALISTS
    max_reasoning_rounds: int = MAX_REASONING_ROUNDS
    max_source_bytes: int = MAX_SOURCE_BYTES
    max_target_bytes: int = MAX_TARGET_BYTES
    work_weights: tuple[tuple[str, int], ...] = DEFAULT_WORK_WEIGHTS
    activation_enabled: bool = ADAPTER_ACTIVATION_ENABLED
    scientific_promotion_allowed: bool = SCIENTIFIC_PROMOTION_ALLOWED

    def __post_init__(self) -> None:
        _require(self.adapter_id == ADAPTER_ID, "adapter identity drift")
        _require(
            self.source_observation_schema == "mop-edcm1-visible-observation/v3",
            "observation schema drift",
        )
        _require(
            self.source_proposal_schema == EDCM_PROPOSAL_SCHEMA
            and edcm.PROPOSAL_SCHEMA == EDCM_PROPOSAL_SCHEMA,
            "proposal schema drift",
        )
        _require(
            self.source_verification_schema == EDCM_VERIFICATION_SCHEMA
            and edcm.VERIFICATION_SCHEMA == EDCM_VERIFICATION_SCHEMA,
            "verification schema drift",
        )
        _require(self.max_initial_specialists == len(edcm.PROPOSER_ORDER), "specialist cap drift")
        _require(self.max_reasoning_rounds == 2, "EDCM adapter requires exactly two bounded rounds")
        _require(
            self.max_source_bytes == MAX_SOURCE_BYTES and self.max_target_bytes == MAX_TARGET_BYTES,
            "adapter byte cap drift",
        )
        _require(self.work_weights == DEFAULT_WORK_WEIGHTS, "EDCM work-weight authority drift")
        _require(self.activation_enabled is False, "EDCM adapter activation must remain disabled")
        _require(
            self.scientific_promotion_allowed is False,
            "EDCM adapter cannot grant scientific promotion",
        )

    @property
    def weights(self) -> dict[str, int]:
        return dict(self.work_weights)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "adapter_id": self.adapter_id,
            "source_observation_schema": self.source_observation_schema,
            "source_proposal_schema": self.source_proposal_schema,
            "source_verification_schema": self.source_verification_schema,
            "max_initial_specialists": self.max_initial_specialists,
            "max_reasoning_rounds": self.max_reasoning_rounds,
            "max_source_bytes": self.max_source_bytes,
            "max_target_bytes": self.max_target_bytes,
            "work_weights": self.weights,
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class TranslationAccounting:
    stage: str
    source_work_components: tuple[tuple[str, int], ...]
    source_work_weights: tuple[tuple[str, int], ...]
    source_work_units: int
    source_work_buckets: tuple[tuple[str, int], ...]
    adapter_work_bucket: str
    source_payload_bytes: int
    target_payload_bytes: int
    adapter_validation_operations: int
    adapter_bytes_serialized: int
    adapter_bytes_hashed: int
    adapter_operations: int
    work: WorkVector
    accounting_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.stage.strip()), "accounting stage is empty")
        components = dict(self.source_work_components)
        weights = dict(self.source_work_weights)
        _require(len(components) == len(self.source_work_components), "duplicate source work component")
        _require(len(weights) == len(self.source_work_weights), "duplicate source work weight")
        _require(set(components) == set(weights), "source work/weight component mismatch")
        for name, value in (*self.source_work_components, *self.source_work_weights):
            _nonnegative_int(value, name)
        expected_source = sum(components[name] * weights[name] for name in components)
        _require(self.source_work_units == expected_source, "source work total mismatch")
        source_buckets = dict(self.source_work_buckets)
        _require(
            len(source_buckets) == len(self.source_work_buckets) and set(source_buckets) <= _WORK_BUCKETS,
            "source work buckets are duplicated or unknown",
        )
        _require(
            sum(_nonnegative_int(value, f"source bucket {name}") for name, value in source_buckets.items())
            == self.source_work_units,
            "source work bucket total mismatch",
        )
        _require(self.adapter_work_bucket in _WORK_BUCKETS, "adapter work bucket is unknown")
        for label, value in (
            ("source_payload_bytes", self.source_payload_bytes),
            ("target_payload_bytes", self.target_payload_bytes),
            ("adapter_validation_operations", self.adapter_validation_operations),
            ("adapter_bytes_serialized", self.adapter_bytes_serialized),
            ("adapter_bytes_hashed", self.adapter_bytes_hashed),
            ("adapter_operations", self.adapter_operations),
        ):
            _nonnegative_int(value, label)
        _require(
            self.adapter_bytes_serialized == self.source_payload_bytes + self.target_payload_bytes,
            "adapter serialization bytes are incomplete",
        )
        _require(
            self.adapter_bytes_hashed == self.source_payload_bytes + self.target_payload_bytes,
            "adapter hash bytes are incomplete",
        )
        _require(
            self.adapter_operations
            == self.adapter_validation_operations + self.adapter_bytes_serialized + self.adapter_bytes_hashed,
            "adapter operation total mismatch",
        )
        expected_buckets = dict(source_buckets)
        expected_buckets[self.adapter_work_bucket] = (
            expected_buckets.get(self.adapter_work_bucket, 0) + self.adapter_operations
        )
        _require(self.work == _work_vector_with(expected_buckets), "ESCS/source/adapter work join mismatch")
        _digest(self.accounting_sha256, "accounting_sha256")
        _require(
            self.accounting_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "translation accounting self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        source_work: edcm.AbstractWork,
        weights: Mapping[str, int],
        source_bucket: str | None,
        adapter_bucket: str,
        source_bucket_totals: Mapping[str, int] | None = None,
        source_payload: Any,
        target_payload: Any,
        validation_operations: int,
        source_limit: int = MAX_SOURCE_BYTES,
        target_limit: int = MAX_TARGET_BYTES,
    ) -> Self:
        components = _source_work_payload(source_work)
        normalized_weights = {
            name: _nonnegative_int(weights.get(name), f"weight {name}") for name in components
        }
        _require(set(weights) == set(components), "source work weights have incompatible schema")
        source_bytes = _canonical_limited(source_payload, source_limit, "adapter source payload")
        target_bytes = _canonical_limited(target_payload, target_limit, "adapter target payload")
        source_total = sum(components[name] * normalized_weights[name] for name in components)
        serialized = len(source_bytes) + len(target_bytes)
        hashed = serialized
        adapter_ops = _nonnegative_int(validation_operations, "validation_operations") + serialized + hashed
        if source_bucket_totals is None:
            _require(source_bucket in _WORK_BUCKETS, "source work bucket is unknown")
            source_buckets = {str(source_bucket): source_total}
        else:
            _require(source_bucket is None, "use either one source bucket or explicit source buckets")
            source_buckets = {
                name: _nonnegative_int(value, f"source bucket {name}")
                for name, value in source_bucket_totals.items()
            }
            _require(set(source_buckets) <= _WORK_BUCKETS, "source work bucket is unknown")
            _require(sum(source_buckets.values()) == source_total, "source work bucket total mismatch")
        _require(adapter_bucket in _WORK_BUCKETS, "adapter work bucket is unknown")
        buckets = dict(source_buckets)
        buckets[adapter_bucket] = buckets.get(adapter_bucket, 0) + adapter_ops
        work = _work_vector_with(buckets)
        core = {
            "schema": ACCOUNTING_SCHEMA,
            "stage": stage,
            "source_work_components": dict(sorted(components.items())),
            "source_work_weights": dict(sorted(normalized_weights.items())),
            "source_work_units": source_total,
            "source_work_buckets": dict(sorted(source_buckets.items())),
            "adapter_work_bucket": adapter_bucket,
            "source_payload_bytes": len(source_bytes),
            "target_payload_bytes": len(target_bytes),
            "adapter_validation_operations": validation_operations,
            "adapter_bytes_serialized": serialized,
            "adapter_bytes_hashed": hashed,
            "adapter_operations": adapter_ops,
            "work": work.payload(),
        }
        return cls(
            stage=stage,
            source_work_components=tuple(sorted(components.items())),
            source_work_weights=tuple(sorted(normalized_weights.items())),
            source_work_units=source_total,
            source_work_buckets=tuple(sorted(source_buckets.items())),
            adapter_work_bucket=adapter_bucket,
            source_payload_bytes=len(source_bytes),
            target_payload_bytes=len(target_bytes),
            adapter_validation_operations=validation_operations,
            adapter_bytes_serialized=serialized,
            adapter_bytes_hashed=hashed,
            adapter_operations=adapter_ops,
            work=work,
            accounting_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": ACCOUNTING_SCHEMA,
            "stage": self.stage,
            "source_work_components": dict(self.source_work_components),
            "source_work_weights": dict(self.source_work_weights),
            "source_work_units": self.source_work_units,
            "source_work_buckets": dict(self.source_work_buckets),
            "adapter_work_bucket": self.adapter_work_bucket,
            "source_payload_bytes": self.source_payload_bytes,
            "target_payload_bytes": self.target_payload_bytes,
            "adapter_validation_operations": self.adapter_validation_operations,
            "adapter_bytes_serialized": self.adapter_bytes_serialized,
            "adapter_bytes_hashed": self.adapter_bytes_hashed,
            "adapter_operations": self.adapter_operations,
            "work": self.work.payload(),
        }
        if include_digest:
            result["accounting_sha256"] = self.accounting_sha256
        return result

    def charge(
        self,
        ledger: LifecycleLedger,
        *,
        start_tick: int,
        end_tick: int,
        causal_event_ids: Sequence[EventRef],
    ) -> LifecycleCharge:
        return ledger.charge(
            owner=f"adapter:{ADAPTER_ID}",
            reason=f"edcm-to-escs:{self.stage}",
            work=self.work,
            start_tick=start_tick,
            end_tick=end_tick,
            branch_id=FACTUAL_BRANCH,
            causal_event_ids=tuple(sorted(causal_event_ids, key=str)),
        )


@dataclass(frozen=True, slots=True)
class VerifiedEDCMAuthority:
    producer_file: tuple[tuple[str, Any], ...]
    verification_file: tuple[tuple[str, Any], ...]
    producer_receipt_sha256: str
    verification_artifact_sha256: str
    config_authority_sha256: str
    implementation_authority_sha256: str
    complementarity_gate_sha256: str
    complementarity_passed: bool
    terminal_execution_status: str
    verifier_mode: str
    authority_sha256: str
    _validation_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for label, items in (
            ("producer_file", self.producer_file),
            ("verification_file", self.verification_file),
        ):
            receipt = dict(items)
            _exact_keys(receipt, {"path", "bytes", "sha256"}, label)
            _require(isinstance(receipt["path"], str) and bool(receipt["path"]), f"{label} path")
            _nonnegative_int(receipt["bytes"], f"{label} bytes")
            _digest(receipt["sha256"], f"{label} sha256")
        for label, value in (
            ("producer_receipt_sha256", self.producer_receipt_sha256),
            ("verification_artifact_sha256", self.verification_artifact_sha256),
            ("config_authority_sha256", self.config_authority_sha256),
            ("implementation_authority_sha256", self.implementation_authority_sha256),
            ("complementarity_gate_sha256", self.complementarity_gate_sha256),
            ("authority_sha256", self.authority_sha256),
        ):
            _digest(value, label)
        _require(self.terminal_execution_status in {"complete", "terminal_scientific_stop"}, "nonterminal")
        _require(self.verifier_mode == edcm.OFFICIAL_VERIFIER_MODE, "nonofficial verifier mode")
        _require(isinstance(self.complementarity_passed, bool), "gate pass state must be boolean")
        _require(
            self._validation_token is _AUTHORITY_VALIDATION_TOKEN,
            "verified EDCM authority must come from the current-authority loader",
        )
        _require(
            self.authority_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "verified EDCM authority self-hash mismatch",
        )

    @classmethod
    def _create(
        cls,
        *,
        producer_file: Mapping[str, Any],
        verification_file: Mapping[str, Any],
        producer_receipt_sha256: str,
        verification_artifact_sha256: str,
        config_authority_sha256: str,
        implementation_authority_sha256: str,
        complementarity_gate_sha256: str,
        complementarity_passed: bool,
        terminal_execution_status: str,
        verifier_mode: str,
    ) -> Self:
        core = {
            "schema": AUTHORITY_SCHEMA,
            "producer_file": dict(producer_file),
            "verification_file": dict(verification_file),
            "producer_receipt_sha256": producer_receipt_sha256,
            "verification_artifact_sha256": verification_artifact_sha256,
            "config_authority_sha256": config_authority_sha256,
            "implementation_authority_sha256": implementation_authority_sha256,
            "complementarity_gate_sha256": complementarity_gate_sha256,
            "complementarity_passed": complementarity_passed,
            "terminal_execution_status": terminal_execution_status,
            "verifier_mode": verifier_mode,
        }
        return cls(
            producer_file=tuple(sorted(producer_file.items())),
            verification_file=tuple(sorted(verification_file.items())),
            producer_receipt_sha256=producer_receipt_sha256,
            verification_artifact_sha256=verification_artifact_sha256,
            config_authority_sha256=config_authority_sha256,
            implementation_authority_sha256=implementation_authority_sha256,
            complementarity_gate_sha256=complementarity_gate_sha256,
            complementarity_passed=complementarity_passed,
            terminal_execution_status=terminal_execution_status,
            verifier_mode=verifier_mode,
            authority_sha256=canonical_sha256(core),
            _validation_token=_AUTHORITY_VALIDATION_TOKEN,
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": AUTHORITY_SCHEMA,
            "producer_file": dict(self.producer_file),
            "verification_file": dict(self.verification_file),
            "producer_receipt_sha256": self.producer_receipt_sha256,
            "verification_artifact_sha256": self.verification_artifact_sha256,
            "config_authority_sha256": self.config_authority_sha256,
            "implementation_authority_sha256": self.implementation_authority_sha256,
            "complementarity_gate_sha256": self.complementarity_gate_sha256,
            "complementarity_passed": self.complementarity_passed,
            "terminal_execution_status": self.terminal_execution_status,
            "verifier_mode": self.verifier_mode,
        }
        if include_digest:
            result["authority_sha256"] = self.authority_sha256
        return result


def load_verified_edcm_authority(
    producer_path: Path | str = edcm.DEFAULT_OUTPUT_PATH,
    verification_path: Path | str = edcm.DEFAULT_VERIFICATION_OUTPUT_PATH,
    *,
    config_path: Path | str = edcm.DEFAULT_CONFIG_PATH,
    implementation_authority_path: Path | str = edcm.DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
) -> VerifiedEDCMAuthority:

    producer_source = Path(producer_path)
    verification_source = Path(verification_path)
    config_source = Path(config_path)
    implementation_source = Path(implementation_authority_path)
    paths = {
        producer_source.resolve(),
        verification_source.resolve(),
        config_source.resolve(),
        implementation_source.resolve(),
    }
    _require(len(paths) == 4, "EDCM authority paths must be distinct")
    config_file = _file_receipt(config_source)
    try:
        config = edcm.load_config(config_source)
    except (OSError, ValueError) as exc:
        raise AdapterContractError(f"current EDCM config authority is invalid: {exc}") from exc
    _require(config_file == _file_receipt(config_source), "EDCM config changed during validation")
    config_authority = edcm.canonical_sha256(config)
    _require(
        config.get("abstract_work", {}).get("weights") == dict(DEFAULT_WORK_WEIGHTS),
        "EDCM current abstract-work authority is incompatible with the adapter",
    )
    producer, producer_file = _read_json(
        producer_source,
        max_bytes=MAX_EDCM_RECEIPT_BYTES,
        label="EDCM producer receipt",
    )
    _exact_keys(producer, set(edcm.RECEIPT_KEYS), "EDCM producer receipt")
    _require(
        producer.get("schema") == EDCM_RECEIPT_SCHEMA and edcm.RECEIPT_SCHEMA == EDCM_RECEIPT_SCHEMA,
        "EDCM producer schema mismatch",
    )
    producer_without_receipt = dict(producer)
    producer_receipt_sha = producer_without_receipt.pop("receipt_sha256", None)
    _require(
        producer_receipt_sha == edcm.canonical_sha256(producer_without_receipt),
        "EDCM producer self-hash mismatch",
    )
    deterministic_sha = producer_without_receipt.pop("deterministic_core_sha256", None)
    _require(
        deterministic_sha == edcm.canonical_sha256(producer_without_receipt),
        "EDCM producer deterministic-core mismatch",
    )
    _require(
        producer.get("execution_status") in {"complete", "terminal_scientific_stop"}
        and producer.get("all_ok") is True
        and producer.get("problems") == []
        and producer.get("resumable") is False,
        "EDCM producer is not terminal and valid",
    )
    _require(producer.get("exploratory") is False, "exploratory EDCM receipt cannot authorize adapter")
    _require(
        producer.get("study_id") == "edcm1-event-triggered-heterogeneous-coalition-crossover-v3"
        and producer.get("claim_scope") == edcm.CLAIM_SCOPE,
        "EDCM producer study/claim scope mismatch",
    )
    _require(producer.get("scientific_promotion") is False, "EDCM scientific promotion escaped")
    _require(producer.get("verifier_mode") == edcm.OFFICIAL_VERIFIER_MODE, "EDCM verifier mode drift")
    _require(producer.get("authority_sha256") == config_authority, "EDCM current config authority drift")
    _require(
        producer.get("config_source") == config_file,
        "EDCM producer/current config file mismatch",
    )
    declared_implementation_authority = _digest(
        producer.get("implementation_authority_sha256"),
        "EDCM producer implementation authority",
    )
    implementation_file = _file_receipt(implementation_source)
    try:
        implementation = edcm.load_implementation_authority(
            implementation_source,
            config,
            expected_sha256=declared_implementation_authority,
            exploratory=False,
        )
    except (OSError, ValueError) as exc:
        raise AdapterContractError(f"current EDCM implementation authority is invalid: {exc}") from exc
    _require(
        implementation_file == _file_receipt(implementation_source),
        "EDCM implementation authority changed during validation",
    )
    implementation_authority = str(implementation["manifest_sha256"])
    _require(
        producer.get("implementation_authority_sha256") == implementation_authority,
        "EDCM current implementation authority drift",
    )
    _require(producer.get("implementation") == implementation["files"], "EDCM implementation files drift")
    current_runtime = edcm._runtime_identity()
    _require(producer.get("runtime_identity") == current_runtime, "EDCM current runtime identity drift")
    _require(
        producer.get("implementation_sha256")
        == edcm.canonical_sha256(
            {
                "implementation_authority_sha256": implementation_authority,
                "runtime": current_runtime,
            }
        ),
        "EDCM producer implementation/runtime join mismatch",
    )
    expected_implementation_receipt = {
        "source": implementation_file,
        "mode": implementation["mode"],
        "review_status": implementation["review_status"],
        "manifest_sha256": implementation_authority,
    }
    _require(
        producer.get("implementation_authority") == expected_implementation_receipt,
        "EDCM implementation authority source drift",
    )
    gate = producer.get("gate")
    if not (
        isinstance(gate, dict) and gate.get("status") == "complete" and isinstance(gate.get("passed"), bool)
    ):
        raise AdapterContractError("EDCM complementarity gate is not complete")

    verification, verification_file = _read_json(
        verification_source,
        max_bytes=MAX_EDCM_VERIFICATION_BYTES,
        label="EDCM verification artifact",
    )
    _exact_keys(
        verification,
        {
            "schema",
            "study_id",
            "claim_scope",
            "verification",
            "scientific_promotion",
            "verification_artifact_sha256",
        },
        "EDCM verification artifact",
    )
    _require(
        verification.get("schema") == EDCM_VERIFICATION_ARTIFACT_SCHEMA
        and edcm.VERIFICATION_ARTIFACT_SCHEMA == EDCM_VERIFICATION_ARTIFACT_SCHEMA,
        "EDCM verification artifact schema mismatch",
    )
    _require(
        verification.get("study_id") == "edcm1-event-triggered-heterogeneous-coalition-crossover-v3"
        and verification.get("claim_scope") == edcm.CLAIM_SCOPE,
        "EDCM verification study/claim scope mismatch",
    )
    verification_core = dict(verification)
    verification_sha = verification_core.pop("verification_artifact_sha256", None)
    _require(
        verification_sha == edcm.canonical_sha256(verification_core),
        "EDCM verification artifact self-hash mismatch",
    )
    _require(verification.get("scientific_promotion") is False, "verification promotion escaped")
    result = verification.get("verification")
    if not isinstance(result, dict):
        raise AdapterContractError("EDCM verification result is missing")
    _exact_keys(
        result,
        {
            "valid",
            "gate_seed_ids",
            "heldout_seed_ids",
            "verdict",
            "execution_status",
            "verifier_mode",
            "regeneration",
            "authority_sha256",
            "implementation_authority_sha256",
            "verified_sources",
            "scientific_promotion",
        },
        "EDCM verification result",
    )
    _require(
        result.get("valid") is True
        and result.get("execution_status") == producer.get("execution_status")
        and result.get("verifier_mode") == edcm.OFFICIAL_VERIFIER_MODE
        and result.get("scientific_promotion") is False,
        "EDCM verification result is not terminal full regeneration",
    )
    _require(
        isinstance(producer.get("aggregate"), dict)
        and result.get("verdict") == producer["aggregate"].get("verdict"),
        "EDCM producer/verifier verdict mismatch",
    )
    _require(result.get("authority_sha256") == config_authority, "verification config authority drift")
    _require(
        result.get("implementation_authority_sha256") == implementation_authority,
        "verification implementation authority drift",
    )
    regeneration = result.get("regeneration")
    if not (isinstance(regeneration, dict) and regeneration.get("mode") == edcm.OFFICIAL_VERIFIER_MODE):
        raise AdapterContractError("EDCM full regeneration evidence is missing")
    _require(
        regeneration.get("regenerated_gate_seeds") == result.get("gate_seed_ids")
        and regeneration.get("regenerated_heldout_seeds") == result.get("heldout_seed_ids")
        and result.get("gate_seed_ids") == producer.get("completed_gate_seeds")
        and result.get("heldout_seed_ids") == producer.get("completed_heldout_seeds"),
        "EDCM regenerated seed authority mismatch",
    )
    sources = result.get("verified_sources")
    if not isinstance(sources, dict):
        raise AdapterContractError("EDCM verified sources are missing")
    _exact_keys(
        sources,
        {
            "receipt",
            "receipt_path",
            "checkpoint",
            "checkpoint_path",
            "config",
            "implementation_authority",
        },
        "EDCM verified sources",
    )
    _require(sources["receipt"] == producer_file, "verifier is not bound to producer bytes")
    _require(
        Path(sources["receipt_path"]).resolve() == producer_source.resolve(),
        "verified producer path drift",
    )
    _require(sources["config"] == config_file, "verified current config drift")
    _require(
        sources["implementation_authority"] == implementation_file,
        "verified current implementation authority drift",
    )
    checkpoint_source = Path(sources["checkpoint_path"])
    _require(sources["checkpoint"] == _file_receipt(checkpoint_source), "verified checkpoint drift")
    _require(
        Path(producer["resume"]["checkpoint_path"]).resolve() == checkpoint_source.resolve(),
        "producer/verifier checkpoint path drift",
    )
    return VerifiedEDCMAuthority._create(
        producer_file=producer_file,
        verification_file=verification_file,
        producer_receipt_sha256=str(producer_receipt_sha),
        verification_artifact_sha256=str(verification_sha),
        config_authority_sha256=config_authority,
        implementation_authority_sha256=implementation_authority,
        complementarity_gate_sha256=edcm.canonical_sha256(gate),
        complementarity_passed=bool(gate["passed"]),
        terminal_execution_status=str(result["execution_status"]),
        verifier_mode=str(result["verifier_mode"]),
    )


@dataclass(frozen=True, slots=True)
class ActivationAssessment:
    verified_current_authority: bool
    complementarity_passed: bool
    activation_enabled: bool
    blockers: tuple[str, ...]
    authority_sha256: str | None
    assessment_sha256: str

    def __post_init__(self) -> None:
        _require(self.activation_enabled is False, "activation assessment escaped disabled state")
        _require(self.blockers == tuple(sorted(set(self.blockers))), "activation blockers not canonical")
        if self.authority_sha256 is not None:
            _digest(self.authority_sha256, "assessment authority_sha256")
        _digest(self.assessment_sha256, "assessment_sha256")
        _require(
            self.assessment_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "activation assessment self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        authority: VerifiedEDCMAuthority | None,
        authority_problem: str | None = None,
    ) -> Self:
        blockers = {"adapter-activation-disabled"}
        if authority is None:
            blockers.add("verified-current-edcm-authority-missing")
            if authority_problem:
                blockers.add(f"authority-invalid:{authority_problem}")
        elif not authority.complementarity_passed:
            blockers.add("edcm-complementarity-gate-failed")
        canonical_blockers = tuple(sorted(blockers))
        core = {
            "schema": ASSESSMENT_SCHEMA,
            "verified_current_authority": authority is not None,
            "complementarity_passed": bool(authority and authority.complementarity_passed),
            "activation_enabled": False,
            "blockers": list(canonical_blockers),
            "authority_sha256": authority.authority_sha256 if authority else None,
        }
        return cls(
            verified_current_authority=authority is not None,
            complementarity_passed=bool(authority and authority.complementarity_passed),
            activation_enabled=False,
            blockers=canonical_blockers,
            authority_sha256=authority.authority_sha256 if authority else None,
            assessment_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": ASSESSMENT_SCHEMA,
            "verified_current_authority": self.verified_current_authority,
            "complementarity_passed": self.complementarity_passed,
            "activation_enabled": self.activation_enabled,
            "blockers": list(self.blockers),
            "authority_sha256": self.authority_sha256,
        }
        if include_digest:
            result["assessment_sha256"] = self.assessment_sha256
        return result


@dataclass(frozen=True, slots=True)
class ObservationTranslation:
    source_observation: edcm.VisibleObservation
    observation_event: ObservationEvent
    hypothesis_event: HypothesisEvent
    accounting: TranslationAccounting
    translation_sha256: str

    def __post_init__(self) -> None:
        _coerce_observation(self.source_observation)
        _require(self.observation_event.branch_id == FACTUAL_BRANCH, "observation branch mutation")
        _require(self.hypothesis_event.branch_id == FACTUAL_BRANCH, "hypothesis branch mutation")
        _require(
            self.hypothesis_event.envelope.causal_parent_ids == (self.observation_event.event_id,),
            "observation/hypothesis causal join mismatch",
        )
        _digest(self.translation_sha256, "observation translation_sha256")
        _require(
            self.translation_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "observation translation self-hash mismatch",
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": TRANSLATION_SCHEMA,
            "kind": "observation",
            "source_sha256": edcm.canonical_sha256(dataclasses.asdict(self.source_observation)),
            "target_event_ids": [str(self.observation_event.event_id), str(self.hypothesis_event.event_id)],
            "branch_id": str(self.observation_event.branch_id),
            "accounting_sha256": self.accounting.accounting_sha256,
            "activation_enabled": False,
        }
        if include_digest:
            result["translation_sha256"] = self.translation_sha256
        return result


@dataclass(frozen=True, slots=True)
class DecisionTranslation:
    source_observation_sha256: str
    proposal_claims: tuple[ClaimMessage, ...]
    verification_claim: ClaimMessage | None
    action_intent: ActionIntent
    commitment_event: CommitmentEvent
    accounting: TranslationAccounting
    translation_sha256: str

    def __post_init__(self) -> None:
        _digest(self.source_observation_sha256, "decision source observation sha256")
        _require(isinstance(self.proposal_claims, tuple), "proposal claims must be immutable")
        _require(all(message.integrity_valid() for message in self.proposal_claims), "claim mutation")
        if self.verification_claim is not None:
            _require(self.verification_claim.integrity_valid(), "verification claim mutation")
        _require(self.action_intent.integrity_valid(), "action intent mutation")
        _require(self.commitment_event.branch_id == FACTUAL_BRANCH, "commitment branch mutation")
        committed = self.commitment_event.committed_payload.value()
        _require(
            isinstance(committed, dict) and committed.get("action_id") == self.action_intent.action_id,
            "action/commitment identity mismatch",
        )
        _digest(self.translation_sha256, "decision translation_sha256")
        _require(
            self.translation_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "decision translation self-hash mismatch",
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": TRANSLATION_SCHEMA,
            "kind": "decision",
            "source_observation_sha256": self.source_observation_sha256,
            "proposal_claim_ids": [message.header.message_id for message in self.proposal_claims],
            "verification_claim_id": (
                self.verification_claim.header.message_id if self.verification_claim else None
            ),
            "action_id": self.action_intent.action_id,
            "commitment_event_id": str(self.commitment_event.event_id),
            "branch_id": str(self.commitment_event.branch_id),
            "accounting_sha256": self.accounting.accounting_sha256,
            "activation_enabled": False,
        }
        if include_digest:
            result["translation_sha256"] = self.translation_sha256
        return result


@dataclass(frozen=True, slots=True)
class ConsequenceTranslation:
    source_transition_sha256: str
    consequence_event: ConsequenceEvent
    accounting: TranslationAccounting
    translation_sha256: str

    def __post_init__(self) -> None:
        _digest(self.source_transition_sha256, "consequence source transition sha256")
        _require(self.consequence_event.branch_id == FACTUAL_BRANCH, "consequence branch mutation")
        _digest(self.translation_sha256, "consequence translation_sha256")
        _require(
            self.translation_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "consequence translation self-hash mismatch",
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": TRANSLATION_SCHEMA,
            "kind": "consequence",
            "source_transition_sha256": self.source_transition_sha256,
            "consequence_event_id": str(self.consequence_event.event_id),
            "commitment_event_id": str(self.consequence_event.commitment_event_id),
            "branch_id": str(self.consequence_event.branch_id),
            "accounting_sha256": self.accounting.accounting_sha256,
            "activation_enabled": False,
        }
        if include_digest:
            result["translation_sha256"] = self.translation_sha256
        return result


def _coerce_observation(value: edcm.VisibleObservation | Mapping[str, Any]) -> edcm.VisibleObservation:
    if isinstance(value, Mapping):
        expected = {field.name for field in fields(edcm.VisibleObservation)}
        _exact_keys(value, expected, "EDCM visible observation")
        value = edcm.VisibleObservation(
            world_id=value["world_id"],
            event_id=value["event_id"],
            tick=value["tick"],
            local_blocked=tuple(value["local_blocked"]),
            relative_goal=tuple(value["relative_goal"]),
            previous_action=value["previous_action"],
            previous_reward=value["previous_reward"],
            novelty_channels=tuple(value["novelty_channels"]),
        )
    if type(value) is not edcm.VisibleObservation:
        if isinstance(value, edcm.EvaluatorTransition):
            raise AdapterContractError("evaluator-only EDCM transition cannot enter the factual adapter")
        raise AdapterContractError("source observation has an incompatible schema")
    payload = dataclasses.asdict(value)
    _reject_forbidden(payload, "EDCM visible observation")
    _require(isinstance(value.world_id, str) and bool(value.world_id.strip()), "world_id is empty")
    _digest(value.event_id, "EDCM observation event_id")
    _require(
        value.event_id == edcm.canonical_sha256({"world_id": value.world_id, "tick": value.tick}),
        "EDCM observation event identity mismatch",
    )
    _nonnegative_int(value.tick, "EDCM observation tick")
    _require(
        isinstance(value.local_blocked, tuple)
        and len(value.local_blocked) == 4
        and all(type(item) is int and item in {0, 1} for item in value.local_blocked),
        "local_blocked must contain four exact binary integers",
    )
    _require(
        isinstance(value.relative_goal, tuple)
        and len(value.relative_goal) == 2
        and all(type(item) is int for item in value.relative_goal),
        "relative_goal must contain two exact integers",
    )
    _require(type(value.previous_action) is int and 0 <= value.previous_action < 4, "previous action")
    _finite(value.previous_reward, "previous reward")
    _require(
        isinstance(value.novelty_channels, tuple)
        and len(value.novelty_channels) <= 64
        and all(type(item) is int and item in {0, 1} for item in value.novelty_channels),
        "novelty channels must be bounded binary integers",
    )
    return value


def _coerce_proposal(value: edcm.ProposalMessage | Mapping[str, Any]) -> edcm.ProposalMessage:
    if isinstance(value, Mapping):
        expected = {field.name for field in fields(edcm.ProposalMessage)}
        _exact_keys(value, expected, "EDCM proposal")
        provenance = value["provenance"]
        _exact_keys(
            provenance,
            {field.name for field in fields(edcm.Provenance)},
            "EDCM proposal provenance",
        )
        value = edcm.ProposalMessage(
            schema=value["schema"],
            message_id=value["message_id"],
            referent_id=value["referent_id"],
            source_event_id=value["source_event_id"],
            created_tick=value["created_tick"],
            max_age=value["max_age"],
            specialist_id=value["specialist_id"],
            specialist_kind=value["specialist_kind"],
            proposed_action=value["proposed_action"],
            confidence=value["confidence"],
            expected_progress=value["expected_progress"],
            evidence=tuple(value["evidence"]),
            producer_work_units=value["producer_work_units"],
            provenance=edcm.Provenance(**provenance),
        )
    if type(value) is not edcm.ProposalMessage:
        raise AdapterContractError("source specialist output has an incompatible schema")
    _reject_forbidden(value.payload(), "EDCM proposal")
    return value


class EDCMToESCSAdapter:

    def __init__(
        self,
        *,
        config: AdapterConfig = AdapterConfig(),
        verified_authority: VerifiedEDCMAuthority | None = None,
    ) -> None:
        self.config = config
        self.verified_authority = verified_authority

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(
            {
                "adapter_config_sha256": self.config.authority_sha256,
                "verified_edcm_authority_sha256": (
                    self.verified_authority.authority_sha256 if self.verified_authority else None
                ),
                "activation_enabled": False,
            }
        )

    def assess_activation(self, *, authority_problem: str | None = None) -> ActivationAssessment:
        return ActivationAssessment.create(
            authority=self.verified_authority,
            authority_problem=authority_problem,
        )

    def require_activation(self) -> None:
        assessment = self.assess_activation()
        if not assessment.verified_current_authority:
            raise AdapterActivationError("activation requires a verified EDCM result and current authority")
        if not assessment.complementarity_passed:
            raise AdapterActivationError("activation refuses a failed EDCM complementarity gate")
        raise AdapterActivationError("EDCM-to-ESCS adapter activation is disabled in this revision")

    def activate(self, *_: object, **__: object) -> None:
        self.require_activation()

    def _provenance(self, *, source_kind: str, source_id: str, source_sha256: str) -> dict[str, Any]:
        return {
            "adapter_id": ADAPTER_ID,
            "adapter_authority_sha256": self.authority_sha256,
            "source_study": "edcm1-v3",
            "source_kind": source_kind,
            "source_id": source_id,
            "source_sha256": source_sha256,
            "verified_edcm_authority_sha256": (
                self.verified_authority.authority_sha256 if self.verified_authority else None
            ),
            "branch_id": str(FACTUAL_BRANCH),
            "activation_enabled": False,
            "scientific_promotion_allowed": False,
        }

    def translate_observation(
        self,
        observation: edcm.VisibleObservation | Mapping[str, Any],
        *,
        source_work: edcm.AbstractWork | None = None,
    ) -> ObservationTranslation:
        source = _coerce_observation(observation)
        work = source_work or edcm.AbstractWork()
        source_payload = dataclasses.asdict(source)
        source_sha = edcm.canonical_sha256(source_payload)
        source_bytes = _canonical_limited(source_payload, self.config.max_source_bytes, "observation")
        pre_cost = _work_vector_with(
            {
                "raw_transport_and_adapters": len(source_bytes) * 2 + len(source_payload),
            }
        )
        provenance = self._provenance(
            source_kind=self.config.source_observation_schema,
            source_id=source.event_id,
            source_sha256=source_sha,
        )
        observed = ObservationEvent.create(
            raw_packet_or_delta_refs=(f"packet:edcm/{source.event_id}",),
            adapter_version=f"{ADAPTER_ID}:{self.config.authority_sha256}",
            sensor_scope={
                "world_id": source.world_id,
                "edcm_event_id": source.event_id,
                "tick": source.tick,
                "local_blocked": list(source.local_blocked),
                "relative_goal": list(source.relative_goal),
                "previous_action": source.previous_action,
                "previous_reward": source.previous_reward,
                "novelty_channels": list(source.novelty_channels),
            },
            transport_and_detection_cost=pre_cost,
            clock_start_tick=source.tick,
            clock_end_tick=source.tick,
            source_and_provenance=provenance,
            measured_creation_cost=pre_cost,
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )
        hypothesis = HypothesisEvent.create(
            origin=HypothesisOrigin.EVENT_FORMER,
            epistemic_status=EpistemicStatus.OBSERVED_CANDIDATE,
            referent_hypotheses={f"goal@{source.world_id}": 1.0},
            factor_change_distribution={
                "local_blocked": list(source.local_blocked),
                "relative_goal": list(source.relative_goal),
                "previous_action": source.previous_action,
                "previous_reward": source.previous_reward,
                "novelty_channels": list(source.novelty_channels),
            },
            decision_relevance_distribution={
                "status": "unverified-structured-input",
                "factor_scope": list(FACTOR_SCOPE),
            },
            reducibility_distribution={"status": "not-estimated-by-adapter"},
            supporting_event_ids=(observed.event_id,),
            calibrated_confidence=0.0,
            abstention_reason="structured-observation-only-activation-disabled",
            predicted_value_of_further_computation=0.0,
            causal_parent_ids=(observed.event_id,),
            clock_start_tick=source.tick,
            clock_end_tick=source.tick,
            source_and_provenance=provenance,
            measured_creation_cost=WorkVector(event_formation=len(source_bytes)),
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )
        target_payload = [observed.payload(), hypothesis.payload()]
        accounting = TranslationAccounting.create(
            stage="structured-observation",
            source_work=work,
            weights=self.config.weights,
            source_bucket="raw_transport_and_adapters",
            adapter_bucket="raw_transport_and_adapters",
            source_payload=source_payload,
            target_payload=target_payload,
            validation_operations=len(source_payload) + len(FACTOR_SCOPE),
            source_limit=self.config.max_source_bytes,
            target_limit=self.config.max_target_bytes,
        )
        core = {
            "schema": TRANSLATION_SCHEMA,
            "kind": "observation",
            "source_sha256": source_sha,
            "target_event_ids": [str(observed.event_id), str(hypothesis.event_id)],
            "branch_id": str(FACTUAL_BRANCH),
            "accounting_sha256": accounting.accounting_sha256,
            "activation_enabled": False,
        }
        return ObservationTranslation(source, observed, hypothesis, accounting, canonical_sha256(core))

    def _validate_proposal(
        self,
        proposal: edcm.ProposalMessage,
        observation: edcm.VisibleObservation,
    ) -> None:
        _require(proposal.schema == self.config.source_proposal_schema, "EDCM proposal schema mismatch")
        _require(proposal.integrity_valid(), "EDCM proposal integrity mismatch")
        _require(proposal.source_event_id == observation.event_id, "proposal source event mismatch")
        _require(proposal.created_tick == observation.tick, "delayed/future proposal is not factual-current")
        _require(proposal.max_age in {0, 1}, "proposal age bound exceeds EDCM authority")
        _require(proposal.specialist_kind in edcm.PROPOSER_ORDER, "unknown EDCM specialist kind")
        _require(bool(proposal.specialist_id.strip()), "empty EDCM specialist id")
        _require(type(proposal.proposed_action) is int and 0 <= proposal.proposed_action < 4, "action")
        confidence = _finite(proposal.confidence, "proposal confidence")
        _require(0.0 <= confidence <= 1.0, "proposal confidence outside [0, 1]")
        _finite(proposal.expected_progress, "proposal expected progress")
        _nonnegative_int(proposal.producer_work_units, "proposal producer work")
        _digest(proposal.provenance.state_digest, "proposal state digest")
        _require(
            proposal.provenance.producer_id == proposal.specialist_id
            and proposal.provenance.producer_kind == proposal.specialist_kind
            and proposal.provenance.world_id == observation.world_id
            and proposal.provenance.source_event_id == observation.event_id,
            "proposal provenance mismatch",
        )
        _require(proposal.referent_id == f"goal@{observation.world_id}", "proposal referent mismatch")
        for evidence in proposal.evidence:
            _require(isinstance(evidence, str) and bool(evidence), "proposal evidence must be text")
            lowered = evidence.lower()
            _require(
                not any(token in lowered for token in _FORBIDDEN_TEXT),
                "proposal evidence names an evaluator/future-only field",
            )

    def _proposal_claim(
        self,
        source: ObservationTranslation,
        proposal: edcm.ProposalMessage,
    ) -> ClaimMessage:
        self._validate_proposal(proposal, source.source_observation)
        payload = _canonical_limited(
            proposal.payload(),
            PROPOSAL_CLAIM_SCHEMA.max_payload_bytes,
            "EDCM proposal claim payload",
        )
        return ClaimMessage.create(
            schema=PROPOSAL_CLAIM_SCHEMA,
            source_hypothesis_event_ids=(str(source.hypothesis_event.event_id),),
            referent_hypotheses=(proposal.referent_id,),
            branch_id=str(FACTUAL_BRANCH),
            factor_scope=FACTOR_SCOPE,
            claim_type="action_proposal",
            epistemic_status=EpistemicStatus.INFERRED,
            supporting_event_ids=(
                str(source.observation_event.event_id),
                str(source.hypothesis_event.event_id),
            ),
            producer_actor_id=f"actor:edcm/{proposal.specialist_id}",
            producer_state_version=proposal.provenance.state_digest,
            calibrated_confidence=proposal.confidence,
            created_tick=proposal.created_tick,
            expiry_tick=proposal.created_tick,
            predicted_utility=(proposal.expected_progress,),
            producer_operations=proposal.producer_work_units,
            payload_form="edcm1-canonical-json",
            payload_bytes=payload,
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )

    def _validate_verification(
        self,
        verification: edcm.VerificationMessage,
        *,
        observation: edcm.VisibleObservation,
        proposal_ids: set[str],
    ) -> None:
        _require(type(verification) is edcm.VerificationMessage, "verification schema incompatible")
        _require(verification.schema == self.config.source_verification_schema, "verification schema drift")
        payload = dataclasses.asdict(verification)
        _reject_forbidden(payload, "EDCM verification message")
        body = dict(payload)
        message_id = body.pop("message_id")
        _require(message_id == edcm.canonical_sha256(body), "EDCM verification integrity mismatch")
        _require(verification.source_event_id == observation.event_id, "verification source mismatch")
        _require(verification.created_tick == observation.tick, "verification is future/delayed")
        _require(
            len(verification.contradicted_message_ids) == len(set(verification.contradicted_message_ids)),
            "verification contradiction ids are duplicated",
        )
        referenced = set(verification.contradicted_message_ids)
        if verification.endorsed_message_id is not None:
            referenced.add(verification.endorsed_message_id)
        _require(referenced <= proposal_ids, "verification references an unknown proposal")
        _require(
            (verification.abstained and verification.endorsed_message_id is None)
            or (not verification.abstained and verification.endorsed_message_id is not None),
            "verification abstention/endorsement mismatch",
        )
        confidence = _finite(verification.confidence, "verification confidence")
        _require(0.0 <= confidence <= 1.0, "verification confidence outside [0, 1]")
        _digest(verification.state_digest, "verification state digest")
        for reason in verification.reason_codes:
            _require(isinstance(reason, str) and bool(reason), "verification reason must be text")
            _require(
                not any(token in reason.lower() for token in _FORBIDDEN_TEXT),
                "verification reason names evaluator/future-only data",
            )

    def _verification_claim(
        self,
        source: ObservationTranslation,
        verification: edcm.VerificationMessage,
        proposal_ids: set[str],
    ) -> ClaimMessage:
        self._validate_verification(
            verification,
            observation=source.source_observation,
            proposal_ids=proposal_ids,
        )
        payload_value = dataclasses.asdict(verification)
        payload = _canonical_limited(
            payload_value,
            VERIFICATION_CLAIM_SCHEMA.max_payload_bytes,
            "EDCM verification claim payload",
        )
        return ClaimMessage.create(
            schema=VERIFICATION_CLAIM_SCHEMA,
            source_hypothesis_event_ids=(str(source.hypothesis_event.event_id),),
            referent_hypotheses=(f"goal@{source.source_observation.world_id}",),
            branch_id=str(FACTUAL_BRANCH),
            factor_scope=FACTOR_SCOPE,
            claim_type="proposal_verification",
            epistemic_status=EpistemicStatus.INFERRED,
            supporting_event_ids=(
                str(source.observation_event.event_id),
                str(source.hypothesis_event.event_id),
            ),
            producer_actor_id="actor:edcm/contradiction_verifier",
            producer_state_version=verification.state_digest,
            calibrated_confidence=verification.confidence,
            created_tick=verification.created_tick,
            expiry_tick=verification.created_tick,
            predicted_utility=(),
            producer_operations=len(payload) * 2 + len(proposal_ids),
            payload_form="edcm1-canonical-json",
            payload_bytes=payload,
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )

    def translate_decision(
        self,
        source: ObservationTranslation,
        prepared: edcm.PreparedDecision,
        resolution: edcm.Resolution,
    ) -> DecisionTranslation:
        _require(type(prepared) is edcm.PreparedDecision, "prepared decision schema incompatible")
        _require(type(resolution) is edcm.Resolution, "resolution schema incompatible")
        _require(prepared.observation == source.source_observation, "prepared observation mutation")
        activation = prepared.activation
        _require(type(activation) is edcm.ActivationRecord, "activation schema incompatible")
        expected_order = tuple(kind for kind in edcm.PROPOSER_ORDER if kind in activation.initial)
        _require(
            activation.initial == expected_order
            and len(activation.initial) == len(set(activation.initial))
            and len(activation.initial) <= self.config.max_initial_specialists,
            "activation specialists are unbounded, duplicated, or noncanonical",
        )
        round_count = 1 + int(bool(activation.extra_round))
        _require(round_count <= self.config.max_reasoning_rounds, "unbounded reasoning rounds")
        _require(
            activation.extra_round in ((), (edcm.VERIFIER_ID,)),
            "only one bounded verifier round may follow proposals",
        )
        _require(prepared.active_ids == activation.initial, "active specialist identity mismatch")
        _digest(prepared.pre_bus_state_sha256, "EDCM evaluator fork-audit digest")
        proposals = tuple(_coerce_proposal(value) for value in prepared.proposals)
        _require(len(proposals) == len(activation.initial), "proposal/activation count mismatch")
        _require(
            tuple(proposal.specialist_kind for proposal in proposals) == activation.initial,
            "proposal specialist order/identity mismatch",
        )
        proposal_ids = {proposal.message_id for proposal in proposals}
        _require(len(proposal_ids) == len(proposals), "duplicate EDCM proposal identity")
        claims = tuple(self._proposal_claim(source, proposal) for proposal in proposals)
        delivered = tuple(_coerce_proposal(value) for value in resolution.delivered)
        delivered_ids = tuple(proposal.message_id for proposal in delivered)
        _require(
            delivered_ids == tuple(proposal.message_id for proposal in proposals),
            "factual adapter rejects lesion, delay, dropped, reordered, or mutated messages",
        )
        delayed_planner = resolution.delayed_planner_for_next_tick
        current_planner = next(
            (proposal for proposal in proposals if proposal.specialist_kind == "short_horizon_planner"),
            None,
        )
        _require(
            (delayed_planner is None and current_planner is None)
            or (
                delayed_planner is not None
                and current_planner is not None
                and delayed_planner.payload() == current_planner.payload()
            ),
            "EDCM delay-control cache does not bind the current planner message",
        )
        verification_claim: ClaimMessage | None = None
        if activation.extra_round:
            verification_message = resolution.verification
            if resolution.verifier_executed is not True or verification_message is None:
                raise AdapterContractError("declared verifier round did not execute")
            verification_claim = self._verification_claim(
                source,
                verification_message,
                proposal_ids,
            )
        else:
            _require(
                resolution.verifier_executed is False and resolution.verification is None,
                "undeclared verifier output",
            )
        expected_message_bytes = sum(proposal.encoded_bytes for proposal in delivered)
        if resolution.verification is not None:
            expected_message_bytes += resolution.verification.encoded_bytes
        _require(resolution.message_bytes == expected_message_bytes, "EDCM message-byte accounting mismatch")
        _require(type(resolution.action) is int and 0 <= resolution.action < 4, "resolved action invalid")
        selected = None
        if resolution.chosen_message_id is not None:
            selected = next(
                (proposal for proposal in proposals if proposal.message_id == resolution.chosen_message_id),
                None,
            )
            if selected is None:
                raise AdapterContractError("chosen proposal is absent")
            _require(selected.proposed_action == resolution.action, "chosen proposal/action mutation")
        else:
            fallback = (
                source.source_observation.previous_action
                if not source.source_observation.local_blocked[source.source_observation.previous_action]
                else next(
                    (action for action in range(4) if not source.source_observation.local_blocked[action]),
                    0,
                )
            )
            _require(resolution.action == fallback, "fallback action mutation")
        source_work = _add_source_work(prepared.work, resolution.work)
        prepared_work_units = edcm.AbstractWork(**_source_work_payload(prepared.work)).total(
            self.config.weights
        )
        resolution_work_units = edcm.AbstractWork(**_source_work_payload(resolution.work)).total(
            self.config.weights
        )
        _require(
            sum(proposal.producer_work_units for proposal in proposals) <= prepared_work_units,
            "proposal producer work exceeds fully charged prepared-decision work",
        )
        source_payload = {
            "observation": dataclasses.asdict(prepared.observation),
            "activation": dataclasses.asdict(activation),
            "proposals": [proposal.payload() for proposal in proposals],
            "resolution": {
                "action": resolution.action,
                "chosen_message_id": resolution.chosen_message_id,
                "delivered_message_ids": list(delivered_ids),
                "verification": (
                    dataclasses.asdict(resolution.verification) if resolution.verification else None
                ),
                "message_bytes": resolution.message_bytes,
                "verifier_executed": resolution.verifier_executed,
            },
        }
        _reject_forbidden(source_payload, "EDCM decision")
        public_state_version = canonical_sha256(
            {
                "source_hypothesis_event_id": str(source.hypothesis_event.event_id),
                "activation": dataclasses.asdict(activation),
                "proposal_message_ids": sorted(proposal_ids),
                "verification_message_id": (
                    resolution.verification.message_id if resolution.verification else None
                ),
            }
        )
        action_payload = _canonical_limited(
            {
                "schema": "mop-escs-edcm-action/v1",
                "action": resolution.action,
                "source_edcm_event_id": source.source_observation.event_id,
                "chosen_edcm_message_id": resolution.chosen_message_id,
            },
            16 * 1024,
            "EDCM action intent",
        )
        source_work_units = edcm.AbstractWork(**_source_work_payload(source_work)).total(self.config.weights)
        action_intent = ActionIntent.create(
            source_event_id=str(source.hypothesis_event.event_id),
            branch_id=str(FACTUAL_BRANCH),
            referent_hypotheses=(f"goal@{source.source_observation.world_id}",),
            epistemic_status=EpistemicStatus.INFERRED,
            producer_actor_id="actor:edcm/coalition_arbiter",
            producer_state_version=public_state_version,
            created_tick=source.source_observation.tick,
            expiry_tick=source.source_observation.tick + 1,
            producer_operations=source_work_units,
            payload_form="edcm1-canonical-json",
            payload_bytes=action_payload,
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )
        coalition_digest = canonical_sha256(
            {
                "source_hypothesis_event_id": str(source.hypothesis_event.event_id),
                "initial": list(activation.initial),
                "extra_round": list(activation.extra_round),
                "temporary": True,
            }
        )
        predicted_progress = selected.expected_progress if selected is not None else 0.0
        predicted_confidence = selected.confidence if selected is not None else 0.0
        pre_adapter = len(_canonical_limited(source_payload, self.config.max_source_bytes, "decision"))
        commitment = CommitmentEvent.create(
            coalition_id=f"coalition:edcm/{coalition_digest}",
            commitment_kind=CommitmentKind.EXTERNAL_ACTION,
            committed_payload={
                "action_id": action_intent.action_id,
                "action": resolution.action,
                "source_edcm_event_id": source.source_observation.event_id,
                "chosen_edcm_message_id": resolution.chosen_message_id,
                "proposal_claim_ids": [claim.header.message_id for claim in claims],
                "verification_claim_id": (
                    verification_claim.header.message_id if verification_claim else None
                ),
            },
            decision_distribution={str(action): float(action == resolution.action) for action in range(4)},
            deadline_tick=source.source_observation.tick + 1,
            predicted_utility_vector={
                "expected_progress": predicted_progress,
                "source_confidence": predicted_confidence,
            },
            predicted_full_cost=_work_vector_with(
                {
                    "actor_execution": source_work_units,
                    "messages": pre_adapter * 2 + len(source_payload),
                }
            ),
            causal_parent_ids=(source.hypothesis_event.event_id,),
            clock_start_tick=source.source_observation.tick,
            clock_end_tick=source.source_observation.tick,
            source_and_provenance=self._provenance(
                source_kind="mop-edcm1-clean-resolution/v3",
                source_id=source.source_observation.event_id,
                source_sha256=edcm.canonical_sha256(source_payload),
            ),
            measured_creation_cost=WorkVector(messages=pre_adapter * 2 + len(source_payload)),
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )
        target_payload = {
            "claims": [claim.wire_payload() for claim in claims],
            "verification_claim": verification_claim.wire_payload() if verification_claim else None,
            "action_intent": {
                "action_id": action_intent.action_id,
                "header": action_intent.identity_payload(),
                "payload_base64": base64.b64encode(action_intent.payload_bytes).decode("ascii"),
            },
            "commitment": commitment.payload(),
        }
        accounting = TranslationAccounting.create(
            stage="specialist-resolution-and-commitment",
            source_work=source_work,
            weights=self.config.weights,
            source_bucket=None,
            adapter_bucket="messages",
            source_bucket_totals={
                "actor_execution": prepared_work_units,
                "messages": resolution_work_units,
            },
            source_payload=source_payload,
            target_payload=target_payload,
            validation_operations=(
                len(proposals) * 16 + len(activation.initial) + int(bool(activation.extra_round)) * 12 + 12
            ),
            source_limit=self.config.max_source_bytes,
            target_limit=self.config.max_target_bytes,
        )
        core = {
            "schema": TRANSLATION_SCHEMA,
            "kind": "decision",
            "source_observation_sha256": edcm.canonical_sha256(dataclasses.asdict(source.source_observation)),
            "proposal_claim_ids": [claim.header.message_id for claim in claims],
            "verification_claim_id": (verification_claim.header.message_id if verification_claim else None),
            "action_id": action_intent.action_id,
            "commitment_event_id": str(commitment.event_id),
            "branch_id": str(FACTUAL_BRANCH),
            "accounting_sha256": accounting.accounting_sha256,
            "activation_enabled": False,
        }
        source_observation_sha256 = edcm.canonical_sha256(dataclasses.asdict(source.source_observation))
        return DecisionTranslation(
            source_observation_sha256=source_observation_sha256,
            proposal_claims=claims,
            verification_claim=verification_claim,
            action_intent=action_intent,
            commitment_event=commitment,
            accounting=accounting,
            translation_sha256=canonical_sha256(core),
        )

    def translate_consequence(
        self,
        decision: DecisionTranslation,
        transition: edcm.VisibleTransition,
        *,
        update_work: edcm.AbstractWork,
        successor_observation: ObservationTranslation | None = None,
    ) -> ConsequenceTranslation:
        if isinstance(transition, edcm.EvaluatorTransition):
            raise AdapterContractError("evaluator-only transition cannot enter an ESCS consequence")
        _require(type(transition) is edcm.VisibleTransition, "visible transition schema incompatible")
        before = _coerce_observation(transition.before)
        _require(
            edcm.canonical_sha256(dataclasses.asdict(before)) == decision.source_observation_sha256,
            "consequence before-state does not match commitment authority",
        )
        _require(decision.action_intent.integrity_valid(), "committed action intent is corrupt")
        action_payload = json.loads(decision.action_intent.payload_bytes)
        _exact_keys(
            action_payload,
            {"schema", "action", "source_edcm_event_id", "chosen_edcm_message_id"},
            "EDCM action intent payload",
        )
        _require(action_payload["schema"] == "mop-escs-edcm-action/v1", "action payload schema drift")
        _require(
            action_payload["source_edcm_event_id"] == before.event_id,
            "action payload source-event mutation",
        )
        _require(transition.action == action_payload["action"], "transition action/commitment mismatch")
        feedback = transition.feedback
        _require(type(feedback) is edcm.PublicFeedback, "public feedback schema incompatible")
        _require(
            feedback.source_event_id == before.event_id
            and feedback.tick == before.tick
            and feedback.action == transition.action,
            "public feedback authority mismatch",
        )
        _finite(feedback.reward, "feedback reward")
        _require(type(feedback.blocked) is bool and type(feedback.reached_goal) is bool, "feedback flags")
        after_value = transition.after
        if after_value is None:
            raise AdapterContractError("official EDCM visible transition requires successor state")
        after = _coerce_observation(after_value)
        _require(after.world_id == before.world_id, "transition silently changed factual world identity")
        _require(after.tick == before.tick + 1, "transition successor tick mismatch")
        _require(after.previous_action == transition.action, "successor previous-action mismatch")
        _require(after.previous_reward == feedback.reward, "successor previous-reward mismatch")
        _require(type(transition.terminal) is bool, "transition terminal flag must be boolean")
        if successor_observation is not None:
            _require(
                successor_observation.source_observation == after,
                "successor observation translation mismatch",
            )
        source_payload = {
            "before": dataclasses.asdict(before),
            "action": transition.action,
            "feedback": dataclasses.asdict(feedback),
            "after": dataclasses.asdict(after),
            "terminal": transition.terminal,
        }
        _reject_forbidden(source_payload, "EDCM visible transition")
        parents = [decision.commitment_event.event_id]
        if successor_observation is not None:
            parents.append(successor_observation.observation_event.event_id)
        source_sha = edcm.canonical_sha256(source_payload)
        work_payload = _source_work_payload(update_work)
        source_work_units = edcm.AbstractWork(**work_payload).total(self.config.weights)
        pre_adapter = len(
            _canonical_limited(source_payload, self.config.max_source_bytes, "visible transition")
        )
        consequence = ConsequenceEvent.create(
            commitment_event_id=decision.commitment_event.event_id,
            observed_outcome={
                "source_edcm_event_id": before.event_id,
                "action": transition.action,
                "reward": feedback.reward,
                "blocked": feedback.blocked,
                "reached_goal": feedback.reached_goal,
                "successor_edcm_event_id": after.event_id,
                "successor_observation_event_id": (
                    str(successor_observation.observation_event.event_id) if successor_observation else None
                ),
                "terminal": transition.terminal,
            },
            realized_utility_vector={"reward": feedback.reward},
            delayed_or_partial=False,
            observation_uncertainty=0.0,
            realized_full_cost=_work_vector_with(
                {
                    "learning": source_work_units,
                    "messages": pre_adapter * 2 + len(source_payload),
                }
            ),
            causal_parent_ids=tuple(parents),
            clock_start_tick=after.tick,
            clock_end_tick=after.tick,
            source_and_provenance=self._provenance(
                source_kind="mop-edcm1-visible-transition/v3",
                source_id=after.event_id,
                source_sha256=source_sha,
            ),
            measured_creation_cost=WorkVector(messages=pre_adapter * 2 + len(source_payload)),
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )
        target_payload = consequence.payload()
        accounting = TranslationAccounting.create(
            stage="visible-consequence",
            source_work=update_work,
            weights=self.config.weights,
            source_bucket="learning",
            adapter_bucket="messages",
            source_payload=source_payload,
            target_payload=target_payload,
            validation_operations=18,
            source_limit=self.config.max_source_bytes,
            target_limit=self.config.max_target_bytes,
        )
        core = {
            "schema": TRANSLATION_SCHEMA,
            "kind": "consequence",
            "source_transition_sha256": source_sha,
            "consequence_event_id": str(consequence.event_id),
            "commitment_event_id": str(consequence.commitment_event_id),
            "branch_id": str(FACTUAL_BRANCH),
            "accounting_sha256": accounting.accounting_sha256,
            "activation_enabled": False,
        }
        return ConsequenceTranslation(
            source_transition_sha256=source_sha,
            consequence_event=consequence,
            accounting=accounting,
            translation_sha256=canonical_sha256(core),
        )


__all__ = [
    "ACCOUNTING_SCHEMA",
    "ADAPTER_ACTIVATION_ENABLED",
    "ADAPTER_ID",
    "ADAPTER_SCHEMA",
    "ASSESSMENT_SCHEMA",
    "AUTHORITY_SCHEMA",
    "AdapterActivationError",
    "AdapterConfig",
    "AdapterContractError",
    "ActivationAssessment",
    "ConsequenceTranslation",
    "DecisionTranslation",
    "EDCMToESCSAdapter",
    "ObservationTranslation",
    "PROPOSAL_CLAIM_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TRANSLATION_SCHEMA",
    "TranslationAccounting",
    "VERIFICATION_CLAIM_SCHEMA",
    "VerifiedEDCMAuthority",
    "load_verified_edcm_authority",
]
