
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import FrozenJSON, canonical_sha256

HARNESS_SCHEMA = "mop-expansion-harness/v1"
SENTINEL_RESULT_SCHEMA = "mop-expansion-sentinel-result/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
REQUIRED_CONTROLS = (
    "wrong-time",
    "wrong-event",
    "appearance-only",
    "action-blind",
    "action-shuffled",
    "stale-memory",
    "unavailable-memory",
    "identity-ambiguity-abstain",
    "exact-replay",
)
_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ArmSpec:
    id: str
    role: str
    description: str

    def __post_init__(self) -> None:
        if _ID_RE.fullmatch(self.id) is None:
            raise ValueError("arm id must use stable lowercase characters")
        if self.role not in {"primary", "baseline", "negative-control"}:
            raise ValueError(f"unsupported arm role {self.role!r}")
        if not self.description.strip():
            raise ValueError("arm description must not be empty")

    def payload(self) -> dict[str, str]:
        return {"id": self.id, "role": self.role, "description": self.description}


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    device: str
    cpu_threads: int
    memory_limit_bytes: int
    wall_limit_seconds: int
    accelerator_required: bool
    model_weights_loaded: bool

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ValueError("Wave E0 mechanics are CPU-only")
        if self.cpu_threads < 1 or self.memory_limit_bytes < 1 or self.wall_limit_seconds < 1:
            raise ValueError("resource envelope limits must be positive")
        if self.accelerator_required or self.model_weights_loaded:
            raise ValueError("Wave E0 cannot require an accelerator or model weights")

    def payload(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cpu_threads": self.cpu_threads,
            "memory_limit_bytes": self.memory_limit_bytes,
            "wall_limit_seconds": self.wall_limit_seconds,
            "accelerator_required": self.accelerator_required,
            "model_weights_loaded": self.model_weights_loaded,
        }


@dataclass(frozen=True, slots=True)
class IndependentUnit:
    ref: str
    seed: int
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.ref.startswith("unit:") or _ID_RE.fullmatch(self.ref) is None:
            raise ValueError("independent unit ref must use the unit namespace")
        if self.seed < 0:
            raise ValueError("independent unit seed must be nonnegative")
        if _SHA256_RE.fullmatch(self.artifact_sha256) is None:
            raise ValueError("independent unit artifact digest is invalid")

    def payload(self) -> dict[str, Any]:
        return {"ref": self.ref, "seed": self.seed, "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class StopContract:
    minimum_independent_units: int
    require_all_controls: bool
    stop_on_identity_drift: bool
    stop_on_metric_failure: bool
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.minimum_independent_units < 1:
            raise ValueError("minimum independent unit count must be positive")
        if not self.require_all_controls or not self.stop_on_identity_drift:
            raise ValueError("Wave E0 must fail closed on control or identity drift")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("Wave E0 claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "minimum_independent_units": self.minimum_independent_units,
            "require_all_controls": self.require_all_controls,
            "stop_on_identity_drift": self.stop_on_identity_drift,
            "stop_on_metric_failure": self.stop_on_metric_failure,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class ExpansionHarnessContract:
    arms: tuple[ArmSpec, ...]
    controls: tuple[str, ...]
    resources: ResourceEnvelope
    independent_units: tuple[IndependentUnit, ...]
    stop: StopContract
    schema: str = HARNESS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HARNESS_SCHEMA:
            raise ValueError(f"unsupported harness schema {self.schema!r}")
        if tuple(self.controls) != REQUIRED_CONTROLS:
            raise ValueError("harness control set or order drift")
        if len({row.id for row in self.arms}) != len(self.arms):
            raise ValueError("harness arm ids must be unique")
        if {row.role for row in self.arms} != {"primary", "baseline", "negative-control"}:
            raise ValueError("harness requires primary, baseline, and negative-control arms")
        if len({row.ref for row in self.independent_units}) != len(self.independent_units):
            raise ValueError("independent unit references must be unique")
        if len({row.seed for row in self.independent_units}) != len(self.independent_units):
            raise ValueError("independent unit seeds must be unique")
        if len(self.independent_units) < self.stop.minimum_independent_units:
            raise ValueError("contract has fewer independent units than its stop minimum")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms": [row.payload() for row in self.arms],
            "controls": list(self.controls),
            "resources": self.resources.payload(),
            "independent_units": [row.payload() for row in self.independent_units],
            "stop": self.stop.payload(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class SentinelSpec:
    id: str
    title: str
    metric_names: tuple[str, ...]
    harness_contract_sha256: str

    def __post_init__(self) -> None:
        if _ID_RE.fullmatch(self.id) is None:
            raise ValueError("sentinel id must use stable lowercase characters")
        if not self.title.strip() or not self.metric_names:
            raise ValueError("sentinel title and metric names are required")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("sentinel metric names must be unique")
        if any(_ID_RE.fullmatch(name) is None for name in self.metric_names):
            raise ValueError("sentinel metric names must use stable lowercase characters")
        if _SHA256_RE.fullmatch(self.harness_contract_sha256) is None:
            raise ValueError("sentinel harness digest is invalid")

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "metric_names": list(self.metric_names),
            "harness_contract_sha256": self.harness_contract_sha256,
        }


@dataclass(frozen=True, slots=True)
class UnitArtifact:
    unit: IndependentUnit
    content: FrozenJSON

    def __post_init__(self) -> None:
        if self.unit.artifact_sha256 != self.content.sha256:
            raise ValueError("unit metadata and content digest disagree")

    def payload(self) -> dict[str, Any]:
        return {
            "unit": self.unit.payload(),
            "artifact": self.content.value(),
            "artifact_sha256": self.content.sha256,
        }


@dataclass(frozen=True, slots=True)
class SentinelResult:
    sentinel: SentinelSpec
    per_unit: tuple[dict[str, Any], ...]
    all_units_pass: bool
    status: str
    claim_scope: str = CLAIM_SCOPE
    schema: str = SENTINEL_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SENTINEL_RESULT_SCHEMA:
            raise ValueError(f"unsupported sentinel result schema {self.schema!r}")
        expected_status = "mechanics-pass" if self.all_units_pass else "mechanics-fail"
        if self.status != expected_status:
            raise ValueError("sentinel status does not match its aggregate")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("sentinel claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sentinel": self.sentinel.payload(),
            "per_unit": list(self.per_unit),
            "all_units_pass": self.all_units_pass,
            "status": self.status,
            "claim_scope": self.claim_scope,
        }


MetricEvaluator = Callable[[dict[str, Any]], dict[str, bool]]


def make_contract(units: Sequence[IndependentUnit]) -> ExpansionHarnessContract:
    return ExpansionHarnessContract(
        arms=(
            ArmSpec(
                id="shared-mechanics",
                role="primary",
                description="typed event, intervention, and lifecycle mechanics",
            ),
            ArmSpec(
                id="identity-only",
                role="baseline",
                description="identity links without the tested transition",
            ),
            ArmSpec(
                id="mismatched-control",
                role="negative-control",
                description="wrong clock, event, action, or memory join",
            ),
        ),
        controls=REQUIRED_CONTROLS,
        resources=ResourceEnvelope(
            device="cpu",
            cpu_threads=1,
            memory_limit_bytes=64 * 1024 * 1024,
            wall_limit_seconds=30,
            accelerator_required=False,
            model_weights_loaded=False,
        ),
        independent_units=tuple(units),
        stop=StopContract(
            minimum_independent_units=3,
            require_all_controls=True,
            stop_on_identity_drift=True,
            stop_on_metric_failure=True,
        ),
    )


def run_sentinel(
    *,
    spec: SentinelSpec,
    contract: ExpansionHarnessContract,
    artifacts: Mapping[str, UnitArtifact],
    evaluator: MetricEvaluator,
) -> SentinelResult:

    if spec.harness_contract_sha256 != contract.sha256:
        raise ValueError("sentinel does not reference the supplied harness contract")
    rows: list[dict[str, Any]] = []
    for unit in contract.independent_units:
        artifact = artifacts.get(unit.ref)
        if artifact is None:
            raise ValueError(f"missing independent unit artifact {unit.ref}")
        if artifact.unit != unit or artifact.content.sha256 != unit.artifact_sha256:
            raise ValueError(f"independent unit identity drift for {unit.ref}")
        raw = artifact.content.value()
        if not isinstance(raw, dict):
            raise ValueError(f"independent unit {unit.ref} is not a JSON mapping")
        metrics = evaluator(raw)
        if tuple(metrics) != spec.metric_names:
            raise ValueError(f"sentinel {spec.id} metric schema drift")
        if any(type(value) is not bool for value in metrics.values()):
            raise ValueError(f"sentinel {spec.id} metrics must be booleans")
        row_pass = all(metrics.values())
        rows.append(
            {
                "unit_ref": unit.ref,
                "seed": unit.seed,
                "artifact_sha256": unit.artifact_sha256,
                "metrics": metrics,
                "all_metrics_pass": row_pass,
            }
        )
        if not row_pass and contract.stop.stop_on_metric_failure:
            break
    all_pass = len(rows) >= contract.stop.minimum_independent_units and all(
        row["all_metrics_pass"] is True for row in rows
    )
    return SentinelResult(
        sentinel=spec,
        per_unit=tuple(rows),
        all_units_pass=all_pass,
        status="mechanics-pass" if all_pass else "mechanics-fail",
    )
