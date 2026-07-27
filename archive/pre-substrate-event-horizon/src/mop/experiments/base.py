from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..evidence import canonical_bytes, canonical_sha256

Executor = Callable[[DictConfig, DeviceInfo, Path], dict[str, Any]]
Verifier = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
PROGRAM = ("execute_provider", "verify", "project")
REQUIRED = {
    "id",
    "name",
    "question",
    "null_hypothesis",
    "metrics",
    "controls",
    "source",
    "split",
    "unit",
    "treatments",
    "sesoi",
    "multiplicity",
    "budget",
    "stop",
    "claim_ceiling",
    "provider",
    "verifier",
    "program",
    "status",
    "resource_tier",
}


class RecordRefused(ValueError):
    pass


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_bytes(value))


def validate_declaration(record: Mapping[str, Any]) -> None:
    missing = REQUIRED - record.keys()
    if missing:
        raise RecordRefused(f"experiment declaration is missing {sorted(missing)}")
    for field in ("id", "name", "question", "null_hypothesis", "provider", "verifier"):
        if not str(record[field]).strip():
            raise RecordRefused(f"experiment declaration has an empty {field}")
    if tuple(record["program"]) != PROGRAM:
        raise RecordRefused("experiment program is unknown or reordered")
    if not tuple(record["metrics"]) or not tuple(record["controls"]):
        raise RecordRefused("metrics and controls must be nonempty")
    claims = record["claim_ceiling"]
    if not isinstance(claims, Mapping):
        raise RecordRefused("claim_ceiling must be a mapping")
    for field in ("activation_allowed", "scientific_promotion", "independent_confirmation"):
        if claims.get(field) is not False:
            raise RecordRefused(f"claim_ceiling.{field} must be false")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    declaration: dict[str, Any]
    record_sha256: str
    executor: Executor | None
    verifier: Verifier | None

    @property
    def id(self) -> str:
        return str(self.declaration["id"])

    @property
    def null_hypothesis(self) -> str:
        return str(self.declaration["null_hypothesis"])

    def contract(self) -> dict[str, Any]:
        return {"record": _copy(self.declaration), "record_sha256": self.record_sha256}


def bind(
    declaration: Mapping[str, Any],
    executor: Executor | None,
    verifier: Verifier | None,
) -> ExperimentSpec:
    record = _copy(declaration)
    validate_declaration(record)
    if record["status"] != "historical" and (executor is None or verifier is None):
        raise RecordRefused("a nonhistorical experiment requires execution and verification providers")
    return ExperimentSpec(record, canonical_sha256(record), executor, verifier)


def interpret(spec: ExperimentSpec, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict[str, Any]:
    validate_declaration(spec.declaration)
    if canonical_sha256(spec.declaration) != spec.record_sha256:
        raise RecordRefused("the experiment declaration authority has drifted")
    if spec.executor is None or spec.verifier is None:
        raise RecordRefused(f"experiment {spec.id} is historical and not executable")
    result = spec.executor(cfg, device, run_dir)
    if not isinstance(result, dict):
        raise RecordRefused("execution provider did not return a metric mapping")
    missing_metrics = set(spec.declaration["metrics"]) - result.keys()
    if missing_metrics:
        raise RecordRefused(f"execution provider omitted metrics {sorted(missing_metrics)}")
    verification = spec.verifier(result, spec.declaration)
    if verification.get("verified") is not True:
        raise RecordRefused("verification provider refused the experiment result")
    if verification.get("independent_scientific_confirmation") is not False:
        raise RecordRefused("local execution cannot claim independent scientific confirmation")
    return result


__all__ = [
    "PROGRAM",
    "ExperimentSpec",
    "RecordRefused",
    "bind",
    "interpret",
    "validate_declaration",
]
