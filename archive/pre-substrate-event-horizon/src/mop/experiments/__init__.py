from __future__ import annotations

from typing import Any

import yaml

from ..config import REPO_ROOT
from .base import ExperimentSpec, RecordRefused, bind, interpret


def _registry_rows() -> list[dict[str, Any]]:
    payload = yaml.safe_load((REPO_ROOT / "registry/experiments.yaml").read_text())
    rows = payload.get("experiments") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RecordRefused("registry/experiments.yaml must contain experiment mappings")
    return rows


_ROWS = _registry_rows()
if any(row.get("status") != "historical" for row in _ROWS):
    raise RecordRefused("a maintained experiment declaration must be historical or have a provider binding")
REGISTRY = {row["id"]: bind(row, None, None) for row in _ROWS}


def get_experiment(experiment_id: str) -> ExperimentSpec:
    if experiment_id not in REGISTRY:
        raise KeyError(f"unknown experiment {experiment_id!r}; have {sorted(REGISTRY)}")
    return REGISTRY[experiment_id]


__all__ = ["ExperimentSpec", "REGISTRY", "RecordRefused", "get_experiment", "interpret"]
