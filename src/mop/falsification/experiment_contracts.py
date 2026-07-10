"""Exact experiment-contract equality checks.

The experiment bank is the preregistration source of truth, but a runnable experiment also repeats
part of that contract in its class and its composed config.  A run is not canonical when those three
surfaces disagree, even if the code executes and emits all expected keys.  This module makes that
rule explicit and reusable by registry validation, the harness doctor, and durable result promotion.

Only fields that have the same meaning on all three surfaces are compared: id, ordered metric list,
null hypothesis, and run tier.  ``controls``/``falsifier`` are intentionally not compared to the
class ``baseline``/``ablation`` prose because those pairs are related but not equivalent schemas.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..config import REPO_ROOT

SCHEMA = "mop-experiment-contract-audit/v1"
CONTRACT_FIELDS = ("id", "metric", "null_hypothesis", "tier")


def compare_contract_sources(
    registry_row: dict[str, Any],
    experiment_class: type,
    config: DictConfig | dict[str, Any],
) -> dict[str, Any]:
    """Compare one registry row, runnable class, and experiment config exactly."""
    eid = str(registry_row.get("id") or "<missing-id>")
    cfg = _plain(config)
    registry_values = {
        "id": registry_row.get("id"),
        "metric": list(registry_row.get("metrics") or []),
        "null_hypothesis": registry_row.get("null_hypothesis"),
        "tier": registry_row.get("exp_tier"),
    }
    class_values = {
        "id": getattr(experiment_class, "id", None),
        "metric": list(getattr(experiment_class, "metric", ()) or ()),
        "null_hypothesis": getattr(experiment_class, "null_hypothesis", None),
        "tier": getattr(experiment_class, "tier", None),
    }
    config_values = {
        "id": cfg.get("id"),
        "metric": list(cfg.get("metric") or []),
        "null_hypothesis": cfg.get("null_hypothesis"),
        "tier": cfg.get("tier"),
    }
    comparisons: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for field in CONTRACT_FIELDS:
        values = {
            "registry": registry_values[field],
            "class": class_values[field],
            "config": config_values[field],
        }
        equal = values["registry"] == values["class"] == values["config"]
        comparisons[field] = {**values, "equal": equal}
        if not equal:
            problems.append(
                f"{eid}: contract {field} differs across registry/class/config: "
                f"registry={values['registry']!r}, class={values['class']!r}, "
                f"config={values['config']!r}"
            )
    return {
        "experiment_id": eid,
        "required": registry_row.get("status") == "implemented",
        "comparisons": comparisons,
        "canonical": registry_values,
        "problems": problems,
        "all_ok": not problems,
    }


def build_contract_audit(
    *,
    repo_root: Path | str = REPO_ROOT,
    series: str | None = "F",
    implemented_only: bool = True,
) -> dict[str, Any]:
    """Audit live runnable contracts against their registry rows and config files.

    Missing class/config sources are explicit failures for implemented rows.  Registry-only rows are
    omitted by default because they do not yet have a runnable contract to compare.
    """
    from ..devel.registries import load_experiments
    from ..experiments import REGISTRY

    root = Path(repo_root)
    records: list[dict[str, Any]] = []
    for row in load_experiments(root / "registry" / "experiments.yaml"):
        if series is not None and str(row.get("series")) != series:
            continue
        if implemented_only and row.get("status") != "implemented":
            continue
        eid = str(row.get("id") or "")
        cls = REGISTRY.get(eid)
        cfg_path = root / "configs" / "experiment" / f"{eid}.yaml"
        if cls is None or not cfg_path.exists():
            missing: list[str] = []
            if cls is None:
                missing.append("runnable class")
            if not cfg_path.exists():
                missing.append(f"config {cfg_path}")
            records.append(
                {
                    "experiment_id": eid,
                    "required": row.get("status") == "implemented",
                    "comparisons": {},
                    "canonical": _canonical_from_row(row),
                    "problems": [f"{eid}: missing {' and '.join(missing)}"],
                    "all_ok": False,
                }
            )
            continue
        loaded_config = OmegaConf.load(cfg_path)
        if not isinstance(loaded_config, DictConfig):
            records.append(
                {
                    "experiment_id": eid,
                    "required": True,
                    "comparisons": {},
                    "canonical": _canonical_from_row(row),
                    "problems": [f"{eid}: config {cfg_path} must be a mapping"],
                    "all_ok": False,
                }
            )
            continue
        records.append(compare_contract_sources(row, cls, loaded_config))
    problems = [problem for record in records for problem in record["problems"]]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "series": series,
        "implemented_only": bool(implemented_only),
        "records": records,
        "summary": {
            "total": len(records),
            "aligned": sum(1 for record in records if record["all_ok"]),
            "misaligned": sum(1 for record in records if not record["all_ok"]),
        },
        "problems": problems,
        "all_ok": not problems,
    }


def validate_manifest_contract(
    manifest: dict[str, Any],
    contract_record: dict[str, Any],
    run_config: DictConfig | dict[str, Any],
) -> list[str]:
    """Reject a run whose embedded/snapshotted contract is stale relative to the live audit."""
    eid = str(contract_record.get("experiment_id") or "<missing-id>")
    problems = list(contract_record.get("problems") or [])
    canonical = dict(contract_record.get("canonical") or {})
    embedded = dict((manifest.get("extra") or {}).get("contract") or {})
    embedded_normalized = {
        "id": embedded.get("id"),
        "metric": list(embedded.get("metric") or []),
        "null_hypothesis": embedded.get("null_hypothesis"),
        "tier": embedded.get("tier"),
    }
    snapshot = _plain(run_config)
    snapshot_normalized = {
        "id": snapshot.get("id"),
        "metric": list(snapshot.get("metric") or []),
        "null_hypothesis": snapshot.get("null_hypothesis"),
        "tier": snapshot.get("tier"),
    }
    if str(manifest.get("name") or "") != eid:
        problems.append(f"{eid}: manifest name is {manifest.get('name')!r}")
    for label, values in (("manifest", embedded_normalized), ("run config", snapshot_normalized)):
        for field in CONTRACT_FIELDS:
            if values.get(field) != canonical.get(field):
                problems.append(
                    f"{eid}: {label} {field} is stale: {values.get(field)!r} != "
                    f"canonical {canonical.get(field)!r}"
                )
    return problems


def write_contract_audit(audit: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, default=str) + "\n")


def _canonical_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "metric": list(row.get("metrics") or []),
        "null_hypothesis": row.get("null_hypothesis"),
        "tier": row.get("exp_tier"),
    }


def _plain(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        data = OmegaConf.to_container(config, resolve=True)
        return {str(key): value for key, value in data.items()} if isinstance(data, dict) else {}
    return dict(config)
