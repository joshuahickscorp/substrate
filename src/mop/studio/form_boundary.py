"""Machine-readable pre-Studio boundary for the F1-F20 campaign.

Resource-tier labels are plans, not proof that a laptop was exhausted.  This receipt separates local
obligations, Studio-scale hardware walls, external environment/license blockers, and work explicitly
beyond the proposed Studio.  A Studio-only boundary can become true only from durable receipts; it
cannot be inferred from ``resource_tier: studio-scale`` prose.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from ..devices import apple_silicon_info
from ..falsification.form_evidence import (
    PROOF_ROOT,
    SCORECARD_SCHEMA,
    build_campaign_scorecard,
    load_form_campaign,
)

SCHEMA = "mop-form-pre-studio-boundary/v1"
BOUNDARY_EVIDENCE_SCHEMA = "mop-form-scale-boundary-evidence/v1"
DEFAULT_PATH = PROOF_ROOT / "PRE_STUDIO_BOUNDARY.json"
ARTIFACT_INDEX_PATH = Path("proof/ARTIFACT_INDEX/form_substrate.json")
LIMIT_TYPES = {"memory", "wall_time", "disk", "model_weights", "dataset_scale", "throughput"}


def validate_scale_boundary_evidence(receipt: dict[str, Any], experiment_id: str) -> list[str]:
    """Validate a measured local-to-Studio scale boundary receipt."""
    problems: list[str] = []
    if receipt.get("schema") != BOUNDARY_EVIDENCE_SCHEMA:
        problems.append(f"{experiment_id}: unexpected scale-boundary schema {receipt.get('schema')!r}")
    if receipt.get("experiment_id") != experiment_id:
        problems.append(f"{experiment_id}: scale-boundary receipt names {receipt.get('experiment_id')!r}")
    if receipt.get("local_attempted") is not True:
        problems.append(f"{experiment_id}: local_attempted must be true")
    if receipt.get("limit_type") not in LIMIT_TYPES:
        problems.append(f"{experiment_id}: limit_type must be one of {sorted(LIMIT_TYPES)}")
    measurement = receipt.get("measurement")
    if not isinstance(measurement, dict):
        problems.append(f"{experiment_id}: measurement must be a mapping")
    else:
        for field in ("local_available", "required_or_observed", "unit", "method"):
            if measurement.get(field) in (None, ""):
                problems.append(f"{experiment_id}: measurement missing {field!r}")
    if not str(receipt.get("studio_profile") or "").startswith("studio-"):
        problems.append(f"{experiment_id}: studio_profile must name an explicit Studio profile")
    if not receipt.get("source_receipts") or not isinstance(receipt.get("source_receipts"), list):
        problems.append(f"{experiment_id}: source_receipts must be a non-empty list")
    if not isinstance(receipt.get("full_scale_command"), list) or not receipt.get("full_scale_command"):
        problems.append(f"{experiment_id}: full_scale_command must be a non-empty argv list")
    return problems


def _studio_only_boundary(
    *,
    local_exhausted: bool,
    scientific_ledger_ready: bool,
    verified_studio_boundaries: list[str],
    unproved_studio_boundaries: list[str],
    non_hardware_blockers: list[dict[str, Any]],
    beyond_studio: list[str],
) -> bool:
    """Return true only for an evidenced, exclusive Studio hardware boundary.

    Absence of an unproved Studio claim is not proof of a Studio boundary.  At least one measured
    and validated Studio boundary must exist, the scientific ledger must be ready, and no data,
    rights, environment, or beyond-Studio blocker may remain.  This prevents a fully local or
    externally blocked campaign from satisfying the hardware conclusion vacuously.
    """

    return bool(
        local_exhausted
        and scientific_ledger_ready
        and verified_studio_boundaries
        and not unproved_studio_boundaries
        and not non_hardware_blockers
        and not beyond_studio
    )


def build_form_pre_studio_boundary(
    *, repo_root: Path | str = REPO_ROOT, scorecard: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Classify every F-series leg and decide whether Studio is the sole remaining hardware wall."""
    root = Path(repo_root)
    campaign = load_form_campaign(repo_root=root)
    score = scorecard or build_campaign_scorecard(repo_root=root)
    score_states = {str(state["experiment_id"]): state for state in score.get("legs", [])}
    classifications: list[dict[str, Any]] = []
    non_hardware_blockers: list[dict[str, Any]] = []
    verified_studio_boundaries: list[str] = []
    unproved_studio_boundaries: list[str] = []
    beyond_studio: list[str] = []
    for leg in campaign.get("legs", []):
        eid = str(leg["id"])
        state = score_states.get(eid, {})
        local_ready = bool(state.get("all_ok"))
        scale = str(leg.get("scale_boundary"))
        boundary_path = root / PROOF_ROOT / "BOUNDARY_EVIDENCE" / f"{eid}.json"
        boundary_receipt = _load_json(boundary_path)
        boundary_problems: list[str] = []
        boundary_verified = False
        non_hardware_inputs = list(leg.get("non_hardware_inputs") or [])
        if non_hardware_inputs:
            non_hardware_blockers.append(
                {
                    "experiment_id": eid,
                    "kind": "license-or-data",
                    "inputs": non_hardware_inputs,
                    "local_ready": local_ready,
                }
            )
        if scale in {"studio", "beyond-studio"}:
            if boundary_receipt is None:
                boundary_problems.append(f"{eid}: measured scale-boundary receipt missing")
            else:
                boundary_problems.extend(validate_scale_boundary_evidence(boundary_receipt, eid))
            boundary_verified = not boundary_problems
        if scale == "local":
            classification = "local-complete" if local_ready else "local-work-remaining"
        elif scale == "studio":
            if not local_ready:
                classification = "local-scaffold-or-run-remaining"
            elif boundary_verified:
                classification = "studio-scale-only"
                verified_studio_boundaries.append(eid)
            else:
                classification = "studio-scale-claim-unproved"
                unproved_studio_boundaries.append(eid)
        elif scale == "environment":
            classification = "external-environment-scale" if local_ready else "local-smoke-remaining"
            if not non_hardware_inputs:
                non_hardware_blockers.append(
                    {
                        "experiment_id": eid,
                        "kind": "environment",
                        "inputs": [],
                        "local_ready": local_ready,
                    }
                )
        else:
            classification = "beyond-studio-scope" if local_ready else "local-preflight-remaining"
            beyond_studio.append(eid)
            if not non_hardware_inputs:
                non_hardware_blockers.append(
                    {
                        "experiment_id": eid,
                        "kind": "beyond-studio",
                        "inputs": [],
                        "local_ready": local_ready,
                    }
                )
        classifications.append(
            {
                "experiment_id": eid,
                "local_requirement": leg.get("local_requirement"),
                "local_ready": local_ready,
                "scale_boundary": scale,
                "classification": classification,
                "boundary_evidence_path": str(boundary_path.relative_to(root)),
                "boundary_evidence_verified": boundary_verified,
                "boundary_evidence_problems": boundary_problems,
                "non_hardware_inputs": non_hardware_inputs,
                "future_extension_inputs": list(leg.get("future_extension_inputs") or []),
            }
        )

    artifact_index = _load_json(root / ARTIFACT_INDEX_PATH)
    artifact_index_ok = bool(
        artifact_index
        and artifact_index.get("schema") == "mop-artifact-bundle/v1"
        and artifact_index.get("all_ok")
    )
    local_exhausted = bool(score.get("local_obligations_exhausted"))
    scientific_ledger_ready = bool(score.get("scientific_ledger_ready"))
    studio_only = _studio_only_boundary(
        local_exhausted=local_exhausted,
        scientific_ledger_ready=scientific_ledger_ready,
        verified_studio_boundaries=verified_studio_boundaries,
        unproved_studio_boundaries=unproved_studio_boundaries,
        non_hardware_blockers=non_hardware_blockers,
        beyond_studio=beyond_studio,
    )
    boundary_reasons: list[str] = []
    if not local_exhausted:
        boundary_reasons.append("local obligations remain")
    if not scientific_ledger_ready:
        boundary_reasons.append("scientific verdict ledger is not ready")
    if not verified_studio_boundaries:
        boundary_reasons.append("no measured and validated Studio hardware boundary exists")
    if unproved_studio_boundaries:
        boundary_reasons.append("one or more Studio boundary claims are unproved")
    if non_hardware_blockers:
        boundary_reasons.append("data, rights, environment, or provenance blockers remain")
    if beyond_studio:
        boundary_reasons.append("one or more campaign legs are beyond Studio scope")
    disk = shutil.disk_usage(root)
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "F1-F20 form-substrate campaign; this is not a project-wide boundary claim",
        "repo_root": str(root),
        "host": {
            **apple_silicon_info(),
            "disk_total_gb": round(disk.total / 1e9, 3),
            "disk_free_gb": round(disk.free / 1e9, 3),
        },
        "scorecard": {
            "schema": score.get("schema"),
            "valid_schema": score.get("schema") == SCORECARD_SCHEMA,
            "local_obligations_exhausted": local_exhausted,
            "scientific_ledger_ready": scientific_ledger_ready,
        },
        "classifications": classifications,
        "non_hardware_blockers": non_hardware_blockers,
        "verified_studio_boundaries": verified_studio_boundaries,
        "unproved_studio_boundaries": unproved_studio_boundaries,
        "beyond_studio_scope": beyond_studio,
        "artifact_index": {
            "path": str(ARTIFACT_INDEX_PATH),
            "all_ok": artifact_index_ok,
        },
        "local_resources_exhausted": local_exhausted,
        "studio_is_only_remaining_hardware_boundary": studio_only,
        "boundary_decision_reasons": boundary_reasons,
        "ready_for_studio_handoff": bool(studio_only and artifact_index_ok),
        "all_ok": bool(studio_only and artifact_index_ok),
    }


def write_form_pre_studio_boundary(receipt: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None
