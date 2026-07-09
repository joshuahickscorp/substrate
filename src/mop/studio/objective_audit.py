"""Studio objective audit.

This receipt is a conservative, point-by-point audit of the active Studio 10/10 objective. It does not
award scientific credit for local launch prep. Instead it names which requirement points are proved by
current receipts, which are only launch-prepared, and which still need M1 Ultra evidence.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT

SCHEMA = "mop-studio-objective-audit/v1"
DEFAULT_PATHS = {
    "transfer": "runs/studio_wave0/transfer_check.json",
    "wave0_report": "runs/studio_wave0/wave0_report.json",
    "spine_plan": "runs/studio_spine/spine_plan.json",
    "spine_status": "runs/studio_spine/spine_status_local.json",
    "dr1_verification": "data/cache/vjepa2_vitl_comp_video/dr1_verification.json",
    "pr9_verdict": "runs/mot/pr9_verdict_ledger.json",
    "dense_gate": "runs/mot/dense_atlas_cache_gate.json",
    "atlas_verdict": "runs/mot/atlas_verdict_ledger.json",
    "process_c_gate": "runs/mot/process_c_license_gate.json",
    "scorecard": "runs/studio_scorecard_local.json",
    "density_receipt": "runs/studio_wave0/density_receipt.json",
    "native_lanes": "runs/studio_native_lanes_manifest.json",
    "wave0_index": "proof/ARTIFACT_INDEX/wave0.json",
    "dr1_index": "proof/ARTIFACT_INDEX/dr1.json",
    "pr9_index": "proof/ARTIFACT_INDEX/pr9.json",
    "atlas_index": "proof/ARTIFACT_INDEX/atlas.json",
    "spine_index": "proof/ARTIFACT_INDEX/spine.json",
}


def build_studio_objective_audit(
    *,
    repo_root: Path | str = REPO_ROOT,
    paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a point-by-point audit of the Studio objective from local receipts."""
    root = Path(repo_root)
    resolved = {**DEFAULT_PATHS, **{k: str(v) for k, v in (paths or {}).items()}}
    receipts = {name: _load_json(root / rel) for name, rel in resolved.items()}
    requirements = [
        _launch_prep(receipts),
        _dr1_real_video(receipts),
        _pr9_long_stream(receipts),
        _dense_atlas(receipts),
        _process_c(receipts),
        _adversarial_and_nulls(root),
        _durability(root, receipts),
        _native_lanes(root, receipts),
    ]
    summary = {
        "points_possible": sum(float(r["point_value"]) for r in requirements),
        "points_earned": round(sum(float(r["credit"]) for r in requirements), 3),
        "complete": sum(1 for r in requirements if r["status"] == "complete"),
        "prepared": sum(1 for r in requirements if r["status"] == "prepared"),
        "blocked": sum(1 for r in requirements if r["status"] == "blocked"),
        "pending": sum(1 for r in requirements if r["status"] == "pending"),
        "failed": sum(1 for r in requirements if r["status"] == "failed"),
    }
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "score_kind": "objective checklist credit, not a scientific score",
        "studio_10_ready": bool(summary["points_earned"] == summary["points_possible"]),
        "summary": summary,
        "requirements": requirements,
        "receipt_paths": resolved,
    }


def load_json(path: Path | str | None) -> dict[str, Any] | None:
    """Load a JSON object if it exists."""
    if path is None:
        return None
    return _load_json(Path(path))


def write_studio_objective_audit(audit: dict[str, Any], path: Path | str) -> None:
    """Write the objective-audit receipt."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, default=str) + "\n")


def _launch_prep(receipts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    checks = [
        _check("transfer_check", _all_ok(receipts["transfer"]), "Wave 0 transfer check all_ok"),
        _check("wave0_report", _all_ok(receipts["wave0_report"]), "Wave 0 report all_ok"),
        _check(
            "spine_plan", _schema(receipts["spine_plan"], "mop-studio-spine-plan/v1"), "spine plan exists"
        ),
        _check(
            "spine_status",
            _schema(receipts["spine_status"], "mop-studio-spine-status/v1"),
            "spine status exists",
        ),
    ]
    return _requirement(
        "wave0_launch_prep",
        "Wave 0 launch prep and spine contract",
        checks,
        complete_detail="Wave 0 receipts and staged spine are complete",
        prepared_detail="transfer/spine prep exists but Studio Wave 0 still has missing execution receipts",
    )


def _dr1_real_video(receipts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    dr1 = receipts["dr1_verification"]
    checks = [
        _check("dr1_verifier_schema", _schema(dr1, "mop-dr1-adversarial-verification/v1"), "DR1 verifier"),
        _check("dr1_integrity", bool((dr1 or {}).get("integrity_ok")), "DR1 artifact integrity"),
        _check("dr1_a6_decision", isinstance((dr1 or {}).get("a6_survives"), bool), "A6 decision exists"),
    ]
    return _requirement(
        "dr1_real_bound_video",
        "DR1 real bound-attribute video with adversarial verification",
        checks,
        complete_detail="DR1 real-video verifier is decisive",
        prepared_detail="DR1 launch gates exist, but decisive real-video verifier evidence is missing",
    )


def _pr9_long_stream(receipts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    pr9 = receipts["pr9_verdict"]
    checks = [
        _check("pr9_verdict_schema", _schema(pr9, "mop-pr9-verdict-ledger/v1"), "PR9 verdict ledger"),
        _check("pr9_verdict_all_ok", _all_ok(pr9), "PR9 verdict all_ok"),
        _check(
            "pr9_dr1_cache",
            str((pr9 or {}).get("dr1_cache") or "") == "data/cache/vjepa2_vitl_comp_video",
            "PR9 evaluated the DR1 real cache",
        ),
    ]
    return _requirement(
        "pr9_long_stream_plasticity",
        "PR9 long-stream plasticity with run-state and verdict ledger",
        checks,
        complete_detail="PR9 long-stream verdict is complete on the DR1 cache",
        prepared_detail=(
            "PR9 run-state/verdict machinery exists, but Studio PR9 evidence is missing or non-scoring"
        ),
    )


def _dense_atlas(receipts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    checks = [
        _check(
            "dense_gate",
            _schema(receipts["dense_gate"], "mop-dense-atlas-cache-gate/v1")
            and _all_ok(receipts["dense_gate"]),
            "dense real/control cache gate all_ok",
        ),
        _check(
            "atlas_verdict",
            _schema(receipts["atlas_verdict"], "mop-atlas-verdict-ledger/v1")
            and _all_ok(receipts["atlas_verdict"]),
            "atlas verdict all_ok",
        ),
    ]
    return _requirement(
        "dense_cache_and_atlas",
        "Dense cache plus full multi-encoder atlas",
        checks,
        complete_detail="dense cache gate and atlas verdict are scoring",
        prepared_detail=(
            "dense/atlas gates exist, but real/control dense caches or atlas evidence are missing"
        ),
    )


def _process_c(receipts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    gate = receipts["process_c_gate"]
    decisive = _schema(gate, "mop-process-c-license-gate/v1") and _all_ok(gate)
    checks = [
        _check("process_c_gate_schema", _schema(gate, "mop-process-c-license-gate/v1"), "Process C gate"),
        _check("process_c_decisive", decisive, "Process C gate is decisive"),
    ]
    return _requirement(
        "process_c_authorization",
        "Process C only if licensed by PR9 or DR1 evidence",
        checks,
        complete_detail=f"Process C decision is {(gate or {}).get('status')}",
        prepared_detail="Process C gate exists but is not decisive without Studio PR9/DR1 receipts",
    )


def _adversarial_and_nulls(root: Path) -> dict[str, Any]:
    required = (
        "proof/NULL_CARDS/mop_dr1_video_cache.md",
        "proof/NULL_CARDS/pr9_long_stream_plasticity.md",
        "proof/NULL_CARDS/atlas_dense_multiencoder.md",
        "proof/NULL_CARDS/process_c_dense_token_pilot.md",
        "scripts/verdict_gate.py",
        "scripts/studio/dr1_verify.py",
    )
    checks = [_check(Path(path).name, (root / path).exists(), path) for path in required]
    return _requirement(
        "adversarial_verification_and_null_cards",
        "Adversarial verification and null-card preservation",
        checks,
        complete_detail="core null cards and verifier gates are present",
        prepared_detail="some verifier/null-card artifacts are missing",
    )


def _durability(root: Path, receipts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    index_names = ("wave0_index", "dr1_index", "pr9_index", "atlas_index", "spine_index")
    checks = [
        _check(
            "studio_run_report",
            (root / "docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md").exists(),
            "run report",
        ),
        _check(
            "density_receipt",
            _schema(receipts["density_receipt"], "mop-studio-density-receipt/v1")
            and _all_ok(receipts["density_receipt"]),
            "density/artifact-mass receipt",
        ),
        _check("scorecard", _schema(receipts["scorecard"], "mop-studio-scorecard/v1"), "scorecard receipt"),
        *[
            _check(name, _schema(receipts[name], "mop-artifact-bundle/v1") and _all_ok(receipts[name]), name)
            for name in index_names
        ],
    ]
    return _requirement(
        "durable_artifacts_and_reports",
        "Durable artifact indexes and Studio run reports",
        checks,
        complete_detail="run report, scorecard, and all artifact indexes are durable",
        prepared_detail="run report exists, but one or more artifact indexes are missing or incomplete",
    )


def _native_lanes(root: Path, receipts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    manifest = receipts["native_lanes"]
    cli_exists = (root / "scripts/studio/__main__.py").exists() or (
        root / "scripts/studio_native_lanes.py"
    ).exists()
    checks = [
        _check("native_lane_cli", cli_exists, "native lane CLI"),
        _check("native_lane_manifest", _schema(manifest, "mop-studio-native-lanes/v1"), "native manifest"),
        _check(
            "native_lanes_accounted",
            bool((manifest or {}).get("lanes")),
            "native lanes have ready/blocked records",
        ),
    ]
    return _requirement(
        "studio_native_lanes",
        "Studio-native frontier lanes are stood up or walled",
        checks,
        complete_detail="native lane manifest records ready/blocked states",
        prepared_detail="native lane CLI exists, but a current manifest receipt is missing",
    )


def _requirement(
    req_id: str,
    title: str,
    checks: list[dict[str, Any]],
    *,
    complete_detail: str,
    prepared_detail: str,
) -> dict[str, Any]:
    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    if passed == total:
        status = "complete"
        credit = 1.0
        detail = complete_detail
    elif passed > 0:
        status = "prepared"
        credit = round(0.5 * passed / total, 3)
        detail = prepared_detail
    else:
        status = "pending"
        credit = 0.0
        detail = prepared_detail
    return {
        "id": req_id,
        "title": title,
        "status": status,
        "point_value": 1.0,
        "credit": credit,
        "detail": detail,
        "checks": checks,
    }


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _schema(data: dict[str, Any] | None, schema: str) -> bool:
    return isinstance(data, dict) and data.get("schema") == schema


def _all_ok(data: dict[str, Any] | None) -> bool:
    return isinstance(data, dict) and bool(data.get("all_ok"))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None
