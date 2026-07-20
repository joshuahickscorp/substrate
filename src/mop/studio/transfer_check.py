from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from ..substrate.cache_manifest import validate_cache_manifest
from .profiles import get_profile

SCHEMA = "mop-studio-transfer-check/v1"
DEFAULT_AUDIT_PATH = REPO_ROOT / "FORM_SUBSTRATE_IMPLEMENTATION_PLAN.md"

REQUIRED_PATHS = (
    "docs/mixture_of_perspectives/EXPAND_PHASE_PLAN.md",
    "docs/mixture_of_perspectives/STUDIO_POTENTIAL_AUDIT.md",
    "docs/mixture_of_perspectives/STUDIO_GOAL_PROMPT.md",
    "docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md",
    "docs/mixture_of_perspectives/STUDIO_TURNKEY_PLAN.md",
    "scripts/studio/__main__.py",
    "scripts/studio_pipeline.py",
    "scripts/mop_dr13_predictor_fidelity.py",
    "scripts/mop_dr13_readout_adapter.py",
    "scripts/studio/dr1_smoke.py",
    "scripts/studio/dr1_source_card.py",
    "scripts/studio/dr1_source_intake.py",
    "scripts/studio/dr1_schedule_plan.py",
    "scripts/studio/dr1_curate_bound_video.py",
    "scripts/studio/dr1_verify.py",
    "scripts/studio/dense_atlas_gate.py",
    "scripts/studio/atlas_multi_encoder_grid.py",
    "scripts/studio/atlas_verdict_ledger.py",
    "scripts/studio/pr9_continual_backprop.py",
    "scripts/studio/pr9_verdict_ledger.py",
    "scripts/studio/process_c_license_gate.py",
    "scripts/mop_encode_autoselect.py",
    "scripts/null_card_tool.py",
    "scripts/form_substrate_campaign.py",
    "proof/NULL_CARDS/null_card.schema.json",
    "proof/ARTIFACT_INDEX/pre_studio.json",
    "campaign/form_substrate_campaign.yaml",
    "proof/FORM_SUBSTRATE/README.md",
    "proof/FORM_SUBSTRATE/SCORECARD.json",
    "proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json",
    "proof/ARTIFACT_INDEX/form_substrate.json",
)

DURABLE_RECEIPTS = (
    "runs/pre_studio/RESULTS_PRE_STUDIO.md",
    "runs/pre_studio/close_b5_degeneracy.json",
    "runs/pre_studio/close_e7_sparse.json",
    "runs/pre_studio/close_ex2_planning.json",
    "runs/pre_studio/close_ex5_local_rules.json",
    "runs/pre_studio/frozen_random_census.json",
    "runs/pre_studio/census_reaudit.json",
    "runs/mot/dr13_predictor_fidelity.json",
)


@dataclass(frozen=True)
class TransferCheckConfig:
    repo_root: Path = REPO_ROOT
    audit_path: Path | None = DEFAULT_AUDIT_PATH
    profile_name: str = "studio-m1ultra"
    allow_dirty: bool = False
    require_receipts: bool = True


def run_transfer_check(config: TransferCheckConfig | None = None) -> dict[str, Any]:
    cfg = config or TransferCheckConfig()
    root = Path(cfg.repo_root)
    checks: list[dict[str, Any]] = []
    checks.append(_profile_check(cfg.profile_name))
    checks.append(_profile_host_check(cfg.profile_name, root))
    checks.append(_audit_check(cfg.audit_path))
    checks.extend(_path_checks(root, REQUIRED_PATHS))
    checks.append(_schema_json_check(root / "proof/NULL_CARDS/null_card.schema.json"))
    checks.append(_git_check(root, allow_dirty=cfg.allow_dirty))
    if cfg.require_receipts:
        checks.extend(_receipt_checks(root, DURABLE_RECEIPTS))
    checks.append(_cache_manifest_check(root / "data" / "cache"))

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "profile": cfg.profile_name,
        "checks": checks,
    }
    report["summary"] = {
        "total": len(checks),
        "passed": sum(1 for c in checks if c["ok"]),
        "failed": sum(1 for c in checks if not c["ok"]),
    }
    report["all_ok"] = report["summary"]["failed"] == 0
    return report


def write_transfer_report(report: dict[str, Any], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str) + "\n")


def _profile_check(profile_name: str) -> dict[str, Any]:
    try:
        profile = get_profile(profile_name)
    except Exception as e:
        return _check("profile", False, str(e))
    return _check(
        "profile",
        profile.name.startswith("studio-"),
        f"{profile.name}: min_free_disk_gb={profile.min_free_disk_gb:g}, max_wall_min={profile.max_wall_min}",
        profile=profile.as_dict(),
    )


def _profile_host_check(profile_name: str, root: Path) -> dict[str, Any]:
    try:
        profile = get_profile(profile_name)
        ok, problems, measured = profile.host_compatibility(disk_root=root)
    except Exception as e:
        return _check("profile_host_match", False, str(e))
    detail = f"measured={measured}; requirements={profile.as_dict()}"
    if problems:
        detail += f"; mismatch={problems}"
    return _check("profile_host_match", ok, detail, measured=measured, problems=problems)


def _audit_check(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _check("governing_audit", True, "not required by this invocation")
    ok = Path(path).exists()
    return _check("governing_audit", ok, str(path))


def _path_checks(root: Path, rels: tuple[str, ...]) -> list[dict[str, Any]]:
    return [_check(f"path:{rel}", (root / rel).exists(), rel) for rel in rels]


def _schema_json_check(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return _check("null_card_schema_json", False, str(e))
    return _check("null_card_schema_json", data.get("$schema") is not None, str(path))


def _git_check(root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    try:
        head = _git(root, "rev-parse", "--short", "HEAD")
        branch = _git(root, "branch", "--show-current")
        dirty = bool(_git(root, "status", "--short"))
    except Exception as e:
        return _check("git_state", False, str(e))
    ok = allow_dirty or not dirty
    detail = f"branch={branch or '(detached)'}, head={head}, dirty={dirty}, allow_dirty={allow_dirty}"
    return _check("git_state", ok, detail, branch=branch, head=head, dirty=dirty)


def _receipt_checks(root: Path, rels: tuple[str, ...]) -> list[dict[str, Any]]:
    checks = []
    for rel in rels:
        path = root / rel
        exists = path.exists()
        tracked = _git_tracked(root, rel) if exists else False
        checks.append(_check(f"receipt:{rel}", exists and tracked, f"exists={exists}, tracked={tracked}"))
    return checks


def _cache_manifest_check(cache_root: Path) -> dict[str, Any]:
    stores = (
        sorted(path for path in cache_root.iterdir() if path.is_dir() and (path / "meta.json").exists())
        if cache_root.exists()
        else []
    )
    problems: list[str] = []
    for store in stores:
        problems.extend(f"{store.name}: {p}" for p in validate_cache_manifest(store, citable=True))
    if not stores:
        problems.append("no latent stores found to transfer")
    detail = f"{len(stores)} cache stores checked for citable manifests"
    if problems:
        detail += f"; first problem: {problems[0]}"
    return _check("cache_manifests", not problems, detail, problems=problems)


def _git_tracked(root: Path, rel: str) -> bool:
    try:
        _git(root, "ls-files", "--error-unmatch", rel)
    except subprocess.CalledProcessError:
        return False
    return True


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def _check(name: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    out = {"name": name, "ok": bool(ok), "detail": detail}
    out.update(extra)
    return out
