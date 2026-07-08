"""Studio transfer checklist.

Wave 0 needs a boring proof that the repo arrived with the governing docs, gate scripts, durable
receipts, and profile envelope intact before it spends Studio compute. This module is read-only: it
checks paths, git tracking, JSON/schema parseability, optional cache manifests, and the active profile.
"""

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
DEFAULT_AUDIT_PATH = Path("/Users/scammermike/Downloads/project_audits/mop_deep_audit_2026_07_08.md")

REQUIRED_PATHS = (
    "docs/mixture_of_perspectives/EXPAND_PHASE_PLAN.md",
    "docs/mixture_of_perspectives/STUDIO_POTENTIAL_AUDIT.md",
    "docs/mixture_of_perspectives/STUDIO_GOAL_PROMPT.md",
    "docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md",
    "docs/mixture_of_perspectives/STUDIO_TURNKEY_PLAN.md",
    "scripts/studio_doctor.py",
    "scripts/studio_pipeline.py",
    "scripts/studio_daemon.py",
    "scripts/studio_transfer_check.py",
    "scripts/studio/dr1_smoke.py",
    "scripts/studio/dr1_curate_bound_video.py",
    "scripts/studio/pr9_continual_backprop.py",
    "scripts/mop_encode_autoselect.py",
    "scripts/null_card_tool.py",
    "proof/NULL_CARDS/null_card.schema.json",
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
    """Run the read-only transfer checklist."""
    cfg = config or TransferCheckConfig()
    root = Path(cfg.repo_root)
    checks: list[dict[str, Any]] = []
    checks.append(_profile_check(cfg.profile_name))
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
        profile.name == "studio-m1ultra",
        f"{profile.name}: min_free_disk_gb={profile.min_free_disk_gb:g}, max_wall_min={profile.max_wall_min}",
        profile=profile.as_dict(),
    )


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
    manifests = sorted(cache_root.glob("*/cache_manifest.json")) if cache_root.exists() else []
    problems: list[str] = []
    for manifest in manifests:
        problems.extend(f"{manifest.parent.name}: {p}" for p in validate_cache_manifest(manifest.parent))
    detail = f"{len(manifests)} cache manifests checked"
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
