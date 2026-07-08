"""Durable artifact index and small-receipt bundle.

Studio waves produce JSON and Markdown receipts under `runs/`, which is ignored by default. This module
turns a list of receipt paths into a hash index and, when requested, copies small text receipts into a
durable bundle directory. Large cache arrays stay out of the bundle; their cache manifests carry hashes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from .transfer_check import DURABLE_RECEIPTS

SCHEMA = "mop-artifact-bundle/v1"
TEXT_EXTS = {".json", ".md", ".txt", ".yaml", ".yml", ".csv", ".tsv"}
PRESETS: dict[str, tuple[str, ...]] = {
    "pre-studio": (
        "docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md",
        *DURABLE_RECEIPTS,
    ),
    "wave0": (
        "docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md",
        "runs/studio_wave0/transfer_check.json",
        "runs/studio_wave0/daemon_state.json",
        "runs/studio_wave0/wave0_report.json",
        "runs/mot/encode_device.json",
        "runs/mot/encode_schedule.json",
    ),
}


def build_artifact_index(
    paths: list[str | Path],
    *,
    repo_root: Path | str = REPO_ROOT,
    copy_dir: Path | str | None = None,
    max_copy_bytes: int = 5_000_000,
    require_durable: bool = False,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Build a receipt index, optionally copying small text artifacts into `copy_dir`."""
    root = Path(repo_root)
    bundle = Path(copy_dir) if copy_dir is not None else None
    artifacts = [_artifact_record(root, raw, bundle, max_copy_bytes=max_copy_bytes) for raw in _dedupe(paths)]
    problems: list[str] = []
    for art in artifacts:
        if not art["exists"] and not allow_missing:
            problems.append(f"missing artifact: {art['path']}")
        if art["exists"] and art["json_ok"] is False:
            problems.append(f"invalid JSON artifact: {art['path']}")
        if require_durable and art["exists"] and not art["durable"]:
            problems.append(f"artifact is not durable: {art['path']}")
        problems.extend(str(p) for p in art.get("problems", []))
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "copy_dir": str(bundle) if bundle is not None else None,
        "max_copy_bytes": int(max_copy_bytes),
        "require_durable": bool(require_durable),
        "allow_missing": bool(allow_missing),
        "artifacts": artifacts,
        "summary": {
            "total": len(artifacts),
            "exists": sum(1 for a in artifacts if a["exists"]),
            "durable": sum(1 for a in artifacts if a["durable"]),
            "copied": sum(1 for a in artifacts if a["copied"]),
            "tracked": sum(1 for a in artifacts if a["git_tracked"]),
            "missing": sum(1 for a in artifacts if not a["exists"]),
        },
        "problems": problems,
        "all_ok": not problems,
    }


def write_artifact_index(index: dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, default=str) + "\n")


def preset_paths(name: str) -> tuple[str, ...]:
    try:
        return PRESETS[name]
    except KeyError as e:
        raise ValueError(f"unknown artifact preset {name!r}; choose from {sorted(PRESETS)}") from e


def _artifact_record(
    root: Path, raw: str | Path, bundle: Path | None, *, max_copy_bytes: int
) -> dict[str, Any]:
    path = _resolve_path(root, raw)
    rel = _rel_display(root, path)
    exists = path.exists() and path.is_file()
    size = path.stat().st_size if exists else None
    sha = _sha256(path) if exists else None
    tracked = _git_tracked(root, path) if exists else False
    json_ok = _json_ok(path) if exists and path.suffix == ".json" else None
    copied = False
    copy_path: str | None = None
    problems: list[str] = []
    if exists and bundle is not None and not tracked:
        if path.suffix not in TEXT_EXTS:
            problems.append(f"not copied, extension {path.suffix!r} is not a small-text receipt")
        elif size is not None and size > max_copy_bytes:
            problems.append(f"not copied, {size} bytes exceeds max_copy_bytes={max_copy_bytes}")
        else:
            dest = bundle / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied = True
            copy_path = str(dest)
    return {
        "path": str(path),
        "display_path": rel,
        "exists": exists,
        "size_bytes": size,
        "sha256": sha,
        "json_ok": json_ok,
        "git_tracked": tracked,
        "copied": copied,
        "copy_path": copy_path,
        "durable": bool(tracked or copied),
        "problems": problems,
    }


def _resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _rel_display(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        safe = path.name or hashlib.sha256(str(path).encode()).hexdigest()[:12]
        return str(Path("external") / safe)


def _json_ok(path: Path) -> bool:
    try:
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_tracked(root: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", str(rel)],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _dedupe(paths: list[str | Path]) -> list[str | Path]:
    seen: set[str] = set()
    out: list[str | Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out
