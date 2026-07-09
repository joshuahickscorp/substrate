"""Studio density and artifact-mass receipt.

The shared Studio 10/10 standard asks for a density receipt: repo size, largest files, artifact mass,
and cleanup deltas. This module is read-only. It does not delete anything; it summarizes the current
workspace and folds in the disk-recovery receipt when one exists.
"""

from __future__ import annotations

import heapq
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT

SCHEMA = "mop-studio-density-receipt/v1"
DEFAULT_DISK_RECOVERY = Path("runs") / "studio_wave0" / "disk_recovery.json"
ARTIFACT_ROOTS = (
    "runs",
    "data/cache",
    "proof/ARTIFACT_BUNDLES",
    "proof/ARTIFACT_INDEX",
    "proof/NULL_CARDS",
)
EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)
LOC_SUFFIXES = frozenset(
    {
        ".py",
        ".md",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
    }
)


@dataclass(frozen=True)
class DensityReceiptConfig:
    repo_root: Path = REPO_ROOT
    disk_recovery_path: Path | str | None = DEFAULT_DISK_RECOVERY
    largest_limit: int = 25
    artifact_roots: tuple[str, ...] = ARTIFACT_ROOTS


def build_density_receipt(config: DensityReceiptConfig | None = None) -> dict[str, Any]:
    """Build a read-only Studio density receipt."""
    cfg = config or DensityReceiptConfig()
    root = Path(cfg.repo_root).resolve()
    limit = max(1, int(cfg.largest_limit))
    inventory = _inventory(root, root, largest_limit=limit)
    cleanup = _cleanup_summary(root, cfg.disk_recovery_path)
    artifact_mass = {
        rel: _inventory(root, root / rel, largest_limit=min(limit, 10)) for rel in cfg.artifact_roots
    }
    loc = _tracked_loc(root)
    after_bytes = int(inventory["total_bytes"])
    deleted_bytes = int(cleanup["deleted_bytes"])
    problems = list(cleanup.get("problems", []))
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "density_kind": "workspace and artifact-mass receipt, not scientific evidence",
        "workspace": inventory,
        "source_loc": loc,
        "artifact_mass": artifact_mass,
        "cleanup": cleanup,
        "before_after": {
            "basis": "current scan plus deleted_bytes from disk recovery receipt when available",
            "repo_bytes_before_known_cleanup": after_bytes + deleted_bytes,
            "repo_bytes_after_known_cleanup": after_bytes,
            "known_cleanup_delta_bytes": deleted_bytes,
            "largest_files_after": inventory["largest_files"],
            "largest_cleanup_targets": cleanup["largest_targets"],
        },
        "problems": problems,
        "all_ok": not problems,
    }


def write_density_receipt(receipt: dict[str, Any], path: Path | str) -> None:
    """Write the density receipt."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")


def _inventory(root: Path, path: Path, *, largest_limit: int) -> dict[str, Any]:
    exists = path.exists()
    total_files = 0
    total_bytes = 0
    largest: list[tuple[int, str]] = []
    if exists:
        for f in _iter_files(path):
            size = _safe_size(f)
            total_files += 1
            total_bytes += size
            item = (size, _display(root, f))
            if len(largest) < largest_limit:
                heapq.heappush(largest, item)
            elif item > largest[0]:
                heapq.heapreplace(largest, item)
    return {
        "path": str(path),
        "display_path": _display(root, path),
        "exists": exists,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_human": _human_bytes(total_bytes),
        "largest_files": [
            {"display_path": display, "size_bytes": size, "size_human": _human_bytes(size)}
            for size, display in sorted(largest, reverse=True)
        ],
    }


def _cleanup_summary(root: Path, raw_path: Path | str | None) -> dict[str, Any]:
    path = None if raw_path is None else _resolve(root, raw_path)
    data = _load_json(path)
    problems: list[str] = []
    if data is not None and data.get("schema") != "mop-disk-recovery-plan/v1":
        problems.append(f"unexpected disk recovery schema {data.get('schema')!r}")
    summary = data.get("summary", {}) if data else {}
    actions = data.get("actions", []) if data else []
    deleted_bytes = int(summary.get("deleted_bytes") or _action_bytes(actions, "deleted"))
    would_delete_bytes = int(summary.get("would_delete_bytes") or _action_bytes(actions, "would_delete"))
    targets = [
        {
            "display_path": str(action.get("display_path") or action.get("path") or ""),
            "kind": action.get("kind"),
            "status": action.get("status"),
            "size_bytes": int(action.get("size_bytes") or 0),
            "size_human": _human_bytes(int(action.get("size_bytes") or 0)),
        }
        for action in actions
        if action.get("status") in {"deleted", "would_delete"}
    ]
    targets.sort(key=lambda item: item["size_bytes"], reverse=True)
    return {
        "path": None if path is None else str(path),
        "exists": data is not None,
        "schema": None if data is None else data.get("schema"),
        "all_ok": None if data is None else bool(data.get("all_ok")),
        "dry_run": None if data is None else bool(data.get("dry_run")),
        "execute_requested": None if data is None else bool(data.get("execute_requested")),
        "deleted_bytes": deleted_bytes,
        "would_delete_bytes": would_delete_bytes,
        "deleted_human": _human_bytes(deleted_bytes),
        "would_delete_human": _human_bytes(would_delete_bytes),
        "safe_bytes": int(summary.get("safe_bytes") or 0),
        "blocked_bytes": int(summary.get("blocked_bytes") or 0),
        "largest_targets": targets[:10],
        "problems": problems,
    }


def _tracked_loc(root: Path) -> dict[str, Any]:
    suffix_counts: dict[str, dict[str, int]] = {}
    total_lines = 0
    total_files = 0
    for rel in _git_ls_files(root):
        path = root / rel
        suffix = path.suffix.lower()
        if suffix not in LOC_SUFFIXES or not path.exists() or not path.is_file():
            continue
        lines = _line_count(path)
        total_lines += lines
        total_files += 1
        rec = suffix_counts.setdefault(suffix or "(none)", {"files": 0, "lines": 0})
        rec["files"] += 1
        rec["lines"] += lines
    return {
        "basis": "git ls-files tracked text/code suffixes",
        "total_files": total_files,
        "total_lines": total_lines,
        "by_suffix": dict(sorted(suffix_counts.items())),
    }


def _iter_files(path: Path):
    if path.is_file() and not path.is_symlink():
        yield path
        return
    if not path.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_DIR_NAMES and not (Path(dirpath) / name).is_symlink()
        ]
        for name in filenames:
            f = Path(dirpath) / name
            if not f.is_symlink():
                yield f


def _git_ls_files(root: Path) -> list[str]:
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [p.decode() for p in raw.split(b"\0") if p]


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _action_bytes(actions: Any, status: str) -> int:
    if not isinstance(actions, list):
        return 0
    return sum(
        int(a.get("size_bytes") or 0) for a in actions if isinstance(a, dict) and a.get("status") == status
    )


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _resolve(root: Path, path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000.0
    return f"{value:.1f} TB"
