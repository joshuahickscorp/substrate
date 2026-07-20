
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from .artifact_bundle import TEXT_EXTS
from .profiles import get_profile

SCHEMA = "mop-disk-recovery-plan/v1"
SAFE_CLASSES = frozenset({"repo_tool_cache", "tmp_mop", "repo_ignored_run"})
_TOOL_CACHE_NAMES = frozenset({".pytest_cache", ".ruff_cache", ".mypy_cache"})


@dataclass(frozen=True)
class DiskRecoveryConfig:
    repo_root: Path = REPO_ROOT
    profile_name: str = "m3pro-local-max"
    scan_paths: tuple[Path | str, ...] = ()
    include_defaults: bool = True
    execute: bool = False
    allow_classes: tuple[str, ...] = ()
    allow_paths: tuple[Path | str, ...] = ()
    max_receipts: int = 50


def build_disk_recovery_plan(config: DiskRecoveryConfig | None = None) -> dict[str, Any]:
    cfg = config or DiskRecoveryConfig()
    root = Path(cfg.repo_root).resolve()
    profile = get_profile(cfg.profile_name)
    free_ok, free_gb = profile.free_disk_ok(root)
    candidates = [
        _candidate_record(root, p, max_receipts=cfg.max_receipts)
        for p in _candidate_paths(root, cfg.scan_paths, cfg.include_defaults)
    ]
    allow_classes = tuple(dict.fromkeys(str(c) for c in cfg.allow_classes))
    allow_paths = tuple(_resolve(root, p).resolve() for p in cfg.allow_paths)
    problems: list[str] = []
    if cfg.execute and not allow_classes and not allow_paths:
        problems.append("execute requires at least one --allow-class or --allow-path")
    actions = _actions(
        root,
        candidates,
        execute=cfg.execute and not problems,
        allow_classes=allow_classes,
        allow_paths=allow_paths,
    )
    deleted_bytes = sum(a["size_bytes"] or 0 for a in actions if a["status"] == "deleted")
    would_delete_bytes = sum(a["size_bytes"] or 0 for a in actions if a["status"] == "would_delete")
    failed = [a for a in actions if a["status"] == "delete_failed"]
    problems.extend(f"{a['display_path']}: {a['error']}" for a in failed)
    safe = [c for c in candidates if c["safe_to_delete"]]
    blocked = [c for c in candidates if not c["safe_to_delete"]]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "profile": profile.as_dict(),
        "free_disk": {
            "free_gb": round(free_gb, 3),
            "floor_gb": profile.min_free_disk_gb,
            "ok": free_ok,
        },
        "dry_run": not cfg.execute,
        "execute_requested": bool(cfg.execute),
        "allow_classes": list(allow_classes),
        "allow_paths": [str(p) for p in allow_paths],
        "candidates": candidates,
        "actions": actions,
        "summary": {
            "candidate_count": len(candidates),
            "safe_count": len(safe),
            "blocked_count": len(blocked),
            "safe_bytes": sum(c["size_bytes"] or 0 for c in safe),
            "blocked_bytes": sum(c["size_bytes"] or 0 for c in blocked),
            "would_delete_bytes": would_delete_bytes,
            "deleted_bytes": deleted_bytes,
            "would_delete_human": _human_bytes(would_delete_bytes),
            "deleted_human": _human_bytes(deleted_bytes),
        },
        "problems": problems,
        "all_ok": not problems,
    }


def write_disk_recovery_plan(plan: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, default=str) + "\n")


def default_candidate_paths(repo_root: Path | str = REPO_ROOT) -> list[Path]:
    root = Path(repo_root).resolve()
    out: list[Path] = []
    for name in sorted(_TOOL_CACHE_NAMES):
        p = root / name
        if p.exists():
            out.append(p)
    e1 = root / "runs" / "e1_baseline"
    if e1.exists():
        out.extend(p for p in sorted(e1.iterdir()) if p.is_dir())
    tmp = Path(tempfile.gettempdir())
    out.extend(p for p in sorted(tmp.glob("mop_*")) if p.exists())
    return _dedupe_paths(out)


def _candidate_paths(root: Path, scan_paths: tuple[Path | str, ...], include_defaults: bool) -> list[Path]:
    paths: list[Path] = []
    if include_defaults:
        paths.extend(default_candidate_paths(root))
    paths.extend(_resolve(root, p) for p in scan_paths)
    return _dedupe_paths(paths)


def _candidate_record(root: Path, raw: Path | str, *, max_receipts: int) -> dict[str, Any]:
    path = _resolve(root, raw).resolve()
    exists = path.exists()
    size = _path_size(path) if exists else None
    inside_repo = _inside(path, root)
    kind = _classify(root, path)
    tracked_files = _git_tracked_files_under(root, path) if inside_repo and exists else []
    git_ignored = _git_ignored(root, path) if inside_repo and exists else False
    receipt_files = _receipt_files(root, path, max_receipts=max_receipts) if exists else []
    undurable = [r for r in receipt_files if not r["git_tracked"]]
    blockers: list[str] = []
    if not exists:
        blockers.append("path does not exist")
    if _protected_path(root, path):
        blockers.append("path is protected")
    if exists and path.is_symlink():
        blockers.append("path is a symlink")
    if inside_repo and tracked_files:
        blockers.append(f"contains {len(tracked_files)} git-tracked file(s)")
    if inside_repo and kind in SAFE_CLASSES and not git_ignored:
        blockers.append("repo candidate is not git-ignored")
    if inside_repo and kind not in SAFE_CLASSES:
        blockers.append(f"class {kind} is not a default safe recovery class")
    if not inside_repo and kind != "tmp_mop":
        blockers.append("outside-repo candidate is not an approved tmp_mop path")
    if undurable and kind != "repo_tool_cache":
        blockers.append(f"contains {len(undurable)} unbundled receipt-like text artifact(s)")
    safe = exists and not blockers
    return {
        "path": str(path),
        "display_path": _display(root, path),
        "exists": exists,
        "kind": kind,
        "size_bytes": size,
        "size_human": _human_bytes(size or 0),
        "inside_repo": inside_repo,
        "git_ignored": git_ignored if inside_repo else None,
        "git_tracked_count": len(tracked_files),
        "git_tracked_examples": tracked_files[:10],
        "receipt_like_count": len(receipt_files),
        "receipt_like_examples": receipt_files,
        "safe_to_delete": safe,
        "blockers": blockers,
        "delete_requires": {
            "explicit_class_or_path": True,
            "allowed_classes": sorted(SAFE_CLASSES),
            "receipt_policy": "small text receipts must be git-tracked or bundled first",
        },
    }


def _actions(
    root: Path,
    candidates: list[dict[str, Any]],
    *,
    execute: bool,
    allow_classes: tuple[str, ...],
    allow_paths: tuple[Path, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candidates:
        status = "blocked"
        error = None
        allowed = _candidate_allowed(root, c, allow_classes=allow_classes, allow_paths=allow_paths)
        if c["safe_to_delete"]:
            if not execute:
                status = "would_delete"
            elif not allowed:
                status = "skipped_not_allowed"
            else:
                try:
                    _delete_path(Path(c["path"]))
                    status = "deleted"
                except Exception as e:
                    status = "delete_failed"
                    error = str(e)
        out.append(
            {
                "path": c["path"],
                "display_path": c["display_path"],
                "kind": c["kind"],
                "size_bytes": c["size_bytes"],
                "status": status,
                "allowed": allowed,
                "error": error,
            }
        )
    return out


def _candidate_allowed(
    root: Path,
    candidate: dict[str, Any],
    *,
    allow_classes: tuple[str, ...],
    allow_paths: tuple[Path, ...],
) -> bool:
    if candidate["kind"] in allow_classes:
        return True
    path = Path(candidate["path"]).resolve()
    return any(path == p or _inside(path, p) for p in allow_paths if _path_allows(root, p))


def _path_allows(root: Path, path: Path) -> bool:
    return not _protected_path(root, path)


def _classify(root: Path, path: Path) -> str:
    if _inside(path, root):
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in _TOOL_CACHE_NAMES:
            return "repo_tool_cache"
        if len(rel.parts) >= 3 and rel.parts[0] == "runs" and rel.parts[1] == "e1_baseline":
            return "repo_ignored_run"
        if rel.parts and rel.parts[0] == "runs":
            return "repo_run_protected"
        return "repo_path_protected"
    tmp = Path(tempfile.gettempdir()).resolve()
    if _inside(path, tmp) and path.name.startswith("mop_"):
        return "tmp_mop"
    return "outside_policy"


def _receipt_files(root: Path, path: Path, *, max_receipts: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in _iter_files(path):
        if f.suffix.lower() not in TEXT_EXTS:
            continue
        tracked = _git_tracked(root, f) if _inside(f, root) else False
        out.append(
            {
                "path": str(f),
                "display_path": _display(root, f),
                "size_bytes": _safe_size(f),
                "git_tracked": tracked,
                "durable": tracked,
            }
        )
        if len(out) >= max_receipts:
            break
    return out


def _iter_files(path: Path):
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            yield p


def _path_size(path: Path) -> int:
    if path.is_file():
        return _safe_size(path)
    total = 0
    for p in _iter_files(path):
        total += _safe_size(p)
    return total


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _delete_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _git_ignored(root: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    p = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(rel)],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return p.returncode == 0


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


def _git_tracked_files_under(root: Path, path: Path) -> list[str]:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return []
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(rel)],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [p.decode() for p in raw.split(b"\0") if p][:1000]


def _protected_path(root: Path, path: Path) -> bool:
    protected = {root.resolve(), Path.home().resolve(), Path(tempfile.gettempdir()).resolve(), Path("/")}
    return path.resolve() in protected


def _resolve(root: Path, path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000.0
    return f"{value:.1f} TB"
