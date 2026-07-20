from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from ..logging_utils import get_logger
from ..provenance import provenance as _provenance
from .profiles import Profile

log = get_logger("downloader")

LOCAL_METHODS = frozenset({"generate", "local-path"})
REMOTE_METHODS = frozenset(
    {"http", "http-token", "mirror-or-metadata", "epic-download-script", "ego4d-cli", "ego-exo4d-cli"}
)
# statuses requiring a license acknowledgement before a real fetch
TERMS_STATUSES = frozenset({"manual"})

GenFn = Callable[[dict, Path], dict]
FetchFn = Callable[[dict, Path, int], dict]


def unsafe_archive_members(names: list[str]) -> list[str]:
    import ntpath
    import posixpath

    bad: list[str] = []
    for n in names:
        norm = n.replace("\\", "/")  # normalize Windows separators so .. is caught either way
        win_abs = ntpath.isabs(n) or ntpath.splitdrive(n)[0] != ""  # C:\, \\server\share, C:rel
        if posixpath.isabs(norm) or win_abs or n.startswith(("/", "\\")):
            bad.append(n)
            continue
        if ".." in norm.split("/"):
            bad.append(n)
    return bad


def safe_extract(archive_members: list[str], dest: Path) -> list[str]:
    bad = unsafe_archive_members(archive_members)
    if bad:
        raise RuntimeError(f"refusing to extract {len(bad)} unsafe archive member(s) into {dest}: {bad[:5]}")
    return archive_members


def hash_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_chunk), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_dir(root: Path) -> tuple[list[str], int]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    hashes = [hash_file(p) for p in files]
    total = sum(p.stat().st_size for p in files)
    return hashes, total


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def acquire(
    plan: dict,
    profile: Profile,
    out_dir: Path,
    execute: bool = False,
    budget_gb: float | None = None,
    accept_license: bool = False,
    gen_fn: GenFn | None = None,
    fetch_fn: FetchFn | None = None,
    manifest_path: Path | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or (out_dir / "acquire_manifest.json")
    budget_bytes = int(profile.effective_budget_gb(budget_gb) * 1e9)

    prior = _load_prior(manifest_path)
    done = {e["slug"]: e for e in prior.get("sources", []) if e.get("status") == "complete"}

    manifest = {
        "mode": "execute" if execute else "dry-run",
        "profile": profile.name,
        "budget_gb": round(budget_bytes / 1e9, 3),
        "accept_license": accept_license,
        "started": time.time(),
        "provenance": _provenance(seed=0, device="cpu", result_tag="provisional", extra={"stage": "acquire"}),
        "sources": [],
        "seen_hashes": prior.get("seen_hashes", []),
    }
    seen_hashes: set[str] = set(manifest["seen_hashes"])
    spent = 0

    for sel in plan.get("selected", []):
        slug = sel["slug"]
        method = sel.get("download_method", "")
        planned_bytes = int((float(sel.get("raw_gb", 0.0)) + float(sel.get("cache_gb", 0.0))) * 1e9)
        rec: dict = {
            "slug": slug,
            "method": method,
            "planned_bytes": planned_bytes,
            "planned_human": _human(planned_bytes),
        }

        prior_rec = done.get(slug)
        if prior_rec is not None:  # resume candidate
            dest_prev = prior_rec.get("dest")
            if dest_prev and Path(dest_prev).exists():  # data really still on disk
                rec.update(prior_rec)
                rec["status"] = "complete"
                rec["resumed"] = True
                spent += int(prior_rec.get("bytes", 0))  # cumulative budget across resumes (Frontier 16)
                manifest["sources"].append(rec)
                _write_manifest(manifest_path, manifest)
                continue
            log.warning(
                "resume: %s was marked complete but data is missing at %s; re-acquiring", slug, dest_prev
            )

        # license gate (independent of dry-run): a manual source needs acknowledgement to even plan
        needs_terms = sel.get("status") in TERMS_STATUSES
        if needs_terms and not accept_license:
            rec["status"] = "needs-license" if not execute else "blocked"
            rec["detail"] = "manual license not acknowledged (pass --accept-license)"
            manifest["sources"].append(rec)
            _write_manifest(manifest_path, manifest)
            continue

        if not execute:
            rec["status"] = "planned"
            rec["detail"] = "dry-run (no bytes written); pass --execute to acquire"
            manifest["sources"].append(rec)
            _write_manifest(manifest_path, manifest)
            continue

        if spent + planned_bytes > budget_bytes:
            rec["status"] = "skipped-budget"
            rec["detail"] = f"would exceed budget ({_human(budget_bytes)}; {_human(spent)} spent)"
            manifest["sources"].append(rec)
            _write_manifest(manifest_path, manifest)
            log.warning("budget reached at %s; stopping real acquisition", slug)
            break

        dest = out_dir / "data" / slug
        try:
            if method == "generate":
                if gen_fn is None:
                    raise RuntimeError("no generator supplied for a 'generate' source")
                res = gen_fn(sel, dest)
            elif method == "local-path":
                if not sel.get("local_path"):  # the user supplies the dir separately; not an error
                    rec["status"] = "needs-source"
                    rec["detail"] = "local-path lane: pass sel['local_path'] (a class-foldered dir) to ingest"
                    manifest["sources"].append(rec)
                    _write_manifest(manifest_path, manifest)
                    continue
                res = _ingest_local(sel, dest)
            elif method in REMOTE_METHODS:
                if fetch_fn is None:
                    rec["status"] = "blocked"
                    rec["detail"] = (
                        f"remote method {method!r}: no fetcher/credentials on this device (clean stop)"
                    )
                    manifest["sources"].append(rec)
                    _write_manifest(manifest_path, manifest)
                    continue
                res = fetch_fn(sel, dest, budget_bytes - spent)
            else:
                raise RuntimeError(f"unknown download_method {method!r}")
        except Exception as e:  # fail cleanly: record, remove the partial, keep the manifest consistent
            _remove_partial(out_dir, dest)  # no orphaned, untracked bytes left behind
            rec["status"] = "error"
            rec["detail"] = f"{type(e).__name__}: {e} (partial output removed)"
            manifest["sources"].append(rec)
            _write_manifest(manifest_path, manifest)
            log.warning("acquire %s failed: %s", slug, e)
            continue

        got_bytes = int(res.get("bytes", 0))
        hashes, dups = _dedup(res.get("hashes") or _hashes_of(dest), seen_hashes)
        spent += got_bytes
        over = spent > budget_bytes
        rec.update(
            status="over-budget" if over else "complete",
            bytes=got_bytes,
            human=_human(got_bytes),
            n_files=len(hashes),
            n_duplicates=dups,
            dest=res.get("dest", str(dest)),  # local-path records the user's real dir, not the scratch dest
            n_clips=res.get("n_clips"),
        )
        manifest["sources"].append(rec)
        manifest["seen_hashes"] = sorted(seen_hashes)
        _write_manifest(manifest_path, manifest)
        if over:
            rec["detail"] = (
                f"fetched {_human(got_bytes)} taking total to {_human(spent)} OVER budget "
                f"{_human(budget_bytes)}; stopping acquisition"
            )
            log.warning("budget OVERRUN at %s: %s", slug, rec["detail"])
            break

    manifest["finished"] = time.time()
    manifest["totals"] = _totals(manifest["sources"], budget_bytes, spent)
    _write_manifest(manifest_path, manifest)
    log.info(
        "acquire %s: %d sources, %s spent of %s budget",
        manifest["mode"],
        len(manifest["sources"]),
        _human(spent),
        _human(budget_bytes),
    )
    return manifest


def _ingest_local(sel: dict, dest: Path) -> dict:
    src = sel.get("local_path")
    if not src:
        raise RuntimeError("local-path source needs sel['local_path'] (a class-foldered dir)")
    p = Path(src)
    if not p.exists():
        raise RuntimeError(f"local source {p} does not exist")
    hashes, total = _hash_dir(p)
    return {"bytes": total, "hashes": hashes, "dest": str(p), "n_clips": len(hashes)}


def _hashes_of(dest: Path) -> list[str]:
    if not Path(dest).exists():
        return []
    hashes, _ = _hash_dir(Path(dest))
    return hashes


def _dedup(hashes: list[str], seen: set[str]) -> tuple[list[str], int]:
    dups = 0
    for h in hashes:
        if h in seen:
            dups += 1
        else:
            seen.add(h)
    return hashes, dups


def _remove_partial(out_dir: Path, dest: Path) -> None:
    import shutil

    try:
        dest = Path(dest)
        scratch = Path(out_dir) / "data"
        if dest.exists() and scratch in dest.parents:
            shutil.rmtree(dest)
    except OSError as e:
        log.warning("could not remove partial %s: %s", dest, e)


def _load_prior(manifest_path: Path) -> dict:
    if Path(manifest_path).exists():
        try:
            return json.loads(Path(manifest_path).read_text())
        except Exception as e:  # corrupt/truncated manifest: do NOT silently reset resume+dedup state
            log.warning(
                "prior manifest %s is unreadable (%s); resume + dedup state is reset, completed "
                "sources may be re-fetched. Inspect/restore the manifest to resume cleanly.",
                manifest_path,
                e,
            )
            return {}
    return {}


def _totals(sources: list[dict], budget_bytes: int, spent: int) -> dict:
    by_status: dict[str, int] = {}
    for s in sources:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
    return {
        "n_sources": len(sources),
        "by_status": by_status,
        "bytes_spent": spent,
        "human_spent": _human(spent),
        "budget_bytes": budget_bytes,
        "budget_human": _human(budget_bytes),
        "headroom_human": _human(max(0, budget_bytes - spent)),
    }


def _human(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1000 or u == "TB":
            return f"{int(n)} B" if u == "B" else f"{x:.2f} {u}"
        x /= 1000.0
    return f"{x:.2f} TB"
