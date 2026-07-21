"""Corpus custody: one canonical root, one inventory, and a deletion guard that can say no.

The prior cleanup removed the only local copies of PAMAP2 and HARTH because two worktrees pointed absolute
paths at each other's data directories. Nothing scientific depended on them, which is exactly why it went
unnoticed. The operational method was still invalid, so this module makes the failure impossible to repeat
rather than merely unlikely.

The guard is the load bearing part. It answers one question: if this directory disappeared right now, what
would be unrecoverable.

House style: no dashes.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mop.temporal import io

CANONICAL_ROOT = io.DATA_ROOT

RETENTION_CLASSES = (
    "principal_active",  # a live principal bed depends on it
    "secondary_active",  # a secondary or replication bed depends on it
    "historical_reproducibility",  # only sealed evidence depends on it
    "publicly_recoverable_inactive",  # nothing depends on it and the source is public
    "derived_rebuildable",  # a cache that a recorded command regenerates
    "temporary",  # safe to remove at any time
)

# what makes a path unique and therefore undeletable when it is the only copy
UNIQUE_KINDS = (
    "raw_data",
    "non_rebuildable_cache",
    "split_authority",
    "principal_checkpoint",
    "unindexed_evidence",
)

REQUIRED_FIELDS = (
    "logical_identity",
    "official_source",
    "license",
    "citation",
    "archive_url_authority",
    "archive_hash",
    "extracted_hashes",
    "canonical_path",
    "derived_caches",
    "experiments_using_it",
    "redownload_command",
    "rebuild_command",
    "retention_class",
)


@dataclass
class Corpus:
    logical_identity: str
    official_source: str
    license: str
    citation: str
    archive_url_authority: str
    canonical_path: str
    redownload_command: str
    rebuild_command: str
    retention_class: str
    archive_hash: str = ""
    extracted_hashes: dict = field(default_factory=dict)
    derived_caches: list = field(default_factory=list)
    experiments_using_it: list = field(default_factory=list)
    kind: str = "raw_data"

    def violations(self) -> list[str]:
        v = [f"{self.logical_identity}: {f} is empty" for f in REQUIRED_FIELDS
             if not getattr(self, f, None) and f not in ("archive_hash", "extracted_hashes",
                                                         "derived_caches", "experiments_using_it")]
        if self.retention_class not in RETENTION_CLASSES:
            v.append(f"{self.logical_identity}: unknown retention class {self.retention_class!r}")
        if self.kind not in UNIQUE_KINDS:
            v.append(f"{self.logical_identity}: unknown kind {self.kind!r}")
        p = Path(self.canonical_path)
        if not p.is_absolute():
            v.append(f"{self.logical_identity}: canonical path is not absolute")
        elif not str(p).startswith(str(CANONICAL_ROOT)):
            v.append(f"{self.logical_identity}: canonical path sits outside {CANONICAL_ROOT}")
        return v

    def present(self) -> bool:
        return Path(self.canonical_path).exists()

    def as_dict(self) -> dict:
        d = {f: getattr(self, f) for f in REQUIRED_FIELDS}
        d["kind"] = self.kind
        d["present"] = self.present()
        d["violations"] = self.violations()
        return d


# ---------------------------------------------------------------- inventory


def inventory(corpora: list[Corpus]) -> dict:
    rows = [c.as_dict() for c in corpora]
    return {
        "schema": "mop-data-custody-inventory/v1",
        "canonical_root": str(CANONICAL_ROOT),
        "retention_classes": list(RETENTION_CLASSES),
        "unique_kinds": list(UNIQUE_KINDS),
        "corpora": rows,
        "n": len(rows),
        "present": [r["logical_identity"] for r in rows if r["present"]],
        "absent": [r["logical_identity"] for r in rows if not r["present"]],
        "violations": [x for r in rows for x in r["violations"]],
        "all_declared": not any(r["violations"] for r in rows),
    }


# ---------------------------------------------------------------- the deletion guard


def _dir_size(p: Path, cap: int = 4000) -> int:
    n = 0
    for i, f in enumerate(p.rglob("*")):
        if i > cap:
            break
        if f.is_file():
            n += 1
    return n


def unique_holdings(target: Path, corpora: list[Corpus], evidence_index: set[str] | None = None) -> list[dict]:
    """What would become unrecoverable if target were removed right now.

    A corpus counts as held only inside target when its canonical path is inside target and no other declared
    copy exists. A derived cache is exempt only when its rebuild command names a source that survives.
    """
    target = Path(target).resolve()
    out = []
    for c in corpora:
        p = Path(c.canonical_path)
        inside = str(p.resolve()) == str(target) or str(p.resolve()).startswith(str(target) + os.sep)
        if not inside or not p.exists():
            continue
        if c.retention_class == "temporary":
            continue
        rebuildable = c.retention_class == "derived_rebuildable" and bool(c.rebuild_command)
        source_survives = any(
            Path(o.canonical_path).exists()
            and not str(Path(o.canonical_path).resolve()).startswith(str(target) + os.sep)
            for o in corpora
            if o.logical_identity in c.derived_caches or c.logical_identity in o.derived_caches
        )
        if rebuildable and source_survives:
            continue
        out.append({
            "logical_identity": c.logical_identity,
            "kind": c.kind,
            "retention_class": c.retention_class,
            "canonical_path": c.canonical_path,
            "publicly_recoverable": bool(c.archive_url_authority and c.redownload_command),
            "rebuildable": rebuildable,
            "source_survives_removal": source_survives,
        })
    for rel in sorted(evidence_index or []):
        p = (io.ROOT / rel).resolve()
        if str(p).startswith(str(target) + os.sep):
            out.append({"logical_identity": rel, "kind": "unindexed_evidence",
                        "retention_class": "historical_reproducibility", "canonical_path": str(p),
                        "publicly_recoverable": False, "rebuildable": False,
                        "source_survives_removal": False})
    return out


def guard(target: Path, corpora: list[Corpus], evidence_index: set[str] | None = None,
          allow_publicly_recoverable: bool = False) -> dict:
    """Decide whether target may be removed. Refusal is the default when anything unique lives inside it."""
    held = unique_holdings(target, corpora, evidence_index)
    blocking = [h for h in held if not (allow_publicly_recoverable and h["publicly_recoverable"])]
    return {
        "target": str(target),
        "exists": Path(target).exists(),
        "unique_holdings": held,
        "blocking": blocking,
        "allowed": not blocking,
        "reason": "" if not blocking else (
            f"{len(blocking)} unique holding(s) live only here: "
            + ", ".join(sorted(h["logical_identity"] for h in blocking))
        ),
        "override_available": bool(held) and all(h["publicly_recoverable"] for h in held),
    }


def remove_worktree(target: Path, corpora: list[Corpus], evidence_index: set[str] | None = None,
                    allow_publicly_recoverable: bool = False, dry_run: bool = True) -> dict:
    g = guard(target, corpora, evidence_index, allow_publicly_recoverable)
    g["dry_run"] = dry_run
    g["removed"] = False
    if g["allowed"] and not dry_run:
        shutil.rmtree(target)
        g["removed"] = True
    return g


# ---------------------------------------------------------------- integrity checks


def verify_corpus(c: Corpus) -> dict:
    p = Path(c.canonical_path)
    checks = {"present": p.exists()}
    if not p.exists():
        checks["all_pass"] = False
        return {"logical_identity": c.logical_identity, "checks": checks,
                "status": "absent", "recovery": c.redownload_command}
    live = {}
    for rel, want in (c.extracted_hashes or {}).items():
        f = p / rel
        if not f.is_file():
            live[rel] = "missing"
        else:
            live[rel] = io.sha_file(f)
    checks["no_missing_files"] = "missing" not in live.values()
    checks["hashes_match"] = all(live.get(k) == v for k, v in (c.extracted_hashes or {}).items())
    checks["not_empty"] = _dir_size(p) > 0 if p.is_dir() else p.stat().st_size > 0
    checks["all_pass"] = all(v for v in checks.values() if isinstance(v, bool))
    return {
        "logical_identity": c.logical_identity,
        "checks": checks,
        "live_hashes": live,
        "status": "intact" if checks["all_pass"] else "damaged",
        "recovery": c.redownload_command if not checks["all_pass"] else "",
    }


def load_inventory(path: Path | None = None) -> list[Corpus]:
    p = Path(path or (io.PROOF / "MOP_DATA_CUSTODY_INVENTORY.json"))
    doc = json.loads(p.read_text())
    return [Corpus(**{k: v for k, v in row.items() if k in REQUIRED_FIELDS + ("kind",)})
            for row in doc["corpora"]]
