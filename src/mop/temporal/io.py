"""Sealing helpers for the temporal core mechanism program.

Same hash rule and commit stamp as the method kernel, different roots. A sibling rather than an edit,
because mop.method.io is the sealing helper behind already sealed immutable evidence.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
PROGRAM = "mop-temporal-core-mechanism-v1"
PROOF = ROOT / "proof" / "substrate" / PROGRAM
RUNS = ROOT / "runs" / "substrate" / PROGRAM
STOP = Path.home() / ".mop_temporal_core_mechanism_stop"
DATA_ROOT = Path("/Users/scammermike/Downloads/mop-data")
SESOI = 0.05
EQUIVALENCE_MARGIN = 0.02  # two configurations are equivalent when the bound sits inside this margin

def sha_obj(v) -> str:
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def _atomic_text(path: Path, payload: str) -> None:
    """Land a receipt atomically; an interrupted writer leaves a visible partial, never a forged final."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    partial.write_text(payload)
    partial.replace(path)


def seal(name: str, obj: dict, subdir: str = "") -> Path:
    obj = json.loads(json.dumps(dict(obj), default=str)); obj.setdefault("program", PROGRAM)
    obj.setdefault("source_commit", commit())
    obj.setdefault("SESOI", SESOI)
    obj.setdefault("sha256_version", "canonical_json_v2")
    obj["sha256"] = sha_obj({k: v for k, v in obj.items() if k != "sha256"})
    out = PROOF / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    _atomic_text(p, json.dumps(obj, indent=2, default=str))
    return p


def seal_md(name: str, text: str, subdir: str = "") -> Path:
    out = PROOF / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    _atomic_text(p, text)
    return p


def load(name: str, subdir: str = "") -> dict:
    return json.loads((PROOF / subdir / name).read_text())


def exists(name: str, subdir: str = "") -> bool:
    return (PROOF / subdir / name).is_file()


def run_json(name: str, obj: dict, subdir: str = "") -> Path:
    out = RUNS / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    doc = json.loads(json.dumps(dict(obj), default=str))
    doc.setdefault("program", PROGRAM)
    doc.setdefault("source_commit", commit())
    doc.setdefault("result_hash_version", "canonical_json_v2")
    doc["result_sha256"] = sha_obj({k: v for k, v in doc.items() if k != "result_sha256"})
    _atomic_text(p, json.dumps(doc, indent=2, default=str))
    return p


def run_load(name: str, subdir: str = "") -> dict:
    return json.loads((RUNS / subdir / name).read_text())
