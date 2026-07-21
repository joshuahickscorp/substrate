"""Sealing helpers for the experimental method reformation.

One proof root, one hash rule, one commit stamp, shared by every stage of this program. Deliberately a
sibling of fastforge.runs.io rather than an edit of it, because that module is the sealing helper behind
already sealed immutable evidence.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = "mop-experimental-method-reformation-v1"
PROOF = ROOT / "proof" / "method" / PROGRAM
RUNS = ROOT / "runs" / "method" / PROGRAM
STOP = Path.home() / ".mop_experimental_method_stop"
SESOI = 0.05


def sha_obj(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def seal(name: str, obj: dict, subdir: str = "") -> Path:
    obj.setdefault("program", PROGRAM)
    obj.setdefault("source_commit", commit())
    obj["sha256"] = sha_obj({k: v for k, v in obj.items() if k != "sha256"})
    out = PROOF / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def seal_md(name: str, text: str, subdir: str = "") -> Path:
    out = PROOF / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(text)
    return p


def load(name: str, subdir: str = "") -> dict:
    return json.loads((PROOF / subdir / name).read_text())


def exists(name: str, subdir: str = "") -> bool:
    return (PROOF / subdir / name).is_file()


def run_json(name: str, obj: dict, subdir: str = "") -> Path:
    out = RUNS / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p
