"""Sealing helpers shared by every runner: one proof root, one hash rule, one commit stamp.

Hashing, serialization and file reading are deliberately shared with the independent verifier. Effect
computation, group inference and scientific classification are not.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = "mop-fast-state-plasticity-forge-v1"
PROOF = ROOT / "proof" / "substrate" / PROGRAM
RUNS = ROOT / "runs" / "substrate" / PROGRAM
SESOI = 0.05


def sha_obj(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def seal(name: str, obj: dict, subdir: str = "") -> Path:
    obj.setdefault("program", PROGRAM)
    obj.setdefault("source_commit", commit())
    obj.setdefault("SESOI", SESOI)
    obj["sha256"] = sha_obj({k: v for k, v in obj.items() if k != "sha256"})
    out = PROOF / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(obj, indent=2))
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
