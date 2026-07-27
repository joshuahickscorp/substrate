"""Sealing for the Substrate master program.

One proof root, one hash rule, one commit stamp, shared by every Substrate stage. A sibling of
mop.method.io and mop.temporal.io for the same reason those are siblings of each other: the older modules
are the sealing helpers behind already sealed immutable evidence and must not change.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = "mop-substrate-master-v1"
PROOF = ROOT / "proof" / "substrate" / PROGRAM
RUNS = ROOT / "runs" / "substrate" / PROGRAM
STOP = Path.home() / ".mop_substrate_stop"

# Activation stays false for the whole program. Nothing here is licensed to act on the world.
ACTIVATION = False


def data_root() -> Path:
    """Where the corpora live, which is deliberately outside every worktree.

    Resolving this as ROOT.parent / mop-data works in the checkout it was written in and nowhere else. A
    clean clone lands in a temporary directory whose parent holds no corpora, so every data backed module
    silently loses its bed. The order below is explicit override, then the custody authority that already
    declares the canonical root, then the historical layout.
    """
    import json
    import os

    env = os.environ.get("MOP_DATA_ROOT")
    if env:
        return Path(env)
    custody = (ROOT / "proof" / "substrate" / "mop-temporal-core-mechanism-v1"
               / "MOP_DATA_CUSTODY_AUTHORITY.json")
    try:
        declared = json.loads(custody.read_text()).get("canonical_root")
        if declared and Path(declared).is_dir():
            return Path(declared)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return ROOT.parent / "mop-data"


def sha_obj(v) -> str:
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def seal(name: str, obj: dict, subdir: str = "") -> Path:
    obj.setdefault("program", PROGRAM)
    obj.setdefault("source_commit", commit())
    obj.setdefault("activation", ACTIVATION)
    # The digest is taken over the round tripped form, not the in memory object. JSON has no integer
    # keys, so a dict keyed by int is written with string keys and read back with string keys, and a
    # seal computed before the round trip can never verify after it. One artifact with integer keys was
    # already unverifiable for exactly this reason.
    obj = json.loads(json.dumps({k: v for k, v in obj.items() if k != "sha256"}, default=str))
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
