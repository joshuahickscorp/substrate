"""The single Substrate evidence and run-state writer.

Historical predecessor evidence remains byte-identical under ``proof/``. New Substrate evidence is
written only under ``evidence/substrate/v1`` and mutable execution receipts only under
``runs/substrate/v1``.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = "substrate-v1"
PROOF = ROOT / "evidence" / "substrate" / "v1"
RUNS = ROOT / "runs" / "substrate" / "v1"
ARTIFACTS = ROOT / "artifacts" / "substrate" / "v1"
STATE = Path(os.environ.get("SUBSTRATE_STATE_ROOT", Path.home() / ".substrate"))
STOP = STATE / "stop"

# Activation stays false for the whole program. Nothing here is licensed to act on the world.
ACTIVATION = False


def data_root() -> Path:
    """Where the corpora live, which is deliberately outside every worktree.

    A clean clone must resolve the same Substrate custody authority as the development checkout. Historical
    environment names and predecessor directory guesses are intentionally not accepted.
    """
    from substrate import data

    return data.root()


def sha_obj(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@lru_cache(maxsize=1)
def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _atomic_write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


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
    return _atomic_write(PROOF / subdir / name, json.dumps(obj, indent=2, default=str))


def seal_md(name: str, text: str, subdir: str = "") -> Path:
    return _atomic_write(PROOF / subdir / name, text)


def load(name: str, subdir: str = "") -> dict:
    path = PROOF / subdir / name
    if not path.is_file():
        from substrate import historical

        path = historical.predecessor_evidence(name, subdir)
    return json.loads(path.read_text())


def exists(name: str, subdir: str = "") -> bool:
    if (PROOF / subdir / name).is_file():
        return True
    from substrate import historical

    return historical.predecessor_evidence(name, subdir).is_file()


def run_json(name: str, obj: dict, subdir: str = "") -> Path:
    return _atomic_write(RUNS / subdir / name, json.dumps(obj, indent=2, default=str))


def stop() -> Path:
    return _atomic_write(STOP, "operator stop\n")


def resume() -> None:
    STOP.unlink(missing_ok=True)
