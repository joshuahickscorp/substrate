"""The single Substrate evidence and run-state writer.

Historical predecessor evidence remains byte-identical under ``proof/``. New Substrate evidence is
written only under ``evidence/substrate/v1`` and mutable execution receipts only under
``runs/substrate/v1``.

"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

# An editable checkout resolves naturally from the source path. A regular wheel cannot contain the
# multi-gigabyte immutable proof and receipt history, so an installed command binds to an explicit
# checkout through this Substrate-native variable.
ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
PROGRAM = "substrate-v1"
PROOF = ROOT / "evidence" / "substrate" / "v1"
RUNS = ROOT / "runs" / "substrate" / "v1"
ARTIFACTS = ROOT / "artifacts" / "substrate" / "v1"
STATE = Path(os.environ.get("SUBSTRATE_STATE_ROOT", Path.home() / ".substrate"))
STOP = STATE / "stop"

# Activation stays false for the whole program. Nothing here is licensed to act on the world.
ACTIVATION = False
_ACTIVE_FABRIC = None


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


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def atomic_write(path: Path, payload: str) -> Path:
    return atomic_write_bytes(path, payload.encode("utf-8"))


# Internal compatibility alias for the runtime canary helpers. New writers use
# the public name above; keeping this alias avoids changing the canary surface.
_atomic_write = atomic_write


class ArtifactFabric:
    """Supervisor owned staging for deterministic evidence publication.

    Producers construct bytes in memory.  The supervisor validates those bytes and is the only writer
    allowed to publish them.  Byte identical artifacts are cache hits and do not touch the filesystem.
    """

    def __init__(self) -> None:
        self.proposals: dict[Path, bytes] = {}
        self.proposed_bytes = 0
        self.published_bytes = 0
        self.writes = 0
        self.cache_hits = 0

    def propose(self, path: Path, payload: str) -> Path:
        encoded = payload.encode("utf-8")
        self.proposals[path] = encoded
        self.proposed_bytes += len(encoded)
        return path

    def has_proposal(self, path: Path) -> bool:
        return path in self.proposals

    def validate(self, paths: list[Path] | tuple[Path, ...]) -> dict:
        missing, invalid_seals, activation_violations = [], [], []
        for path in paths:
            payload = self.proposals.get(path)
            if payload is None:
                missing.append(path.name)
                continue
            if path.suffix != ".json":
                continue
            try:
                document = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid_seals.append(path.name)
                continue
            expected = sha_obj({key: value for key, value in document.items() if key != "sha256"})
            if document.get("sha256") != expected:
                invalid_seals.append(path.name)
            if document.get("activation") is not False:
                activation_violations.append(path.name)
        return {
            "missing": sorted(missing),
            "invalid_seals": sorted(invalid_seals),
            "activation_violations": sorted(activation_violations),
            "all_pass": not missing and not invalid_seals and not activation_violations,
        }

    def publish(self, paths: list[Path] | tuple[Path, ...]) -> dict:
        published, reused = [], []
        for path in paths:
            payload = self.proposals[path]
            try:
                unchanged = path.is_file() and path.read_bytes() == payload
            except OSError:
                unchanged = False
            if unchanged:
                self.cache_hits += 1
                reused.append(path.name)
            else:
                atomic_write(path, payload.decode("utf-8"))
                self.writes += 1
                self.published_bytes += len(payload)
                published.append(path.name)
        return {"published": published, "reused": reused}

    def stats(self) -> dict:
        return {
            "proposals": len(self.proposals),
            "proposed_bytes": self.proposed_bytes,
            "authoritative_writes": self.writes,
            "published_bytes": self.published_bytes,
            "byte_identical_cache_hits": self.cache_hits,
        }


@contextmanager
def artifact_transaction(fabric: ArtifactFabric | None = None):
    """Make one fabric active for direct in process producers."""
    global _ACTIVE_FABRIC
    if _ACTIVE_FABRIC is not None:
        raise RuntimeError("nested artifact transactions are refused")
    active = fabric or ArtifactFabric()
    _ACTIVE_FABRIC = active
    try:
        yield active
    finally:
        _ACTIVE_FABRIC = None


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
    path = PROOF / subdir / name
    payload = json.dumps(obj, indent=2, default=str)
    return _ACTIVE_FABRIC.propose(path, payload) if _ACTIVE_FABRIC is not None else atomic_write(path, payload)


def seal_md(name: str, text: str, subdir: str = "") -> Path:
    path = PROOF / subdir / name
    return _ACTIVE_FABRIC.propose(path, text) if _ACTIVE_FABRIC is not None else atomic_write(path, text)


def load(name: str, subdir: str = "") -> dict:
    path = PROOF / subdir / name
    if _ACTIVE_FABRIC is not None and path in _ACTIVE_FABRIC.proposals:
        return json.loads(_ACTIVE_FABRIC.proposals[path])
    if not path.is_file():
        from substrate import historical

        path = historical.predecessor_evidence(name, subdir)
    return json.loads(path.read_text())


def exists(name: str, subdir: str = "") -> bool:
    path = PROOF / subdir / name
    if _ACTIVE_FABRIC is not None and path in _ACTIVE_FABRIC.proposals:
        return True
    if path.is_file():
        return True
    from substrate import historical

    return historical.predecessor_evidence(name, subdir).is_file()


def run_json(name: str, obj: dict, subdir: str = "") -> Path:
    return atomic_write(RUNS / subdir / name, json.dumps(obj, indent=2, default=str))


def stop() -> Path:
    return atomic_write(STOP, "operator stop\n")


def resume() -> None:
    STOP.unlink(missing_ok=True)
