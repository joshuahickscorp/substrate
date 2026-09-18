"""Small, shared contracts. Nothing here grants authority by being deserialized.

Canonical JSON is the only interchange representation. Host credentials and
private evaluator state must never be put into portable cognitive records.
"""
from __future__ import annotations

import base64
import hashlib
import json
from functools import lru_cache
import math
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

SCHEMA = "substrate-v2"
MAX_JSON = 4 * 1024 * 1024
ZERO = "0" * 64


class Refused(ValueError):
    """A contract failed. Refusal is not a successful research observation."""


class Conflict(Refused):
    """A compare-and-swap or fencing token is stale."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refused(message)


def identifier(value: Any) -> str:
    require(isinstance(value, str) and bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", value)),
            "invalid identifier")
    return value


def digest(value: Any) -> str:
    require(isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value)), "invalid SHA-256")
    return value


def numeric(value: Any, low: float = -1e100, high: float = 1e100) -> float:
    require(type(value) in (int, float) and low <= value <= high and math.isfinite(value),
            "finite numeric value outside permitted range")
    return float(value)


def integer(value: Any, low: int = 0, high: int = 2**63 - 1) -> int:
    require(type(value) is int and low <= value <= high, "integer outside permitted range")
    return value


def fields(value: Any, required: set[str], optional: set[str] | None = None) -> dict:
    require(type(value) is dict, "expected an object")
    require(required <= value.keys() <= required | (optional or set()), "missing or undeclared fields")
    return value


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise Refused("expected finite JSON-only data") from exc


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse(raw: bytes | str, limit: int = MAX_JSON) -> Any:
    require(isinstance(raw, (bytes, str)), "JSON must be text or bytes")
    require(len(raw.encode() if isinstance(raw, str) else raw) <= limit, "JSON size limit")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise Refused(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        canonical(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise Refused(f"invalid JSON: {exc}") from exc


def clone(value: Any) -> Any:
    return parse(canonical(value))


def scope_matches(required: Mapping, actual: Mapping) -> bool:
    return all(key in actual and sha(actual[key]) == sha(value) for key, value in required.items())


def regular_bytes(path: Path, limit: int = MAX_JSON) -> bytes:
    """Refuse file links/devices; parent directories must be operator-controlled."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "expected unlinked regular file")
        require(info.st_size <= limit, "file size limit")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        require(len(raw) <= limit, "file grew past limit")
        return raw
    finally:
        os.close(fd)


def read_json(path: Path, limit: int = MAX_JSON) -> Any:
    return parse(regular_bytes(path, limit), limit)


def atomic_bytes(path: Path, content: bytes, *, replace: bool = False) -> None:
    """Atomic, fsynced publication; never silently overwrite an operator artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            require(not path.is_symlink(), "refusing destination symlink")
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    atomic_bytes(path, canonical(value) + b"\n", replace=replace)


def source_digest(root: Path) -> str:
    """Bind every executable/configuration input; do not hash run outputs into source."""
    paths = []
    for folder in ("src", "tests", "tools", "configs", ".github"):
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(p for name in ("pyproject.toml", "Makefile", "Dockerfile", ".dockerignore", ".gitignore") if (p := root / name).is_file())
    return sha([[p.relative_to(root).as_posix(), bytes_sha(regular_bytes(p, 16 * MAX_JSON))]
                for p in sorted(paths)])


@dataclass(frozen=True)
class Cost:
    """Unknown measurements remain None, never an invented zero."""
    elapsed_ns: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    peak_memory_bytes: int | None = None
    energy_j: float | None = None
    source: str = "measured-wall-only"

    def __post_init__(self):
        for name in ("elapsed_ns", "model_calls", "input_tokens", "output_tokens"):
            integer(getattr(self, name))
        if self.peak_memory_bytes is not None:
            integer(self.peak_memory_bytes)
        if self.energy_j is not None:
            numeric(self.energy_j, 0)
        identifier(self.source)

    def document(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Budget:
    calls: int = 500
    tokens: int = 1_000_000
    wall_seconds: float = 3600
    retained_bytes: int = 256 * 1024 * 1024
    workers: int = 2

    def __post_init__(self):
        integer(self.calls)
        integer(self.tokens)
        numeric(self.wall_seconds, 0.001, 31_536_000)
        integer(self.retained_bytes, 1024)
        integer(self.workers, 1, 64)

    def document(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrustRule:
    public_key: str
    purposes: tuple[str, ...]
    domains: tuple[str, ...]
    revoked: bool = False

    def key(self) -> Ed25519PublicKey:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        try:
            return Ed25519PublicKey.from_public_bytes(base64.b64decode(self.public_key, validate=True))
        except (ValueError, TypeError) as exc:
            raise Refused("invalid public key") from exc


class Trust:
    """Host-local scoped verifier trust. A signature is not proof of a sound oracle."""
    def __init__(self, rules: Mapping[str, TrustRule]):
        self.rules = dict(rules)
        for name, rule in self.rules.items():
            identifier(name)
            rule.key()

    @classmethod
    def load(cls, path: Path) -> Trust:
        document = read_json(path)
        fields(document, {"schema", "rules"})
        require(document["schema"] == "substrate-trust-v2", "unsupported trust schema")
        return cls({name: TrustRule(**value) for name, value in document["rules"].items()})

    def verify(self, certificate: dict, *, purpose: str, domain: str, subject: Any,
               producer: str, now: float | None = None) -> dict:
        from cryptography.exceptions import InvalidSignature
        fields(certificate, {"schema", "signer", "producer", "purpose", "domain", "subject",
                             "issued", "expires", "evidence", "findings", "signature"})
        require(certificate["schema"] == "substrate-attestation-v2", "wrong attestation schema")
        signer = identifier(certificate["signer"])
        rule = self.rules.get(signer)
        require(rule is not None and not rule.revoked, "untrusted or revoked verifier")
        require(signer != producer and certificate["producer"] == producer, "producer cannot self-verify")
        require(purpose in rule.purposes and domain in rule.domains, "verifier scope refused")
        require(certificate["purpose"] == purpose and certificate["domain"] == domain,
                "attestation purpose mismatch")
        require(certificate["subject"] == sha(subject), "attestation subject mismatch")
        current = time.time() if now is None else numeric(now, 0)
        require(numeric(certificate["issued"], 0) <= current < numeric(certificate["expires"], 0),
                "expired or future attestation")
        digest(certificate["evidence"])
        require(type(certificate["findings"]) is dict, "findings must be an object")
        unsigned = {key: value for key, value in certificate.items() if key != "signature"}
        try:
            rule.key().verify(base64.b64decode(certificate["signature"], validate=True), canonical(unsigned))
        except (InvalidSignature, ValueError, TypeError) as exc:
            raise Refused("invalid attestation signature") from exc
        return clone(certificate["findings"])


def attest(private: Ed25519PrivateKey, *, signer: str, producer: str, purpose: str,
           domain: str, subject: Any, evidence: str, findings: dict, ttl: float = 3600,
           now: float | None = None) -> dict:
    """For independent evaluator/operator processes; never called by an LLM broker."""
    issued = time.time() if now is None else numeric(now, 0)
    numeric(ttl, 0.001, 31_536_000)
    unsigned = {"schema": "substrate-attestation-v2", "signer": identifier(signer),
                "producer": identifier(producer), "purpose": identifier(purpose),
                "domain": identifier(domain), "subject": sha(subject), "issued": issued,
                "expires": issued + ttl, "evidence": digest(evidence), "findings": clone(findings)}
    return {**unsigned, "signature": base64.b64encode(private.sign(canonical(unsigned))).decode()}


@lru_cache(maxsize=4096)
def valid_signature(encoded: bytes, public_key: str) -> bool:
    """Signature cache is keyed by exact bytes AND current trusted key, not just signer name."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    certificate = parse(encoded)
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key, validate=True))
        unsigned = {name: value for name, value in certificate.items() if name != "signature"}
        key.verify(base64.b64decode(certificate["signature"], validate=True), canonical(unsigned))
        return True
    except (InvalidSignature, ValueError, TypeError, KeyError):
        return False


def inspect_body(machine_tag: str) -> dict:
    """Read-only host facts; not GPU discovery, an energy measurement or a hardware qualification."""
    import importlib.metadata
    import platform
    import sqlite3
    identifier(machine_tag)
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        memory = None
    try:
        crypto = importlib.metadata.version("cryptography")
    except importlib.metadata.PackageNotFoundError:
        crypto = None
    document = {"schema": "substrate-body-profile-v2", "machine_tag": machine_tag,
                "system": platform.system(), "release": platform.release(), "architecture": platform.machine(),
                "python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "cryptography": crypto,
                "logical_cpus": os.cpu_count(), "physical_memory_bytes": memory,
                "gpu": None, "fpga": None, "energy_instrument": None, "qualified": False}
    return {"profile": document, "host_digest": sha(document)}
