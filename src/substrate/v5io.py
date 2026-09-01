"""Deterministic, content-addressed storage owned exclusively by Substrate v5.

The v5 writer fails closed on path escape, non-JSON values, invalid seals, and any
attempt to enable external activation.  Named publications are atomic convenience
indexes; their immutable content-addressed copies are the durable authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from substrate import evidence as v1

try:
    from json.encoder import c_make_encoder, encode_basestring
except ImportError:  # pragma: no cover - the fallback is for non-CPython runtimes.
    _CANONICAL_JSON_ENCODER = None
else:
    _CANONICAL_JSON_ENCODER = (
        c_make_encoder(None, None, encode_basestring, None, ":", ",", True, False, False)
        if c_make_encoder is not None
        else None
    )

ROOT = v1.ROOT
EVIDENCE = ROOT / "evidence" / "substrate" / "v5"
RUNS = ROOT / "runs" / "substrate" / "v5"
ARTIFACTS = ROOT / "evidence" / "artifacts" / "substrate" / "v5"
CONFIGS = ROOT / "ops" / "configs" / "substrate" / "v5"
MODELS = ROOT / "models" / "substrate" / "v5"
DATA = ROOT / "data" / "substrate" / "v5"
CACHE = ROOT / "cache" / "substrate" / "v5"
STATE = RUNS / "state"
STOP = STATE / "stop"
PROGRAM = "substrate-v5"
ACTIVATION = False

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class Refused(RuntimeError):
    """A v5 storage operation violated custody or integrity constraints."""


def roots() -> tuple[Path, ...]:
    """Return the live v5 roots.

    This is a function rather than a constant so isolated tests may redirect a
    root without leaving a stale allow-list behind.
    """

    return (CONFIGS, EVIDENCE, RUNS, ARTIFACTS, MODELS, DATA, CACHE)


def _normal_json(value: Any) -> JSONValue:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise Refused(f"value is not finite canonical JSON: {error}") from error
    return json.loads(encoded)


def canonical_json(value: Any) -> bytes:
    """Encode finite JSON in the byte-stable v5 canonical form.

    ``json.dumps`` already performs the same type and finite-number
    validation used by ``_normal_json``. The old path dumped, parsed, and
    dumped the value again even though the first byte representation was
    already canonical; retaining that second pass added CPU and allocation
    cost to every digest and publication.
    """

    try:
        if _CANONICAL_JSON_ENCODER is not None:
            try:
                encoded = "".join(_CANONICAL_JSON_ENCODER(value, 0))
            except RecursionError:
                # The no-marker C fast path does not detect cycles early. Let
                # the reference encoder preserve its ValueError refusal.
                encoded = json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
        else:
            encoded = json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        return (encoded + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Refused(f"value is not finite canonical JSON: {error}") from error


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_obj(value: Any) -> str:
    return sha_bytes(canonical_json(value))


@lru_cache(maxsize=1)
def commit() -> str:
    """Return the process-scoped source commit used by sealed documents."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def source_inventory() -> dict[str, str]:
    roots_to_scan = (ROOT / "src" / "substrate", ROOT / "tests" / "substrate")
    return {
        path.relative_to(ROOT).as_posix(): sha_bytes(path.read_bytes())
        for source_root in roots_to_scan
        for path in sorted(source_root.glob("*.py"))
    }


@lru_cache(maxsize=1)
def source_digest() -> str:
    return sha_obj(source_inventory())


def _contains_true_activation(value: JSONValue) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "activation" and child is not False:
                return True
            if _contains_true_activation(child):
                return True
    elif isinstance(value, list):
        for child in value:
            if _contains_true_activation(child):
                return True
    return False


def assert_activation_false(value: Any) -> None:
    normal = _normal_json(value)
    _assert_normalized_activation_false(normal)


def _assert_normalized_activation_false(value: JSONValue) -> None:
    """Reject activation in a tree already normalized as JSON."""

    if _contains_true_activation(value):
        raise Refused("v5 activation must remain exactly false")


def _owned_path(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    resolved = candidate.resolve(strict=False)
    for root in roots():
        owned_root = root.expanduser().absolute().resolve(strict=False)
        try:
            resolved.relative_to(owned_root)
        except ValueError:
            continue
        return candidate
    raise Refused(f"path is outside the Substrate v5 roots: {path}")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, immutable: bool = False) -> Path:
    """Atomically publish bytes beneath a v5 root.

    Immutable publication is idempotent for identical bytes and refuses a
    collision.  Parent directory fsync makes the rename durable across a crash.
    """

    destination = _owned_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if immutable:
            if existing != payload:
                raise Refused(f"immutable v5 object collision at {destination}")
            return destination
        if existing == payload:
            return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write(path: Path, payload: str, *, immutable: bool = False) -> Path:
    return atomic_write_bytes(path, payload.encode("utf-8"), immutable=immutable)


def sealed_document(document: dict[str, Any]) -> dict[str, JSONValue]:
    """Return a canonical JSON document whose digest covers every other field."""

    if not isinstance(document, dict):
        raise Refused("a sealed document must be a JSON object")
    normal = _normal_json({key: value for key, value in document.items() if key != "sha256"})
    if not isinstance(normal, dict):
        raise Refused("a sealed document must be a JSON object")
    normal.setdefault("program", PROGRAM)
    normal.setdefault("source_commit", commit())
    normal.setdefault("source_digest", source_digest())
    normal.setdefault("activation", ACTIVATION)
    _assert_normalized_activation_false(normal)
    normal["sha256"] = sha_obj(normal)
    return normal


def _is_sealable_normalized_json(value: Any) -> bool:
    """Recognize a finite JSON tree that contains no activation violation."""

    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is dict:
        for key, child in value.items():
            if type(key) is not str:
                return False
            if key == "activation" and child is not False:
                return False
            if not _is_sealable_normalized_json(child):
                return False
        return True
    if value_type is list:
        return all(_is_sealable_normalized_json(child) for child in value)
    return False


def _sealed_normalized_document(document: dict[str, Any]) -> dict[str, JSONValue]:
    """Seal an already-normalized tree without a redundant JSON round trip.

    Internal checkpoint construction supplies exact JSON containers, but public
    event objects remain mutable for compatibility. Fall back to the general
    path when a caller has injected a tuple, subclass, or other non-canonical
    value so checkpoint behavior remains unchanged.
    """

    if not isinstance(document, dict):
        raise Refused("a sealed document must be a JSON object")
    if not _is_sealable_normalized_json(document):
        return sealed_document(document)
    normal = {key: value for key, value in document.items() if key != "sha256"}
    normal.setdefault("program", PROGRAM)
    normal.setdefault("source_commit", commit())
    normal.setdefault("source_digest", source_digest())
    normal.setdefault("activation", ACTIVATION)
    normal["sha256"] = sha_obj(normal)
    return cast(dict[str, JSONValue], normal)


def validate_seal(document: dict[str, Any]) -> dict[str, JSONValue]:
    if not isinstance(document, dict):
        raise Refused("sealed JSON is not an object")
    normal = _normal_json(document)
    if not isinstance(normal, dict):
        raise Refused("sealed JSON is not an object")
    supplied = normal.get("sha256")
    body = {key: value for key, value in normal.items() if key != "sha256"}
    if not isinstance(supplied, str) or supplied != sha_obj(body):
        raise Refused("invalid v5 JSON self-seal")
    if normal.get("program") != PROGRAM:
        raise Refused("sealed JSON is not owned by Substrate v5")
    source_commit = normal.get("source_commit")
    source_digest_value = normal.get("source_digest")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or not isinstance(source_digest_value, str)
        or len(source_digest_value) != 64
    ):
        raise Refused("sealed JSON is missing exact source identity")
    _assert_normalized_activation_false(normal)
    return normal


def _content_path(root: Path, digest: str, *, namespace: str) -> Path:
    if not namespace or Path(namespace).is_absolute() or ".." in Path(namespace).parts:
        raise Refused("content-address namespace must be a safe relative path")
    return _owned_path(root) / namespace / digest[:2] / f"{digest}.json"


def content_addressed_json(
    document: dict[str, Any],
    *,
    root: Path = RUNS,
    namespace: str = "objects",
) -> Path:
    """Seal and immutably store a JSON object at a digest-derived path."""

    owned_root = _owned_path(root)
    sealed = sealed_document(document)
    digest = str(sealed["sha256"])
    destination = _content_path(owned_root, digest, namespace=namespace)
    return atomic_write_bytes(destination, canonical_json(sealed), immutable=True)


def publish_json(
    path: Path,
    document: dict[str, Any],
    *,
    object_namespace: str = ".objects",
) -> Path:
    """Atomically publish a named sealed document and its immutable authority."""

    destination = _owned_path(path)
    if destination.suffix != ".json":
        raise Refused("named v5 JSON publications must end in .json")
    sealed = sealed_document(document)
    owner = next(
        root
        for root in roots()
        if destination.resolve(strict=False).is_relative_to(root.resolve(strict=False))
    )
    digest = str(sealed["sha256"])
    object_path = _content_path(owner, digest, namespace=object_namespace)
    payload = canonical_json(sealed)
    atomic_write_bytes(object_path, payload, immutable=True)
    return atomic_write_bytes(destination, payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refused(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, JSONValue]:
    source = _owned_path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Refused(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Refused(f"invalid v5 JSON at {source}: {error}") from error
    return validate_seal(value)


def seal(name: str, document: dict[str, Any], *, artifact: bool = False) -> Path:
    root = ARTIFACTS if artifact else EVIDENCE
    return publish_json(root / name, document)


def load(name: str, *, artifact: bool = False) -> dict[str, JSONValue]:
    root = ARTIFACTS if artifact else EVIDENCE
    return load_json(root / name)


def run_json(relative: str, document: dict[str, Any]) -> Path:
    return publish_json(RUNS / relative, document)


def config_json(relative: str, document: dict[str, Any]) -> Path:
    return publish_json(CONFIGS / relative, document)


def stop() -> Path:
    return atomic_write(STOP, "operator stop\n")


def resume() -> None:
    _owned_path(STOP).unlink(missing_ok=True)
