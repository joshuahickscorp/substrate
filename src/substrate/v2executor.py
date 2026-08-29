"""Supervisor owned immutable receipt publication for Substrate v2.

Workers return documents.  Only the supervisor validates source, configuration, split, seed, activation,
and content identity before atomically publishing a unit receipt.

"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager, suppress
from pathlib import Path

from substrate import v2config as C
from substrate import v2io as io


class Refused(RuntimeError):
    """An execution or publication context that fails closed."""


def frozen_configuration() -> dict:
    path = io.CONFIGS / "frozen_configuration.json"
    document = json.loads(path.read_text())
    expected_seal = io.sha_obj({key: value for key, value in document.items() if key != "sha256"})
    if document.get("sha256") != expected_seal:
        raise Refused("frozen configuration seal mismatch")
    live = C.configuration()
    if document.get("configuration_digest") != live["configuration_digest"]:
        raise Refused("frozen configuration differs from executable configuration")
    return document


def context() -> dict:
    configuration = frozen_configuration()
    return {
        "source_digest": io.source_digest(),
        "configuration_digest": configuration["configuration_digest"],
        "split_digest": io.sha_obj(configuration["splits"]),
        "activation": False,
    }


def validate_context(
    supplied: dict,
    *,
    split: str,
    seed: int,
    expected: dict | None = None,
) -> dict:
    expected = expected or context()
    violations = []
    for key in ("source_digest", "configuration_digest", "split_digest"):
        if supplied.get(key) != expected[key]:
            violations.append(key)
    if supplied.get("activation") is not False:
        violations.append("activation")
    if split not in C.SPLITS:
        violations.append("split")
    elif seed not in C.SPLITS[split]:
        violations.append("seed")
    if violations:
        raise Refused(f"execution context refused: {sorted(set(violations))}")
    return expected


@contextmanager
def exclusive_claim(lock: Path):
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise Refused(f"duplicate live claim for {lock.stem}") from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        with suppress(OSError):
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def receipt_body(identity: str, payload: dict, supplied_context: dict) -> dict:
    body = {
        "schema": "substrate-v2-unit-receipt/v1",
        "identity": identity,
        "payload": payload,
        **supplied_context,
        "activation": False,
    }
    body["receipt_sha256"] = io.sha_obj(body)
    return body


def validate_receipt(document: dict) -> bool:
    received = document.get("receipt_sha256")
    body = {key: value for key, value in document.items() if key != "receipt_sha256"}
    return (
        received == io.sha_obj(body)
        and document.get("activation") is False
        and document.get("schema") == "substrate-v2-unit-receipt/v1"
    )


def publish_unit(relative: str, document: dict) -> dict:
    """Publish once.  Byte identical repeats are cache hits and divergent repeats are refused."""
    if not validate_receipt(document):
        raise Refused("unit receipt validation failed")
    target = io.RUNS / relative
    encoded = json.dumps(document, indent=2)
    if target.is_file():
        prior = json.loads(target.read_text())
        if prior == document:
            return {"published": False, "cache_hit": True, "path": target.relative_to(io.ROOT).as_posix()}
        raise Refused(f"divergent duplicate publication for {relative}")
    io.atomic_write(target, encoded)
    return {"published": True, "cache_hit": False, "path": target.relative_to(io.ROOT).as_posix()}
