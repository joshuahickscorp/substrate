"""Production candidate/control adapters for a sealed Substrate Odyssey.

The Odyssey worker owns scheduling and custody.  This module owns the one
operation that the worker intentionally leaves to a scientific arm: turn a
candidate-visible task into a locally-modelled response and a durable receipt.

It implements two deliberately different, but resource-matched, persistence
laws:

* ``candidate`` is an L1-style flat exact associative monolith.  It keeps one
  mutable content-addressed microstore and rewrites entries as new visible
  experience is assimilated.  It has no hierarchy, topology, learned model
  weights, hidden evaluator input, or tool side channel.
* ``control`` keeps only append-only task/response history and retrieves that
  history deterministically.  It never builds a mutable fact, summary,
  competence, or unfinished-work projection.

Both arms make exactly one request to the same Ollama-compatible local HTTP
endpoint with the same generation options, transport budget, substantive answer
bound, retries, timeouts, and bounded context window.  Their only intended
causal difference is whether the context comes from mutable owned associations
or append-only project history.

Protocol v2 separates the *transport generation budget* (``GENERATION_NUM_PREDICT``)
from the frozen *substantive answer payload* bound (``MAX_OUTPUT_TOKENS``).
Reasoning models such as ``gpt-oss:20b`` charge thinking tokens against
``num_predict``; the transport budget covers that.  After parse, the adapter
enforces the 64-token substantive ceiling on required answer fields only.
Transport repairs may fix syntax wrappers; they never invent or correct answers.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from substrate.evidence import canonical_source_digest
from substrate.odyssey_density import (
    GLOBAL_OVERLAP,
    PINNED_OLLAMA_NUM_PARALLEL,
    assert_ollama_num_parallel_pinned,
)
from substrate.odyssey_density import (
    DensityRefused as _DensityRefused,
)

REQUEST_SCHEMA = "SUBSTRATE_ODYSSEY_ADAPTER_REQUEST/v1"
RECEIPT_SCHEMA = "SUBSTRATE_ODYSSEY_ADAPTER_RECEIPT/v1"
OUTPUT_SCHEMA = "SUBSTRATE_ODYSSEY_ARM_OUTPUT/v1"
EVENT_SCHEMA = "SUBSTRATE_ODYSSEY_ARM_EVENT/v1"
PROGRAM = "substrate-odyssey-7d-v1"
FRONTIERS = frozenset("ABCDEFGH")
PHASES = frozenset(("retrieval", "exposure", "transfer", "repair_checkpoint"))
ROLES = frozenset(("candidate", "control"))
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# Frozen substantive answer payload bound (G06 envelope / parity token_budget).
# This is enforced after JSON parse on required receipt fields; it is NOT the
# raw Ollama generation cap.
MAX_OUTPUT_TOKENS = 64
# Transport generation budget for Ollama num_predict.  Reasoning models charge
# thinking tokens against num_predict.  Host measurement of gpt-oss:20b with the
# same endpoint/options produced valid JSON at eval_count 109/102/375 across
# three frontier shapes; 256 was insufficient.  1024 freezes headroom above the
# observed 375 maximum without expanding the substantive answer bound.
GENERATION_NUM_PREDICT = 1024
# Byte-identical request policy for both arms.  Any asymmetry is a parity
# violation and must refuse.
ARM_REQUEST_TIMEOUT_SECONDS = 900.0
# Single attempt: no retry ladder that could leak task information or diverge
# between candidate and control.  Both arms share this exact constant.
ARM_TRANSPORT_ATTEMPTS = 1
MAX_CONTEXT_BYTES = 8 * 1024
MAX_CONTEXT_ITEMS = 12
MAX_MEMORY_UPDATES = 16
MAX_UNFINISHED = 16
MAX_COMPETENCE = 16
MAX_MEMORY_VALUE_BYTES = 2 * 1024
MAX_RESPONSE_BYTES = 256 * 1024

_FORBIDDEN_PATH_TOKENS = ("evaluator", "answer", "scorer", "hidden")
_FORBIDDEN_TASK_KEYS = _FORBIDDEN_PATH_TOKENS
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOKEN = re.compile(r"[a-z0-9_]{2,}")
# Transport-only wrappers that may be stripped without changing answer values.
_MARKDOWN_FENCE = re.compile(r"^```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*$", re.IGNORECASE | re.DOTALL)
# Channel / protocol markers that must never appear in arm content.
_CHANNEL_MARKER = re.compile(
    r"<\|(?:channel|message|end|start|assistant|system|user|redacted_reasoning)\|>|"
    r"</?(?:think|thinking|reasoning|analysis|final)>|"
    r"(?:^|\n)\s*(?:analysis|final|commentary)\s+to\s*=",
    re.IGNORECASE,
)


class Refused(RuntimeError):
    """Raised when an arm would cross an execution or custody boundary."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Refused(f"{label} must be non-empty text")
    return value.strip()


def _nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Refused(f"{label} must be a non-negative integer")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{label} must contain a JSON object")
    return value


def _contains_forbidden_task_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                return True
            lowered = key.casefold()
            if any(token in lowered for token in _FORBIDDEN_TASK_KEYS):
                return True
            if _contains_forbidden_task_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_task_key(item) for item in value)
    return False


def _assert_plain_relative(value: object, *, label: str, allow_empty: bool = False) -> Path:
    text = _nonempty_text(value, label=label) if not allow_empty else str(value)
    path = Path(text)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise Refused(f"{label} must be a non-escaping root-relative path")
    if any(any(token in part.casefold() for token in _FORBIDDEN_PATH_TOKENS) for part in path.parts):
        raise Refused(f"{label} may not name an evaluator-only namespace")
    return path


def _safe_existing_or_created_directory(root: Path, relative: Path, *, label: str) -> Path:
    """Resolve a directory without accepting a symlink in its path."""
    current = root
    for part in relative.parts:
        candidate = current / part
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            try:
                candidate.mkdir(mode=0o700)
            except OSError as error:
                raise Refused(f"cannot create {label}: {error}") from error
            metadata = candidate.lstat()
        except OSError as error:
            raise Refused(f"cannot inspect {label}: {error}") from error
        if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise Refused(f"{label} must be a real directory path")
        current = candidate
    return current


def _safe_existing_file_parent(root: Path, relative: Path, *, label: str) -> Path:
    if len(relative.parts) < 2:
        raise Refused(f"{label} must have a directory parent below the repository root")
    parent = _safe_existing_or_created_directory(root, Path(*relative.parts[:-1]), label=f"{label} parent")
    target = parent / relative.name
    if target.is_symlink():
        raise Refused(f"{label} must not be a symlink")
    return target


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise Refused(f"refusing to replace symlinked output {path}")
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)
    except OSError as error:
        raise Refused(f"cannot durably write {path}: {error}") from error


def _append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one durable event, or validate the same event after a restart."""
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    request_sha = event["request_sha256"]
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise Refused("arm event log must be a regular file")
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if isinstance(row, dict) and row.get("request_sha256") == request_sha:
                    if row != event:
                        raise Refused("existing arm event does not match the request")
                    return
        except (OSError, json.JSONDecodeError) as error:
            raise Refused(f"cannot validate arm event log: {error}") from error
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise Refused(f"cannot append arm event log: {error}") from error


def _document_size(value: object) -> int:
    return len(canonical(value))


def _tokens(value: object) -> set[str]:
    return set(_TOKEN.findall(json.dumps(value, sort_keys=True, ensure_ascii=False).casefold()))


def _bounded_context(rows: list[dict[str, Any]], task: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose a deterministic, bounded state/history window for either arm."""
    task_tokens = _tokens(task)
    ranked: list[tuple[int, int, str, dict[str, Any]]] = []
    for ordinal, row in enumerate(rows):
        score = len(task_tokens & _tokens(row))
        stable = digest(row)
        ranked.append((-score, ordinal, stable, row))
    ranked.sort()
    selected: list[dict[str, Any]] = []
    used = 2  # JSON list brackets.
    for _negative_score, _ordinal, _stable, row in ranked[:MAX_CONTEXT_ITEMS]:
        row_size = _document_size(row)
        separator = 1 if selected else 0
        if selected and used + separator + row_size > MAX_CONTEXT_BYTES:
            continue
        if not selected and row_size > MAX_CONTEXT_BYTES:
            # The record is locally generated and normally bounded already.
            # Do not slice a JSON object into invalid context if a corrupted
            # historical row somehow exceeds the sealed context allowance.
            continue
        selected.append(row)
        used += separator + row_size
    return selected


def _request_digest(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("request_sha256", None)
    if not _is_sha256(claimed) or claimed != digest(unsigned):
        raise Refused("adapter request has an invalid request_sha256")
    return claimed


def validate_request(root: Path, request_path: Path, *, expected_role: str) -> tuple[dict[str, Any], Path]:
    """Validate the exact worker request surface before loading any state."""
    if expected_role not in ROLES:
        raise Refused("adapter role is invalid")
    requested = request_path if request_path.is_absolute() else root / request_path
    if requested.is_symlink():
        raise Refused("adapter request must be a regular non-symlink file")
    try:
        raw_request = requested.resolve(strict=True)
    except OSError as error:
        raise Refused(f"adapter request is unreadable: {error}") from error
    try:
        raw_request.relative_to(root)
    except ValueError as error:
        raise Refused("adapter request must live below the repository root") from error
    if raw_request.is_symlink() or not raw_request.is_file():
        raise Refused("adapter request must be a regular non-symlink file")
    request = _read_json(raw_request, label="adapter request")
    required = {
        "schema",
        "activation",
        "authority_sha256",
        "run_id",
        "frontier",
        "role",
        "cycle",
        "phase",
        "task",
        "candidate_manifest_sha256",
        "receipt_path",
        "request_sha256",
    }
    if set(request) != required:
        raise Refused("adapter request fields are not exact")
    if request.get("schema") != REQUEST_SCHEMA or request.get("activation") is not False:
        raise Refused("adapter request has the wrong schema or activation state")
    if not _is_sha256(request.get("authority_sha256")):
        raise Refused("adapter request authority digest is invalid")
    if not _is_sha256(request.get("candidate_manifest_sha256")):
        raise Refused("adapter request candidate-manifest digest is invalid")
    _nonempty_text(request.get("run_id"), label="adapter request run_id")
    frontier = request.get("frontier")
    if frontier not in FRONTIERS:
        raise Refused("adapter request frontier is invalid")
    if request.get("role") != expected_role:
        raise Refused("adapter request role differs from the pinned adapter role")
    _nonnegative_int(request.get("cycle"), label="adapter request cycle")
    if request.get("phase") not in PHASES:
        raise Refused("adapter request phase is invalid")
    task = request.get("task")
    if not isinstance(task, dict) or _contains_forbidden_task_key(task):
        raise Refused("adapter request task is malformed or exposes evaluator-only material")
    if task.get("activation") is not False or task.get("frontier") != frontier or task.get("program") != PROGRAM:
        raise Refused("adapter request task has the wrong program, frontier, or activation state")
    _nonempty_text(task.get("task_id"), label="adapter task_id")
    required_receipt = task.get("required_receipt")
    if (
        not isinstance(required_receipt, list)
        or not required_receipt
        or len(set(required_receipt)) != len(required_receipt)
        or not all(isinstance(name, str) and _FIELD_NAME.fullmatch(name) for name in required_receipt)
    ):
        raise Refused("adapter task required_receipt fields are invalid")
    receipt_relative = _assert_plain_relative(request.get("receipt_path"), label="adapter receipt_path")
    receipt_path = _safe_existing_file_parent(root, receipt_relative, label="adapter receipt_path")
    _request_digest(request)
    return request, receipt_path


def _assert_self_digest(expected: str) -> str:
    if not _is_sha256(expected):
        raise Refused("adapter --self-sha256 must be a sha256 digest")
    observed = canonical_source_digest(Path(__file__).resolve())
    if observed != expected:
        raise Refused("adapter source drifted from its sealed --self-sha256")
    return observed


@contextlib.contextmanager
def _state_lock(state_root: Path) -> Iterator[None]:
    lock_path = state_root / "state.lock"
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as error:
        raise Refused(f"cannot open arm state lock: {error}") from error
    try:
        os.chmod(lock_path, 0o600)
        with os.fdopen(descriptor, "r+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as error:
                raise Refused(f"cannot lock arm state: {error}") from error
            yield
    finally:
        # ``handle`` closes on context exit and releases the lock.  If opening
        # the text wrapper failed, descriptor may already be invalid; closing
        # it again would mask the real refusal.
        pass


def _open_database(state_root: Path) -> sqlite3.Connection:
    path = state_root / "state.sqlite3"
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise Refused("arm state database must be a regular non-symlink file")
    try:
        connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        os.chmod(path, 0o600)
        return connection
    except (OSError, sqlite3.Error) as error:
        raise Refused(f"cannot open arm state database: {error}") from error


def _initialize_database(connection: sqlite3.Connection, *, role: str) -> None:
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS dispatches (
                request_sha256 TEXT PRIMARY KEY NOT NULL,
                task_id TEXT NOT NULL,
                cycle INTEGER NOT NULL,
                phase TEXT NOT NULL,
                output_path TEXT NOT NULL,
                output_sha256 TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                state_before_sha256 TEXT NOT NULL,
                state_after_sha256 TEXT NOT NULL
            )"""
        )
        if role == "candidate":
            connection.execute(
                """CREATE TABLE IF NOT EXISTS associations (
                    address_sha256 TEXT PRIMARY KEY NOT NULL,
                    kind TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    provenance_json TEXT NOT NULL,
                    update_count INTEGER NOT NULL,
                    last_cycle INTEGER NOT NULL,
                    last_phase TEXT NOT NULL
                )"""
            )
        else:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS history (
                    request_sha256 TEXT PRIMARY KEY NOT NULL,
                    task_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    output_sha256 TEXT NOT NULL
                )"""
            )
    except sqlite3.Error as error:
        raise Refused(f"cannot initialize arm state schema: {error}") from error


def _meta_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def _bind_state_metadata(
    connection: sqlite3.Connection,
    *,
    role: str,
    request: dict[str, Any],
    model: str,
    adapter_sha256: str,
) -> None:
    expected = {
        "schema": "SUBSTRATE_ODYSSEY_ARM_STATE/v1",
        "role": role,
        "authority_sha256": request["authority_sha256"],
        "run_id": request["run_id"],
        "frontier": request["frontier"],
        "model": model,
        "adapter_sha256": adapter_sha256,
    }
    try:
        connection.execute("BEGIN IMMEDIATE")
        for key, value in expected.items():
            existing = _meta_value(connection, key)
            if existing is not None and existing != value:
                raise Refused(f"arm state metadata drifted: {key}")
            if existing is None:
                connection.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (key, value))
        connection.execute("COMMIT")
    except Refused:
        connection.execute("ROLLBACK")
        raise
    except sqlite3.Error as error:
        connection.execute("ROLLBACK")
        raise Refused(f"cannot bind arm state metadata: {error}") from error


def _json_load(value: str, *, label: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise Refused(f"durable {label} is invalid JSON") from error


def _candidate_state_digest(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT address_sha256, kind, memory_key, value_json, confidence, provenance_json, update_count, last_cycle, last_phase "
        "FROM associations ORDER BY address_sha256"
    ).fetchall()
    material = [
        {
            "address_sha256": str(row["address_sha256"]),
            "kind": str(row["kind"]),
            "memory_key": str(row["memory_key"]),
            "value": _json_load(str(row["value_json"]), label="candidate association value"),
            "confidence": float(row["confidence"]),
            "provenance": _json_load(str(row["provenance_json"]), label="candidate association provenance"),
            "update_count": int(row["update_count"]),
            "last_cycle": int(row["last_cycle"]),
            "last_phase": str(row["last_phase"]),
        }
        for row in rows
    ]
    return digest({"role": "candidate", "associations": material})


def _control_state_digest(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT request_sha256, task_id, cycle, phase, task_json, response_json, output_sha256 FROM history ORDER BY request_sha256"
    ).fetchall()
    material = [
        {
            "request_sha256": str(row["request_sha256"]),
            "task_id": str(row["task_id"]),
            "cycle": int(row["cycle"]),
            "phase": str(row["phase"]),
            "task": _json_load(str(row["task_json"]), label="control history task"),
            "response": _json_load(str(row["response_json"]), label="control history response"),
            "output_sha256": str(row["output_sha256"]),
        }
        for row in rows
    ]
    return digest({"role": "control", "history": material})


def _state_digest(connection: sqlite3.Connection, *, role: str) -> str:
    return _candidate_state_digest(connection) if role == "candidate" else _control_state_digest(connection)


def _candidate_context(connection: sqlite3.Connection, task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT address_sha256, kind, memory_key, value_json, confidence, provenance_json, update_count, last_cycle, last_phase "
        "FROM associations ORDER BY last_cycle DESC, last_phase DESC, address_sha256"
    ).fetchall()
    available = [
        {
            "address": str(row["address_sha256"]),
            "kind": str(row["kind"]),
            "key": str(row["memory_key"]),
            "value": _json_load(str(row["value_json"]), label="candidate association value"),
            "confidence": float(row["confidence"]),
            "provenance": _json_load(str(row["provenance_json"]), label="candidate association provenance"),
            "updates": int(row["update_count"]),
            "cycle": int(row["last_cycle"]),
            "phase": str(row["last_phase"]),
        }
        for row in rows
    ]
    return _bounded_context(available, task)


def _control_context(connection: sqlite3.Connection, task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT request_sha256, task_id, cycle, phase, task_json, response_json FROM history ORDER BY cycle DESC, phase DESC, request_sha256"
    ).fetchall()
    available = [
        {
            "request_sha256": str(row["request_sha256"]),
            "task_id": str(row["task_id"]),
            "cycle": int(row["cycle"]),
            "phase": str(row["phase"]),
            "task": _json_load(str(row["task_json"]), label="control history task"),
            "response": _json_load(str(row["response_json"]), label="control history response"),
        }
        for row in rows
    ]
    return _bounded_context(available, task)


def _declared_tool_operations(request: dict[str, Any]) -> frozenset[str]:
    """Resolve the closed, frontier-scoped tool surface for this request."""
    from substrate.odyssey_tools import declared_operations_for_task

    return declared_operations_for_task(request["task"], frontier=str(request["frontier"]))


def _system_prompt(required_fields: list[str], *, frontier: str, declared_operations: frozenset[str]) -> str:
    from substrate.odyssey_tools import allowed_operations_prompt_block

    fields = ", ".join(required_fields)
    tool_rule = allowed_operations_prompt_block(frontier, declared_operations)
    return (
        "You are a local, deterministic Odyssey reasoning organ. Work only from the visible task, the supplied "
        "bounded prior context, and admitted tool results. "
        f"{tool_rule} "
        "Do not use hidden answer keys, evaluator material, or unstated facts. "
        "Return one JSON object, with every required response field present: "
        f"{fields}. You may additionally emit candidate-safe memory_updates, unfinished, competence, "
        "and tool_proposals fields. "
        "Do not include markdown fences. "
        f"Keep the required answer fields extremely compact: their combined substantive payload must fit within "
        f"{MAX_OUTPUT_TOKENS} tokens (short phrases, minimal derivation). Prefer terse values over prose."
    )


def _prompt_messages(request: dict[str, Any], context: list[dict[str, Any]]) -> list[dict[str, str]]:
    task = request["task"]
    declared = _declared_tool_operations(request)
    payload = {
        "task": task,
        "prior_context": context,
        "context_limit_bytes": MAX_CONTEXT_BYTES,
        "allowed_operations": sorted(declared),
        "output_contract": {
            "required_fields": task["required_receipt"],
            "optional_fields": ["memory_updates", "unfinished", "competence", "tool_proposals"],
        },
    }
    return [
        {
            "role": "system",
            "content": _system_prompt(
                task["required_receipt"],
                frontier=str(request["frontier"]),
                declared_operations=declared,
            ),
        },
        {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
    ]


def _paired_seed(request: dict[str, Any]) -> int:
    """Derive the same fixed seed for the two arms of one paired task."""
    pairing = {
        "authority_sha256": request["authority_sha256"],
        "run_id": request["run_id"],
        "frontier": request["frontier"],
        "cycle": request["cycle"],
        "phase": request["phase"],
        "task_id": request["task"]["task_id"],
        "candidate_manifest_sha256": request["candidate_manifest_sha256"],
    }
    return int(digest(pairing)[:8], 16)


def _transport_refuse(reason: str) -> None:
    raise Refused(f"transport: {reason}")


def _semantic_refuse(reason: str) -> None:
    raise Refused(f"semantic: {reason}")


def _strip_transport_wrappers(content: str) -> str:
    """Remove markdown fences only.  Never alter answer values or fill fields."""
    text = content.strip()
    match = _MARKDOWN_FENCE.fullmatch(text)
    if match is not None:
        return match.group(1).strip()
    return text


def _parse_json_object_transport(text: str) -> dict[str, Any]:
    """Transport gate: UTF-8 text → exact JSON object, no non-finite constants."""

    def reject_constant(token: str) -> Any:
        _transport_refuse(f"non-finite JSON constant {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _value in pairs:
            if key in seen:
                _transport_refuse(f"duplicate JSON key {key!r}")
            seen.add(key)
        return dict(pairs)

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        _transport_refuse(f"content is not valid JSON: {error}")
    except Refused:
        raise
    if not isinstance(value, dict):
        _transport_refuse("content must be a JSON object")
    return value


def _assert_transport_content(raw_content: str) -> tuple[dict[str, Any], str, bool]:
    """Validate transport properties of model content.

    Returns ``(object, cleaned_content, fence_stripped)``.  Transport repair is
    limited to stripping a single surrounding markdown fence.  Prose, channel
    markers, truncated JSON, and non-objects remain transport failures.
    """
    if not isinstance(raw_content, str):
        _transport_refuse("message content is not a string")
    try:
        raw_content.encode("utf-8")
    except UnicodeEncodeError as error:
        _transport_refuse(f"content is not valid UTF-8: {error}")
    if len(raw_content.encode("utf-8")) > MAX_RESPONSE_BYTES:
        _transport_refuse("content exceeds the arm response bound")
    if not raw_content.strip():
        _transport_refuse("content is empty (likely truncated by generation budget into thinking only)")
    if _CHANNEL_MARKER.search(raw_content):
        _transport_refuse("content contains channel or protocol markers")
    cleaned = _strip_transport_wrappers(raw_content)
    fence_stripped = cleaned != raw_content.strip()
    # Leading analysis / trailing prose around a JSON object is not repaired.
    # Allow only a pure JSON object after optional fence strip.
    if not cleaned.startswith("{"):
        _transport_refuse("content is not a pure JSON object (prose, fence residue, or premature EOS)")
    parsed = _parse_json_object_transport(cleaned)
    # Round-trip canonical serialization proves the object is plain JSON data.
    try:
        canonical(parsed)
    except (TypeError, ValueError) as error:
        _transport_refuse(f"content is not canonically serializable: {error}")
    return parsed, cleaned, fence_stripped


def _count_substantive_tokens(response: dict[str, Any], required_fields: list[str]) -> int:
    """Approximate token count of the substantive required-field payload.

    Optional developmental fields (memory_updates, unfinished, competence) are
    bounded separately and do not consume the frozen 64-token answer envelope.
    Counts whitespace-delimited tokens over the serialized required fields only
    (JSON structure included), which is a standard approximate ceiling when the
    host has no model tokenizer in-process.
    """
    payload = {field: response[field] for field in required_fields}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    tokens = text.split()
    return len(tokens) if tokens else 0


def _validate_semantic_response(response: dict[str, Any], required_fields: list[str]) -> int:
    """Semantic gate: required fields present, non-null, model-produced, bounded.

    Never fills missing fields, never substitutes defaults, never rewrites
    values.  A semantic failure is a genuine model failure.
    """
    if not isinstance(response, dict):
        _semantic_refuse("response is not an object")
    missing = [field for field in required_fields if field not in response]
    if missing:
        _semantic_refuse(f"missing required task fields: {missing}")
    for field in required_fields:
        value = response[field]
        if value is None:
            _semantic_refuse(f"null required task field: {field}")
        if isinstance(value, str) and not value.strip():
            _semantic_refuse(f"empty required task field: {field}")
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            _semantic_refuse(f"non-finite required task field: {field}")
    token_count = _count_substantive_tokens(response, required_fields)
    if token_count > MAX_OUTPUT_TOKENS:
        _semantic_refuse(
            f"substantive answer payload is {token_count} tokens; "
            f"frozen bound is {MAX_OUTPUT_TOKENS}"
        )
    return token_count


def _validate_model_response(response: dict[str, Any], required_fields: list[str]) -> int:
    """Backward-compatible semantic validation entrypoint."""
    return _validate_semantic_response(response, required_fields)


def _generation_options(*, seed: int) -> dict[str, Any]:
    """Byte-identical generation options for every arm call."""
    return {
        "temperature": 0,
        "seed": seed,
        "num_predict": GENERATION_NUM_PREDICT,
    }


def _ollama_chat(
    *,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    seed: int,
    timeout_seconds: float = ARM_REQUEST_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the narrow local Ollama-compatible HTTP seam used by both arms.

    Candidate and control share this exact function, budget, options, attempt
    count, and timeout.  Content is read from ``message.content`` only;
    ``message.thinking`` length is recorded as diagnostic telemetry and is
    never treated as answer content.
    """
    if not url.startswith("http://127.0.0.1:") and not url.startswith("http://localhost:"):
        raise Refused("arm model endpoint must be a loopback Ollama-compatible URL")
    if ARM_TRANSPORT_ATTEMPTS != 1:
        raise Refused("arm transport attempt count drifted from the parity constant")
    if timeout_seconds != ARM_REQUEST_TIMEOUT_SECONDS:
        # Callers may not widen or shrink one arm's budget independently.
        raise Refused("arm request timeout drifted from the parity constant")
    # Density contract: refuse if the shared gateway is serializing (or has any
    # parallel-slot value other than the pinned contract).  Do not silently
    # inherit an operator-only environment.
    try:
        pin_status = assert_ollama_num_parallel_pinned(require_running=True)
    except _DensityRefused as error:
        raise Refused(f"model gateway pin: {error}") from error
    options = _generation_options(seed=seed)
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        # gpt-oss:20b is a reasoning model.  With think=false it often returns
        # empty or garbage content while still consuming eval tokens; with
        # think=true it emits message.thinking (telemetry only) and places the
        # JSON answer in message.content.  Thinking tokens still charge against
        # num_predict — hence GENERATION_NUM_PREDICT.  Both arms share this
        # exact flag; never promote thinking into the answer.
        "think": True,
        "format": "json",
        "messages": messages,
        "options": options,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=canonical(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with (
            GLOBAL_OVERLAP.span(lane_id="arm", arm="shared", kind="model"),
            urllib.request.urlopen(request, timeout=timeout_seconds) as response,  # noqa: S310 - checked loopback endpoint
        ):
                raw_body = response.read()
                raw = json.loads(raw_body.decode("utf-8"))
    except TimeoutError as error:
        raise Refused(f"transport: local Ollama request timed out: {error}") from error
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise Refused(f"transport: local Ollama request failed: {error}") from error
    # pin_status retained for resource_usage attachment by callers when present.
    _ = pin_status
    if not isinstance(raw, dict):
        _transport_refuse("local Ollama response must be an object")
    message = raw.get("message")
    if not isinstance(message, dict):
        _transport_refuse("local Ollama response lacks message object")
    # Content is the only answer channel.  Thinking is telemetry only.
    content = message.get("content")
    if not isinstance(content, str):
        _transport_refuse("local Ollama response lacks message content")
    thinking_raw = message.get("thinking")
    thinking = thinking_raw if isinstance(thinking_raw, str) else ""
    # Never promote thinking into the answer.
    parsed, _cleaned, fence_stripped = _assert_transport_content(content)

    def observed_integer(name: str) -> int | None:
        value = raw.get(name)
        if value is None:
            return None
        return _nonnegative_int(value, label=f"local Ollama {name}")

    usage: dict[str, Any] = {
        "prompt_eval_count": observed_integer("prompt_eval_count"),
        "eval_count": observed_integer("eval_count"),
        "total_duration_ns": observed_integer("total_duration"),
        "load_duration_ns": observed_integer("load_duration"),
        "eval_duration_ns": observed_integer("eval_duration"),
        "thinking_char_count": len(thinking),
        "content_char_count": len(content),
        "generation_num_predict": GENERATION_NUM_PREDICT,
        "substantive_max_tokens": MAX_OUTPUT_TOKENS,
        "transport_gate": "pass",
        "transport_fence_stripped": fence_stripped,
        "transport_attempts": ARM_TRANSPORT_ATTEMPTS,
        "request_timeout_seconds": ARM_REQUEST_TIMEOUT_SECONDS,
        "ollama_num_parallel_pinned": PINNED_OLLAMA_NUM_PARALLEL,
    }
    return parsed, usage


def _bounded_memory_updates(response: dict[str, Any]) -> list[dict[str, Any]]:
    raw = response.get("memory_updates", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > MAX_MEMORY_UPDATES:
        raise Refused("candidate memory_updates are invalid")
    updates: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or set(row) - {"kind", "key", "value", "confidence"}:
            raise Refused(f"candidate memory update {index} has unsupported fields")
        kind = _nonempty_text(row.get("kind", "fact"), label=f"candidate memory update {index}.kind")
        key = _nonempty_text(row.get("key"), label=f"candidate memory update {index}.key")
        if len(kind) > 128 or len(key) > 512 or _document_size(row.get("value")) > MAX_MEMORY_VALUE_BYTES:
            raise Refused("candidate memory update exceeds its bounded storage allowance")
        confidence = row.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
            raise Refused(f"candidate memory update {index}.confidence is invalid")
        updates.append({"kind": kind, "key": key, "value": row.get("value"), "confidence": float(confidence)})
    return updates


def _bounded_unfinished(response: dict[str, Any]) -> list[str]:
    raw = response.get("unfinished", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > MAX_UNFINISHED:
        raise Refused("candidate unfinished work list is invalid")
    values = [_nonempty_text(item, label="candidate unfinished work") for item in raw]
    if any(len(item) > 1024 for item in values):
        raise Refused("candidate unfinished work entry is too large")
    return values


def _bounded_competence(response: dict[str, Any]) -> dict[str, Any]:
    raw = response.get("competence", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or len(raw) > MAX_COMPETENCE:
        raise Refused("candidate competence update is invalid")
    result: dict[str, Any] = {}
    for key, value in raw.items():
        name = _nonempty_text(key, label="candidate competence key")
        if len(name) > 128 or _document_size(value) > MAX_MEMORY_VALUE_BYTES:
            raise Refused("candidate competence update exceeds its bounded storage allowance")
        result[name] = value
    return result


def _association_address(*, frontier: str, kind: str, key: str) -> str:
    return digest({"frontier": frontier, "kind": kind, "key": key})


def _upsert_association(
    connection: sqlite3.Connection,
    *,
    frontier: str,
    kind: str,
    key: str,
    value: Any,
    confidence: float,
    provenance: list[str],
    cycle: int,
    phase: str,
) -> bool:
    address = _association_address(frontier=frontier, kind=kind, key=key)
    existing = connection.execute(
        "SELECT value_json, confidence, provenance_json, update_count FROM associations WHERE address_sha256 = ?", (address,)
    ).fetchone()
    value_json = canonical(value).decode("utf-8")
    provenance_json = canonical(sorted(set(provenance))).decode("utf-8")
    if existing is not None and (
        str(existing["value_json"]) == value_json and float(existing["confidence"]) == confidence and str(existing["provenance_json"]) == provenance_json
    ):
        return False
    count = 1 if existing is None else int(existing["update_count"]) + 1
    connection.execute(
        """INSERT INTO associations(address_sha256, kind, memory_key, value_json, confidence, provenance_json, update_count, last_cycle, last_phase)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(address_sha256) DO UPDATE SET
             value_json = excluded.value_json,
             confidence = excluded.confidence,
             provenance_json = excluded.provenance_json,
             update_count = excluded.update_count,
             last_cycle = excluded.last_cycle,
             last_phase = excluded.last_phase""",
        (address, kind, key, value_json, confidence, provenance_json, count, cycle, phase),
    )
    return True


def _apply_candidate_state(
    connection: sqlite3.Connection,
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    output_sha256: str,
) -> dict[str, int]:
    """Apply L1's flat exact associative update law and unconditional consolidation."""
    task = request["task"]
    updates = _bounded_memory_updates(response)
    unfinished = _bounded_unfinished(response)
    competence = _bounded_competence(response)
    writes = 0
    revisions = 0

    def write(**kwargs: Any) -> None:
        nonlocal writes, revisions
        address = _association_address(frontier=request["frontier"], kind=kwargs["kind"], key=kwargs["key"])
        was_existing = connection.execute("SELECT 1 FROM associations WHERE address_sha256 = ?", (address,)).fetchone() is not None
        if _upsert_association(connection, frontier=request["frontier"], cycle=request["cycle"], phase=request["phase"], **kwargs):
            writes += 1
            revisions += int(was_existing)

    # The visible episode itself is an exact association.  This means the
    # treatment has a real persistent update even if the model elects to emit
    # no optional memory hints.
    write(
        kind="episode",
        key=task["task_id"],
        value={"task": task, "response": response, "output_sha256": output_sha256},
        confidence=1.0,
        provenance=[task["task_id"]],
    )
    for update in updates:
        write(
            kind=update["kind"],
            key=update["key"],
            value=update["value"],
            confidence=update["confidence"],
            provenance=[task["task_id"]],
        )
    # These are all entries in the same flat associative table, not a field,
    # graph, compiled procedure, or additional learned component.
    write(
        kind="unfinished",
        key="open-work",
        value=unfinished,
        confidence=1.0,
        provenance=[task["task_id"]],
    )
    for name, value in sorted(competence.items()):
        write(
            kind="competence",
            key=name,
            value=value,
            confidence=0.5,
            provenance=[task["task_id"]],
        )
    # L1's unconditional consolidation: a bounded materialized association is
    # refreshed on every accepted visible episode.  It carries no rule fitting
    # or topology, just exact addresses and current work state.
    recent = connection.execute(
        "SELECT address_sha256 FROM associations ORDER BY last_cycle DESC, last_phase DESC, address_sha256 LIMIT ?",
        (MAX_CONTEXT_ITEMS,),
    ).fetchall()
    write(
        kind="consolidated",
        key="current",
        value={
            "last_task_id": task["task_id"],
            "recent_addresses": [str(row["address_sha256"]) for row in recent],
            "unfinished_count": len(unfinished),
        },
        confidence=1.0,
        provenance=[task["task_id"]],
    )
    return {"associations_written": writes, "association_revisions": revisions, "history_events_appended": 0}


def _apply_control_state(
    connection: sqlite3.Connection,
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    output_sha256: str,
) -> dict[str, int]:
    """Append one immutable history row; intentionally no mutable projection."""
    connection.execute(
        "INSERT INTO history(request_sha256, task_id, cycle, phase, task_json, response_json, output_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            request["request_sha256"],
            request["task"]["task_id"],
            request["cycle"],
            request["phase"],
            canonical(request["task"]).decode("utf-8"),
            canonical(response).decode("utf-8"),
            output_sha256,
        ),
    )
    return {"associations_written": 0, "association_revisions": 0, "history_events_appended": 1}


def _output_path(state_root: Path, request: dict[str, Any]) -> Path:
    name = f"{int(request['cycle']):03d}-{request['phase']}.json"
    output_root = state_root / "outputs"
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir()):
        raise Refused("arm output directory must be a real directory")
    output_root.mkdir(mode=0o700, exist_ok=True)
    target = output_root / name
    if target.is_symlink():
        raise Refused("arm output artifact must not be a symlink")
    return target


def _build_output(
    *,
    request: dict[str, Any],
    role: str,
    model: str,
    adapter_sha256: str,
    messages: list[dict[str, str]],
    response: dict[str, Any],
    resource_usage: dict[str, int | None],
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body = {
        "schema": OUTPUT_SCHEMA,
        "activation": False,
        "authority_sha256": request["authority_sha256"],
        "run_id": request["run_id"],
        "frontier": request["frontier"],
        "role": role,
        "cycle": request["cycle"],
        "phase": request["phase"],
        "task_id": request["task"]["task_id"],
        "request_sha256": request["request_sha256"],
        "candidate_manifest_sha256": request["candidate_manifest_sha256"],
        "adapter_sha256": adapter_sha256,
        "model": model,
        "prompt_sha256": digest({"messages": messages}),
        "response": response,
        "resource_usage": resource_usage,
        "tool_results": list(tool_results or []),
    }
    body["sha256"] = digest(body)
    return body


def _execute_response_tools(
    root: Path,
    *,
    request: dict[str, Any],
    role: str,
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate and execute optional model tool_proposals through the shared broker.

    Tool/CPU work is recorded on the overlap ledger so concurrent lanes can
    hold the GPU while this arm runs formal checks, hashing, and media metadata.
    """
    proposals = response.get("tool_proposals")
    if proposals is None:
        return []
    if not isinstance(proposals, list):
        raise Refused("tool_proposals must be a list when present")
    if not proposals:
        return []
    from substrate.odyssey_tools import ToolBudget, ToolRefused, budget_for_frontier, execute_tool_proposals

    frontier = str(request.get("frontier", ""))
    # Frontier resource class — candidate and control of a pair share the same
    # frontier id so budgets remain byte-identical.
    try:
        budget = budget_for_frontier(frontier) if frontier else ToolBudget()
    except Exception:
        budget = ToolBudget()
    lane = frontier or "arm"
    try:
        with GLOBAL_OVERLAP.span(lane_id=lane, arm=role, kind="tool"):
            return execute_tool_proposals(
                root=root,
                request=request,
                role=role,
                proposals=proposals,
                budget=budget,
                peer_budget_sha256=budget.budget_sha256(),
            )
    except ToolRefused as error:
        raise Refused(f"tool: {error}") from error


def _existing_output(path: Path, request: dict[str, Any], *, role: str, adapter_sha256: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    output = _read_json(path, label="existing arm output")
    unsigned = dict(output)
    claimed = unsigned.pop("sha256", None)
    checks = {
        "digest": _is_sha256(claimed) and claimed == digest(unsigned),
        "schema": output.get("schema") == OUTPUT_SCHEMA,
        "request": output.get("request_sha256") == request["request_sha256"],
        "role": output.get("role") == role,
        "adapter": output.get("adapter_sha256") == adapter_sha256,
    }
    if not all(checks.values()):
        raise Refused("existing arm output cannot be reused for this request")
    return output


def _receipt_from_existing(connection: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any] | None:
    row = connection.execute("SELECT receipt_json FROM dispatches WHERE request_sha256 = ?", (request["request_sha256"],)).fetchone()
    if row is None:
        return None
    receipt = _json_load(str(row["receipt_json"]), label="existing arm receipt")
    if not isinstance(receipt, dict):
        raise Refused("existing arm receipt has invalid shape")
    unsigned = dict(receipt)
    claimed = unsigned.pop("sha256", None)
    if not _is_sha256(claimed) or claimed != digest(unsigned):
        raise Refused("existing arm receipt has invalid self-digest")
    if receipt.get("request_sha256") != request["request_sha256"]:
        raise Refused("existing arm receipt belongs to a different request")
    return receipt


def _ensure_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path, label="existing adapter receipt")
        if existing != receipt:
            raise Refused("existing adapter receipt differs from durable arm state")
        return
    _atomic_write_json(path, receipt)


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError as error:
        raise Refused("arm artifact escaped the repository root") from error


def run(
    root: Path,
    *,
    role: str,
    model: str,
    state_root: str,
    self_sha256: str,
    request_path: Path,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    """Run one pinned candidate/control dispatch and durably write its receipt."""
    root = root.resolve()
    if role not in ROLES:
        raise Refused("adapter role is invalid")
    model = _nonempty_text(model, label="adapter model")
    adapter_sha256 = _assert_self_digest(self_sha256)
    request, receipt_path = validate_request(root, request_path, expected_role=role)
    state_relative = _assert_plain_relative(state_root, label="adapter state_root")
    state_directory = _safe_existing_or_created_directory(root, state_relative, label="adapter state_root")
    # A command can be copied between rows by mistake.  The request itself is
    # the final binding of the lane, and a role-specific root prevents it from
    # becoming a shared candidate/control cache.
    if state_directory == receipt_path.parent or state_directory in receipt_path.parents:
        raise Refused("adapter state root must not be the worker receipt directory")
    started = time.monotonic()
    with _state_lock(state_directory):
        connection = _open_database(state_directory)
        try:
            _initialize_database(connection, role=role)
            _bind_state_metadata(
                connection,
                role=role,
                request=request,
                model=model,
                adapter_sha256=adapter_sha256,
            )
            existing_receipt = _receipt_from_existing(connection, request)
            event_path = state_directory / "events.jsonl"
            if existing_receipt is not None:
                _append_event(
                    event_path,
                    {
                        "schema": EVENT_SCHEMA,
                        "activation": False,
                        "authority_sha256": request["authority_sha256"],
                        "run_id": request["run_id"],
                        "frontier": request["frontier"],
                        "role": role,
                        "cycle": request["cycle"],
                        "phase": request["phase"],
                        "task_id": request["task"]["task_id"],
                        "request_sha256": request["request_sha256"],
                        "output_sha256": existing_receipt["response_sha256"],
                        "state_before_sha256": existing_receipt["state_before_sha256"],
                        "state_after_sha256": existing_receipt["state_after_sha256"],
                    },
                )
                _ensure_receipt(receipt_path, existing_receipt)
                return existing_receipt

            before = _state_digest(connection, role=role)
            context = _candidate_context(connection, request["task"]) if role == "candidate" else _control_context(connection, request["task"])
            messages = _prompt_messages(request, context)
            seed = _paired_seed(request)
            output_path = _output_path(state_directory, request)
            output = _existing_output(output_path, request, role=role, adapter_sha256=adapter_sha256)
            required_fields = list(request["task"]["required_receipt"])
            tool_results: list[dict[str, Any]] = []
            if output is None:
                # Transport gate runs inside _ollama_chat.  Semantic gate is
                # separate and must never be satisfied by adapter defaults.
                response, usage = _ollama_chat(
                    url=ollama_url,
                    model=model,
                    messages=messages,
                    seed=seed,
                    timeout_seconds=ARM_REQUEST_TIMEOUT_SECONDS,
                )
                substantive_tokens = _validate_semantic_response(response, required_fields)
                usage = dict(usage)
                usage["semantic_gate"] = "pass"
                usage["substantive_token_count"] = substantive_tokens
                # Model proposes → broker validates → sandbox executes → cache
                # quarantines → verifier admits.  Only admitted results are kept.
                tool_results = _execute_response_tools(root, request=request, role=role, response=response)
                usage["tool_calls_admitted"] = len(tool_results)
                output = _build_output(
                    request=request,
                    role=role,
                    model=model,
                    adapter_sha256=adapter_sha256,
                    messages=messages,
                    response=response,
                    resource_usage=usage,
                    tool_results=tool_results,
                )
                _atomic_write_json(output_path, output)
            else:
                response = output.get("response")
                if not isinstance(response, dict):
                    raise Refused("existing arm output lacks a response object")
                substantive_tokens = _validate_semantic_response(response, required_fields)
                usage_value = output.get("resource_usage")
                usage = dict(usage_value) if isinstance(usage_value, dict) else {}
                usage.setdefault("semantic_gate", "pass")
                usage.setdefault("substantive_token_count", substantive_tokens)
                usage.setdefault("substantive_max_tokens", MAX_OUTPUT_TOKENS)
                usage.setdefault("generation_num_predict", GENERATION_NUM_PREDICT)
                existing_tools = output.get("tool_results")
                if isinstance(existing_tools, list):
                    tool_results = existing_tools
                usage.setdefault("tool_calls_admitted", len(tool_results))
            output_sha256 = output["sha256"]
            if not _is_sha256(output_sha256):
                raise Refused("arm output self-digest is invalid")
            output_file_sha256 = file_digest(output_path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                if role == "candidate":
                    state_change = _apply_candidate_state(
                        connection,
                        request=request,
                        response=response,
                        output_sha256=output_sha256,
                    )
                else:
                    state_change = _apply_control_state(
                        connection,
                        request=request,
                        response=response,
                        output_sha256=output_sha256,
                    )
                after = _state_digest(connection, role=role)
                elapsed = round(max(0.0, time.monotonic() - started), 6)
                receipt: dict[str, Any] = {
                    "schema": RECEIPT_SCHEMA,
                    "activation": False,
                    "authority_sha256": request["authority_sha256"],
                    "run_id": request["run_id"],
                    "frontier": request["frontier"],
                    "role": role,
                    "cycle": request["cycle"],
                    "phase": request["phase"],
                    "task_id": request["task"]["task_id"],
                    "candidate_manifest_sha256": request["candidate_manifest_sha256"],
                    "request_sha256": request["request_sha256"],
                    "elapsed_seconds": elapsed,
                    "adapter_sha256": adapter_sha256,
                    "model": {"id": model, "endpoint": ollama_url},
                    "output_artifacts": [{"path": _relative(root, output_path), "sha256": output_file_sha256}],
                    "response_sha256": output_sha256,
                    "state_before_sha256": before,
                    "state_after_sha256": after,
                    "state_change": {
                        "mode": "flat_exact_associative_monolith" if role == "candidate" else "append_only_history_retrieval",
                        **state_change,
                    },
                    "resource_usage": usage,
                    "tool_results": [
                        {
                            "operation": row.get("operation"),
                            "receipt_sha256": row.get("receipt_sha256"),
                            "output_digests": row.get("output_digests"),
                            "admitted": row.get("admitted"),
                            "tool_revision": row.get("tool_revision"),
                        }
                        for row in tool_results
                        if isinstance(row, dict)
                    ],
                }
                receipt["sha256"] = digest(receipt)
                connection.execute(
                    """INSERT INTO dispatches(request_sha256, task_id, cycle, phase, output_path, output_sha256, receipt_json,
                       state_before_sha256, state_after_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        request["request_sha256"],
                        request["task"]["task_id"],
                        request["cycle"],
                        request["phase"],
                        _relative(root, output_path),
                        output_sha256,
                        canonical(receipt).decode("utf-8"),
                        before,
                        after,
                    ),
                )
                connection.execute("COMMIT")
            except Refused:
                connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as error:
                connection.execute("ROLLBACK")
                raise Refused(f"cannot commit arm state: {error}") from error
            event = {
                "schema": EVENT_SCHEMA,
                "activation": False,
                "authority_sha256": request["authority_sha256"],
                "run_id": request["run_id"],
                "frontier": request["frontier"],
                "role": role,
                "cycle": request["cycle"],
                "phase": request["phase"],
                "task_id": request["task"]["task_id"],
                "request_sha256": request["request_sha256"],
                "output_sha256": output_sha256,
                "state_before_sha256": before,
                "state_after_sha256": after,
            }
            _append_event(event_path, event)
            _ensure_receipt(receipt_path, receipt)
            with contextlib.suppress(sqlite3.Error):
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                # The committed database and receipt are already durable.  A
                # failed opportunistic compaction must not rewrite an observed
                # result or conceal the valid dispatch.
            return receipt
        finally:
            connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--self-sha256", required=True)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.root,
            role=args.role,
            model=args.model,
            state_root=args.state_root,
            self_sha256=args.self_sha256,
            request_path=args.request,
            ollama_url=args.ollama_url,
        )
    except Refused as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
