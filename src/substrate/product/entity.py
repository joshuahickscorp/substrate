"""Portable entity state with a hash-linked developmental receipt ledger."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported product hosts are POSIX.
    fcntl = None  # type: ignore[assignment]

from substrate.product.codec import (
    atomic_write_json,
    atomic_write_json_lines,
    durable_unlink,
    fsync_directory,
    read_json,
    read_json_lines,
    sha256,
)
from substrate.product.contracts import PRODUCT_SCHEMA_VERSION, EntityManifest, ProductRefused
from substrate.product.packs import resolve_packs

MANIFEST_NAME = "entity.json"
STATE_NAME = "developmental-state.json"
LEDGER_NAME = "receipts.jsonl"
CHECKPOINT_NAME = "checkpoint.json"
PENDING_TRANSACTION_NAME = ".pending-transaction.json"
WRITER_LOCK_NAME = ".writer.lock"
CREATION_LOCK_SUFFIX = ".creation.lock"


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductRefused(message)


def _entry_exists(path: Path) -> bool:
    """Check for an entry without treating a broken symlink as absent."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ProductRefused(f"cannot inspect {path.name}: {exc}") from exc
    return True


def _require_entity_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductRefused(f"entity directory does not exist: {path}") from exc
    _require(
        stat.S_ISDIR(metadata.st_mode) and not path.is_symlink(),
        f"entity directory does not exist: {path}",
    )


def _lock_file_is_safe(metadata: os.stat_result) -> bool:
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


@contextmanager
def _exclusive_file_lock(path: Path, description: str):
    """Take an advisory lock without following or sharing a lock-file link.

    The lock files are coordination artifacts, not authority.  Still, refusing
    links and non-regular files prevents a malformed entity directory from
    turning a normal record/create operation into an arbitrary file open.
    """

    if fcntl is None:
        raise ProductRefused("the product entity store requires a POSIX file-lock backend")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ProductRefused("the product entity store requires O_NOFOLLOW for its lock files")
    descriptor: int | None = None
    locked = False
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | no_follow | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        metadata = os.fstat(descriptor)
        _require(_lock_file_is_safe(metadata), f"cannot acquire {description} lock: expected a regular non-linked file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        metadata = os.fstat(descriptor)
        _require(_lock_file_is_safe(metadata), f"cannot acquire {description} lock: lock file changed while waiting")
        path_metadata = path.lstat()
        _require(
            _lock_file_is_safe(path_metadata)
            and (path_metadata.st_dev, path_metadata.st_ino) == (metadata.st_dev, metadata.st_ino),
            f"cannot acquire {description} lock: lock file changed while waiting",
        )
        yield
    except ProductRefused:
        raise
    except OSError as exc:
        raise ProductRefused(f"cannot acquire {description} lock: {exc}") from exc
    finally:
        if descriptor is not None:
            if locked:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _parse_manifest_document(document: dict[str, Any], description: str) -> EntityManifest:
    """Reject manifest fields that the portable format does not serialize.

    Receipt and checkpoint hashes bind the canonical manifest.  Without this
    equality check, an added JSON field would be dropped by ``from_dict`` and
    would evade that binding on load.  This is deliberately entity-local so it
    does not change the public manifest schema or migrate existing entities.
    """

    manifest = EntityManifest.from_dict(document)
    _require(document == manifest.to_dict(), f"{description} contains undeclared or noncanonical fields")
    resolve_packs(manifest.selected_packs)
    return manifest


def _default_state(manifest: EntityManifest) -> dict[str, Any]:
    return {
        "active_apprenticeship": None,
        "active_apprenticeship_plan_sha256": None,
        "active_source_policy_sha256": None,
        "competence": {},
        "entity_id": manifest.entity_id,
        "evidence_assimilated": 0,
        "phase": "initialized",
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "unfinished_work": [],
    }


def _validate_state(state: dict[str, Any], entity_id: str) -> None:
    _require(state.get("schema_version") == PRODUCT_SCHEMA_VERSION, "unsupported developmental-state schema version")
    _require(state.get("entity_id") == entity_id, "developmental state belongs to a different entity")
    _require(isinstance(state.get("phase"), str) and bool(state["phase"]), "developmental state phase must be nonempty")
    _require(state.get("active_apprenticeship") is None or isinstance(state["active_apprenticeship"], str), "active apprenticeship is invalid")
    for field in ("active_apprenticeship_plan_sha256", "active_source_policy_sha256"):
        value = state.get(field)
        valid_hash = isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        _require(value is None or valid_hash, f"{field} is invalid")
    if state.get("phase") == "apprenticeship_planned":
        _require(isinstance(state.get("active_apprenticeship_plan_sha256"), str), "planned apprenticeship is missing its plan hash")
        _require(isinstance(state.get("active_source_policy_sha256"), str), "planned apprenticeship is missing its source policy hash")
    _require(isinstance(state.get("competence"), dict), "competence must be an object")
    _require(
        isinstance(state.get("evidence_assimilated"), int)
        and not isinstance(state["evidence_assimilated"], bool)
        and state["evidence_assimilated"] >= 0,
        "evidence count is invalid",
    )
    _require(isinstance(state.get("unfinished_work"), list) and all(isinstance(item, str) for item in state["unfinished_work"]), "unfinished work is invalid")


def _unsigned_receipt(sequence: int, kind: str, payload: dict[str, Any], previous_sha256: str | None) -> dict[str, Any]:
    _require(isinstance(sequence, int) and sequence > 0, "receipt sequence must be positive")
    _require(isinstance(kind, str) and bool(kind), "receipt kind must be nonempty")
    _require(isinstance(payload, dict), "receipt payload must be an object")
    return {
        "kind": kind,
        "payload": payload,
        "previous_sha256": previous_sha256,
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "sequence": sequence,
        "timestamp": _timestamp(),
    }


def _seal_receipt(sequence: int, kind: str, payload: dict[str, Any], previous_sha256: str | None) -> dict[str, Any]:
    receipt = _unsigned_receipt(sequence, kind, payload, previous_sha256)
    return {**receipt, "sha256": sha256(receipt)}


def _verify_ledger(rows: list[dict[str, Any]]) -> str | None:
    _require(bool(rows), "receipt ledger cannot be empty")
    previous_sha256: str | None = None
    for expected_sequence, receipt in enumerate(rows, start=1):
        _require(receipt.get("schema_version") == PRODUCT_SCHEMA_VERSION, "receipt has an unsupported schema version")
        _require(receipt.get("sequence") == expected_sequence, "receipt sequence is not contiguous")
        _require(receipt.get("previous_sha256") == previous_sha256, "receipt previous hash does not match the ledger")
        _require(isinstance(receipt.get("kind"), str) and bool(receipt["kind"]), "receipt kind is invalid")
        _require(isinstance(receipt.get("payload"), dict), "receipt payload is invalid")
        _require(isinstance(receipt.get("timestamp"), str) and bool(receipt["timestamp"]), "receipt timestamp is invalid")
        recorded_sha256 = receipt.get("sha256")
        unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
        _require(isinstance(recorded_sha256, str) and recorded_sha256 == sha256(unsigned), "receipt hash verification failed")
        previous_sha256 = recorded_sha256
    return previous_sha256


def _checkpoint(manifest: EntityManifest, state: dict[str, Any], rows: list[dict[str, Any]], tail_sha256: str | None) -> dict[str, Any]:
    return {
        "developmental_state_sha256": sha256(state),
        "ledger_length": len(rows),
        "ledger_tail_sha256": tail_sha256,
        "manifest_sha256": sha256(manifest.to_dict()),
        "schema_version": PRODUCT_SCHEMA_VERSION,
    }


def _transaction(manifest: EntityManifest, state: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    tail_sha256 = _verify_ledger(rows)
    document = {
        "checkpoint": _checkpoint(manifest, state, rows, tail_sha256),
        "manifest": manifest.to_dict(),
        "manifest_sha256": sha256(manifest.to_dict()),
        "receipts": rows,
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "state": state,
    }
    return {**document, "sha256": sha256(document)}


def _validated_transaction(
    document: dict[str, Any], current_manifest: EntityManifest | None
) -> tuple[EntityManifest, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    recorded_sha256 = document.get("sha256")
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    _require(document.get("schema_version") == PRODUCT_SCHEMA_VERSION, "pending transaction has an unsupported schema version")
    _require(isinstance(recorded_sha256, str) and recorded_sha256 == sha256(unsigned), "pending transaction hash verification failed")
    manifest_value = document.get("manifest")
    if not isinstance(manifest_value, dict):
        raise ProductRefused("pending transaction manifest is invalid")
    manifest = _parse_manifest_document(manifest_value, "pending transaction manifest")
    _require(document.get("manifest_sha256") == sha256(manifest.to_dict()), "pending transaction manifest hash is invalid")
    if current_manifest is not None:
        _require(sha256(current_manifest.to_dict()) == sha256(manifest.to_dict()), "pending transaction belongs to a different entity manifest")
    state = document.get("state")
    rows = document.get("receipts")
    checkpoint = document.get("checkpoint")
    if not isinstance(state, dict):
        raise ProductRefused("pending transaction state is invalid")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ProductRefused("pending transaction receipt ledger is invalid")
    if not isinstance(checkpoint, dict):
        raise ProductRefused("pending transaction checkpoint is invalid")
    _validate_state(state, manifest.entity_id)
    tail_sha256 = _verify_ledger(rows)
    _require(checkpoint == _checkpoint(manifest, state, rows, tail_sha256), "pending transaction checkpoint is invalid")
    return manifest, state, rows, checkpoint


@dataclass(frozen=True)
class EntitySnapshot:
    manifest: EntityManifest
    state: dict[str, Any]
    receipts: tuple[dict[str, Any], ...]
    checkpoint: dict[str, Any]

    @property
    def revision_sha256(self) -> str:
        """Stable compare-and-commit token for this verified snapshot."""

        return sha256(self.checkpoint)

    def status(self) -> dict[str, Any]:
        return {
            "entity": self.manifest.to_dict(),
            "execution_permitted": False,
            "ledger": {"receipt_count": len(self.receipts), "tail_sha256": self.receipts[-1]["sha256"]},
            "state": self.state,
            "valid": True,
        }


class EntityStore:
    """An entity-root-local store. It does not use campaign evidence paths."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def state_path(self) -> Path:
        return self.root / STATE_NAME

    @property
    def ledger_path(self) -> Path:
        return self.root / LEDGER_NAME

    @property
    def checkpoint_path(self) -> Path:
        return self.root / CHECKPOINT_NAME

    @property
    def pending_transaction_path(self) -> Path:
        return self.root / PENDING_TRANSACTION_NAME

    @property
    def writer_lock_path(self) -> Path:
        return self.root / WRITER_LOCK_NAME

    @property
    def creation_lock_path(self) -> Path:
        return self.root.parent / f".{self.root.name}{CREATION_LOCK_SUFFIX}"

    @contextmanager
    def _writer_lock(self):
        with _exclusive_file_lock(self.writer_lock_path, "entity writer"):
            yield

    @contextmanager
    def _creation_lock(self):
        with _exclusive_file_lock(self.creation_lock_path, "entity creation"):
            yield

    @classmethod
    def create(cls, root: Path, manifest: EntityManifest) -> EntityStore:
        store = cls(root)
        resolve_packs(manifest.selected_packs)
        try:
            store.root.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProductRefused(f"cannot prepare entity parent directory: {exc}") from exc
        with store._creation_lock():
            _require(not _entry_exists(store.root), f"entity directory already exists: {store.root}")
            try:
                staging_root = Path(tempfile.mkdtemp(prefix=f".{store.root.name}.staging-", dir=store.root.parent))
            except OSError as exc:
                raise ProductRefused(f"cannot create entity staging directory: {exc}") from exc
            staging_store = cls(staging_root)
            try:
                state = _default_state(manifest)
                receipt = _seal_receipt(1, "entity_initialized", {"manifest_sha256": sha256(manifest.to_dict())}, None)
                rows = [receipt]
                transaction = _transaction(manifest, state, rows)
                atomic_write_json(staging_store.pending_transaction_path, transaction)
                atomic_write_json(staging_store.manifest_path, manifest.to_dict())
                atomic_write_json(staging_store.state_path, state)
                atomic_write_json_lines(staging_store.ledger_path, rows)
                atomic_write_json(staging_store.checkpoint_path, transaction["checkpoint"])
                durable_unlink(staging_store.pending_transaction_path)
                staging_store.validate()
                os.replace(staging_store.root, store.root)
                fsync_directory(store.root.parent)
                store.validate()
                return store
            except ProductRefused:
                if _entry_exists(staging_root) and not _entry_exists(store.root):
                    shutil.rmtree(staging_root, ignore_errors=True)
                raise
            except OSError as exc:
                if _entry_exists(staging_root) and not _entry_exists(store.root):
                    shutil.rmtree(staging_root, ignore_errors=True)
                raise ProductRefused(f"cannot publish initialized entity directory: {exc}") from exc

    def load(self) -> EntitySnapshot:
        _require_entity_directory(self.root)
        _require(not _entry_exists(self.pending_transaction_path), "entity has a pending transaction; call recover before loading it")
        manifest = _parse_manifest_document(read_json(self.manifest_path), "entity manifest")
        state = read_json(self.state_path)
        _validate_state(state, manifest.entity_id)
        rows = read_json_lines(self.ledger_path)
        tail_sha256 = _verify_ledger(rows)
        initial_receipt = rows[0]
        _require(initial_receipt["kind"] == "entity_initialized", "receipt ledger has no initialization receipt")
        _require(
            initial_receipt["payload"] == {"manifest_sha256": sha256(manifest.to_dict())},
            "initialization receipt does not bind the entity manifest",
        )
        checkpoint = read_json(self.checkpoint_path)
        expected_checkpoint = _checkpoint(manifest, state, rows, tail_sha256)
        _require(checkpoint == expected_checkpoint, "checkpoint does not match the verified state and receipt ledger")
        return EntitySnapshot(manifest=manifest, state=state, receipts=tuple(rows), checkpoint=checkpoint)

    def validate(self) -> dict[str, Any]:
        snapshot = self.load()
        return {
            "entity_id": snapshot.manifest.entity_id,
            "ledger_tail_sha256": snapshot.receipts[-1]["sha256"],
            "receipt_count": len(snapshot.receipts),
            "valid": True,
        }

    def status(self) -> dict[str, Any]:
        return self.load().status()

    def record(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        state_update: dict[str, Any] | None = None,
        expected_checkpoint_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Append a verified receipt, then checkpoint its explicitly supplied state update.

        This is intentionally a single-writer operation. Future workers return
        receipts to this method rather than writing the entity directory.
        """

        if expected_checkpoint_sha256 is not None:
            _require(
                isinstance(expected_checkpoint_sha256, str)
                and len(expected_checkpoint_sha256) == 64
                and all(character in "0123456789abcdef" for character in expected_checkpoint_sha256),
                "expected entity snapshot revision is invalid",
            )
        if state_update is not None:
            _require(
                expected_checkpoint_sha256 is not None,
                "state updates require an expected entity snapshot revision",
            )
        with self._writer_lock():
            snapshot = self.load()
            if expected_checkpoint_sha256 is not None:
                _require(
                    snapshot.revision_sha256 == expected_checkpoint_sha256,
                    "entity changed since its expected snapshot; reload and retry",
                )
            state = dict(snapshot.state)
            if state_update is not None:
                _require(isinstance(state_update, dict), "state update must be an object")
                allowed_state_fields = {
                    "active_apprenticeship",
                    "active_apprenticeship_plan_sha256",
                    "active_source_policy_sha256",
                    "competence",
                    "evidence_assimilated",
                    "phase",
                    "unfinished_work",
                }
                unknown_state_fields = sorted(set(state_update) - allowed_state_fields)
                _require(not unknown_state_fields, f"state update touches undeclared fields: {', '.join(unknown_state_fields)}")
                state.update(state_update)
            _validate_state(state, snapshot.manifest.entity_id)
            receipt = _seal_receipt(
                len(snapshot.receipts) + 1,
                kind,
                payload,
                snapshot.receipts[-1]["sha256"],
            )
            rows = [*snapshot.receipts, receipt]
            transaction = _transaction(snapshot.manifest, state, rows)
            atomic_write_json(self.pending_transaction_path, transaction)
            atomic_write_json_lines(self.ledger_path, rows)
            atomic_write_json(self.state_path, state)
            atomic_write_json(self.checkpoint_path, transaction["checkpoint"])
            durable_unlink(self.pending_transaction_path)
            self.validate()
            return receipt

    def recover(self) -> dict[str, Any]:
        """Explicitly apply a complete, verified transaction left by an interrupted write."""

        with self._writer_lock():
            _require(_entry_exists(self.pending_transaction_path), "entity has no pending transaction to recover")
            _require(_entry_exists(self.manifest_path), "entity manifest is missing; cannot recover a pending transaction")
            current_manifest = _parse_manifest_document(read_json(self.manifest_path), "entity manifest")
            manifest, state, rows, checkpoint = _validated_transaction(read_json(self.pending_transaction_path), current_manifest)
            atomic_write_json(self.manifest_path, manifest.to_dict())
            atomic_write_json_lines(self.ledger_path, rows)
            atomic_write_json(self.state_path, state)
            atomic_write_json(self.checkpoint_path, checkpoint)
            durable_unlink(self.pending_transaction_path)
            return self.validate()
