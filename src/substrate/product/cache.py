"""Content-addressed quarantine and immutable artifact cache.

This is the non-executing evidence boundary for the product runtime.  It only
admits bytes supplied by a local operator path; it performs no fetching,
archive extraction, media decode, browser operation, or tool invocation.

Objects enter quarantine first.  A typed, explicit attestation plus a content
digest/length/type re-check is required before an object moves into a verified
or processed immutable zone.  Portable entities retain only object digests and
safe plan digests, never arbitrary file paths or credential-bearing URLs.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from substrate.product.codec import atomic_write_json, fsync_directory, sha256
from substrate.product.contracts import ProductRefused, SourceReceipt

try:
    import fcntl
except ImportError:  # pragma: no cover - mutation requires a POSIX cache host.
    fcntl = None  # type: ignore[assignment]

CACHE_SCHEMA_VERSION = "substrate-artifact-cache-v1"
CACHE_ATTESTATION_SCHEMA_VERSION = "substrate-artifact-cache-attestation-v1"
CACHE_VERIFIER_TRUST_SCHEMA_VERSION = "substrate-artifact-cache-verifier-trust-v1"
CACHE_VERIFICATION_RECEIPT_SCHEMA_VERSION = "substrate-artifact-cache-verification-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_RIGHTS_STATUSES = frozenset({"licensed", "public", "user-provided"})
_VERIFICATION_STATUSES = frozenset({"quarantined", "verified"})
_ARTIFACT_KINDS = frozenset({"derived", "raw"})
_OBJECT_ZONES = ("quarantine", "verified", "processed")
_DIRECTORIES = ("incoming", "quarantine", "verified", "processed", "workspaces", "evaluator-only", "exports", "pins")
_MAXIMUM_MEDIA_BYTES_TO_SNIFF = 8192
# Structured adapter evidence is intentionally much narrower than a generic
# ``+json`` media type.  It stays in the Substrate namespace, must fit within a
# bounded parser window, and must be a strict JSON object.  Other JSON-looking
# bytes remain ordinary ``text/plain`` until a separately typed cache adapter
# is introduced.
_SUBSTRATE_STRUCTURED_JSON_MEDIA_TYPE = re.compile(r"^application/x-substrate-[a-z0-9][a-z0-9._-]{0,63}\+json$")
_MAXIMUM_SUBSTRATE_STRUCTURED_JSON_BYTES = 64 * 1024 * 1024
_VERIFICATION_RECEIPT_NAME = "verification.json"
_PROCESSING_LINEAGE_FIELDS = frozenset({"input_sha256", "recipe_id", "tool_artifact_sha256"})
_ATTESTATION_FIELDS = frozenset(
    {
        "artifact_sha256",
        "cache_id",
        "decision",
        "descriptor_sha256",
        "expires_at",
        "issued_at",
        "key_id",
        "public_key_b64",
        "rights_status",
        "schema_version",
        "signature_b64",
        "verifier_id",
    }
)
_VERIFIER_TRUST_FIELDS = frozenset({"allowed_rights_statuses", "key_id", "public_key_b64", "schema_version", "verifier_id"})
_DESCRIPTOR_FIELDS = frozenset(
    {
        "artifact_kind",
        "byte_length",
        "derived_objects",
        "media_type",
        "processing_lineage",
        "quarantine_reason",
        "retrieved_at",
        "rights_status",
        "schema_version",
        "sha256",
        "source_reference_sha256",
        "verification_receipt_sha256",
        "verification_status",
    }
)
_VERIFICATION_RECEIPT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "attestation",
        "attestation_sha256",
        "attested_descriptor",
        "cache_id",
        "descriptor_sha256",
        "execution_permitted",
        "schema_version",
        "verified_at",
    }
)
# The cache may add reverse derivative references and change transition
# bookkeeping after verification.  All other descriptor fields are immutable
# provenance and must remain equal to the descriptor that the verifier signed.
_MUTABLE_AFTER_VERIFICATION_DESCRIPTOR_FIELDS = frozenset(
    {"derived_objects", "quarantine_reason", "verification_receipt_sha256", "verification_status"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductRefused(message)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProductRefused(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProductRefused(f"{label} must be a lowercase identifier")
    return value


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ProductRefused(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProductRefused(f"{label} must be an ISO-8601 timestamp") from exc
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _validate_timestamp(value: object, label: str) -> str:
    _parse_timestamp(value, label)
    return cast(str, value)


def _public_key_id(public_key_b64: str) -> str:
    return sha256({"algorithm": "ed25519", "public_key_b64": public_key_b64})


def _public_key_b64(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def _load_public_key(path: Path) -> Ed25519PublicKey:
    try:
        descriptor = _open_regular_read(path, "verifier public key", reject_hardlinks=True)
        with os.fdopen(descriptor, "rb") as handle:
            key = serialization.load_pem_public_key(handle.read())
    except (OSError, TypeError, ValueError) as exc:
        raise ProductRefused("verifier public key is not a valid PEM key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ProductRefused("verifier public key must use Ed25519")
    return key


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    try:
        descriptor = _open_regular_read(path, "verifier private key", reject_hardlinks=True)
        with os.fdopen(descriptor, "rb") as handle:
            key = serialization.load_pem_private_key(handle.read(), password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise ProductRefused("verifier private key is not a valid unencrypted PEM key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ProductRefused("verifier private key must use Ed25519")
    return key


def _regular_file(path: Path, label: str, *, reject_hardlinks: bool = False) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductRefused(f"cannot inspect {label}: {exc}") from exc
    _require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(), f"{label} must be a regular non-symlink file")
    if reject_hardlinks:
        _require(metadata.st_nlink == 1, f"{label} must not be a hard-linked file")
    return metadata


def _open_regular_read(path: Path, label: str, *, reject_hardlinks: bool = False) -> int:
    """Open the exact regular file just checked, without following a racey link."""

    expected = _regular_file(path, label, reject_hardlinks=reject_hardlinks)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ProductRefused(f"cannot safely open {label}: {exc}") from exc
    try:
        observed = os.fstat(descriptor)
        _require(
            stat.S_ISREG(observed.st_mode)
            and (not reject_hardlinks or observed.st_nlink == 1)
            and (observed.st_dev, observed.st_ino, observed.st_size) == (expected.st_dev, expected.st_ino, expected.st_size),
            f"{label} changed before it could be opened safely",
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _strict_substrate_json_object(path: Path) -> None:
    """Require a bounded, duplicate-free JSON object for typed adapter data.

    The cache does not infer arbitrary application semantics from text.  This
    helper only recognizes a declared ``application/x-substrate-…+json``
    artifact after proving that the exact blob is strict JSON rather than a
    binary/file masquerading as one.  Schema-specific validation remains the
    responsibility of the source-plan or future output-receipt contract.
    """

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        descriptor = _open_regular_read(path, "cache blob", reject_hardlinks=True)
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read(_MAXIMUM_SUBSTRATE_STRUCTURED_JSON_BYTES + 1)
    except ProductRefused:
        raise
    except OSError as exc:
        raise ProductRefused(f"cannot inspect structured JSON cache blob: {exc}") from exc
    _require(
        len(raw) <= _MAXIMUM_SUBSTRATE_STRUCTURED_JSON_BYTES,
        "Substrate structured JSON cache blob exceeds its maximum size",
    )
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProductRefused("cache blob declared as Substrate structured JSON is not a strict JSON object") from exc
    _require(isinstance(document, dict), "cache blob declared as Substrate structured JSON must contain an object")


def _sniff_media_type(path: Path, *, declared_media_type: str | None = None) -> str:
    """Conservatively identify bytes, including declared structured evidence."""

    if isinstance(declared_media_type, str) and _SUBSTRATE_STRUCTURED_JSON_MEDIA_TYPE.fullmatch(declared_media_type):
        _strict_substrate_json_object(path)
        return declared_media_type
    try:
        descriptor = _open_regular_read(path, "cache blob", reject_hardlinks=True)
        with os.fdopen(descriptor, "rb") as handle:
            prefix = handle.read(_MAXIMUM_MEDIA_BYTES_TO_SNIFF)
    except OSError as exc:
        raise ProductRefused(f"cannot inspect cache blob type: {exc}") from exc
    if prefix.startswith(b"%PDF-"):
        return "application/pdf"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if prefix.startswith(b"PK\x03\x04"):
        return "application/zip"
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WAVE":
        return "audio/wav"
    if prefix.startswith(b"ID3") or prefix.startswith(b"\xff\xfb"):
        return "audio/mpeg"
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        return "video/mp4"
    try:
        decoded = prefix.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    if any(ord(character) < 9 or (13 < ord(character) < 32) for character in decoded):
        return "application/octet-stream"
    return "text/plain"


def _media_type_matches_modality(media_type: str, modality: str) -> bool:
    """Use a conservative cross-check before cache evidence reaches an entity."""

    rules = {
        "audio": lambda: media_type.startswith("audio/"),
        "document": lambda: media_type in {"application/pdf", "text/plain"},
        "image": lambda: media_type.startswith("image/"),
        "repository": lambda: media_type == "text/plain",
        "text": lambda: media_type == "text/plain",
        "three-d": lambda: False,
        "video": lambda: media_type.startswith("video/"),
    }
    return rules.get(modality, lambda: False)()


def _stream_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_length = 0
    try:
        descriptor = _open_regular_read(path, "cache blob", reject_hardlinks=True)
        with os.fdopen(descriptor, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_length += len(chunk)
    except OSError as exc:
        raise ProductRefused(f"cannot hash cache blob: {exc}") from exc
    return digest.hexdigest(), byte_length


@dataclass(frozen=True)
class ProcessingLineage:
    """A typed derived-object recipe; shell commands are intentionally absent."""

    input_sha256: str
    recipe_id: str
    tool_artifact_sha256: str

    def __post_init__(self) -> None:
        _digest(self.input_sha256, "lineage input sha256")
        _identifier(self.recipe_id, "lineage recipe id")
        _digest(self.tool_artifact_sha256, "lineage tool artifact sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "input_sha256": self.input_sha256,
            "recipe_id": self.recipe_id,
            "tool_artifact_sha256": self.tool_artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessingLineage:
        _require(isinstance(value, Mapping), "processing lineage is malformed")
        _require(set(value) == _PROCESSING_LINEAGE_FIELDS, "processing lineage fields are malformed")
        try:
            return cls(
                input_sha256=value["input_sha256"],
                recipe_id=value["recipe_id"],
                tool_artifact_sha256=value["tool_artifact_sha256"],
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("processing lineage is malformed") from exc


@dataclass(frozen=True)
class CacheAttestation:
    """A signed local-verifier approval bound to one exact cache object."""

    artifact_sha256: str
    cache_id: str
    descriptor_sha256: str
    verifier_id: str
    rights_status: str
    issued_at: str
    expires_at: str
    public_key_b64: str
    key_id: str
    signature_b64: str
    decision: str = "approved"
    schema_version: str = CACHE_ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _digest(self.artifact_sha256, "attestation artifact sha256")
        _identifier(self.cache_id, "attestation cache id")
        _digest(self.descriptor_sha256, "attestation descriptor sha256")
        _identifier(self.verifier_id, "attestation verifier id")
        _require(self.rights_status in _RIGHTS_STATUSES, "attestation rights status is unsupported")
        issued_at = _parse_timestamp(self.issued_at, "attestation issuance time")
        expires_at = _parse_timestamp(self.expires_at, "attestation expiry time")
        _require(expires_at > issued_at, "attestation expiry must be after issuance")
        _require(isinstance(self.public_key_b64, str) and bool(self.public_key_b64), "attestation public key is malformed")
        _require(isinstance(self.key_id, str) and bool(_SHA256.fullmatch(self.key_id)), "attestation key id is malformed")
        _require(self.key_id == _public_key_id(self.public_key_b64), "attestation key id does not match its public key")
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(self.public_key_b64, validate=True))
        except (ValueError, binascii.Error, TypeError) as exc:
            raise ProductRefused("attestation public key must be an Ed25519 public key") from exc
        _require(isinstance(self.signature_b64, str) and bool(self.signature_b64), "attestation signature is malformed")
        _require(self.decision == "approved", "cache attestation decision must be approved")
        _require(self.schema_version == CACHE_ATTESTATION_SCHEMA_VERSION, "unsupported cache attestation schema version")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "cache_id": self.cache_id,
            "decision": self.decision,
            "descriptor_sha256": self.descriptor_sha256,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "key_id": self.key_id,
            "public_key_b64": self.public_key_b64,
            "rights_status": self.rights_status,
            "schema_version": self.schema_version,
            "signature_b64": self.signature_b64,
            "verifier_id": self.verifier_id,
        }

    def signing_payload(self) -> dict[str, str]:
        """Return every signed field, excluding only the signature itself."""

        document = self.to_dict()
        del document["signature_b64"]
        return document

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CacheAttestation:
        _require(isinstance(value, Mapping), "cache attestation is malformed")
        _require(set(value) == _ATTESTATION_FIELDS, "cache attestation fields are malformed")
        try:
            return cls(
                artifact_sha256=value["artifact_sha256"],
                cache_id=value["cache_id"],
                descriptor_sha256=value["descriptor_sha256"],
                verifier_id=value["verifier_id"],
                rights_status=value["rights_status"],
                issued_at=value["issued_at"],
                expires_at=value["expires_at"],
                public_key_b64=value["public_key_b64"],
                key_id=value["key_id"],
                signature_b64=value["signature_b64"],
                decision=value["decision"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("cache attestation is malformed") from exc


@dataclass(frozen=True)
class CacheVerifierTrustRule:
    """One local authorization rule for a cache-verifier signing key."""

    verifier_id: str
    key_id: str
    public_key_b64: str
    allowed_rights_statuses: tuple[str, ...]
    schema_version: str = CACHE_VERIFIER_TRUST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.verifier_id, "cache verifier id")
        _digest(self.key_id, "cache verifier key id")
        _require(isinstance(self.public_key_b64, str) and bool(self.public_key_b64), "cache verifier public key is malformed")
        _require(self.key_id == _public_key_id(self.public_key_b64), "cache verifier key id does not match its public key")
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(self.public_key_b64, validate=True))
        except (ValueError, binascii.Error, TypeError) as exc:
            raise ProductRefused("cache verifier public key must be an Ed25519 public key") from exc
        _require(isinstance(self.allowed_rights_statuses, tuple) and bool(self.allowed_rights_statuses), "cache verifier rights scope cannot be empty")
        _require(set(self.allowed_rights_statuses).issubset(_RIGHTS_STATUSES), "cache verifier rights scope is unsupported")
        _require(len(set(self.allowed_rights_statuses)) == len(self.allowed_rights_statuses), "cache verifier rights scope cannot contain duplicates")
        _require(self.allowed_rights_statuses == tuple(sorted(self.allowed_rights_statuses)), "cache verifier rights scope must be sorted")
        _require(self.schema_version == CACHE_VERIFIER_TRUST_SCHEMA_VERSION, "unsupported cache verifier trust schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_rights_statuses": list(self.allowed_rights_statuses),
            "key_id": self.key_id,
            "public_key_b64": self.public_key_b64,
            "schema_version": self.schema_version,
            "verifier_id": self.verifier_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CacheVerifierTrustRule:
        _require(isinstance(value, Mapping), "cache verifier trust rule is malformed")
        _require(set(value) == _VERIFIER_TRUST_FIELDS, "cache verifier trust rule fields are malformed")
        try:
            return cls(
                verifier_id=value["verifier_id"],
                key_id=value["key_id"],
                public_key_b64=value["public_key_b64"],
                allowed_rights_statuses=tuple(value["allowed_rights_statuses"]),
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("cache verifier trust rule is malformed") from exc


class LocalCacheVerifierTrustStore:
    """Local explicit trust for signed cache-verifier attestations."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, verifier_id: str, key_id: str) -> Path:
        return self.root / "verifiers" / _identifier(verifier_id, "cache verifier id") / f"{_digest(key_id, 'cache verifier key id')}.json"

    def trust(
        self,
        *,
        verifier_id: str,
        public_key_path: Path,
        allowed_rights_statuses: tuple[str, ...],
    ) -> CacheVerifierTrustRule:
        public_key_b64 = _public_key_b64(_load_public_key(public_key_path))
        rule = CacheVerifierTrustRule(
            verifier_id=verifier_id,
            key_id=_public_key_id(public_key_b64),
            public_key_b64=public_key_b64,
            allowed_rights_statuses=tuple(sorted(allowed_rights_statuses)),
        )
        target = self._path(rule.verifier_id, rule.key_id)
        if target.exists() or target.is_symlink():
            existing = CacheVerifierTrustRule.from_dict(_read_descriptor(target))
            _require(existing == rule, "refusing to replace an existing cache verifier trust decision")
            return existing
        atomic_write_json(target, rule.to_dict())
        return rule

    def load(self, verifier_id: str, key_id: str) -> CacheVerifierTrustRule:
        target = self._path(verifier_id, key_id)
        _require(target.exists() and not target.is_symlink(), "cache verifier is not trusted locally")
        return CacheVerifierTrustRule.from_dict(_read_descriptor(target))


def _attestation_signature_payload(attestation: CacheAttestation) -> bytes:
    return b"substrate-artifact-cache-attestation-v1\x00" + sha256(attestation.signing_payload()).encode("ascii")


def sign_cache_attestation(
    *,
    artifact_sha256: str,
    cache_id: str,
    descriptor_sha256: str,
    verifier_id: str,
    rights_status: str,
    private_key_path: Path,
    issued_at: str | None = None,
    expires_at: str,
) -> CacheAttestation:
    """Create an explicit signed verifier decision for one cache descriptor."""

    private_key = _load_private_key(private_key_path)
    public_key_b64 = _public_key_b64(private_key.public_key())
    provisional = CacheAttestation(
        artifact_sha256=artifact_sha256,
        cache_id=cache_id,
        descriptor_sha256=descriptor_sha256,
        verifier_id=verifier_id,
        rights_status=rights_status,
        issued_at=_timestamp() if issued_at is None else issued_at,
        expires_at=expires_at,
        public_key_b64=public_key_b64,
        key_id=_public_key_id(public_key_b64),
        signature_b64="pending",
    )
    signature = private_key.sign(_attestation_signature_payload(provisional))
    return replace(provisional, signature_b64=base64.b64encode(signature).decode("ascii"))


def verify_cache_attestation(attestation: CacheAttestation, trust_store: LocalCacheVerifierTrustStore) -> CacheVerifierTrustRule:
    """Verify signature, expiry, and explicit local verifier trust."""

    _require(isinstance(attestation, CacheAttestation), "cache attestation is malformed")
    _require(isinstance(trust_store, LocalCacheVerifierTrustStore), "cache verifier trust store is malformed")
    _require(_parse_timestamp(attestation.expires_at, "attestation expiry time") > datetime.now(UTC), "cache attestation has expired")
    trust = trust_store.load(attestation.verifier_id, attestation.key_id)
    _require(trust.public_key_b64 == attestation.public_key_b64, "local cache verifier key does not match the attestation")
    _require(attestation.rights_status in trust.allowed_rights_statuses, "local cache verifier is not authorized for this rights status")
    try:
        raw_key = base64.b64decode(attestation.public_key_b64, validate=True)
        raw_signature = base64.b64decode(attestation.signature_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(raw_key).verify(raw_signature, _attestation_signature_payload(attestation))
    except (InvalidSignature, ValueError, binascii.Error, TypeError) as exc:
        raise ProductRefused("cache attestation signature verification failed") from exc
    return trust


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Provenance-bearing metadata for immutable cache bytes."""

    sha256: str
    byte_length: int
    media_type: str
    source_reference_sha256: str
    retrieved_at: str
    rights_status: str
    artifact_kind: str = "raw"
    processing_lineage: tuple[ProcessingLineage, ...] = ()
    derived_objects: tuple[str, ...] = ()
    verification_status: str = "quarantined"
    verification_receipt_sha256: str | None = None
    quarantine_reason: str | None = "awaiting-explicit-verification"
    schema_version: str = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _digest(self.sha256, "artifact sha256")
        _require(isinstance(self.byte_length, int) and not isinstance(self.byte_length, bool) and self.byte_length >= 0, "artifact byte length is invalid")
        _require(isinstance(self.media_type, str) and bool(_MEDIA_TYPE.fullmatch(self.media_type)), "artifact media type is invalid")
        _digest(self.source_reference_sha256, "artifact source reference")
        _validate_timestamp(self.retrieved_at, "artifact retrieval time")
        _require(self.rights_status in _RIGHTS_STATUSES, "artifact rights status is unsupported")
        _require(self.artifact_kind in _ARTIFACT_KINDS, "artifact kind is unsupported")
        _require(
            isinstance(self.processing_lineage, tuple) and all(isinstance(item, ProcessingLineage) for item in self.processing_lineage),
            "artifact lineage is malformed",
        )
        _require(
            (self.artifact_kind == "raw" and not self.processing_lineage) or (self.artifact_kind == "derived" and bool(self.processing_lineage)),
            "artifact kind and processing lineage do not agree",
        )
        _require(isinstance(self.derived_objects, tuple), "derived objects must be a tuple")
        _require(all(isinstance(item, str) and bool(_SHA256.fullmatch(item)) for item in self.derived_objects), "derived object digest is invalid")
        _require(len(set(self.derived_objects)) == len(self.derived_objects), "derived object digests cannot contain duplicates")
        _require(self.verification_status in _VERIFICATION_STATUSES, "artifact verification status is unsupported")
        _require(
            self.verification_receipt_sha256 is None or bool(_SHA256.fullmatch(self.verification_receipt_sha256)),
            "verification receipt digest is invalid",
        )
        if self.verification_status == "quarantined":
            _require(isinstance(self.quarantine_reason, str) and bool(self.quarantine_reason.strip()), "quarantined artifact requires a reason")
            _require(self.verification_receipt_sha256 is None, "quarantined artifact cannot retain a verification receipt")
        else:
            _require(self.quarantine_reason is None, "verified artifact cannot retain a quarantine reason")
            _require(self.verification_receipt_sha256 is not None, "verified artifact requires a verification receipt")
        _require(self.schema_version == CACHE_SCHEMA_VERSION, "unsupported artifact cache schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "byte_length": self.byte_length,
            "derived_objects": list(self.derived_objects),
            "media_type": self.media_type,
            "processing_lineage": [item.to_dict() for item in self.processing_lineage],
            "quarantine_reason": self.quarantine_reason,
            "retrieved_at": self.retrieved_at,
            "rights_status": self.rights_status,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "source_reference_sha256": self.source_reference_sha256,
            "verification_receipt_sha256": self.verification_receipt_sha256,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactDescriptor:
        _require(isinstance(value, Mapping), "artifact descriptor is malformed")
        _require(set(value) == _DESCRIPTOR_FIELDS, "artifact descriptor fields are malformed")
        try:
            return cls(
                sha256=value["sha256"],
                byte_length=value["byte_length"],
                media_type=value["media_type"],
                source_reference_sha256=value["source_reference_sha256"],
                retrieved_at=value["retrieved_at"],
                rights_status=value["rights_status"],
                artifact_kind=value["artifact_kind"],
                processing_lineage=tuple(ProcessingLineage.from_dict(item) for item in value.get("processing_lineage", ())),
                derived_objects=tuple(value.get("derived_objects", ())),
                verification_status=value["verification_status"],
                verification_receipt_sha256=value.get("verification_receipt_sha256"),
                quarantine_reason=value.get("quarantine_reason"),
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("artifact descriptor is malformed") from exc


def _immutable_descriptor_provenance(descriptor: ArtifactDescriptor) -> dict[str, Any]:
    """Return descriptor fields that no cache transition may later rewrite.

    ``derived_objects`` is a reverse index maintained after children are
    admitted.  The verification-status fields are transition bookkeeping.
    Neither may become a way to alter source, rights, bytes, media type, or
    typed processing lineage after a verifier signed the quarantined object.
    """

    document = descriptor.to_dict()
    for field in _MUTABLE_AFTER_VERIFICATION_DESCRIPTOR_FIELDS:
        del document[field]
    return document


class ArtifactStore:
    """Local content-addressed storage with explicit zone transitions.

    A handle must be bound to a :class:`LocalCacheVerifierTrustStore` before
    it can report, consume, pin, or derive from a verified object.  Binding is
    explicit on ``open(..., verifier_trust_store=...)`` or occurs for the
    lifetime of a handle after ``verify(..., verifier_trust_store)``.  This
    prevents mutable cache metadata from turning a self-consistent but
    untrusted receipt into an object the control plane treats as verified.

    ``quarantine`` is deliberately the one monotonic recovery operation that
    may inspect an untrusted verified-looking entry: it only moves it back to
    quarantine and never returns it as trusted evidence.
    """

    def __init__(
        self,
        root: Path,
        capacity_bytes: int,
        cache_id: str,
        *,
        verifier_trust_store: LocalCacheVerifierTrustStore | None = None,
    ) -> None:
        _require(isinstance(capacity_bytes, int) and not isinstance(capacity_bytes, bool) and capacity_bytes > 0, "cache capacity must be positive")
        _identifier(cache_id, "cache id")
        self.root = root
        self.capacity_bytes = capacity_bytes
        self.cache_id = cache_id
        self._verifier_trust_store: LocalCacheVerifierTrustStore | None = None
        if verifier_trust_store is not None:
            self._bind_verifier_trust_store(verifier_trust_store)

    def _bind_verifier_trust_store(self, verifier_trust_store: LocalCacheVerifierTrustStore) -> None:
        """Bind one explicit local verifier trust root to this cache handle.

        A cache handle cannot silently switch trust roots midway through a
        read/derive operation.  Multiple trusted verifier keys belong under
        one local trust store; callers that need another policy create a new
        cache handle with that policy explicitly.
        """

        _require(isinstance(verifier_trust_store, LocalCacheVerifierTrustStore), "cache verifier trust store is malformed")
        if self._verifier_trust_store is None:
            self._verifier_trust_store = verifier_trust_store
            return
        _require(
            self._verifier_trust_store.root == verifier_trust_store.root,
            "cache verifier trust store differs from the trust root bound to this cache handle",
        )

    def _trusted_verifier_trust_store(self) -> LocalCacheVerifierTrustStore:
        """Return the required trust root for a verified-object operation."""

        trust_store = self._verifier_trust_store
        if trust_store is None:
            raise ProductRefused("verified cache objects require an explicit local verifier trust store")
        return trust_store

    @property
    def config_path(self) -> Path:
        return self.root / "cache.json"

    @property
    def lock_path(self) -> Path:
        return self.root / ".cache.lock"

    @contextmanager
    def _cache_lock(self, *, exclusive: bool = True):
        """Serialize cache transitions and capacity reservations on POSIX hosts."""

        if fcntl is None:
            raise ProductRefused("artifact cache mutation requires a POSIX file-lock backend")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            metadata = os.fstat(descriptor)
            _require(
                stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                "artifact cache lock must be a regular non-linked file",
            )
            with os.fdopen(descriptor, "a+", encoding="utf-8", closefd=True) as handle:
                descriptor = None
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise ProductRefused(f"cannot acquire artifact cache lock: {exc}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        capacity_bytes: int,
        verifier_trust_store: LocalCacheVerifierTrustStore | None = None,
    ) -> ArtifactStore:
        store = cls(root, capacity_bytes, f"cache-{uuid.uuid4().hex}", verifier_trust_store=verifier_trust_store)
        _require(not root.exists() and not root.is_symlink(), f"cache root already exists: {root}")
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
            for directory in _DIRECTORIES:
                (staging / directory).mkdir()
            atomic_write_json(
                staging / "cache.json",
                {"cache_id": store.cache_id, "capacity_bytes": capacity_bytes, "schema_version": CACHE_SCHEMA_VERSION},
            )
            os.replace(staging, root)
            fsync_directory(root.parent)
        except (OSError, ProductRefused) as exc:
            if "staging" in locals() and staging.exists() and not root.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, ProductRefused):
                raise
            raise ProductRefused(f"cannot create artifact cache: {exc}") from exc
        return store

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        verifier_trust_store: LocalCacheVerifierTrustStore | None = None,
    ) -> ArtifactStore:
        """Open cache metadata without implicitly trusting promoted objects.

        Callers that need to read a verified/processed object must supply the
        local trust root that approved its verifier.  An unbound handle may
        still operate on quarantine-only state or explicitly revoke a suspect
        object, but it cannot represent that object as trusted evidence.
        """

        _require(root.is_dir() and not root.is_symlink(), "artifact cache root must be a real directory")
        _regular_file(root / "cache.json", "cache configuration", reject_hardlinks=True)
        config = _read_descriptor(root / "cache.json")
        _require(set(config) == {"cache_id", "capacity_bytes", "schema_version"}, "artifact cache configuration is malformed")
        _require(config["schema_version"] == CACHE_SCHEMA_VERSION, "unsupported artifact cache schema version")
        store = cls(root, config["capacity_bytes"], config["cache_id"], verifier_trust_store=verifier_trust_store)
        for directory in _DIRECTORIES:
            _require((root / directory).is_dir() and not (root / directory).is_symlink(), f"artifact cache is missing its {directory} zone")
        with store._cache_lock():
            store._recover_transitions()
            if store._verifier_trust_store is not None:
                store._reconcile_lineage()
        return store

    @staticmethod
    def _zone_for_descriptor(descriptor: ArtifactDescriptor) -> str:
        return "quarantine" if descriptor.verification_status == "quarantined" else "processed" if descriptor.artifact_kind == "derived" else "verified"

    @staticmethod
    def _transition_record(descriptor: ArtifactDescriptor, verification_receipt: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "descriptor": descriptor.to_dict(),
            "schema_version": "substrate-artifact-cache-transition-v1",
            "verification_receipt": verification_receipt,
        }

    def _apply_object_state(
        self,
        directory: Path,
        descriptor: ArtifactDescriptor,
        verification_receipt: dict[str, Any] | None,
    ) -> None:
        """Write an object state described by a durable transition record."""

        receipt_path = directory / _VERIFICATION_RECEIPT_NAME
        if verification_receipt is None:
            if receipt_path.exists() or receipt_path.is_symlink():
                _regular_file(receipt_path, "cache verification receipt", reject_hardlinks=True)
                receipt_path.unlink()
                fsync_directory(directory)
        else:
            atomic_write_json(receipt_path, verification_receipt)
        atomic_write_json(directory / "descriptor.json", descriptor.to_dict())

    def _recover_transitions(self) -> None:
        """Finish an interrupted zone transition before admitting cache use."""

        records: list[Path] = []
        for zone in _OBJECT_ZONES:
            root = self.root / zone / "sha256"
            if root.exists():
                records.extend(root.glob("*/*/.transition.json"))
        for marker in sorted(records):
            _regular_file(marker, "cache transition record", reject_hardlinks=True)
            record = _read_descriptor(marker)
            _require(
                set(record) == {"descriptor", "schema_version", "verification_receipt"}
                and record["schema_version"] == "substrate-artifact-cache-transition-v1",
                "cache transition record is malformed",
            )
            descriptor = ArtifactDescriptor.from_dict(record["descriptor"])
            expected_directory = self._object_directory(self._zone_for_descriptor(descriptor), descriptor.sha256)
            current_directory = marker.parent
            _require(current_directory.is_dir() and not current_directory.is_symlink(), "cache transition directory is unsafe")
            _regular_file(current_directory / "blob", "cache transition blob", reject_hardlinks=True)
            receipt = record["verification_receipt"]
            _require(receipt is None or isinstance(receipt, dict), "cache transition receipt is malformed")
            self._apply_object_state(current_directory, descriptor, receipt)
            if current_directory != expected_directory:
                _require(not expected_directory.exists() and not expected_directory.is_symlink(), "cache transition target already exists")
                expected_directory.parent.mkdir(parents=True, exist_ok=True)
                os.replace(current_directory, expected_directory)
                fsync_directory(current_directory.parent)
                fsync_directory(expected_directory.parent)
                current_directory = expected_directory
            marker_path = current_directory / ".transition.json"
            _regular_file(marker_path, "cache transition record", reject_hardlinks=True)
            marker_path.unlink()
            fsync_directory(current_directory)

    def _transition_object(
        self,
        *,
        source_zone: str,
        digest: str,
        descriptor: ArtifactDescriptor,
        verification_receipt: dict[str, Any] | None,
    ) -> ArtifactDescriptor:
        """Durably move an object and its metadata between cache zones."""

        source_directory = self._object_directory(source_zone, digest)
        target_zone = self._zone_for_descriptor(descriptor)
        target_directory = self._object_directory(target_zone, digest)
        _require(source_directory.is_dir() and not source_directory.is_symlink(), "cache transition source is unsafe")
        if source_directory == target_directory:
            self._apply_object_state(source_directory, descriptor, verification_receipt)
            fsync_directory(source_directory)
            return descriptor
        _require(not target_directory.exists() and not target_directory.is_symlink(), "cache transition target already exists")
        staging: Path | None = None
        moved = False
        try:
            target_directory.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{digest}.transition-", dir=target_directory.parent))
            staging.rmdir()
            # Persist intent at the source before moving it. If the process
            # dies between either rename, `open()` can finish this transition
            # from the marker rather than accepting a zone/descriptor split.
            atomic_write_json(source_directory / ".transition.json", self._transition_record(descriptor, verification_receipt))
            os.replace(source_directory, staging)
            moved = True
            fsync_directory(source_directory.parent)
            fsync_directory(target_directory.parent)
            self._apply_object_state(staging, descriptor, verification_receipt)
            os.replace(staging, target_directory)
            fsync_directory(target_directory.parent)
            marker = target_directory / ".transition.json"
            _regular_file(marker, "cache transition record", reject_hardlinks=True)
            marker.unlink()
            fsync_directory(target_directory)
        except (OSError, ProductRefused) as exc:
            if staging is not None and staging.exists() and not moved:
                shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, ProductRefused):
                raise
            raise ProductRefused(f"cannot transition artifact cache object: {exc}") from exc
        return descriptor

    def _object_directory(self, zone: str, digest: str) -> Path:
        _require(zone in _OBJECT_ZONES, "unknown artifact zone")
        _digest(digest, "artifact sha256")
        return self.root / zone / "sha256" / digest[:2] / digest

    def _descriptor_path(self, zone: str, digest: str) -> Path:
        return self._object_directory(zone, digest) / "descriptor.json"

    def _blob_path(self, zone: str, digest: str) -> Path:
        return self._object_directory(zone, digest) / "blob"

    def _verification_receipt_path(self, zone: str, digest: str) -> Path:
        return self._object_directory(zone, digest) / _VERIFICATION_RECEIPT_NAME

    def _validate_verified_blob(self, zone: str, descriptor: ArtifactDescriptor) -> None:
        """Revalidate bytes and observed type before accepting a verified object.

        The descriptor/receipt binding protects provenance labels.  It cannot
        make filesystem permissions an integrity boundary against an owner of
        the cache directory, so every read path that treats an object as
        verified repeats the content and media-type checks.
        """

        if descriptor.verification_status != "verified":
            return
        blob = self._blob_path(zone, descriptor.sha256)
        observed_digest, observed_length = _stream_digest(blob)
        _require(
            observed_digest == descriptor.sha256 and observed_length == descriptor.byte_length,
            "cache blob digest or byte length does not match its descriptor",
        )
        observed_media_type = _sniff_media_type(blob, declared_media_type=descriptor.media_type)
        _require(
            observed_media_type == descriptor.media_type,
            "cache blob media type does not match its declared descriptor",
        )

    def _validate_verification_receipt(
        self,
        descriptor: ArtifactDescriptor,
        receipt: Mapping[str, Any],
        *,
        require_trusted: bool = True,
    ) -> None:
        """Validate one cache-generated receipt against current provenance.

        A receipt stores the exact quarantined descriptor the verifier signed.
        The live descriptor must retain every immutable provenance field from
        that record, even though a later child may extend its reverse index.
        This prevents an owner of the cache directory from relabeling an
        already-attested blob with a different source, rights status, media
        type, or processing lineage.
        """

        _require(set(receipt) == _VERIFICATION_RECEIPT_FIELDS, "cache verification receipt is malformed")
        _require(receipt["schema_version"] == CACHE_VERIFICATION_RECEIPT_SCHEMA_VERSION, "cache verification receipt schema is unsupported")
        _require(receipt["execution_permitted"] is False, "cache verification receipt cannot permit execution")
        _require(receipt["cache_id"] == self.cache_id, "cache verification receipt belongs to a different cache")
        _require(receipt["artifact_sha256"] == descriptor.sha256, "cache verification receipt belongs to a different artifact")
        _digest(receipt["descriptor_sha256"], "cache verification receipt descriptor sha256")
        _validate_timestamp(receipt["verified_at"], "cache verification receipt verification time")
        _require(isinstance(receipt["attested_descriptor"], Mapping), "cache verification receipt attested descriptor is malformed")
        attested_descriptor = ArtifactDescriptor.from_dict(receipt["attested_descriptor"])
        _require(
            attested_descriptor.verification_status == "quarantined"
            and attested_descriptor.verification_receipt_sha256 is None,
            "cache verification receipt must retain a quarantined descriptor",
        )
        attestation = CacheAttestation.from_dict(receipt["attestation"])
        _require(attestation.artifact_sha256 == descriptor.sha256, "cache verification attestation belongs to a different artifact")
        _require(attestation.cache_id == self.cache_id, "cache verification attestation belongs to a different cache")
        _require(
            attestation.descriptor_sha256 == receipt["descriptor_sha256"] == sha256(attested_descriptor.to_dict()),
            "cache verification attestation descriptor digest is invalid",
        )
        _require(attestation.rights_status == attested_descriptor.rights_status, "cache verification attestation rights status is invalid")
        _require(receipt["attestation_sha256"] == sha256(attestation.to_dict()), "cache verification receipt attestation digest is invalid")
        _require(
            _immutable_descriptor_provenance(descriptor) == _immutable_descriptor_provenance(attested_descriptor),
            "cache descriptor provenance differs from its attested descriptor",
        )
        _require(descriptor.verification_receipt_sha256 == sha256(dict(receipt)), "cache verification receipt digest does not match its descriptor")
        if require_trusted:
            # Receipt and descriptor digests make metadata internally
            # consistent, but an owner of the cache directory can recompute
            # those hashes.  The signature plus *separate* local trust root is
            # the authority boundary for every operation that treats this
            # object as verified evidence.
            _require(
                _parse_timestamp(attestation.expires_at, "cache verification attestation expiry") > datetime.now(UTC),
                "cache verification attestation has expired",
            )
            verify_cache_attestation(attestation, self._trusted_verifier_trust_store())

    def _locate(self, digest: str, *, require_trusted: bool = True) -> tuple[str, ArtifactDescriptor]:
        _digest(digest, "artifact sha256")
        found: list[tuple[str, ArtifactDescriptor]] = []
        for zone in _OBJECT_ZONES:
            directory = self._object_directory(zone, digest)
            if not directory.exists() and not directory.is_symlink():
                continue
            _require(directory.is_dir() and not directory.is_symlink(), "artifact directory is unsafe")
            _require(
                not (directory / ".transition.json").exists() and not (directory / ".transition.json").is_symlink(),
                "artifact transition requires cache recovery",
            )
            descriptor = ArtifactDescriptor.from_dict(_read_descriptor(self._descriptor_path(zone, digest)))
            _require(descriptor.sha256 == digest, "artifact descriptor belongs to a different object")
            _regular_file(self._blob_path(zone, digest), "cache blob", reject_hardlinks=True)
            expected_zone = self._zone_for_descriptor(descriptor)
            _require(zone == expected_zone, "artifact descriptor verification status does not match its cache zone")
            receipt_path = self._verification_receipt_path(zone, digest)
            if descriptor.verification_status == "verified":
                receipt = _read_descriptor(receipt_path)
                self._validate_verification_receipt(descriptor, receipt, require_trusted=require_trusted)
            else:
                _require(not receipt_path.exists() and not receipt_path.is_symlink(), "quarantined artifact cannot retain a verification receipt")
            self._validate_verified_blob(zone, descriptor)
            found.append((zone, descriptor))
        _require(len(found) == 1, "artifact digest is absent or appears in multiple cache zones")
        return found[0]

    def _used_bytes(self) -> int:
        total = 0
        for incoming in (self.root / "incoming").glob("*.partial"):
            _regular_file(incoming, "cache incoming object", reject_hardlinks=True)
            total += incoming.stat().st_size
        for zone in _OBJECT_ZONES:
            root = self.root / zone / "sha256"
            if not root.exists():
                continue
            for blob in root.glob("*/*/blob"):
                _regular_file(blob, "cache blob")
                total += blob.stat().st_size
        return total

    def _iter_objects(self, *, require_trusted: bool = True) -> list[tuple[str, ArtifactDescriptor]]:
        """Return coherent objects, cryptographically validating verified ones.

        ``require_trusted=False`` exists solely for monotonic cleanup and
        quarantine paths.  Callers must not use such a scan to consume,
        describe, pin, or derive from a verified object.
        """

        objects: list[tuple[str, ArtifactDescriptor]] = []
        for zone in _OBJECT_ZONES:
            root = self.root / zone / "sha256"
            if not root.exists():
                continue
            for descriptor_path in root.glob("*/*/descriptor.json"):
                descriptor = ArtifactDescriptor.from_dict(_read_descriptor(descriptor_path))
                _require(
                    descriptor_path == self._descriptor_path(zone, descriptor.sha256),
                    "artifact descriptor path does not match its digest",
                )
                _require(
                    zone == self._zone_for_descriptor(descriptor),
                    "artifact descriptor verification status does not match its cache zone",
                )
                _regular_file(self._blob_path(zone, descriptor.sha256), "cache blob", reject_hardlinks=True)
                receipt_path = self._verification_receipt_path(zone, descriptor.sha256)
                if descriptor.verification_status == "verified":
                    self._validate_verification_receipt(
                        descriptor,
                        _read_descriptor(receipt_path),
                        require_trusted=require_trusted,
                    )
                else:
                    _require(
                        not receipt_path.exists() and not receipt_path.is_symlink(),
                        "quarantined artifact cannot retain a verification receipt",
                    )
                self._validate_verified_blob(zone, descriptor)
                objects.append((zone, descriptor))
        return objects

    def _direct_children(self, parent_digest: str, *, require_trusted: bool = True) -> tuple[str, ...]:
        """Return derivatives that depend on an input *or* tool artifact."""

        _digest(parent_digest, "parent artifact sha256")
        children = {
            descriptor.sha256
            for _, descriptor in self._iter_objects(require_trusted=require_trusted)
            if any(
                lineage.input_sha256 == parent_digest or lineage.tool_artifact_sha256 == parent_digest
                for lineage in descriptor.processing_lineage
            )
        }
        return tuple(sorted(children))

    def _references(self, digest: str, *, require_trusted: bool = True) -> tuple[str, ...]:
        """Return every live derivative that names an input or tool digest."""

        _digest(digest, "referenced artifact sha256")
        references = {
            descriptor.sha256
            for _, descriptor in self._iter_objects(require_trusted=require_trusted)
            if any(lineage.input_sha256 == digest or lineage.tool_artifact_sha256 == digest for lineage in descriptor.processing_lineage)
        }
        return tuple(sorted(references))

    def _record_derived_object(self, parent_digest: str, child_digest: str) -> None:
        """Maintain a reverse lineage reference after a child is admitted."""

        zone, parent = self._locate(parent_digest)
        _require(parent.verification_status == "verified", "derived artifact parent is no longer verified")
        if child_digest in parent.derived_objects:
            return
        updated = replace(parent, derived_objects=tuple(sorted((*parent.derived_objects, child_digest))))
        atomic_write_json(self._descriptor_path(zone, parent_digest), updated.to_dict())

    def _reconcile_lineage(self) -> None:
        """Repair missing reverse references from authoritative forward lineage."""

        for _, descriptor in self._iter_objects():
            if descriptor.artifact_kind != "derived":
                continue
            for lineage in descriptor.processing_lineage:
                try:
                    self._record_derived_object(lineage.input_sha256, descriptor.sha256)
                except ProductRefused:
                    # A revoked/quarantined parent must not be silently made
                    # valid; the forward relation stays visible for explicit
                    # lineage inspection and recursive revocation.
                    continue

    def _assert_lineage_closure(self, descriptor: ArtifactDescriptor, visited: set[str] | None = None) -> None:
        """Fail closed unless every derived input and tool is still verified."""

        seen = set() if visited is None else visited
        _require(descriptor.sha256 not in seen, "artifact lineage contains a cycle")
        seen.add(descriptor.sha256)
        try:
            for lineage in descriptor.processing_lineage:
                parent_zone, parent = self._locate(lineage.input_sha256)
                tool_zone, tool = self._locate(lineage.tool_artifact_sha256)
                _require(parent_zone in {"verified", "processed"} and parent.verification_status == "verified", "artifact lineage input is not verified")
                _require(tool_zone in {"verified", "processed"} and tool.verification_status == "verified", "artifact lineage tool is not verified")
                _require(
                    parent.source_reference_sha256 == descriptor.source_reference_sha256, "artifact lineage source reference does not match its derivative"
                )
                _require(parent.rights_status == descriptor.rights_status, "artifact lineage rights status does not match its derivative")
                self._assert_lineage_closure(parent, seen)
                self._assert_lineage_closure(tool, seen)
        finally:
            seen.remove(descriptor.sha256)

    def _ensure_capacity(self, byte_length: int) -> None:
        _require(isinstance(byte_length, int) and byte_length >= 0, "artifact byte length is invalid")
        _require(byte_length <= self.capacity_bytes, "artifact exceeds the cache capacity")
        _require(self._used_bytes() + byte_length <= self.capacity_bytes, "artifact cache has insufficient reserved capacity")

    def _copy_to_incoming(self, source_path: Path, *, expected_byte_length: int | None) -> tuple[Path, str, int]:
        metadata = _regular_file(source_path, "ingress source", reject_hardlinks=True)
        if expected_byte_length is not None:
            _require(
                isinstance(expected_byte_length, int) and expected_byte_length >= 0 and expected_byte_length == metadata.st_size,
                "declared ingress byte length does not match the source file",
            )
        self._ensure_capacity(metadata.st_size)
        available_bytes = self.capacity_bytes - self._used_bytes()
        _require(available_bytes >= metadata.st_size, "artifact cache has insufficient reserved capacity")
        incoming_path = self.root / "incoming" / f"{uuid.uuid4().hex}.partial"
        digest = hashlib.sha256()
        byte_length = 0
        source_descriptor: int | None = None
        target_descriptor: int | None = None
        try:
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            source_descriptor = os.open(source_path, os.O_RDONLY | nofollow)
            opened_metadata = os.fstat(source_descriptor)
            _require(
                stat.S_ISREG(opened_metadata.st_mode)
                and opened_metadata.st_nlink == 1
                and (opened_metadata.st_dev, opened_metadata.st_ino, opened_metadata.st_size) == (metadata.st_dev, metadata.st_ino, metadata.st_size),
                "ingress source changed before it could be opened safely",
            )
            target_descriptor = os.open(incoming_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
            target_metadata = os.fstat(target_descriptor)
            _require(
                stat.S_ISREG(target_metadata.st_mode) and target_metadata.st_nlink == 1,
                "cache incoming object is not a safe regular file",
            )
            with os.fdopen(source_descriptor, "rb", closefd=True) as source, os.fdopen(target_descriptor, "wb", closefd=True) as target:
                source_descriptor = None
                target_descriptor = None
                while chunk := source.read(1024 * 1024):
                    byte_length += len(chunk)
                    _require(byte_length <= available_bytes, "ingress object exceeds the cache capacity reservation")
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            _require(byte_length == metadata.st_size, "ingress source changed while it was being copied")
            _require(expected_byte_length is None or byte_length == expected_byte_length, "ingress byte length changed while copying")
            return incoming_path, digest.hexdigest(), byte_length
        except (OSError, ProductRefused) as exc:
            incoming_path.unlink(missing_ok=True)
            if isinstance(exc, ProductRefused):
                raise
            raise ProductRefused(f"cannot ingest source file: {exc}") from exc
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            if target_descriptor is not None:
                os.close(target_descriptor)

    def _publish_quarantine(self, incoming_path: Path, descriptor: ArtifactDescriptor) -> ArtifactDescriptor:
        target = self._object_directory("quarantine", descriptor.sha256)
        try:
            try:
                existing_zone, existing = self._locate(descriptor.sha256)
            except ProductRefused as exc:
                if "absent" not in str(exc):
                    raise
            else:
                incoming_path.unlink(missing_ok=True)
                _require(existing.to_dict() == descriptor.to_dict(), "content digest is already bound to different provenance")
                _require(existing_zone == "quarantine", "content digest is already immutable in a different cache zone")
                return existing
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{descriptor.sha256}.staging-", dir=target.parent))
            os.replace(incoming_path, staging / "blob")
            os.chmod(staging / "blob", 0o444)
            atomic_write_json(staging / "descriptor.json", descriptor.to_dict())
            os.replace(staging, target)
            fsync_directory(target.parent)
            return descriptor
        except OSError as exc:
            incoming_path.unlink(missing_ok=True)
            if "staging" in locals() and staging.exists() and not target.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise ProductRefused(f"cannot publish quarantined artifact: {exc}") from exc

    def ingest_file(
        self,
        source_path: Path,
        *,
        media_type: str,
        source_reference_sha256: str,
        rights_status: str,
        expected_byte_length: int | None = None,
    ) -> ArtifactDescriptor:
        """Serialize local ingress and quarantine publication."""

        with self._cache_lock():
            return self._ingest_file(
                source_path,
                media_type=media_type,
                source_reference_sha256=source_reference_sha256,
                rights_status=rights_status,
                expected_byte_length=expected_byte_length,
            )

    def _ingest_file(
        self,
        source_path: Path,
        *,
        media_type: str,
        source_reference_sha256: str,
        rights_status: str,
        expected_byte_length: int | None = None,
    ) -> ArtifactDescriptor:
        """Stream a local regular file into quarantine; no remote transport occurs."""

        _require(isinstance(media_type, str) and bool(_MEDIA_TYPE.fullmatch(media_type)), "artifact media type is invalid")
        _digest(source_reference_sha256, "artifact source reference")
        _require(rights_status in _RIGHTS_STATUSES, "artifact rights status is unsupported")
        incoming_path, digest, byte_length = self._copy_to_incoming(source_path, expected_byte_length=expected_byte_length)
        descriptor = ArtifactDescriptor(
            sha256=digest,
            byte_length=byte_length,
            media_type=media_type,
            source_reference_sha256=source_reference_sha256,
            retrieved_at=_timestamp(),
            rights_status=rights_status,
        )
        return self._publish_quarantine(incoming_path, descriptor)

    def ingest_derivative_file(
        self,
        source_path: Path,
        *,
        media_type: str,
        source_reference_sha256: str,
        rights_status: str,
        processing_lineage: tuple[ProcessingLineage, ...],
        expected_byte_length: int | None = None,
    ) -> ArtifactDescriptor:
        """Serialize derivative admission and source/tool lineage checks."""

        with self._cache_lock():
            return self._ingest_derivative_file(
                source_path,
                media_type=media_type,
                source_reference_sha256=source_reference_sha256,
                rights_status=rights_status,
                processing_lineage=processing_lineage,
                expected_byte_length=expected_byte_length,
            )

    def _ingest_derivative_file(
        self,
        source_path: Path,
        *,
        media_type: str,
        source_reference_sha256: str,
        rights_status: str,
        processing_lineage: tuple[ProcessingLineage, ...],
        expected_byte_length: int | None = None,
    ) -> ArtifactDescriptor:
        """Stage a typed derivative in quarantine until it receives an attestation."""

        _require(isinstance(media_type, str) and bool(_MEDIA_TYPE.fullmatch(media_type)), "artifact media type is invalid")
        _digest(source_reference_sha256, "artifact source reference")
        _require(rights_status in _RIGHTS_STATUSES, "artifact rights status is unsupported")
        _require(isinstance(processing_lineage, tuple) and bool(processing_lineage), "derived artifact requires processing lineage")
        _require(
            len({lineage.input_sha256 for lineage in processing_lineage if isinstance(lineage, ProcessingLineage)}) == len(processing_lineage),
            "derived artifact inputs cannot repeat",
        )
        for lineage in processing_lineage:
            _require(isinstance(lineage, ProcessingLineage), "derived artifact lineage is malformed")
            parent_zone, parent = self._locate(lineage.input_sha256)
            _require(
                parent_zone in {"verified", "processed"} and parent.verification_status == "verified", "derived artifact input is not immutable and verified"
            )
            tool_zone, tool = self._locate(lineage.tool_artifact_sha256)
            _require(tool_zone in {"verified", "processed"} and tool.verification_status == "verified", "derived artifact tool is not immutable and verified")
            _require(parent.source_reference_sha256 == source_reference_sha256, "derived artifact source reference does not match its input")
            _require(parent.rights_status == rights_status, "derived artifact rights status does not match its input")
            self._assert_lineage_closure(parent)
            self._assert_lineage_closure(tool)
        incoming_path, digest, byte_length = self._copy_to_incoming(source_path, expected_byte_length=expected_byte_length)
        descriptor = ArtifactDescriptor(
            sha256=digest,
            byte_length=byte_length,
            media_type=media_type,
            source_reference_sha256=source_reference_sha256,
            retrieved_at=_timestamp(),
            rights_status=rights_status,
            artifact_kind="derived",
            processing_lineage=processing_lineage,
        )
        return self._publish_quarantine(incoming_path, descriptor)

    def verify(
        self,
        digest: str,
        attestation: CacheAttestation,
        verifier_trust_store: LocalCacheVerifierTrustStore,
    ) -> ArtifactDescriptor:
        """Serialize validation and promotion of one quarantined object."""

        with self._cache_lock():
            return self._verify(digest, attestation, verifier_trust_store)

    def _verify(
        self,
        digest: str,
        attestation: CacheAttestation,
        verifier_trust_store: LocalCacheVerifierTrustStore,
    ) -> ArtifactDescriptor:
        """Rehash, type-check, and atomically promote one attested object."""

        _digest(digest, "artifact sha256")
        _require(isinstance(attestation, CacheAttestation), "cache attestation is malformed")
        _require(attestation.artifact_sha256 == digest, "cache attestation is bound to a different object")
        zone, descriptor = self._locate(digest)
        _require(zone == "quarantine" and descriptor.verification_status == "quarantined", "only quarantined objects can be verified")
        _require(attestation.rights_status == descriptor.rights_status, "cache attestation rights status does not match the artifact")
        _require(attestation.cache_id == self.cache_id, "cache attestation is bound to a different cache")
        _require(
            attestation.descriptor_sha256 == sha256(descriptor.to_dict()),
            "cache attestation does not bind the quarantined descriptor",
        )
        verify_cache_attestation(attestation, verifier_trust_store)
        # Do not bind a failed candidate trust root.  Once an attestation is
        # actually accepted, this handle must use the same root for every
        # subsequent verified-object read or lineage check.
        self._bind_verifier_trust_store(verifier_trust_store)
        blob = self._blob_path(zone, digest)
        observed_digest, observed_length = _stream_digest(blob)
        _require(
            observed_digest == descriptor.sha256 and observed_length == descriptor.byte_length,
            "cache blob digest or byte length does not match its descriptor",
        )
        observed_media_type = _sniff_media_type(blob, declared_media_type=descriptor.media_type)
        _require(observed_media_type != "application/octet-stream", "opaque binary objects require a dedicated typed adapter")
        _require(observed_media_type != "application/zip", "archive extraction is unavailable until bounded archive validation exists")
        _require(observed_media_type == descriptor.media_type, "cache blob media type does not match its declared descriptor")
        if descriptor.artifact_kind == "derived":
            self._assert_lineage_closure(descriptor)
        receipt = {
            "artifact_sha256": digest,
            "attestation": attestation.to_dict(),
            "attestation_sha256": sha256(attestation.to_dict()),
            "attested_descriptor": descriptor.to_dict(),
            "cache_id": self.cache_id,
            "descriptor_sha256": attestation.descriptor_sha256,
            "execution_permitted": False,
            "schema_version": CACHE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
            "verified_at": _timestamp(),
        }
        receipt_digest = sha256(receipt)
        promoted = replace(
            descriptor,
            verification_status="verified",
            verification_receipt_sha256=receipt_digest,
            quarantine_reason=None,
        )
        admitted = self._transition_object(
            source_zone="quarantine",
            digest=digest,
            descriptor=promoted,
            verification_receipt=receipt,
        )
        if admitted.artifact_kind == "derived":
            for lineage in admitted.processing_lineage:
                self._record_derived_object(lineage.input_sha256, admitted.sha256)
        return admitted

    def quarantine(self, digest: str, *, reason: str) -> ArtifactDescriptor:
        """Serialize revocation of an object and derived descendants."""

        with self._cache_lock():
            return self._quarantine(digest, reason=reason)

    def _quarantine(self, digest: str, *, reason: str) -> ArtifactDescriptor:
        """Revoke materialization for an object and every derived descendant."""

        _digest(digest, "artifact sha256")
        _require(isinstance(reason, str) and bool(reason.strip()), "quarantine reason must be nonempty")
        pending = [digest]
        visited: set[str] = set()
        first: ArtifactDescriptor | None = None
        while pending:
            current = pending.pop(0)
            if current in visited:
                continue
            visited.add(current)
            # Quarantine is intentionally monotonic: it may inspect a
            # verified-looking object without a trust root only to revoke it.
            # It never returns that state as trusted evidence.
            children = self._direct_children(current, require_trusted=False)
            zone, descriptor = self._locate(current, require_trusted=False)
            current_reason = reason if current == digest else f"ancestor {digest} was quarantined: {reason}"
            quarantined = replace(
                descriptor,
                verification_status="quarantined",
                verification_receipt_sha256=None,
                quarantine_reason=current_reason,
            )
            transitioned = self._transition_object(
                source_zone=zone,
                digest=current,
                descriptor=quarantined,
                verification_receipt=None,
            )
            if current == digest:
                first = transitioned
            pending.extend(children)
        if first is None:
            raise ProductRefused("artifact to quarantine is absent")
        return first

    def pin(self, digest: str, *, reason: str) -> dict[str, Any]:
        """Serialize a pin update with the object state it protects."""

        with self._cache_lock():
            return self._pin(digest, reason=reason)

    def _pin(self, digest: str, *, reason: str) -> dict[str, Any]:
        """Protect an immutable object from explicit garbage collection."""

        _digest(digest, "artifact sha256")
        _require(isinstance(reason, str) and bool(reason.strip()), "pin reason must be nonempty")
        zone, descriptor = self._locate(digest)
        _require(zone in {"verified", "processed"} and descriptor.verification_status == "verified", "only verified artifacts can be pinned")
        pin = {
            "artifact_sha256": digest,
            "execution_permitted": False,
            "reason": reason,
            "schema_version": CACHE_SCHEMA_VERSION,
        }
        target = self.root / "pins" / f"{digest}.json"
        if target.exists():
            existing = _read_descriptor(target)
            _require(existing == pin, "refusing to replace an existing artifact pin")
        else:
            atomic_write_json(target, pin)
        return pin

    def _pins(self) -> set[str]:
        pinned: set[str] = set()
        for path in (self.root / "pins").glob("*.json"):
            record = _read_descriptor(path)
            _require(
                set(record) == {"artifact_sha256", "execution_permitted", "reason", "schema_version"},
                "artifact pin is malformed",
            )
            _digest(record["artifact_sha256"], "artifact pin digest")
            _require(record["execution_permitted"] is False, "artifact pins may not permit execution")
            _require(isinstance(record["reason"], str) and bool(record["reason"].strip()), "artifact pin reason is invalid")
            _require(record["schema_version"] == CACHE_SCHEMA_VERSION, "artifact pin schema version is invalid")
            pinned.add(record["artifact_sha256"])
        return pinned

    def gc(self, *, include_verified: bool = False, maximum_objects: int | None = None) -> dict[str, Any]:
        """Serialize garbage collection with live lineage inspection."""

        with self._cache_lock():
            return self._gc(include_verified=include_verified, maximum_objects=maximum_objects)

    def _gc(self, *, include_verified: bool = False, maximum_objects: int | None = None) -> dict[str, Any]:
        """Remove explicitly selected unpinned cache objects; verified objects stay by default."""

        _require(isinstance(include_verified, bool), "include_verified must be boolean")
        _require(maximum_objects is None or (isinstance(maximum_objects, int) and maximum_objects > 0), "gc maximum objects is invalid")
        pinned = self._pins()
        zones = ("quarantine", "verified", "processed") if include_verified else ("quarantine",)
        candidates: list[tuple[str, str, int]] = []
        # A default GC only removes quarantine entries.  It may scan
        # verified-looking metadata to preserve lineage, but it does not
        # represent or consume it.  GC that selects verified objects must
        # validate their attestations under the bound local trust root.
        for zone, descriptor in self._iter_objects(require_trusted=include_verified):
            if zone in zones and descriptor.sha256 not in pinned:
                candidates.append((zone, descriptor.sha256, descriptor.byte_length))
        candidates.sort()
        if maximum_objects is not None:
            candidates = candidates[:maximum_objects]
        removed: list[dict[str, Any]] = []
        retained_due_to_lineage: list[dict[str, Any]] = []
        for zone, digest, byte_length in candidates:
            references = self._references(digest, require_trusted=include_verified)
            if references:
                retained_due_to_lineage.append({"artifact_sha256": digest, "referenced_by": list(references)})
                continue
            target = self._object_directory(zone, digest)
            _require(target.is_dir() and not target.is_symlink(), "gc target is unsafe")
            try:
                shutil.rmtree(target)
                fsync_directory(target.parent)
            except OSError as exc:
                raise ProductRefused(f"cannot garbage collect artifact: {exc}") from exc
            removed.append({"byte_length": byte_length, "sha256": digest, "zone": zone})
        return {
            "execution_permitted": False,
            "include_verified": include_verified,
            "removed": removed,
            "removed_bytes": sum(item["byte_length"] for item in removed),
            "retained_due_to_lineage": retained_due_to_lineage,
        }

    def explain(self, digest: str) -> dict[str, Any]:
        """Read one coherent object explanation without racing a transition."""

        with self._cache_lock(exclusive=False):
            return self._explain(digest)

    def _explain(self, digest: str) -> dict[str, Any]:
        zone, descriptor = self._locate(digest)
        return {
            "descriptor": descriptor.to_dict(),
            "execution_permitted": False,
            "pinned": digest in self._pins(),
            "zone": zone,
        }

    def status(self) -> dict[str, Any]:
        """Read coherent cache accounting without racing a mutation."""

        with self._cache_lock(exclusive=False):
            return self._status()

    def _status(self) -> dict[str, Any]:
        counts = {zone: 0 for zone in _OBJECT_ZONES}
        bytes_by_zone = {zone: 0 for zone in _OBJECT_ZONES}
        for zone, descriptor in self._iter_objects():
            counts[zone] += 1
            bytes_by_zone[zone] += descriptor.byte_length
        return {
            "bytes_by_zone": bytes_by_zone,
            "cache_id": self.cache_id,
            "capacity_bytes": self.capacity_bytes,
            "execution_permitted": False,
            "object_counts": counts,
            "pinned_objects": len(self._pins()),
            "schema_version": CACHE_SCHEMA_VERSION,
            "used_bytes": sum(bytes_by_zone.values()),
        }

    def trusted_source_receipt_verifier(
        self,
        receipt: SourceReceipt,
        source_plan: Mapping[str, Any],
        verifier_trust_store: LocalCacheVerifierTrustStore,
    ) -> bool:
        """Read and revalidate admitted cache evidence under a shared lock."""

        with self._cache_lock(exclusive=False):
            return self._trusted_source_receipt_verifier(receipt, source_plan, verifier_trust_store)

    def _trusted_source_receipt_verifier(
        self,
        receipt: SourceReceipt,
        source_plan: Mapping[str, Any],
        verifier_trust_store: LocalCacheVerifierTrustStore,
    ) -> bool:
        """Bridge verified cache evidence to existing source assimilation safely.

        This is intentionally a strict predicate suitable for the existing
        `assimilate_source_receipt(..., verifier=...)` hook.  It does not read
        source paths or receive network authority.
        """

        if not isinstance(receipt, SourceReceipt) or not isinstance(source_plan, Mapping) or not isinstance(verifier_trust_store, LocalCacheVerifierTrustStore):
            return False
        try:
            self._bind_verifier_trust_store(verifier_trust_store)
            zone, descriptor = self._locate(receipt.content_sha256)
            blob = self._blob_path(zone, descriptor.sha256)
            observed_digest, observed_length = _stream_digest(blob)
            observed_media_type = _sniff_media_type(blob, declared_media_type=descriptor.media_type)
            self._assert_lineage_closure(descriptor)
        except ProductRefused:
            return False
        return (
            zone in {"verified", "processed"}
            and descriptor.verification_status == "verified"
            and observed_digest == descriptor.sha256
            and observed_length == descriptor.byte_length
            and observed_media_type == descriptor.media_type
            and _media_type_matches_modality(descriptor.media_type, receipt.request.modality)
            and descriptor.source_reference_sha256 == receipt.acquisition_plan_sha256
            and descriptor.rights_status == receipt.request.access_status
            and source_plan.get("plan_sha256") == receipt.acquisition_plan_sha256
        )


def _read_descriptor(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        descriptor = _open_regular_read(path, "cache metadata", reject_hardlinks=True)
        with os.fdopen(descriptor, "rb") as handle:
            value = json.loads(
                handle.read().decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_nonstandard_constant,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProductRefused(f"cannot read cache metadata: {exc}") from exc
    _require(isinstance(value, dict), "cache metadata must contain a JSON object")
    return value
