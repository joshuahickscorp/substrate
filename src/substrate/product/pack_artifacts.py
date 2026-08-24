"""Signed, local-first capability-pack artifacts.

This module implements Band 1 of the product program.  It deliberately builds,
signs, verifies, trusts, and installs *manifests*, not executable tool bundles.
In particular it does not download, vendor, invoke, or inspect Chromium,
FFmpeg, yt-dlp, Git, Docker, model weights, or any other upstream component.

A valid signature is necessary but insufficient: a local trust rule must also
scope the publisher, pack name, capabilities, supported host, and runtime.
Even an installed pack remains non-executing until a later, independently
verified sandbox broker is enabled.
"""

from __future__ import annotations

import base64
import binascii
import os
import platform
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from substrate.product.codec import atomic_write_json, canonical_json, fsync_directory, read_json, sha256
from substrate.product.contracts import PRODUCT_SCHEMA_VERSION, ProductRefused
from substrate.product.packs import CapabilityPack, resolve_packs

PACK_ARTIFACT_SCHEMA_VERSION = "substrate-capability-pack-v1"
PACK_SIGNATURE_SCHEMA_VERSION = "substrate-capability-pack-signature-v1"
PACK_TRUST_SCHEMA_VERSION = "substrate-capability-pack-trust-v1"
PACK_INSTALL_RECEIPT_SCHEMA_VERSION = "substrate-capability-pack-install-v1"

MANIFEST_NAME = "manifest.json"
SIGNATURE_NAME = "signature.json"
INSTALL_RECEIPT_NAME = "installation.json"

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CAPABILITIES = frozenset(
    {
        "browser",
        "camera",
        "container-nesting",
        "desktop",
        "filesystem-write",
        "gpu",
        "media-decode",
        "microphone",
        "network",
        "subprocess",
    }
)
_PORTABLE_FILESYSTEM_GRANTS = (
    {"access": "read-only", "source": "content-addressed-inputs"},
    {"access": "ephemeral-write", "source": "task-workspace"},
    {"access": "quarantine-write", "source": "untrusted-output"},
)
_SUPPORTED_HOSTS = ("darwin-arm64", "linux-amd64", "linux-arm64", "windows-amd64")
_MANIFEST_FIELDS = frozenset(
    {
        "artifact_descriptors",
        "capability_denial_canary",
        "capability_grants",
        "filesystem_grants",
        "health_checks",
        "license_metadata",
        "name",
        "network_grants",
        "publisher",
        "required_binaries",
        "required_model_organs",
        "required_runtime_version",
        "resource_class",
        "schema_version",
        "source_adapters",
        "supported_hosts",
        "tool_adapters",
        "uninstall_policy",
        "version",
    }
)
_TRUST_RULE_FIELDS = frozenset(
    {"allowed_capabilities", "allowed_pack_names", "key_id", "public_key_b64", "publisher", "schema_version"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductRefused(message)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProductRefused(f"{label} must be a lowercase identifier")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProductRefused(f"{label} must be a lowercase SHA-256 digest")
    return value


def _string_list(value: object, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ProductRefused(f"{label} must be a list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ProductRefused(f"{label} must contain nonempty strings")
    values = cast(list[str], value)
    if len(set(values)) != len(values):
        raise ProductRefused(f"{label} cannot contain duplicates")
    if nonempty and not values:
        raise ProductRefused(f"{label} cannot be empty")
    return values


def _semantic_version(version: str) -> str:
    """Promote legacy declarative pack version ``1`` to semantic ``1.0.0``."""

    if version.isdigit():
        version = f"{version}.0.0"
    _require(bool(_SEMVER.fullmatch(version)), "pack version must be a semantic version")
    return version


def host_platform_id() -> str:
    """Return the host label used for local compatibility checks, without probing tools."""

    system = platform.system().lower()
    machine = platform.machine().lower()
    machine_aliases = {"aarch64": "arm64", "x86_64": "amd64", "x64": "amd64"}
    normalized_machine = machine_aliases.get(machine, machine)
    return f"{system}-{normalized_machine}"


def _require_regular_file(path: Path, label: str, *, reject_hardlinks: bool = False) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductRefused(f"cannot inspect {label}: {exc}") from exc
    _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular non-symlink file")
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    if reject_hardlinks:
        _require(metadata.st_nlink == 1, f"{label} must not be a hard-linked file")
    return metadata


def _read_regular_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label, reject_hardlinks=True)
    return read_json(path)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    """Read the exact non-linked PEM file checked by ``lstat``.

    Key material must not switch from a regular file to a symlink or a
    different inode in the gap between the policy check and deserialization.
    """

    expected = _require_regular_file(path, label, reject_hardlinks=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        observed = os.fstat(descriptor)
        _require(
            stat.S_ISREG(observed.st_mode)
            and observed.st_nlink == 1
            and (observed.st_dev, observed.st_ino, observed.st_size) == (expected.st_dev, expected.st_ino, expected.st_size),
            f"{label} changed before it could be opened safely",
        )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            return handle.read()
    except OSError as exc:
        raise ProductRefused(f"cannot safely read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _signature_payload(manifest: Mapping[str, Any]) -> bytes:
    """Return a domain-separated canonical message for Ed25519 signing."""

    return b"substrate-capability-pack-signature-v1\x00" + canonical_json(dict(manifest)).encode("utf-8")


def _public_key_b64(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def _key_id(public_key_b64: str) -> str:
    return sha256({"algorithm": "ed25519", "public_key_b64": public_key_b64})


def _load_public_key(path: Path) -> Ed25519PublicKey:
    try:
        loaded = serialization.load_pem_public_key(_read_regular_bytes(path, "public key"))
    except ProductRefused:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ProductRefused("public key is not a valid PEM key") from exc
    if not isinstance(loaded, Ed25519PublicKey):
        raise ProductRefused("public key must use Ed25519")
    return loaded


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    try:
        loaded = serialization.load_pem_private_key(_read_regular_bytes(path, "private key"), password=None)
    except ProductRefused:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ProductRefused("private key is not a valid unencrypted PEM key") from exc
    if not isinstance(loaded, Ed25519PrivateKey):
        raise ProductRefused("private key must use Ed25519")
    return loaded


def _write_exclusive(path: Path, value: bytes, *, mode: int) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite existing key path {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor != -1:
                os.close(descriptor)
        fsync_directory(path.parent)
    except OSError as exc:
        raise ProductRefused(f"cannot write key material: {exc}") from exc


def generate_ed25519_keypair(private_key_path: Path, public_key_path: Path) -> dict[str, Any]:
    """Generate an operator-managed Ed25519 keypair without an implicit key store."""

    _require(private_key_path != public_key_path, "private and public key paths must differ")
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    private_bytes = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_exclusive(private_key_path, private_bytes, mode=0o600)
    try:
        _write_exclusive(public_key_path, public_bytes, mode=0o644)
    except ProductRefused:
        # The private key is intentionally not deleted automatically: only an
        # operator can decide whether a newly generated key should be retained.
        raise
    public_key_b64 = _public_key_b64(public)
    return {
        "algorithm": "ed25519",
        "execution_permitted": False,
        "key_id": _key_id(public_key_b64),
        "private_key_path": str(private_key_path),
        "public_key_path": str(public_key_path),
    }


def _adapter_specs(pack: CapabilityPack) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return typed roles, source adapters, and requested body capabilities.

    These names are protocol roles.  They are intentionally not executable
    paths, commands, container images, browser profiles, or downloader flags.
    """

    specs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]] = {
        "engineering": (
            # Snapshot inspection itself consumes immutable inputs; it must
            # not inherit the write/process grants that a later, separately
            # typed compiler or patch-work adapter might require.
            [{"adapter_id": "repository-inspection-v1", "required_capabilities": [], "role": "repository-inspection"}],
            [{"adapter_id": "approved-repository-v1", "mode": "operator-import"}],
            ["filesystem-write", "subprocess"],
        ),
        "research": (
            [{"adapter_id": "document-observation-v1", "required_capabilities": [], "role": "document-observation"}],
            [{"adapter_id": "approved-document-v1", "mode": "operator-import"}],
            [],
        ),
        "media": (
            [{"adapter_id": "media-observation-v1", "required_capabilities": ["media-decode"], "role": "media-observation"}],
            [{"adapter_id": "approved-media-v1", "mode": "brokered-rights-gated"}],
            ["media-decode"],
        ),
        "browser": (
            [{"adapter_id": "browser-observation-v1", "required_capabilities": ["browser"], "role": "browser-observation"}],
            [{"adapter_id": "approved-web-observation-v1", "mode": "brokered-rights-gated"}],
            ["browser"],
        ),
        "mathematics": (
            [{"adapter_id": "formal-solver-v1", "required_capabilities": ["subprocess"], "role": "formal-solver"}],
            [{"adapter_id": "approved-theorem-library-v1", "mode": "operator-import"}],
            ["subprocess"],
        ),
        "formal-math": (
            [{"adapter_id": "formal-solver-v1", "required_capabilities": ["subprocess"], "role": "formal-solver"}],
            [{"adapter_id": "approved-theorem-library-v1", "mode": "operator-import"}],
            ["subprocess"],
        ),
        "three-d": (
            [{"adapter_id": "three-d-observation-v1", "required_capabilities": ["media-decode", "subprocess"], "role": "three-d-observation"}],
            [{"adapter_id": "approved-scene-v1", "mode": "operator-import"}],
            ["media-decode", "subprocess"],
        ),
        "3d": (
            [{"adapter_id": "three-d-observation-v1", "required_capabilities": ["media-decode", "subprocess"], "role": "three-d-observation"}],
            [{"adapter_id": "approved-scene-v1", "mode": "operator-import"}],
            ["media-decode", "subprocess"],
        ),
        "desktop": (
            [{"adapter_id": "desktop-observation-v1", "required_capabilities": ["desktop"], "role": "desktop-observation"}],
            [{"adapter_id": "operator-desktop-fixture-v1", "mode": "operator-import"}],
            ["desktop"],
        ),
        "data-science": (
            [{"adapter_id": "data-analysis-v1", "required_capabilities": ["filesystem-write", "subprocess"], "role": "data-analysis"}],
            [{"adapter_id": "approved-dataset-v1", "mode": "operator-import"}],
            ["filesystem-write", "subprocess"],
        ),
        "robotics": (
            [{"adapter_id": "simulation-observation-v1", "required_capabilities": ["subprocess"], "role": "simulation-observation"}],
            [{"adapter_id": "approved-simulation-v1", "mode": "operator-import"}],
            ["subprocess"],
        ),
    }
    try:
        return specs[pack.name]
    except KeyError as exc:
        raise ProductRefused(f"no artifact adapter contract exists for capability pack {pack.name!r}") from exc


def _binary_id(requirement: str) -> str:
    stem = requirement.split(" (")[0].strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-")
    return _identifier(normalized, "required binary id")


def build_manifest(pack_name: str, *, publisher: str) -> dict[str, Any]:
    """Construct the fixed, data-only artifact manifest for a built-in pack."""

    pack = resolve_packs((pack_name,))[0]
    publisher = _identifier(publisher, "publisher")
    tool_adapters, source_adapters, capabilities = _adapter_specs(pack)
    _require(set(capabilities).issubset(_CAPABILITIES), "pack requests an unsupported capability")
    capability_grants = sorted(capabilities)
    manifest = {
        "artifact_descriptors": [],
        "capability_denial_canary": {
            "denied_capabilities": sorted(_CAPABILITIES - set(capability_grants)),
            "execution_permitted": False,
            "network_mode": "none",
        },
        "capability_grants": capability_grants,
        "filesystem_grants": [dict(item) for item in _PORTABLE_FILESYSTEM_GRANTS],
        "health_checks": [{"kind": "manifest-validation-only"}],
        "license_metadata": {
            "operator_review_required": True,
            "pack_license": "LicenseRef-Substrate-Operator-Review",
            "upstream_components_not_vendored": True,
        },
        "name": pack.name,
        "network_grants": [],
        "publisher": publisher,
        "required_binaries": [
            {
                "binary_id": _binary_id(requirement),
                "delivery": "operator-provided-or-verified-image",
                "license_review_required": True,
                "vendored": False,
            }
            for requirement in pack.tool_requirements
        ],
        "required_model_organs": [],
        "required_runtime_version": PRODUCT_SCHEMA_VERSION,
        "resource_class": pack.worker_profile.to_dict(),
        "schema_version": PACK_ARTIFACT_SCHEMA_VERSION,
        "source_adapters": source_adapters,
        "supported_hosts": list(_SUPPORTED_HOSTS),
        "tool_adapters": tool_adapters,
        "uninstall_policy": {
            "action": "remove-registry-reference-only",
            "execution_permitted": False,
            "shared_artifacts": "retain-until-explicit-cache-gc",
        },
        "version": _semantic_version(pack.version),
    }
    _validate_manifest(manifest)
    return manifest


def _validate_adapter(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"adapter_id", "required_capabilities", "role"}:
        raise ProductRefused(f"{label} is malformed")
    adapter = cast(dict[str, Any], value)
    _identifier(adapter["adapter_id"], f"{label} id")
    _identifier(adapter["role"], f"{label} role")
    requested = _string_list(adapter["required_capabilities"], f"{label} capabilities")
    _require(set(requested).issubset(_CAPABILITIES), f"{label} requests an unsupported capability")


def _validate_source_adapter(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"adapter_id", "mode"}:
        raise ProductRefused("source adapter is malformed")
    adapter = cast(dict[str, Any], value)
    _identifier(adapter["adapter_id"], "source adapter id")
    _require(adapter["mode"] in {"brokered-rights-gated", "operator-import"}, "source adapter mode is unsupported")


def _validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise ProductRefused("pack manifest must be an object")
    manifest = cast(dict[str, Any], manifest)
    unknown = sorted(set(manifest) - _MANIFEST_FIELDS)
    missing = sorted(_MANIFEST_FIELDS - set(manifest))
    _require(not unknown and not missing, f"pack manifest fields are invalid; missing={missing}, unknown={unknown}")
    _require(manifest["schema_version"] == PACK_ARTIFACT_SCHEMA_VERSION, "unsupported pack artifact schema version")
    _identifier(manifest["name"], "pack name")
    _identifier(manifest["publisher"], "publisher")
    _require(isinstance(manifest["version"], str) and bool(_SEMVER.fullmatch(manifest["version"])), "pack version must be a semantic version")
    _require(manifest["required_runtime_version"] == PRODUCT_SCHEMA_VERSION, "pack requires an unsupported runtime version")
    hosts = _string_list(manifest["supported_hosts"], "supported hosts", nonempty=True)
    _require(set(hosts).issubset(_SUPPORTED_HOSTS), "pack supports an unknown host")
    _require(
        manifest["filesystem_grants"] == [dict(item) for item in _PORTABLE_FILESYSTEM_GRANTS],
        "pack filesystem grants must retain the portable mount contract",
    )
    _require(manifest["network_grants"] == [], "pack network grants must be empty until a future broker exists")
    requested_capabilities = _string_list(manifest["capability_grants"], "capability grants")
    _require(set(requested_capabilities).issubset(_CAPABILITIES), "pack requests an unsupported capability")
    _require(requested_capabilities == sorted(requested_capabilities), "pack capabilities must be sorted")
    _require(isinstance(manifest["resource_class"], dict), "pack resource class is malformed")
    for field in ("cpu_cores", "memory_mib", "disk_mib"):
        value = manifest["resource_class"].get(field)
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"pack resource class {field} must be positive")
    required_binaries = manifest["required_binaries"]
    _require(isinstance(required_binaries, list), "required binaries must be a list")
    for binary in required_binaries:
        _require(
            isinstance(binary, dict) and set(binary) == {"binary_id", "delivery", "license_review_required", "vendored"},
            "binary requirement is malformed",
        )
        _identifier(binary["binary_id"], "required binary id")
        _require(binary["delivery"] == "operator-provided-or-verified-image", "binary delivery must remain operator-provided")
        _require(
            binary["license_review_required"] is True and binary["vendored"] is False,
            "binary requirement must require license review and avoid vendoring",
        )
    _string_list(manifest["required_model_organs"], "required model organs")
    _require(isinstance(manifest["tool_adapters"], list), "tool adapters must be a list")
    for adapter in manifest["tool_adapters"]:
        _validate_adapter(adapter, "tool adapter")
        _require(
            set(adapter["required_capabilities"]).issubset(set(requested_capabilities)),
            "tool adapter requires a capability not granted by its pack",
        )
    _require(isinstance(manifest["source_adapters"], list) and bool(manifest["source_adapters"]), "source adapters cannot be empty")
    for adapter in manifest["source_adapters"]:
        _validate_source_adapter(adapter)
    _require(manifest["health_checks"] == [{"kind": "manifest-validation-only"}], "arbitrary pack health checks are forbidden")
    license_metadata = manifest["license_metadata"]
    _require(
        isinstance(license_metadata, dict)
        and license_metadata
        == {
            "operator_review_required": True,
            "pack_license": "LicenseRef-Substrate-Operator-Review",
            "upstream_components_not_vendored": True,
        },
        "pack license metadata is invalid",
    )
    denial_canary = manifest["capability_denial_canary"]
    _require(isinstance(denial_canary, dict), "capability denial canary is malformed")
    _require(
        denial_canary
        == {
            "denied_capabilities": sorted(_CAPABILITIES - set(requested_capabilities)),
            "execution_permitted": False,
            "network_mode": "none",
        },
        "capability denial canary does not match pack grants",
    )
    uninstall = manifest["uninstall_policy"]
    _require(
        isinstance(uninstall, dict)
        and uninstall
        == {
            "action": "remove-registry-reference-only",
            "execution_permitted": False,
            "shared_artifacts": "retain-until-explicit-cache-gc",
        },
        "pack uninstall policy is invalid",
    )
    descriptors = manifest["artifact_descriptors"]
    _require(isinstance(descriptors, list), "artifact descriptors must be a list")
    for descriptor in descriptors:
        _require(isinstance(descriptor, dict) and set(descriptor) == {"byte_length", "media_type", "sha256"}, "artifact descriptor is malformed")
        _sha256(descriptor["sha256"], "artifact descriptor sha256")
        _require(isinstance(descriptor["byte_length"], int) and descriptor["byte_length"] >= 0, "artifact descriptor byte length is invalid")
        _require(
            isinstance(descriptor["media_type"], str) and bool(descriptor["media_type"].strip()),
            "artifact descriptor media type is invalid",
        )


def _manifest_envelope(manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    return {
        "manifest": manifest,
        "manifest_sha256": sha256(manifest),
        "schema_version": PACK_ARTIFACT_SCHEMA_VERSION,
    }


def _load_manifest(artifact_directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(artifact_directory.is_dir() and not artifact_directory.is_symlink(), "pack artifact directory must be a real directory")
    envelope = _read_regular_json(artifact_directory / MANIFEST_NAME, "pack manifest")
    _require(set(envelope) == {"manifest", "manifest_sha256", "schema_version"}, "pack manifest envelope is malformed")
    _require(envelope["schema_version"] == PACK_ARTIFACT_SCHEMA_VERSION, "unsupported pack manifest envelope version")
    manifest = envelope["manifest"]
    _validate_manifest(manifest)
    _require(envelope["manifest_sha256"] == sha256(manifest), "pack manifest digest does not match its contents")
    return manifest, envelope


def build_pack_artifact(pack_name: str, artifact_directory: Path, *, publisher: str) -> dict[str, Any]:
    """Build an unsigned manifest-only pack artifact atomically."""

    _require(not artifact_directory.exists() and not artifact_directory.is_symlink(), f"pack artifact path already exists: {artifact_directory}")
    manifest = build_manifest(pack_name, publisher=publisher)
    try:
        artifact_directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{artifact_directory.name}.staging-", dir=artifact_directory.parent))
    except OSError as exc:
        raise ProductRefused(f"cannot create pack artifact staging directory: {exc}") from exc
    try:
        atomic_write_json(staging / MANIFEST_NAME, _manifest_envelope(manifest))
        os.replace(staging, artifact_directory)
        fsync_directory(artifact_directory.parent)
    except (OSError, ProductRefused) as exc:
        if staging.exists() and not artifact_directory.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, ProductRefused):
            raise
        raise ProductRefused(f"cannot publish pack artifact: {exc}") from exc
    return inspect_pack_artifact(artifact_directory)


def sign_pack_artifact(artifact_directory: Path, private_key_path: Path) -> dict[str, Any]:
    """Create a detached Ed25519 signature over a canonical manifest only."""

    manifest, envelope = _load_manifest(artifact_directory)
    signature_path = artifact_directory / SIGNATURE_NAME
    _require(not signature_path.exists() and not signature_path.is_symlink(), "pack artifact already has a signature")
    private_key = _load_private_key(private_key_path)
    public_key_b64 = _public_key_b64(private_key.public_key())
    signature = private_key.sign(_signature_payload(manifest))
    record = {
        "algorithm": "ed25519",
        "key_id": _key_id(public_key_b64),
        "manifest_sha256": envelope["manifest_sha256"],
        "public_key_b64": public_key_b64,
        "publisher": manifest["publisher"],
        "schema_version": PACK_SIGNATURE_SCHEMA_VERSION,
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    atomic_write_json(signature_path, record)
    return record


def _load_signature(artifact_directory: Path, manifest: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    record = _read_regular_json(artifact_directory / SIGNATURE_NAME, "pack signature")
    expected_fields = {"algorithm", "key_id", "manifest_sha256", "public_key_b64", "publisher", "schema_version", "signature_b64"}
    _require(set(record) == expected_fields, "pack signature fields are malformed")
    _require(record["schema_version"] == PACK_SIGNATURE_SCHEMA_VERSION, "unsupported pack signature schema version")
    _require(record["algorithm"] == "ed25519", "unsupported pack signature algorithm")
    _require(record["publisher"] == manifest["publisher"], "pack signature publisher does not match the manifest")
    _require(record["manifest_sha256"] == envelope["manifest_sha256"], "pack signature does not bind the manifest digest")
    _require(isinstance(record["public_key_b64"], str) and bool(record["public_key_b64"]), "pack signature public key is malformed")
    _require(record["key_id"] == _key_id(record["public_key_b64"]), "pack signature key id does not match its public key")
    _require(isinstance(record["signature_b64"], str) and bool(record["signature_b64"]), "pack signature is malformed")
    try:
        raw_key = base64.b64decode(record["public_key_b64"], validate=True)
        raw_signature = base64.b64decode(record["signature_b64"], validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(raw_key)
        public_key.verify(raw_signature, _signature_payload(manifest))
    except (InvalidSignature, ValueError, binascii.Error, TypeError) as exc:
        raise ProductRefused("pack signature verification failed") from exc
    return record


@dataclass(frozen=True)
class TrustRule:
    """One local, scoped publisher trust decision."""

    publisher: str
    key_id: str
    public_key_b64: str
    allowed_pack_names: tuple[str, ...]
    allowed_capabilities: tuple[str, ...]
    schema_version: str = PACK_TRUST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.publisher, "trust publisher")
        _sha256(self.key_id, "trust key id")
        _require(isinstance(self.public_key_b64, str) and bool(self.public_key_b64), "trust public key is malformed")
        _require(self.key_id == _key_id(self.public_key_b64), "trust key id does not match its public key")
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(self.public_key_b64, validate=True))
        except (ValueError, binascii.Error, TypeError) as exc:
            raise ProductRefused("trust public key must be an Ed25519 public key") from exc
        _require(isinstance(self.allowed_pack_names, tuple) and bool(self.allowed_pack_names), "trusted pack names cannot be empty")
        _require(all(bool(_IDENTIFIER.fullmatch(name)) for name in self.allowed_pack_names), "trusted pack name is malformed")
        _require(len(set(self.allowed_pack_names)) == len(self.allowed_pack_names), "trusted pack names cannot contain duplicates")
        _require(self.allowed_pack_names == tuple(sorted(self.allowed_pack_names)), "trusted pack names must be sorted")
        _require(isinstance(self.allowed_capabilities, tuple), "trusted capabilities must be a tuple")
        _require(set(self.allowed_capabilities).issubset(_CAPABILITIES), "trusted capability is unsupported")
        _require(len(set(self.allowed_capabilities)) == len(self.allowed_capabilities), "trusted capabilities cannot contain duplicates")
        _require(self.allowed_capabilities == tuple(sorted(self.allowed_capabilities)), "trusted capabilities must be sorted")
        _require(self.schema_version == PACK_TRUST_SCHEMA_VERSION, "unsupported trust rule schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_capabilities": list(self.allowed_capabilities),
            "allowed_pack_names": list(self.allowed_pack_names),
            "key_id": self.key_id,
            "public_key_b64": self.public_key_b64,
            "publisher": self.publisher,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrustRule:
        _require(isinstance(value, Mapping), "trust rule is malformed")
        _require(set(value) == _TRUST_RULE_FIELDS, "trust rule fields are malformed")
        try:
            return cls(
                publisher=value["publisher"],
                key_id=value["key_id"],
                public_key_b64=value["public_key_b64"],
                allowed_pack_names=tuple(value["allowed_pack_names"]),
                allowed_capabilities=tuple(value["allowed_capabilities"]),
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("trust rule is malformed") from exc


class LocalTrustStore:
    """A local explicit trust store; artifacts never trust themselves."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, publisher: str, key_id: str) -> Path:
        return self.root / "publishers" / _identifier(publisher, "trust publisher") / f"{_sha256(key_id, 'trust key id')}.json"

    def trust(
        self,
        *,
        publisher: str,
        public_key_path: Path,
        allowed_pack_names: tuple[str, ...],
        allowed_capabilities: tuple[str, ...],
    ) -> TrustRule:
        public_key_b64 = _public_key_b64(_load_public_key(public_key_path))
        rule = TrustRule(
            publisher=publisher,
            key_id=_key_id(public_key_b64),
            public_key_b64=public_key_b64,
            allowed_pack_names=tuple(sorted(allowed_pack_names)),
            allowed_capabilities=tuple(sorted(allowed_capabilities)),
        )
        target = self._path(rule.publisher, rule.key_id)
        if target.exists() or target.is_symlink():
            existing = TrustRule.from_dict(_read_regular_json(target, "trust rule"))
            _require(existing == rule, "refusing to replace an existing local trust decision")
            return existing
        atomic_write_json(target, rule.to_dict())
        return rule

    def load(self, publisher: str, key_id: str) -> TrustRule:
        target = self._path(publisher, key_id)
        _require(target.exists(), "pack signer is not trusted locally")
        return TrustRule.from_dict(_read_regular_json(target, "trust rule"))


def verify_pack_artifact(
    artifact_directory: Path,
    trust_store: LocalTrustStore,
) -> dict[str, Any]:
    """Verify signature plus local trust scope; never authorize execution."""

    _require(isinstance(trust_store, LocalTrustStore), "trust store is malformed")
    manifest, envelope = _load_manifest(artifact_directory)
    signature = _load_signature(artifact_directory, manifest, envelope)
    trust = trust_store.load(manifest["publisher"], signature["key_id"])
    _require(trust.public_key_b64 == signature["public_key_b64"], "local trust key does not match the pack signature")
    _require(manifest["name"] in trust.allowed_pack_names, "local trust rule does not permit this pack name")
    _require(set(manifest["capability_grants"]).issubset(trust.allowed_capabilities), "local trust rule does not permit this pack capability set")
    resolved_host = host_platform_id()
    _require(resolved_host in _SUPPORTED_HOSTS, "host identity is unsupported")
    _require(resolved_host in manifest["supported_hosts"], "pack is not compatible with this host")
    return {
        "execution_permitted": False,
        "manifest_sha256": envelope["manifest_sha256"],
        "pack": {"name": manifest["name"], "publisher": manifest["publisher"], "version": manifest["version"]},
        "signature_sha256": sha256(signature),
        "trust_rule_sha256": sha256(trust.to_dict()),
        "verified": True,
    }


class LocalPackRegistry:
    """An installed-pack registry that stores verified manifest references only."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _destination(self, manifest: Mapping[str, Any], manifest_sha256: str) -> Path:
        return self.root / "packs" / manifest["name"] / manifest["version"] / manifest_sha256

    def install(self, artifact_directory: Path, trust_store: LocalTrustStore) -> dict[str, Any]:
        verification = verify_pack_artifact(artifact_directory, trust_store)
        manifest, envelope = _load_manifest(artifact_directory)
        destination = self._destination(manifest, envelope["manifest_sha256"])
        _require(not destination.exists() and not destination.is_symlink(), "this verified pack is already installed")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
            for filename in (MANIFEST_NAME, SIGNATURE_NAME):
                source = artifact_directory / filename
                _require_regular_file(source, f"pack {filename}", reject_hardlinks=True)
                shutil.copyfile(source, staging / filename)
            # Re-verify the copied bytes rather than assuming the source stayed
            # stable between the initial verification and the copy. The registry
            # must publish exactly the manifest/signature pair it records.
            staged_verification = verify_pack_artifact(staging, trust_store)
            _require(staged_verification == verification, "pack artifact changed while it was being installed")
            receipt = {
                "execution_permitted": False,
                "manifest_sha256": envelope["manifest_sha256"],
                "schema_version": PACK_INSTALL_RECEIPT_SCHEMA_VERSION,
                "verification": verification,
            }
            atomic_write_json(staging / INSTALL_RECEIPT_NAME, receipt)
            os.replace(staging, destination)
            fsync_directory(destination.parent)
        except (OSError, ProductRefused) as exc:
            if "staging" in locals() and staging.exists() and not destination.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, ProductRefused):
                raise
            raise ProductRefused(f"cannot install verified pack: {exc}") from exc
        return {"installation": receipt, "installation_path": str(destination)}

    def remove(self, *, pack_name: str, version: str, manifest_sha256: str) -> dict[str, Any]:
        _identifier(pack_name, "pack name")
        _require(isinstance(version, str) and bool(_SEMVER.fullmatch(version)), "pack version must be a semantic version")
        _sha256(manifest_sha256, "manifest sha256")
        destination = self.root / "packs" / pack_name / version / manifest_sha256
        _require(destination.is_dir() and not destination.is_symlink(), "installed pack reference does not exist")
        try:
            shutil.rmtree(destination)
            fsync_directory(destination.parent)
        except OSError as exc:
            raise ProductRefused(f"cannot remove installed pack reference: {exc}") from exc
        return {
            "execution_permitted": False,
            "manifest_sha256": manifest_sha256,
            "pack_name": pack_name,
            "removed_registry_reference_only": True,
            "version": version,
        }


def inspect_pack_artifact(artifact_directory: Path) -> dict[str, Any]:
    """Inspect an artifact without treating an unsigned artifact as trusted."""

    manifest, envelope = _load_manifest(artifact_directory)
    signature_path = artifact_directory / SIGNATURE_NAME
    signature: dict[str, Any] | None = None
    if signature_path.exists() or signature_path.is_symlink():
        signature = _load_signature(artifact_directory, manifest, envelope)
    return {
        "artifact_directory": str(artifact_directory),
        "execution_permitted": False,
        "manifest": manifest,
        "manifest_sha256": envelope["manifest_sha256"],
        "signature": signature,
        "signed": signature is not None,
    }
