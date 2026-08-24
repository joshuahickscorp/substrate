"""Immutable, non-executing manifests for operator-provided tool bundles.

This is a *contract* for a future sandbox broker, not a package manager or a
tool launcher.  It deliberately does not download, unpack, open, inspect, or
invoke FFmpeg, yt-dlp, Chromium, Git, compilers, proof tools, Blender, or an
OCI runtime.  Instead it records the exact content digest and legal material
that a separately controlled installer/broker would have to verify before a
tool could ever be considered for a bounded execution lane.

There are no fields for commands, arguments, executable paths, image tags,
working directories, mounts, credentials, browser profiles, URLs, or network
allowlists.  Unknown fields are refused during parsing so those capabilities
cannot be smuggled into an otherwise valid, digest-bound manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from substrate.product.codec import sha256
from substrate.product.contracts import ProductRefused

TOOL_BUNDLE_MANIFEST_SCHEMA_VERSION = "substrate-tool-bundle-manifest-v1"

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
# Upstream browser releases commonly carry a fourth numeric component, while
# the bundle itself uses ordinary semantic versioning.  Keep the artifact
# field strict and tag-free, but do not make a legitimate pinned Chromium
# build impossible to represent.
_TOOL_RELEASE_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){1,4}(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPDX_IDENTIFIER = re.compile(r"^[A-Za-z0-9.+-]{1,80}$")

# A bundle is deliberately limited to a host/platform-specific immutable
# artifact.  A future multi-platform release publishes one manifest per target,
# rather than allowing a mutable tag or host-dependent lookup here.
SUPPORTED_PLATFORMS = frozenset({"darwin-arm64", "linux-amd64", "linux-arm64", "windows-amd64"})
DISTRIBUTION_KINDS = frozenset({"binary-archive", "oci-image"})

# These are product vocabulary names, not executable filenames.  Keeping this
# set closed stops a manifest from introducing an arbitrary local program under
# a seemingly legitimate adapter role.
SUPPORTED_TOOL_IDS = frozenset({"blender", "clang", "chromium", "ffmpeg", "git", "go", "lean", "rustc", "yt-dlp", "z3"})

CAPABILITIES = frozenset(
    {
        "browser-observation",
        "compiler-analysis",
        "formal-verification",
        "media-decode",
        "repository-inspection",
        "scene-observation",
        "source-staging",
    }
)

NETWORK_MODE_NONE = "none"
_MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024


class _AdapterSpec:
    """Closed policy for one semantic adapter role."""

    def __init__(self, *, tools: frozenset[str], operations: frozenset[str], capability: str) -> None:
        self.tools = tools
        self.operations = operations
        self.capability = capability


# Names shared with source_adapters.py are intentionally protocol roles.  The
# remaining roles are equally declarative; none selects a command line or
# permits process/network execution.
_ADAPTER_SPECS: dict[str, _AdapterSpec] = {
    "approved-media-staging-v1": _AdapterSpec(
        tools=frozenset({"yt-dlp"}),
        operations=frozenset({"stage-approved-media", "stage-approved-metadata"}),
        capability="source-staging",
    ),
    "browser-observation-v1": _AdapterSpec(
        tools=frozenset({"chromium"}),
        operations=frozenset({"accessibility", "audio-capture", "dom", "frame-capture", "screenshot"}),
        capability="browser-observation",
    ),
    "compiler-analysis-v1": _AdapterSpec(
        tools=frozenset({"clang", "go", "rustc"}),
        operations=frozenset({"compile-test", "parse-source", "typecheck"}),
        capability="compiler-analysis",
    ),
    "formal-verification-v1": _AdapterSpec(
        tools=frozenset({"lean", "z3"}),
        operations=frozenset({"check-proof", "check-smt"}),
        capability="formal-verification",
    ),
    "media-observation-v1": _AdapterSpec(
        tools=frozenset({"ffmpeg"}),
        operations=frozenset({"extract-audio", "metadata", "probe", "sample-frames", "subtitles"}),
        capability="media-decode",
    ),
    "repository-inspection-v1": _AdapterSpec(
        tools=frozenset({"git"}),
        operations=frozenset({"commit-metadata", "file-metadata", "text-content", "tree"}),
        capability="repository-inspection",
    ),
    "scene-observation-v1": _AdapterSpec(
        tools=frozenset({"blender"}),
        operations=frozenset({"inspect-scene", "render-frame-samples"}),
        capability="scene-observation",
    ),
}

ADAPTER_ROLES = frozenset(_ADAPTER_SPECS)

_TOOL_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "artifact_size_bytes",
        "distribution_kind",
        "license_document_sha256",
        "license_spdx",
        "notices_sha256",
        "release_version",
        "sbom_sha256",
        "target_platform",
        "tool_id",
        "verification_receipt_sha256",
    }
)
_ADAPTER_BINDING_FIELDS = frozenset({"adapter_role", "operations", "tool_id"})
_MANIFEST_FIELDS = frozenset(
    {
        "adapter_bindings",
        "capabilities",
        "bundle_id",
        "execution_permitted",
        "manifest_sha256",
        "network_mode",
        "schema_version",
        "target_platform",
        "tools",
        "version",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductRefused(message)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProductRefused(f"{label} must be a lowercase identifier")
    return value


def _semver(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SEMVER.fullmatch(value):
        raise ProductRefused(f"{label} must be a semantic version")
    return value


def _tool_release_version(value: object, label: str) -> str:
    if not isinstance(value, str) or not _TOOL_RELEASE_VERSION.fullmatch(value):
        raise ProductRefused(f"{label} must be a pinned numeric release version")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProductRefused(f"{label} must be a lowercase SHA-256 digest")
    return value


def _platform(value: object, label: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_PLATFORMS:
        raise ProductRefused(f"{label} must be a supported platform")
    return value


def _tool_id(value: object, label: str) -> str:
    identifier = _identifier(value, label)
    if identifier not in SUPPORTED_TOOL_IDS:
        raise ProductRefused(f"{label} is not an approved upstream utility")
    return identifier


def _closed_tuple(values: object, *, allowed: frozenset[str], label: str, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ProductRefused(f"{label} must be a tuple")
    if not all(isinstance(value, str) and value in allowed for value in values):
        raise ProductRefused(f"{label} includes an unsupported value")
    result = cast(tuple[str, ...], values)
    if len(set(result)) != len(result):
        raise ProductRefused(f"{label} cannot contain duplicates")
    if nonempty and not result:
        raise ProductRefused(f"{label} cannot be empty")
    if result != tuple(sorted(result)):
        raise ProductRefused(f"{label} must be lexicographically sorted")
    return result


def _bounded_bytes(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAXIMUM_ARTIFACT_BYTES:
        raise ProductRefused(f"{label} must be between 1 and {_MAXIMUM_ARTIFACT_BYTES}")
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProductRefused(f"{label} must be an object")
    return cast(dict[str, object], value)


def _exact_fields(value: object, *, fields: frozenset[str], label: str) -> dict[str, object]:
    document = _mapping(value, label)
    if set(document) != fields:
        unknown = sorted(set(document) - fields)
        missing = sorted(fields - set(document))
        details: list[str] = []
        if unknown:
            details.append(f"unknown={unknown}")
        if missing:
            details.append(f"missing={missing}")
        detail = ", ".join(details)
        raise ProductRefused(f"{label} has an invalid field set ({detail})")
    return document


def _object_list(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ProductRefused(f"{label} must be a nonempty list")
    return [_mapping(item, f"{label} item") for item in value]


def _string_list(value: object, *, allowed: frozenset[str], label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProductRefused(f"{label} must be a list")
    return _closed_tuple(tuple(value), allowed=allowed, label=label)


@dataclass(frozen=True)
class ToolArtifact:
    """One operator-provided binary archive or OCI image pinned by digest.

    The four evidence digests intentionally make legal and verification
    material content-addressable alongside the artifact itself.  This object
    never resolves any digest to a file, registry, image tag, or executable.
    """

    tool_id: str
    release_version: str
    distribution_kind: str
    target_platform: str
    artifact_sha256: str
    artifact_size_bytes: int
    sbom_sha256: str
    notices_sha256: str
    license_document_sha256: str
    license_spdx: str
    verification_receipt_sha256: str

    def __post_init__(self) -> None:
        _tool_id(self.tool_id, "tool id")
        _tool_release_version(self.release_version, "tool release version")
        _require(self.distribution_kind in DISTRIBUTION_KINDS, "tool distribution kind is unsupported")
        _platform(self.target_platform, "tool target platform")
        _digest(self.artifact_sha256, "tool artifact sha256")
        _bounded_bytes(self.artifact_size_bytes, "tool artifact size bytes")
        _digest(self.sbom_sha256, "tool SBOM sha256")
        _digest(self.notices_sha256, "tool notices sha256")
        _digest(self.license_document_sha256, "tool license document sha256")
        if not isinstance(self.license_spdx, str) or not _SPDX_IDENTIFIER.fullmatch(self.license_spdx):
            raise ProductRefused("tool license SPDX identifier is malformed")
        _digest(self.verification_receipt_sha256, "tool verification receipt sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "distribution_kind": self.distribution_kind,
            "license_document_sha256": self.license_document_sha256,
            "license_spdx": self.license_spdx,
            "notices_sha256": self.notices_sha256,
            "release_version": self.release_version,
            "sbom_sha256": self.sbom_sha256,
            "target_platform": self.target_platform,
            "tool_id": self.tool_id,
            "verification_receipt_sha256": self.verification_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ToolArtifact:
        document = _exact_fields(value, fields=_TOOL_ARTIFACT_FIELDS, label="tool artifact")
        return cls(
            tool_id=cast(str, document["tool_id"]),
            release_version=cast(str, document["release_version"]),
            distribution_kind=cast(str, document["distribution_kind"]),
            target_platform=cast(str, document["target_platform"]),
            artifact_sha256=cast(str, document["artifact_sha256"]),
            artifact_size_bytes=cast(int, document["artifact_size_bytes"]),
            sbom_sha256=cast(str, document["sbom_sha256"]),
            notices_sha256=cast(str, document["notices_sha256"]),
            license_document_sha256=cast(str, document["license_document_sha256"]),
            license_spdx=cast(str, document["license_spdx"]),
            verification_receipt_sha256=cast(str, document["verification_receipt_sha256"]),
        )


@dataclass(frozen=True)
class AdapterBinding:
    """A closed semantic role and operation set for one pinned utility."""

    adapter_role: str
    tool_id: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_role, str) or self.adapter_role not in _ADAPTER_SPECS:
            raise ProductRefused("adapter role is unsupported")
        tool_id = _tool_id(self.tool_id, "adapter tool id")
        spec = _ADAPTER_SPECS[self.adapter_role]
        _require(tool_id in spec.tools, "adapter tool is incompatible with its role")
        _closed_tuple(self.operations, allowed=spec.operations, label="adapter operations")

    def to_dict(self) -> dict[str, object]:
        return {"adapter_role": self.adapter_role, "operations": list(self.operations), "tool_id": self.tool_id}

    @classmethod
    def from_dict(cls, value: object) -> AdapterBinding:
        document = _exact_fields(value, fields=_ADAPTER_BINDING_FIELDS, label="adapter binding")
        role = document["adapter_role"]
        if not isinstance(role, str) or role not in _ADAPTER_SPECS:
            raise ProductRefused("adapter role is unsupported")
        return cls(
            adapter_role=role,
            tool_id=cast(str, document["tool_id"]),
            operations=_string_list(document["operations"], allowed=_ADAPTER_SPECS[role].operations, label="adapter operations"),
        )


@dataclass(frozen=True)
class ToolBundleManifest:
    """A portable manifest that can be hashed but cannot authorize execution."""

    bundle_id: str
    version: str
    target_platform: str
    tools: tuple[ToolArtifact, ...]
    adapter_bindings: tuple[AdapterBinding, ...]
    capabilities: tuple[str, ...]
    network_mode: str = NETWORK_MODE_NONE
    execution_permitted: bool = False

    def __post_init__(self) -> None:
        _identifier(self.bundle_id, "bundle id")
        _semver(self.version, "bundle version")
        _platform(self.target_platform, "bundle target platform")
        if not isinstance(self.tools, tuple) or not self.tools or not all(isinstance(tool, ToolArtifact) for tool in self.tools):
            raise ProductRefused("bundle tools must be a nonempty tuple of tool artifacts")
        if not isinstance(self.adapter_bindings, tuple) or not self.adapter_bindings or not all(
            isinstance(binding, AdapterBinding) for binding in self.adapter_bindings
        ):
            raise ProductRefused("bundle adapter bindings must be a nonempty tuple of adapter bindings")
        tool_ids = tuple(tool.tool_id for tool in self.tools)
        if len(set(tool_ids)) != len(tool_ids):
            raise ProductRefused("bundle tools cannot contain duplicate tool ids")
        if tool_ids != tuple(sorted(tool_ids)):
            raise ProductRefused("bundle tools must be sorted by tool id")
        _require(
            all(tool.target_platform == self.target_platform for tool in self.tools),
            "tool artifact platform must match the bundle target platform",
        )
        binding_keys = tuple((binding.adapter_role, binding.tool_id) for binding in self.adapter_bindings)
        if len(set(binding_keys)) != len(binding_keys):
            raise ProductRefused("bundle adapter bindings cannot duplicate a role and tool")
        if binding_keys != tuple(sorted(binding_keys)):
            raise ProductRefused("bundle adapter bindings must be sorted by role and tool")
        if len({binding.tool_id for binding in self.adapter_bindings}) != len(self.adapter_bindings):
            raise ProductRefused("bundle adapter bindings cannot reuse a tool across roles")
        bound_tool_ids = {binding.tool_id for binding in self.adapter_bindings}
        _require(bound_tool_ids == set(tool_ids), "every bundle tool must have exactly one adapter binding")
        declared_capabilities = _closed_tuple(self.capabilities, allowed=CAPABILITIES, label="bundle capabilities")
        required_capabilities = tuple(sorted({_ADAPTER_SPECS[binding.adapter_role].capability for binding in self.adapter_bindings}))
        _require(declared_capabilities == required_capabilities, "bundle capabilities must exactly match adapter roles")
        _require(self.network_mode == NETWORK_MODE_NONE, "tool bundle network mode must be none")
        _require(self.execution_permitted is False, "tool bundle execution_permitted must be false")

    def unsigned_dict(self) -> dict[str, object]:
        """Return the full content-addressed payload, without its self-digest."""

        return {
            "adapter_bindings": [binding.to_dict() for binding in self.adapter_bindings],
            "capabilities": list(self.capabilities),
            "bundle_id": self.bundle_id,
            "execution_permitted": False,
            "network_mode": NETWORK_MODE_NONE,
            "schema_version": TOOL_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "target_platform": self.target_platform,
            "tools": [tool.to_dict() for tool in self.tools],
            "version": self.version,
        }

    @property
    def manifest_sha256(self) -> str:
        return sha256(self.unsigned_dict())

    def to_document(self) -> dict[str, object]:
        """Return a canonical-data-compatible document with a self-digest."""

        return {**self.unsigned_dict(), "manifest_sha256": self.manifest_sha256}


def parse_tool_bundle_manifest(value: object) -> ToolBundleManifest:
    """Parse and re-hash an untrusted manifest document without touching tools.

    A caller still needs to verify that the operator-provided artifact, SBOM,
    notices, license document, and verification receipt really match these
    digests.  This function only makes it impossible for this manifest itself
    to carry mutable tags, paths, commands, flags, or a network grant.
    """

    document = _exact_fields(value, fields=_MANIFEST_FIELDS, label="tool bundle manifest")
    if document["schema_version"] != TOOL_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ProductRefused("tool bundle manifest schema version is unsupported")
    tools = tuple(ToolArtifact.from_dict(item) for item in _object_list(document["tools"], "bundle tools"))
    bindings = tuple(AdapterBinding.from_dict(item) for item in _object_list(document["adapter_bindings"], "bundle adapter bindings"))
    manifest = ToolBundleManifest(
        bundle_id=cast(str, document["bundle_id"]),
        version=cast(str, document["version"]),
        target_platform=cast(str, document["target_platform"]),
        tools=tools,
        adapter_bindings=bindings,
        capabilities=_string_list(document["capabilities"], allowed=CAPABILITIES, label="bundle capabilities"),
        network_mode=cast(str, document["network_mode"]),
        execution_permitted=cast(bool, document["execution_permitted"]),
    )
    expected_digest = _digest(document["manifest_sha256"], "tool bundle manifest sha256")
    _require(expected_digest == manifest.manifest_sha256, "tool bundle manifest digest does not match its content")
    return manifest
