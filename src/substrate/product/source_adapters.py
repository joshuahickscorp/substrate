"""Typed, cache-bound observation plans for future source adapters.

This module is intentionally a contract boundary, not an adapter implementation.
It never opens a local path, starts a process, contacts a remote endpoint, or
loads a browser profile.  Each plan references evidence by an immutable cache
identifier and digest, describes a small closed set of observations, and
requires every prospective output to enter cache quarantine for independent
verification.

Acquisition is deliberately outside this module.  A future broker may create
an approved source receipt and cache object through the product's separate
source-policy and cache boundaries.  Only then may it be referenced here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from substrate.product.codec import sha256
from substrate.product.contracts import ProductRefused

SOURCE_ADAPTER_PLAN_SCHEMA_VERSION = "substrate-source-adapter-plan-v1"
QUARANTINE_ZONE = "quarantine"
# Raw inputs live in ``verified`` and independently verified derivatives live
# in ``processed``.  Both are immutable cache evidence; neither is a live
# browser/session/URL/path handle.
REQUIRED_INPUT_ZONES = ("verified", "processed")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")

LOCAL_FILE_VIEWS = frozenset({"metadata", "text", "document-structure"})
REPOSITORY_SNAPSHOT_VIEWS = frozenset({"commit-metadata", "file-metadata", "text-content", "tree"})
MEDIA_OPERATIONS = frozenset({"extract-audio", "metadata", "probe", "sample-frames", "subtitles"})
BROWSER_OPERATIONS = frozenset({"accessibility", "audio-capture", "dom", "frame-capture", "screenshot"})

_PLAN_FIELDS = frozenset(
    {
        "denied_capabilities",
        "execution_permitted",
        "execution_refusal",
        "input_requirements",
        "output_descriptors",
        "output_requirements",
        "plan_sha256",
        "request",
        "schema_version",
    }
)
_CACHE_ARTIFACT_REF_FIELDS = frozenset({"artifact_sha256", "cache_id"})
_LOCAL_FILE_REQUEST_FIELDS = frozenset({"artifact", "expected_media_types", "type", "views"})
_REPOSITORY_REQUEST_FIELDS = frozenset({"maximum_files", "maximum_text_bytes", "repository_id", "snapshot", "type", "views"})
_MEDIA_REQUEST_FIELDS = frozenset({"artifact", "expected_media_type", "frame_sample_count", "maximum_audio_seconds", "operations", "type"})
_BROWSER_REQUEST_FIELDS = frozenset(
    {"capture", "expected_capture_media_type", "frame_sample_count", "maximum_audio_seconds", "operations", "type"}
)

_MAXIMUM_REPOSITORY_FILES = 10_000
_MAXIMUM_REPOSITORY_TEXT_BYTES = 64 * 1024 * 1024
_MAXIMUM_FRAME_SAMPLES = 120
_MAXIMUM_AUDIO_SECONDS = 3_600

# These are policy-level refusals, rather than advisory warnings.  A future
# executable broker must make every one of them true before it can carry out a
# plan created here.
DENIED_CAPABILITIES = (
    "arbitrary-command-execution",
    "arbitrary-tool-flags",
    "browser-cookies",
    "browser-profile-access",
    "cache-promotion",
    "credential-access",
    "downloader-invocation",
    "host-filesystem-path-access",
    "network-egress",
    "process-execution",
    "shell-execution",
)


class _ObservationRequest:
    """Marker base class for the closed request union used by ``plan_observation``."""


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


def _media_type(value: object, label: str) -> str:
    if not isinstance(value, str) or not _MEDIA_TYPE.fullmatch(value):
        raise ProductRefused(f"{label} must be a lowercase media type")
    return value


def _closed_tuple(values: object, *, allowed: frozenset[str], label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ProductRefused(f"{label} must be a nonempty tuple")
    if not all(isinstance(value, str) and value in allowed for value in values):
        raise ProductRefused(f"{label} includes an unsupported value")
    typed_values = tuple(values)
    if len(set(typed_values)) != len(typed_values):
        raise ProductRefused(f"{label} cannot contain duplicates")
    return typed_values


def _bounded_integer(value: object, *, minimum: int, maximum: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ProductRefused(f"{label} must be between {minimum} and {maximum}")
    return value


def _exact_object(value: object, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    """Return a JSON-object-shaped mapping with no extension fields."""

    if not isinstance(value, dict):
        raise ProductRefused(f"{label} must be an object")
    document = cast(dict[str, Any], value)
    _require(set(document) == fields, f"{label} fields are malformed")
    return document


def _string_tuple_from_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProductRefused(f"{label} must be a list")
    items = cast(list[object], value)
    _require(all(isinstance(item, str) for item in items), f"{label} must contain strings")
    return tuple(cast(str, item) for item in items)


def _parse_cache_artifact_ref(value: object, label: str) -> CacheArtifactRef:
    document = _exact_object(value, fields=_CACHE_ARTIFACT_REF_FIELDS, label=label)
    try:
        return CacheArtifactRef(cache_id=document["cache_id"], artifact_sha256=document["artifact_sha256"])
    except (KeyError, TypeError) as exc:
        raise ProductRefused(f"{label} is malformed") from exc


@dataclass(frozen=True)
class CacheArtifactRef:
    """An immutable evidence reference; never a path, URI, or live handle."""

    cache_id: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.cache_id, "cache id")
        _digest(self.artifact_sha256, "cache artifact sha256")

    def to_dict(self) -> dict[str, str]:
        return {"artifact_sha256": self.artifact_sha256, "cache_id": self.cache_id}


@dataclass(frozen=True)
class LocalFileObservationRequest(_ObservationRequest):
    """Observe an operator-imported cache artifact without reopening its path."""

    artifact: CacheArtifactRef
    views: tuple[str, ...] = ("metadata",)
    expected_media_types: tuple[str, ...] = ("text/plain",)

    def __post_init__(self) -> None:
        _require(isinstance(self.artifact, CacheArtifactRef), "local file artifact reference is malformed")
        _closed_tuple(self.views, allowed=LOCAL_FILE_VIEWS, label="local file views")
        _require(isinstance(self.expected_media_types, tuple) and bool(self.expected_media_types), "local file media types must be a nonempty tuple")
        _require(len(set(self.expected_media_types)) == len(self.expected_media_types), "local file media types cannot contain duplicates")
        for media_type in self.expected_media_types:
            _media_type(media_type, "local file media type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "expected_media_types": list(self.expected_media_types),
            "type": "local-file-observation",
            "views": list(self.views),
        }


@dataclass(frozen=True)
class RepositorySnapshotObservationRequest(_ObservationRequest):
    """Inspect a cache-backed, immutable repository snapshot manifest."""

    repository_id: str
    snapshot: CacheArtifactRef
    views: tuple[str, ...] = ("tree",)
    maximum_files: int = 1_024
    maximum_text_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        _identifier(self.repository_id, "repository id")
        _require(isinstance(self.snapshot, CacheArtifactRef), "repository snapshot reference is malformed")
        _closed_tuple(self.views, allowed=REPOSITORY_SNAPSHOT_VIEWS, label="repository snapshot views")
        _bounded_integer(self.maximum_files, minimum=1, maximum=_MAXIMUM_REPOSITORY_FILES, label="repository maximum files")
        _bounded_integer(
            self.maximum_text_bytes,
            minimum=1,
            maximum=_MAXIMUM_REPOSITORY_TEXT_BYTES,
            label="repository maximum text bytes",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_files": self.maximum_files,
            "maximum_text_bytes": self.maximum_text_bytes,
            "repository_id": self.repository_id,
            "snapshot": self.snapshot.to_dict(),
            "type": "repository-snapshot-observation",
            "views": list(self.views),
        }


@dataclass(frozen=True)
class MediaObservationRequest(_ObservationRequest):
    """Describe fixed media observations over an immutable cache artifact."""

    artifact: CacheArtifactRef
    operations: tuple[str, ...]
    expected_media_type: str
    frame_sample_count: int = 0
    maximum_audio_seconds: int = 0

    def __post_init__(self) -> None:
        _require(isinstance(self.artifact, CacheArtifactRef), "media artifact reference is malformed")
        _closed_tuple(self.operations, allowed=MEDIA_OPERATIONS, label="media operations")
        _media_type(self.expected_media_type, "media expected media type")
        _bounded_integer(self.frame_sample_count, minimum=0, maximum=_MAXIMUM_FRAME_SAMPLES, label="media frame sample count")
        _bounded_integer(self.maximum_audio_seconds, minimum=0, maximum=_MAXIMUM_AUDIO_SECONDS, label="media maximum audio seconds")
        _require(
            ("sample-frames" in self.operations) == (self.frame_sample_count > 0),
            "media frame sample count must be set only for sample-frames",
        )
        _require(
            ("extract-audio" in self.operations) == (self.maximum_audio_seconds > 0),
            "media maximum audio seconds must be set only for extract-audio",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "expected_media_type": self.expected_media_type,
            "frame_sample_count": self.frame_sample_count,
            "maximum_audio_seconds": self.maximum_audio_seconds,
            "operations": list(self.operations),
            "type": "media-observation",
        }


@dataclass(frozen=True)
class BrowserObservationRequest(_ObservationRequest):
    """Observe an already-captured page bundle without opening a live browser."""

    capture: CacheArtifactRef
    operations: tuple[str, ...]
    expected_capture_media_type: str = "application/x-substrate-browser-capture+json"
    frame_sample_count: int = 0
    maximum_audio_seconds: int = 0

    def __post_init__(self) -> None:
        _require(isinstance(self.capture, CacheArtifactRef), "browser capture reference is malformed")
        _closed_tuple(self.operations, allowed=BROWSER_OPERATIONS, label="browser operations")
        _media_type(self.expected_capture_media_type, "browser capture media type")
        _bounded_integer(self.frame_sample_count, minimum=0, maximum=_MAXIMUM_FRAME_SAMPLES, label="browser frame sample count")
        _bounded_integer(self.maximum_audio_seconds, minimum=0, maximum=_MAXIMUM_AUDIO_SECONDS, label="browser maximum audio seconds")
        _require(
            ("frame-capture" in self.operations) == (self.frame_sample_count > 0),
            "browser frame sample count must be set only for frame-capture",
        )
        _require(
            ("audio-capture" in self.operations) == (self.maximum_audio_seconds > 0),
            "browser maximum audio seconds must be set only for audio-capture",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": self.capture.to_dict(),
            "expected_capture_media_type": self.expected_capture_media_type,
            "frame_sample_count": self.frame_sample_count,
            "maximum_audio_seconds": self.maximum_audio_seconds,
            "operations": list(self.operations),
            "type": "browser-observation",
        }


ObservationRequest: TypeAlias = (
    LocalFileObservationRequest | RepositorySnapshotObservationRequest | MediaObservationRequest | BrowserObservationRequest
)


def _parse_observation_request(value: object) -> ObservationRequest:
    """Rebuild one closed request from an untrusted serialized document."""

    if not isinstance(value, dict):
        raise ProductRefused("source adapter request must be an object")
    request_document = cast(dict[str, Any], value)
    request_type = request_document.get("type")
    try:
        if request_type == "local-file-observation":
            document = _exact_object(request_document, fields=_LOCAL_FILE_REQUEST_FIELDS, label="local file source adapter request")
            return LocalFileObservationRequest(
                artifact=_parse_cache_artifact_ref(document["artifact"], "local file cache artifact"),
                views=_string_tuple_from_list(document["views"], "local file views"),
                expected_media_types=_string_tuple_from_list(document["expected_media_types"], "local file media types"),
            )
        if request_type == "repository-snapshot-observation":
            document = _exact_object(request_document, fields=_REPOSITORY_REQUEST_FIELDS, label="repository source adapter request")
            return RepositorySnapshotObservationRequest(
                repository_id=document["repository_id"],
                snapshot=_parse_cache_artifact_ref(document["snapshot"], "repository snapshot cache artifact"),
                views=_string_tuple_from_list(document["views"], "repository snapshot views"),
                maximum_files=document["maximum_files"],
                maximum_text_bytes=document["maximum_text_bytes"],
            )
        if request_type == "media-observation":
            document = _exact_object(request_document, fields=_MEDIA_REQUEST_FIELDS, label="media source adapter request")
            return MediaObservationRequest(
                artifact=_parse_cache_artifact_ref(document["artifact"], "media cache artifact"),
                operations=_string_tuple_from_list(document["operations"], "media operations"),
                expected_media_type=document["expected_media_type"],
                frame_sample_count=document["frame_sample_count"],
                maximum_audio_seconds=document["maximum_audio_seconds"],
            )
        if request_type == "browser-observation":
            document = _exact_object(request_document, fields=_BROWSER_REQUEST_FIELDS, label="browser source adapter request")
            return BrowserObservationRequest(
                capture=_parse_cache_artifact_ref(document["capture"], "browser capture cache artifact"),
                operations=_string_tuple_from_list(document["operations"], "browser operations"),
                expected_capture_media_type=document["expected_capture_media_type"],
                frame_sample_count=document["frame_sample_count"],
                maximum_audio_seconds=document["maximum_audio_seconds"],
            )
    except (KeyError, TypeError) as exc:
        raise ProductRefused("source adapter request is malformed") from exc
    raise ProductRefused("source adapter request type is unsupported")


def _output_descriptors(request: ObservationRequest) -> list[dict[str, str]]:
    """Return fixed output roles without naming a binary, command, or output path."""

    if isinstance(request, LocalFileObservationRequest):
        return [{"media_type": "application/x-substrate-local-file-observation+json", "role": view} for view in request.views]
    if isinstance(request, RepositorySnapshotObservationRequest):
        return [{"media_type": "application/x-substrate-repository-observation+json", "role": view} for view in request.views]
    if isinstance(request, MediaObservationRequest):
        roles = {
            "extract-audio": "audio-derivative",
            "metadata": "metadata",
            "probe": "probe",
            "sample-frames": "frame-sample-set",
            "subtitles": "subtitle-tracks",
        }
        return [{"media_type": "application/x-substrate-media-observation+json", "role": roles[operation]} for operation in request.operations]
    if isinstance(request, BrowserObservationRequest):
        roles = {
            "accessibility": "accessibility-tree",
            "audio-capture": "audio-capture",
            "dom": "dom-snapshot",
            "frame-capture": "frame-sample-set",
            "screenshot": "screenshot",
        }
        return [{"media_type": "application/x-substrate-browser-observation+json", "role": roles[operation]} for operation in request.operations]
    raise ProductRefused("source adapter request is malformed")


def plan_observation(request: ObservationRequest) -> dict[str, Any]:
    """Build a deterministic, non-executing, quarantine-only observation plan.

    The plan is intentionally not enough to invoke a tool: it contains no URL,
    file path, command, flag, browser profile, cookie, credential, network
    grant, or executable implementation selector.
    """

    _require(
        isinstance(
            request,
            (LocalFileObservationRequest, RepositorySnapshotObservationRequest, MediaObservationRequest, BrowserObservationRequest),
        ),
        "source adapter request is malformed",
    )
    request_document = request.to_dict()
    plan: dict[str, Any] = {
        "denied_capabilities": list(DENIED_CAPABILITIES),
        "execution_permitted": False,
        "execution_refusal": "source adapter execution is not configured",
        "input_requirements": {
            "cache_zones": list(REQUIRED_INPUT_ZONES),
            "descriptor_verification_status": "verified",
            "immutable_digest_match": True,
            "live_source_access": "forbidden",
        },
        "output_descriptors": _output_descriptors(request),
        "output_requirements": {
            "artifact_kind": "derived",
            "cache_zone": QUARANTINE_ZONE,
            "execution_permitted": False,
            "promotion_requires": "separate-cache-attestation-and-revalidation",
        },
        "request": request_document,
        "schema_version": SOURCE_ADAPTER_PLAN_SCHEMA_VERSION,
    }
    plan["plan_sha256"] = sha256(plan)
    return plan


def parse_observation_plan(value: object) -> dict[str, Any]:
    """Strictly validate and re-hash serialized source-adapter plan data.

    This is the future broker-facing counterpart to :func:`plan_observation`.
    It accepts no extension fields, rebuilds the closed typed request, verifies
    the self-digest, and requires every remaining plan field to be exactly the
    one derived from that request.  It remains a parser only: it does not open
    the referenced cache object, select a tool, or execute an operation.

    Callers receiving raw JSON bytes should decode them with
    :func:`substrate.product.codec.read_json` first so duplicate keys and
    non-standard JSON constants are rejected before this object-level check.
    """

    document = _exact_object(value, fields=_PLAN_FIELDS, label="source adapter plan")
    _require(document["schema_version"] == SOURCE_ADAPTER_PLAN_SCHEMA_VERSION, "source adapter plan schema version is unsupported")
    recorded_digest = _digest(document["plan_sha256"], "source adapter plan sha256")
    _require(
        recorded_digest == sha256({key: item for key, item in document.items() if key != "plan_sha256"}),
        "source adapter plan digest does not match its contents",
    )
    request = _parse_observation_request(document["request"])
    expected = plan_observation(request)
    _require(document == expected, "source adapter plan does not match its typed contract")
    return expected


def plan_local_file_observation(request: LocalFileObservationRequest) -> dict[str, Any]:
    """Build one local-file observation plan after strict type validation."""

    _require(isinstance(request, LocalFileObservationRequest), "local file observation request is malformed")
    return plan_observation(request)


def plan_repository_snapshot_observation(request: RepositorySnapshotObservationRequest) -> dict[str, Any]:
    """Build one repository-snapshot observation plan after strict type validation."""

    _require(isinstance(request, RepositorySnapshotObservationRequest), "repository snapshot observation request is malformed")
    return plan_observation(request)


def plan_media_observation(request: MediaObservationRequest) -> dict[str, Any]:
    """Build one media observation plan after strict type validation."""

    _require(isinstance(request, MediaObservationRequest), "media observation request is malformed")
    return plan_observation(request)


def plan_browser_observation(request: BrowserObservationRequest) -> dict[str, Any]:
    """Build one browser-capture observation plan after strict type validation."""

    _require(isinstance(request, BrowserObservationRequest), "browser observation request is malformed")
    return plan_observation(request)
