"""Typed, fail-closed contracts for the post-Odyssey product foundation."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import SplitResult, unquote, urlsplit

PRODUCT_SCHEMA_VERSION = "substrate-product-v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")
_DOMAIN_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SOURCE_MODALITIES = frozenset({"audio", "document", "image", "repository", "text", "three-d", "video"})
_RETRIEVAL_MODES = frozenset({"download", "import", "metadata", "stream"})
_PROCESSING_HISTORY_FIELDS = frozenset(
    {"adapter_id", "extractor", "operation_id", "recipe_id", "tool_artifact_sha256", "tool_id", "version"}
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?:api[-_ ]?key|access[-_ ]?token|authorization|credential|cookie|password|secret|token)\s*[:=]"
)
_SENSITIVE_BEARER = re.compile(r"(?i)\b(?:basic|bearer)\s+[a-z0-9._~+/=-]{8,}")


class ProductRefused(ValueError):
    """A product manifest, plan, or receipt failed a required control."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductRefused(message)


def _identifier(value: str, label: str) -> str:
    _require(isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value)), f"{label} must be a lowercase identifier")
    return value


def _string_tuple(values: tuple[str, ...], label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    _require(isinstance(values, tuple), f"{label} must be a tuple")
    _require(all(isinstance(value, str) and value.strip() for value in values), f"{label} must contain nonempty strings")
    _require(len(set(values)) == len(values), f"{label} cannot contain duplicates")
    _require(not nonempty or bool(values), f"{label} cannot be empty")
    return values


def _portable_sha256(value: Any) -> str:
    """Use the product canonical digest without importing ``codec`` at module load.

    ``codec`` imports :class:`ProductRefused` from this module, so importing it
    here would create a module-initialization cycle.  A local import is safe
    once a source request is being serialized.
    """

    from substrate.product.codec import sha256

    return sha256(value)


def _assert_non_sensitive_text(value: str, label: str) -> None:
    """Reject obvious credentials instead of merely hashing them into a receipt."""

    _require(isinstance(value, str), f"{label} must be a string")
    _require(not _SENSITIVE_ASSIGNMENT.search(value), f"{label} cannot contain credentials")
    _require(not _SENSITIVE_BEARER.search(value), f"{label} cannot contain credentials")
    _require("-----BEGIN " not in value.upper(), f"{label} cannot contain credentials")


def _source_origin(value: str) -> dict[str, str]:
    """Return a portable source location summary without a path or full URI."""

    parsed = _parse_source_uri(value)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        # The root/path is an operator-local mount detail.  It must not become
        # portable entity state merely because a source plan was recorded.
        return {"scheme": scheme, "scope": "operator-approved-file-root"}
    return {"authority": _remote_uri_host(parsed), "scheme": scheme}


def _file_uri_path(parsed: SplitResult) -> str:
    """Return one safe, local POSIX path from a structurally valid file URI."""

    _require(not parsed.netloc, "file source URI cannot include an authority")
    _require(parsed.path.startswith("/"), "file source URI must be an absolute path")
    decoded_path = unquote(parsed.path)
    _require(decoded_path.startswith("/"), "file source URI must be an absolute path")
    # A double-leading slash may carry implementation-defined/UNC semantics on
    # some consumers.  A capability broker must never reinterpret it as a
    # hosted path after this contract accepted it as local.
    _require(not decoded_path.startswith("//"), "file source URI must name a local absolute path")
    normalized_path = posixpath.normpath(decoded_path)
    _require(normalized_path.startswith("/") and not normalized_path.startswith("//"), "file source URI is malformed")
    return normalized_path


def _remote_uri_host(parsed: SplitResult) -> str:
    """Return the normalized host for a remote URI or refuse it structurally."""

    _require(bool(parsed.netloc), "remote source URI must include an authority")
    try:
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError as exc:
        raise ProductRefused(f"source URI is malformed: {exc}") from exc
    _require(bool(host), "remote source URI must include a hostname")
    _require(port is None, "remote source URI cannot include an explicit port")
    return host.lower()


def _approved_domain(value: str) -> str:
    """Validate an exact authority allowlist value, not a URL or wildcard."""

    _require(isinstance(value, str) and bool(value), "allowed_domains must contain nonempty strings")
    _require(value == value.lower(), "allowed_domains must be lowercase")
    _require(value == value.strip(), "allowed_domains cannot contain surrounding whitespace")
    _require(
        all(_DOMAIN_LABEL.fullmatch(label) for label in value.split(".")),
        "allowed_domains must contain exact lowercase hostnames",
    )
    return value


def _parse_source_uri(value: str) -> SplitResult:
    _require(isinstance(value, str) and bool(value), "source URI must be a nonempty string")
    _require(value == value.strip(), "source URI cannot have leading or trailing whitespace")
    _require(not any(character.isspace() or ord(character) < 32 for character in value), "source URI cannot contain whitespace or control characters")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ProductRefused(f"source URI is malformed: {exc}") from exc
    _require(bool(_URI_SCHEME.fullmatch(parsed.scheme)), "source URI must include a valid scheme")
    _require(parsed.username is None and parsed.password is None, "source URI cannot include credentials")
    _require(not parsed.query and not parsed.fragment, "source URI cannot include query or fragment data")
    if parsed.scheme.lower() == "file":
        _file_uri_path(parsed)
    else:
        _remote_uri_host(parsed)
    return parsed


@dataclass(frozen=True)
class ResourceBudget:
    """A declared per-worker or host resource vector, measured in MiB."""

    cpu_cores: int
    memory_mib: int
    disk_mib: int

    def __post_init__(self) -> None:
        for label, value in (
            ("cpu_cores", self.cpu_cores),
            ("memory_mib", self.memory_mib),
            ("disk_mib", self.disk_mib),
        ):
            _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{label} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {"cpu_cores": self.cpu_cores, "memory_mib": self.memory_mib, "disk_mib": self.disk_mib}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResourceBudget:
        try:
            _require(isinstance(value, dict) and set(value) == {"cpu_cores", "memory_mib", "disk_mib"}, "resource budget is malformed")
            return cls(**{name: value[name] for name in ("cpu_cores", "memory_mib", "disk_mib")})
        except (KeyError, TypeError) as exc:
            raise ProductRefused("resource budget is malformed") from exc


@dataclass(frozen=True)
class OrganRequirement:
    """A replaceable model or solver interface requirement, never model weights."""

    organ_id: str
    interface_version: str = "substrate-organ/v1"
    modalities: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        _identifier(self.organ_id, "organ_id")
        _require(isinstance(self.interface_version, str) and bool(self.interface_version.strip()), "organ interface version must be nonempty")
        _string_tuple(self.modalities, "organ modalities", nonempty=True)
        _require(all(modality in _SOURCE_MODALITIES for modality in self.modalities), "organ modality is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_version": self.interface_version,
            "modalities": list(self.modalities),
            "organ_id": self.organ_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OrganRequirement:
        try:
            return cls(
                organ_id=value["organ_id"],
                interface_version=value["interface_version"],
                modalities=tuple(value["modalities"]),
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("organ requirement is malformed") from exc


@dataclass(frozen=True)
class EntityManifest:
    """Stable identity and declared scope for one portable specialist."""

    entity_id: str
    specialty: str
    selected_packs: tuple[str, ...]
    organ_requirements: tuple[OrganRequirement, ...] = ()
    schema_version: str = PRODUCT_SCHEMA_VERSION
    _organ_requirements_declared: bool = False

    def __post_init__(self) -> None:
        _require(self.schema_version == PRODUCT_SCHEMA_VERSION, "unsupported entity schema version")
        _identifier(self.entity_id, "entity_id")
        _require(isinstance(self.specialty, str) and bool(self.specialty.strip()), "specialty must be nonempty")
        _string_tuple(self.selected_packs, "selected_packs", nonempty=True)
        for pack in self.selected_packs:
            _identifier(pack, "pack name")
        _require(isinstance(self.organ_requirements, tuple), "organ_requirements must be a tuple")
        _require(all(isinstance(organ, OrganRequirement) for organ in self.organ_requirements), "organ_requirements are malformed")
        _require(len({organ.organ_id for organ in self.organ_requirements}) == len(self.organ_requirements), "organ_requirements cannot contain duplicates")
        _require(isinstance(self._organ_requirements_declared, bool), "organ requirement serialization marker is invalid")

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "entity_id": self.entity_id,
            "schema_version": self.schema_version,
            "selected_packs": list(self.selected_packs),
            "specialty": self.specialty,
        }
        # Preserve the original v1 on-disk representation for entities that do
        # not declare an organ requirement, rather than invalidating their
        # initialization receipt merely by reading and reserializing them.
        if self.organ_requirements or self._organ_requirements_declared:
            document["organ_requirements"] = [organ.to_dict() for organ in self.organ_requirements]
        return document

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EntityManifest:
        try:
            return cls(
                entity_id=value["entity_id"],
                specialty=value["specialty"],
                selected_packs=tuple(value["selected_packs"]),
                organ_requirements=tuple(OrganRequirement.from_dict(organ) for organ in value.get("organ_requirements", ())),
                schema_version=value["schema_version"],
                _organ_requirements_declared="organ_requirements" in value,
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("entity manifest is malformed") from exc


@dataclass(frozen=True)
class SourcePolicy:
    """A source allowlist and media-rights gate; it never grants live access."""

    allowed_schemes: tuple[str, ...]
    allowed_domains: tuple[str, ...] = ()
    allowed_file_roots: tuple[str, ...] = ()
    permitted_access_statuses: tuple[str, ...] = ("licensed", "public", "user-provided")
    allow_download: bool = False

    def __post_init__(self) -> None:
        _string_tuple(self.allowed_schemes, "allowed_schemes", nonempty=True)
        _string_tuple(self.allowed_domains, "allowed_domains")
        _string_tuple(self.allowed_file_roots, "allowed_file_roots")
        _string_tuple(self.permitted_access_statuses, "permitted_access_statuses", nonempty=True)
        _require(
            all(value == value.lower() and bool(_URI_SCHEME.fullmatch(value)) for value in self.allowed_schemes),
            "allowed_schemes must contain lowercase URI schemes",
        )
        for domain in self.allowed_domains:
            _approved_domain(domain)
        for status in self.permitted_access_statuses:
            _identifier(status, "permitted access status")
        _require(isinstance(self.allow_download, bool), "allow_download must be boolean")
        if "file" in self.allowed_schemes:
            _require(bool(self.allowed_file_roots), "file source policy requires at least one approved file root")
        if any(scheme != "file" for scheme in self.allowed_schemes):
            _require(bool(self.allowed_domains), "remote source policy requires at least one approved domain")
        for root in self.allowed_file_roots:
            normalized = posixpath.normpath(root)
            _require(
                root == normalized and normalized.startswith("/") and normalized != "/" and not normalized.startswith("//"),
                "approved file roots must be normalized non-root absolute paths",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_download": self.allow_download,
            "allowed_domains": list(self.allowed_domains),
            "allowed_file_roots": list(self.allowed_file_roots),
            "allowed_schemes": list(self.allowed_schemes),
            "permitted_access_statuses": list(self.permitted_access_statuses),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourcePolicy:
        try:
            _require(isinstance(value, dict), "source policy is malformed")
            _require(
                set(value)
                == {
                    "allow_download",
                    "allowed_domains",
                    "allowed_file_roots",
                    "allowed_schemes",
                    "permitted_access_statuses",
                },
                "source policy is malformed",
            )
            allowed_schemes = value["allowed_schemes"]
            allowed_domains = value["allowed_domains"]
            allowed_file_roots = value["allowed_file_roots"]
            permitted_access_statuses = value["permitted_access_statuses"]
            _require(
                all(isinstance(item, (list, tuple)) for item in (allowed_schemes, allowed_domains, allowed_file_roots, permitted_access_statuses)),
                "source policy is malformed",
            )
            return cls(
                allowed_schemes=tuple(allowed_schemes),
                allowed_domains=tuple(allowed_domains),
                allowed_file_roots=tuple(allowed_file_roots),
                permitted_access_statuses=tuple(permitted_access_statuses),
                allow_download=value["allow_download"],
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("source policy is malformed") from exc

    def assert_permits(self, request: SourceRequest) -> None:
        _require(isinstance(request, SourceRequest), "source request is malformed")
        parsed = _parse_source_uri(request.source_uri)
        scheme = parsed.scheme.lower()
        _require(scheme in self.allowed_schemes, f"source scheme {scheme!r} is not approved")
        if scheme == "file":
            normalized_path = _file_uri_path(parsed)
            approved_roots = tuple(posixpath.normpath(root) for root in self.allowed_file_roots)
            _require(
                any(normalized_path == root or normalized_path.startswith(f"{root}/") for root in approved_roots),
                "file source path is outside approved roots",
            )
        else:
            host = _remote_uri_host(parsed)
            _require(host in self.allowed_domains, f"source domain {host!r} is not approved")
        _require(request.access_status in self.permitted_access_statuses, "source access status is not approved")
        _require(bool(request.declared_rights.strip()), "source request must declare rights or access basis")
        if request.retrieval_mode == "download":
            _require(self.allow_download, "persistent download is not approved by this source policy")


@dataclass(frozen=True)
class SourceRequest:
    """A request to acquire evidence, before any network or tool operation occurs."""

    source_uri: str
    modality: str
    access_status: str
    declared_rights: str
    retrieval_mode: str = "metadata"

    def __post_init__(self) -> None:
        _require(isinstance(self.source_uri, str) and bool(self.source_uri.strip()), "source_uri must be nonempty")
        _parse_source_uri(self.source_uri)
        _require(isinstance(self.modality, str) and self.modality in _SOURCE_MODALITIES, f"unknown source modality {self.modality!r}")
        _identifier(self.access_status, "access_status")
        _require(isinstance(self.declared_rights, str) and bool(self.declared_rights.strip()), "declared_rights must be nonempty")
        _require(len(self.declared_rights) <= 512, "declared_rights is too long")
        _assert_non_sensitive_text(self.declared_rights, "declared_rights")
        _require(isinstance(self.retrieval_mode, str) and self.retrieval_mode in _RETRIEVAL_MODES, f"unknown retrieval mode {self.retrieval_mode!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            "access_status": self.access_status,
            "declared_rights": self.declared_rights,
            "modality": self.modality,
            "retrieval_mode": self.retrieval_mode,
            "source_uri": self.source_uri,
        }

    def to_provenance_dict(self) -> dict[str, str | dict[str, str]]:
        """Return the only source-request representation fit for entity receipts.

        The live request remains available to a future broker in memory, but a
        portable entity ledger receives only policy-relevant labels and
        digests.  This avoids retaining an operator-local path, an opaque URL
        path, or an accidentally sensitive rights note.
        """

        return {
            "access_status": self.access_status,
            "declared_rights_sha256": _portable_sha256({"declared_rights": self.declared_rights}),
            "modality": self.modality,
            "retrieval_mode": self.retrieval_mode,
            "source_origin": _source_origin(self.source_uri),
            "source_request_sha256": _portable_sha256(self.to_dict()),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceRequest:
        try:
            _require(
                isinstance(value, dict)
                and set(value) == {"access_status", "declared_rights", "modality", "retrieval_mode", "source_uri"},
                "source request is malformed",
            )
            return cls(**{name: value[name] for name in ("source_uri", "modality", "access_status", "declared_rights", "retrieval_mode")})
        except (KeyError, TypeError) as exc:
            raise ProductRefused("source request is malformed") from exc


@dataclass(frozen=True)
class SourceReceipt:
    """Provenance for content received through a separately implemented backend."""

    request: SourceRequest
    received_at: str
    retrieval_method: str
    content_sha256: str
    acquisition_plan_sha256: str
    processing_history: tuple[dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require(isinstance(self.request, SourceRequest), "source receipt request is malformed")
        _require(isinstance(self.received_at, str) and bool(self.received_at.strip()), "received_at must be nonempty")
        try:
            parsed_timestamp = datetime.fromisoformat(self.received_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProductRefused("received_at must be an ISO-8601 timestamp") from exc
        _require("T" in self.received_at and parsed_timestamp.tzinfo is not None, "received_at must include time and timezone")
        _identifier(self.retrieval_method, "retrieval_method")
        _require(isinstance(self.content_sha256, str) and bool(_SHA256.fullmatch(self.content_sha256)), "content_sha256 must be a lowercase SHA-256 digest")
        _require(
            isinstance(self.acquisition_plan_sha256, str) and bool(_SHA256.fullmatch(self.acquisition_plan_sha256)),
            "acquisition_plan_sha256 must be a lowercase SHA-256 digest",
        )
        _require(isinstance(self.processing_history, tuple), "processing_history must be a tuple")
        _require(all(isinstance(item, dict) and item for item in self.processing_history), "processing_history entries must be nonempty objects")
        for item in self.processing_history:
            _require(set(item) <= _PROCESSING_HISTORY_FIELDS, "processing_history includes an unsupported field")
            _require(
                all(isinstance(key, str) and isinstance(value, str) for key, value in item.items()),
                "processing_history entries must contain string fields",
            )
            for key, history_value in item.items():
                _assert_non_sensitive_text(history_value, f"processing_history {key}")
                if key == "tool_artifact_sha256":
                    _require(bool(_SHA256.fullmatch(history_value)), "processing_history tool_artifact_sha256 must be a lowercase SHA-256 digest")
                else:
                    _identifier(history_value, f"processing_history {key}")
        object.__setattr__(self, "processing_history", tuple(tuple(sorted(item.items())) for item in self.processing_history))

    def to_dict(self) -> dict[str, Any]:
        return {
            "acquisition_plan_sha256": self.acquisition_plan_sha256,
            "content_sha256": self.content_sha256,
            "processing_history": [dict(item) for item in self.processing_history],
            "received_at": self.received_at,
            "request": self.request.to_dict(),
            "retrieval_method": self.retrieval_method,
        }

    def to_provenance_dict(self) -> dict[str, Any]:
        """Return a ledger-safe source receipt without raw request material."""

        processing_history = [dict(item) for item in self.processing_history]
        return {
            "acquisition_plan_sha256": self.acquisition_plan_sha256,
            "content_sha256": self.content_sha256,
            "processing_history_sha256": _portable_sha256(processing_history),
            "processing_step_count": len(processing_history),
            "received_at": self.received_at,
            "request": self.request.to_provenance_dict(),
            "retrieval_method": self.retrieval_method,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceReceipt:
        try:
            _require(
                isinstance(value, dict)
                and set(value)
                == {
                    "acquisition_plan_sha256",
                    "content_sha256",
                    "processing_history",
                    "received_at",
                    "request",
                    "retrieval_method",
                },
                "source receipt is malformed",
            )
            return cls(
                request=SourceRequest.from_dict(value["request"]),
                received_at=value["received_at"],
                retrieval_method=value["retrieval_method"],
                content_sha256=value["content_sha256"],
                acquisition_plan_sha256=value["acquisition_plan_sha256"],
                processing_history=tuple(value["processing_history"]),
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("source receipt is malformed") from exc


@dataclass(frozen=True)
class ApprenticeshipSpec:
    """A bounded learning objective. Workers may gather evidence; one writer assimilates it."""

    name: str
    objective: str
    evaluators: tuple[str, ...]
    source_policy: SourcePolicy
    worker_budget: ResourceBudget
    maximum_workers: int
    wall_clock_minutes: int

    def __post_init__(self) -> None:
        _identifier(self.name, "apprenticeship name")
        _require(isinstance(self.objective, str) and bool(self.objective.strip()), "apprenticeship objective must be nonempty")
        _string_tuple(self.evaluators, "evaluators", nonempty=True)
        _require(isinstance(self.source_policy, SourcePolicy), "source_policy is malformed")
        _require(isinstance(self.worker_budget, ResourceBudget), "worker_budget is malformed")
        _require(
            isinstance(self.maximum_workers, int) and not isinstance(self.maximum_workers, bool) and self.maximum_workers > 0,
            "maximum_workers must be positive",
        )
        _require(
            isinstance(self.wall_clock_minutes, int) and not isinstance(self.wall_clock_minutes, bool) and self.wall_clock_minutes > 0,
            "wall_clock_minutes must be positive",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluators": list(self.evaluators),
            "maximum_workers": self.maximum_workers,
            "name": self.name,
            "objective": self.objective,
            "source_policy": self.source_policy.to_dict(),
            "wall_clock_minutes": self.wall_clock_minutes,
            "worker_budget": self.worker_budget.to_dict(),
        }
