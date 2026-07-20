
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self

from mop.substrate.events import canonical_sha256

from .perspective_registry import PerspectiveCandidateRegistry
from .substrate_assembly import SubstrateAssembly
from .topology_grammar import GrammarStatus, TopologyGrammar

SUBSTRATE_PREFLIGHT_MANIFEST_SCHEMA = "mop-escs-substrate-preflight-manifest/v1"
SUBSTRATE_PREFLIGHT_REPORT_SCHEMA = "mop-escs-substrate-preflight-report/v1"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_file_receipt_problems(
    root: Path,
    role: str,
    rows: object,
) -> list[str]:

    if not isinstance(rows, list) or not rows:
        return [f"binding-file-receipts-missing:{role}"]
    problems: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"binding-file-receipt-invalid:{role}:{index}")
            continue
        try:
            relative = _relative_path(row.get("path"), f"{role} file receipt path")
            expected_sha256 = _digest(row.get("sha256"), f"{role} file receipt sha256")
        except ValueError:
            problems.append(f"binding-file-receipt-invalid:{role}:{index}")
            continue
        if relative in seen:
            problems.append(f"binding-file-receipt-duplicate:{role}:{relative}")
            continue
        seen.add(relative)
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            problems.append(f"binding-live-file-missing:{role}:{relative}")
            continue
        try:
            observed_sha256 = _sha256_file(target)
            observed_bytes = target.stat().st_size
        except OSError:
            problems.append(f"binding-live-file-unreadable:{role}:{relative}")
            continue
        if observed_sha256 != expected_sha256:
            problems.append(f"binding-live-file-drift:{role}:{relative}")
        expected_bytes = row.get("bytes")
        if expected_bytes is not None and (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or observed_bytes != expected_bytes
        ):
            problems.append(f"binding-live-file-size-drift:{role}:{relative}")
    return problems


def _binding_integrity_problems(
    root: Path,
    binding: PreflightBinding,
    payload: Mapping[str, Any],
) -> list[str]:

    problems: list[str] = []
    if binding.role == "escs-mechanics-proof" or binding.schema == "mop-escs-mechanics-proof/v1":
        core = dict(payload)
        declared = core.pop("proof_sha256", None)
        if declared != canonical_sha256(core):
            problems.append(f"binding-self-hash-mismatch:{binding.role}")
        implementation = payload.get("implementation_receipt")
        if not isinstance(implementation, dict):
            problems.append(f"binding-implementation-receipt-missing:{binding.role}")
        else:
            files = implementation.get("files")
            if implementation.get("manifest_sha256") != canonical_sha256(files):
                problems.append(f"binding-file-receipt-manifest-mismatch:{binding.role}")
            declared_paths = (
                [str(row.get("path")) for row in files if isinstance(row, dict)]
                if isinstance(files, list)
                else []
            )
            expected_sources = [
                root / "scripts/run_escs_mechanics_chassis.py",
                *sorted((root / "src/mop/escs").glob("*.py")),
            ]
            expected_paths = [str(path.resolve().relative_to(root)) for path in expected_sources]
            if declared_paths != expected_paths:
                problems.append(f"binding-file-receipt-coverage-mismatch:{binding.role}")
            problems.extend(_live_file_receipt_problems(root, binding.role, files))
    elif binding.schema.endswith("implementation-authority/v1"):
        core = dict(payload)
        declared = core.pop("manifest_sha256", None)
        if declared != canonical_sha256(core):
            problems.append(f"binding-self-hash-mismatch:{binding.role}")
        files = payload.get("files", payload.get("scoped_files"))
        problems.extend(_live_file_receipt_problems(root, binding.role, files))
    return problems


def _dotted_value(payload: Any, dotted: str) -> Any:
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


@dataclass(frozen=True, slots=True)
class PreflightBinding:
    role: str
    path: str
    sha256: str
    schema: str
    required_fields: tuple[tuple[str, Any], ...]
    activation_authority: bool

    def __post_init__(self) -> None:
        if not self.role or self.role.strip() != self.role:
            raise ValueError("preflight binding role must be canonical nonempty text")
        _relative_path(self.path, "preflight binding path")
        _digest(self.sha256, "preflight binding sha256")
        if not self.schema:
            raise ValueError("preflight binding schema must be nonempty")
        if not self.required_fields:
            raise ValueError("preflight binding requires at least one semantic field")
        keys = tuple(key for key, _value in self.required_fields)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("preflight binding fields must be unique and canonically sorted")
        if self.activation_authority is not False:
            raise ValueError("substrate preflight bindings cannot grant activation authority")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {"role", "path", "sha256", "schema", "required_fields", "activation_authority"},
            "PreflightBinding",
        )
        fields = payload["required_fields"]
        if not isinstance(fields, dict) or not fields:
            raise ValueError("preflight binding required_fields must be a nonempty mapping")
        return cls(
            role=payload["role"],
            path=payload["path"],
            sha256=payload["sha256"],
            schema=payload["schema"],
            required_fields=tuple(sorted(fields.items())),
            activation_authority=payload["activation_authority"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "schema": self.schema,
            "required_fields": dict(self.required_fields),
            "activation_authority": self.activation_authority,
        }


@dataclass(frozen=True, slots=True)
class SubstratePreflightManifest:
    manifest_id: str
    registry_role: str
    assembly_role: str
    topology_role: str
    mechanics_role: str
    bindings: tuple[PreflightBinding, ...]
    activation_enabled: bool
    scientific_promotion_allowed: bool
    manifest_sha256: str
    schema: str = SUBSTRATE_PREFLIGHT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUBSTRATE_PREFLIGHT_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported substrate preflight schema {self.schema!r}")
        if not self.manifest_id:
            raise ValueError("substrate preflight manifest_id must be nonempty")
        roles = tuple(binding.role for binding in self.bindings)
        if not roles or roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError("preflight bindings must be nonempty, unique, and sorted by role")
        required_roles = {
            self.registry_role,
            self.assembly_role,
            self.topology_role,
            self.mechanics_role,
        }
        if len(required_roles) != 4 or not required_roles <= set(roles):
            raise ValueError("preflight semantic roles must be distinct declared bindings")
        if self.activation_enabled is not False:
            raise ValueError("substrate preflight must remain activation-disabled")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("substrate preflight cannot grant scientific promotion")
        _digest(self.manifest_sha256, "substrate preflight manifest sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.manifest_sha256:
            raise ValueError("substrate preflight manifest self-hash mismatch")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "schema",
                "manifest_id",
                "registry_role",
                "assembly_role",
                "topology_role",
                "mechanics_role",
                "bindings",
                "activation_enabled",
                "scientific_promotion_allowed",
                "manifest_sha256",
            },
            "SubstratePreflightManifest",
        )
        rows = payload["bindings"]
        if not isinstance(rows, list):
            raise ValueError("substrate preflight bindings must be a list")
        return cls(
            schema=payload["schema"],
            manifest_id=payload["manifest_id"],
            registry_role=payload["registry_role"],
            assembly_role=payload["assembly_role"],
            topology_role=payload["topology_role"],
            mechanics_role=payload["mechanics_role"],
            bindings=tuple(PreflightBinding.from_payload(row) for row in rows),
            activation_enabled=payload["activation_enabled"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            manifest_sha256=payload["manifest_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "registry_role": self.registry_role,
            "assembly_role": self.assembly_role,
            "topology_role": self.topology_role,
            "mechanics_role": self.mechanics_role,
            "bindings": [binding.payload() for binding in self.bindings],
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def binding(self, role: str) -> PreflightBinding:
        matches = [binding for binding in self.bindings if binding.role == role]
        if len(matches) != 1:
            raise ValueError(f"substrate preflight role {role!r} is not unique")
        return matches[0]


@dataclass(frozen=True, slots=True)
class SubstratePreflightReport:
    manifest_sha256: str
    binding_count: int
    perspective_count: int
    installed_slot_count: int
    exact_binding_count: int
    default_quiescent: bool
    topology_status: str
    topology_implementation_complete: bool
    scaffold_ready: bool
    activation_ready: bool
    scientific_promotion_allowed: bool
    problems: tuple[str, ...]
    report_sha256: str
    schema: str = SUBSTRATE_PREFLIGHT_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUBSTRATE_PREFLIGHT_REPORT_SCHEMA:
            raise ValueError(f"unsupported substrate report schema {self.schema!r}")
        _digest(self.manifest_sha256, "report manifest sha256")
        if (
            min(
                self.binding_count,
                self.perspective_count,
                self.installed_slot_count,
                self.exact_binding_count,
            )
            < 0
        ):
            raise ValueError("substrate preflight counts must be nonnegative")
        if tuple(sorted(set(self.problems))) != self.problems:
            raise ValueError("substrate preflight problems must be unique and sorted")
        if self.scaffold_ready != (not self.problems):
            raise ValueError("substrate scaffold readiness must be exactly problem-derived")
        if self.activation_ready is not False:
            raise ValueError("this preflight cannot declare runtime activation ready")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("this preflight cannot grant scientific promotion")
        _digest(self.report_sha256, "substrate preflight report sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.report_sha256:
            raise ValueError("substrate preflight report self-hash mismatch")

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "manifest_sha256": self.manifest_sha256,
            "binding_count": self.binding_count,
            "perspective_count": self.perspective_count,
            "installed_slot_count": self.installed_slot_count,
            "exact_binding_count": self.exact_binding_count,
            "default_quiescent": self.default_quiescent,
            "topology_status": self.topology_status,
            "topology_implementation_complete": self.topology_implementation_complete,
            "scaffold_ready": self.scaffold_ready,
            "activation_ready": self.activation_ready,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "problems": list(self.problems),
        }
        if include_digest:
            result["report_sha256"] = self.report_sha256
        return result


def load_substrate_preflight_manifest(path: str | Path) -> SubstratePreflightManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("substrate preflight manifest must be a JSON object")
    return SubstratePreflightManifest.from_payload(payload)


def _binding_path(root: Path, binding: PreflightBinding) -> Path:
    path = (root / binding.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing-or-escaping-path")
    return path


def assess_substrate_preflight(
    manifest: SubstratePreflightManifest,
    *,
    repository_root: str | Path,
) -> SubstratePreflightReport:
    root = Path(repository_root).resolve()
    problems: list[str] = []
    exact_count = 0
    loaded: dict[str, dict[str, Any]] = {}
    for binding in manifest.bindings:
        try:
            path = _binding_path(root, binding)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != binding.sha256:
                problems.append(f"binding-digest-mismatch:{binding.role}")
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                problems.append(f"binding-not-object:{binding.role}")
                continue
            if payload.get("schema") != binding.schema:
                problems.append(f"binding-schema-mismatch:{binding.role}")
                continue
            field_mismatches = [
                key for key, expected in binding.required_fields if _dotted_value(payload, key) != expected
            ]
            if field_mismatches:
                problems.extend(f"binding-field-mismatch:{binding.role}:{key}" for key in field_mismatches)
                continue
            integrity_problems = _binding_integrity_problems(root, binding, payload)
            if integrity_problems:
                problems.extend(integrity_problems)
                continue
            loaded[binding.role] = payload
            exact_count += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"binding-unreadable:{binding.role}:{type(exc).__name__}")

    registry = None
    assembly = None
    grammar = None
    try:
        registry = PerspectiveCandidateRegistry.from_payload(loaded[manifest.registry_role])
        assembly = SubstrateAssembly.from_payload(loaded[manifest.assembly_role])
        grammar = TopologyGrammar.from_payload(loaded[manifest.topology_role])
        problems.extend(assembly.validate_registry(registry))
        registry_binding = manifest.binding(manifest.registry_role)
        if grammar.candidate_registry_path != registry_binding.path:
            problems.append("candidate-registry-path-authority-mismatch")
        if grammar.candidate_registry_sha256 != registry.sha256:
            problems.append("candidate-registry-authority-mismatch")
        if any(slot.activation_enabled for slot in assembly.slots):
            problems.append("assembly-slot-activation-enabled")
        if grammar.status is not GrammarStatus.SCAFFOLD:
            problems.append("topology-grammar-not-scaffold")
        if grammar.activation_enabled:
            problems.append("topology-grammar-activation-enabled")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        problems.append(f"semantic-join-failed:{type(exc).__name__}")

    perspective_count = len(registry.candidates) if registry is not None else 0
    installed_count = len(assembly.slots) if assembly is not None else 0
    topology_status = grammar.status.value if grammar is not None else "unavailable"
    topology_complete = (
        grammar.construction_language.implementation_complete if grammar is not None else False
    )
    default_quiescent = bool(assembly is not None and assembly.default_quiescent)
    canonical_problems = tuple(sorted(set(problems)))
    core = {
        "schema": SUBSTRATE_PREFLIGHT_REPORT_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "binding_count": len(manifest.bindings),
        "perspective_count": perspective_count,
        "installed_slot_count": installed_count,
        "exact_binding_count": exact_count,
        "default_quiescent": default_quiescent,
        "topology_status": topology_status,
        "topology_implementation_complete": topology_complete,
        "scaffold_ready": not canonical_problems,
        "activation_ready": False,
        "scientific_promotion_allowed": False,
        "problems": list(canonical_problems),
    }
    return SubstratePreflightReport(
        manifest_sha256=manifest.manifest_sha256,
        binding_count=len(manifest.bindings),
        perspective_count=perspective_count,
        installed_slot_count=installed_count,
        exact_binding_count=exact_count,
        default_quiescent=default_quiescent,
        topology_status=topology_status,
        topology_implementation_complete=topology_complete,
        scaffold_ready=not canonical_problems,
        activation_ready=False,
        scientific_promotion_allowed=False,
        problems=canonical_problems,
        report_sha256=canonical_sha256(core),
    )


def create_substrate_preflight_manifest(
    *,
    manifest_id: str,
    registry_role: str,
    assembly_role: str,
    topology_role: str,
    mechanics_role: str,
    bindings: Sequence[PreflightBinding],
) -> SubstratePreflightManifest:
    rows = tuple(sorted(bindings, key=lambda binding: binding.role))
    core = {
        "schema": SUBSTRATE_PREFLIGHT_MANIFEST_SCHEMA,
        "manifest_id": manifest_id,
        "registry_role": registry_role,
        "assembly_role": assembly_role,
        "topology_role": topology_role,
        "mechanics_role": mechanics_role,
        "bindings": [binding.payload() for binding in rows],
        "activation_enabled": False,
        "scientific_promotion_allowed": False,
    }
    return SubstratePreflightManifest(
        manifest_id=manifest_id,
        registry_role=registry_role,
        assembly_role=assembly_role,
        topology_role=topology_role,
        mechanics_role=mechanics_role,
        bindings=rows,
        activation_enabled=False,
        scientific_promotion_allowed=False,
        manifest_sha256=canonical_sha256(core),
    )


__all__ = [
    "SUBSTRATE_PREFLIGHT_MANIFEST_SCHEMA",
    "SUBSTRATE_PREFLIGHT_REPORT_SCHEMA",
    "PreflightBinding",
    "SubstratePreflightManifest",
    "SubstratePreflightReport",
    "assess_substrate_preflight",
    "create_substrate_preflight_manifest",
    "load_substrate_preflight_manifest",
]
