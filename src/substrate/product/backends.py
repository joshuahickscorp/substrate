"""Non-executing local sandbox backend discovery and dry-run selection.

This module only inspects supplied platform facts and executable paths. It
never launches a VM or container, contacts a daemon, opens a socket, invokes a
binary, installs anything, or mutates the host filesystem.
"""

from __future__ import annotations

import platform
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from substrate.product.codec import sha256
from substrate.product.contracts import PRODUCT_SCHEMA_VERSION, ProductRefused, ResourceBudget
from substrate.product.entity import EntityStore

WhichLookup = Callable[[str], str | None]

_APPLE_SILICON_MACHINES = frozenset({"arm64", "aarch64"})
_BACKEND_IDS = frozenset({"apple-container", "docker"})
_CANDIDATE_SELECTION_STATUSES = frozenset({"candidate", "eligible-not-selected", "rejected", "selected"})
_SANDBOX_MOUNT_SOURCES = frozenset({"content-addressed-inputs", "task-workspace", "untrusted-output"})
_SANDBOX_REQUIRED_KEYS = frozenset(
    {
        "backend",
        "entity_id",
        "execution_permitted",
        "mounts",
        "network_mode",
        "plan_sha256",
        "resource_budget",
    }
)


def _require_string_tuple(values: tuple[str, ...], label: str, *, nonempty: bool = False) -> None:
    if not isinstance(values, tuple):
        raise ProductRefused(f"{label} must be a tuple")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ProductRefused(f"{label} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ProductRefused(f"{label} cannot contain duplicates")
    if nonempty and not values:
        raise ProductRefused(f"{label} cannot be empty")


@dataclass(frozen=True)
class PlatformFacts:
    """Host facts used for eligibility. Callers may inject fixtures for tests."""

    system: str
    machine: str

    def __post_init__(self) -> None:
        if not isinstance(self.system, str) or not self.system.strip():
            raise ProductRefused("platform system must be a nonempty string")
        if not isinstance(self.machine, str) or not self.machine.strip():
            raise ProductRefused("platform machine must be a nonempty string")

    @classmethod
    def discover(cls) -> PlatformFacts:
        """Read the local platform identity without probing daemons or sockets."""

        return cls(system=platform.system(), machine=platform.machine())

    @property
    def is_darwin_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine.lower() in _APPLE_SILICON_MACHINES

    def to_dict(self) -> dict[str, str]:
        return {"machine": self.machine, "system": self.system}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlatformFacts:
        if not isinstance(value, Mapping):
            raise ProductRefused("platform facts are malformed")
        try:
            return cls(system=value["system"], machine=value["machine"])
        except (KeyError, TypeError) as exc:
            raise ProductRefused("platform facts are malformed") from exc


@dataclass(frozen=True)
class BackendCandidate:
    """One discovered-or-rejected backend with an explicit safety posture."""

    backend_id: str
    executable_name: str
    executable_path: str | None
    platform_eligible: bool
    executable_discovered: bool
    eligible: bool
    priority: int
    role: str
    safety_posture: str
    capability_claims: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    daemon_status: str = "not-probed"
    image_availability: str = "not-probed"
    selection_status: str = "candidate"

    def __post_init__(self) -> None:
        if self.backend_id not in _BACKEND_IDS:
            raise ProductRefused(f"unknown backend id {self.backend_id!r}")
        if not isinstance(self.executable_name, str) or not self.executable_name.strip():
            raise ProductRefused("backend executable name must be nonempty")
        if self.executable_path is not None and (
            not isinstance(self.executable_path, str) or not self.executable_path.strip()
        ):
            raise ProductRefused("backend executable path must be nonempty when present")
        for label, value in (
            ("platform_eligible", self.platform_eligible),
            ("executable_discovered", self.executable_discovered),
            ("eligible", self.eligible),
        ):
            if not isinstance(value, bool):
                raise ProductRefused(f"{label} must be boolean")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 1:
            raise ProductRefused("backend priority must be a positive integer")
        expected_priority = 1 if self.backend_id == "apple-container" else 2
        if self.priority != expected_priority:
            raise ProductRefused("backend priority must match the fixed backend selection order")
        for label, text_value in (("backend role", self.role), ("backend safety posture", self.safety_posture)):
            if not isinstance(text_value, str) or not text_value.strip():
                raise ProductRefused(f"{label} must be nonempty")
        _require_string_tuple(self.capability_claims, "backend capability claims", nonempty=True)
        _require_string_tuple(self.rejection_reasons, "backend rejection reasons")
        if self.eligible and self.rejection_reasons:
            raise ProductRefused("eligible backends cannot carry rejection reasons")
        if not self.eligible and not self.rejection_reasons:
            raise ProductRefused("ineligible backends must explain their rejection")
        if self.daemon_status != "not-probed":
            raise ProductRefused("backend probes must not claim daemon health")
        if self.image_availability != "not-probed":
            raise ProductRefused("backend probes must not claim image availability")
        if self.selection_status not in _CANDIDATE_SELECTION_STATUSES:
            raise ProductRefused("backend candidate selection status is invalid")
        if self.eligible != (self.platform_eligible and self.executable_discovered):
            raise ProductRefused("backend eligibility must match platform and executable discovery")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "capability_claims": list(self.capability_claims),
            "daemon_status": self.daemon_status,
            "eligible": self.eligible,
            "executable_discovered": self.executable_discovered,
            "executable_name": self.executable_name,
            "executable_path": self.executable_path,
            "image_availability": self.image_availability,
            "platform_eligible": self.platform_eligible,
            "priority": self.priority,
            "rejection_reasons": list(self.rejection_reasons),
            "role": self.role,
            "safety_posture": self.safety_posture,
            "selection_status": self.selection_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BackendCandidate:
        if not isinstance(value, Mapping):
            raise ProductRefused("backend candidate is malformed")
        try:
            return cls(
                backend_id=value["backend_id"],
                executable_name=value["executable_name"],
                executable_path=value.get("executable_path"),
                platform_eligible=value["platform_eligible"],
                executable_discovered=value["executable_discovered"],
                eligible=value["eligible"],
                priority=value["priority"],
                role=value["role"],
                safety_posture=value["safety_posture"],
                capability_claims=tuple(value.get("capability_claims", ())),
                rejection_reasons=tuple(value.get("rejection_reasons", ())),
                daemon_status=value.get("daemon_status", "not-probed"),
                image_availability=value.get("image_availability", "not-probed"),
                selection_status=value.get("selection_status", "candidate"),
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("backend candidate is malformed") from exc


def select_preferred_backend(candidates: tuple[BackendCandidate, ...]) -> tuple[str | None, str]:
    """Prefer eligible Apple Container, then Docker; otherwise report absence."""

    if not isinstance(candidates, tuple) or not candidates:
        raise ProductRefused("backend selection requires probed candidates")
    if not all(isinstance(candidate, BackendCandidate) for candidate in candidates):
        raise ProductRefused("backend selection candidates are malformed")
    ordered = sorted(candidates, key=lambda candidate: candidate.priority)
    for candidate in ordered:
        if candidate.eligible:
            return candidate.backend_id, "recommended"
    return None, "unavailable"


@dataclass(frozen=True)
class BackendProbeResult:
    """Deterministic probe outcome and preferred dry-run selection."""

    platform: PlatformFacts
    candidates: tuple[BackendCandidate, ...]
    preferred_backend: str | None
    selection_status: str
    probe_method: str = "executable-path-discovery-only"
    selection_is_authorization: bool = False
    selection_is_dry_run: bool = True
    execution_permitted: bool = False
    execution_refusal: str = (
        "backend selection is a dry-run recommendation only; no local sandbox backend is authorized to execute"
    )
    schema_version: str = PRODUCT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.platform, PlatformFacts):
            raise ProductRefused("backend probe platform facts are malformed")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ProductRefused("backend probe must include at least one candidate")
        if not all(isinstance(candidate, BackendCandidate) for candidate in self.candidates):
            raise ProductRefused("backend probe candidates are malformed")
        backend_ids = [candidate.backend_id for candidate in self.candidates]
        if len(set(backend_ids)) != len(backend_ids):
            raise ProductRefused("backend probe cannot contain duplicate candidates")
        if set(backend_ids) != _BACKEND_IDS:
            raise ProductRefused("backend probe must cover every supported backend candidate")
        expected_backend, expected_status = select_preferred_backend(self.candidates)
        if (self.preferred_backend, self.selection_status) != (expected_backend, expected_status):
            raise ProductRefused("backend probe selection does not match candidate priority and eligibility")
        for candidate in self.candidates:
            expected_candidate_status = (
                "selected"
                if candidate.backend_id == self.preferred_backend
                else "eligible-not-selected"
                if candidate.eligible
                else "rejected"
            )
            if candidate.selection_status != expected_candidate_status:
                raise ProductRefused("backend candidate selection status does not match probe selection")
        if self.selection_is_authorization is not False:
            raise ProductRefused("backend selection must not authorize execution")
        if self.selection_is_dry_run is not True:
            raise ProductRefused("backend selection must remain a dry-run")
        if self.execution_permitted is not False:
            raise ProductRefused("backend probe cannot permit execution")
        if self.probe_method != "executable-path-discovery-only":
            raise ProductRefused("unsupported backend probe method")
        if not isinstance(self.execution_refusal, str) or not self.execution_refusal.strip():
            raise ProductRefused("backend probe execution refusal must be nonempty")
        if self.schema_version != PRODUCT_SCHEMA_VERSION:
            raise ProductRefused("unsupported backend probe schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "execution_permitted": self.execution_permitted,
            "execution_refusal": self.execution_refusal,
            "platform": self.platform.to_dict(),
            "preferred_backend": self.preferred_backend,
            "probe_method": self.probe_method,
            "schema_version": self.schema_version,
            "selection_is_authorization": self.selection_is_authorization,
            "selection_is_dry_run": self.selection_is_dry_run,
            "selection_status": self.selection_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BackendProbeResult:
        if not isinstance(value, Mapping):
            raise ProductRefused("backend probe result is malformed")
        try:
            return cls(
                platform=PlatformFacts.from_dict(value["platform"]),
                candidates=tuple(BackendCandidate.from_dict(item) for item in value["candidates"]),
                preferred_backend=value["preferred_backend"],
                selection_status=value["selection_status"],
                probe_method=value.get("probe_method", "executable-path-discovery-only"),
                selection_is_authorization=value.get("selection_is_authorization", False),
                selection_is_dry_run=value.get("selection_is_dry_run", True),
                execution_permitted=value.get("execution_permitted", False),
                execution_refusal=value.get(
                    "execution_refusal",
                    "backend selection is a dry-run recommendation only; no local sandbox backend is authorized to execute",
                ),
                schema_version=value.get("schema_version", PRODUCT_SCHEMA_VERSION),
            )
        except (KeyError, TypeError) as exc:
            raise ProductRefused("backend probe result is malformed") from exc


def _discover_executable(name: str, which: WhichLookup) -> str | None:
    try:
        path = which(name)
    except Exception as exc:
        raise ProductRefused(f"executable lookup failed for {name!r}") from exc
    if path is None:
        return None
    if not isinstance(path, str) or not path.strip():
        raise ProductRefused(f"executable lookup for {name!r} returned an invalid path")
    return path


def _apple_container_candidate(facts: PlatformFacts, which: WhichLookup) -> BackendCandidate:
    path = _discover_executable("container", which)
    platform_eligible = facts.is_darwin_apple_silicon
    discovered = path is not None
    reasons: list[str] = []
    if not platform_eligible:
        reasons.append("apple-container requires Darwin on Apple Silicon")
    if not discovered:
        reasons.append("container executable was not discovered on PATH")
    return BackendCandidate(
        backend_id="apple-container",
        executable_name="container",
        executable_path=path,
        platform_eligible=platform_eligible,
        executable_discovered=discovered,
        eligible=platform_eligible and discovered,
        priority=1,
        role="preferred-local-sandbox",
        safety_posture=(
            "Apple Container is intended as an OCI runtime that starts each container in a lightweight Linux VM; "
            "this probe only discovered an executable path and did not verify daemon readiness, image availability, "
            "VM configuration, or permission to run"
        ),
        capability_claims=(
            "executable-path-discovery-only",
            "no-daemon-health-claim",
            "no-image-availability-claim",
            "no-execution-authorization",
        ),
        rejection_reasons=tuple(reasons),
    )


def _docker_candidate(facts: PlatformFacts, which: WhichLookup) -> BackendCandidate:
    del facts  # Docker eligibility is executable-path based; platform is informational only.
    path = _discover_executable("docker", which)
    discovered = path is not None
    return BackendCandidate(
        backend_id="docker",
        executable_name="docker",
        executable_path=path,
        platform_eligible=True,
        executable_discovered=discovered,
        eligible=discovered,
        priority=2,
        role="compatibility-fallback",
        safety_posture=(
            "Docker is a compatibility fallback only. Discovering a docker executable does not prove the daemon is "
            "healthy, that images are available, or that host mounts and shared paths are configured safely; treat "
            "any future adapter as high scrutiny"
        ),
        capability_claims=(
            "executable-path-discovery-only",
            "no-daemon-health-claim",
            "no-image-availability-claim",
            "no-execution-authorization",
        ),
        rejection_reasons=() if discovered else ("docker executable was not discovered on PATH",),
    )


def probe_backends(
    *,
    platform_facts: PlatformFacts | None = None,
    which: WhichLookup | None = None,
) -> BackendProbeResult:
    """Probe local backend eligibility without executing any discovered binary."""

    facts = platform_facts if platform_facts is not None else PlatformFacts.discover()
    if not isinstance(facts, PlatformFacts):
        raise ProductRefused("platform facts are malformed")
    lookup = which if which is not None else shutil.which
    if not callable(lookup):
        raise ProductRefused("executable lookup must be callable")
    candidates = (
        _apple_container_candidate(facts, lookup),
        _docker_candidate(facts, lookup),
    )
    preferred, selection_status = select_preferred_backend(candidates)
    annotated = tuple(
        BackendCandidate(
            backend_id=candidate.backend_id,
            executable_name=candidate.executable_name,
            executable_path=candidate.executable_path,
            platform_eligible=candidate.platform_eligible,
            executable_discovered=candidate.executable_discovered,
            eligible=candidate.eligible,
            priority=candidate.priority,
            role=candidate.role,
            safety_posture=candidate.safety_posture,
            capability_claims=candidate.capability_claims,
            rejection_reasons=candidate.rejection_reasons,
            selection_status=(
                "selected"
                if candidate.backend_id == preferred
                else "eligible-not-selected"
                if candidate.eligible
                else "rejected"
            ),
        )
        for candidate in candidates
    )
    return BackendProbeResult(
        platform=facts,
        candidates=annotated,
        preferred_backend=preferred,
        selection_status=selection_status,
    )


def _require_sandbox_plan(sandbox_plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(sandbox_plan, Mapping):
        raise ProductRefused("sandbox plan is malformed")
    missing = sorted(_SANDBOX_REQUIRED_KEYS - set(sandbox_plan))
    if missing:
        raise ProductRefused(f"sandbox plan is missing required field(s): {', '.join(missing)}")
    plan = dict(sandbox_plan)
    if plan.get("execution_permitted") is True:
        raise ProductRefused("execution-enabled sandbox plan is refused by the dry-run backend layer")
    if plan.get("execution_permitted") is not False:
        raise ProductRefused("sandbox plan execution_permitted must be false")
    if plan.get("backend") != "unconfigured":
        raise ProductRefused("sandbox plan backend must remain unconfigured for a dry-run binding")
    recorded_digest = plan.get("plan_sha256")
    if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
        raise ProductRefused("sandbox plan digest is malformed")
    if recorded_digest != sha256({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ProductRefused("sandbox plan digest does not match its declaration")
    if not isinstance(plan.get("entity_id"), str) or not plan["entity_id"].strip():
        raise ProductRefused("sandbox plan entity_id is malformed")
    if plan.get("network_mode") != "none":
        raise ProductRefused("sandbox plan network_mode must remain none for dry-run binding")
    mounts = plan.get("mounts")
    if not isinstance(mounts, list) or not mounts:
        raise ProductRefused("sandbox plan mounts are malformed")
    for mount in mounts:
        if not isinstance(mount, Mapping):
            raise ProductRefused("sandbox plan mounts are malformed")
        mode = mount.get("mode")
        source = mount.get("source")
        if mode not in {"read-only", "ephemeral-write", "quarantine-write"}:
            raise ProductRefused("sandbox plan includes an unsupported mount mode")
        if source not in _SANDBOX_MOUNT_SOURCES:
            raise ProductRefused("sandbox plan must use an approved non-host mount source")
    try:
        ResourceBudget.from_dict(plan["resource_budget"])
    except ProductRefused as exc:
        raise ProductRefused("sandbox plan resource budget is malformed") from exc
    return plan


def plan_backend_dry_run(
    sandbox_plan: Mapping[str, Any],
    *,
    platform_facts: PlatformFacts | None = None,
    which: WhichLookup | None = None,
    probe: BackendProbeResult | None = None,
) -> dict[str, Any]:
    """Bind a non-executing backend recommendation to an existing sandbox plan."""

    plan = _require_sandbox_plan(sandbox_plan)
    if probe is None:
        probe_result = probe_backends(platform_facts=platform_facts, which=which)
    else:
        if not isinstance(probe, BackendProbeResult):
            raise ProductRefused("backend probe result is malformed")
        if probe.execution_permitted is not False or probe.selection_is_authorization is not False:
            raise ProductRefused("execution-enabled backend probe is refused")
        if probe.selection_is_dry_run is not True:
            raise ProductRefused("non-dry-run backend probe is refused")
        expected_backend, expected_status = select_preferred_backend(probe.candidates)
        if (probe.preferred_backend, probe.selection_status) != (expected_backend, expected_status):
            raise ProductRefused("backend probe selection is inconsistent")
        probe_result = probe
    selected = None
    rejected: list[dict[str, Any]] = []
    for candidate in probe_result.candidates:
        document = candidate.to_dict()
        if candidate.backend_id == probe_result.preferred_backend and candidate.eligible:
            selected = document
        else:
            rejected.append(document)
    dry_run: dict[str, Any] = {
        "backend_probe": probe_result.to_dict(),
        "entity_id": plan["entity_id"],
        "execution_permitted": False,
        "execution_refusal": (
            "backend dry-run plans never authorize execution; a separately reviewed trusted backend and operator grant "
            "are still required"
        ),
        "filesystem_posture": "non-host mounts only; read-only inputs, ephemeral work, quarantined output",
        "kind": "backend-dry-run-plan",
        "mounts": list(plan["mounts"]),
        "network_mode": "none",
        "preferred_backend": probe_result.preferred_backend,
        "rejected_candidates": rejected,
        "resource_budget": dict(plan["resource_budget"]),
        "sandbox_backend_declaration": plan["backend"],
        "sandbox_plan_sha256": plan["plan_sha256"],
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "selected_candidate": selected,
        "selection_is_authorization": False,
        "selection_is_dry_run": True,
        "selection_status": probe_result.selection_status,
    }
    dry_run["plan_sha256"] = sha256(dry_run)
    return dry_run


def plan_backend_dry_run_for_entity(
    store: EntityStore,
    *,
    platform_facts: PlatformFacts | None = None,
    which: WhichLookup | None = None,
    probe: BackendProbeResult | None = None,
) -> dict[str, Any]:
    """Emit a dry-run backend plan for an entity's active apprenticeship sandbox plan."""

    if not isinstance(store, EntityStore):
        raise ProductRefused("entity store is malformed")
    snapshot = store.load()
    if snapshot.state["phase"] != "apprenticeship_planned" or not snapshot.state["active_apprenticeship"]:
        raise ProductRefused("operation requires an active apprenticeship plan")
    active_plan_sha256 = snapshot.state["active_apprenticeship_plan_sha256"]
    matching_plan = next(
        (
            row["payload"]["plan"]
            for row in snapshot.receipts
            if row["kind"] == "apprenticeship_planned"
            and isinstance(row["payload"].get("plan"), dict)
            and row["payload"]["plan"].get("plan_sha256") == active_plan_sha256
        ),
        None,
    )
    if matching_plan is None:
        raise ProductRefused("active apprenticeship plan is not present in the verified receipt ledger")
    if active_plan_sha256 != sha256({key: value for key, value in matching_plan.items() if key != "plan_sha256"}):
        raise ProductRefused("active apprenticeship plan digest does not match its declaration")
    sandbox = matching_plan.get("sandbox")
    if not isinstance(sandbox, Mapping):
        raise ProductRefused("active apprenticeship plan is missing a sandbox declaration")
    dry_run = plan_backend_dry_run(
        sandbox,
        platform_facts=platform_facts,
        which=which,
        probe=probe,
    )
    bound_plan = {
        **{key: value for key, value in dry_run.items() if key != "plan_sha256"},
        "active_apprenticeship": snapshot.state["active_apprenticeship"],
        "active_apprenticeship_plan_sha256": active_plan_sha256,
        "entity_revision_sha256": snapshot.revision_sha256,
    }
    bound_plan["plan_sha256"] = sha256(bound_plan)
    return bound_plan
