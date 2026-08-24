"""Declarative capability packs and non-executable sandbox plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from substrate.product.codec import sha256
from substrate.product.contracts import ProductRefused, ResourceBudget, SourcePolicy


@dataclass(frozen=True)
class CapabilityPack:
    """Requirements for one pack; this never represents installed host binaries."""

    name: str
    version: str
    tool_requirements: tuple[str, ...]
    permitted_modalities: tuple[str, ...]
    worker_profile: ResourceBudget
    network_default: str
    filesystem_posture: str
    isolation_expectation: str
    source_policy_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "filesystem_posture": self.filesystem_posture,
            "isolation_expectation": self.isolation_expectation,
            "name": self.name,
            "network_default": self.network_default,
            "permitted_modalities": list(self.permitted_modalities),
            "source_policy_required": self.source_policy_required,
            "tool_requirements": list(self.tool_requirements),
            "version": self.version,
            "worker_profile": self.worker_profile.to_dict(),
        }


BUILTIN_PACKS: dict[str, CapabilityPack] = {
    "engineering": CapabilityPack(
        name="engineering",
        version="1.0.0",
        tool_requirements=("git", "compiler", "test-runner", "version-control-adapter"),
        permitted_modalities=("repository", "text"),
        worker_profile=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=8192),
        network_default="none",
        filesystem_posture="read-only inputs; ephemeral work and quarantined output only",
        isolation_expectation="task-scoped Linux VM or OCI worker; no host git identity, hooks, or credentials",
    ),
    "research": CapabilityPack(
        name="research",
        version="1.0.0",
        tool_requirements=("document-parser", "retrieval-adapter"),
        permitted_modalities=("document", "repository", "text"),
        worker_profile=ResourceBudget(cpu_cores=1, memory_mib=2048, disk_mib=4096),
        network_default="none",
        filesystem_posture="read-only source snapshots; immutable derived artifacts",
        isolation_expectation="task-scoped worker with a brokered, source-specific egress grant only",
        source_policy_required=True,
    ),
    "media": CapabilityPack(
        name="media",
        version="1.0.0",
        tool_requirements=("ffmpeg (optional host/image requirement)", "yt-dlp (optional host/image requirement)"),
        permitted_modalities=("audio", "image", "video"),
        worker_profile=ResourceBudget(cpu_cores=4, memory_mib=8192, disk_mib=16384),
        network_default="none",
        filesystem_posture="read-only approved media; ephemeral derivatives; quarantine before promotion",
        isolation_expectation="short-lived task VM with no browser profile, credentials, DRM bypass, or direct host mounts",
        source_policy_required=True,
    ),
    "mathematics": CapabilityPack(
        name="mathematics",
        version="1.0.0",
        tool_requirements=("lean", "symbolic-algebra", "solver-adapter"),
        permitted_modalities=("document", "text"),
        worker_profile=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=4096),
        network_default="none",
        filesystem_posture="read-only theorem/library snapshots; ephemeral proof workspaces",
        isolation_expectation="task-scoped VM with verified tool/image versions",
    ),
    "three-d": CapabilityPack(
        name="three-d",
        version="1.0.0",
        tool_requirements=("blender", "geometry-library", "simulation-adapter"),
        permitted_modalities=("image", "three-d", "video"),
        worker_profile=ResourceBudget(cpu_cores=4, memory_mib=12288, disk_mib=16384),
        network_default="none",
        filesystem_posture="read-only scene inputs; ephemeral renderer/simulation output",
        isolation_expectation="task-scoped VM with a bounded GPU and output quota when a GPU backend exists",
    ),
    "browser": CapabilityPack(
        name="browser",
        version="1.0.0",
        tool_requirements=("chromium", "browser-automation-adapter"),
        permitted_modalities=("document", "image", "text", "video"),
        worker_profile=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=8192),
        network_default="none",
        filesystem_posture="fresh browser context in a task VM; no host profile, cookies, or DevTools exposure",
        isolation_expectation="browser runs inside the task VM; a browser context is not the outer sandbox boundary",
        source_policy_required=True,
    ),
    # These names complete the initial product-pack set. ``mathematics`` and
    # ``three-d`` remain valid legacy ids so existing portable entities do not
    # need a migration merely to inspect or validate themselves.
    "formal-math": CapabilityPack(
        name="formal-math",
        version="1.0.0",
        tool_requirements=("lean", "symbolic-algebra", "solver-adapter"),
        permitted_modalities=("document", "text"),
        worker_profile=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=4096),
        network_default="none",
        filesystem_posture="read-only theorem/library snapshots; ephemeral proof workspaces",
        isolation_expectation="task-scoped VM with verified tool/image versions",
    ),
    "3d": CapabilityPack(
        name="3d",
        version="1.0.0",
        tool_requirements=("blender", "geometry-library", "simulation-adapter"),
        permitted_modalities=("image", "three-d", "video"),
        worker_profile=ResourceBudget(cpu_cores=4, memory_mib=12288, disk_mib=16384),
        network_default="none",
        filesystem_posture="read-only scene inputs; ephemeral renderer/simulation output",
        isolation_expectation="task-scoped VM with a bounded GPU and output quota when a GPU backend exists",
    ),
    "desktop": CapabilityPack(
        name="desktop",
        version="1.0.0",
        tool_requirements=("desktop-automation-adapter",),
        permitted_modalities=("audio", "image", "text", "video"),
        worker_profile=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=8192),
        network_default="none",
        filesystem_posture="fresh task desktop only; no host desktop, profile, clipboard, or credentials",
        isolation_expectation="short-lived virtual desktop inside the task boundary; no external side effects by default",
        source_policy_required=True,
    ),
    "data-science": CapabilityPack(
        name="data-science",
        version="1.0.0",
        tool_requirements=("python-runtime", "notebook-adapter", "dataframe-library"),
        permitted_modalities=("document", "repository", "text"),
        worker_profile=ResourceBudget(cpu_cores=4, memory_mib=8192, disk_mib=16384),
        network_default="none",
        filesystem_posture="read-only approved datasets; private ephemeral analysis; quarantine all exports",
        isolation_expectation="task-scoped worker with no notebook server, host credentials, or unbounded data egress",
        source_policy_required=True,
    ),
    "robotics": CapabilityPack(
        name="robotics",
        version="1.0.0",
        tool_requirements=("simulator-adapter", "robotics-middleware-adapter", "geometry-library"),
        permitted_modalities=("image", "three-d", "video"),
        worker_profile=ResourceBudget(cpu_cores=4, memory_mib=12288, disk_mib=16384),
        network_default="none",
        filesystem_posture="read-only approved scenes and telemetry; simulation-only private workspaces",
        isolation_expectation="simulation-first worker with no physical actuator or local-network authority",
        source_policy_required=True,
    ),
}


def list_packs() -> list[dict[str, Any]]:
    return [BUILTIN_PACKS[name].to_dict() for name in sorted(BUILTIN_PACKS)]


def resolve_packs(names: tuple[str, ...]) -> tuple[CapabilityPack, ...]:
    if not names:
        raise ProductRefused("at least one capability pack is required")
    if len(set(names)) != len(names):
        raise ProductRefused("capability pack selection cannot contain duplicates")
    unknown = sorted(set(names) - set(BUILTIN_PACKS))
    if unknown:
        raise ProductRefused(f"unknown capability pack(s): {', '.join(unknown)}")
    return tuple(BUILTIN_PACKS[name] for name in names)


def plan_sandbox(
    *,
    entity_id: str,
    selected_packs: tuple[str, ...],
    worker_budget: ResourceBudget,
    source_policy: SourcePolicy,
) -> dict[str, Any]:
    """Return the explicit declaration a future trusted backend must enforce.

    A manifest is not a security boundary.  This function intentionally emits a
    non-executable plan rather than selecting a host runtime or launching one.
    """

    packs = resolve_packs(selected_packs)
    if not isinstance(worker_budget, ResourceBudget):
        raise ProductRefused("worker_budget is malformed")
    if not isinstance(source_policy, SourcePolicy):
        raise ProductRefused("source_policy is malformed")
    required_profile = ResourceBudget(
        cpu_cores=max(pack.worker_profile.cpu_cores for pack in packs),
        memory_mib=max(pack.worker_profile.memory_mib for pack in packs),
        disk_mib=max(pack.worker_profile.disk_mib for pack in packs),
    )
    if (
        worker_budget.cpu_cores < required_profile.cpu_cores
        or worker_budget.memory_mib < required_profile.memory_mib
        or worker_budget.disk_mib < required_profile.disk_mib
    ):
        raise ProductRefused("worker_budget does not meet the selected capability packs' minimum resource profile")
    policy_required = [pack.name for pack in packs if pack.source_policy_required]
    plan = {
        "backend": "unconfigured",
        "entity_id": entity_id,
        "execution_permitted": False,
        "execution_refusal": "no trusted local sandbox backend is configured",
        "mounts": [
            {"destination": "/inputs", "mode": "read-only", "source": "content-addressed-inputs"},
            {"destination": "/work", "mode": "ephemeral-write", "source": "task-workspace"},
            {"destination": "/output", "mode": "quarantine-write", "source": "untrusted-output"},
        ],
        "network_mode": "none",
        "pack_ids": [pack.name for pack in packs],
        "resource_budget": worker_budget.to_dict(),
        "required_pack_profile": required_profile.to_dict(),
        "source_policy": source_policy.to_dict(),
        "source_policy_required_by": policy_required,
        "trusted_host_responsibilities": [
            "validate declared argument vectors",
            "materialize read-only inputs",
            "apply resource and egress grants",
            "hash, scan, and promote outputs explicitly",
        ],
    }
    plan["plan_sha256"] = sha256(plan)
    return plan


def refuse_execution(_: dict[str, Any]) -> None:
    """Prevent callers from mistaking a capability declaration for a sandbox."""

    raise ProductRefused("sandbox execution is unavailable until a separately configured trusted backend exists")
