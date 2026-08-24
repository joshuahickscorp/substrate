"""Bounded apprenticeship planning and single-writer evidence assimilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from substrate.product.cache import ArtifactStore, LocalCacheVerifierTrustStore
from substrate.product.codec import sha256
from substrate.product.contracts import ApprenticeshipSpec, ProductRefused, ResourceBudget, SourcePolicy, SourceReceipt, SourceRequest
from substrate.product.entity import EntityStore
from substrate.product.packs import plan_sandbox, resolve_packs
from substrate.product.sources import plan_acquisition, validate_acquisition_plan, validate_receipt


@dataclass(frozen=True)
class CacheAttestedEvidenceAuthority:
    """The sole v1 bridge from a verified local cache object to entity evidence.

    This is deliberately a small typed authority, not a callable supplied by a
    worker.  It delegates to the cache's existing signed-attestation and
    revalidation path; it does not ingest content, fetch sources, invoke a
    tool, or grant a worker any new authority.
    """

    artifact_store: ArtifactStore
    verifier_trust_store: LocalCacheVerifierTrustStore

    def __post_init__(self) -> None:
        # Exact concrete types avoid accepting a worker-made look-alike with a
        # ``trusted_source_receipt_verifier`` method.  The local cache and its
        # explicit trust root remain the trusted computing base for this v1
        # integration.
        if type(self.artifact_store) is not ArtifactStore:
            raise ProductRefused("cache evidence authority requires an ArtifactStore")
        if type(self.verifier_trust_store) is not LocalCacheVerifierTrustStore:
            raise ProductRefused("cache evidence authority requires a local cache verifier trust store")

    def verifies(self, receipt: SourceReceipt, source_plan: dict[str, Any]) -> bool:
        """Delegate read-only receipt verification to the already trusted cache."""

        return self.artifact_store.trusted_source_receipt_verifier(
            receipt,
            source_plan,
            self.verifier_trust_store,
        )


@dataclass(frozen=True)
class HostResources:
    """An operator-supplied host snapshot. The scheduler does not guess at capacity."""

    cpu_cores: int
    memory_mib: int
    disk_mib: int
    control_plane_reserve: ResourceBudget = ResourceBudget(cpu_cores=1, memory_mib=1024, disk_mib=1024)
    assimilation_reserve: ResourceBudget = ResourceBudget(cpu_cores=1, memory_mib=1024, disk_mib=1024)

    def __post_init__(self) -> None:
        ResourceBudget(self.cpu_cores, self.memory_mib, self.disk_mib)
        if not isinstance(self.control_plane_reserve, ResourceBudget):
            raise ProductRefused("control_plane_reserve is malformed")
        if not isinstance(self.assimilation_reserve, ResourceBudget):
            raise ProductRefused("assimilation_reserve is malformed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assimilation_reserve": self.assimilation_reserve.to_dict(),
            "control_plane_reserve": self.control_plane_reserve.to_dict(),
            "cpu_cores": self.cpu_cores,
            "memory_mib": self.memory_mib,
            "disk_mib": self.disk_mib,
        }


@dataclass(frozen=True)
class WorkerPlan:
    """Resource admission for independent workers and exactly one assimilator."""

    concurrent_workers: int
    capacity_by_resource: dict[str, int]
    available_for_workers: ResourceBudget
    maximum_workers: int
    host: HostResources
    per_worker: ResourceBudget
    reserved_for_assimilation: ResourceBudget
    reserved_for_control_plane: ResourceBudget
    authoritative_assimilation_writers: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "authoritative_assimilation_writers": self.authoritative_assimilation_writers,
            "available_for_workers": self.available_for_workers.to_dict(),
            "capacity_by_resource": dict(self.capacity_by_resource),
            "concurrent_workers": self.concurrent_workers,
            "host": self.host.to_dict(),
            "maximum_workers": self.maximum_workers,
            "per_worker": self.per_worker.to_dict(),
            "reserved_for_assimilation": self.reserved_for_assimilation.to_dict(),
            "reserved_for_control_plane": self.reserved_for_control_plane.to_dict(),
            "worker_identity_claim": "workers gather evidence; they are not a shared continuous mind",
        }


def _active_plan_and_policy(store: EntityStore) -> tuple[Any, dict[str, Any], SourcePolicy]:
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
    specification = matching_plan.get("specification")
    if not isinstance(specification, dict):
        raise ProductRefused("active apprenticeship plan has an invalid specification")
    source_policy = SourcePolicy.from_dict(specification.get("source_policy", {}))
    if snapshot.state["active_source_policy_sha256"] != sha256(source_policy.to_dict()):
        raise ProductRefused("active apprenticeship policy hash does not match its recorded plan")
    return snapshot, matching_plan, source_policy


def _assert_permitted_modality(snapshot: Any, receipt_or_request: SourceReceipt | Any) -> None:
    modality = receipt_or_request.request.modality if isinstance(receipt_or_request, SourceReceipt) else receipt_or_request.modality
    permitted_modalities = {supported for pack in resolve_packs(snapshot.manifest.selected_packs) for supported in pack.permitted_modalities}
    if modality not in permitted_modalities:
        raise ProductRefused("source modality is not permitted by the entity's selected capability packs")


def plan_workers(host: HostResources, per_worker: ResourceBudget, maximum_workers: int) -> WorkerPlan:
    """Calculate the strict resource-vector bound for a proposed worker pool."""

    if not isinstance(maximum_workers, int) or isinstance(maximum_workers, bool) or maximum_workers <= 0:
        raise ProductRefused("maximum_workers must be a positive integer")
    if not isinstance(host, HostResources) or not isinstance(per_worker, ResourceBudget):
        raise ProductRefused("host resources and worker budget must be declared resource vectors")
    available_values = {
        "cpu_cores": host.cpu_cores - host.control_plane_reserve.cpu_cores - host.assimilation_reserve.cpu_cores,
        "memory_mib": host.memory_mib - host.control_plane_reserve.memory_mib - host.assimilation_reserve.memory_mib,
        "disk_mib": host.disk_mib - host.control_plane_reserve.disk_mib - host.assimilation_reserve.disk_mib,
    }
    if any(value <= 0 for value in available_values.values()):
        raise ProductRefused("host resources cannot admit a worker after control-plane and assimilation reserves")
    available_for_workers = ResourceBudget(
        **available_values,
    )
    capacity_by_resource = {
        "cpu": available_for_workers.cpu_cores // per_worker.cpu_cores,
        "disk": available_for_workers.disk_mib // per_worker.disk_mib,
        "memory": available_for_workers.memory_mib // per_worker.memory_mib,
    }
    concurrent_workers = min(maximum_workers, *capacity_by_resource.values())
    if concurrent_workers < 1:
        raise ProductRefused("host resources cannot admit one worker with the declared budget")
    return WorkerPlan(
        concurrent_workers=concurrent_workers,
        capacity_by_resource=capacity_by_resource,
        available_for_workers=available_for_workers,
        maximum_workers=maximum_workers,
        host=host,
        per_worker=per_worker,
        reserved_for_assimilation=host.assimilation_reserve,
        reserved_for_control_plane=host.control_plane_reserve,
    )


def plan_apprenticeship(store: EntityStore, spec: ApprenticeshipSpec, host: HostResources) -> dict[str, Any]:
    """Record an apprenticeship plan; this does not create or run workers."""

    if not isinstance(store, EntityStore) or not isinstance(spec, ApprenticeshipSpec) or not isinstance(host, HostResources):
        raise ProductRefused("apprenticeship inputs are malformed")
    snapshot = store.load()
    worker_plan = plan_workers(host, spec.worker_budget, spec.maximum_workers)
    sandbox = plan_sandbox(
        entity_id=snapshot.manifest.entity_id,
        selected_packs=snapshot.manifest.selected_packs,
        worker_budget=spec.worker_budget,
        source_policy=spec.source_policy,
    )
    plan = {
        "assimilator": {
            "authority": "one entity-local receipt writer",
            "mode": "controlled assimilation only",
            "workers_write_entity_state": False,
        },
        "entity_id": snapshot.manifest.entity_id,
        "kind": "autonomous-apprenticeship-plan",
        "sandbox": sandbox,
        "specification": spec.to_dict(),
        "worker_plan": worker_plan.to_dict(),
    }
    plan["plan_sha256"] = sha256(plan)
    receipt = store.record(
        "apprenticeship_planned",
        {"plan": plan},
        state_update={
            "active_apprenticeship": spec.name,
            "active_apprenticeship_plan_sha256": plan["plan_sha256"],
            "active_source_policy_sha256": sha256(spec.source_policy.to_dict()),
            "phase": "apprenticeship_planned",
        },
        expected_checkpoint_sha256=snapshot.revision_sha256,
    )
    return {"plan": plan, "receipt": receipt}


def plan_source_acquisition(store: EntityStore, request: SourceRequest) -> dict[str, Any]:
    """Bind one approved source request to the active apprenticeship ledger."""

    if not isinstance(request, SourceRequest):
        raise ProductRefused("source acquisition request is malformed")
    snapshot, _, source_policy = _active_plan_and_policy(store)
    _assert_permitted_modality(snapshot, request)
    plan = plan_acquisition(request, source_policy)
    receipt = store.record(
        "source_acquisition_planned",
        {"plan": plan},
        expected_checkpoint_sha256=snapshot.revision_sha256,
    )
    return {"plan": plan, "receipt": receipt}


def assimilate_source_receipt(
    store: EntityStore,
    receipt: SourceReceipt,
    *,
    verifier: CacheAttestedEvidenceAuthority | None = None,
) -> dict[str, Any]:
    """Promote only cache-attested evidence into the authoritative ledger.

    A bare callback is not an authority: v1 accepts only
    :class:`CacheAttestedEvidenceAuthority`, which delegates to the existing
    local signed-cache-attestation verifier.  This function neither fetches nor
    executes a source/tool; it only binds previously verified cache evidence to
    the currently active plan and records a sanitized receipt.
    """

    snapshot, _, source_policy = _active_plan_and_policy(store)
    _assert_permitted_modality(snapshot, receipt)
    source_plan = next(
        (
            row["payload"]["plan"]
            for row in snapshot.receipts
            if row["kind"] == "source_acquisition_planned"
            and isinstance(row["payload"].get("plan"), dict)
            and row["payload"]["plan"].get("plan_sha256") == receipt.acquisition_plan_sha256
        ),
        None,
    )
    if source_plan is None:
        raise ProductRefused("source receipt is not bound to an approved acquisition plan")
    sealed_source_plan = validate_acquisition_plan(source_plan, source_policy, request=receipt.request)
    # Bind the recorded acquisition plan to the active policy before applying
    # policy semantics to the receipt.  This makes a replaced policy fail as a
    # plan-policy mismatch rather than accidentally looking like a malformed
    # new source request.
    receipt_envelope = validate_receipt(receipt, source_policy)
    if type(verifier) is not CacheAttestedEvidenceAuthority:
        raise ProductRefused("evidence cannot be assimilated without a cache-attested evidence authority")
    try:
        trusted = verifier.verifies(receipt, sealed_source_plan)
    except ProductRefused as exc:
        raise ProductRefused("cache-attested evidence authority failed") from exc
    if trusted is not True:
        raise ProductRefused("cache-attested evidence authority did not attest the source receipt")
    assimilation = {
        "artifact_cache_id": verifier.artifact_store.cache_id,
        "content_sha256": receipt.content_sha256,
        "receipt_sha256": receipt_envelope["receipt_sha256"],
        "source_receipt": receipt_envelope["receipt"],
        "status": "accepted-after-cache-attested-verification",
    }
    developmental_receipt = store.record(
        "source_evidence_assimilated",
        assimilation,
        state_update={"evidence_assimilated": snapshot.state["evidence_assimilated"] + 1},
        expected_checkpoint_sha256=snapshot.revision_sha256,
    )
    return {"assimilation_receipt": developmental_receipt, "accepted": True}
