"""Focused fail-closed checks for the v1 source-to-assimilation boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate.product import ProductRefused
from substrate.product.apprenticeship import (
    CacheAttestedEvidenceAuthority,
    HostResources,
    assimilate_source_receipt,
    plan_apprenticeship,
    plan_source_acquisition,
    plan_workers,
)
from substrate.product.cache import ArtifactStore, LocalCacheVerifierTrustStore, sign_cache_attestation
from substrate.product.codec import sha256
from substrate.product.contracts import ApprenticeshipSpec, EntityManifest, ResourceBudget, SourcePolicy, SourceReceipt, SourceRequest
from substrate.product.entity import EntityStore
from substrate.product.pack_artifacts import generate_ed25519_keypair
from substrate.product.sources import plan_acquisition, validate_acquisition_plan, validate_receipt


def _policy() -> SourcePolicy:
    return SourcePolicy(allowed_schemes=("file",), allowed_file_roots=("/approved",))


def _spec() -> ApprenticeshipSpec:
    return ApprenticeshipSpec(
        name="bounded-source-study",
        objective="Inspect approved text without activating a source backend",
        evaluators=("hidden-source-suite",),
        source_policy=_policy(),
        worker_budget=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=8192),
        maximum_workers=2,
        wall_clock_minutes=30,
    )


def _store(tmp_path: Path) -> EntityStore:
    return EntityStore.create(
        tmp_path / "source-study.substrate",
        EntityManifest(entity_id="source-study", specialty="bounded source review", selected_packs=("engineering", "research")),
    )


def _request() -> SourceRequest:
    return SourceRequest(
        source_uri="file:///approved/private/never-persist-this-title.txt",
        modality="text",
        access_status="user-provided",
        declared_rights="operator-provided-license-2026",
        retrieval_mode="import",
    )


def _receipt(request: SourceRequest, plan_sha256: str, *, content_sha256: str = "a" * 64) -> SourceReceipt:
    return SourceReceipt(
        request=request,
        received_at="2026-08-04T00:00:00Z",
        retrieval_method="operator-import",
        content_sha256=content_sha256,
        acquisition_plan_sha256=plan_sha256,
        processing_history=({"extractor": "plain-text-v1"},),
    )


def _cache_attested_authority(
    tmp_path: Path,
    *,
    source_plan_sha256: str,
) -> tuple[CacheAttestedEvidenceAuthority, str]:
    """Build existing local cache evidence for this contract-level integration test."""

    source = tmp_path / "approved-source.txt"
    source.write_text("bounded source material\n", encoding="utf-8")
    private_key = tmp_path / "verifier-private.pem"
    public_key = tmp_path / "verifier-public.pem"
    generate_ed25519_keypair(private_key, public_key)
    trust = LocalCacheVerifierTrustStore(tmp_path / "verifier-trust")
    trust.trust(
        verifier_id="operator-local-review",
        public_key_path=public_key,
        allowed_rights_statuses=("user-provided",),
    )
    cache = ArtifactStore.create(tmp_path / "artifact-cache", capacity_bytes=1024 * 1024)
    descriptor = cache.ingest_file(
        source,
        media_type="text/plain",
        source_reference_sha256=source_plan_sha256,
        rights_status="user-provided",
    )
    cache.verify(
        descriptor.sha256,
        sign_cache_attestation(
            artifact_sha256=descriptor.sha256,
            cache_id=cache.cache_id,
            descriptor_sha256=sha256(descriptor.to_dict()),
            verifier_id="operator-local-review",
            rights_status="user-provided",
            private_key_path=private_key,
            expires_at="2035-01-01T00:00:00Z",
        ),
        trust,
    )
    return CacheAttestedEvidenceAuthority(cache, trust), descriptor.sha256


def test_source_plans_and_receipt_envelopes_keep_live_references_out_of_the_ledger(tmp_path: Path) -> None:
    store = _store(tmp_path)
    plan_apprenticeship(store, _spec(), HostResources(cpu_cores=8, memory_mib=16_384, disk_mib=65_536))
    request = _request()
    planned = plan_source_acquisition(store, request)

    serialized_plan = json.dumps(planned["plan"], sort_keys=True)
    serialized_ledger = store.ledger_path.read_text(encoding="utf-8")
    for sensitive_value in (
        request.source_uri,
        "private/never-persist-this-title.txt",
        request.declared_rights,
    ):
        assert sensitive_value not in serialized_plan
        assert sensitive_value not in serialized_ledger
    assert planned["plan"]["source_request"]["source_origin"] == {
        "scheme": "file",
        "scope": "operator-approved-file-root",
    }

    envelope = validate_receipt(_receipt(request, planned["plan"]["plan_sha256"]), _policy())
    serialized_envelope = json.dumps(envelope, sort_keys=True)
    assert request.source_uri not in serialized_envelope
    assert request.declared_rights not in serialized_envelope
    assert "processing_history" not in envelope["receipt"]


def test_assimilation_revalidates_the_sealed_plan_against_the_active_policy() -> None:
    request = _request()
    policy = _policy()
    plan = plan_acquisition(request, policy)

    assert validate_acquisition_plan(plan, policy, request=request)["plan_sha256"] == plan["plan_sha256"]

    request_tampered = {**plan, "source_request": request.to_dict()}
    request_tampered["plan_sha256"] = sha256({key: value for key, value in request_tampered.items() if key != "plan_sha256"})
    with pytest.raises(ProductRefused, match="does not bind its source request"):
        validate_acquisition_plan(request_tampered, policy, request=request)

    policy_tampered = {**plan, "source_policy_sha256": "b" * 64}
    policy_tampered["plan_sha256"] = sha256({key: value for key, value in policy_tampered.items() if key != "plan_sha256"})
    with pytest.raises(ProductRefused, match="does not bind the active source policy"):
        validate_acquisition_plan(policy_tampered, policy, request=request)


def test_contracts_refuse_secret_like_fields_and_unknown_serialized_fields() -> None:
    with pytest.raises(ProductRefused, match="declared_rights cannot contain credentials"):
        SourceRequest(
            source_uri="file:///approved/notes.txt",
            modality="text",
            access_status="user-provided",
            declared_rights="licensed material api_key=not-for-ledgers",
            retrieval_mode="import",
        )
    with pytest.raises(ProductRefused, match="processing_history includes an unsupported field"):
        SourceReceipt(
            request=_request(),
            received_at="2026-08-04T00:00:00Z",
            retrieval_method="operator-import",
            content_sha256="a" * 64,
            acquisition_plan_sha256="b" * 64,
            processing_history=({"token": "not-for-ledgers"},),
        )
    with pytest.raises(ProductRefused, match="source request is malformed"):
        SourceRequest.from_dict({**_request().to_dict(), "credential": "not-for-ledgers"})
    with pytest.raises(ProductRefused, match="source policy is malformed"):
        SourcePolicy.from_dict({**_policy().to_dict(), "credential": "not-for-ledgers"})


def test_bare_verifier_callback_cannot_increment_entity_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    plan_apprenticeship(store, _spec(), HostResources(cpu_cores=8, memory_mib=16_384, disk_mib=65_536))
    request = _request()
    planned = plan_source_acquisition(store, request)
    receipt = _receipt(request, planned["plan"]["plan_sha256"])
    calls = 0

    def fake_verifier(_receipt_value: SourceReceipt, _plan: dict[str, object]) -> bool:
        nonlocal calls
        calls += 1
        return True

    with pytest.raises(ProductRefused, match="without a cache-attested evidence authority"):
        assimilate_source_receipt(store, receipt, verifier=fake_verifier)
    assert calls == 0
    assert store.status()["state"]["evidence_assimilated"] == 0

    with pytest.raises(ProductRefused, match="without a cache-attested evidence authority"):
        assimilate_source_receipt(store, receipt)
    assert store.status()["state"]["evidence_assimilated"] == 0


def test_only_verified_cache_evidence_can_cross_into_the_entity_ledger(tmp_path: Path) -> None:
    store = _store(tmp_path)
    plan_apprenticeship(store, _spec(), HostResources(cpu_cores=8, memory_mib=16_384, disk_mib=65_536))
    request = _request()
    planned = plan_source_acquisition(store, request)
    authority, content_sha256 = _cache_attested_authority(
        tmp_path,
        source_plan_sha256=planned["plan"]["plan_sha256"],
    )

    result = assimilate_source_receipt(
        store,
        _receipt(request, planned["plan"]["plan_sha256"], content_sha256=content_sha256),
        verifier=authority,
    )

    assert result["accepted"] is True
    assert store.status()["state"]["evidence_assimilated"] == 1
    payload = result["assimilation_receipt"]["payload"]
    assert payload["status"] == "accepted-after-cache-attested-verification"
    assert payload["content_sha256"] == content_sha256
    assert request.source_uri not in json.dumps(payload, sort_keys=True)
    assert request.declared_rights not in json.dumps(payload, sort_keys=True)


def test_worker_admission_reserves_the_single_assimilator_before_workers() -> None:
    with pytest.raises(ProductRefused, match="cannot admit one worker"):
        plan_workers(
            HostResources(cpu_cores=3, memory_mib=6144, disk_mib=10_240),
            ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=8192),
            maximum_workers=1,
        )

    plan = plan_workers(
        HostResources(cpu_cores=10, memory_mib=10_240, disk_mib=100_000),
        ResourceBudget(cpu_cores=2, memory_mib=3072, disk_mib=25_000),
        maximum_workers=4,
    )
    assert plan.capacity_by_resource == {"cpu": 4, "disk": 3, "memory": 2}
    assert plan.reserved_for_assimilation == ResourceBudget(cpu_cores=1, memory_mib=1024, disk_mib=1024)
    assert plan.reserved_for_control_plane == ResourceBudget(cpu_cores=1, memory_mib=1024, disk_mib=1024)

    with pytest.raises(ProductRefused, match="resource budget is malformed"):
        ResourceBudget.from_dict({"cpu_cores": 1, "memory_mib": 1024, "disk_mib": 1024, "extra": 1})
