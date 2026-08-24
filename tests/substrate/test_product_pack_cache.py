"""Adversarial checks for signed packs and the non-executing artifact cache."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from substrate.product import ProductRefused
from substrate.product import cache as cache_module
from substrate.product.cache import (
    ArtifactDescriptor,
    ArtifactStore,
    CacheAttestation,
    LocalCacheVerifierTrustStore,
    ProcessingLineage,
    sign_cache_attestation,
)
from substrate.product.cli import main as product_main
from substrate.product.codec import sha256
from substrate.product.contracts import SourceReceipt, SourceRequest
from substrate.product.pack_artifacts import (
    LocalPackRegistry,
    LocalTrustStore,
    build_manifest,
    build_pack_artifact,
    generate_ed25519_keypair,
    inspect_pack_artifact,
    sign_pack_artifact,
    verify_pack_artifact,
)


def _keys(tmp_path: Path) -> tuple[Path, Path]:
    private_key = tmp_path / "publisher-private.pem"
    public_key = tmp_path / "publisher-public.pem"
    generate_ed25519_keypair(private_key, public_key)
    return private_key, public_key


def _signed_media_pack(tmp_path: Path) -> tuple[Path, Path]:
    private_key, public_key = _keys(tmp_path)
    artifact = tmp_path / "media.pack"
    build_pack_artifact("media", artifact, publisher="substrate-local")
    sign_pack_artifact(artifact, private_key)
    return artifact, public_key


def _trusted_media_store(tmp_path: Path, public_key: Path) -> LocalTrustStore:
    trust = LocalTrustStore(tmp_path / "trust")
    trust.trust(
        publisher="substrate-local",
        public_key_path=public_key,
        allowed_pack_names=("media",),
        allowed_capabilities=("media-decode",),
    )
    return trust


@pytest.mark.parametrize(
    "pack_name",
    ("engineering", "formal-math", "research", "media", "3d", "browser", "desktop", "data-science", "robotics"),
)
def test_every_initial_pack_builds_as_a_manifest_only_artifact(tmp_path: Path, pack_name: str) -> None:
    artifact = tmp_path / f"{pack_name}.pack"
    inspection = build_pack_artifact(pack_name, artifact, publisher="substrate-local")

    assert inspection["execution_permitted"] is False
    assert inspection["signed"] is False
    assert inspection["manifest"]["name"] == pack_name
    assert inspection["manifest"]["network_grants"] == []
    assert inspection["manifest"]["license_metadata"]["upstream_components_not_vendored"] is True


def test_repository_inspection_role_does_not_inherit_engineering_write_or_process_grants() -> None:
    manifest = build_manifest("engineering", publisher="substrate-local")

    assert manifest["capability_grants"] == ["filesystem-write", "subprocess"]
    assert manifest["tool_adapters"] == [
        {
            "adapter_id": "repository-inspection-v1",
            "required_capabilities": [],
            "role": "repository-inspection",
        }
    ]


def test_signed_pack_requires_scoped_local_trust_and_detects_tampering(tmp_path: Path) -> None:
    artifact, public_key = _signed_media_pack(tmp_path)

    assert inspect_pack_artifact(artifact)["signed"] is True
    with pytest.raises(ProductRefused, match="not trusted locally"):
        verify_pack_artifact(artifact, LocalTrustStore(tmp_path / "empty-trust"))

    too_narrow = LocalTrustStore(tmp_path / "too-narrow-trust")
    too_narrow.trust(
        publisher="substrate-local",
        public_key_path=public_key,
        allowed_pack_names=("media",),
        allowed_capabilities=(),
    )
    with pytest.raises(ProductRefused, match="does not permit this pack capability"):
        verify_pack_artifact(artifact, too_narrow)

    verification = verify_pack_artifact(artifact, _trusted_media_store(tmp_path, public_key))
    assert verification["verified"] is True
    assert verification["execution_permitted"] is False
    assert verification["pack"] == {"name": "media", "publisher": "substrate-local", "version": "1.0.0"}

    signature_path = artifact / "signature.json"
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    signature["signature_b64"] = "A" * len(signature["signature_b64"])
    signature_path.write_text(json.dumps(signature), encoding="utf-8")
    with pytest.raises(ProductRefused, match="signature verification failed"):
        verify_pack_artifact(artifact, _trusted_media_store(tmp_path, public_key))


def test_pack_key_loading_refuses_symlinked_or_hardlinked_public_keys(tmp_path: Path) -> None:
    _, public_key = _keys(tmp_path)
    symlinked = tmp_path / "public-link.pem"
    symlinked.symlink_to(public_key)
    with pytest.raises(ProductRefused, match="public key must be a regular non-symlink file"):
        LocalTrustStore(tmp_path / "symlink-trust").trust(
            publisher="substrate-local",
            public_key_path=symlinked,
            allowed_pack_names=("media",),
            allowed_capabilities=("media-decode",),
        )

    hardlinked = tmp_path / "public-hardlink.pem"
    os.link(public_key, hardlinked)
    with pytest.raises(ProductRefused, match="public key must not be a hard-linked file"):
        LocalTrustStore(tmp_path / "hardlink-trust").trust(
            publisher="substrate-local",
            public_key_path=hardlinked,
            allowed_pack_names=("media",),
            allowed_capabilities=("media-decode",),
        )


def test_pack_trust_rules_refuse_unknown_fields(tmp_path: Path) -> None:
    _, public_key = _keys(tmp_path)
    trust = LocalTrustStore(tmp_path / "trust")
    rule = trust.trust(
        publisher="substrate-local",
        public_key_path=public_key,
        allowed_pack_names=("media",),
        allowed_capabilities=("media-decode",),
    )
    path = trust._path(rule.publisher, rule.key_id)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["unexpected_scope"] = "all"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ProductRefused, match="trust rule fields are malformed"):
        trust.load(rule.publisher, rule.key_id)


def test_pack_install_is_a_verified_manifest_reference_only(tmp_path: Path) -> None:
    artifact, public_key = _signed_media_pack(tmp_path)
    trust = _trusted_media_store(tmp_path, public_key)
    registry = LocalPackRegistry(tmp_path / "registry")

    installed = registry.install(artifact, trust)

    installed_path = Path(installed["installation_path"])
    assert installed["installation"]["execution_permitted"] is False
    assert {path.name for path in installed_path.iterdir()} == {"installation.json", "manifest.json", "signature.json"}
    with pytest.raises(ProductRefused, match="already installed"):
        registry.install(artifact, trust)
    removed = registry.remove(
        pack_name="media",
        version="1.0.0",
        manifest_sha256=installed["installation"]["manifest_sha256"],
    )
    assert removed["removed_registry_reference_only"] is True
    assert not installed_path.exists()


def _source_reference() -> str:
    return sha256({"acquisition-plan": "approved-local-import"})


def _cache_verifier(tmp_path: Path) -> tuple[Path, LocalCacheVerifierTrustStore]:
    private_key, public_key = _keys(tmp_path)
    trust = LocalCacheVerifierTrustStore(tmp_path / "cache-verifier-trust")
    trust.trust(
        verifier_id="operator-local-review",
        public_key_path=public_key,
        allowed_rights_statuses=("licensed", "public", "user-provided"),
    )
    return private_key, trust


def _attestation(
    store: ArtifactStore,
    descriptor: ArtifactDescriptor,
    private_key: Path,
    *,
    rights_status: str = "user-provided",
) -> CacheAttestation:
    return sign_cache_attestation(
        artifact_sha256=descriptor.sha256,
        cache_id=store.cache_id,
        descriptor_sha256=sha256(descriptor.to_dict()),
        verifier_id="operator-local-review",
        rights_status=rights_status,
        private_key_path=private_key,
        expires_at="2035-01-01T00:00:00Z",
    )


def test_cache_quarantines_then_verifies_immutable_content_and_binds_source_receipts(tmp_path: Path) -> None:
    input_path = tmp_path / "lesson.txt"
    input_path.write_text("bounded source material\n", encoding="utf-8")
    source_reference = _source_reference()
    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)

    descriptor = store.ingest_file(
        input_path,
        media_type="text/plain",
        source_reference_sha256=source_reference,
        rights_status="user-provided",
    )
    assert store.status()["object_counts"]["quarantine"] == 1

    request = SourceRequest(
        source_uri="file:///approved/lesson.txt",
        modality="text",
        access_status="user-provided",
        declared_rights="operator-provided source",
        retrieval_mode="import",
    )
    receipt = SourceReceipt(
        request=request,
        received_at="2026-08-04T00:00:00Z",
        retrieval_method="operator-import",
        content_sha256=descriptor.sha256,
        acquisition_plan_sha256=source_reference,
    )
    assert store.trusted_source_receipt_verifier(receipt, {"plan_sha256": source_reference}, verifier_trust) is False

    verified = store.verify(descriptor.sha256, _attestation(store, descriptor, private_key), verifier_trust)
    assert verified.verification_status == "verified"
    explanation = store.explain(descriptor.sha256)
    assert explanation["zone"] == "verified"
    assert explanation["execution_permitted"] is False
    assert store.trusted_source_receipt_verifier(receipt, {"plan_sha256": source_reference}, verifier_trust) is True

    pin = store.pin(descriptor.sha256, reason="required by a pending apprenticeship")
    assert pin["artifact_sha256"] == descriptor.sha256
    assert store.gc(include_verified=True)["removed"] == []
    quarantined = store.quarantine(descriptor.sha256, reason="rights review reopened")
    assert quarantined.verification_status == "quarantined"
    assert store.trusted_source_receipt_verifier(receipt, {"plan_sha256": source_reference}, verifier_trust) is False


def test_cache_refuses_unsafe_bytes_and_cross_provenance_deduplication(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)
    opaque_input = tmp_path / "opaque.bin"
    opaque_input.write_bytes(b"\x00\x01\x02")
    descriptor = store.ingest_file(
        opaque_input,
        media_type="application/octet-stream",
        source_reference_sha256=_source_reference(),
        rights_status="user-provided",
    )
    with pytest.raises(ProductRefused, match="opaque binary"):
        store.verify(descriptor.sha256, _attestation(store, descriptor, private_key), verifier_trust)
    assert store.explain(descriptor.sha256)["zone"] == "quarantine"

    text_input = tmp_path / "same-content.txt"
    text_input.write_text("same immutable bytes", encoding="utf-8")
    text_descriptor = store.ingest_file(
        text_input,
        media_type="text/plain",
        source_reference_sha256=sha256({"plan": "first"}),
        rights_status="user-provided",
    )
    with pytest.raises(ProductRefused, match="different provenance"):
        store.ingest_file(
            text_input,
            media_type="text/plain",
            source_reference_sha256=sha256({"plan": "second"}),
            rights_status="user-provided",
        )
    assert store.explain(text_descriptor.sha256)["zone"] == "quarantine"

    link_input = tmp_path / "link.txt"
    link_input.symlink_to(text_input)
    with pytest.raises(ProductRefused, match="non-symlink"):
        store.ingest_file(
            link_input,
            media_type="text/plain",
            source_reference_sha256=sha256({"plan": "link"}),
            rights_status="user-provided",
        )


def test_cache_accepts_only_strict_substrate_json_for_structured_evidence(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)
    source_reference = _source_reference()

    capture = tmp_path / "browser-capture.json"
    capture.write_text('{"capture":{"dom":"bounded"}}', encoding="utf-8")
    capture_descriptor = store.ingest_file(
        capture,
        media_type="application/x-substrate-browser-capture+json",
        source_reference_sha256=source_reference,
        rights_status="user-provided",
    )
    promoted = store.verify(capture_descriptor.sha256, _attestation(store, capture_descriptor, private_key), verifier_trust)
    assert promoted.verification_status == "verified"
    assert store.explain(capture_descriptor.sha256)["descriptor"]["media_type"] == "application/x-substrate-browser-capture+json"

    duplicate_keys = tmp_path / "duplicate-keys.json"
    duplicate_keys.write_text('{"capture":1,"capture":2}', encoding="utf-8")
    duplicate_descriptor = store.ingest_file(
        duplicate_keys,
        media_type="application/x-substrate-browser-capture+json",
        source_reference_sha256=sha256({"plan": "duplicate-keys"}),
        rights_status="user-provided",
    )
    with pytest.raises(ProductRefused, match="strict JSON object"):
        store.verify(duplicate_descriptor.sha256, _attestation(store, duplicate_descriptor, private_key), verifier_trust)

    generic_json = tmp_path / "generic.json"
    generic_json.write_text('{"capture":"not a product structured type"}', encoding="utf-8")
    generic_descriptor = store.ingest_file(
        generic_json,
        media_type="application/json",
        source_reference_sha256=sha256({"plan": "generic-json"}),
        rights_status="user-provided",
    )
    with pytest.raises(ProductRefused, match="media type does not match"):
        store.verify(generic_descriptor.sha256, _attestation(store, generic_descriptor, private_key), verifier_trust)


def test_derived_artifacts_require_verified_inputs_and_promote_to_processed_cache(tmp_path: Path) -> None:
    source_reference = _source_reference()
    source = tmp_path / "source.txt"
    source.write_text("source material", encoding="utf-8")
    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)
    parent = store.ingest_file(
        source,
        media_type="text/plain",
        source_reference_sha256=source_reference,
        rights_status="user-provided",
    )
    lineage = ProcessingLineage(
        input_sha256=parent.sha256,
        recipe_id="plain-text-normalize-v1",
        tool_artifact_sha256="a" * 64,
    )
    derived = tmp_path / "derived.txt"
    derived.write_text("normalized source material", encoding="utf-8")
    with pytest.raises(ProductRefused, match="not immutable and verified"):
        store.ingest_derivative_file(
            derived,
            media_type="text/plain",
            source_reference_sha256=source_reference,
            rights_status="user-provided",
            processing_lineage=(lineage,),
        )

    store.verify(parent.sha256, _attestation(store, parent, private_key), verifier_trust)
    tool = tmp_path / "tool.txt"
    tool.write_text("verified tool artifact", encoding="utf-8")
    tool_descriptor = store.ingest_file(
        tool,
        media_type="text/plain",
        source_reference_sha256=source_reference,
        rights_status="user-provided",
    )
    store.verify(tool_descriptor.sha256, _attestation(store, tool_descriptor, private_key), verifier_trust)
    lineage = ProcessingLineage(
        input_sha256=parent.sha256,
        recipe_id="plain-text-normalize-v1",
        tool_artifact_sha256=tool_descriptor.sha256,
    )
    staged = store.ingest_derivative_file(
        derived,
        media_type="text/plain",
        source_reference_sha256=source_reference,
        rights_status="user-provided",
        processing_lineage=(lineage,),
    )
    store.verify(staged.sha256, _attestation(store, staged, private_key), verifier_trust)
    assert store.explain(staged.sha256)["zone"] == "processed"
    assert staged.sha256 in store.explain(parent.sha256)["descriptor"]["derived_objects"]

    store.quarantine(tool_descriptor.sha256, reason="tool provenance was revoked")
    assert store.explain(staged.sha256)["zone"] == "quarantine"
    revoked_derivative = ArtifactDescriptor.from_dict(store.explain(staged.sha256)["descriptor"])
    with pytest.raises(ProductRefused, match="artifact lineage tool is not verified"):
        store.verify(staged.sha256, _attestation(store, revoked_derivative, private_key), verifier_trust)

    store.quarantine(parent.sha256, reason="input rights were revoked")
    assert store.explain(staged.sha256)["zone"] == "quarantine"


def test_cache_requires_signed_local_verifier_and_recovers_interrupted_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)
    source = tmp_path / "lesson.txt"
    source.write_text("cache review fixture", encoding="utf-8")
    descriptor = store.ingest_file(
        source,
        media_type="text/plain",
        source_reference_sha256=_source_reference(),
        rights_status="user-provided",
    )
    attestation = _attestation(store, descriptor, private_key)

    with pytest.raises(ProductRefused, match="not trusted locally"):
        store.verify(descriptor.sha256, attestation, LocalCacheVerifierTrustStore(tmp_path / "untrusted"))

    tampered = attestation.to_dict()
    tampered["signature_b64"] = "A" * len(tampered["signature_b64"])
    with pytest.raises(ProductRefused, match="signature verification failed"):
        store.verify(descriptor.sha256, CacheAttestation.from_dict(tampered), verifier_trust)

    real_replace = cache_module.os.replace
    promoted_directory = store._object_directory("verified", descriptor.sha256)

    def refuse_final_publish(source_path: object, target_path: object) -> None:
        if target_path == promoted_directory and Path(source_path).name.startswith(f".{descriptor.sha256}.transition-"):
            raise OSError("injected final publish failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(cache_module.os, "replace", refuse_final_publish)
    with pytest.raises(ProductRefused, match="cannot transition artifact cache object"):
        store.verify(descriptor.sha256, attestation, verifier_trust)
    monkeypatch.setattr(cache_module.os, "replace", real_replace)

    reopened = ArtifactStore.open(store.root, verifier_trust_store=verifier_trust)
    assert reopened.explain(descriptor.sha256)["zone"] == "verified"

    blob = reopened._blob_path("verified", descriptor.sha256)
    blob.chmod(0o644)
    blob.write_text("tampered after promotion", encoding="utf-8")
    blob.chmod(0o444)
    request = SourceRequest(
        source_uri="file:///approved/lesson.txt",
        modality="text",
        access_status="user-provided",
        declared_rights="operator-provided source",
        retrieval_mode="import",
    )
    receipt = SourceReceipt(
        request=request,
        received_at="2026-08-04T00:00:00Z",
        retrieval_method="operator-import",
        content_sha256=descriptor.sha256,
        acquisition_plan_sha256=_source_reference(),
    )
    assert reopened.trusted_source_receipt_verifier(receipt, {"plan_sha256": _source_reference()}, verifier_trust) is False
    with pytest.raises(ProductRefused, match="cache blob digest or byte length"):
        reopened.status()
    with pytest.raises(ProductRefused, match="cache blob digest or byte length"):
        reopened.explain(descriptor.sha256)


def test_cache_refuses_post_promotion_provenance_relabeling(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)
    source = tmp_path / "lesson.txt"
    source.write_text("verified source material", encoding="utf-8")
    descriptor = store.ingest_file(
        source,
        media_type="text/plain",
        source_reference_sha256=_source_reference(),
        rights_status="user-provided",
    )
    store.verify(descriptor.sha256, _attestation(store, descriptor, private_key), verifier_trust)

    descriptor_path = store._descriptor_path("verified", descriptor.sha256)
    tampered = json.loads(descriptor_path.read_text(encoding="utf-8"))
    tampered["source_reference_sha256"] = sha256({"acquisition-plan": "unrelated"})
    descriptor_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ProductRefused, match="provenance differs from its attested descriptor"):
        store.explain(descriptor.sha256)


def test_cache_never_treats_a_self_consistent_forged_receipt_as_verified(tmp_path: Path) -> None:
    """The cache directory cannot mint a trusted promotion by rewriting JSON."""

    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)
    source = tmp_path / "lesson.txt"
    source.write_text("verified source material", encoding="utf-8")
    descriptor = store.ingest_file(
        source,
        media_type="text/plain",
        source_reference_sha256=_source_reference(),
        rights_status="user-provided",
    )
    store.verify(descriptor.sha256, _attestation(store, descriptor, private_key), verifier_trust)

    receipt_path = store._verification_receipt_path("verified", descriptor.sha256)
    descriptor_path = store._descriptor_path("verified", descriptor.sha256)
    forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # An attacker with cache-directory write access can keep every local hash
    # self-consistent.  It still cannot make the attestation valid under the
    # separate local verifier trust root.
    forged_receipt["attestation"]["signature_b64"] = "not-a-valid-signature"
    forged_receipt["attestation_sha256"] = sha256(forged_receipt["attestation"])
    forged_descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    forged_descriptor["verification_receipt_sha256"] = sha256(forged_receipt)
    receipt_path.write_text(json.dumps(forged_receipt), encoding="utf-8")
    descriptor_path.write_text(json.dumps(forged_descriptor), encoding="utf-8")

    with pytest.raises(ProductRefused, match="cache attestation signature verification failed"):
        store.explain(descriptor.sha256)
    with pytest.raises(ProductRefused, match="cache attestation signature verification failed"):
        store.status()
    with pytest.raises(ProductRefused, match="cache attestation signature verification failed"):
        store.pin(descriptor.sha256, reason="must not pin forged evidence")
    with pytest.raises(ProductRefused, match="cache attestation signature verification failed"):
        ArtifactStore.open(store.root, verifier_trust_store=verifier_trust)

    unbound = ArtifactStore.open(store.root)
    with pytest.raises(ProductRefused, match="explicit local verifier trust store"):
        unbound.explain(descriptor.sha256)
    # Emergency revocation remains available without accepting the forged
    # object as trusted evidence.
    assert unbound.quarantine(descriptor.sha256, reason="invalid local receipt").verification_status == "quarantined"


def test_cache_refuses_an_expired_attestation_on_later_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ArtifactStore.create(tmp_path / "cache", capacity_bytes=1024 * 1024)
    private_key, verifier_trust = _cache_verifier(tmp_path)
    source = tmp_path / "lesson.txt"
    source.write_text("expiry fixture", encoding="utf-8")
    descriptor = store.ingest_file(
        source,
        media_type="text/plain",
        source_reference_sha256=_source_reference(),
        rights_status="user-provided",
    )
    store.verify(descriptor.sha256, _attestation(store, descriptor, private_key), verifier_trust)

    class FutureDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return datetime(2036, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(cache_module, "datetime", FutureDateTime)
    with pytest.raises(ProductRefused, match="cache verification attestation has expired"):
        store.explain(descriptor.sha256)


def test_pack_and_cache_cli_are_explicit_local_nonexecuting_operations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    artifact = tmp_path / "media.pack"
    trust_directory = tmp_path / "trust"
    registry = tmp_path / "registry"
    cache_directory = tmp_path / "cache"
    cache_verifier_trust = tmp_path / "cache-verifier-trust"

    product_main(["pack", "keygen", "--private-key", str(private_key), "--public-key", str(public_key)])
    assert json.loads(capsys.readouterr().out)["execution_permitted"] is False
    product_main(["pack", "build", "media", str(artifact), "--publisher", "substrate-local"])
    assert json.loads(capsys.readouterr().out)["signed"] is False
    product_main(["pack", "sign", str(artifact), "--private-key", str(private_key)])
    assert json.loads(capsys.readouterr().out)["algorithm"] == "ed25519"
    product_main(
        [
            "pack",
            "trust",
            "--trust-store",
            str(trust_directory),
            "--publisher",
            "substrate-local",
            "--public-key",
            str(public_key),
            "--pack",
            "media",
            "--capability",
            "media-decode",
        ]
    )
    assert json.loads(capsys.readouterr().out)["execution_permitted"] is False
    product_main(["pack", "verify", str(artifact), "--trust-store", str(trust_directory)])
    assert json.loads(capsys.readouterr().out)["verified"] is True
    product_main(
        [
            "pack",
            "install",
            str(artifact),
            "--trust-store",
            str(trust_directory),
            "--registry",
            str(registry),
        ]
    )
    assert json.loads(capsys.readouterr().out)["installation"]["execution_permitted"] is False

    product_main(
        [
            "cache",
            "trust-verifier",
            "--trust-store",
            str(cache_verifier_trust),
            "--verifier-id",
            "operator-local-review",
            "--public-key",
            str(public_key),
            "--rights-status",
            "user-provided",
        ]
    )
    assert json.loads(capsys.readouterr().out)["execution_permitted"] is False

    source = tmp_path / "lesson.txt"
    source.write_text("local cache source", encoding="utf-8")
    product_main(["cache", "init", str(cache_directory), "--capacity-bytes", "1048576"])
    assert json.loads(capsys.readouterr().out)["execution_permitted"] is False
    source_reference = _source_reference()
    product_main(
        [
            "cache",
            "add",
            str(cache_directory),
            str(source),
            "--media-type",
            "text/plain",
            "--source-reference-sha256",
            source_reference,
            "--rights-status",
            "user-provided",
        ]
    )
    descriptor = json.loads(capsys.readouterr().out)["descriptor"]
    product_main(
        [
            "cache",
            "attest",
            str(cache_directory),
            descriptor["sha256"],
            "--verifier-id",
            "operator-local-review",
            "--private-key",
            str(private_key),
            "--rights-status",
            "user-provided",
            "--expires-at",
            "2035-01-01T00:00:00Z",
        ]
    )
    attestation = json.loads(capsys.readouterr().out)["attestation"]
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    product_main(
        [
            "cache",
            "verify",
            str(cache_directory),
            descriptor["sha256"],
            "--attestation",
            str(attestation_path),
            "--verifier-trust-store",
            str(cache_verifier_trust),
        ]
    )
    assert json.loads(capsys.readouterr().out)["descriptor"]["verification_status"] == "verified"
    product_main(
        [
            "cache",
            "pin",
            str(cache_directory),
            descriptor["sha256"],
            "--reason",
            "held for review",
            "--verifier-trust-store",
            str(cache_verifier_trust),
        ]
    )
    assert json.loads(capsys.readouterr().out)["execution_permitted"] is False
