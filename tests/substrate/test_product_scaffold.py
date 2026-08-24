"""Focused tests for the bounded post-Odyssey product foundation."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from substrate import cli as root_cli
from substrate import sandbox
from substrate.product import ProductRefused
from substrate.product.apprenticeship import HostResources, assimilate_source_receipt, plan_apprenticeship, plan_source_acquisition, plan_workers
from substrate.product.backends import (
    BackendCandidate,
    BackendProbeResult,
    PlatformFacts,
    plan_backend_dry_run,
    plan_backend_dry_run_for_entity,
    probe_backends,
    select_preferred_backend,
)
from substrate.product.cli import main as product_main
from substrate.product.codec import sha256
from substrate.product.contracts import ApprenticeshipSpec, EntityManifest, ResourceBudget, SourcePolicy, SourceReceipt, SourceRequest
from substrate.product.entity import EntityStore
from substrate.product.packs import BUILTIN_PACKS, plan_sandbox, refuse_execution
from substrate.product.sources import plan_acquisition


def _manifest() -> EntityManifest:
    return EntityManifest(
        entity_id="go-systems-engineer",
        specialty="Go distributed systems",
        selected_packs=("engineering", "research"),
    )


def _store(tmp_path: Path) -> EntityStore:
    return EntityStore.create(tmp_path / "go-systems-engineer.substrate", _manifest())


def _source_policy() -> SourcePolicy:
    return SourcePolicy(allowed_schemes=("file",), allowed_file_roots=("/approved",))


def _spec() -> ApprenticeshipSpec:
    return ApprenticeshipSpec(
        name="go-systems-foundation",
        objective="Build and verify bounded Go systems exercises",
        evaluators=("hidden-go-suite",),
        source_policy=_source_policy(),
        worker_budget=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=8192),
        maximum_workers=8,
        wall_clock_minutes=480,
    )


def _parallel_record(entity_root: str, start: multiprocessing.synchronize.Event, result: multiprocessing.queues.Queue, label: str) -> None:
    try:
        start.wait(timeout=10)
        receipt = EntityStore(Path(entity_root)).record("parallel_note", {"label": label})
        result.put(("ok", receipt["sequence"]))
    except Exception as exc:  # pragma: no cover - child failures are asserted by the parent.
        result.put(("error", repr(exc)))


def test_portable_entity_round_trip_and_checkpoint_validation(tmp_path: Path) -> None:
    store = _store(tmp_path)

    status = store.status()
    verification = store.validate()

    assert status["entity"]["entity_id"] == "go-systems-engineer"
    assert status["entity"]["selected_packs"] == ["engineering", "research"]
    assert status["execution_permitted"] is False
    assert status["ledger"]["receipt_count"] == 1
    assert verification["valid"] is True
    assert {path.name for path in store.root.iterdir()} == {
        "checkpoint.json",
        "developmental-state.json",
        "entity.json",
        "receipts.jsonl",
    }


def test_empty_organ_requirement_preserves_the_original_v1_manifest_shape() -> None:
    legacy_manifest = {
        "entity_id": "go-systems-engineer",
        "schema_version": "substrate-product-v1",
        "selected_packs": ["engineering", "research"],
        "specialty": "Go distributed systems",
    }

    assert EntityManifest.from_dict(legacy_manifest).to_dict() == legacy_manifest
    explicit_empty_manifest = {**legacy_manifest, "organ_requirements": []}
    assert EntityManifest.from_dict(explicit_empty_manifest).to_dict() == explicit_empty_manifest


def test_receipt_tampering_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("operator_note", {"note": "first bounded plan"})
    store.ledger_path.write_text(store.ledger_path.read_text(encoding="utf-8").replace("operator_note", "tampered_note"), encoding="utf-8")

    with pytest.raises(ProductRefused, match="receipt hash verification failed"):
        store.validate()


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["specialty"] = "tampered specialty"
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProductRefused, match="initialization receipt does not bind"):
        store.validate()


def test_interprocess_writer_lock_serializes_receipt_sequences(tmp_path: Path) -> None:
    store = _store(tmp_path)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    workers = [context.Process(target=_parallel_record, args=(str(store.root), start, results, label)) for label in ("a", "b")]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    observed = [results.get(timeout=2) for _ in workers]

    assert sorted(observed) == [("ok", 2), ("ok", 3)]
    assert store.validate()["receipt_count"] == 3


def test_interrupted_write_requires_explicit_verified_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from substrate.product import entity as entity_module

    store = _store(tmp_path)
    original_write = entity_module.atomic_write_json

    def fail_state_write(path: Path, value: dict) -> None:
        if path == store.state_path:
            raise ProductRefused("simulated interrupted state write")
        original_write(path, value)

    monkeypatch.setattr(entity_module, "atomic_write_json", fail_state_write)
    with pytest.raises(ProductRefused, match="simulated interrupted"):
        store.record("interrupted_note", {"note": "must recover"})
    assert store.pending_transaction_path.exists()
    with pytest.raises(ProductRefused, match="pending transaction"):
        store.load()

    monkeypatch.setattr(entity_module, "atomic_write_json", original_write)
    assert store.recover()["valid"] is True
    assert store.validate()["receipt_count"] == 2


def test_failed_initial_staging_leaves_the_requested_entity_path_retriable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from substrate.product import entity as entity_module

    entity_root = tmp_path / "retriable.substrate"
    original_write = entity_module.atomic_write_json

    def fail_pending_write(path: Path, value: dict) -> None:
        if path.name == ".pending-transaction.json":
            raise ProductRefused("simulated pending transaction failure")
        original_write(path, value)

    monkeypatch.setattr(entity_module, "atomic_write_json", fail_pending_write)
    with pytest.raises(ProductRefused, match="simulated pending transaction failure"):
        EntityStore.create(entity_root, _manifest())
    assert not entity_root.exists()
    assert not list(tmp_path.glob(".retriable.substrate.staging-*"))

    monkeypatch.setattr(entity_module, "atomic_write_json", original_write)
    assert EntityStore.create(entity_root, _manifest()).validate()["valid"] is True


def test_invalid_packs_and_resource_values_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ProductRefused, match="duplicates"):
        EntityManifest(entity_id="duplicate", specialty="test", selected_packs=("engineering", "engineering"))
    with pytest.raises(ProductRefused, match="unknown capability"):
        EntityStore.create(
            tmp_path / "unknown-pack.substrate",
            EntityManifest(entity_id="unknown-pack", specialty="test", selected_packs=("not-a-pack",)),
        )
    with pytest.raises(ProductRefused, match="positive"):
        ResourceBudget(cpu_cores=0, memory_mib=1024, disk_mib=1024)
    with pytest.raises(ProductRefused, match="cannot admit"):
        plan_workers(
            HostResources(cpu_cores=1, memory_mib=1024, disk_mib=1024),
            ResourceBudget(cpu_cores=2, memory_mib=1024, disk_mib=1024),
            maximum_workers=1,
        )


def test_media_and_video_are_policy_gated_and_never_execute() -> None:
    request = SourceRequest(
        source_uri="https://youtube.com/watch",
        modality="video",
        access_status="public",
        declared_rights="approved by rights holder",
        retrieval_mode="download",
    )
    policy = SourcePolicy(allowed_schemes=("https",), allowed_domains=("youtube.com",), allow_download=False)

    with pytest.raises(ProductRefused, match="persistent download"):
        policy.assert_permits(request)

    plan = plan_sandbox(
        entity_id="media-specialist",
        selected_packs=("media",),
        worker_budget=ResourceBudget(cpu_cores=4, memory_mib=8192, disk_mib=16384),
        source_policy=policy,
    )
    assert plan["execution_permitted"] is False
    assert plan["network_mode"] == "none"
    assert "yt-dlp (optional host/image requirement)" in BUILTIN_PACKS["media"].tool_requirements
    with pytest.raises(ProductRefused, match="sandbox execution is unavailable"):
        refuse_execution(plan)

    with pytest.raises(ProductRefused, match="minimum resource profile"):
        plan_sandbox(
            entity_id="media-specialist",
            selected_packs=("media",),
            worker_budget=ResourceBudget(cpu_cores=1, memory_mib=1024, disk_mib=1024),
            source_policy=policy,
        )


def test_source_policy_rejects_unsafe_paths_remote_bypass_and_uri_credentials() -> None:
    local_policy = _source_policy()
    request = SourceRequest(
        source_uri="file:///etc/passwd",
        modality="text",
        access_status="user-provided",
        declared_rights="operator-provided material",
        retrieval_mode="import",
    )
    with pytest.raises(ProductRefused, match="outside approved roots"):
        local_policy.assert_permits(request)

    source_fields = {
        "modality": "document",
        "access_status": "licensed",
        "declared_rights": "licensed archive",
    }
    for source_uri, error in (
        ("file:relative/notes.txt", "absolute path"),
        ("file://evil.example/approved/notes.txt", "cannot include an authority"),
        ("file:////evil.example/approved/notes.txt", "local absolute path"),
        ("file:///%2Fevil.example/approved/notes.txt", "local absolute path"),
        ("https://operator:secret@approved.example/notes.txt", "credentials"),
        ("https://approved.example/notes.txt?token=secret", "query"),
        ("https://approved.example/notes.txt#chapter", "query or fragment"),
        ("https:/approved.example/notes.txt", "authority"),
        ("https://approved.example:443/notes.txt", "explicit port"),
    ):
        with pytest.raises(ProductRefused, match=error):
            SourceRequest(source_uri=source_uri, **source_fields)

    remote_policy = SourcePolicy(allowed_schemes=("s3",), allowed_domains=("approved-bucket",))
    request = SourceRequest(
        source_uri="s3://evil-bucket/notes.txt",
        modality="document",
        access_status="licensed",
        declared_rights="licensed archive",
        retrieval_mode="metadata",
    )
    with pytest.raises(ProductRefused, match="not approved"):
        remote_policy.assert_permits(request)

    https_policy = SourcePolicy(allowed_schemes=("https",), allowed_domains=("approved.example",))
    for source_uri in ("https://approved.example.evil/notes.txt", "https://evil.example/approved.example/notes.txt"):
        remote_request = SourceRequest(source_uri=source_uri, **source_fields)
        with pytest.raises(ProductRefused, match="not approved"):
            https_policy.assert_permits(remote_request)

    with pytest.raises(ProductRefused, match="requires at least one approved domain"):
        SourcePolicy(allowed_schemes=("https",))
    for domain in (
        "https://approved.example",
        "approved.example:443",
        "approved.example/path",
        "approved.example@evil.example",
        "approved..example",
        "approved.example.",
    ):
        with pytest.raises(ProductRefused, match="exact lowercase hostnames"):
            SourcePolicy(allowed_schemes=("https",), allowed_domains=(domain,))

    with pytest.raises(ProductRefused, match="source request is malformed"):
        plan_acquisition(object(), local_policy)  # type: ignore[arg-type]
    with pytest.raises(ProductRefused, match="source policy is malformed"):
        plan_acquisition(request, object())  # type: ignore[arg-type]
    malformed_policy = local_policy.to_dict()
    malformed_policy["allowed_schemes"] = "file"
    with pytest.raises(ProductRefused, match="source policy is malformed"):
        SourcePolicy.from_dict(malformed_policy)


def test_source_acquisition_plan_is_hashable_and_non_executing() -> None:
    request = SourceRequest(
        source_uri="file:///approved/notes.txt",
        modality="document",
        access_status="user-provided",
        declared_rights="operator-provided study material",
        retrieval_mode="import",
    )
    plan = plan_acquisition(request, _source_policy())

    assert plan["adapter"] == "operator-import"
    assert plan["execution_permitted"] is False
    assert len(plan["plan_sha256"]) == 64


def test_worker_plan_honors_cpu_memory_disk_and_caller_cap() -> None:
    plan = plan_workers(
        HostResources(cpu_cores=10, memory_mib=10_240, disk_mib=100_000),
        ResourceBudget(cpu_cores=2, memory_mib=3072, disk_mib=25_000),
        maximum_workers=4,
    )

    # One control-plane and one authoritative-assimilation reserve are held
    # outside worker capacity.  The memory vector, not CPU, is now limiting.
    assert plan.capacity_by_resource == {"cpu": 4, "disk": 3, "memory": 2}
    assert plan.concurrent_workers == 2
    assert plan.authoritative_assimilation_writers == 1


def test_apprenticeship_assimilation_rejects_an_untrusted_callback(tmp_path: Path) -> None:
    store = _store(tmp_path)
    planned = plan_apprenticeship(
        store,
        _spec(),
        HostResources(cpu_cores=20, memory_mib=65_536, disk_mib=262_144),
    )
    assert planned["plan"]["worker_plan"]["concurrent_workers"] == 8
    assert store.status()["state"]["active_apprenticeship"] == "go-systems-foundation"

    with pytest.raises(ProductRefused, match="declared_rights"):
        SourceRequest(
            source_uri="file:///approved/lesson.txt",
            modality="text",
            access_status="user-provided",
            declared_rights="",
            retrieval_mode="import",
        )

    request = SourceRequest(
        source_uri="file:///approved/lesson.txt",
        modality="text",
        access_status="user-provided",
        declared_rights="operator-provided training material",
        retrieval_mode="import",
    )
    source_plan = plan_source_acquisition(store, request)
    valid = SourceReceipt(
        request=request,
        received_at="2026-08-03T00:00:00Z",
        retrieval_method="operator-import",
        content_sha256="a" * 64,
        acquisition_plan_sha256=source_plan["plan"]["plan_sha256"],
        processing_history=({"extractor": "plain-text-v1"},),
    )
    with pytest.raises(ProductRefused, match="cache-attested evidence authority"):
        assimilate_source_receipt(store, valid, verifier=lambda _receipt, _plan: True)

    # The focused assimilation-hardening suite covers the accepted path with
    # a signed locally trusted cache object.  This scaffold check makes sure
    # a caller-provided callback cannot bypass that evidence boundary.
    assert store.status()["state"]["evidence_assimilated"] == 0
    assert store.validate()["receipt_count"] == 3


def test_assimilation_requires_active_plan_matching_policy_pack_and_adapter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    request = SourceRequest(
        source_uri="file:///approved/lesson.txt",
        modality="text",
        access_status="user-provided",
        declared_rights="operator-provided training material",
        retrieval_mode="import",
    )
    with pytest.raises(ProductRefused, match="requires an active apprenticeship plan"):
        plan_source_acquisition(store, request)

    plan_apprenticeship(store, _spec(), HostResources(cpu_cores=20, memory_mib=65_536, disk_mib=262_144))
    source_plan = plan_source_acquisition(store, request)
    text_receipt = SourceReceipt(
        request=request,
        received_at="2026-08-03T00:00:00Z",
        retrieval_method="operator-import",
        content_sha256="b" * 64,
        acquisition_plan_sha256=source_plan["plan"]["plan_sha256"],
    )
    with pytest.raises(ProductRefused, match="cache-attested evidence authority"):
        assimilate_source_receipt(store, text_receipt)

    bad_adapter = SourceReceipt(
        request=text_receipt.request,
        received_at=text_receipt.received_at,
        retrieval_method="untrusted-import",
        content_sha256="c" * 64,
        acquisition_plan_sha256=text_receipt.acquisition_plan_sha256,
    )
    with pytest.raises(ProductRefused, match="does not match its declared source adapter"):
        assimilate_source_receipt(store, bad_adapter, verifier=lambda _receipt, _plan: True)

    engineering_only = EntityStore.create(
        tmp_path / "engineering-only.substrate",
        EntityManifest(entity_id="engineering-only", specialty="systems", selected_packs=("engineering",)),
    )
    plan_apprenticeship(engineering_only, _spec(), HostResources(cpu_cores=20, memory_mib=65_536, disk_mib=262_144))
    video_request = SourceRequest(
        source_uri="file:///approved/lesson.mp4",
        modality="video",
        access_status="user-provided",
        declared_rights="operator-provided training material",
        retrieval_mode="import",
    )
    with pytest.raises(ProductRefused, match="not permitted by the entity's selected"):
        plan_source_acquisition(engineering_only, video_request)


def test_assimilation_refuses_a_source_plan_after_its_active_policy_is_replaced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    host = HostResources(cpu_cores=20, memory_mib=65_536, disk_mib=262_144)
    plan_apprenticeship(store, _spec(), host)
    request = SourceRequest(
        source_uri="file:///approved/lesson.txt",
        modality="text",
        access_status="user-provided",
        declared_rights="operator-provided training material",
        retrieval_mode="import",
    )
    source_plan = plan_source_acquisition(store, request)
    receipt = SourceReceipt(
        request=request,
        received_at="2026-08-03T00:00:00Z",
        retrieval_method="operator-import",
        content_sha256="d" * 64,
        acquisition_plan_sha256=source_plan["plan"]["plan_sha256"],
    )
    replacement = ApprenticeshipSpec(
        name="remote-replacement",
        objective="Use only a newly approved remote corpus",
        evaluators=("hidden-go-suite",),
        source_policy=SourcePolicy(allowed_schemes=("https",), allowed_domains=("approved.example",)),
        worker_budget=ResourceBudget(cpu_cores=2, memory_mib=4096, disk_mib=8192),
        maximum_workers=8,
        wall_clock_minutes=480,
    )

    plan_apprenticeship(store, replacement, host)
    with pytest.raises(ProductRefused, match="does not bind the active source policy"):
        assimilate_source_receipt(store, receipt)
    status = store.status()
    assert status["state"]["active_apprenticeship"] == "remote-replacement"
    assert status["state"]["evidence_assimilated"] == 0
    assert status["ledger"]["receipt_count"] == 4


def _which_map(mapping: dict[str, str | None]):
    def lookup(name: str) -> str | None:
        return mapping.get(name)

    return lookup


def _sealed_sandbox_variant(sandbox_plan: dict, **updates: object) -> dict:
    variant = {**sandbox_plan, **updates}
    variant["plan_sha256"] = sha256({key: value for key, value in variant.items() if key != "plan_sha256"})
    return variant


def test_backend_probe_prefers_apple_container_on_darwin_apple_silicon() -> None:
    probe = probe_backends(
        platform_facts=PlatformFacts(system="Darwin", machine="arm64"),
        which=_which_map({"container": "/usr/local/bin/container", "docker": "/usr/local/bin/docker"}),
    )

    assert probe.preferred_backend == "apple-container"
    assert probe.selection_status == "recommended"
    assert probe.execution_permitted is False
    assert probe.selection_is_authorization is False
    assert probe.selection_is_dry_run is True
    by_id = {candidate.backend_id: candidate for candidate in probe.candidates}
    assert by_id["apple-container"].selection_status == "selected"
    assert by_id["apple-container"].daemon_status == "not-probed"
    assert by_id["docker"].eligible is True
    assert by_id["docker"].selection_status == "eligible-not-selected"


def test_backend_probe_falls_back_to_docker_or_reports_unavailable() -> None:
    linux = probe_backends(
        platform_facts=PlatformFacts(system="Linux", machine="x86_64"),
        which=_which_map({"container": "/usr/bin/container", "docker": "/usr/bin/docker"}),
    )
    unavailable = probe_backends(
        platform_facts=PlatformFacts(system="Darwin", machine="x86_64"),
        which=_which_map({}),
    )

    assert linux.preferred_backend == "docker"
    assert "Apple Silicon" in " ".join(linux.candidates[0].rejection_reasons)
    assert unavailable.preferred_backend is None
    assert unavailable.selection_status == "unavailable"
    assert all(not candidate.eligible for candidate in unavailable.candidates)
    assert select_preferred_backend(unavailable.candidates) == (None, "unavailable")


def test_backend_dry_run_binds_verified_sandbox_digest_and_refuses_unsafe_input(tmp_path: Path) -> None:
    store = _store(tmp_path)
    planned = plan_apprenticeship(
        store,
        _spec(),
        HostResources(cpu_cores=20, memory_mib=65_536, disk_mib=262_144),
    )
    sandbox_plan = planned["plan"]["sandbox"]
    probe = probe_backends(
        platform_facts=PlatformFacts(system="Darwin", machine="arm64"),
        which=_which_map({"container": "/opt/homebrew/bin/container"}),
    )
    dry_run = plan_backend_dry_run(sandbox_plan, probe=probe)

    assert dry_run["sandbox_plan_sha256"] == sandbox_plan["plan_sha256"]
    assert dry_run["preferred_backend"] == "apple-container"
    assert dry_run["network_mode"] == "none"
    assert dry_run["filesystem_posture"].startswith("non-host mounts only")
    assert dry_run["resource_budget"] == sandbox_plan["resource_budget"]
    assert dry_run["execution_permitted"] is False
    assert dry_run["selection_is_authorization"] is False
    assert dry_run["selected_candidate"]["backend_id"] == "apple-container"
    assert sandbox_plan["backend"] == "unconfigured"

    with pytest.raises(ProductRefused, match="execution-enabled"):
        plan_backend_dry_run({**sandbox_plan, "execution_permitted": True}, probe=probe)
    with pytest.raises(ProductRefused, match="digest does not match"):
        plan_backend_dry_run({**sandbox_plan, "network_mode": "egress"}, probe=probe)
    with pytest.raises(ProductRefused, match="backend must remain unconfigured"):
        plan_backend_dry_run(_sealed_sandbox_variant(sandbox_plan, backend="docker"), probe=probe)
    with pytest.raises(ProductRefused, match="approved non-host mount source"):
        plan_backend_dry_run(
            _sealed_sandbox_variant(
                sandbox_plan,
                mounts=[{"destination": "/inputs", "mode": "read-only", "source": "/Users/shared"}],
            ),
            probe=probe,
        )


def test_entity_backend_dry_run_requires_active_plan_and_binds_entity_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ProductRefused, match="active apprenticeship plan"):
        plan_backend_dry_run_for_entity(
            store,
            platform_facts=PlatformFacts(system="Linux", machine="x86_64"),
            which=_which_map({"docker": "/usr/bin/docker"}),
        )

    planned = plan_apprenticeship(
        store,
        _spec(),
        HostResources(cpu_cores=20, memory_mib=65_536, disk_mib=262_144),
    )
    dry_run = plan_backend_dry_run_for_entity(
        store,
        platform_facts=PlatformFacts(system="Linux", machine="x86_64"),
        which=_which_map({"docker": "/usr/bin/docker"}),
    )

    assert dry_run["active_apprenticeship"] == "go-systems-foundation"
    assert dry_run["active_apprenticeship_plan_sha256"] == planned["plan"]["plan_sha256"]
    assert dry_run["sandbox_plan_sha256"] == planned["plan"]["sandbox"]["plan_sha256"]
    assert len(dry_run["entity_revision_sha256"]) == 64
    assert dry_run["preferred_backend"] == "docker"
    assert dry_run["execution_permitted"] is False


def test_backend_probe_rejects_malformed_and_inconsistent_inputs() -> None:
    with pytest.raises(ProductRefused, match="nonempty"):
        PlatformFacts(system="", machine="arm64")
    with pytest.raises(ProductRefused, match="daemon health"):
        BackendCandidate(
            backend_id="docker",
            executable_name="docker",
            executable_path="/usr/bin/docker",
            platform_eligible=True,
            executable_discovered=True,
            eligible=True,
            priority=2,
            role="compatibility-fallback",
            safety_posture="test",
            capability_claims=("executable-path-discovery-only",),
            rejection_reasons=(),
            daemon_status="healthy",
        )
    with pytest.raises(ProductRefused, match="fixed backend selection order"):
        BackendCandidate(
            backend_id="docker",
            executable_name="docker",
            executable_path="/usr/bin/docker",
            platform_eligible=True,
            executable_discovered=True,
            eligible=True,
            priority=1,
            role="compatibility-fallback",
            safety_posture="test",
            capability_claims=("executable-path-discovery-only",),
            rejection_reasons=(),
        )

    apple = BackendCandidate(
        backend_id="apple-container",
        executable_name="container",
        executable_path="/usr/bin/container",
        platform_eligible=True,
        executable_discovered=True,
        eligible=True,
        priority=1,
        role="preferred-local-sandbox",
        safety_posture="test",
        capability_claims=("executable-path-discovery-only",),
        rejection_reasons=(),
        selection_status="eligible-not-selected",
    )
    docker = BackendCandidate(
        backend_id="docker",
        executable_name="docker",
        executable_path="/usr/bin/docker",
        platform_eligible=True,
        executable_discovered=True,
        eligible=True,
        priority=2,
        role="compatibility-fallback",
        safety_posture="test",
        capability_claims=("executable-path-discovery-only",),
        rejection_reasons=(),
        selection_status="selected",
    )
    with pytest.raises(ProductRefused, match="selection does not match"):
        BackendProbeResult(
            platform=PlatformFacts(system="Darwin", machine="arm64"),
            candidates=(apple, docker),
            preferred_backend="docker",
            selection_status="recommended",
        )
    with pytest.raises(ProductRefused, match="cover every supported backend candidate"):
        BackendProbeResult(
            platform=PlatformFacts(system="Linux", machine="x86_64"),
            candidates=(docker,),
            preferred_backend="docker",
            selection_status="recommended",
        )


def test_product_cli_and_legacy_dispatch_are_isolated(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    entity_root = tmp_path / "cli-engineer.substrate"
    product_main(["packs"])
    assert any(pack["name"] == "media" for pack in json.loads(capsys.readouterr().out)["packs"])
    product_main(
        [
            "init",
            str(entity_root),
            "--entity-id",
            "cli-engineer",
            "--specialty",
            "CLI engineering",
            "--pack",
            "engineering",
            "--organ",
            "primary-reasoning",
        ]
    )
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["entity"]["entity_id"] == "cli-engineer"
    assert initialized["entity"]["organ_requirements"][0]["organ_id"] == "primary-reasoning"

    product_main(["status", str(entity_root)])
    assert json.loads(capsys.readouterr().out)["valid"] is True

    product_main(
        [
            "plan-apprenticeship",
            str(entity_root),
            "--name",
            "cli-foundation",
            "--objective",
            "Plan a bounded coding curriculum",
            "--evaluator",
            "hidden-cli-suite",
            "--host-cpu-cores",
            "8",
            "--host-memory-mib",
            "16384",
            "--host-disk-mib",
            "65536",
            "--worker-cpu-cores",
            "2",
            "--worker-memory-mib",
            "4096",
            "--worker-disk-mib",
            "8192",
            "--maximum-workers",
            "4",
            "--wall-clock-minutes",
            "120",
            "--source-file-root",
            "/approved",
        ]
    )
    assert json.loads(capsys.readouterr().out)["plan"]["worker_plan"]["concurrent_workers"] == 3

    product_main(
        [
            "plan-source",
            str(entity_root),
            "--source-uri",
            "file:///approved/lesson.txt",
            "--modality",
            "text",
            "--access-status",
            "user-provided",
            "--declared-rights",
            "operator-provided training material",
            "--retrieval-mode",
            "import",
        ]
    )
    assert json.loads(capsys.readouterr().out)["plan"]["execution_permitted"] is False

    product_main(["backends"])
    backends_output = json.loads(capsys.readouterr().out)
    assert backends_output["execution_permitted"] is False
    assert backends_output["selection_is_dry_run"] is True
    assert backends_output["selection_is_authorization"] is False
    assert backends_output["probe"]["probe_method"] == "executable-path-discovery-only"
    assert {candidate["backend_id"] for candidate in backends_output["probe"]["candidates"]} == {
        "apple-container",
        "docker",
    }

    product_main(["dry-run-backend", str(entity_root)])
    dry_run_output = json.loads(capsys.readouterr().out)
    assert dry_run_output["execution_permitted"] is False
    assert dry_run_output["selection_is_authorization"] is False
    assert dry_run_output["network_mode"] == "none"
    assert dry_run_output["active_apprenticeship"] == "cli-foundation"
    assert len(dry_run_output["sandbox_plan_sha256"]) == 64

    root_cli.main(["product", "validate", str(entity_root)])
    assert json.loads(capsys.readouterr().out)["valid"] is True

    root_cli.main(["product", "backends"])
    assert json.loads(capsys.readouterr().out)["execution_permitted"] is False

    called: list[list[str]] = []
    monkeypatch.setattr(sandbox, "main", lambda argv: called.append(argv))
    root_cli.main(["sandbox", "status"])
    assert called == [["status"]]
