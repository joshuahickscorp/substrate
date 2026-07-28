from __future__ import annotations

from types import SimpleNamespace

import pytest

from substrate import v5pilot


@pytest.fixture(scope="module")
def reports() -> dict:
    return v5pilot.run()


def _no_true_activation(value: object) -> bool:
    if isinstance(value, dict):
        return all(key != "activation" or child is False for key, child in value.items()) and all(_no_true_activation(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_no_true_activation(child) for child in value)
    return True


def test_moderate_pilot_executes_the_exact_continuing_entity_sequence(reports: dict) -> None:
    pilot = reports["pilot"]
    assert pilot["independent_histories"] == 16
    assert 8 <= pilot["focused_arm_count"] <= 14
    assert 25_000 <= pilot["episodes"] <= 100_000
    assert pilot["modality_count"] >= 6
    assert pilot["model_equivalent_count"] >= 6
    assert pilot["continuing_entity_sequence"] == list(v5pilot.INTEGRATED_SEQUENCE)
    assert all(row["sequence_exact"] and row["entity_continued"] for row in pilot["trajectories"])
    assert all(row["all_stages_pass"] and row["transfer"] and row["retention"] for row in pilot["trajectories"])
    assert pilot["analysis"]["all_pass"]
    assert pilot["passed"]


def test_destructive_rehearsal_injects_and_contains_the_full_matrix(reports: dict) -> None:
    failures = reports["failures"]
    assert failures["scenario_count"] == 15
    assert tuple(row["failure"] for row in failures["scenarios"]) == v5pilot.FAILURE_SCENARIOS
    assert all(row["injected"] and row["detected"] and row["recovered"] for row in failures["scenarios"])
    assert all(row["contained"] for row in failures["scenarios"])
    assert failures["live_processes_signaled"] == 0
    assert failures["live_processes_modified"] == 0
    assert failures["injected_failure_rate"] == 1.0
    assert failures["failure_rate"] == 0.0
    assert failures["uncontained_failure_rate"] == 0.0
    assert failures["all_pass"]


def test_resources_download_kernel_and_admission_are_measured(reports: dict) -> None:
    resources = reports["resources"]
    assert resources["observed_worker_count"] >= 1
    assert resources["worker_tasks_completed"] == 16
    assert resources["peak_rss_mib"] > 0
    assert resources["mean_checkpoint_bytes"] > 0
    assert resources["mean_checkpoint_elapsed_ns"] > 0
    assert resources["mean_model_startup_elapsed_ns"] > 0
    assert resources["hawking_coexistence"]["passed"]
    assert reports["download_benchmark"]["digest_verified"]
    assert reports["download_benchmark"]["preprocessing_exact"]
    assert reports["kernel"]["selected"] == "candidate_d_hybrid_explicit_latent"
    assert reports["kernel"]["passed"]
    assert reports["admission"]["admitted"]
    assert reports["admission"]["principal_launch_authorized"]
    assert all(reports["admission"]["gates"].values())
    assert _no_true_activation(reports)


def test_admission_binds_exact_authority_and_frozen_input_digests(
    reports: dict,
) -> None:
    admission = reports["admission"]
    bindings = admission["authority_bindings"]

    assert set(bindings) == set(v5pilot.ADMISSION_AUTHORITY_PATHS)
    assert all(row["path"] == v5pilot.ADMISSION_AUTHORITY_PATHS[identity] and len(row["sha256"]) == 64 for identity, row in bindings.items())
    assert admission["gates"]["authority_bindings_complete"]
    assert admission["model_registry_digest"] == bindings["model"]["sha256"]
    assert admission["corpus_catalog_digest"] == bindings["corpus"]["sha256"]
    assert len(admission["configuration_digest"]) == 64
    assert len(admission["source_commit"]) == 40
    assert len(admission["source_digest"]) == 64

    writer = v5pilot._v5io()
    generated = {
        "pilot": reports["pilot"],
        "failure": reports["failures"],
        "kernel": v5pilot._kernel_documents(reports["kernel"])["SUBSTRATE_V5_KERNEL_SELECTION.json"],
    }
    for identity, document in generated.items():
        sealed = writer.sealed_document(document)
        assert bindings[identity]["sha256"] == sealed["sha256"]
    for identity in ("configuration", "model", "corpus"):
        current = writer.load_json(writer.ROOT / v5pilot.ADMISSION_AUTHORITY_PATHS[identity])
        assert bindings[identity]["sha256"] == current["sha256"]


def test_publication_is_explicit_and_covers_all_authorities(
    reports: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, dict, bool]] = []
    fake_io = SimpleNamespace(
        seal=lambda name, document, *, artifact=False: writes.append((name, document, artifact)),
    )
    monkeypatch.setattr(v5pilot, "_v5io", lambda: fake_io)

    published = v5pilot.publish_authorities(reports)

    assert tuple(published) == v5pilot.AUTHORITY_NAMES
    assert {name for name, _, _ in writes} == set(v5pilot.AUTHORITY_NAMES)
    assert all(not artifact for _, _, artifact in writes)
    assert all(_no_true_activation(document) for _, document, _ in writes)
