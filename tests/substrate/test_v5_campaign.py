from __future__ import annotations

from types import SimpleNamespace

import pytest

from substrate import v5campaign
from substrate import v5config as C


def _no_true_activation(value: object) -> bool:
    if isinstance(value, dict):
        return all(key != "activation" or child is False for key, child in value.items()) and all(_no_true_activation(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_no_true_activation(child) for child in value)
    return True


def test_v5_configuration_exactly_encodes_the_master_plan_surface() -> None:
    assert len(C.HYPOTHESES) == 15
    assert tuple(C.HYPOTHESES) == tuple(f"H_M{index}" for index in range(1, 16))
    assert len(C.PHASES) == 20
    assert len(C.ARMS) == 18
    assert len(C.MODALITIES) >= 8
    assert len(C.MODEL_ROLES) == 18
    assert "independent_performer" in C.MODEL_ROLES
    assert tuple(f"C{index:02d}" for index in range(1, 51)) == C.CANARIES
    assert C.CLASSIFICATIONS[-1] == "multimodal_nous_ready_for_review"
    assert C.CLAIM_BOUNDARY["unqualified_nous"] is False
    assert C.PRE_TAG == "substrate-v5-pre-sensorium"
    assert C.READY_TAG == "substrate-v5-sensorium-ready"
    assert C.TERMINAL_TAG == "substrate-v5-terminal"
    assert C.IMPLEMENTATION_BRANCH == "agent/substrate-v5-sensorium-model-fabric"
    assert _no_true_activation(C.configuration())


def test_v5_candidate_ladders_are_bounded() -> None:
    assert set(C.CANDIDATE_LADDERS) == {
        "kernel",
        "video",
        "spatial_3d",
        "cross_modal_binding",
        "model_routing",
        "active_perception",
    }
    assert len(C.CANDIDATE_LADDERS["kernel"]) == 6
    assert all(len(ladder) <= 6 for ladder in C.CANDIDATE_LADDERS.values())
    assert "extended_v4_reference" in C.CANDIDATE_LADDERS["kernel"]
    assert "alternative_non_neural_core_if_concrete" in C.CANDIDATE_LADDERS["kernel"]


def test_deliverables_include_extended_master_plan_authorities() -> None:
    required = {
        "SUBSTRATE_V5_PREFLIGHT.json",
        "SUBSTRATE_V5_V1_V2_V3_V4_IMMUTABILITY.json",
        "SUBSTRATE_V5_MODEL_RUNTIME_MATRIX.json",
        "SUBSTRATE_V5_REPRESENTATIONAL_HIERARCHY.json",
        "SUBSTRATE_V5_SENSOR_FUSION.json",
        "SUBSTRATE_V5_CURRICULUM_AUTHORITY.json",
        "SUBSTRATE_V5_SIMULATOR_VALIDATION.json",
        "SUBSTRATE_V5_DISASTER_RECOVERY.json",
        "SUBSTRATE_V5_COMPLETION_SCORECARD.json",
        "SUBSTRATE_V5_TERMINAL_REPORT.md",
    }
    assert required <= set(C.DELIVERABLES)
    assert len(C.DELIVERABLES) == len(set(C.DELIVERABLES))


def test_immutability_requires_annotated_remote_exact_tags_and_local_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = {tag: {"tag_object": f"object-{tag}", "peeled_commit": f"commit-{tag}"} for tag in C.PRIOR_TAGS}
    monkeypatch.setattr(v5campaign, "_remote_tag_refs", lambda: remote)
    monkeypatch.setattr(
        v5campaign,
        "_tag_snapshot",
        lambda tag, _remote: {
            "name": tag,
            "tag_object": f"object-{tag}",
            "peeled_commit": f"commit-{tag}",
            "object_type": "tag",
            "annotated": True,
            "remote_tag_object": f"object-{tag}",
            "remote_peeled_commit": f"commit-{tag}",
            "tag_object_matches_remote": True,
            "peeled_commit_matches_remote": True,
        },
    )
    monkeypatch.setattr(
        v5campaign,
        "_git",
        lambda *arguments: "\n".join(C.PRIOR_TAGS) if arguments[:2] == ("tag", "--list") else "",
    )
    monkeypatch.setattr(
        v5campaign,
        "_tree_integrity",
        lambda tag, roots: {
            "tag": tag,
            "roots": roots,
            "tag_tree_hash": "same",
            "local_tree_hash": "same",
            "tree_hashes_match": True,
            "byte_identical": True,
        },
    )
    monkeypatch.setattr(
        v5campaign,
        "_seal_validation",
        lambda version: {"version": version, "all_valid": True},
    )
    monkeypatch.setattr(
        v5campaign,
        "_classification_snapshot",
        lambda: {version: {"classification": classification, "preserved": True} for version, classification in C.PRIOR_CLASSIFICATIONS.items()},
    )

    result = v5campaign.immutability()

    assert result["all_pass"]
    assert result["failed"] == []
    assert result["tag_authority"]["expected"] == list(C.PRIOR_TAGS)
    assert all(tree["tree_hashes_match"] for tree in result["trees"].values())
    assert result["activation"] is False


def test_inventory_is_explicitly_read_only_and_detects_no_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v5campaign,
        "_process_snapshots",
        lambda: {
            "hawking": {
                "processes": [],
                "active_process_count": 0,
                "observation_only": True,
                "signals_sent": 0,
                "processes_modified": 0,
                "controllers_modified": 0,
                "mps_adopted": False,
            },
            "v5_workers": {
                "processes": [],
                "active_process_count": 0,
                "observation_only": True,
                "signals_sent": 0,
            },
        },
    )
    monkeypatch.setattr(
        v5campaign,
        "_v5_namespace_snapshot",
        lambda: {
            "families": {},
            "principal_files": [],
            "principal_exists": False,
        },
    )
    monkeypatch.setattr(v5campaign, "_tool_snapshot", lambda: {"git": {"available": True}})
    monkeypatch.setattr(v5campaign, "_hardware_snapshot", lambda: {"logical_cores": 8})
    monkeypatch.setattr(v5campaign, "_resource_snapshot", lambda: {"disk_available_gib": 100.0})
    monkeypatch.setattr(v5campaign, "_network_snapshot", lambda: {"active_bandwidth_probe_performed": False})
    monkeypatch.setattr(v5campaign, "_model_snapshot", lambda: {"roots": [], "inventory_only": True})
    monkeypatch.setattr(v5campaign, "_corpus_snapshot", lambda: {"roots": [], "inventory_only": True})

    result = v5campaign.inventory()

    assert result["read_only"]
    assert result["files_written"] == 0
    assert result["processes_modified"] == 0
    assert result["v5_principal"]["pre_existing"] is False
    assert result["activation"] is False


def test_worktree_cleanliness_allows_only_declared_roots() -> None:
    allowed = "\0".join(
        (
            " M artifacts/substrate/v5/SUBSTRATE_V5_PREFLIGHT.json",
            "?? artifacts/substrate/v5/.objects/aa/object.json",
            "",
        )
    )
    clean = v5campaign.worktree_cleanliness(
        v5campaign.PREFLIGHT_GENERATED_ROOTS,
        status_output=allowed,
    )

    assert clean["clean_except_allowed_roots"]
    assert clean["undeclared_dirty_paths"] == []
    assert clean["activation"] is False

    undeclared = allowed + " M src/substrate/v5campaign.py\0"
    dirty = v5campaign.worktree_cleanliness(
        v5campaign.PREFLIGHT_GENERATED_ROOTS,
        status_output=undeclared,
    )

    assert not dirty["clean_except_allowed_roots"]
    assert dirty["undeclared_dirty_paths"] == ["src/substrate/v5campaign.py"]

    runtime_only = "\0".join(
        (
            " M evidence/substrate/v5/SUBSTRATE_V5_PRINCIPAL_AUTHORITY.json",
            "?? runs/substrate/v5/principal/units/unit.json",
            "?? cache/substrate/v5/features/object.bin",
            "",
        )
    )
    runtime = v5campaign.worktree_cleanliness(
        v5campaign.PRINCIPAL_RUNTIME_ROOTS,
        status_output=runtime_only,
    )
    assert runtime["clean_except_allowed_roots"]

    frozen_config_drift = runtime_only + (" M configs/substrate/v5/frozen_configuration.json\0")
    frozen = v5campaign.worktree_cleanliness(
        v5campaign.PRINCIPAL_RUNTIME_ROOTS,
        status_output=frozen_config_drift,
    )
    assert not frozen["clean_except_allowed_roots"]
    assert frozen["undeclared_dirty_paths"] == ["configs/substrate/v5/frozen_configuration.json"]


def test_preflight_fails_on_undeclared_dirty_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = "a" * 40
    local_inventory = {
        "resources": {"disk_available_gib": 100.0},
        "processes": {
            "hawking": {
                "observation_only": True,
                "signals_sent": 0,
                "processes_modified": 0,
                "controllers_modified": 0,
            }
        },
        "v5_principal": {"pre_existing": False, "workers": []},
        "read_only": True,
        "files_written": 0,
        "processes_modified": 0,
    }
    integrity = {
        "all_pass": True,
        "tag_authority": {"tags": {C.TERMINAL_TAGS["v4"]: {"peeled_commit": base}}},
    }
    monkeypatch.setattr(v5campaign, "_remote_tag_refs", lambda: {})
    monkeypatch.setattr(
        v5campaign,
        "_tag_snapshot",
        lambda _tag, _remote: {
            "annotated": True,
            "tag_object_matches_remote": True,
            "peeled_commit_matches_remote": True,
            "peeled_commit": base,
        },
    )
    monkeypatch.setattr(
        v5campaign,
        "_ref_or_none",
        lambda reference: {
            "refs/heads/main": base,
            "refs/remotes/origin/main": base,
            "HEAD": "b" * 40,
        }.get(reference),
    )
    monkeypatch.setattr(v5campaign, "_remote_ref", lambda _reference: base)

    def optional(arguments: list[str], **_kwargs: object) -> dict:
        if arguments[:3] == ["git", "branch", "--show-current"]:
            return {"returncode": 0, "stdout": C.IMPLEMENTATION_BRANCH}
        if arguments[:3] == ["git", "worktree", "list"]:
            return {"returncode": 0, "stdout": "worktree /repo\n"}
        if arguments[:3] == ["git", "merge-base", "--is-ancestor"]:
            return {"returncode": 0, "stdout": ""}
        raise AssertionError(arguments)

    monkeypatch.setattr(v5campaign, "_optional_command", optional)
    monkeypatch.setattr(
        v5campaign,
        "worktree_cleanliness",
        lambda _roots: {
            "entries": [{"status": " M", "path": "src/substrate/v5.py"}],
            "clean_except_allowed_roots": False,
            "undeclared_dirty_paths": ["src/substrate/v5.py"],
            "activation": False,
        },
    )

    report = v5campaign.preflight(
        inventory_snapshot=local_inventory,
        integrity_snapshot=integrity,
    )

    assert not report["all_pass"]
    assert "worktree_clean_except_preflight_authorities" in report["failed"]
    assert report["entry"]["cleanliness"]["undeclared_dirty_paths"] == ["src/substrate/v5.py"]


def test_seal_preflight_publishes_all_nine_entry_authorities_via_lazy_v5io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, bool]] = []
    fake_io = SimpleNamespace(
        ACTIVATION=False,
        seal=lambda name, document, *, artifact=False: writes.append((name, artifact)),
    )
    local_inventory = {
        "resources": {"disk_available_gib": 100.0},
        "storage": {},
        "network": {},
        "tools": {},
        "hardware": {},
        "models": {},
        "corpora": {},
        "processes": {
            "hawking": {
                "observation_only": True,
                "signals_sent": 0,
                "processes_modified": 0,
                "controllers_modified": 0,
            },
            "v5_workers": {"processes": []},
        },
    }
    integrity = {"all_pass": True, "activation": False}
    entry = {"all_pass": True, "activation": False}
    monkeypatch.setattr(v5campaign, "_v5io", lambda: fake_io)
    monkeypatch.setattr(v5campaign, "inventory", lambda: local_inventory)
    monkeypatch.setattr(v5campaign, "immutability", lambda: integrity)
    monkeypatch.setattr(v5campaign, "preflight", lambda **kwargs: entry)

    result = v5campaign.seal_preflight()

    assert len(result["sealed"]) == 9
    assert {name for name, _ in writes} == {
        "SUBSTRATE_V5_PREFLIGHT.json",
        "SUBSTRATE_V5_V1_V2_V3_V4_IMMUTABILITY.json",
        "SUBSTRATE_V5_HAWKING_COEXISTENCE.json",
        "SUBSTRATE_V5_LOCAL_CAPABILITY_INVENTORY.json",
        "SUBSTRATE_V5_STORAGE_AND_NETWORK_PLAN.json",
        "SUBSTRATE_V5_SPEED_CONSTITUTION.json",
        "SUBSTRATE_V5_DOWNLOAD_AUTHORITY.json",
        "SUBSTRATE_V5_RESOURCE_TELEMETRY.json",
        "SUBSTRATE_V5_PERFORMANCE_LEDGER.json",
    }
    assert all(artifact for _, artifact in writes)
    assert result["activation"] is False


def test_freeze_uses_v5io_and_keeps_every_authority_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed: list[tuple[str, dict]] = []
    configs: list[tuple[str, dict]] = []
    fake_io = SimpleNamespace(
        ACTIVATION=False,
        seal=lambda name, document, *, artifact=False: sealed.append((name, document)),
        config_json=lambda name, document: configs.append((name, document)),
    )
    monkeypatch.setattr(v5campaign, "_v5io", lambda: fake_io)

    result = v5campaign.freeze()

    assert len(sealed) == 6
    assert {name for name, _ in configs} == {
        "frozen_configuration.json",
        "candidate_ladders.json",
    }
    assert all(_no_true_activation(document) for _, document in sealed + configs)
    assert result["configuration"]["configuration_digest"]
    assert result["activation"] is False
