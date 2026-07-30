from __future__ import annotations

import plistlib
from pathlib import Path

from substrate import sandbox
from substrate import sandbox_campaign as campaign
from substrate import sandbox_config as C
from substrate import sandbox_execution as execution
from substrate.final_revision_io import digest


def _preflight_fixture(*, free: int, total: int = 1000 * C.GIB) -> dict:
    floor = C.disk_floor_bytes(total)
    return campaign.authority(
        "SUBSTRATE_SANDBOX_PREFLIGHT",
        {
            "disk": {
                "capacity_bytes": total,
                "available_bytes": free,
                "required_floor_bytes": floor,
                "floor_pass": free >= floor,
                "core_start_deficit_bytes": max(
                    0, floor + C.CORE_MINIMUM_ACQUISITION_BYTES - free
                ),
            },
            "bounded_repair": {"alternate_mounted_core_volume_found": False},
            "admission": {
                "terminal_outcome_c_admitted": free < floor,
                "critical_blocker": "protected_disk_floor" if free < floor else None,
            },
            "historical_identity": {"preserved": True},
            "docker": {"server_available": False},
            "tools": {
                "vmrun": {"available": False},
                "emulator": {"available": False},
                "adb": {"available": False},
            },
            "model_and_data_credentials": {},
        },
        status="fixture",
    )


def test_r2_constitution_preserves_parent_and_claim_boundary() -> None:
    assert C.ACTIVATION is False
    assert C.UNQUALIFIED_NOUS is False
    assert C.PARENT_SELECTED_MATERIAL == "L1_associative_monolith"
    assert C.PARENT_CLASSIFICATION == "cognitive_material_genesis_ii_complete"
    assert C.PARENT_STATUS == "compositional_advantage_unproven"
    assert C.PARENT_READINESS == "tangible_sandbox_ready"
    assert C.OUTCOMES["C"]["classification"] == "terminal_tangible_sandbox_null"
    assert C.SESOI == 0.05


def test_disk_floor_is_larger_of_50_gib_and_twenty_percent() -> None:
    assert C.disk_floor_bytes(100 * C.GIB) == 50 * C.GIB
    assert C.disk_floor_bytes(1000 * C.GIB) == 200 * C.GIB


def test_core_admission_requires_floor_plus_reservation() -> None:
    total = 1000 * C.GIB
    floor = C.disk_floor_bytes(total)
    refused = campaign.acquisition_plan(
        _preflight_fixture(free=floor - 1, total=total),
        persist=False,
    )
    assert refused["safe_to_start"] is False
    assert refused["disk"]["usable_above_floor_bytes"] == 0
    assert refused["refusals"] == [
        "protected_disk_floor",
        "core_acquisition_reservation",
    ]

    admitted = campaign.acquisition_plan(
        _preflight_fixture(
            free=floor
            + max(
                C.CORE_MINIMUM_ACQUISITION_BYTES,
                sum(row["bytes"] for row in C.CORE_BINARY_ASSETS),
            ),
            total=total,
        ),
        persist=False,
    )
    assert admitted["safe_to_start"] is True


def test_outcome_c_requires_a_terminal_preflight_failure() -> None:
    total = 1000 * C.GIB
    refused = _preflight_fixture(free=1 * C.GIB, total=total)
    classification = campaign._classify(refused)
    assert classification["outcome"] == "C"
    assert classification["principal_launched"] is False
    assert classification["H_T12"]["status"] == "not_tested"
    assert classification["external_activation"] is False


def test_authorities_are_hash_sealed_and_inactive() -> None:
    document = campaign.authority("TEST", {"value": 1}, status="test")
    claimed = document.pop("sha256")
    assert claimed == digest(document)
    assert document["activation"] is False
    assert document["unqualified_nous"] is False


def test_acquisition_state_machine_and_four_pools_are_complete() -> None:
    assert set(C.ACQUISITION_POOLS) == {
        "network",
        "hash",
        "extraction",
        "preprocessing",
    }
    assert C.ACQUISITION_STATES[-3:] == ("QUARANTINED", "GATED", "REFUSED")
    assert len(set(C.ACQUISITION_STATES)) == len(C.ACQUISITION_STATES)
    selected_bytes = sum(row["bytes"] for row in C.CORE_BINARY_ASSETS)
    assert selected_bytes >= C.CORE_MINIMUM_ACQUISITION_BYTES
    assert selected_bytes <= C.CORE_PREFERRED_ACQUISITION_BYTES
    assert {row["source_id"] for row in C.CORE_BINARY_ASSETS} == {
        "fsd50k",
        "librispeech",
    }


def test_required_arms_include_strengthened_practical_controls() -> None:
    assert len(C.REQUIRED_ARMS) == 13
    assert "project_state_database" in C.REQUIRED_ARMS
    assert "best_of_n_direct_model" in C.REQUIRED_ARMS
    assert "oracle" in C.REQUIRED_ARMS


def test_stsc_schema_has_separate_roots_and_all_splits() -> None:
    schema = campaign._stsc_schema()
    assert schema["version"] == "1.0.0-r2"
    assert schema["roots"] == [
        "builder_visible",
        "executor_visible",
        "evaluator_only",
        "publication_safe",
    ]
    assert len(schema["splits"]) == 7
    assert len(schema["families"]) == 16
    assert schema["materialized"] is False


def test_integrity_canaries_use_real_temporary_bytes() -> None:
    checks = campaign._checksum_canaries()
    assert checks == {
        "C02_checksum_mismatch_detected": True,
        "C03_partial_download_resumes": True,
        "C04_duplicate_download_avoided": True,
    }


def test_official_source_catalog_has_unique_ids_and_pins() -> None:
    ids = [row["source_id"] for row in C.OFFICIAL_SOURCES]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 17
    assert all(row["official_url"].startswith("https://") for row in C.OFFICIAL_SOURCES)
    assert all(row["selected_revision"] for row in C.OFFICIAL_SOURCES)
    osworld = next(row for row in C.OFFICIAL_SOURCES if row["source_id"] == "osworld_v2")
    assert osworld["selected_revision"] == "v2026.06.24"


def test_common_voice_and_fsd50k_license_boundaries_are_explicit() -> None:
    common_voice = next(
        row for row in C.OFFICIAL_SOURCES if row["source_id"] == "common_voice"
    )
    fsd50k = next(row for row in C.OFFICIAL_SOURCES if row["source_id"] == "fsd50k")
    assert common_voice["redistribution_class"] == "local_evaluation_only"
    assert "no_rehosting" in common_voice["access"]
    assert fsd50k["redistribution_class"] == "filter_to_CC0_and_CC-BY_with_attribution"


def test_ungated_access_is_not_misclassified_as_gated() -> None:
    assert campaign._access_is_gated("ungated") is False
    assert campaign._access_is_gated("ungated_code_and_dataset") is False
    assert campaign._access_is_gated("gated_task_classes_and_assets") is True
    assert campaign._access_is_gated("Kaggle_credentials_and_terms_required") is True


def test_required_deliverable_names_are_unique() -> None:
    assert len(C.REQUIRED_DELIVERABLES) == len(set(C.REQUIRED_DELIVERABLES))
    assert "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json" in C.REQUIRED_DELIVERABLES
    assert "SUBSTRATE_SANDBOX_TERMINAL_REPORT.md" in C.REQUIRED_DELIVERABLES


def test_adapter_contract_never_exposes_evaluator_root() -> None:
    contract = campaign._adapter_contract()
    assert contract["candidate_can_read_evaluator_only"] is False
    assert contract["implemented_environment_adapters"] == []


def test_not_run_receipt_does_not_counterfeit_a_null_effect() -> None:
    document = campaign._not_run("TEST", reason="preflight")
    assert document["status"] == "not_run"
    assert document["principal_units"] == 0
    assert "effect" not in document


def test_package_configuration_is_machine_readable() -> None:
    configuration = C.configuration()
    assert configuration["eta"]["planning_hours"] == 36
    assert configuration["eta"]["expected_hours"] == [32, 44]
    assert configuration["required_public_floor"]["software_engineering"].endswith(
        "minimum 25"
    )
    assert configuration["activation"] is False


def test_publication_paths_stay_inside_repository() -> None:
    assert campaign.EVIDENCE.is_relative_to(campaign.ROOT)
    assert campaign.PUBLICATION.is_relative_to(campaign.ROOT)
    assert campaign.CORPUS.is_relative_to(campaign.ROOT)
    assert isinstance(campaign.ROOT, Path)


def test_admitted_custom_design_meets_counts_and_history_floors() -> None:
    assert execution.SPLIT_COUNTS["principal"] == 1024
    assert execution.SPLIT_COUNTS["replication"] >= (
        execution.SPLIT_COUNTS["principal"] + 2
    ) // 3
    assert execution.SPLIT_COUNTS["hidden_composition"] >= (
        execution.SPLIT_COUNTS["principal"] + 2
    ) // 3
    assert execution.HISTORY_COUNTS == {
        "principal": 64,
        "replication": 24,
        "hidden_composition": 24,
    }
    assert sum(execution.SPLIT_COUNTS.values()) == 2000


def test_preoutcome_review_has_48_distinct_roles_in_all_cells() -> None:
    assert len(execution.GROK_ROLES) == 48
    assert len(set(execution.GROK_ROLES)) == 48
    assert len({role.rsplit("-", 1)[0] for role in execution.GROK_ROLES}) == 12


def test_strong_project_database_is_not_weakened() -> None:
    assert set(C.REQUIRED_ARMS) == execution.DIRECT_ARMS
    assert "project_state_database" in execution.PERSISTENT_ARMS
    assert "L1_full" in execution.PERSISTENT_ARMS
    assert "L1_no_development" not in execution.PERSISTENT_ARMS
    assert "fresh_model" not in execution.PERSISTENT_ARMS


def test_webarena_subset_is_preregistered_across_four_sites() -> None:
    assert set(execution.WEBARENA_ENDPOINTS) == {
        "shopping",
        "shopping_admin",
        "reddit",
        "gitlab",
    }


def test_longitudinal_schedule_meets_the_r2_minimums_before_launch() -> None:
    assert C.CONTINUITY_SCHEDULE_VERSION == "2.0.0"
    assert [row[0] for row in C.LONGITUDINAL_SCHEDULE] == list(range(0, 25, 3))
    assert len(C.LONGITUDINAL_SCHEDULE) >= C.LONGITUDINAL_MINIMUMS["checkpoints"]
    events = [row[1] for row in C.LONGITUDINAL_SCHEDULE]
    activities = [row[2] for row in C.LONGITUDINAL_SCHEDULE]
    assert sum(event.startswith("restart_") for event in events) >= C.LONGITUDINAL_MINIMUMS[
        "process_restarts"
    ]
    assert "model_replacement" in events
    assert "restart_2_tool_body_change" in events
    assert "sensor_interruption" in events
    assert sum(event.startswith("human_correction") for event in events) >= 2
    assert sum(activity.startswith("return_old_work") for activity in activities) >= 2
    assert sum("requires_earlier_history" in activity for activity in activities) >= 2


def test_supervised_longitudinal_job_is_one_shot_and_receipt_bound() -> None:
    manifest = execution.RUNS / execution.SUPERVISION_ROOT_NAME / "r2-test" / "manifest.json"
    job = execution._launchd_job_plist(
        label="org.substrate.tangible-sandbox-r2.r2-test",
        manifest_path=manifest,
        stdout_path=manifest.parent / "stdout.log",
        stderr_path=manifest.parent / "stderr.log",
    )
    round_tripped = plistlib.loads(plistlib.dumps(job))
    assert execution.SUPERVISION_VERSION == "1.0.0"
    assert round_tripped["KeepAlive"] is False
    assert round_tripped["RunAtLoad"] is False
    assert round_tripped["AbandonProcessGroup"] is False
    assert round_tripped["EnvironmentVariables"]["SUBSTRATE_LONGITUDINAL_SUPERVISOR"] == "launchd"
    assert round_tripped["ProgramArguments"][-2:] == [
        "--supervision-manifest",
        str(manifest),
    ]


def test_supervision_cli_requires_an_explicit_manifest_for_the_worker() -> None:
    commands = sandbox.parser()._subparsers._group_actions[0].choices
    assert {
        "launch-longitudinal",
        "supervised-longitudinal",
        "longitudinal-supervision-status",
        "seal-continuity-supervision-repair",
    }.issubset(commands)
    arguments = sandbox.parser().parse_args(
        ["supervised-longitudinal", "--supervision-manifest", "/tmp/manifest.json"]
    )
    assert arguments.supervision_manifest == Path("/tmp/manifest.json")


def test_terminal_refusal_requires_an_actual_prelaunch_refusal() -> None:
    common = {"activation": False}
    checks = execution._continuity_refusal_checks(
        classification={"outcome": "C", **common},
        preflight={"fresh_launch_admitted": False, **common},
        failure={
            "classification": "invalid_terminal_trace",
            "trace_is_not_terminal_evidence": True,
            **common,
        },
        root_cause={"fresh_lane_permitted": False, **common},
        longitudinal_result={
            "status": "refused_before_launch",
            "continuity_passing": False,
            "actual_wall_hours": 0.0,
            **common,
        },
        clean_clone_result={"all_pass": True, **common},
    )
    assert all(checks.values())

    counterfeit = execution._continuity_refusal_checks(
        classification={"outcome": "C", **common},
        preflight={"fresh_launch_admitted": False, **common},
        failure={
            "classification": "invalid_terminal_trace",
            "trace_is_not_terminal_evidence": True,
            **common,
        },
        root_cause={"fresh_lane_permitted": False, **common},
        longitudinal_result={
            "status": "refused_before_launch",
            "continuity_passing": True,
            "actual_wall_hours": 0.0,
            **common,
        },
        clean_clone_result={"all_pass": True, **common},
    )
    assert counterfeit["longitudinal_not_counterfeited"] is False


def test_terminal_refusal_cli_is_explicit() -> None:
    commands = sandbox.parser()._subparsers._group_actions[0].choices
    assert "publish-continuity-refusal" in commands
