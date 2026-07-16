from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _builder() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/build_generation1_program.py"
    specification = importlib.util.spec_from_file_location("generation1_program_builder", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_v2_campaign_profile_freezes_independent_execution_count() -> None:
    builder = _builder()
    config = builder.load_config(builder.CORPUS_CONFIG)

    profile = builder._campaign_profile(config)

    assert profile["mode_counts"] == {"fixed": 1, "mechanics": 2, "varied": 125}
    assert profile["evidence_class_counts"] == {
        "fixed_case_noninferential": 1,
        "inferential": 125,
        "mechanics_noninferential": 2,
    }
    assert profile["effective_executions"] == 3003
    assert profile["execute_once_ids"] == [
        "mop_p4_capability_density_screen",
        "mop_p5_context_capability",
        "s5_code_stability",
    ]

    drifted = copy.deepcopy(config)
    drifted["seed_authority"]["mechanics_only_experiment_ids"].pop()
    with pytest.raises(ValueError, match="seed-mode census drifted"):
        builder._campaign_profile(drifted)


def test_v2_program_stays_sealed_after_live_source_evolution() -> None:
    builder = _builder()
    with pytest.raises(ValueError, match="resource_canary.*source authority drifted"):
        builder.build_program()
    program = json.loads(builder.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    core = {key: value for key, value in program.items() if key != "program_sha256"}
    assert program["program_sha256"] == builder.canonical_sha256(core)

    assert program["program_id"] == "generation1-empirical-cognitive-corpus-v2"
    assert program["program_root"] == (
        "runs/generation1/generation1-empirical-cognitive-corpus-v2"
    )
    capsules = program["capsules"]
    seed_capsules = [row for row in capsules if row["id"].startswith("g1_cognitive_seed_")]
    assert len(seed_capsules) == 24
    worker_counts: set[int] = set()
    memory_limits: set[float] = set()
    for index, capsule in enumerate(seed_capsules):
        if index:
            assert capsule["depends_on"] == [seed_capsules[index - 1]["id"]]
        workers = int(capsule["command"][capsule["command"].index("--max-workers") + 1])
        worker_counts.add(workers)
        memory_limits.add(float(capsule["resources"]["estimated_unified_memory_gb"]))
        assert capsule["resources"]["cpu_cores"] == workers
        assert "3003 actual executions" in capsule["resources"]["resource_basis"]
        assert "0.8 GB" not in capsule["resources"]["resource_basis"]
        assert capsule["artifacts"][0]["schema"] == "mop-generation1-cognitive-seed/v2"
        assert any(
            authority["path"] == "proof/GENERATION1_RESOURCE_CANARY.json"
            for authority in capsule["authorities"]
        )
    assert len(worker_counts) == 1
    assert len(memory_limits) == 1
    max_workers = next(iter(worker_counts))
    memory_gb = next(iter(memory_limits))

    by_id = {row["id"]: row for row in capsules}
    aggregate = by_id["g1_cognitive_corpus_aggregate"]
    assert aggregate["depends_on"] == [row["id"] for row in seed_capsules]
    assert aggregate["artifacts"][0]["schema"] == "mop-generation1-cognitive-corpus/v2"
    verifier = by_id["g1_cognitive_corpus_verify"]
    assert verifier["artifacts"][0]["schema"] == (
        "mop-generation1-cognitive-corpus-verification/v2"
    )
    fields = verifier["artifacts"][0]["fields"]
    for check in builder.ADVANCED_VERIFIER_CHECKS:
        assert fields[f"checks.{check}"] is True
    assert fields["authority_audit.expected_effective_cell_count"] == 3003
    assert fields["authority_audit.selected_effective_cell_count"] == 3003
    report = by_id["g1_empirical_report"]
    assert report["artifacts"][0]["schema"] == "mop-generation1-empirical-report/v2"
    report_fields = report["artifacts"][0]["fields"]
    assert report_fields[
        "resource_authority.recommendation.recommended_max_workers"
    ] == max_workers
    assert report_fields[
        "resource_authority.recommendation.recommended_estimated_unified_memory_gb"
    ] == memory_gb
    assert any(
        authority["path"] == "proof/GENERATION1_RESOURCE_CANARY.json"
        for authority in report["authorities"]
    )
