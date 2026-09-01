from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from substrate import v5config as C
from substrate import v5io as io
from substrate import v5principal as P
from substrate import v5sensorium as VS
from substrate import v5verify as V


def _memory_raw() -> dict:
    receipts = {}
    checkpoints = {}
    paths = {}
    for split, split_seeds in P.SPLIT_SEEDS.items():
        seed = split_seeds[0]
        for arm in C.ARMS:
            predecessor = None
            for shard in range(P.SHARDS):
                unit = P.WorkUnit(split, seed, arm, shard)
                receipt, checkpoint = P.execute_unit(unit, predecessor)
                receipts[unit.identity] = receipt
                checkpoints[unit.identity] = checkpoint
                paths[unit.identity] = {
                    "receipt": f"runs/substrate/v5/{split}/units/{unit.identity}.json",
                    "checkpoint": f"runs/substrate/v5/{split}/checkpoints/{unit.identity}.json",
                }
                predecessor = checkpoint
    return {
        "receipts": receipts,
        "checkpoints": checkpoints,
        "paths": paths,
        "expected": len(receipts),
        "valid": len(receipts),
        "missing": [],
        "invalid": {},
        "seal_errors": {},
        "all_pass": True,
        "activation": False,
    }


def _redirect_v5_roots(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    monkeypatch.setattr(io, "ROOT", root)
    monkeypatch.setattr(io, "commit", lambda: "c" * 40)
    monkeypatch.setattr(io, "source_digest", lambda: "d" * 64)
    for name in (
        "EVIDENCE",
        "RUNS",
        "ARTIFACTS",
        "CONFIGS",
        "MODELS",
        "DATA",
        "CACHE",
    ):
        monkeypatch.setattr(io, name, root / name.lower())


def test_v5_verifier_source_rebinding_preserves_mutation_isolation() -> None:
    sealed = {
        "payload": {"items": [{"value": 1}]},
        "sha256": "ignored",
        "source_commit": "c" * 40,
        "source_digest": "d" * 64,
    }

    stripped = V._strip_seal(sealed)
    assert set(stripped) == {"payload"}

    rebound = V._source_bound_seal(sealed, ("e" * 40, "f" * 64))
    rebound["payload"]["items"][0]["value"] = 3
    assert sealed["payload"]["items"][0]["value"] == 1


def test_v5_independent_cached_public_tasks_remain_isolated_between_callers() -> None:
    first_identity, first_observation, first_target = V._independent_public_task(
        "principal", 5_000, 0, 0
    )
    first_observation["modality_cues"]["text"] = 999.0
    first_observation["mechanism_cues"]["model_fabric"] = 999.0
    first_observation["modalities"].append("mutated")

    second_identity, second_observation, second_target = V._independent_public_task(
        "principal", 5_000, 0, 0
    )
    assert second_identity == first_identity
    assert second_target == first_target
    assert "mutated" not in second_observation["modalities"]
    assert second_observation["modality_cues"]["text"] != 999.0
    assert second_observation["mechanism_cues"]["model_fabric"] != 999.0


def test_v5_independent_cached_sensor_events_retain_boundary_and_digest() -> None:
    uncached = V._independent_sensor_event_uncached(
        "task:sensor-cache",
        "text",
        0.25,
        2,
        4,
        "model:text-specialist:v5",
    )
    first, first_digest = V._independent_sensor_event_with_digest(
        "task:sensor-cache",
        "text",
        0.25,
        2,
        4,
        "model:text-specialist:v5",
    )
    assert first == uncached
    assert first_digest == VS.canonical_event_digest(uncached)

    first.observation["target"] = True
    with pytest.raises(VS.SensoriumError, match="hidden target authority"):
        VS.Sensorium()._ingest_cached(first)

    second, second_digest = V._independent_sensor_event_with_digest(
        "task:sensor-cache",
        "text",
        0.25,
        2,
        4,
        "model:text-specialist:v5",
    )
    assert second is not first
    assert second == uncached
    assert second.observation["observable_cue"] == 0.25
    assert second_digest == VS.canonical_event_digest(second)


def test_v5_raw_verifier_loads_seals_and_regenerates_complete_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_v5_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(V.v4io, "RUNS", tmp_path / "absent-v4-runs")
    units = [P.WorkUnit("principal", 5_000, "full_v5", shard) for shard in range(P.SHARDS)]
    predecessor = None
    for unit in units:
        receipt, checkpoint = P.execute_unit(unit, predecessor)
        io.run_json(
            f"{unit.split}/units/{unit.identity}.json",
            receipt,
        )
        io.run_json(
            f"{unit.split}/checkpoints/{unit.identity}.json",
            checkpoint,
        )
        predecessor = checkpoint
    monkeypatch.setattr(
        P,
        "execute_unit",
        lambda *_args, **_kwargs: pytest.fail("independent verifier called principal execution"),
    )

    result = V.raw(units)

    assert result["all_pass"]
    assert result["hash_chains_valid"]
    assert result["deterministic_regeneration_exact"]
    assert result["valid"] == P.SHARDS
    assert result["activation"] is False


def test_v5_raw_verifier_refuses_an_altered_sealed_unit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_v5_roots(monkeypatch, tmp_path)
    unit = P.WorkUnit("principal", 5_000, "full_v5", 0)
    receipt, checkpoint = P.execute_unit(unit)
    receipt_path = io.run_json(
        f"{unit.split}/units/{unit.identity}.json",
        receipt,
    )
    io.run_json(
        f"{unit.split}/checkpoints/{unit.identity}.json",
        checkpoint,
    )
    receipt_path.write_text(
        receipt_path.read_text().replace('"activation":false', '"activation":true'),
        encoding="utf-8",
    )

    result = V.raw([unit])

    assert not result["all_pass"]
    assert unit.identity in result["seal_errors"]


def test_v5_raw_verifier_preserves_ready_source_after_verifier_transition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_v5_roots(monkeypatch, tmp_path)
    ready_source = ("c" * 40, "d" * 64)
    units = [P.WorkUnit("principal", 5_000, "full_v5", shard) for shard in range(P.SHARDS)]
    predecessor = None
    for unit in units:
        receipt, checkpoint = P.execute_unit(unit, predecessor)
        io.run_json(f"{unit.split}/units/{unit.identity}.json", receipt)
        io.run_json(
            f"{unit.split}/checkpoints/{unit.identity}.json",
            checkpoint,
        )
        predecessor = checkpoint
    monkeypatch.setattr(io, "commit", lambda: "e" * 40)
    monkeypatch.setattr(io, "source_digest", lambda: "f" * 64)

    result = V.raw(units)

    assert result["all_pass"]
    assert result["principal_source"] == {
        "source_commit": ready_source[0],
        "source_digest": ready_source[1],
    }


def test_v5_sample_regeneration_remains_ready_source_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_source = ("c" * 40, "d" * 64)
    monkeypatch.setattr(io, "commit", lambda: ready_source[0])
    monkeypatch.setattr(io, "source_digest", lambda: ready_source[1])
    unit = next(unit for unit in P.work_units("principal") if unit.arm == "full_v5" and unit.shard == 0)
    receipt, checkpoint = V._independent_execute_unit(unit)
    raw_report = {
        "all_pass": True,
        "receipts": {unit.identity: receipt},
        "principal_source": {
            "source_commit": ready_source[0],
            "source_digest": ready_source[1],
        },
    }
    expected = io.sha_obj(
        {
            "receipt": receipt,
            "checkpoint": checkpoint,
        }
    )
    monkeypatch.setattr(io, "commit", lambda: "e" * 40)
    monkeypatch.setattr(io, "source_digest", lambda: "f" * 64)

    _, actual = V._sample_regeneration(raw_report)

    assert actual == expected


def test_v5_recomputation_rebuilds_every_split_effect_cost_and_continuity() -> None:
    report = V.recompute(_memory_raw())

    assert len(report["effects"]) == 15
    assert set(report["splits"]) == set(P.SPLIT_SEEDS)
    assert all(len(split["effects"]) == 15 for split in report["splits"].values())
    assert all(set(cost["arms"]) == set(C.ARMS) for cost in report["costs"].values())
    assert report["continuity"]["identity"]
    assert report["continuity"]["model_replacement"]
    assert report["continuity"]["body_continuity"]
    assert report["continuity"]["learning"]
    assert report["historical_v4"]["preserved"]
    assert report["independent_recomputation_complete"]
    assert report["activation"] is False


def test_v5_mutations_cover_full_master_list_with_no_survivors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_report = _memory_raw()
    checkpoint = next(iter(raw_report["checkpoints"].values()))
    principal_source = checkpoint["entity_checkpoint"]
    raw_report["principal_source"] = {
        "source_commit": principal_source["source_commit"],
        "source_digest": principal_source["source_digest"],
    }
    monkeypatch.setattr(io, "commit", lambda: "e" * 40)
    monkeypatch.setattr(io, "source_digest", lambda: "f" * 64)

    result = V.mutations(raw_report)

    assert result["total"] == len(V.MUTATION_CLASSES) == 21
    assert result["detected"] == result["total"]
    assert result["survived"] == []
    assert result["zero_survived"]
    assert all(row["input_changed"] for row in result["mutations"])
    assert all(row["detectors"] for row in result["mutations"])


def test_v5_clean_clone_supports_explicit_injected_commands() -> None:
    raw_report = _memory_raw()
    python = "{python}"
    commands = {
        "clone": [
            python,
            "-c",
            "from pathlib import Path;Path(r'{clone}').mkdir(parents=True)",
        ],
        "install": [
            python,
            "-c",
            "from pathlib import Path;Path(r'{installed}').mkdir(parents=True)",
        ],
        "tests": [python, "-c", "pass"],
        "ruff": [python, "-c", "pass"],
        "ruff_format": [python, "-c", "pass"],
        "regeneration": [
            python,
            "-c",
            "print({'receipt': '{expected_digest}'}['receipt'])",
        ],
        "ready": [python, "-c", "print('frozen-ready-commit')"],
    }

    result = V.clean_clone(raw_report, commands=commands)

    assert result["all_pass"]
    assert result["ready_commit"] == "frozen-ready-commit"
    assert result["exact_reproduction"]
    assert result["normalized_double_regeneration_exact"]
    assert result["commands_injected"]
    assert not result["cache_reused"]
    assert set(result["stages"]) == {
        "clone",
        "install",
        "tests",
        "ruff",
        "ruff_format",
    }


def test_v5_clean_clone_reuses_only_exact_default_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_digest = "e" * 64
    cached = {
        "schema": "substrate-v5-clean-clone/v1",
        "commands_injected": False,
        "ready_ref": "frozen-ready",
        "ready_commit": "ready-commit",
        "ready_ref_returncode": 0,
        "expected_digest": expected_digest,
        "actual_digests": [expected_digest, expected_digest],
        "exact_reproduction": True,
        "normalized_double_regeneration_exact": True,
        "all_pass": True,
        "source_worktree_clean": True,
        "source_commit": "c" * 40,
        "source_digest": "d" * 64,
        "program": "substrate-v5",
        "sha256": "sealed",
    }
    monkeypatch.setattr(V.io, "load", lambda _name: cached)
    monkeypatch.setattr(V.io, "commit", lambda: "c" * 40)
    monkeypatch.setattr(V.io, "source_digest", lambda: "d" * 64)
    calls: list[list[str]] = []

    def run_command(command, *, cwd, env=None):
        del cwd, env
        calls.append(list(command))
        if command[1] == "status":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "ready-commit\n", "")

    monkeypatch.setattr(V, "_run_command", run_command)

    result = V._reusable_clean_clone(expected_digest, "frozen-ready")

    assert result is not None
    assert result["cache_reused"]
    assert "source_commit" not in result
    assert "sha256" not in result
    assert ["git", "status", "--porcelain", "--untracked-files=all"] in calls
    assert ["git", "rev-parse", "frozen-ready^{}"] in calls

    cached["source_digest"] = "f" * 64
    assert V._reusable_clean_clone(expected_digest, "frozen-ready") is None


def test_v5_classification_is_ordered_and_never_assigns_unqualified_nous() -> None:
    raw_report = {"all_pass": True}
    effects = {name: {"passes": True} for name in C.HYPOTHESES}
    verification = {
        "effects": effects,
        "metrics": {"modality_count": 8},
        "continuity": {
            "identity": True,
            "sensor_recovery": True,
            "model_identities": ["model-a", "model-b"],
            "executed_model_families": ["family-a", "family-b"],
            "model_replacement": True,
            "body_continuity": True,
            "body_identities": ["desktop", "simulator"],
            "sensor_environments": ["environment-a", "environment-b"],
            "body_variants": ["desktop", "simulator"],
            "diversity_records_complete": True,
            "learning": True,
        },
        "historical_v4": {"preserved": True},
        "replication_pass": True,
        "open_world_pass": True,
        "all_pass": True,
    }
    mutation = {"zero_survived": True}
    clone = {"all_pass": True}

    embodied = V.classify(
        raw_report,
        verification,
        mutation,
        clone,
        review_complete=False,
    )
    review = V.classify(
        raw_report,
        verification,
        mutation,
        clone,
        review_complete=True,
    )

    assert embodied["classification"] == "persistent_embodied_proto_nous_candidate"
    assert review["classification"] == "multimodal_nous_ready_for_review"
    assert review["unqualified_nous"] is False
    broken = copy.deepcopy(verification)
    broken["effects"]["H_M3"]["passes"] = False
    downgraded = V.classify(
        raw_report,
        broken,
        mutation,
        clone,
        review_complete=True,
    )
    assert downgraded["classification"] == "multimodal_cognitive_substrate"
    names_only = copy.deepcopy(verification)
    names_only["continuity"]["executed_model_families"] = []
    names_only["continuity"]["sensor_environments"] = []
    names_only["continuity"]["body_variants"] = []
    names_only["continuity"]["diversity_records_complete"] = False
    refused_diversity = V.classify(
        raw_report,
        names_only,
        mutation,
        clone,
        review_complete=True,
    )
    assert refused_diversity["classification"] == "persistent_sensorium"


def test_v5_finalize_is_pure_until_publication_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_report = {"all_pass": True, "valid": 1}
    effects = {name: {"passes": False} for name in C.HYPOTHESES}
    verification = {
        "effects": effects,
        "metrics": {"modality_count": 0},
        "continuity": {
            "identity": False,
            "sensor_recovery": False,
            "model_identities": [],
            "executed_model_families": [],
            "model_replacement": False,
            "body_continuity": False,
            "body_identities": [],
            "sensor_environments": [],
            "body_variants": [],
            "diversity_records_complete": False,
            "learning": False,
        },
        "historical_v4": {"preserved": True},
        "replication_pass": False,
        "open_world_pass": False,
        "all_pass": True,
    }
    mutation = {"zero_survived": True, "survived": []}
    clone = {"all_pass": True}
    monkeypatch.setattr(
        io,
        "seal",
        lambda *_args, **_kwargs: pytest.fail("pure finalization published"),
    )

    result = V.finalize(
        raw_report,
        verification,
        mutation,
        clone,
        publish=False,
    )

    assert result["classification"]["classification"] == "functional_proto_nous_candidate"
    assert result["review"]["published"] is False
    assert result["activation"] is False


def test_v5_all_null_requires_and_accepts_explicit_verified_refusal() -> None:
    null_effects = {name: {"passes": False} for name in C.HYPOTHESES}
    verification = {
        "effects": null_effects,
        "splits": {
            split: {
                "effects": copy.deepcopy(null_effects),
                "all_pass": False,
            }
            for split in P.SPLIT_SEEDS
        },
        "all_pass": True,
        "independent_recomputation_complete": True,
    }
    raw_report = {"all_pass": True}
    mutation = {"zero_survived": True}
    clone = {"all_pass": True}
    refusal = V.terminal_refusal_authority(
        raw_report,
        verification,
        mutation,
        clone,
    )
    result = {
        "raw": raw_report,
        "verification": verification,
        "mutation": mutation,
        "clean_clone": clone,
        "final": {
            "classification": {
                "classification": "functional_proto_nous_candidate",
                "unqualified_nous": False,
            },
            "final_state": {"review_package_complete": False},
            "terminal_refusal": refusal,
        },
        "activation": False,
    }

    assert refusal["terminal_refusal"]
    assert refusal["independently_verified"]
    assert refusal["null_count"] == 45
    assert V._terminal_verification_passed(result)
    result["final"]["terminal_refusal"] = {
        "terminal_refusal": False,
        "independently_verified": False,
        "null_count": 0,
    }
    assert not V._terminal_verification_passed(result)
