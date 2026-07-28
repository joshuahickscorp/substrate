from __future__ import annotations

import copy
from pathlib import Path

import pytest

from substrate import v5config as C
from substrate import v5io as io
from substrate import v5principal as P
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


def test_v5_raw_verifier_loads_seals_and_regenerates_complete_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_v5_roots(monkeypatch, tmp_path)
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


def test_v5_mutations_cover_full_master_list_with_no_survivors() -> None:
    result = V.mutations(_memory_raw())

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
        "regeneration": [python, "-c", "print('{expected_digest}')"],
        "ready": [python, "-c", "print('frozen-ready-commit')"],
    }

    result = V.clean_clone(raw_report, commands=commands)

    assert result["all_pass"]
    assert result["ready_commit"] == "frozen-ready-commit"
    assert result["exact_reproduction"]
    assert result["normalized_double_regeneration_exact"]
    assert set(result["stages"]) == {
        "clone",
        "install",
        "tests",
        "ruff",
        "ruff_format",
    }


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
