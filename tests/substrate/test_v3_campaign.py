from __future__ import annotations

import json

from substrate import v3config as C
from substrate import v3principal as P
from substrate import v3state as S


def test_splits_are_disjoint():
    values = [seed for split in C.SPLITS.values() for seed in split]
    assert len(values) == len(set(values))


def test_principal_dag_is_unique_and_nous_scale():
    manifest = P.manifest()
    assert manifest["unit_count"] == 1224
    assert manifest["principal_units"] == 1152
    assert manifest["expected_episodes"] >= 100_000
    assert manifest["unique_identities"]


def test_work_unit_plan_is_cached_as_frozen_units():
    first = P.work_units()
    second = P.work_units()
    assert first is second
    assert all(unit.__dataclass_params__.frozen for unit in first)


def test_principal_manifest_parser_reuses_unchanged_bytes():
    P._parse_manifest.cache_clear()

    first = P._load_manifest()
    second = P._load_manifest()

    assert first is second
    assert P._parse_manifest.cache_info().hits == 1
    P._parse_manifest.cache_clear()


def test_principal_manifest_parser_rechecks_changed_bytes():
    P._parse_manifest.cache_clear()
    raw = P.MANIFEST.read_bytes()
    P._parse_manifest(raw)
    tampered = json.loads(raw)
    tampered["activation"] = True

    try:
        P._parse_manifest(json.dumps(tampered).encode())
    except P.io.Refused:
        pass
    else:
        raise AssertionError("changed manifest bytes bypassed the seal check")
    finally:
        P._parse_manifest.cache_clear()


def test_principal_receipt_validation_cache_is_content_bound(monkeypatch, tmp_path):
    unit = P.work_units()[0]
    receipt = {
        "schema": "substrate-v3-principal-unit/v1",
        "unit": {"identity": unit.identity},
        "summary": {"episodes": 128, "checkpoint_exact": True, "body_continuity": True},
        "activation": False,
    }
    receipt["receipt_identity"] = P.io.sha_obj(
        {key: value for key, value in receipt.items() if key != "receipt_identity"}
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    calls = []
    original = P.validate_receipt

    def counted(document, candidate):
        calls.append(candidate.identity)
        return original(document, candidate)

    monkeypatch.setattr(P, "validate_receipt", counted)

    assert P._validate_receipt_file(path, unit) is True
    assert P._validate_receipt_file(path, unit) is True
    assert calls == [unit.identity]

    receipt["activation"] = True
    path.write_text(json.dumps(receipt))
    assert P._validate_receipt_file(path, unit) is False
    assert calls == [unit.identity, unit.identity]


def test_source_ref_parser_requires_two_real_commit_lines():
    head = "a" * 40
    ready = "b" * 40
    assert P._parse_source_refs(f"{head}\n{ready}\n", 0) == (head, ready)
    assert P._parse_source_refs(f"{head}\n{P.READY_TAG}^{{}}\n", 128) == (head, None)
    assert P._parse_source_refs(f"{head}\n{ready}\nextra\n", 0) == (head, None)
    assert P._parse_source_refs("not-a-commit\n", 0) == ("", None)


def test_source_ready_coalesces_head_and_ready_tag_lookup(monkeypatch, tmp_path):
    head = "a" * 40
    ready = "b" * 40
    calls = []

    class Result:
        returncode = 0
        stdout = f"{head}\n{ready}\n"

    def run(command, **_kwargs):
        calls.append(command)
        return Result()

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("valid combined source lookup should not need a fallback")

    monkeypatch.setattr(P.subprocess, "run", run)
    monkeypatch.setattr(P.subprocess, "check_output", unexpected_fallback)
    monkeypatch.setattr(P, "MANIFEST", tmp_path / "missing-manifest.json")

    source = P._source_ready()

    assert calls == [["git", "rev-parse", "HEAD", f"{P.READY_TAG}^{{}}"]]
    assert source["ready_tag_exists"] is True
    assert source["ready_commit"] == ready
    assert source["head"] == head


def test_one_unit_is_deterministic_and_valid():
    unit = P.work_units()[1]
    first = P.execute_unit(unit)
    second = P.execute_unit(unit)
    first["summary"]["runtime_seconds"] = 0.0
    second["summary"]["runtime_seconds"] = 0.0
    first["summary"]["peak_rss_mib"] = 0.0
    second["summary"]["peak_rss_mib"] = 0.0
    first.pop("receipt_identity")
    second.pop("receipt_identity")
    assert first == second


def test_v2_preservation_maps_frozen_v3_histories_to_authorized_seeds():
    mapped = [P._authorized_v2_seed(seed) for seed in C.SPLITS["principal"]]
    assert len(mapped) == len(set(mapped))
    unit = next(
        unit
        for unit in P.work_units()
        if unit.history_seed == 1047 and unit.arm == "full_v3" and unit.shard == 0
    )
    receipt = P.execute_unit(unit)
    assert P.validate_receipt(receipt, unit)
    assert receipt["summary"]["v2_preservation"]["v3_history_seed"] == 1047
    assert receipt["summary"]["v2_preservation"]["authorized_v2_generator_seed"] in {
        seed for split in P.v2config.SPLITS.values() for seed in split
    }


def test_arm_features_are_focused_ablations():
    full = S.ARM_FEATURES["full_v3"]
    assert "ontology" in full
    assert "ontology" not in S.ARM_FEATURES["fixed_ontology"]
    assert "epistemology" not in S.ARM_FEATURES["confidence_only_epistemology"]
    assert "reasoning" not in S.ARM_FEATURES["fixed_reasoning"]
    assert "understanding" not in S.ARM_FEATURES["no_understanding_structure"]
    assert "self" not in S.ARM_FEATURES["no_self_model"]
    assert "world" not in S.ARM_FEATURES["no_world_model"]
