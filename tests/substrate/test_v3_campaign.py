from __future__ import annotations

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


def test_arm_features_are_focused_ablations():
    full = S.ARM_FEATURES["full_v3"]
    assert "ontology" in full
    assert "ontology" not in S.ARM_FEATURES["fixed_ontology"]
    assert "epistemology" not in S.ARM_FEATURES["confidence_only_epistemology"]
    assert "reasoning" not in S.ARM_FEATURES["fixed_reasoning"]
    assert "understanding" not in S.ARM_FEATURES["no_understanding_structure"]
    assert "self" not in S.ARM_FEATURES["no_self_model"]
    assert "world" not in S.ARM_FEATURES["no_world_model"]
