"""Unit tests for the STARSS23 counting bed gate-architecture reproduction.

These tests assert the two invariants the design mandates for every reproduction:

1. the separately-authored verifier imports only the standard library (``json``, ``hashlib``,
   ``itertools``, ``dataclasses``, ``__future__``) and nothing under ``mop`` or from the producer, so
   agreement with the producer is real triangulation and not a shared bug;
2. the reproduction is strictly ADDITIVE: it writes only net-new ``count_repro_gate_arch_*`` sources and
   ``proof/STARSS23_COUNTING_REPRO_gate_arch*`` files, uses its own artifact schema id disjoint from the
   sealed bed, and holds the re-authored gate under the 4096-parameter ceiling.

Plus a fast end-to-end producer-and-verifier pass on the real subset (skipped when the data is absent) that
checks the verifier reproduces every graded number with zero mismatches, that the varied-axis gate anchors
are self-consistent, that a lone reproduction never self-certifies scientific confirmation, and that
tampering with the seal is caught.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from mop.beds.starss23 import count_repro_gate_arch_verifier as V
from mop.beds.starss23.count_producer import RealCountBedConfig
from mop.beds.starss23.count_repro_gate_arch_gate import (
    FLOPS_PER_INFERENCE_GATE_ARCH,
    PARAM_CEILING,
    CountReproGateArchGate,
    inference_flops_two_layer,
    param_count_two_layer,
)
from mop.beds.starss23.count_repro_gate_arch_producer import (
    ARTIFACT_SCHEMA,
    DEFAULT_COUNT_REPRO_GATE_ARCH_ARTIFACT_PATH,
    GATE_ARCH_SEEDS,
    build_real_count_repro_gate_arch_artifact,
)
from mop.beds.starss23.count_repro_gate_arch_prereg import (
    DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH,
    build_count_repro_gate_arch_prereg,
)

_FOA = Path("/Users/scammermike/Downloads/mop-data/starss23/foa_subset/foa_dev")
_META = Path("/Users/scammermike/Downloads/mop-data/starss23/metadata_dev_extracted/metadata_dev")
_REAL_PRESENT = _FOA.is_dir() and _META.is_dir()
_TIMESTAMP = "2026-07-18T00:00:00Z"


# ---------------------------------------------------------------------------
# 1. Verifier import surface: standard library only.
# ---------------------------------------------------------------------------


def test_verifier_imports_only_stdlib():
    source = Path(V.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = (
        "mop",
        "count_gate",
        "count_referee",
        "count_producer",
        "count_harness",
        "count_prereg",
        "count_repro_gate_arch_gate",
        "count_repro_gate_arch_producer",
        "count_repro_gate_arch_prereg",
        "stats",
        "harness",
        "referee",
        "numpy",
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        assert not any(name == bad or name.startswith(bad + ".") for bad in forbidden), name
    assert set(n.split(".")[0] for n in imported) <= {
        "json",
        "hashlib",
        "itertools",
        "dataclasses",
        "__future__",
    }


# ---------------------------------------------------------------------------
# 2. Additive boundary: net-new files, disjoint schema, sealed modules untouched.
# ---------------------------------------------------------------------------


def test_reproduction_is_additive_and_disjoint():
    # Net-new proof paths, disjoint from the sealed bed's proof paths.
    assert DEFAULT_COUNT_REPRO_GATE_ARCH_ARTIFACT_PATH == Path(
        "proof/STARSS23_COUNTING_REPRO_gate_arch.json"
    )
    assert DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH == Path(
        "proof/STARSS23_COUNTING_REPRO_gate_arch.prereg.json"
    )
    assert DEFAULT_COUNT_REPRO_GATE_ARCH_ARTIFACT_PATH != Path("proof/STARSS23_COUNTING_BED.json")
    # The artifact schema id is disjoint from the sealed bed's, so the sealed verifier will not accept it
    # and this verifier will not accept the sealed bed's artifact.
    assert ARTIFACT_SCHEMA == "mop-starss23-escs-count-bed-repro-gate-arch/v1"
    assert V.EXPECTED_ARTIFACT_SCHEMA == ARTIFACT_SCHEMA
    assert V.EXPECTED_ARTIFACT_SCHEMA != "mop-starss23-escs-count-bed/v1"
    # The disjoint seed family carries none of the original's (0..4) seed luck.
    assert set(GATE_ARCH_SEEDS).isdisjoint({0, 1, 2, 3, 4})

    # The sealed gate and producer modules do not import anything from this reproduction.
    for sealed in ("count_gate.py", "count_producer.py", "count_verifier.py", "count_prereg.py"):
        text = (Path("src/mop/beds/starss23") / sealed).read_text(encoding="utf-8")
        assert "count_repro_gate_arch" not in text, sealed


# ---------------------------------------------------------------------------
# 3. Re-authored gate contract: topology, ceiling, cost anchors, determinism.
# ---------------------------------------------------------------------------


def test_gate_topology_and_cost_anchors():
    # 264 -> 8 -> 4 -> 1 two-hidden-layer MLP: a genuine depth change, still under the ceiling.
    assert param_count_two_layer() == 2161
    assert param_count_two_layer() <= PARAM_CEILING
    assert inference_flops_two_layer() == 4321
    assert FLOPS_PER_INFERENCE_GATE_ARCH == 4321
    gate = CountReproGateArchGate(seed=40)
    assert gate.n_params() == 2161
    assert gate.flops_per_inference() == 4321
    # C_train uses the new architecture's per-frame inference FLOPs and the reused step factor.
    assert gate.training_flops(25172, 8) == 8 * 25172 * 3 * 4321


def test_gate_determinism_and_seed_independence():
    a = CountReproGateArchGate(seed=40).parameter_digest()
    b = CountReproGateArchGate(seed=40).parameter_digest()
    c = CountReproGateArchGate(seed=41).parameter_digest()
    d = CountReproGateArchGate(seed=0).parameter_digest()
    assert a == b  # deterministic under a fixed seed
    assert a != c  # distinct seeds give distinct inits
    assert a != d  # disjoint from the original seed family


def test_prereg_sesoi_is_label_only_and_above_floor():
    prereg = build_count_repro_gate_arch_prereg(
        timestamp=_TIMESTAMP,
        operating_reestimate_fraction=0.05,
        n_test_clips=21,
        n_test_changes=916,
        n_test_frames=22569,
        train_change_density=0.0502,
        coast_from_zero_mae=0.8,
    )
    # SESOI = 0.5 / n_test_clips, a label-only property independent of the gate architecture.
    assert prereg["sesoi"]["sesoi_mae"] == pytest.approx(0.5 / 21, abs=1e-9)
    assert prereg["sesoi"]["granularity_multiple"] >= 100.0
    assert prereg["preregistered_before_reading_test_scores"] is True
    assert prereg["activation_allowed"] is False and prereg["scientific_promotion"] is False


# ---------------------------------------------------------------------------
# 4. End-to-end producer + verifier on the real subset (fast config).
# ---------------------------------------------------------------------------


# max_frames=300 keeps the run fast while leaving every test clip above the 100x per-frame granularity
# floor the reproduction prereg enforces (the full run averages ~1074 frames per clip).
_SMALL_CONFIG = RealCountBedConfig(
    seeds=(40, 41, 42), target_rates=(0.10, 0.05), noisy_tv_frames=400, max_frames=300
)


@pytest.fixture(scope="module")
def repro_artifact(tmp_path_factory):
    if not _REAL_PRESENT:
        pytest.skip("real STARSS23 subset not present")
    out = tmp_path_factory.mktemp("repro_gate_arch")
    return build_real_count_repro_gate_arch_artifact(
        timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=out / "prereg.json"
    )


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seals_wellformed_artifact_within_ceiling(repro_artifact):
    art = repro_artifact.artifact
    assert isinstance(art.get("seal"), str) and len(art["seal"]) == 64
    assert art["schema"] == ARTIFACT_SCHEMA
    assert art["reproduction_axis"] == "gate_arch"
    assert art["gate"]["params"] == 2161 and art["gate"]["params"] <= 4096
    assert art["gate"]["flops_per_inference"] == 4321
    # Every arm's full-lifecycle FLOPs stay under the 6e10 ceiling.
    ceiling = art["matched_budget"]["flops"]
    assert ceiling <= 60_000_000_000
    for arm in art["harness"]["arm_summaries"]:
        assert arm["max_lifecycle_flops"] <= 60_000_000_000
    # Boundary flags hardcoded false.
    assert art["flags"] == {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_independent_verifier_reproduces_with_zero_mismatches(repro_artifact):
    result = V.verify_count_repro_gate_arch_artifact(repro_artifact.artifact)
    assert result.mismatches == (), result.mismatches
    assert result.seal_intact
    assert result.scores_reproduced
    assert result.stats_reproduced
    assert result.gate_anchors_ok
    assert result.honesty_ok
    assert result.independent_referee_reproduction
    # A lone reproduction can never self-certify scientific confirmation (reproductions counter is 0).
    assert result.independent_scientific_confirmation is False
    assert result.reproductions == 0


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_catches_tampering(repro_artifact):
    # Mutating a stored score without re-sealing must break the seal and the score re-derivation.
    tampered = copy.deepcopy(repro_artifact.artifact)
    seed0 = tampered["per_seed"][0]
    seed0["arm_scores"]["candidate"]["mae"] = float(seed0["arm_scores"]["candidate"]["mae"]) - 0.5
    result = V.verify_count_repro_gate_arch_artifact(tampered)
    assert result.independent_referee_reproduction is False
    assert result.mismatches != ()


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_catches_gate_anchor_tampering(repro_artifact):
    # Claiming a smaller parameter count than the sealed topology implies must be caught.
    tampered = copy.deepcopy(repro_artifact.artifact)
    tampered["gate"]["params"] = 1
    result = V.verify_count_repro_gate_arch_artifact(tampered)
    assert result.gate_anchors_ok is False
    assert result.independent_referee_reproduction is False
