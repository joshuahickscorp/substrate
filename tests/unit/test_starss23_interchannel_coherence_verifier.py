
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from mop.beds.starss23.interchannel_coherence_prereg import (
    FEATURIZER_VARIANTS,
    build_featurizers_prereg,
)
from mop.beds.starss23.verifier_interchannel_coherence import verify_artifact

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER_SRC = REPO_ROOT / "src" / "mop" / "beds" / "starss23" / "verifier_interchannel_coherence.py"
ARTIFACT_PATH = REPO_ROOT / "proof" / "STARSS23_ESCS_BED_interchannel_coherence.json"

_ALLOWED_IMPORT_ROOTS = {"__future__", "argparse", "dataclasses", "hashlib", "itertools", "json"}


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.is_file():
        pytest.skip("sealed artifact not present; run the producer first")
    return json.loads(ARTIFACT_PATH.read_bytes().decode("utf-8"))


# ---------------------------------------------------------------------------
# The verifier shares no code with the producer.
# ---------------------------------------------------------------------------


def test_verifier_imports_no_producer_or_mop_code() -> None:
    tree = ast.parse(VERIFIER_SRC.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                roots.add(name.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    assert "mop" not in roots, "the independent verifier must not import any producer or mop code"
    assert roots <= _ALLOWED_IMPORT_ROOTS, f"verifier import surface widened: {roots - _ALLOWED_IMPORT_ROOTS}"


# ---------------------------------------------------------------------------
# The verifier re-derives the sealed null from raw data.
# ---------------------------------------------------------------------------


def test_verifier_reproduces_sealed_null() -> None:
    artifact = _load_artifact()
    result = verify_artifact(artifact)
    assert result.seal_intact
    assert result.schema_ok
    assert result.scores_reproduced
    assert result.stats_reproduced
    assert result.spread_reproduced
    assert result.budget_ok
    assert result.featurizer_ok
    assert result.honesty_ok
    assert result.independent_referee_reproduction
    assert result.reproduced_verdict == "null"
    # A single real run at n = 5 across a three-featurizer family can never self-certify science.
    assert result.independent_scientific_confirmation is False
    assert result.mismatches == ()


def test_verifier_detects_a_flipped_fire() -> None:
    artifact = _load_artifact()
    tampered = copy.deepcopy(artifact)
    # Inject a fabricated candidate fire on the first clip of the first seed. The stored score no longer
    # matches a re-score from the raw fires, so the independent reproduction must fail.
    clip = tampered["per_seed"][0]["clips"][0]
    fabricated = max(clip["fires"]["candidate"], default=-1) + 1
    if fabricated >= 0:
        clip["fires"]["candidate"].append(fabricated)
    result = verify_artifact(tampered)
    assert not result.independent_referee_reproduction
    assert result.mismatches != ()


def test_verifier_detects_a_broken_seal() -> None:
    artifact = _load_artifact()
    tampered = copy.deepcopy(artifact)
    tampered["seal"] = "0" * 64
    result = verify_artifact(tampered)
    assert not result.seal_intact
    assert not result.independent_referee_reproduction


# ---------------------------------------------------------------------------
# The featurizer-family Bonferroni wall.
# ---------------------------------------------------------------------------


def test_prereg_seals_a_three_featurizer_bonferroni_wall() -> None:
    body = build_featurizers_prereg(
        timestamp="2026-07-17T00:00:00Z",
        operating_firing_fraction=0.06,
        n_test_clips=21,
        n_test_onsets=538,
        train_onset_density=0.06,
        n_test_frames=22569,
    )
    assert body["sesoi"]["sesoi_f1"] == 0.05
    mult = body["multiplicity"]
    assert mult["n_variants"] == 3
    assert mult["correction"] == "Bonferroni"
    # 0.05 / 3 = 0.016667 and the smallest achievable one-sided p at n = 5 is 1/32 = 0.03125, which
    # exceeds it, so no single featurizer can clear family-wise significance from this family alone.
    assert mult["per_variant_alpha"] == pytest.approx(0.05 / 3)
    assert mult["min_achievable_one_sided_p"] == pytest.approx(1.0 / 32)
    assert mult["family_significance_reachable_at_n5"] is False
    assert body["activation_allowed"] is False
    assert body["scientific_promotion"] is False
    # The self seal reproduces.
    stored = dict(body)
    seal = stored.pop("canonical_sha256")
    from mop.substrate.events import canonical_sha256

    assert canonical_sha256(stored) == seal


def test_interchannel_coherence_is_in_the_sealed_family() -> None:
    ids = [entry["variant_id"] for entry in FEATURIZER_VARIANTS]
    assert "interchannel_coherence" in ids
    assert len(ids) == 3
    assert len(set(ids)) == 3
