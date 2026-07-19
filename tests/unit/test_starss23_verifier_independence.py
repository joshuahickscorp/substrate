
from __future__ import annotations

import ast
import copy
import itertools
import pathlib
import random

import pytest

from mop.beds.starss23 import BED_SCHEMA, CLAIM_SCOPE
from mop.beds.starss23 import referee as referee_module
from mop.beds.starss23 import verifier as verifier_module
from mop.beds.starss23.referee import score_arm
from mop.beds.starss23.verifier import (
    MIN_REPRODUCTIONS,
    VerificationRefusal,
    verification_payload,
    verify_artifact,
)
from mop.substrate.events import canonical_sha256

SEEDS = [11, 22, 33, 44, 55]
_CLIP_GT = {
    "clip_a": [10, 20, 30, 40, 50],
    "clip_b": [15, 35, 55, 75, 95],
}



_ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "hashlib",
    "itertools",
    "json",
    "math",
    "dataclasses",
    "typing",
    "collections",
    "numpy",
}


def test_verifier_imports_no_producer_code() -> None:
    source = pathlib.Path(verifier_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    relative_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                relative_imports.append(node.module or "<relative>")
            else:
                imported_roots.add((node.module or "").split(".")[0])

    assert not relative_imports, f"relative imports would pull producer code: {relative_imports}"
    assert "mop" not in imported_roots, "verifier must not import anything under mop"
    for forbidden in ("referee", "stats", "artifact", "schema", "harness", "controls", "gate"):
        assert forbidden not in imported_roots
    unexpected = imported_roots - _ALLOWED_IMPORT_ROOTS
    assert not unexpected, f"verifier imported unexpected modules: {unexpected}"


def test_verifier_module_namespace_holds_no_producer_symbol() -> None:
    for name in dir(verifier_module):
        value = getattr(verifier_module, name)
        module_name = getattr(value, "__module__", "") or ""
        assert not module_name.startswith("mop.beds.starss23.referee")
        assert not module_name.startswith("mop.beds.starss23.stats")
        assert not module_name.startswith("mop.science.statistics")




def _rmr_fires(gt: list[int], seed: int, clip: str) -> list[int]:
    rng = random.Random((seed << 8) ^ hash(clip) & 0xFFFF)
    fires: list[int] = []
    for index, frame in enumerate(gt):
        if (index + seed) % 3 == 0:
            fires.append(frame)  # a genuine hit
        else:
            fires.append(frame + 5)  # shoved three past the two-frame collar
    fires.append(rng.randrange(200, 260))  # a spurious fire far from any onset
    return sorted(set(fires))


def _sign_flip(deltas: list[float]) -> tuple[float, float, int]:
    n = len(deltas)
    t_obs = sum(deltas) / n
    at_least = sum(
        1
        for signs in itertools.product((1.0, -1.0), repeat=n)
        if sum(s * d for s, d in zip(signs, deltas, strict=True)) / n >= t_obs - 1e-12
    )
    return t_obs, at_least / (2**n), 2**n


def _build_sealed_artifact(
    *,
    source_kind: str = "synthetic",
    rights_clean: bool = False,
    reproductions: int = 0,
    noisy_tv_at_chance: bool = True,
) -> dict:
    per_seed = []
    deltas: list[float] = []
    for seed in SEEDS:
        clips = []
        candidate_pairs = []
        rmr_pairs = []
        for clip_id, gt in _CLIP_GT.items():
            candidate = list(gt)  # the candidate detects perfectly
            rmr = _rmr_fires(gt, seed, clip_id)
            clips.append(
                {
                    "clip_id": f"{clip_id}_s{seed}",
                    "gt_onsets": list(gt),
                    "fires": {"candidate": candidate, "rate_matched_random": rmr},
                }
            )
            candidate_pairs.append((gt, candidate))
            rmr_pairs.append((gt, rmr))
        candidate_score = score_arm(candidate_pairs)
        rmr_score = score_arm(rmr_pairs)
        delta = candidate_score.f1 - rmr_score.f1
        deltas.append(delta)
        per_seed.append(
            {
                "seed": seed,
                "clips": clips,
                "arm_scores": {
                    "candidate": candidate_score.payload(),
                    "rate_matched_random": rmr_score.payload(),
                },
                "delta": {"candidate_minus_rate_matched_random": delta},
            }
        )

    t_obs, one_sided_p, n_perm = _sign_flip(deltas)
    body = {
        "schema": BED_SCHEMA,
        "bed_id": "starss23_escs_event_formation",
        "stage": 3,
        "claim_scope": CLAIM_SCOPE,
        "source_kind": source_kind,
        "rights_clean": rights_clean,
        "reproductions": reproductions,
        "collar_frames": 2,
        "frame_ms": 100,
        "primary_control": "rate_matched_random",
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "stats": {
            "control": "rate_matched_random",
            "deltas": deltas,
            "t_obs": t_obs,
            "one_sided_p": one_sided_p,
            "n_permutations": n_perm,
            "two_sided_005_reachable": (2.0 / n_perm) <= 0.05,
            "sesoi_f1": 0.05,
            "claim_verb": "consistent with",
        },
        "controls": {"noisy_tv_at_chance": noisy_tv_at_chance},
        "flags": {
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_scientific_confirmation": False,
        },
    }
    body["seal"] = canonical_sha256(body)
    return body


def _reseal(artifact: dict) -> dict:
    body = {k: v for k, v in artifact.items() if k != "seal"}
    artifact["seal"] = canonical_sha256(body)
    return artifact




def test_verifier_reproduces_referee_f1_from_the_sealed_artifact() -> None:
    artifact = _build_sealed_artifact()
    result = verify_artifact(artifact)
    assert result.seal_intact
    assert result.schema_ok
    assert result.scores_reproduced
    assert result.stats_reproduced
    assert result.honesty_ok
    assert result.independent_referee_reproduction
    assert result.detail["recomputed_one_sided_p"] == pytest.approx(1 / 32)


def test_independent_matcher_agrees_with_the_referee_across_a_random_battery() -> None:
    rng = random.Random(2026)
    for _ in range(400):
        span = rng.randrange(5, 60)
        gt = sorted(rng.sample(range(span), rng.randrange(0, min(span, 8))))
        pred = sorted(rng.sample(range(span), rng.randrange(0, min(span, 8))))
        collar = rng.randrange(0, 4)
        referee_counts = referee_module.score_clip(gt, pred, collar)
        verifier_counts = verifier_module._match_counts(gt, pred, collar)
        assert referee_counts == verifier_counts


def test_sign_flip_floor_and_reachability_match_the_recipe() -> None:
    t_obs, one_sided_p, n_perm = verifier_module._sign_flip_one_sided_p([0.1, 0.2, 0.3, 0.4, 0.5])
    assert n_perm == 32
    assert one_sided_p == pytest.approx(1 / 32)
    assert (2.0 / n_perm) > 0.05




def test_forged_score_without_resealing_is_rejected_by_the_seal() -> None:
    artifact = _build_sealed_artifact()
    artifact["per_seed"][0]["arm_scores"]["rate_matched_random"]["f1"] = 1.0  # no reseal
    result = verify_artifact(artifact)
    assert not result.seal_intact
    assert not result.independent_referee_reproduction
    assert result.rejected


def test_forged_score_with_resealing_is_caught_by_re_derivation() -> None:
    artifact = _build_sealed_artifact()
    artifact["per_seed"][0]["arm_scores"]["candidate"]["f1"] = 0.123
    _reseal(artifact)
    result = verify_artifact(artifact)
    assert result.seal_intact  # the forger fixed the hash
    assert not result.scores_reproduced  # but the fires do not produce that F1
    assert not result.independent_referee_reproduction


def test_tampered_fire_list_is_caught_even_after_resealing() -> None:
    artifact = _build_sealed_artifact()
    seed_block = artifact["per_seed"][0]
    seed_block["clips"][0]["fires"]["rate_matched_random"] = list(seed_block["clips"][0]["gt_onsets"])
    _reseal(artifact)
    result = verify_artifact(artifact)
    assert not result.scores_reproduced
    assert not result.independent_referee_reproduction


def test_forged_p_value_is_caught_after_resealing() -> None:
    artifact = _build_sealed_artifact()
    artifact["stats"]["one_sided_p"] = 0.001  # below the discrete floor
    _reseal(artifact)
    result = verify_artifact(artifact)
    assert result.seal_intact
    assert not result.stats_reproduced
    assert not result.independent_referee_reproduction


def test_self_certified_confirmation_is_rejected_as_dishonest() -> None:
    artifact = _build_sealed_artifact()
    artifact["flags"]["independent_scientific_confirmation"] = True
    _reseal(artifact)
    result = verify_artifact(artifact)
    assert not result.honesty_ok
    assert not result.independent_referee_reproduction


def test_promotion_claim_flag_must_be_false() -> None:
    artifact = _build_sealed_artifact()
    artifact["flags"]["scientific_promotion"] = True
    _reseal(artifact)
    result = verify_artifact(artifact)
    assert not result.honesty_ok
    assert not result.independent_referee_reproduction


def test_overreaching_claim_verb_is_rejected() -> None:
    artifact = _build_sealed_artifact()
    artifact["stats"]["claim_verb"] = "demonstrates"
    _reseal(artifact)
    result = verify_artifact(artifact)
    assert not result.honesty_ok




def test_synthetic_reproduces_but_never_promotes() -> None:
    artifact = _build_sealed_artifact(source_kind="synthetic")
    result = verify_artifact(artifact)
    assert result.independent_referee_reproduction
    assert not result.independent_scientific_confirmation


def test_confirmation_requires_all_conditions_together() -> None:
    ready = _build_sealed_artifact(
        source_kind="real", rights_clean=True, reproductions=MIN_REPRODUCTIONS, noisy_tv_at_chance=True
    )
    assert verify_artifact(ready).independent_scientific_confirmation

    still_synthetic = _build_sealed_artifact(
        source_kind="synthetic", rights_clean=True, reproductions=MIN_REPRODUCTIONS
    )
    assert not verify_artifact(still_synthetic).independent_scientific_confirmation

    dirty_rights = _build_sealed_artifact(
        source_kind="real", rights_clean=False, reproductions=MIN_REPRODUCTIONS
    )
    assert not verify_artifact(dirty_rights).independent_scientific_confirmation

    too_few = _build_sealed_artifact(
        source_kind="real", rights_clean=True, reproductions=MIN_REPRODUCTIONS - 1
    )
    assert not verify_artifact(too_few).independent_scientific_confirmation

    noisy_tv_failed = _build_sealed_artifact(
        source_kind="real",
        rights_clean=True,
        reproductions=MIN_REPRODUCTIONS,
        noisy_tv_at_chance=False,
    )
    assert not verify_artifact(noisy_tv_failed).independent_scientific_confirmation


def test_verification_payload_is_sealed_and_reproducible() -> None:
    artifact = _build_sealed_artifact()
    payload_a = verification_payload(verify_artifact(artifact))
    payload_b = verification_payload(verify_artifact(copy.deepcopy(artifact)))
    assert payload_a == payload_b
    body = {k: v for k, v in payload_a.items() if k != "seal"}
    assert canonical_sha256(body) == payload_a["seal"]
    assert payload_a["independent_referee_reproduction"] is True
    assert payload_a["independent_scientific_confirmation"] is False


def test_malformed_artifact_is_refused() -> None:
    with pytest.raises(VerificationRefusal):
        verify_artifact({"schema": BED_SCHEMA})  # no per_seed, stats, or flags
    with pytest.raises(VerificationRefusal):
        verify_artifact("not a dict")  # type: ignore[arg-type]
