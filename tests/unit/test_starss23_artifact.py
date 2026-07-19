
from __future__ import annotations

from mop.beds.starss23.artifact import (
    ARTIFACT_SCHEMA,
    PRIMARY_CONTROL,
    STAGE,
    build_bed_artifact,
)
from mop.beds.starss23.verifier import verify_artifact
from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_CLEARED,
    RunReceipt,
)


def test_artifact_schema_stage_and_claim_scope_are_the_frozen_contract(starss23_bed_artifact) -> None:
    a = starss23_bed_artifact.artifact
    assert a["schema"] == ARTIFACT_SCHEMA
    assert a["stage"] == STAGE == 3
    assert a["claim_scope"] == "deterministic programmatic mechanics only; no capability or natural-data claim"
    assert a["collar_frames"] == 2
    assert a["primary_control"] == PRIMARY_CONTROL == "rate_matched_random"
    assert a["source_kind"] == "synthetic"


def test_honesty_flags_are_hardcoded_off(starss23_bed_artifact) -> None:
    flags = starss23_bed_artifact.artifact["flags"]
    assert flags["activation_allowed"] is False
    assert flags["scientific_promotion"] is False
    assert flags["independent_scientific_confirmation"] is False


def test_five_paired_seeds_exact_sign_flip_floor(starss23_bed_artifact) -> None:
    stats = starss23_bed_artifact.artifact["stats"]
    assert len(stats["deltas"]) == 5
    assert stats["n_permutations"] == 32
    # All five paired deltas are positive, so the exact sign-flip hits its one-sided floor of 1/32.
    assert stats["one_sided_p"] == 1.0 / 32.0
    # Two-sided 0.05 is unreachable at n = 5 (discrete floor 2/32 = 0.0625).
    assert stats["two_sided_005_reachable"] is False
    assert stats["t_obs"] == sum(stats["deltas"]) / 5


def test_claim_verb_respects_the_clip_limited_ceiling(starss23_bed_artifact) -> None:
    stats = starss23_bed_artifact.artifact["stats"]
    assert stats["claim_verb"] == "consistent with"
    assert stats["experimental_unit"] == "clip"
    assert stats["frame_or_clip_bootstrap_allowed"] is False


def test_candidate_strictly_dominates_rate_matched_random(starss23_bed_artifact) -> None:
    a = starss23_bed_artifact.artifact
    assert a["harness"]["candidate_strictly_dominates_rate_matched_random"] is True
    # Every paired-seed delta (candidate minus rate-matched-random pooled F1) is positive.
    assert all(delta > 0 for delta in a["stats"]["deltas"])


def test_noisy_tv_control_is_at_chance(starss23_bed_artifact) -> None:
    controls = starss23_bed_artifact.artifact["controls"]
    assert controls["noisy_tv_at_chance"] is True
    # The pooled firing rate on the pure-aleatoric channel does not exceed the base rate preferentially.
    assert controls["mean_firing_rate_on_noise"] <= controls["mean_base_rate"] + 0.05 + 1e-9
    assert len(controls["per_seed_noisy_tv"]) == 5


def test_featurizer_is_zero_parameter_and_gate_is_the_only_trained_module(starss23_bed_artifact) -> None:
    a = starss23_bed_artifact.artifact
    assert a["featurizer"]["n_params"] == 0
    assert a["gate"]["params"] <= a["gate"]["param_ceiling"] == 4096
    assert a["gate"]["params"] == 3193
    assert a["gate"]["state_bytes"] <= 8192


def test_matched_budget_within_ceiling_and_break_even_present(starss23_bed_artifact) -> None:
    a = starss23_bed_artifact.artifact
    assert a["matched_budget"]["params"] == 3193
    assert a["matched_budget"]["flops"] <= 60_000_000_000
    assert a["matched_budget"]["seeds"] == 5
    # The candidate charges its amortized training cost, so a positive break-even exists.
    assert a["break_even"]["train_flops"] > 0
    assert a["break_even"]["amortizable"] is True
    assert a["full_scale_anchors"]["c_train_flops"] == 8_274_960_000


def test_per_seed_blocks_carry_raw_fires_for_every_arm(starss23_bed_artifact) -> None:
    a = starss23_bed_artifact.artifact
    arms = {"candidate", "rate_matched_random", "always_on", "best_single"}
    for block in a["per_seed"]:
        assert set(block["arm_scores"]) == arms
        for clip in block["clips"]:
            assert set(clip["fires"]) == arms
            assert isinstance(clip["gt_onsets"], list)
            # rate-matched-random fires the same count as the candidate on every clip.
            assert len(clip["fires"]["rate_matched_random"]) == len(clip["fires"]["candidate"])
            # always-on fires on every frame.
            assert len(clip["fires"]["always_on"]) >= len(clip["fires"]["candidate"])


def test_artifact_is_byte_reproducible(starss23_fast_config, starss23_bed_artifact) -> None:
    rebuilt = build_bed_artifact(starss23_fast_config)
    assert rebuilt.artifact["seal"] == starss23_bed_artifact.artifact["seal"]


def test_independent_verifier_reproduces_but_does_not_confirm(starss23_bed_artifact) -> None:
    result = verify_artifact(starss23_bed_artifact.artifact)
    # Mechanics reproduce: the seal, every score, and the stats are re-derived from specification.
    assert result.independent_referee_reproduction is True
    assert result.mismatches == ()
    # Science does not promote: synthetic data can never be independently confirmed.
    assert result.independent_scientific_confirmation is False
    assert result.source_kind == "synthetic"


def test_demonstration_receipt_is_mechanics_only_and_never_cleared(starss23_bed_artifact) -> None:
    payload = starss23_bed_artifact.artifact["demonstration_receipt"]
    assert payload["kind"] == KIND_DEMONSTRATION
    assert payload["verdict"] != VERDICT_CLEARED
    # The payload is a valid RunReceipt and, being a demonstration, can never be a confirmation.
    receipt = RunReceipt(
        kind=payload["kind"],
        mechanism_id=payload["mechanism_id"],
        stage=payload["stage"],
        requirement_id=payload["requirement_id"],
        verdict=payload["verdict"],
        controls_cleared=tuple(payload["controls_cleared"]),
        evidence_digest=payload["evidence_digest"],
        detail=payload["detail"],
    )
    assert receipt.is_confirmation is False
    assert receipt.stage == 3
    assert "noisy_tv" in receipt.controls_cleared
