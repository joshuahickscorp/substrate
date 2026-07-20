from __future__ import annotations

import math

import pytest

from mop.science import statistics as stats


def test_all_positive_deltas_hit_the_one_sided_floor() -> None:
    result = stats.exact_sign_flip([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result.n == 5
    assert result.permutations == 32
    assert result.count_ge_one_sided == 1
    assert result.one_sided_p == pytest.approx(1 / 32)
    assert result.min_one_sided_p == pytest.approx(1 / 32)
    assert result.count_ge_two_sided == 2
    assert result.two_sided_p == pytest.approx(2 / 32)
    assert result.two_sided_floor == pytest.approx(2 / 32)


def test_known_delta_vector_matches_hand_enumeration() -> None:
    result = stats.exact_sign_flip([-0.1, 0.2, 0.3, 0.4, 0.5])
    assert result.t_observed == pytest.approx(1.3)
    assert result.count_ge_one_sided == 2
    assert result.one_sided_p == pytest.approx(2 / 32)
    assert result.count_ge_two_sided == 4
    assert result.two_sided_p == pytest.approx(4 / 32)


def test_two_sided_005_is_unreachable_at_five_seeds() -> None:
    result = stats.exact_sign_flip([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result.two_sided_floor == pytest.approx(0.0625)
    assert result.two_sided_alpha_reachable is False
    assert stats.two_sided_alpha_reachable(5) is False


def test_two_sided_005_becomes_reachable_at_six_seeds() -> None:
    result = stats.exact_sign_flip([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert result.permutations == 64
    assert result.two_sided_floor == pytest.approx(2 / 64)
    assert result.two_sided_alpha_reachable is True
    assert stats.two_sided_alpha_reachable(6) is True


def test_one_sided_p_is_never_zero() -> None:
    result = stats.exact_sign_flip([0.9, 0.8, 0.7, 0.6, 0.5])
    assert result.one_sided_p >= result.min_one_sided_p
    assert result.one_sided_p == pytest.approx(1 / 32)


def test_phipson_smyth_correction_is_not_applied_to_full_enumeration() -> None:
    result = stats.exact_sign_flip([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result.phipson_smyth_applied is False


def test_below_sesoi_win_is_not_promotable() -> None:
    analysis = stats.analyze_paired_seeds([0.02, 0.02, 0.02, 0.02, 0.02], n_experimental_units=2)
    assert analysis.sign_flip.one_sided_significant is True
    assert analysis.sesoi.exceeds_sesoi is False
    assert analysis.meets_statistical_bar is False
    assert analysis.promotable is False


def test_above_sesoi_win_meets_the_statistical_bar_but_never_scientifically_promotes() -> None:
    analysis = stats.analyze_paired_seeds([0.08, 0.08, 0.08, 0.08, 0.08], n_experimental_units=2)
    assert analysis.sign_flip.one_sided_significant is True
    assert analysis.sesoi.exceeds_sesoi is True
    assert analysis.meets_statistical_bar is True
    assert analysis.promotable is True
    assert analysis.scientific_promotion is False
    assert analysis.independent_scientific_confirmation is False


def test_sesoi_default_is_the_provisional_registered_value() -> None:
    check = stats.sesoi_check(0.05)
    assert check.sesoi_f1 == pytest.approx(stats.PROVISIONAL_SESOI_F1)
    assert check.provisional is True
    assert check.exceeds_sesoi is True


def test_claim_ceiling_is_bounded_to_consistent_with() -> None:
    ceiling = stats.claim_ceiling(n_experimental_units=2, n_seeds=5)
    assert ceiling.experimental_unit == "clip"
    assert ceiling.claim_verb == "consistent with"
    assert "demonstrates" in ceiling.forbidden_verbs
    assert "significant" in ceiling.forbidden_verbs
    assert ceiling.frame_or_clip_bootstrap_allowed is False


def test_paired_deltas_are_candidate_minus_control() -> None:
    deltas = stats.paired_deltas([0.7, 0.6, 0.8], [0.5, 0.5, 0.5])
    assert deltas == pytest.approx((0.2, 0.1, 0.3))


def test_analyze_payload_round_trips_and_digest_is_deterministic() -> None:
    analysis = stats.analyze_paired_seeds([0.06, 0.07, 0.05, 0.08, 0.06], n_experimental_units=2)
    payload = analysis.payload()
    assert payload["schema"] == stats.STATS_SCHEMA
    assert payload["claim_scope"] == stats.CLAIM_SCOPE
    assert payload["scientific_promotion"] is False
    assert payload["sign_flip"]["permutations"] == 32
    digest = analysis.digest()
    assert len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    assert (
        stats.analyze_paired_seeds([0.06, 0.07, 0.05, 0.08, 0.06], n_experimental_units=2).digest() == digest
    )


def test_exact_sign_flip_rejects_empty_and_nonfinite_inputs() -> None:
    with pytest.raises(stats.StatsRefusal):
        stats.exact_sign_flip([])
    with pytest.raises(stats.StatsRefusal):
        stats.exact_sign_flip([0.1, math.nan, 0.2])
    with pytest.raises(stats.StatsRefusal):
        stats.exact_sign_flip([0.1, True, 0.2])


def test_paired_deltas_reject_mismatched_lengths() -> None:
    with pytest.raises(stats.StatsRefusal):
        stats.paired_deltas([0.1, 0.2], [0.1])


def test_claim_ceiling_rejects_nonpositive_counts() -> None:
    with pytest.raises(stats.StatsRefusal):
        stats.claim_ceiling(n_experimental_units=0, n_seeds=5)
    with pytest.raises(stats.StatsRefusal):
        stats.claim_ceiling(n_experimental_units=2, n_seeds=0)


def test_shared_sign_flip_projection_preserves_the_legacy_artifact_shape() -> None:
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = stats.exact_sign_flip(deltas)
    payload = stats.sign_flip_payload(
        result,
        deltas,
        sesoi_key="sesoi_f1",
        sesoi=0.05,
        exceeds_sesoi=True,
        provisional=False,
        prereg_digest="registered",
        extra={"beats_rate_matched_random": True},
    )
    assert payload == {
        "deltas": deltas,
        "t_obs": result.mean_delta,
        "one_sided_p": result.one_sided_p,
        "n_permutations": result.permutations,
        "two_sided_005_reachable": result.two_sided_alpha_reachable,
        "sesoi_f1": 0.05,
        "mean_delta_exceeds_sesoi": True,
        "claim_verb": "consistent with",
        "experimental_unit": "clip",
        "frame_or_clip_bootstrap_allowed": False,
        "sesoi_provisional": False,
        "prereg_canonical_sha256": "registered",
        "beats_rate_matched_random": True,
    }


def test_shared_count_projection_preserves_lower_is_better_fields() -> None:
    deltas = [0.2] * 5
    result = stats.exact_sign_flip(deltas)
    payload = stats.count_sign_flip_payload(
        result,
        deltas,
        sesoi=0.02,
        exceeds_sesoi=True,
        mean_candidate_minus_control=-0.2,
        prereg_digest="registered",
    )
    assert payload["metric"] == "coasted-count-MAE"
    assert payload["mean_delta_control_minus_candidate"] == result.mean_delta
    assert payload["mean_delta_candidate_minus_control"] == -0.2
    assert payload["sesoi_mae"] == 0.02
