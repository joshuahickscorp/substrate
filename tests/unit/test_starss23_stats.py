from __future__ import annotations

import math

import pytest

from mop.evidence import canonical_sha256
from mop.science import statistics as stats


@pytest.mark.parametrize(
    "deltas,mean,p,permutations,significant,two_sided_reachable",
    [
        ([0.1, 0.2, 0.3, 0.4, 0.5], 0.3, 1 / 32, 32, True, False),
        ([-0.1, 0.2, 0.3, 0.4, 0.5], 0.26, 2 / 32, 32, False, False),
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.35, 1 / 64, 64, True, True),
    ],
)
def test_exact_sign_flip_enumerates_registered_tail(
    deltas, mean, p, permutations, significant, two_sided_reachable
):
    result = stats.exact_sign_flip(deltas)
    assert result.mean_delta == pytest.approx(mean)
    assert result.one_sided_p == pytest.approx(p)
    assert result.permutations == permutations
    assert result.one_sided_significant is significant
    assert result.two_sided_alpha_reachable is two_sided_reachable


@pytest.mark.parametrize(
    "deltas,alpha,message",
    [
        ([], 0.05, "at least one"),
        ([0.1, math.nan], 0.05, "finite"),
        ([0.1, True], 0.05, "real number"),
        ([0.1] * 21, 0.05, "capped"),
        ([0.1], 0.0, "probability"),
        ([0.1], True, "probability"),
    ],
)
def test_exact_sign_flip_refuses_invalid_contracts(deltas, alpha, message):
    with pytest.raises(stats.StatsRefusal, match=message):
        stats.exact_sign_flip(deltas, alpha)


def test_sesoi_boundary_and_refusals_are_fail_closed():
    assert stats.sesoi_exceeded(0.05, 0.05)
    assert not stats.sesoi_exceeded(0.049, 0.05)
    for effect, threshold in ((math.nan, 0.05), (0.05, 0.0), (0.05, math.inf), (0.05, True)):
        with pytest.raises(stats.StatsRefusal):
            stats.sesoi_exceeded(effect, threshold)


def test_count_projection_is_exact_and_rejects_shared_field_overlap():
    deltas = [0.2] * 5
    result = stats.exact_sign_flip(deltas)
    payload = stats.count_sign_flip_payload(
        result,
        deltas,
        sesoi=0.02,
        exceeds_sesoi=True,
        mean_candidate_minus_control=-0.2,
        prereg_digest="registered",
        extra={"guard": True},
    )
    assert canonical_sha256(payload) == "e5873e3b2ffe345109297113ac60370da5677fd27d49673d1efe8d1b5b6601a7"
    assert payload["mean_delta_control_minus_candidate"] == pytest.approx(0.2)
    assert payload["mean_delta_candidate_minus_control"] == pytest.approx(-0.2)
    with pytest.raises(stats.StatsRefusal, match="overlap"):
        stats.count_sign_flip_payload(
            result,
            deltas,
            sesoi=0.02,
            exceeds_sesoi=True,
            mean_candidate_minus_control=-0.2,
            prereg_digest="registered",
            extra={"deltas": []},
        )
