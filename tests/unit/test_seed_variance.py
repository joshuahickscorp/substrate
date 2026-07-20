
import math

from mop.studies.seed_variance import (
    CLAIM_TOL,
    SEED_GRID,
    _aggregate,
    _recommend,
    _sem,
    seed_variance,
)


def test_sem_matches_closed_form():
    xs = [0.1, 0.2, 0.3, 0.4]
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    assert math.isclose(_sem(xs), math.sqrt(var) / math.sqrt(len(xs)), rel_tol=1e-9)
    assert _sem([0.5]) == 0.0  # one seed: no measurable spread


def test_aggregate_uses_first_S_seeds_and_flags_sign():
    per_seed = [{"gap": g} for g in (0.05, 0.15, -0.05, 0.11, 0.10, 0.12)]
    rows = _aggregate(per_seed, (1, 2, 3, 6))
    by_S = {r["S"]: r for r in rows}
    assert by_S[2]["sign_stable"] is True
    assert math.isclose(by_S[2]["mean_gap"], 0.10, rel_tol=1e-9)
    assert by_S[3]["sign_stable"] is False
    assert by_S[1]["sign_stable"] is False
    assert by_S[6]["sem"] < by_S[3]["sem"]


def test_recommend_picks_smallest_qualifying_S():
    per_S = [
        {"S": 1, "mean_gap": 0.20, "sem": 0.00, "sign_stable": False},
        {"S": 2, "mean_gap": 0.40, "sem": 0.20, "sign_stable": True},
        {"S": 3, "mean_gap": 0.25, "sem": 0.12, "sign_stable": True},
        {"S": 5, "mean_gap": 0.26, "sem": 0.03, "sign_stable": True},
    ]
    assert _recommend(per_S, tol=0.05) == 5
    assert _recommend(per_S, tol=0.50) == 2


def test_recommend_falls_back_to_largest_when_nothing_qualifies():
    per_S = [
        {"S": 2, "mean_gap": 0.10, "sem": 0.30, "sign_stable": False},
        {"S": 3, "mean_gap": 0.50, "sem": 0.30, "sign_stable": False},
    ]
    assert _recommend(per_S, tol=0.01) == 3


def test_seed_variance_actually_reruns_e1():
    out = seed_variance(max_seeds=2, toy=True)
    assert out["provenance"] == "provisional"
    assert set(out["recommended_seeds"]) == set(CLAIM_TOL)
    assert len(out["per_seed"]) == 2
    for r in out["per_seed"]:
        assert math.isclose(r["gap"], r["protected_bwt"] - r["naive_bwt"], rel_tol=1e-9, abs_tol=1e-9)
    assert [r["S"] for r in out["per_S"]] == [s for s in SEED_GRID if s <= 2]
    assert all(r["S"] >= 0 for r in out["per_S"]) and out["per_S"][-1]["sem"] >= 0.0
    for v in out["recommended_seeds"].values():
        assert isinstance(v, int) and 1 <= v <= 2
