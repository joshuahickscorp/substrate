"""Tests for the STARSS23 ESCS bed matched-budget harness (spec component 6).

These pin the full-lifecycle FLOP accounting: the comparison refuses unless the candidate and the
matched control share byte-equal inference FLOPs and the candidate charges its amortized training cost
C_train; every arm stays under the ~6e10 ceiling; the break-even N* matches the recipe anchor; and the
strict-dominance verdict is capped at mechanics-ok and never opens a gate. House style: no em dashes
and no en dashes.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mop.beds.starss23 import FLOP_CEILING
from mop.beds.starss23.experiments import (
    FEATURIZE_FLOPS_PER_FRAME,
    GATE_INFER_FLOPS_PER_FRAME,
    GATE_PARAMS,
    MAX_GATE_PARAMS,
    ONSET_BUDGET_POLICY,
)
from mop.ladder.stage_ladder import MatchedBudget
from mop.science import budget as harness
from mop.science.budget import (
    TRAIN_BACKWARD_MULTIPLIER,
    Arm,
    BudgetPoint,
    ComputePoint,
    FlopModel,
    SeedResult,
)

SEEDS = (0, 1, 2, 3, 4)
TOTAL_FRAMES = 24_000
C_DOWN = 500_000
FEATURIZE = FEATURIZE_FLOPS_PER_FRAME * TOTAL_FRAMES
GATE_INFER = GATE_INFER_FLOPS_PER_FRAME * TOTAL_FRAMES
C_TRAIN = harness.gate_train_flops(8, 54_000, GATE_INFER_FLOPS_PER_FRAME)

CANDIDATE_FIRINGS = (1200, 1300, 1250, 1280, 1220)


def _seed_results(f1_by_seed, firings_by_seed) -> tuple[SeedResult, ...]:
    return tuple(
        SeedResult(seed=seed, metric_value=f1, actions=firings)
        for seed, f1, firings in zip(SEEDS, f1_by_seed, firings_by_seed, strict=True)
    )


def _candidate(f1_by_seed=(0.62, 0.60, 0.63, 0.61, 0.64), firings=CANDIDATE_FIRINGS) -> Arm:
    return Arm(
        policy=ONSET_BUDGET_POLICY,
        name="candidate_gate",
        kind=harness.ARM_CANDIDATE,
        total_frames=TOTAL_FRAMES,
        params=GATE_PARAMS,
        flop_model=FlopModel(FEATURIZE, GATE_INFER, C_DOWN, C_TRAIN),
        seed_results=_seed_results(f1_by_seed, firings),
    )


def _rate_matched_random(f1_by_seed=(0.30, 0.28, 0.31, 0.29, 0.32), firings=CANDIDATE_FIRINGS) -> Arm:
    # The rate-matched-random control runs an equal-cost inference stub (same FLOP model) and fires the
    # same count per seed as the candidate, but charges no training cost.
    return Arm(
        policy=ONSET_BUDGET_POLICY,
        name="rate_matched_random",
        kind=harness.ARM_RATE_MATCHED_RANDOM,
        total_frames=TOTAL_FRAMES,
        params=0,
        flop_model=FlopModel(FEATURIZE, GATE_INFER, C_DOWN, 0),
        seed_results=_seed_results(f1_by_seed, firings),
    )


def _always_on() -> Arm:
    return Arm(
        policy=ONSET_BUDGET_POLICY,
        name="always_on",
        kind=harness.ARM_ALWAYS_ON,
        total_frames=TOTAL_FRAMES,
        params=0,
        flop_model=FlopModel(FEATURIZE, 0, C_DOWN, 0),
        seed_results=_seed_results((0.66, 0.65, 0.67, 0.66, 0.68), (TOTAL_FRAMES,) * 5),
    )


def _best_single() -> Arm:
    return Arm(
        policy=ONSET_BUDGET_POLICY,
        name="best_single",
        kind=harness.ARM_BEST_SINGLE,
        total_frames=TOTAL_FRAMES,
        params=0,
        flop_model=FlopModel(FEATURIZE, 0, C_DOWN, 0),
        seed_results=_seed_results((0.45, 0.44, 0.46, 0.45, 0.47), (4000, 4200, 3900, 4100, 4050)),
    )


def _budget_point(
    budget_id="theta_0.50",
    candidate_f1=(0.62, 0.60, 0.63, 0.61, 0.64),
    rmr_f1=(0.30, 0.28, 0.31, 0.29, 0.32),
) -> BudgetPoint:
    return BudgetPoint(
        policy=ONSET_BUDGET_POLICY,
        budget_id=budget_id,
        candidate=_candidate(candidate_f1),
        rate_matched_random=_rate_matched_random(rmr_f1),
        always_on=_always_on(),
        reference=_best_single(),
    )


# ---------------------------------------------------------------------------
# Full-lifecycle FLOP accounting.
# ---------------------------------------------------------------------------


def test_lifecycle_flops_include_c_train() -> None:
    model = FlopModel(FEATURIZE, GATE_INFER, C_DOWN, C_TRAIN)
    firings = 1300
    assert model.run_flops(firings) == FEATURIZE + GATE_INFER + firings * C_DOWN
    # The full-lifecycle total is the inference run plus the amortized training cost.
    assert model.lifecycle_flops(firings) - model.run_flops(firings) == C_TRAIN


def test_candidate_stays_within_params_and_flop_ceiling() -> None:
    candidate = _candidate()
    assert candidate.params <= MAX_GATE_PARAMS
    # Every paired-seed run stays under the matched-budget ceiling.
    harness.assert_within_ceiling(candidate)
    assert candidate.max_lifecycle_flops() < FLOP_CEILING


def test_matched_ex_training_passes_for_identical_inference_flops() -> None:
    candidate = _candidate()
    rmr = _rate_matched_random()
    # No exception: same seeds, same firing counts, byte-equal inference FLOPs, candidate charges C_train.
    harness.assert_matched_ex_training(candidate, rmr)


def test_refuses_compare_when_inference_budgets_do_not_match() -> None:
    candidate = _candidate()
    # The control fires a different count on seed 0, so the inference budgets no longer match.
    rmr = _rate_matched_random(firings=(999, 1300, 1250, 1280, 1220))
    with pytest.raises(harness.BudgetMismatch):
        harness.assert_matched_ex_training(candidate, rmr)


def test_refuses_compare_when_downstream_cost_differs() -> None:
    candidate = _candidate()
    # Same firing counts, but a different per-firing downstream cost breaks the byte-equal inference FLOPs.
    rmr = Arm(
        policy=ONSET_BUDGET_POLICY,
        name="rate_matched_random",
        kind=harness.ARM_RATE_MATCHED_RANDOM,
        total_frames=TOTAL_FRAMES,
        params=0,
        flop_model=FlopModel(FEATURIZE, GATE_INFER, C_DOWN + 1, 0),
        seed_results=_seed_results((0.30, 0.28, 0.31, 0.29, 0.32), CANDIDATE_FIRINGS),
    )
    with pytest.raises(harness.BudgetMismatch):
        harness.assert_matched_ex_training(candidate, rmr)


def test_refuses_compare_when_ctrain_is_not_charged() -> None:
    # A candidate that fails to charge its amortized training cost is refused as dishonest accounting.
    candidate = Arm(
        policy=ONSET_BUDGET_POLICY,
        name="candidate_gate",
        kind=harness.ARM_CANDIDATE,
        total_frames=TOTAL_FRAMES,
        params=GATE_PARAMS,
        flop_model=FlopModel(FEATURIZE, GATE_INFER, C_DOWN, 0),
        seed_results=_seed_results((0.62, 0.60, 0.63, 0.61, 0.64), CANDIDATE_FIRINGS),
    )
    with pytest.raises(harness.UnchargedTraining):
        harness.assert_matched_ex_training(candidate, _rate_matched_random())


def test_ceiling_is_enforced() -> None:
    # A downstream cost large enough to blow the ceiling at the always-on firing count is refused.
    over = Arm(
        policy=ONSET_BUDGET_POLICY,
        name="candidate_gate",
        kind=harness.ARM_CANDIDATE,
        total_frames=TOTAL_FRAMES,
        params=GATE_PARAMS,
        flop_model=FlopModel(FEATURIZE, GATE_INFER, 3_000_000, C_TRAIN),
        seed_results=_seed_results((0.62, 0.60, 0.63, 0.61, 0.64), (TOTAL_FRAMES,) * 5),
    )
    with pytest.raises(harness.CeilingExceeded):
        harness.assert_within_ceiling(over)


def test_certify_refuses_a_control_that_charges_training() -> None:
    trained_control = Arm(
        policy=ONSET_BUDGET_POLICY,
        name="rate_matched_random",
        kind=harness.ARM_RATE_MATCHED_RANDOM,
        total_frames=TOTAL_FRAMES,
        params=0,
        flop_model=FlopModel(FEATURIZE, GATE_INFER, C_DOWN, C_TRAIN),
        seed_results=_seed_results((0.30, 0.28, 0.31, 0.29, 0.32), CANDIDATE_FIRINGS),
    )
    point = BudgetPoint(
        policy=ONSET_BUDGET_POLICY,
        budget_id="theta_0.50",
        candidate=_candidate(),
        rate_matched_random=trained_control,
        always_on=_always_on(),
        reference=_best_single(),
    )
    with pytest.raises(harness.BudgetMismatch):
        point.certify()


# ---------------------------------------------------------------------------
# Break-even accounting.
# ---------------------------------------------------------------------------


def test_gate_train_flops_matches_the_spec_anchor() -> None:
    # C_train = E * F_train_frames * 3 * infer = 8 * 54000 * 3 * 6385.
    assert harness.gate_train_flops(8, 54_000, GATE_INFER_FLOPS_PER_FRAME) == 8_274_960_000


def test_per_query_saving_formula() -> None:
    # (F - K) / F * C_down with F = 24000, K = 4800 (20 percent firing), C_down = 50000 gives 40000.
    saving = harness.per_query_saving_vs_always_on(24_000, 4_800, 50_000)
    assert saving == pytest.approx(40_000.0)


def test_break_even_matches_the_recipe_anchor() -> None:
    # N* = C_train / per_query_saving = 8.27e9 / 4e4, which is about 2.07e5 frames.
    saving = harness.per_query_saving_vs_always_on(24_000, 4_800, 50_000)
    break_even = harness.break_even_queries(
        harness.gate_train_flops(8, 54_000, GATE_INFER_FLOPS_PER_FRAME), saving
    )
    assert break_even.amortizable is True
    assert break_even.n_star_frames == 206_874
    assert break_even.n_star_frames == pytest.approx(2.07e5, rel=0.01)


def test_break_even_is_unamortizable_without_a_saving() -> None:
    # If the gate never saves downstream compute there is no break-even.
    break_even = harness.break_even_queries(C_TRAIN, 0.0)
    assert break_even.amortizable is False
    assert break_even.n_star_frames is None


# ---------------------------------------------------------------------------
# Pareto and strict dominance.
# ---------------------------------------------------------------------------


def test_pareto_frontier_drops_dominated_points() -> None:
    points = [
        ComputePoint(ONSET_BUDGET_POLICY, "a", harness.ARM_CANDIDATE, 1.0e10, 0.50, 3193, 1000.0),
        ComputePoint(ONSET_BUDGET_POLICY, "b", harness.ARM_ALWAYS_ON, 2.0e10, 0.60, 0, 24000.0),
        ComputePoint(ONSET_BUDGET_POLICY, "c", harness.ARM_BEST_SINGLE, 2.0e10, 0.40, 0, 4000.0),
    ]
    frontier = harness.pareto_frontier(points)
    kinds = {point.budget_id for point in frontier}
    # Point c is dominated by b (same FLOPs, higher F1); a and b are non-dominated.
    assert kinds == {"a", "b"}
    assert [point.budget_id for point in frontier] == ["a", "b"]


def test_paired_deltas_are_candidate_minus_control() -> None:
    deltas = harness.paired_deltas(_candidate(), _rate_matched_random())
    assert all(delta > 0 for delta in deltas)
    assert len(deltas) == 5


def test_conventional_seed_projection_matches_the_direct_budget_point() -> None:
    direct = _budget_point()
    runs = []
    for seed in SEEDS:
        runs.append(SimpleNamespace(
            seed=seed, total_frames=TOTAL_FRAMES, gate_params=GATE_PARAMS,
            per_budget={direct.budget_id: {
                "arm_scores": {arm.kind: {"f1": arm.result_for_seed(seed).metric_value}
                               for arm in direct.arms()},
                "firings": {arm.kind: arm.result_for_seed(seed).actions for arm in direct.arms()},
            }},
        ))
    models = {arm.kind: arm.flop_model for arm in direct.arms()}
    projected = harness.build_budget_points(
        ONSET_BUDGET_POLICY, runs, score_group="arm_scores", score_field="f1",
        action_group="firings", flop_model=models.__getitem__,
    )
    expected = BudgetPoint(
        policy=direct.policy,
        budget_id=direct.budget_id,
        candidate=replace(direct.candidate, name=f"candidate@{direct.budget_id}"),
        rate_matched_random=replace(
            direct.rate_matched_random, name=f"rate_matched_random@{direct.budget_id}"
        ),
        always_on=replace(direct.always_on, name=f"always_on@{direct.budget_id}"),
        reference=replace(direct.reference, name=f"best_single@{direct.budget_id}"),
    )
    assert projected == [expected]


def test_noise_control_projection_preserves_policy_order_and_sealed_rates() -> None:
    runs = [SimpleNamespace(noisy_tv={"seed": seed}) for seed in (0, 1)]
    block = harness.noise_control_summary(
        ONSET_BUDGET_POLICY, runs, at_chance=True, mean_noise_rate=0.1234567890124,
        mean_base_rate=0.2, rate_key="mean_firing_rate_on_noise",
    )
    assert block == {
        "noisy_tv_at_chance": True,
        "mean_firing_rate_on_noise": 0.123456789012,
        "mean_base_rate": 0.2,
        "per_seed_noisy_tv": [{"seed": 0}, {"seed": 1}],
        "primary_control": harness.ARM_RATE_MATCHED_RANDOM,
        "control_arms": [*ONSET_BUDGET_POLICY.controls, "noisy_tv"],
    }
    with pytest.raises(harness.BudgetRefusal, match="rate_key"):
        harness.noise_control_summary(
            ONSET_BUDGET_POLICY, runs, at_chance=True, mean_noise_rate=0.1,
            mean_base_rate=0.1, rate_key="unknown",
        )


def test_candidate_strictly_dominates_rate_matched_random_when_it_wins_everywhere() -> None:
    report = harness.run_matched_budget([_budget_point()], wall_ns=1_000_000)
    assert report.candidate_strictly_dominates_rate_matched_random is True
    assert report.verdict == "mechanics-ok"


def test_candidate_does_not_dominate_when_it_loses_at_a_budget() -> None:
    # Swap the F1: here the random control matches or beats the candidate, so dominance fails.
    losing = _budget_point(
        candidate_f1=(0.30, 0.28, 0.31, 0.29, 0.32),
        rmr_f1=(0.62, 0.60, 0.63, 0.61, 0.64),
    )
    report = harness.run_matched_budget([losing], wall_ns=1_000_000)
    assert report.candidate_strictly_dominates_rate_matched_random is False
    assert report.verdict == "null"


# ---------------------------------------------------------------------------
# Report assembly, matched budget bridge, and hardcoded boundary flags.
# ---------------------------------------------------------------------------


def test_report_bridges_to_a_positive_matched_budget() -> None:
    report = harness.run_matched_budget([_budget_point()], wall_ns=1_000_000)
    matched = report.matched_budget
    assert isinstance(matched, MatchedBudget)
    assert matched.params == GATE_PARAMS
    assert matched.flops > 0
    assert matched.wall_ns == 1_000_000
    assert matched.seeds == 5
    assert matched.flops < FLOP_CEILING


def test_report_refuses_a_nonpositive_wall_clock() -> None:
    with pytest.raises(harness.BudgetRefusal):
        harness.run_matched_budget([_budget_point()], wall_ns=0)


def test_report_hardcodes_the_boundary_flags() -> None:
    report = harness.run_matched_budget([_budget_point()], wall_ns=1_000_000)
    payload = report.payload()
    assert payload["activation_allowed"] is False
    assert payload["scientific_promotion"] is False
    assert payload["independent_scientific_confirmation"] is False
    assert payload["source_kind"] == "synthetic"
    assert payload["claim_scope"] == ONSET_BUDGET_POLICY.claim_scope
    # A synthetic run can never be cleared: the verdict is capped at mechanics-ok.
    assert payload["verdict"] in ("mechanics-ok", "null")


def test_report_digest_is_deterministic() -> None:
    first = harness.run_matched_budget([_budget_point()], wall_ns=1_000_000)
    second = harness.run_matched_budget([_budget_point()], wall_ns=1_000_000)
    digest = first.digest()
    assert len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    assert digest == "7da5162d604073fa63b3c0663b2b212bc1aaa428f70248a5d881c59b4c62e160"
    assert first.digest() == second.digest()


def test_run_refuses_duplicate_budget_ids() -> None:
    with pytest.raises(harness.BudgetRefusal):
        harness.run_matched_budget(
            [_budget_point("theta_0.50"), _budget_point("theta_0.50")], wall_ns=1_000_000
        )


def test_run_refuses_an_empty_sweep() -> None:
    with pytest.raises(harness.BudgetRefusal):
        harness.run_matched_budget([], wall_ns=1_000_000)


# ---------------------------------------------------------------------------
# Cross-module anchor consistency. The harness carries FLOP-count anchors only for defaults and tests;
# the authoritative counts belong to the featurizer and gate modules. When those siblings are present,
# assert the anchors have not drifted. Guarded so this suite stays green if a sibling is mid-build.
# ---------------------------------------------------------------------------


def test_flop_anchors_match_the_featurizer_and_gate_modules() -> None:
    featurizer = pytest.importorskip("mop.beds.starss23.featurizer")
    gate = pytest.importorskip("mop.beds.starss23.gate")
    assert FEATURIZE_FLOPS_PER_FRAME == featurizer.FLOPS_PER_FRAME
    assert harness.featurize_run_flops(
        TOTAL_FRAMES, FEATURIZE_FLOPS_PER_FRAME
    ) == featurizer.FLOPS_PER_FRAME * TOTAL_FRAMES
    assert gate.inference_flops() == GATE_INFER_FLOPS_PER_FRAME
    assert gate.param_count() == GATE_PARAMS
    assert MAX_GATE_PARAMS == gate.PARAM_CEILING
    assert TRAIN_BACKWARD_MULTIPLIER == gate.TRAIN_STEP_FACTOR
    # C_train anchor: 8 * 54000 * 3 * 6385 = 8_274_960_000, identical to the gate module.
    assert harness.gate_train_flops(
        8, 54_000, GATE_INFER_FLOPS_PER_FRAME
    ) == gate.C_TRAIN_ANCHOR
