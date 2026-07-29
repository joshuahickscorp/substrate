"""Tests for the genesis modules that decide the scientific result."""

from __future__ import annotations

import pytest

from substrate import genesis_campaign as CAMP
from substrate import genesis_claims as CL
from substrate import genesis_config as C
from substrate import genesis_harness as H
from substrate import genesis_history as HI
from substrate import genesis_io as io
from substrate import genesis_material as M
from substrate import genesis_mutations as MU
from substrate import genesis_publication as P
from substrate import genesis_statistics as S
from substrate import genesis_tournament as T
from substrate import genesis_verification as V


def test_module_self_checks_pass() -> None:
    """Every module carries a runnable check; all of them must hold."""
    for module in (S, H, HI, MU, CL, V, P, T):
        module.demo()


# -- constitution ----------------------------------------------------------


def test_s2_has_one_canonical_identity() -> None:
    assert C.CANONICAL_S2_ID in C.CONTROLS
    for alias, target in C.S2_ALIASES.items():
        assert target == C.CANONICAL_S2_ID, alias


def test_every_candidate_and_baseline_is_declared_once() -> None:
    assert len(set(C.CANDIDATES)) == len(C.CANDIDATES) == 11
    assert len(set(C.BASELINES)) == len(C.BASELINES)
    assert len(set(C.REVIEW_CELLS)) == len(C.REVIEW_CELLS)
    assert len(set(C.MUTATIONS)) == len(C.MUTATIONS)
    assert len(C.CHALLENGE_FAMILIES) >= C.TOURNAMENT_MINIMUM_FAMILIES


def test_thresholds_are_not_relaxed() -> None:
    assert C.SESOI >= 0.05
    assert C.OUTCOME_A_REQUIREMENTS["decisive_effect_minimum"] >= 0.05
    assert C.OUTCOME_A_REQUIREMENTS["surviving_mutations"] == 0
    assert C.MINIMUM_ORACLE_HEADROOM >= 0.05
    assert C.CLAIM_BOUNDARY["unqualified_nous"] is False
    assert C.CLAIM_BOUNDARY["external_activation"] is False


def test_every_baseline_declares_its_deprivation() -> None:
    for baseline in C.BASELINES:
        assert baseline in C.BASELINE_DEPRIVATION, baseline
    assert C.BASELINE_DEPRIVATION[C.CANONICAL_S2_ID] == ()


# -- sealing ---------------------------------------------------------------


def test_activation_cannot_be_written() -> None:
    with pytest.raises(io.Refused):
        io.write_json(io.RUNS / "activation_probe.json", {"activation": True})
    with pytest.raises(io.Refused):
        io.write_json(io.RUNS / "activation_probe.json", {"rows": [{"external_activation": True}]})


def test_probe_splits_must_be_disjoint() -> None:
    probes = tuple(M.Probe(index, "f", "c", (0,), 1) for index in range(4))
    H.ProbeSplit(probes[0:1], probes[1:2], probes[2:3])
    with pytest.raises(H.HarnessRefused):
        H.ProbeSplit(probes[0:2], probes[1:3], probes[3:4])


def test_wrong_history_must_actually_differ() -> None:
    observations = [M.Observation(index, "c", (index % 3 - 1,)) for index in range(10)]
    with pytest.raises(H.HarnessRefused):
        H.wrong_stream(observations, observations)

    barely = list(observations)
    barely[0] = M.Observation(0, "c", (2,))
    with pytest.raises(H.HarnessRefused):
        H.wrong_stream(observations, barely)

    genuinely = [M.Observation(index, "d", ((index + 1) % 3 - 1,)) for index in range(10)]
    assert H.wrong_stream(observations, genuinely).transform == "wrong_history"


def test_splits_use_disjoint_history_bands() -> None:
    bands = {split: set(CAMP.split_histories(split, 64)) for split in CAMP.HISTORY_BANDS}
    for left in bands:
        for right in bands:
            if left < right:
                assert not bands[left] & bands[right], (left, right)


# -- statistics ------------------------------------------------------------


def test_a_true_null_is_reported_as_null() -> None:
    scores = [S.HistoryScore(index, arm, 0.5) for index in range(32) for arm in ("K1", C.CANONICAL_S2_ID)]
    analysis = S.decisive_analysis(scores, candidate="K1", comparator=C.CANONICAL_S2_ID)
    assert analysis["effect"] == 0.0
    assert not analysis["primary_gate_pass"]


def test_unpaired_histories_are_refused() -> None:
    scores = [S.HistoryScore(index, "K1", 0.6) for index in range(32)]
    scores += [S.HistoryScore(index, C.CANONICAL_S2_ID, 0.5) for index in range(16)]
    with pytest.raises(S.AnalysisRefused):
        S.decisive_analysis(scores, candidate="K1", comparator=C.CANONICAL_S2_ID)


def test_holm_correction_orders_and_rejects() -> None:
    corrected = S.holm({"P1": 0.0001, "P2": 0.30, "P10": 0.02})
    assert corrected["P1"]["rejected"]
    assert not corrected["P2"]["rejected"]
    assert corrected["P1"]["rank"] == 1


def test_decisive_comparator_must_be_a_registered_plastic_control() -> None:
    scores = [S.HistoryScore(index, "some_unregistered_arm", 0.9) for index in range(4)]
    with pytest.raises(S.AnalysisRefused):
        S.resolve_decisive_comparator(scores, parity_passed={}, separate_implementation={})


# -- mutations -------------------------------------------------------------


def test_mutation_suite_has_no_survivors_and_no_undeclared_entries() -> None:
    report = MU.run()
    assert report["survivors"] == []
    assert report["undeclared"] == []
    assert report["injected_count"] >= 24
    assert report["pending_count"] + report["injected_count"] == len(C.MUTATIONS)


def test_pending_mutations_are_never_reported_as_caught() -> None:
    report = MU.run()
    for row in report["pending"]:
        assert row["caught"] is False
        assert row["pending"] is True


# -- classification --------------------------------------------------------


def test_a_broken_prerequisite_outranks_a_passing_gate() -> None:
    prerequisites = dict.fromkeys(P.PREREQUISITES, True)
    prerequisites["mutations_zero_survivors"] = False
    gates = dict.fromkeys(P.OUTCOME_A_GATES, True)
    assert P.classify(prerequisites=prerequisites, gates=gates)["outcome"] == "C"


def test_a_sound_program_with_a_null_is_outcome_b_not_c() -> None:
    prerequisites = dict.fromkeys(P.PREREQUISITES, True)
    gates = dict.fromkeys(P.OUTCOME_A_GATES, True)
    gates["decisive_effect_at_least_sesoi"] = False
    result = P.classify(prerequisites=prerequisites, gates=gates)
    assert result["outcome"] == "B"
    assert result["classification"] == "cognitive_material_foundation_complete"


def test_no_outcome_assigns_unqualified_nous() -> None:
    prerequisites = dict.fromkeys(P.PREREQUISITES, True)
    for gates in (
        dict.fromkeys(P.OUTCOME_A_GATES, True),
        {**dict.fromkeys(P.OUTCOME_A_GATES, True), "replication_positive": False},
    ):
        result = P.classify(prerequisites=prerequisites, gates=gates)
        assert result["unqualified_nous_assigned"] is False
        assert result["external_activation"] is False
        assert result["nous_status"] == C.STARTING_NOUS_STATUS


# -- verification ----------------------------------------------------------


def test_recomputation_catches_a_published_effect_that_does_not_match() -> None:
    rows = [
        {"history_id": index, "arm": arm, "score": 0.5}
        for index in range(32)
        for arm in ("K1", C.CANONICAL_S2_ID)
    ]
    recomputed = V.recompute_decisive_effect(rows, candidate="K1", comparator=C.CANONICAL_S2_ID)
    assert not V.agrees_with({"effect": 0.3, "confidence_lower": 0.2}, recomputed)["all_pass"]
    assert V.agrees_with({"effect": 0.0, "confidence_lower": 0.0}, recomputed)["all_pass"]


def test_every_material_survives_a_precision_demotion() -> None:
    """A narrowed alphabet must not leave stale wide values in packed state.

    K11 wrote payload values at quinary, was demoted to a narrower precision,
    and then refused to serialize because the stored values no longer fitted
    the alphabet. That took down a full tournament run partway through. Every
    material is checked, not only the one that failed.
    """
    import substrate.genesis_controls  # noqa: F401
    import substrate.genesis_k_advanced  # noqa: F401
    import substrate.genesis_k_basic  # noqa: F401
    import substrate.genesis_k_structural  # noqa: F401

    observations = [
        M.Observation(index, f"c{index % 3}", (index % 5 - 2, (index * 3) % 5 - 2), elapsed_ms=5, teaching=index % 4 == 0)
        for index in range(48)
    ]
    for name in M.registered():
        opportunity = M.equal_opportunity(
            envelope="1GB",
            observations=observations,
            sensor_channels=("c0", "c1", "c2"),
            operation_budget=2_000_000,
            durable_write_budget=4_096,
        )
        material = M.build(name, opportunity)
        for observation in observations:
            material.observe(observation)
        for proposal in material.propose():
            material.apply([M.Verdict(proposal.proposal_id, True, 1.0, 0.0)])
        # Serialising is where an out-of-alphabet value surfaces.
        checkpoint = material.checkpoint()
        assert checkpoint["mechanism"] == material.mechanism, name
        assert material.durable_state_digest(), name


def test_a_counterfeit_that_beats_the_selection_is_reported() -> None:
    rows = [{"arm": "K1", "score": 0.2}, {"arm": "record_store_null", "score": 0.8}]
    report = V.counterfeit_report(rows, selected="K1")
    assert "record_store_null" in report["surviving_counterfeits"]
    assert not report["all_pass"]
