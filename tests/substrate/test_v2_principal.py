from __future__ import annotations

from substrate import v2config as C
from substrate import v2principal as P


def development_unit(arm: str, kind: str = "core") -> P.WorkUnit:
    return P._unit(
        f"test-{arm}-{kind}",
        "test",
        arm,
        C.SPLITS["development"][0],
        "development",
        "general",
        kind,
    )


def test_principal_dag_has_all_preregistered_matched_units():
    units = P.work_units()
    expected = (
        len(C.SPLITS["principal"]) * len(C.CORE_ARMS)
        + len(C.SPLITS["principal"]) * len(C.DIVERGENCE_ARMS)
        + len(C.SPLITS["replication"]) * 2
    )
    assert len(units) == expected == 360
    assert len({unit.identity for unit in units}) == len(units)
    assert len({unit.artifact_family for unit in units}) == len(units)
    assert all(unit.claim_ceiling == C.CLAIM_BOUNDARY["maximum"] for unit in units)


def test_full_history_runs_all_phases_with_transfer_continuity_and_no_activation():
    result = P.run_core(development_unit("full_v2"))
    payload = result["payload"]
    assert payload["episode_count"] == payload["expected_episode_count"] == sum(C.EPISODES_PER_PHASE.values())
    assert len(payload["phase_rows"]) == 11
    assert payload["identity_exact_every_phase"]
    assert payload["B_transfer_early_utility"] > 0.8
    assert payload["C_to_D_transfer_utility"] > 0.8
    assert payload["retention_loss"] <= C.SESOI
    assert not payload["negative_wrong_procedure_selected"]
    assert payload["body_continuity"]
    assert payload["interruption_recovery"]
    assert payload["procedures_induced"] >= 2
    assert payload["procedures_transferred"] >= 2
    assert payload["semantic_records"] >= 4
    assert payload["self_model_probe"]["margin"] > C.SESOI
    assert payload["activation"] is False


def test_procedural_transfer_beats_fresh_and_more_compute_cost_adjusted():
    full = P.run_core(development_unit("full_v2"))["payload"]
    fresh = P.run_core(development_unit("fresh_control"))["payload"]
    more = P.run_core(development_unit("more_compute"))["payload"]
    assert full["B_transfer_early_utility"] - fresh["B_transfer_early_utility"] > C.SESOI
    assert full["B_transfer_early_utility"] - more["B_transfer_early_utility"] > C.SESOI
    assert full["C_to_D_transfer_utility"] - fresh["C_to_D_transfer_utility"] > C.SESOI


def test_identical_divergence_histories_reproduce_exact_state():
    history = P.run_divergence(development_unit("history_A", "divergence"))["payload"]
    replica = P.run_divergence(
        development_unit("identical_history_A_replica", "divergence")
    )["payload"]
    history_b = P.run_divergence(development_unit("history_B", "divergence"))["payload"]
    assert history["state_identity"] == replica["state_identity"]
    assert history["state_identity"] != history_b["state_identity"]
    assert history["evaluation"]["B"] > history_b["evaluation"]["B"]
    assert history_b["evaluation"]["D"] > history["evaluation"]["D"]


def test_power_authority_meets_target_at_frozen_n():
    report = P._power([0.12, 0.23, 0.23, 0.23, 0.12, 0.23], len(C.SPLITS["principal"]))
    assert report["n"] == 24
    assert report["target_met"]
