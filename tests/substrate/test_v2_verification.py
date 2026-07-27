from __future__ import annotations

from substrate import v2config as C
from substrate import v2stats as ST
from substrate import v2verify as V


def test_paired_statistics_are_deterministic_and_use_independent_units():
    values = [0.1 + index / 1000 for index in range(24)]
    first = ST.paired(values, "determinism")
    second = ST.paired(values, "determinism")
    assert first == second
    assert first["n"] == 24
    assert first["bootstrap_95_ci"][0] > 0
    assert first["mean"] > C.SESOI
    assert first["exact_sign_p"] < 0.05


def test_holm_stops_rejection_after_first_failure():
    report = ST.holm({"H_D1": 0.001, "H_D2": 0.04, "H_D3": 0.06})
    assert report["rows"]["H_D1"]["reject_zero"]
    assert not report["rows"]["H_D2"]["reject_zero"]
    assert not report["rows"]["H_D3"]["reject_zero"]


def test_terminal_required_surface_matches_declared_families():
    assert len(V.REQUIRED) == len(set(V.REQUIRED))
    assert "SUBSTRATE_V2_INDEPENDENT_VERIFICATION.json" in V.REQUIRED
    assert "SUBSTRATE_V2_FINAL_STATE.json" in V.REQUIRED
    assert V.PREPRINCIPAL_REQUIRED[-1] == "SUBSTRATE_V2_WORKER_AUTHORITY.json"


def test_allocation_below_sesoi_blocks_reflective_but_not_persistent_classification():
    positive = {"passes": True}
    metrics = {
        "identity_exact": True,
        "interference": 0.0,
        "body_continuity": True,
        "interruption_recovery": True,
        "gates": {
            "H_D1": positive,
            "H_D2": {**positive, "negative_clean": True},
            "H_D3": {"passes": False},
            "H_D4": positive,
            "H_D5": positive,
        },
    }
    report = V.closure(metrics, independent_pass=True, mutation_pass=True)
    assert report["classification"] == "persistent_developmental_cognition"
    assert report["levels"]["persistent_developmental_cognition"]
    assert not report["levels"]["reflective_cognitive_organization"]
    assert not report["levels"]["functional_or_proto_nous_candidate"]
