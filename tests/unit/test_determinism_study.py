
from __future__ import annotations

import math

from mop.studies.determinism_study import CONFIGS, _characterize, determinism_study


def test_study_runs_both_configs_and_reports_spread():
    out = determinism_study(reps=2, toy=True)
    assert out["provenance"] == "provisional"
    assert out["num_threads"] == 1
    assert out["cpu_more_deterministic_than_metal"] is True
    names = {c["name"] for c in out["configs"]}
    assert names == set(CONFIGS)  # both e1 and i4 actually characterized

    for c in out["configs"]:
        assert c["runs"] == 2
        assert len(c["metric_values"]) == 2  # the experiment really ran twice
        assert all(math.isfinite(v) for v in c["metric_values"])
        assert c["max_abs"] >= 0.0 and c["std"] >= 0.0
        assert 0.0 <= c["byte_identical_rate"] <= 1.0
        if c["byte_identical"]:
            assert c["max_abs"] == 0.0


def test_e1_metric_is_protected_bwt_and_finite():
    c = _characterize("e1", CONFIGS["e1"], reps=2)
    assert c["metric"] == "gate.protected_bwt"
    assert all(math.isfinite(v) for v in c["metric_values"])


def test_cpu_serial_is_near_bit_identical():
    out = determinism_study(reps=3, toy=True)
    assert all(c["byte_identical"] for c in out["configs"]), out
    assert all(c["max_abs"] == 0.0 for c in out["configs"])
