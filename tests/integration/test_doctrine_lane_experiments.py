
import pytest

from mop import config, devices
from mop.experiments import get_experiment

DOCTRINE_LANE_IDS = ("d6_sensitive_window", "b8_structural_growth", "y4_hysteresis", "ex10_cross_modal")


@pytest.mark.parametrize("eid", DOCTRINE_LANE_IDS)
def test_doctrine_lane_experiment_runs(eid, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu"])
    dev = devices.resolve("cpu")
    out = get_experiment(eid).run(cfg, dev, tmp_path / eid)
    assert isinstance(out, dict), f"{eid} did not return a dict"
    assert "null_supported" in out and isinstance(out["null_supported"], bool)


def test_doctrine_lane_experiments_declare_contract():
    for eid in DOCTRINE_LANE_IDS:
        exp = get_experiment(eid)
        assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
