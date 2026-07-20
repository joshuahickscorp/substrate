
import pytest

from mop import config, devices
from mop.experiments import get_experiment

SCAFFOLDING_IDS = (
    "ex1_generative_replay",
    "ex4_fast_weights",
    "ex6_active_inference",
    "ex7_meta_learning",
    "ex11_causal_probing",
    "ex14_memory_bakeoff",
    "ex18_self_verification",
)


@pytest.mark.parametrize("eid", SCAFFOLDING_IDS)
def test_scaffolding_experiment_runs(eid, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu"])
    dev = devices.resolve("cpu")
    out = get_experiment(eid).run(cfg, dev, tmp_path / eid)
    assert isinstance(out, dict), f"{eid} did not return a dict"
    assert "null_supported" in out, f"{eid} returned no explicit null check (null_supported)"
    assert isinstance(out["null_supported"], bool), f"{eid} null_supported is not a bool"


def test_scaffolding_experiments_declare_contract():
    for eid in SCAFFOLDING_IDS:
        exp = get_experiment(eid)
        assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
