
import pytest

from mop import config, devices
from mop.experiments import REGISTRY, get_experiment


def _roots_ids() -> list[str]:
    root_prefixes = ("n", "d", "b", "p", "c", "i", "y", "s", "a")
    return sorted(eid for eid in REGISTRY if eid.startswith(root_prefixes) and eid != "i4_backprop_alts")


ROOTS_IDS = _roots_ids()


def test_roots_count():
    assert len(ROOTS_IDS) >= 70, f"expected the ~77 roots experiments, got {len(ROOTS_IDS)}"


@pytest.mark.parametrize("eid", ROOTS_IDS)
def test_roots_experiment_runs(eid, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu"])
    dev = devices.resolve("cpu")
    out = get_experiment(eid).run(cfg, dev, tmp_path / eid)
    assert isinstance(out, dict), f"{eid} did not return a dict"
    assert "null_supported" in out, f"{eid} returned no explicit null check (null_supported)"
    assert isinstance(out["null_supported"], bool), f"{eid} null_supported is not a bool"


def test_roots_experiments_declare_contract():
    for eid in ROOTS_IDS:
        exp = get_experiment(eid)
        assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
