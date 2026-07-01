"""Every cross-disciplinary roots experiment (N/D/B/P/C/I/Y/S/A series) runs end to end on its toy
default config and returns a dict with an explicit null check. This asserts MECHANICS only (the
experiment executes and reports null_supported), never a particular scientific outcome; the nulls may
hold (honest toy results). One parametrized case per registered roots experiment."""

import pytest

from devsys import config, devices
from devsys.experiments import REGISTRY, get_experiment


def _roots_ids() -> list[str]:
    # the roots experiments are every registered id that is not part of the E/EX conducted bank or the
    # I4 comparison (those have their own tests). Series letters n/d/b/p/c/i/y/s/a, ids like n1_*, d4_*.
    return sorted(eid for eid in REGISTRY if not eid.startswith("e") and eid != "i4_backprop_alts")


ROOTS_IDS = _roots_ids()


def test_roots_count():
    # the full-overkill cpu-now build registered the whole roots bank
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
    # the doctrine contract is enforced at class definition; confirm each roots experiment carries it
    for eid in ROOTS_IDS:
        exp = get_experiment(eid)
        assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
