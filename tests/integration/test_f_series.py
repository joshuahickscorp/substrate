"""Harness-level gates for the runnable form-substrate experiments (F-series).

The E and EX series are exercised through the runner; the F-series had only unit tests on
`substrate/form.py`. This closes that parity gap: every IMPLEMENTED F experiment (read live from the
registry, so the list never drifts) is composed from its default config, run through
`get_experiment(...).run(...)`, and checked for MECHANICS only, never a particular scientific outcome
(a null may hold, that is an honest bound, not a failure). Every result must also carry the
performance-density block (Layer 9, PERFORMANCE_DENSITY_DOCTRINE.md).
"""

import pytest

from mop import config, devices
from mop.devel.registries import load_experiments
from mop.diagnostics.performance_density import DENSITY_SCHEMA
from mop.experiments import REGISTRY, get_experiment
from mop.falsification.experiment_contracts import build_contract_audit


def _implemented_f_ids():
    ids = []
    for e in load_experiments():
        if e.get("series") == "F" and e.get("status") == "implemented" and e.get("id") in REGISTRY:
            ids.append(e["id"])
    return sorted(ids)


F_IDS = _implemented_f_ids()


def _run(eid, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu"])
    dev = devices.resolve("cpu")
    return get_experiment(eid).run(cfg, dev, tmp_path / eid)


def test_f_series_is_non_empty():
    assert F_IDS, "no implemented F-series experiments found in the registry"


def test_f_series_registry_class_and_config_contracts_are_exact():
    audit = build_contract_audit(series="F", implemented_only=False)
    assert audit["all_ok"], "\n".join(audit["problems"])
    preregistrations = [record for record in audit["records"] if record.get("preregistration_only") is True]
    assert preregistrations
    assert all(not record["required"] and not record["comparisons"] for record in preregistrations)


@pytest.mark.parametrize("eid", F_IDS)
def test_f_experiment_runs_and_declares_null(eid, tmp_path):
    out = _run(eid, tmp_path)
    assert isinstance(out, dict), f"{eid} did not return a dict"
    assert "null_supported" in out, f"{eid} returned no explicit null check"
    assert isinstance(out["null_supported"], bool), f"{eid} null_supported is not a bool"


@pytest.mark.parametrize("eid", F_IDS)
def test_f_experiment_reports_declared_metrics(eid, tmp_path):
    out = _run(eid, tmp_path)
    exp = get_experiment(eid)
    for name in exp.metric:
        assert name in out, f"{eid} did not report its declared metric {name!r}"


@pytest.mark.parametrize("eid", F_IDS)
def test_f_experiment_reports_density_block(eid, tmp_path):
    out = _run(eid, tmp_path)
    density = out.get("density")
    assert isinstance(density, dict), f"{eid} carries no density block"
    assert density["schema"] == DENSITY_SCHEMA, f"{eid} density schema is wrong"
    assert density["capability"], f"{eid} density block has no capability"
    assert density["cost"], f"{eid} density block has no cost"
    assert density["density"], f"{eid} density block reports no ratio (all costs zero?)"


@pytest.mark.parametrize("eid", F_IDS)
def test_f_experiment_declares_contract(eid):
    exp = get_experiment(eid)
    assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
