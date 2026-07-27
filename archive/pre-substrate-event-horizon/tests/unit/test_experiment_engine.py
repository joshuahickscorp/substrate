from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mop.beds.starss23.experiments import STARSS_RECORDS
from mop.devices import resolve
from mop.experiments.base import PROGRAM, RecordRefused, bind, interpret


def _row(**changes):
    row = {
        "id": "fixture",
        "name": "fixture",
        "question": "does the fixture preserve the declared metric",
        "null_hypothesis": "the fixture does not preserve the metric",
        "metrics": ["score"],
        "controls": ["control"],
        "source": {"kind": "fixture"},
        "split": {"rule": "fixed"},
        "unit": {"experimental": "seed"},
        "treatments": ["candidate"],
        "sesoi": {"value": 0.1},
        "multiplicity": {"rule": "none"},
        "budget": {"rule": "fixed"},
        "stop": {"rule": "one-pass", "tie": "null"},
        "claim_ceiling": {
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_confirmation": False,
        },
        "provider": "fixture.provider",
        "verifier": "fixture.verifier",
        "program": list(PROGRAM),
        "status": "implemented",
        "resource_tier": "cpu-now",
    }
    row.update(changes)
    return row


def _execute(_cfg, _device, _run_dir: Path):
    return {"score": 1.0}


def _verify(_result, _record):
    return {"verified": True, "independent_scientific_confirmation": False}


def test_sealed_record_interpreter_executes_bound_providers(tmp_path):
    spec = bind(_row(), _execute, _verify)
    result = interpret(spec, OmegaConf.create({}), resolve("cpu"), tmp_path)
    assert result == {"score": 1.0}
    assert spec.contract()["record_sha256"] == spec.record_sha256


def test_interpreter_refuses_authority_drift_and_program_reordering(tmp_path):
    spec = bind(_row(), _execute, _verify)
    spec.declaration["question"] = "drifted after sealing"
    with pytest.raises(RecordRefused, match="authority has drifted"):
        interpret(spec, OmegaConf.create({}), resolve("cpu"), tmp_path)
    with pytest.raises(RecordRefused, match="unknown or reordered"):
        bind(_row(program=list(reversed(PROGRAM))), _execute, _verify)


def test_interpreter_refuses_missing_metrics_and_failed_verification(tmp_path):
    cfg, device = OmegaConf.create({}), resolve("cpu")
    missing = bind(_row(), lambda *_args: {}, _verify)
    with pytest.raises(RecordRefused, match="omitted metrics"):
        interpret(missing, cfg, device, tmp_path)
    refused = bind(
        _row(),
        _execute,
        lambda *_args: {"verified": False, "independent_scientific_confirmation": False},
    )
    with pytest.raises(RecordRefused, match="verification provider refused"):
        interpret(refused, cfg, device, tmp_path)


def test_starss_four_axis_records_use_the_global_engine_and_remain_historical(tmp_path):
    assert {record.id for record in STARSS_RECORDS} == {
        "starss23_escs_event_formation",
        "starss23_escs_source_counting",
        "starss23_escs_direction_of_arrival",
        "starss23_escs_source_counting/data_split",
    }
    assert all(record.declaration["program"] == list(PROGRAM) for record in STARSS_RECORDS)
    with pytest.raises(RecordRefused, match="historical and not executable"):
        interpret(STARSS_RECORDS[0], OmegaConf.create({}), resolve("cpu"), tmp_path)
