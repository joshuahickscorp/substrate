"""Unit tests for the Generation 1 General Run closure program (Layer 1).

The live General Run is not terminal while these tests run, so the honest current output is a deferred
closure with ``admitted=false`` and a non-empty refusal list. The tests exercise:

(a) admission refuses on a fixture non-terminal general-run status;
(b) run_closure against the LIVE General Run root seals a deferred closure plus a current frontier;
(c) the structurally separate verifier reports seal_intact and detects a mutated admitted flag;
(d) the rendered report contains the refusal.

Outputs are written to a temporary proof directory so no committed proof is overwritten.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mop.closure import admission as admission_module
from mop.closure.producer import run_closure
from mop.closure.report import render_closure_report
from mop.closure.verifier import verify_closure

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_TREE = Path("/Users/scammermike/Downloads/mop")
LIVE_RUNS = LIVE_TREE / "runs"
LIVE_GR = LIVE_RUNS / "generation1" / "general-run"
TIMESTAMP = "2026-07-18T00:00:00+00:00"

_LIVE_PRESENT = (LIVE_GR / "current_status.json").is_file()
_requires_live = pytest.mark.skipif(
    not _LIVE_PRESENT, reason="the live General Run tree is not present in this environment"
)


def test_admission_refuses_on_nonterminal_status(monkeypatch, tmp_path):
    """Admission must fail closed on a non-terminal general-run status."""

    fixture_status = {
        "schema": "mop-general-run-status/v1",
        "program_id": "generation1-general-run",
        "state": "run_horizon_v2",
        "status_sha256": "0" * 64,
    }
    monkeypatch.setattr(admission_module, "read_general_run_status", lambda root: fixture_status)
    monkeypatch.setattr(
        admission_module,
        "validate_general_run_status",
        lambda status, repo_root=None: "run_horizon_v2",
    )

    decision = admission_module.evaluate_admission(root=tmp_path, repo_root=tmp_path, now_iso=TIMESTAMP)

    assert decision.admitted is False
    assert decision.refusals
    assert any("not terminal" in refusal for refusal in decision.refusals)


@_requires_live
def test_run_closure_seals_deferred_closure_and_frontier(tmp_path):
    """run_closure against the live non-terminal General Run seals a deferred closure and a frontier."""

    summary = run_closure(
        repo_root=REPO_ROOT,
        runs_root=LIVE_RUNS,
        gr_root=LIVE_GR,
        timestamp=TIMESTAMP,
        telegram=False,
        proof_dir=tmp_path,
    )

    assert summary["admitted"] is False
    assert summary["refusals"]

    closure_path = Path(summary["closure_path"])
    frontier_path = Path(summary["frontier_path"])
    assert closure_path.is_file()
    assert frontier_path.is_file()
    assert closure_path.name == "GENERATION1_GENERAL_RUN_CLOSURE.json"
    assert frontier_path.name == "MOP_CURRENT_FRONTIER.json"

    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    assert closure["admitted"] is False
    assert closure["closure_status"] == "deferred_general_run_not_terminal"
    assert closure["admission"]["refusals"]
    assert closure["deferral"] is not None
    assert closure["activation_allowed"] is False
    assert closure["scientific_promotion"] is False
    assert closure["independent_scientific_confirmation"] is False
    assert closure["terminal_lineage"]["derivation_status"] == "deferred_inputs_not_terminal"
    assert "seal" in closure and closure["seal"]["sha256"]


@_requires_live
def test_verifier_reports_seal_intact_and_detects_mutated_admitted(tmp_path):
    """The verifier reproduces the seal, and every semantic mutation (admitted flag included) is caught."""

    run_closure(
        repo_root=REPO_ROOT,
        runs_root=LIVE_RUNS,
        gr_root=LIVE_GR,
        timestamp=TIMESTAMP,
        telegram=False,
        proof_dir=tmp_path,
    )
    closure = json.loads((tmp_path / "GENERATION1_GENERAL_RUN_CLOSURE.json").read_text(encoding="utf-8"))

    raw_artifacts = {
        "general_run_status": json.loads((LIVE_GR / "current_status.json").read_text(encoding="utf-8")),
        "horizon_v1_final_classification": json.loads(
            (
                LIVE_RUNS
                / "generation1"
                / "generation1-successor-horizon-v1"
                / "classifications"
                / "h05.json"
            ).read_text(encoding="utf-8")
        ),
        "horizon_v2_admission": json.loads(
            (LIVE_RUNS / "generation1" / "generation1-successor-horizon-v2" / "admission.json").read_text(
                encoding="utf-8"
            )
        ),
    }

    receipt = verify_closure(closure, raw_artifacts)

    assert receipt["seal_intact"] is True
    assert receipt["classifications_reproduced"] is True
    assert receipt["independent_scientific_confirmation"] is False
    assert receipt["mutations_detected"]["admitted_flag"] is True
    assert receipt["mutations_detected"]["refusal"] is True
    assert receipt["mutations_detected"]["lane_count"] is True
    assert receipt["mutations_detected"]["general_run_state"] is True
    assert receipt["mutations_detected"]["all_detected"] is True
    assert not receipt["mismatches"]

    # A directly tampered artifact whose seal was not refreshed must fail the seal check.
    tampered = json.loads(json.dumps(closure))
    tampered["admitted"] = True
    tampered_receipt = verify_closure(tampered, raw_artifacts)
    assert tampered_receipt["seal_intact"] is False


@_requires_live
def test_report_renders_and_contains_refusal(tmp_path):
    """The rendered markdown report contains the admission refusal and the explicit refusals block."""

    run_closure(
        repo_root=REPO_ROOT,
        runs_root=LIVE_RUNS,
        gr_root=LIVE_GR,
        timestamp=TIMESTAMP,
        telegram=False,
        proof_dir=tmp_path,
    )
    closure = json.loads((tmp_path / "GENERATION1_GENERAL_RUN_CLOSURE.json").read_text(encoding="utf-8"))

    report = render_closure_report(closure)

    assert "# Generation 1 General Run Closure" in report
    assert "activation_allowed: false" in report
    assert "independent_scientific_confirmation: false" in report
    # The admission refusal that defers the closure must be visible in the report.
    admission_refusals = closure["admission"]["refusals"]
    assert admission_refusals
    assert any(refusal in report for refusal in admission_refusals)
