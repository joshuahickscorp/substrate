import json

import scripts.studio.__main__ as audit_cli

from mop.studio.objective_audit import build_studio_objective_audit, write_studio_objective_audit


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_objective_audit_marks_missing_studio_evidence_as_not_ready(tmp_path):
    audit = build_studio_objective_audit(repo_root=tmp_path)
    assert audit["schema"] == "mop-studio-objective-audit/v1"
    assert audit["studio_10_ready"] is False
    assert audit["summary"]["points_possible"] == 9.0
    assert any(r["id"] == "dr1_real_bound_video" and r["status"] == "pending" for r in audit["requirements"])


def test_objective_audit_counts_prepared_launch_receipts_without_science_credit(tmp_path):
    _write(
        tmp_path / "runs" / "studio_wave0" / "transfer_check.json",
        {"schema": "mop-studio-transfer-check/v1", "all_ok": True},
    )
    _write(
        tmp_path / "runs" / "studio_spine" / "spine_plan.json",
        {"schema": "mop-studio-spine-plan/v1"},
    )
    _write(
        tmp_path / "runs" / "studio_spine" / "spine_status_local.json",
        {"schema": "mop-studio-spine-status/v1"},
    )
    audit = build_studio_objective_audit(repo_root=tmp_path)
    launch = next(r for r in audit["requirements"] if r["id"] == "wave0_launch_prep")
    dr1 = next(r for r in audit["requirements"] if r["id"] == "dr1_real_bound_video")
    assert launch["status"] == "prepared"
    assert 0.0 < launch["credit"] < 1.0
    assert dr1["credit"] == 0.0


def test_objective_audit_can_complete_native_lane_point(tmp_path):
    _write(
        tmp_path / "runs" / "studio_native_lanes_manifest.json",
        {
            "schema": "mop-studio-native-lanes/v1",
            "lanes": [{"id": "hosted_corpora_plan", "status": "ready"}],
        },
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "studio_native_lanes.py").write_text("# ok")
    audit = build_studio_objective_audit(repo_root=tmp_path)
    native = next(r for r in audit["requirements"] if r["id"] == "studio_native_lanes")
    assert native["status"] == "complete"


def test_objective_audit_can_complete_form_boundary_point(tmp_path):
    _write(
        tmp_path / "proof/FORM_SUBSTRATE/SCORECARD.json",
        {
            "schema": "mop-form-campaign-scorecard/v1",
            "local_obligations_exhausted": True,
        },
    )
    _write(
        tmp_path / "proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json",
        {
            "schema": "mop-form-pre-studio-boundary/v1",
            "studio_is_only_remaining_hardware_boundary": True,
        },
    )
    _write(
        tmp_path / "proof/ARTIFACT_INDEX/form_substrate.json",
        {"schema": "mop-artifact-bundle/v1", "all_ok": True},
    )
    audit = build_studio_objective_audit(repo_root=tmp_path)
    form = next(r for r in audit["requirements"] if r["id"] == "form_substrate_pre_studio_boundary")
    assert form["status"] == "complete"


def test_objective_audit_writer_and_cli_round_trip(tmp_path, monkeypatch):
    out = tmp_path / "audit.json"
    audit = build_studio_objective_audit(repo_root=tmp_path)
    write_studio_objective_audit(audit, out)
    assert json.loads(out.read_text())["schema"] == "mop-studio-objective-audit/v1"

    monkeypatch.setattr(audit_cli, "REPO_ROOT", tmp_path)
    rc = audit_cli.main(["objective-audit", "--out", str(out)])
    assert rc == 1
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-studio-objective-audit/v1"


def test_objective_audit_cli_can_preserve_not_ready_receipts(tmp_path, monkeypatch):
    out = tmp_path / "audit.json"
    monkeypatch.setattr(audit_cli, "REPO_ROOT", tmp_path)
    rc = audit_cli.main(["objective-audit", "--out", str(out), "--allow-not-ready"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-studio-objective-audit/v1"
    assert data["studio_10_ready"] is False
