import json

import scripts.studio.__main__ as studio_cli

from mop.studio.scorecard import build_studio_scorecard, render_markdown, upsert_report_block


def test_scorecard_missing_receipts_is_incomplete():
    report = build_studio_scorecard()
    assert report["schema"] == "mop-studio-scorecard/v1"
    assert report["studio_10_ready"] is False
    assert report["axes"]["abstraction"]["status"] == "pending"
    assert any(b.startswith("launch:") for b in report["blockers"])


def test_scorecard_accepts_dr1_independent_adversarial_pass():
    report = build_studio_scorecard(
        dr1_verification={
            "schema": "mop-dr1-adversarial-verification/v1",
            "integrity_ok": True,
            "passed": True,
            "all_ok": True,
            "independent": True,
            "adversarial": True,
        }
    )
    assert report["axes"]["abstraction"]["status"] == "evidence"


def test_scorecard_refuses_laptop_pr9_smoke_as_moldability_evidence():
    report = build_studio_scorecard(
        pr9_result={
            "cache": "data/cache/vjepa2_vitl_fpc64_256_real",
            "null_supported": False,
            "any_zero_reinit": False,
            "lr_integral_matched_all": True,
            "certificate": {"fired": True},
        },
        pr9_state={"schema": "mop-pr9-run-state/v1", "status": "complete"},
    )
    assert report["axes"]["moldability"]["status"] == "pending"
    assert "local smoke" in report["axes"]["moldability"]["detail"]


def test_scorecard_marks_dr1_pr9_null_as_process_c_wall():
    report = build_studio_scorecard(
        pr9_result={
            "cache": "data/cache/vjepa2_vitl_comp_video",
            "null_supported": True,
            "any_zero_reinit": False,
            "lr_integral_matched_all": True,
            "certificate": {"fired": False},
        },
        pr9_state={"schema": "mop-pr9-run-state/v1", "status": "complete"},
        pr9_verdict={
            "schema": "mop-pr9-verdict-ledger/v1",
            "all_ok": True,
            "status": "null_no_certificate",
        },
        process_c_gate={
            "schema": "mop-process-c-license-gate/v1",
            "all_ok": True,
            "launch_allowed": True,
            "licensing_sources": ["pr9"],
        },
    )
    assert report["axes"]["moldability"]["status"] == "walled"
    assert "Process C" in report["axes"]["moldability"]["detail"]
    assert report["process_c"]["status"] == "licensed"


def test_scorecard_reports_process_c_not_licensed_from_gate():
    report = build_studio_scorecard(
        process_c_gate={
            "schema": "mop-process-c-license-gate/v1",
            "all_ok": True,
            "launch_allowed": False,
            "blockers": ["pr9:completed but did not license"],
        }
    )
    assert report["process_c"]["status"] == "not_licensed"


def test_scorecard_requires_pr9_verdict_ledger_for_dr1_cache():
    report = build_studio_scorecard(
        pr9_result={
            "cache": "data/cache/vjepa2_vitl_comp_video",
            "null_supported": False,
            "any_zero_reinit": False,
            "lr_integral_matched_all": True,
            "certificate": {"fired": True},
        },
        pr9_state={"schema": "mop-pr9-run-state/v1", "status": "complete"},
    )
    assert report["axes"]["moldability"]["status"] == "pending"
    assert "verdict ledger" in report["axes"]["moldability"]["detail"]


def test_scorecard_partial_atlas_is_pending_not_evidence():
    report = build_studio_scorecard(
        dense_gate={
            "schema": "mop-dense-atlas-cache-gate/v1",
            "all_ok": True,
        },
        atlas_result={
            "full_registered_grid": False,
            "full_registered_pairs": False,
            "null_supported": False,
            "registered_columns_missing": ["vjepa_bound_video"],
            "registered_arms_missing": ["vjepa2_vitl_bound_video"],
        },
        atlas_verdict={
            "schema": "mop-atlas-verdict-ledger/v1",
            "all_ok": False,
            "status": "partial_non_scoring",
            "problems": ["missing registered columns"],
        },
    )
    assert report["axes"]["density"]["status"] == "pending"
    assert "atlas verdict ledger" in report["axes"]["density"]["detail"]


def test_scorecard_requires_dense_atlas_cache_gate_before_atlas_evidence():
    report = build_studio_scorecard(
        atlas_result={
            "full_registered_grid": True,
            "full_registered_pairs": True,
            "null_supported": False,
            "verdict": "candidate",
        }
    )
    assert report["axes"]["density"]["status"] == "pending"
    assert "cache gate" in report["axes"]["density"]["detail"]


def test_scorecard_blocks_bad_dense_atlas_cache_gate():
    report = build_studio_scorecard(
        dense_gate={
            "schema": "mop-dense-atlas-cache-gate/v1",
            "all_ok": False,
            "problems": ["randominit dense manifest missing"],
        },
        atlas_result={
            "full_registered_grid": True,
            "full_registered_pairs": True,
            "null_supported": False,
            "verdict": "candidate",
        },
    )
    assert report["axes"]["density"]["status"] == "pending"
    assert "randominit" in report["axes"]["density"]["problems"][0]


def test_scorecard_requires_atlas_verdict_ledger_before_atlas_evidence():
    report = build_studio_scorecard(
        dense_gate={
            "schema": "mop-dense-atlas-cache-gate/v1",
            "all_ok": True,
        },
        atlas_result={
            "full_registered_grid": True,
            "full_registered_pairs": True,
            "null_supported": False,
            "verdict": "candidate",
        },
    )
    assert report["axes"]["density"]["status"] == "pending"
    assert "verdict ledger" in report["axes"]["density"]["detail"]


def test_scorecard_accepts_atlas_verdict_candidate_positive_as_density_evidence():
    report = build_studio_scorecard(
        dense_gate={
            "schema": "mop-dense-atlas-cache-gate/v1",
            "all_ok": True,
        },
        atlas_result={
            "full_registered_grid": True,
            "full_registered_pairs": True,
            "null_supported": False,
            "verdict": "NULL REJECTED",
        },
        atlas_verdict={
            "schema": "mop-atlas-verdict-ledger/v1",
            "all_ok": True,
            "status": "candidate_positive",
        },
    )
    assert report["axes"]["density"]["status"] == "evidence"
    assert "atlas_verdict_ledger" in report["axes"]["density"]["receipts"]


def test_scorecard_durability_requires_active_artifact_indexes():
    indexes = {
        name: {"schema": "mop-artifact-bundle/v1", "all_ok": True}
        for name in ("wave0", "dr1", "pr9", "atlas", "spine")
    }
    spine = {"schema": "mop-studio-spine-status/v1", "all_complete": True}
    report = build_studio_scorecard(artifact_indexes=indexes, spine_status=spine)
    assert report["axes"]["durability"]["status"] == "complete"


def test_scorecard_markdown_block_upserts(tmp_path):
    report = build_studio_scorecard()
    block = render_markdown(report)
    assert "STUDIO-SCORECARD-AUTO:START" in block
    md = tmp_path / "report.md"
    md.write_text("# Report\n\n## Wave log\n")
    upsert_report_block(md, block)
    upsert_report_block(md, block)
    text = md.read_text()
    assert text.count("STUDIO-SCORECARD-AUTO:START") == 1


def test_scorecard_cli_writes_receipt(tmp_path):
    out = tmp_path / "scorecard.json"
    rc = studio_cli.main(["scorecard", "--out", str(out)])
    assert rc == 1
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-studio-scorecard/v1"


def test_scorecard_cli_allow_incomplete_preserves_receipt(tmp_path):
    out = tmp_path / "scorecard.json"
    rc = studio_cli.main(["scorecard", "--out", str(out), "--allow-incomplete"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-studio-scorecard/v1"
    assert data["all_ok"] is False
