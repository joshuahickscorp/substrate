import json

from mop.studio.dr1_verifier import DR1VerifierConfig, build_dr1_verification, write_dr1_verification


def _dr1_cache(tmp_path, *, a6_survives=True):
    root = tmp_path / "data" / "cache" / "vjepa2_vitl_comp_video"
    leg = root / "leg_0_2"
    leg.mkdir(parents=True)
    (root / "merge_manifest.json").write_text(
        json.dumps(
            {
                "legs": [[0, 2]],
                "contiguous": True,
                "total_encoded": 2,
                "backends": ["vjepa_hf"],
                "factors": ["object", "action"],
                "clip_order_persisted": True,
                "perspective_receipt": str(root / "perspective_matrix_receipt.json"),
            }
        )
    )
    (leg / "cells.json").write_text(
        json.dumps(
            {
                "leg": [0, 2],
                "n_encoded": 2,
                "backend": "vjepa_hf",
                "factors": ["object", "action"],
                "clip_hashes": ["h0", "h1"],
                "acceptance_report": {
                    "object": {"passed": True, "score": 1.0, "chance": 0.5},
                    "action": {"passed": True, "score": 1.0, "chance": 0.5},
                },
            }
        )
    )
    (root / "perspective_matrix_receipt.json").write_text(
        json.dumps(
            {
                "schema": "mop-dr1-perspective-matrix-receipt/v1",
                "ok": True,
                "n_referents": 2,
                "tags": ["caption_text", "vision_vjepa2"],
                "factor_counts": {"object": {"cat": 1, "dog": 1}},
            }
        )
    )
    (root / "a6_residual_guard.json").write_text(
        json.dumps(
            {
                "guard": "a6_residual_alignment (cross-modal caption<->vision nuisance control)",
                "decisive_condition": "minus_factors",
                "conditions": {"minus_factors": {"survives": a6_survives}},
                "verdict": "survives" if a6_survives else "COLLAPSES",
            }
        )
    )
    return root


def test_dr1_verifier_passes_only_when_a6_decisive_condition_survives(tmp_path):
    report = build_dr1_verification(_dr1_cache(tmp_path, a6_survives=True))

    assert report["integrity_ok"] is True
    assert report["passed"] is True
    assert report["all_ok"] is True
    assert report["independent"] is True
    assert report["adversarial"] is True


def test_dr1_verifier_refuses_positive_when_a6_collapses(tmp_path):
    report = build_dr1_verification(_dr1_cache(tmp_path, a6_survives=False))

    assert report["integrity_ok"] is True
    assert report["passed"] is False
    assert any("positive claim refused" in problem for problem in report["problems"])


def test_dr1_verifier_flags_missing_caption_gate(tmp_path):
    root = _dr1_cache(tmp_path)
    cells = json.loads((root / "leg_0_2" / "cells.json").read_text())
    del cells["acceptance_report"]
    (root / "leg_0_2" / "cells.json").write_text(json.dumps(cells))

    report = build_dr1_verification(DR1VerifierConfig(root))

    assert report["passed"] is False
    assert any("caption_gate" in problem for problem in report["problems"])


def test_write_dr1_verification_round_trips(tmp_path):
    root = _dr1_cache(tmp_path)
    report = build_dr1_verification(root)
    out = tmp_path / "dr1_verification.json"
    write_dr1_verification(report, out)
    assert json.loads(out.read_text())["schema"] == "mop-dr1-adversarial-verification/v1"
