import json

import scripts.studio.dr1_source_card as card_cli
import scripts.studio.dr1_source_intake as intake_cli

from mop.studio.dr1_source_intake import (
    build_dr1_source_card,
    build_dr1_source_intake,
    validate_dr1_source_card,
)


def _make_source(root, *, duplicate_stems=False, cells=None, clips_per_cell=1):
    cells = cells or (
        "dog-1-left-running",
        "dog-1-right-running",
        "cat-1-left-running",
        "cat-1-right-running",
    )
    captions = {}
    for ci, cell in enumerate(cells):
        d = root / cell
        d.mkdir(parents=True)
        for j in range(clips_per_cell):
            stem = "clip0" if duplicate_stems else f"clip_{ci}_{j}"
            (d / f"{stem}.mp4").write_bytes(b"")
            captions[stem] = f"{cell.replace('-', ' ')} sample {j}"
    (root / "captions.json").write_text(json.dumps(captions))
    return root


def _source_card(**overrides):
    card = {
        "source_id": "dr1-demo",
        "license": "research-use",
        "allowed_use": "local research encode and derived latent receipts",
        "provenance_tag": "natural-video",
        "non_overlap_proof": {"status": "passed", "detail": "not in benchmark eval set"},
        "requires_manual_license": True,
        "accepted_terms": True,
        "clip_count": 4,
    }
    card.update(overrides)
    return card


def test_source_intake_accepts_bound_source_with_source_card(tmp_path):
    src = _make_source(
        tmp_path / "src",
        cells=("dog-left", "dog-right", "cat-left", "cat-right"),
        clips_per_cell=3,
    )
    receipt = build_dr1_source_intake(
        source=src,
        factors=("object", "relation"),
        min_per_cell=1,
        source_card=_source_card(clip_count=12),
    )
    assert receipt["schema"] == "mop-dr1-source-intake/v1"
    assert receipt["all_ok"] is True
    assert receipt["captions"]["covered"] == 12
    assert receipt["caption_recoverability"]["passed"] is True
    assert receipt["source_card"]["provenance_tag"] == "natural-video"


def test_source_intake_blocks_caption_recoverability_null(tmp_path):
    src = _make_source(
        tmp_path / "src",
        cells=("dog-left", "dog-right", "cat-left", "cat-right"),
        clips_per_cell=3,
    )
    captions_path = src / "captions.json"
    captions = json.loads(captions_path.read_text())
    captions_path.write_text(json.dumps({stem: "same neutral caption" for stem in captions}))
    receipt = build_dr1_source_intake(
        source=src,
        factors=("object", "relation"),
        min_per_cell=1,
        source_card=_source_card(clip_count=12),
    )
    assert receipt["all_ok"] is False
    assert receipt["caption_recoverability"]["passed"] is False
    assert any("caption recoverability" in p for p in receipt["problems"])


def test_source_intake_blocks_missing_source_card(tmp_path):
    src = _make_source(tmp_path / "src")
    receipt = build_dr1_source_intake(source=src, min_per_cell=1)
    assert receipt["all_ok"] is False
    assert any("source card" in p for p in receipt["problems"])


def test_source_intake_blocks_duplicate_clip_stems(tmp_path):
    src = _make_source(tmp_path / "src", duplicate_stems=True)
    receipt = build_dr1_source_intake(source=src, min_per_cell=1, source_card=_source_card())
    assert receipt["all_ok"] is False
    assert "clip0" in receipt["duplicate_clip_stems"]
    assert any("ambiguous" in p for p in receipt["problems"])


def test_source_intake_blocks_non_natural_video_card(tmp_path):
    src = _make_source(tmp_path / "src")
    receipt = build_dr1_source_intake(
        source=src,
        min_per_cell=1,
        source_card=_source_card(provenance_tag="structured-synthetic"),
    )
    assert receipt["all_ok"] is False
    assert any("natural-video" in p for p in receipt["problems"])


def test_source_intake_cli_writes_blocked_receipt_when_card_missing(tmp_path):
    src = _make_source(tmp_path / "src")
    out = tmp_path / "intake.json"
    rc = intake_cli.main(["--source", str(src), "--min-per-cell", "1", "--out", str(out)])
    assert rc == 1
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-dr1-source-intake/v1"
    assert data["all_ok"] is False


def test_source_card_builder_validates_populated_card():
    card = build_dr1_source_card(
        source_id="dr1-demo",
        license_name="research-use",
        allowed_use="Studio DR1 latent encoding and derived receipts",
        non_overlap_proof={"status": "passed", "detail": "hash split checked"},
        clip_count=12,
        requires_manual_license=True,
        accepted_terms=True,
    )
    receipt = validate_dr1_source_card(card, expected_clip_count=12)
    assert receipt["schema"] == "mop-dr1-source-card-validation/v1"
    assert receipt["all_ok"] is True
    assert receipt["source_card"]["schema"] == "mop-dr1-source-card/v1"


def test_source_card_validation_blocks_todo_and_manual_terms():
    card = build_dr1_source_card(
        source_id="TODO",
        license_name="unknown",
        allowed_use="todo",
        non_overlap_proof="todo",
        requires_manual_license=True,
        accepted_terms=False,
    )
    receipt = validate_dr1_source_card(card)
    assert receipt["all_ok"] is False
    assert any("source_id" in p for p in receipt["problems"])
    assert any("accepted_terms" in p for p in receipt["problems"])


def test_source_card_cli_template_and_validate_write_receipts(tmp_path):
    card_path = tmp_path / "card.json"
    validation_path = tmp_path / "validation.json"
    rc = card_cli.main(
        [
            "template",
            "--source-id",
            "dr1-demo",
            "--license",
            "research-use",
            "--allowed-use",
            "Studio DR1 latent encoding and derived receipts",
            "--non-overlap-proof",
            "hash split checked against MOP benchmarks",
            "--clip-count",
            "4",
            "--requires-manual-license",
            "--accepted-terms",
            "--out",
            str(card_path),
        ]
    )
    assert rc == 0
    assert json.loads(card_path.read_text())["schema"] == "mop-dr1-source-card/v1"
    rc = card_cli.main(
        ["validate", str(card_path), "--expected-clip-count", "4", "--out", str(validation_path)]
    )
    assert rc == 0
    data = json.loads(validation_path.read_text())
    assert data["schema"] == "mop-dr1-source-card-validation/v1"
    assert data["all_ok"] is True
