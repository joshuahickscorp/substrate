
from mop.config import REPO_ROOT
from mop.falsification.null_cards import (
    extract_card_yaml,
    generate_from_experiment,
    load_card,
    render_card,
    schema,
    validate_card,
)


def test_generate_from_registry_row_has_contract_fields():
    card = generate_from_experiment("e7_sparse")
    assert card["exp_id"] == "e7_sparse"
    assert card["null_hypothesis"]
    assert card["metric"] == "interference"
    assert card["probe_dependency"]["factor"] == "identity"
    assert card["taxonomy_category"] in range(1, 11)
    assert validate_card(card) == []
    assert any("strict placeholder" in p for p in validate_card(card, strict=True))


def test_render_and_extract_round_trip():
    card = generate_from_experiment("ex13_long_stream")
    md = render_card(card)
    parsed = extract_card_yaml(md)
    assert parsed == card


def test_validate_completed_card_without_todo():
    card = generate_from_experiment("ex13_long_stream")
    card["probe_dependency"]["decodable"] = "yes"
    card["probe_dependency"]["acc_above_chance"] = 0.84
    card["seeds"]["sem"] = 0.01
    card["seeds"]["sign_stability"] = "stable at S>=3"
    card["result"] = "protected-minus-control gap 0.0, tie"
    card["raw_run_id"] = "runs/pre_studio/ex13_long_stream.json"
    assert validate_card(card, strict=True) == []


def test_existing_card_with_colons_in_values_validates():
    card = load_card(REPO_ROOT / "proof" / "NULL_CARDS" / "ex13_long_stream.md")
    assert card["exp_id"] == "ex13_long_stream"
    assert validate_card(card) == []


def test_dr1_null_card_validates_strict():
    card = load_card(REPO_ROOT / "proof" / "NULL_CARDS" / "mop_dr1_video_cache.md")
    assert card["exp_id"] == "mop_dr1_video_cache"
    assert validate_card(card, strict=True) == []


def test_loose_parser_keeps_colon_value():
    text = """
```yaml
exp_id: X
title: colon value
hypothesis: baseline says: keep this prose
null_hypothesis: H0
baseline: tuned baseline
ablation: ablation
metric: bwt
probe_dependency:
  factor: identity
  encoder: enc
  atlas_row: row
  decodable: yes
  acc_above_chance: null
encoder_scale: L
seeds:
  n: 3
  sem: null
  sign_stability: stable
provenance_tag: provisional
result: tie
taxonomy_category: 3
verdict: DOWNGRADE-TIE
badges: [substrate-blindspot]
raw_run_id: runs/x.json
repro_level: R1
```
"""
    card = extract_card_yaml(text)
    assert card["hypothesis"] == "baseline says: keep this prose"
    assert validate_card(card) == []


def test_validate_catches_missing_probe_dependency():
    card = generate_from_experiment("e7_sparse")
    del card["probe_dependency"]
    problems = validate_card(card)
    assert any("probe_dependency" in p for p in problems)


def test_validate_catches_bad_enums_and_seed_count():
    card = generate_from_experiment("e7_sparse")
    card["provenance_tag"] = "made-up"
    card["verdict"] = "MAYBE"
    card["seeds"]["n"] = 1
    problems = validate_card(card)
    assert any("provenance_tag" in p for p in problems)
    assert any("verdict" in p for p in problems)
    assert any("seeds.n" in p for p in problems)


def test_schema_lists_required_fields():
    s = schema()
    assert "exp_id" in s["required"]
    assert s["properties"]["probe_dependency"]["required"] == [
        "factor",
        "encoder",
        "atlas_row",
        "decodable",
        "acc_above_chance",
    ]
