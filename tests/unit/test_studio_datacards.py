"""Data cards + license ledger: every source gets a card with the required fields, the ledger
surfaces manual/blocked/deferred blockers, and live plan/acquire state merges into the card."""

import json

from devsys.studio import datacards, registry


def test_data_card_has_required_fields():
    entry = registry.get_dataset("epic_kitchens_subset")
    card = datacards.data_card(entry)
    for key in (
        "slug",
        "name",
        "status",
        "license",
        "citation",
        "allowed_use",
        "auth_steps",
        "provenance_tag",
        "hash_manifest",
        "command",
        "date",
    ):
        assert key in card
    assert card["provenance_tag"] == entry["kind"]


def test_data_card_merges_live_state():
    entry = registry.get_dataset("synthetic_controls")
    state = {"status": "complete", "dest": "/tmp/x", "bytes": 1234, "n_clips": 9, "subset_clips": 32}
    card = datacards.data_card(entry, state=state)
    assert card["status"] == "complete"
    assert card["local_path"] == "/tmp/x"
    assert card["acquired_bytes"] == 1234
    assert card["acquired_clips"] == 9
    assert card["selected_subset"] == 32


def test_license_ledger_surfaces_blockers():
    ledger = datacards.license_ledger()
    blocker_slugs = {b["slug"] for b in ledger["blockers"]}
    # the known manual/blocked/deferred sources must appear as blockers
    assert "ssv2" in blocker_slugs  # manual
    assert "ego4d_full" in blocker_slugs  # deferred
    assert "laion_tiny_meta" in blocker_slugs  # blocked
    # and an available source must NOT be a blocker
    assert "synthetic_controls" not in blocker_slugs


def test_ledger_counts_sum_to_total():
    ledger = datacards.license_ledger()
    assert sum(ledger["counts"].values()) == ledger["n"]
    assert ledger["n"] == len(registry.load_datasets())


def test_write_data_cards_writes_one_per_source_plus_ledger(tmp_path):
    out = datacards.write_data_cards(tmp_path)
    cards_dir = tmp_path / "datacards"
    n = len(registry.load_datasets())
    assert out["n_cards"] == n
    assert len(list(cards_dir.glob("*.md"))) == n
    assert (tmp_path / "license_ledger.md").exists()
    ledger = json.loads((tmp_path / "license_ledger.json").read_text())
    assert "blockers" in ledger


def test_render_card_and_ledger_md_nonempty():
    card_md = datacards.render_card_md(datacards.data_card(registry.get_dataset("ssv2")))
    assert "Data card" in card_md and "license" in card_md
    ledger_md = datacards.render_ledger_md(datacards.license_ledger())
    assert "License ledger" in ledger_md and "Blockers" in ledger_md
