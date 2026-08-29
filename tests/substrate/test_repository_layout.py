from __future__ import annotations

from pathlib import Path

from substrate import evidence, historical


def test_maintained_areas_have_one_canonical_home() -> None:
    root = evidence.ROOT
    for relative in ("src", "tests", "ops", "docs", "evidence"):
        assert (root / relative).is_dir(), relative
    legacy_configs = root / "configs"
    if legacy_configs.exists():
        assert not legacy_configs.samefile(root / "ops" / "configs")


def test_historical_names_resolve_without_rewriting_the_authority() -> None:
    root = evidence.ROOT
    assert evidence.canonical_current_path(root, "configs/substrate/v4/frozen_configuration.json") == (
        root / "ops" / "configs" / "substrate" / "v4" / "frozen_configuration.json"
    )
    assert evidence.canonical_current_path(root, "proof/substrate/mop-substrate-master-v1") == (
        root / "evidence" / "proof-ledger" / "substrate" / "mop-substrate-master-v1"
    )
    assert historical.authority()["roots"]["predecessor_evidence"].startswith("proof/")
    assert historical.root("predecessor_evidence").is_dir()
    assert historical.verify_all()["all_pass"]


def test_frozen_source_digest_ignores_only_declared_layout_aliases(tmp_path: Path) -> None:
    historical_source = tmp_path / "historical.py"
    current_source = tmp_path / "current.py"
    historical_source.write_bytes(b'PLAN = "plans/substrate/tangible_next_launch"\n')
    current_source.write_bytes(b'PLAN = "docs/plans/substrate/tangible_next_launch"\n')
    assert evidence.canonical_source_digest(historical_source) == evidence.canonical_source_digest(current_source)

    current_source.write_bytes(b'PLAN = "docs/plans/substrate/tangible_next_launch"\n# drift\n')
    assert evidence.canonical_source_digest(historical_source) != evidence.canonical_source_digest(current_source)

    binary = tmp_path / "input.json"
    binary.write_bytes(historical_source.read_bytes())
    assert evidence.canonical_source_digest(binary) != evidence.canonical_source_digest(current_source)
