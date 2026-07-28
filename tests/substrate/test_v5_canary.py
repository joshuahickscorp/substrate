from __future__ import annotations

from pathlib import Path

import pytest

from substrate import v5canary
from substrate import v5config as C


@pytest.fixture(scope="module")
def evidence() -> dict:
    return v5canary.run(publish=False)


def test_all_fifty_canaries_are_complete_terminal_and_in_frozen_order(
    evidence: dict,
) -> None:
    assert evidence["total"] == 50
    assert evidence["passed"] == 50
    assert evidence["failed"] == []
    assert evidence["all_terminal"]
    assert evidence["all_pass"]
    assert evidence["activation"] is False
    assert [row["identity"] for row in evidence["rows"]] == list(C.CANARIES)


def test_every_canary_reports_active_mechanism_controls_and_raw_units(
    evidence: dict,
) -> None:
    required = {
        "identity",
        "description",
        "mechanism_activity",
        "positive_fixture",
        "null_fixture",
        "controls",
        "oracle",
        "headroom",
        "sesoi",
        "raw_independent_values",
        "independent_units",
        "effect",
        "classification",
        "passes",
        "activation",
    }
    for row in evidence["rows"]:
        assert required <= set(row)
        assert row["description"] == C.CANARY_REQUIREMENTS[row["identity"]]
        assert row["mechanism_activity"]
        assert row["positive_fixture"]
        assert row["null_fixture"]
        assert row["controls"]
        assert row["independent_units"] >= 1
        assert row["raw_independent_values"]["positive"]
        assert row["raw_independent_values"]["control"]
        assert row["classification"] in v5canary.TERMINAL_CLASSIFICATIONS
        assert row["passes"]
        assert row["activation"] is False


def test_no_headroom_support_canary_is_a_valid_refusal(evidence: dict) -> None:
    row = next(row for row in evidence["rows"] if row["identity"] == "C07")
    assert row["headroom"] < row["sesoi"]
    assert row["classification"] == "valid_no_headroom"
    assert row["details"]["support_invoked"] is False
    assert row["passes"]


def test_ablation_and_corruption_canaries_have_nonzero_separation(
    evidence: dict,
) -> None:
    by_id = {row["identity"]: row for row in evidence["rows"]}
    for identity in (
        "C37",
        "C38",
        "C39",
        "C40",
        "C41",
        "C43",
        "C44",
        "C45",
        "C46",
        "C47",
        "C48",
        "C49",
    ):
        assert by_id[identity]["effect"] >= by_id[identity]["sesoi"]


def test_publish_false_performs_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        v5canary.io,
        "seal",
        lambda name, document, **kwargs: calls.append(name) or Path(name),
    )
    result = v5canary.run(publish=False)
    assert result["all_pass"]
    assert not result["published"]
    assert calls == []


def test_publish_true_seals_required_and_domain_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    def seal(name: str, document: dict, **_: object) -> Path:
        calls.append((name, document))
        return Path(name)

    monkeypatch.setattr(v5canary.io, "seal", seal)
    result = v5canary.run()
    names = [name for name, _ in calls]
    assert result["published"]
    assert names[:2] == [
        "SUBSTRATE_V5_CHEAP_CANARIES.json",
        "SUBSTRATE_V5_CANARY_LEDGER.json",
    ]
    assert set(v5canary.DOMAIN_AUTHORITIES) <= set(names)
    assert all(document["activation"] is False for _, document in calls)
