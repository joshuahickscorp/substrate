"""Guard against current-state drift between the machine authority and the current-facing docs.

Two failure modes are pinned here:

1. The generated ``proof/MOP_CURRENT_FRONTIER.json`` authority (built by
   ``mop.closure.frontier.build_current_frontier``) must derive the correct current facts from the live
   artifacts: 41 atlas facets, P4 closed rather than the active heavy lane, the live General Run as the
   exclusive heavy lane, the dynamic 1 to 20 worker policy, and both the STARSS23 source-counting
   reproduction and direction-of-arrival outcomes.

2. The current-facing banner of each current-facing doc (``GENERATIONS.md``, ``STATUS.md``,
   ``GOLD_PROMPT.md``) must agree with that authority. Only the clearly delimited current-state banner is
   scanned; historically dated log entries and dated contract sections are exempt by construction because
   they sit outside the banner markers.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mop.closure.frontier import build_current_frontier

REPO_ROOT = Path("/Users/scammermike/Downloads/mop_gen1_closure")
RUNS_ROOT = Path("/Users/scammermike/Downloads/mop/runs")

CURRENT_FACING_DOCS = ("GENERATIONS.md", "STATUS.md", "GOLD_PROMPT.md")

BANNER_START = "<!-- CURRENT-STATE:START -->"
BANNER_END = "<!-- CURRENT-STATE:END -->"


@pytest.fixture(scope="module")
def frontier():
    """The generated current-state authority derived from the live atlas and General Run status."""

    return build_current_frontier(
        repo_root=REPO_ROOT,
        runs_root=RUNS_ROOT,
        timestamp="2026-07-18T00:00Z",
    )


def _extract_banner(doc_name: str) -> str:
    """Return the current-state banner region of a doc, or fail if it is missing."""

    text = (REPO_ROOT / doc_name).read_text(encoding="utf-8")
    assert text.count(BANNER_START) == 1, f"{doc_name} must have exactly one current-state banner start"
    assert text.count(BANNER_END) == 1, f"{doc_name} must have exactly one current-state banner end"
    start = text.index(BANNER_START) + len(BANNER_START)
    end = text.index(BANNER_END)
    assert start < end, f"{doc_name} banner markers are out of order"
    return text[start:end]


# ---------------------------------------------------------------------------
# 1. the generated authority derives the correct current facts
# ---------------------------------------------------------------------------


def test_frontier_derives_current_facts(frontier):
    assert frontier["facet_count"] == 41
    assert frontier["stale_facet_count_rejected"] == 37

    assert frontier["p4_is_active_heavy_lane"] is False
    assert frontier["active_heavy_lane"]["program_id"] == "general-run"

    obsolete_workers = frontier["worker_policy"]["obsolete_descriptions_rejected"]
    assert any(
        "eight-worker" in description and "fixed" in description for description in obsolete_workers
    ), "worker policy must reject a fixed eight-worker description"

    obsolete_commands = frontier["obsolete_commands_rejected"]
    assert any("long_chain" in command for command in obsolete_commands), (
        "obsolete commands must mention the superseded long-chain command"
    )

    starss23 = frontier["starss23_latest_outcomes"]
    assert "source_counting" in starss23
    assert "reproduction" in starss23["source_counting"]["outcome"].lower()
    assert "repro" in starss23["source_counting"]["reproductions"].lower()
    assert "direction_of_arrival" in starss23


def test_frontier_is_not_promotable(frontier):
    """The authority must stay hardcoded to a non-promoted, non-activated posture."""

    assert frontier["activation_allowed"] is False
    assert frontier["scientific_promotion"] is False
    assert frontier["independent_scientific_confirmation"] is False


# ---------------------------------------------------------------------------
# 2. every current-facing banner agrees with the authority (no drift)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc_name", CURRENT_FACING_DOCS)
def test_current_banner_has_no_stale_facet_count(doc_name, frontier):
    banner = _extract_banner(doc_name).lower()
    stale = str(frontier["stale_facet_count_rejected"])
    assert stale not in banner, f"{doc_name} banner still presents {stale} as the current facet count"
    assert re.search(rf"{frontier['facet_count']}\s+facets", banner), (
        f"{doc_name} banner must state the current {frontier['facet_count']}-facet count"
    )


@pytest.mark.parametrize("doc_name", CURRENT_FACING_DOCS)
def test_current_banner_does_not_present_p4_as_active_heavy_lane(doc_name):
    banner = _extract_banner(doc_name).lower()
    # Within any one sentence (period delimited), P4 must never co-occur with "heavy lane".
    assert not re.search(r"p4[^.]*heavy\s+lane", banner), (
        f"{doc_name} banner presents P4 as an active heavy lane"
    )
    assert not re.search(r"heavy\s+lane[^.]*p4", banner), (
        f"{doc_name} banner presents P4 as an active heavy lane"
    )


@pytest.mark.parametrize("doc_name", CURRENT_FACING_DOCS)
def test_current_banner_does_not_present_fixed_eight_worker_pool(doc_name):
    banner = _extract_banner(doc_name).lower()
    assert "eight-worker" not in banner
    assert "eight worker" not in banner
    assert "1 to 20" in banner, f"{doc_name} banner must state the dynamic 1 to 20 worker policy"
    assert "worker" in banner


@pytest.mark.parametrize("doc_name", CURRENT_FACING_DOCS)
def test_current_banner_names_general_run_as_active_heavy_lane(doc_name):
    banner = _extract_banner(doc_name).lower()
    assert re.search(r"general run[^.]*heavy\s+lane", banner), (
        f"{doc_name} banner must name the live General Run as the active heavy lane"
    )
    assert "exclusive" in banner or "active" in banner


@pytest.mark.parametrize("doc_name", CURRENT_FACING_DOCS)
def test_current_banner_mentions_counting_reproduction_and_doa_nulls(doc_name):
    banner = _extract_banner(doc_name).lower()
    assert "counting" in banner and "reproduction" in banner and "null" in banner, (
        f"{doc_name} banner must record the STARSS23 counting-reproduction null"
    )
    assert "direction-of-arrival" in banner or "direction of arrival" in banner, (
        f"{doc_name} banner must record the STARSS23 direction-of-arrival null"
    )


@pytest.mark.parametrize("doc_name", CURRENT_FACING_DOCS)
def test_current_banner_points_to_machine_authority(doc_name):
    banner = _extract_banner(doc_name).lower()
    assert "mop_current_frontier.json" in banner, (
        f"{doc_name} banner must point to proof/MOP_CURRENT_FRONTIER.json as the current authority"
    )
