"""Planner: knapsack scoring, budget enforcement, per-source cap, breadth-first diversity, license
gating, and the hard invariant that full Ego4D is NEVER planned by default."""

import textwrap

from mop.studio import planner
from mop.studio.profiles import M3PRO_LOCAL_MAX, STUDIO, Profile


def _tiny_registry(tmp_path, body: str):
    p = tmp_path / "datasets.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_full_ego4d_never_selected_even_with_license():
    p = planner.plan(STUDIO, budget_gb=900, accept_license=True)
    slugs = {s["slug"] for s in p["selected"]}
    assert "ego4d_full" not in slugs
    # and it shows up as a deferred skip with a clear reason
    skipped = {s["slug"]: s for s in p["skipped"]}
    assert "ego4d_full" in skipped and skipped["ego4d_full"]["status"] == "deferred"


def test_license_blockers_listed_without_acknowledgement():
    p = planner.plan(STUDIO, budget_gb=900, accept_license=False)
    blocker_slugs = {b["slug"] for b in p["license_blockers"]}
    # manual (signed terms) sources are blocked until --accept-license
    assert "ssv2" in blocker_slugs
    assert "ego4d_subset" in blocker_slugs


def test_accept_license_unlocks_manual_sources():
    no = planner.plan(STUDIO, budget_gb=900, accept_license=False)
    yes = planner.plan(STUDIO, budget_gb=900, accept_license=True)
    assert "ssv2" not in {s["slug"] for s in no["selected"]}
    assert "ssv2" in {s["slug"] for s in yes["selected"]}


def test_m3pro_never_selects_manual_even_with_license():
    # the laptop profile disallows manual auth: --accept-license cannot override the profile
    p = planner.plan(M3PRO_LOCAL_MAX, budget_gb=10, accept_license=True)
    assert "ssv2" not in {s["slug"] for s in p["selected"]}


def test_budget_enforced_total_within_usable():
    p = planner.plan(M3PRO_LOCAL_MAX, budget_gb=10)
    t = p["totals"]
    assert t["raw_gb"] + t["cache_gb"] <= 10.0
    assert t["headroom_gb"] >= 0


def test_breadth_first_covers_multiple_modalities():
    p = planner.plan(STUDIO, budget_gb=900)
    # the plan prefers breadth: more than one modality and several domains
    assert len(p["totals"]["modalities_covered"]) >= 2
    assert len(p["totals"]["domains_covered"]) >= 3


def test_synthetic_controls_always_selectable():
    # generated controls have no license/size risk and top priority; always in the plan
    for prof, budget in ((STUDIO, 900), (M3PRO_LOCAL_MAX, 10)):
        p = planner.plan(prof, budget_gb=budget)
        assert "synthetic_controls" in {s["slug"] for s in p["selected"]}


def test_per_source_cap_forces_smaller_subset(tmp_path):
    reg = _tiny_registry(
        tmp_path,
        """
        version: 1
        sources:
          - slug: big
            name: Big
            modality: video
            domain: d
            task_fit: t
            priority: 50
            raw_gb: 100.0
            cache_gb_pooled: 1.0
            recommended_subsets: [1000, 5000, 50000]
            license: open
            citation: c
            allowed_use: ok
            auth_steps: none
            download_method: http
            resume_method: m
            checksum_plan: sha256
            status: available
            risk: {license: low, size: high, link_rot: low, annotation_quality: high, privacy: low, decode: low}
            output_layout: class-folder
            kind: natural-video
        """,
    )
    # a profile whose per-source cap is 5 GB: raw scales 100 * subset/50000, so only the 1000 subset (2 GB) fits
    prof = Profile(
        name="t",
        disk_total_gb=900,
        reserve_gb=0,
        download_budget_gb=900,
        download_hard_cap_gb=900,
        fixture_budget_gb=1,
        raw_smoke_gb=1,
        max_cache_clips=10,
        min_free_disk_gb=0,
        max_run_count=10,
        max_wall_min=10,
        max_source_count=10,
        max_per_source_gb=5.0,
    )
    p = planner.plan(prof, budget_gb=900, datasets_path=reg)
    sel = p["selected"][0]
    assert sel["slug"] == "big"
    assert sel["subset_clips"] == 1000  # the only subset under the 5 GB per-source cap
    assert sel["raw_gb"] <= 5.0


def test_source_count_cap_crowds_out_extras():
    # cap the source count at 2: only 2 selected, the rest skipped with the cap reason
    prof = Profile(
        name="t2",
        disk_total_gb=900,
        reserve_gb=0,
        download_budget_gb=900,
        download_hard_cap_gb=900,
        fixture_budget_gb=1,
        raw_smoke_gb=1,
        max_cache_clips=10,
        min_free_disk_gb=0,
        max_run_count=10,
        max_wall_min=10,
        max_source_count=2,
        max_per_source_gb=500.0,
        allow_manual_auth=True,
    )
    p = planner.plan(prof, budget_gb=900, accept_license=True)
    assert len(p["selected"]) == 2
    assert any("source-count cap" in s["reason"] for s in p["skipped"])


def test_include_exclude_filters():
    p = planner.plan(STUDIO, budget_gb=900, include=["synthetic_controls", "audioset_meta"])
    assert {s["slug"] for s in p["selected"]} <= {"synthetic_controls", "audioset_meta"}
    p2 = planner.plan(STUDIO, budget_gb=900, exclude=["synthetic_controls"])
    assert "synthetic_controls" not in {s["slug"] for s in p2["selected"]}


def test_budget_clamps_to_usable_not_hardcap():
    # asking for more than usable disk is clamped DOWN to usable_gb (the planner spends disk, not
    # the download hard cap); the plan can never commit more than usable_gb of raw+cache.
    p = planner.plan(STUDIO, budget_gb=10_000)  # absurd request
    assert p["totals"]["budget_gb"] <= STUDIO.usable_gb
    assert p["totals"]["raw_gb"] + p["totals"]["cache_gb"] <= STUDIO.usable_gb


def test_negative_budget_rejected():
    import pytest

    with pytest.raises(ValueError):
        planner.plan(STUDIO, budget_gb=-1)


def test_next_commands_present_and_render():
    p = planner.plan(STUDIO, budget_gb=900)
    assert any("acquire" in c for c in p["next_commands"])
    md = planner.render_md(p)
    assert "# Studio acquisition plan" in md
    assert "Selected sources" in md
