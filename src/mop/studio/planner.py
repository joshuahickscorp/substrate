"""Storage planner (Frontier 3 for the 1 TB Studio, Frontier 3B for the current device). A
breadth-first, budget-bounded knapsack over the dataset registry: given a Profile (its disk and
source-count caps) it selects which sources to acquire and at what subset size, preferring breadth
(action video + egocentric + instructional metadata + synthetic controls + local lane) over one
giant dataset, and it NEVER plans full Ego4D.

The selection is a greedy knapsack with a diversity bonus: candidates are scored by registry
priority times a multiplier that rewards covering a new modality/domain and discounts piling onto
an already-covered domain, then taken largest-subset-first while raw+cache fit the usable budget,
the per-source cap holds, and the source count is under the profile cap. Deferred/blocked sources
are never selected; manual (signed-terms) sources are selected ONLY when the profile allows manual
auth AND the caller acknowledges the license. The output records selected + skipped (with reasons),
raw/cache/temp/reserve byte estimates, a risk score, license blockers, and the exact next commands.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from .profiles import Profile
from .registry import ALWAYS_DEFERRED, load_datasets

# risk facet level -> numeric weight (higher is riskier). na/none/user are not numeric risk.
_RISK_LEVEL = {"none": 0, "na": 0, "user": 1, "low": 1, "medium": 2, "high": 3, "extreme": 5}
# temp decode/extract scratch as a fraction of raw (extract archives, transcode); a planning reserve.
TEMP_FRACTION = 0.15


def _risk_score(risk: dict) -> int:
    """Sum the numeric risk facets into one comparable integer (0 == riskless, generated/local)."""
    return sum(_RISK_LEVEL.get(str(v).lower(), 2) for v in risk.values())


def _selectable(d: dict, profile: Profile, accept_license: bool) -> tuple[bool, str]:
    """Can this source be planned under the profile. Returns (ok, reason-if-not).

    Deferred (incl. full Ego4D) and blocked are never selectable. Manual (signed terms) needs both
    the profile to allow manual auth and the caller to acknowledge the license. Everything else
    (available, metadata-only) is selectable."""
    slug, status = d.get("slug"), d.get("status")
    if slug in ALWAYS_DEFERRED or status == "deferred":
        return False, "deferred (beyond a 1 TB local disk; never planned by default)"
    if status == "blocked":
        return False, "blocked (source withdrawn or access-controlled)"
    if status == "manual":
        if not profile.allow_manual_auth:
            return False, "manual license; profile does not allow manual-auth sources"
        if not accept_license:
            return False, "manual license; needs --accept-license acknowledgement"
        return True, "manual license acknowledged"
    return True, "selectable"


def _subset_options(d: dict) -> list[tuple[int | None, float, float]]:
    """Acquisition options for a source as (subset_size, raw_gb, cache_gb), largest first. raw/cache
    scale linearly with the subset fraction of the largest recommended subset; a source with no
    recommended_subsets has a single option at its declared raw_gb/cache_gb_pooled (metadata-only,
    local import, etc.)."""
    raw_full = float(d.get("raw_gb", 0.0))
    cache_full = float(d.get("cache_gb_pooled", 0.0))
    subsets = sorted({int(s) for s in d.get("recommended_subsets", []) if int(s) > 0}, reverse=True)
    if not subsets:
        return [(None, raw_full, cache_full)]
    top = subsets[0]
    return [(s, raw_full * (s / top), cache_full * (s / top)) for s in subsets]


def _diversity_multiplier(d: dict, modalities: set[str | None], domains: set[str | None]) -> float:
    """Breadth-first nudge: 1.5x when the source opens a NEW modality, 1.25x for a new domain, 0.6x
    when both its modality and domain are already covered (diminishing returns on piling on)."""
    mod, dom = d.get("modality"), d.get("domain")
    if mod not in modalities:
        return 1.5
    if dom not in domains:
        return 1.25
    return 0.6


def plan(
    profile: Profile,
    budget_gb: float | None = None,
    accept_license: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    datasets_path=None,
) -> dict:
    """Select sources to acquire under `profile` within the disk budget, breadth-first.

    budget_gb: usable disk to spend (clamped to the profile hard cap); default = profile.usable_gb.
    accept_license: acknowledge signed-terms (manual) sources (still gated by profile.allow_manual_auth).
    include/exclude: optional slug filters (include restricts the candidate set; exclude removes).
    Returns a plan dict with selected, skipped, totals (raw/cache/temp/reserve), risk, blockers, next.
    """
    # the planner spends DISK, so the ceiling is usable_gb (total minus reserve), never the download
    # hard cap; a requested budget is clamped down to usable_gb and a negative budget is rejected.
    if budget_gb is not None and float(budget_gb) < 0:
        raise ValueError(f"budget_gb {budget_gb:g} is negative")
    usable = profile.usable_gb if budget_gb is None else min(float(budget_gb), profile.usable_gb)
    sources = load_datasets(datasets_path)
    if include:
        inc = set(include)
        sources = [d for d in sources if d.get("slug") in inc]
    if exclude:
        exc = set(exclude)
        sources = [d for d in sources if d.get("slug") not in exc]

    # candidates scored by priority x diversity; greedy because diversity depends on what is already
    # picked, so we re-score against the running coverage each step.
    remaining = list(sources)
    selected: list[dict] = []
    skipped: list[dict] = []
    modalities: set[str | None] = set()
    domains: set[str | None] = set()
    raw_used = cache_used = 0.0

    # first, drop the unselectable with reasons (deferred/blocked/manual-not-acked)
    pool: list[dict] = []
    for d in remaining:
        ok, why = _selectable(d, profile, accept_license)
        if not ok:
            skipped.append(
                {"slug": d.get("slug"), "name": d.get("name"), "status": d.get("status"), "reason": why}
            )
        else:
            pool.append(d)

    while pool and len(selected) < profile.max_source_count:
        # score every remaining candidate against current coverage, pick the best
        scored = []
        for d in pool:
            mult = _diversity_multiplier(d, modalities, domains)
            score = float(d.get("priority", 0)) * mult
            scored.append((score, d))
        scored.sort(key=lambda t: (-t[0], str(t[1].get("slug"))))
        best = scored[0][1]
        pool.remove(best)

        # pick the largest subset option that fits the remaining budget AND the per-source cap
        chosen = None
        for size, raw_gb, cache_gb in _subset_options(best):
            if raw_gb > profile.max_per_source_gb:
                continue
            if raw_used + cache_used + raw_gb + cache_gb <= usable:
                chosen = (size, raw_gb, cache_gb)
                break
        if chosen is None:
            biggest = _subset_options(best)[0]
            reason = (
                f"per-source cap {profile.max_per_source_gb:g} GB"
                if biggest[1] > profile.max_per_source_gb
                else f"no subset fits remaining budget ({usable - raw_used - cache_used:.1f} GB left)"
            )
            skipped.append(
                {
                    "slug": best.get("slug"),
                    "name": best.get("name"),
                    "status": best.get("status"),
                    "reason": reason,
                }
            )
            continue
        size, raw_gb, cache_gb = chosen
        raw_used += raw_gb
        cache_used += cache_gb
        modalities.add(best.get("modality"))
        domains.add(best.get("domain"))
        selected.append(
            {
                "slug": best.get("slug"),
                "name": best.get("name"),
                "modality": best.get("modality"),
                "domain": best.get("domain"),
                "kind": best.get("kind"),
                "status": best.get("status"),
                "subset_clips": size,
                "raw_gb": round(raw_gb, 3),
                "cache_gb": round(cache_gb, 3),
                "download_method": best.get("download_method"),
                "output_layout": best.get("output_layout"),
                "risk_score": _risk_score(best.get("risk", {})),
                "license": best.get("license"),
            }
        )

    # anything still in pool was crowded out by the source-count cap
    for d in pool:
        skipped.append(
            {
                "slug": d.get("slug"),
                "name": d.get("name"),
                "status": d.get("status"),
                "reason": f"source-count cap {profile.max_source_count} reached",
            }
        )

    temp_gb = round(raw_used * TEMP_FRACTION, 3)
    blockers = [
        {"slug": s["slug"], "status": s["status"], "reason": s["reason"]}
        for s in skipped
        if s["status"] in ("manual", "blocked", "deferred")
    ]
    totals = {
        "selected_sources": len(selected),
        "raw_gb": round(raw_used, 3),
        "cache_gb": round(cache_used, 3),
        "temp_gb": temp_gb,
        "reserve_gb": profile.reserve_gb,
        "planned_disk_gb": round(raw_used + cache_used + temp_gb, 3),
        "budget_gb": round(usable, 3),
        "headroom_gb": round(usable - raw_used - cache_used, 3),
        "total_risk_score": sum(s["risk_score"] for s in selected),
        "modalities_covered": sorted(m for m in modalities if m),
        "domains_covered": sorted(d for d in domains if d),
    }
    return {
        "profile": profile.as_dict(),
        "selected": selected,
        "skipped": skipped,
        "license_blockers": blockers,
        "totals": totals,
        "next_commands": _next_commands(profile, selected),
        "accept_license": accept_license,
    }


def _next_commands(profile: Profile, selected: list[dict]) -> list[str]:
    """The exact commands to act on this plan, dry-run first, then the gated real acquisition."""
    cmds = [
        "python scripts/studio_pipeline.py acquire --plan runs/studio_pipeline/latest/plan.json  # DRY RUN",
    ]
    if selected:
        cmds.append(
            "python scripts/studio_pipeline.py acquire --plan runs/studio_pipeline/latest/plan.json "
            f"--execute --budget-gb {profile.download_budget_gb:g} --accept-license  # REAL (gated)"
        )
    prof = profile.name
    cmds += [
        "python scripts/studio_pipeline.py validate --plan runs/studio_pipeline/latest/plan.json",
        "python scripts/studio_pipeline.py cache    --plan runs/studio_pipeline/latest/plan.json "
        f"--profile {prof}",
        f"python scripts/studio_pipeline.py run --gated --tiers C --full --profile {prof}",
    ]
    return cmds


def render_md(p: dict) -> str:
    """Readable Markdown plan: header, selected table, skipped table, license blockers, next commands."""
    t = p["totals"]
    prof = p["profile"]
    L = [
        "# Studio acquisition plan",
        "",
        f"Profile **{prof['name']}** | budget {t['budget_gb']:.0f} GB usable | "
        f"selected {t['selected_sources']} sources | "
        f"raw {t['raw_gb']:.1f} GB + cache {t['cache_gb']:.1f} GB + temp {t['temp_gb']:.1f} GB "
        f"= {t['planned_disk_gb']:.1f} GB ({t['headroom_gb']:.1f} GB headroom) | "
        f"reserve {t['reserve_gb']:.0f} GB | risk {t['total_risk_score']}",
        "",
        f"Breadth: modalities {t['modalities_covered']} | domains {t['domains_covered']}",
        "",
        "## Selected sources",
        "",
        "| slug | kind | modality | domain | subset | raw GB | cache GB | risk | method |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in p["selected"]:
        L.append(
            f"| {s['slug']} | {s['kind']} | {s['modality']} | {s['domain']} | "
            f"{s['subset_clips'] if s['subset_clips'] is not None else 'all'} | "
            f"{s['raw_gb']:.2f} | {s['cache_gb']:.3f} | {s['risk_score']} | {s['download_method']} |"
        )
    L += ["", "## Skipped sources", "", "| slug | status | reason |", "| --- | --- | --- |"]
    for s in p["skipped"]:
        L.append(f"| {s['slug']} | {s['status']} | {s['reason']} |")
    if p["license_blockers"]:
        L += ["", "## License blockers (resolve before the Studio run)", ""]
        L += [f"- **{b['slug']}** ({b['status']}): {b['reason']}" for b in p["license_blockers"]]
    L += ["", "## Next commands", "", "```"] + p["next_commands"] + ["```", ""]
    return "\n".join(L)
