from __future__ import annotations

from .registries import load_capacities, load_paradigms

_COST_HOURS = {
    "local": {"low": 0.25, "medium": 1.0, "high": 4.0},
    "studio": {"low": 1.0, "medium": 6.0, "high": 24.0},
}
_STATUS_EIG = {
    "implement-now": 1.0,
    "prototype-local": 0.8,
    "studio-later": 0.4,
    "paper-watch": 0.1,
    "blocked": 0.0,
}
_RUNNABLE = {
    "local": {"implement-now", "prototype-local"},
    "studio": {"implement-now", "prototype-local", "studio-later"},
}


def _cost_hours(compute_risk: str, scope: str) -> float:
    return _COST_HOURS[scope].get(str(compute_risk).lower(), _COST_HOURS[scope]["medium"])


def plan_ablations(scope: str = "local", budget_hours: float | None = None) -> dict:
    if scope not in ("local", "studio"):
        raise ValueError(f"scope must be local or studio, got {scope!r}")
    paradigms = load_paradigms()
    runnable_statuses = _RUNNABLE[scope]

    ranked = []
    for p in paradigms:
        status = p["status"]
        est = _cost_hours(p["compute_risk"], scope)
        eig = _STATUS_EIG.get(status, 0.0)
        is_runnable = status in runnable_statuses
        gate = "" if is_runnable else f"status {status} not runnable in {scope} scope"
        ranked.append(
            {
                "slug": p["slug"],
                "family": p["family"],
                "changes": p["changes"],
                "status": status,
                "est_hours": est,
                "expected_info_gain": eig,
                "eig_per_hour": round(eig / est, 3) if est else 0.0,
                "runnable_now": is_runnable,
                "gate": gate,
                "metric": p["metric"],
                "null_hypothesis": p["null_hypothesis"],
                "basis": "assumption",  # est_hours + eig are levers, NOT measurements (travels with the row)
            }
        )
    ranked.sort(key=lambda r: (r["runnable_now"], r["eig_per_hour"]), reverse=True)

    groups: dict[tuple, list[str]] = {}
    for r in ranked:
        groups.setdefault((r["family"], r["changes"]), []).append(r["slug"])
    competing = [{"family": k[0], "changes": k[1], "candidates": v} for k, v in groups.items() if len(v) > 1]

    runnable = [r for r in ranked if r["runnable_now"]]
    next_best = runnable[0] if runnable else None
    within_budget = []
    if budget_hours is not None:
        spent = 0.0
        for r in runnable:
            if spent + r["est_hours"] <= budget_hours:
                within_budget.append(r["slug"])
                spent += r["est_hours"]
    notes = [
        "est_hours and expected_info_gain are EXPLICIT ASSUMPTIONS (each row carries basis=assumption), "
        "not measurements; wire realized-vs-predicted gain on the Studio to calibrate them.",
        "combine mechanisms only AFTER each one's isolated gate passes (no blind chaining).",
        "competing_groups change the same thing in the same family (same slot); run one before any combo.",
        "within_budget is a greedy hour-prefix of the runnable set, NOT a redundancy-deduplicated set.",
        f"capacity ladder has {len(load_capacities())} rungs; a mechanism targeting an untested rung is "
        f"worth more.",
    ]
    return {
        "scope": scope,
        "ranked": ranked,
        "next_best": next_best,
        "competing_groups": competing,
        "within_budget": within_budget,
        "budget_hours": budget_hours,
        "notes": notes,
    }


def render_md(plan: dict) -> str:
    L = [
        f"# Ablation plan ({plan['scope']} scope)",
        "",
        "Ranked by expected information gain per compute hour (assumptions, not measurements).",
        "",
    ]
    nb = plan["next_best"]
    if nb:
        L += [
            f"**Next best experiment**: `{nb['slug']}` ({nb['family']}, {nb['est_hours']}h, "
            f"eig/h {nb['eig_per_hour']}) -> tests: {nb['metric']}",
            "",
        ]
    else:
        L += ["**Next best experiment**: none runnable in this scope", ""]
    L += [
        "| slug | family | changes | status | est_h | eig/h | runnable | gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in plan["ranked"]:
        L.append(
            f"| {r['slug']} | {r['family']} | {r['changes']} | {r['status']} | {r['est_hours']} | "
            f"{r['eig_per_hour']} | {'yes' if r['runnable_now'] else 'no'} | {r['gate']} |"
        )
    if plan["competing_groups"]:
        L += ["", "## Competing groups (same subsystem; run one before combining)", ""]
        L += [f"- {g['family']}/{g['changes']}: {g['candidates']}" for g in plan["competing_groups"]]
    if plan.get("within_budget"):
        L += ["", f"## Within {plan['budget_hours']}h budget", "", f"- {plan['within_budget']}"]
    L += ["", "## Notes", ""] + [f"- {n}" for n in plan["notes"]]
    return "\n".join(L) + "\n"
