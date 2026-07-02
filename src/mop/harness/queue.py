"""Campaign queue. Reads campaign/run_queue.yaml (the single ordered, tier-tagged,
dependency-aware manifest), validates it (deps resolve, no cycles, tiers valid), topo-orders
it, and runs the enabled legs whose tier is enabled and whose deps are satisfied. Tier
gating: C (cached-latent) is on by default; E (environment) and R (rented CUDA) are off.

Toy scale here for validation; full scale on the Studio by flipping tiers + dropping
toy_overrides in the leg sweeps. Dry-run resolves and prints the plan without running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from ..config import REPO_ROOT
from ..logging_utils import get_logger
from .sweep import load_leg, run_sweep

log = get_logger("queue")
TIERS = {"C", "E", "R"}


@dataclass
class Leg:
    name: str
    track: str
    experiment: str
    tier: str
    sweep: str
    depends_on: list[str] = field(default_factory=list)
    run_units: int = 1  # canonical FULL-scale run-units (full_axes x full_seeds)
    enabled: bool = True
    note: str = ""
    toy_run_units: int = 0  # laptop-validation run-units (axes x seeds)


def load_queue(path: Path | None = None) -> list[Leg]:
    p = Path(path) if path else REPO_ROOT / "campaign" / "run_queue.yaml"
    data = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    assert isinstance(data, dict) and "legs" in data, f"bad run_queue at {p}"
    return [Leg(**dict(leg)) for leg in data["legs"]]


def validate(legs: list[Leg]) -> None:
    names = {leg.name for leg in legs}
    if len(names) != len(legs):
        raise ValueError("duplicate leg names in run_queue")
    for leg in legs:
        if leg.tier not in TIERS:
            raise ValueError(f"{leg.name}: bad tier {leg.tier!r}")
        for d in leg.depends_on:
            if d not in names:
                raise ValueError(f"{leg.name}: depends on unknown leg {d!r}")
    topo_order(legs)  # raises on cycle


def topo_order(legs: list[Leg]) -> list[Leg]:
    by_name = {leg.name: leg for leg in legs}
    order, seen, stack = [], set(), set()

    def visit(n: str):
        if n in seen:
            return
        if n in stack:
            raise ValueError(f"dependency cycle at {n}")
        stack.add(n)
        for d in by_name[n].depends_on:
            visit(d)
        stack.discard(n)
        seen.add(n)
        order.append(by_name[n])

    for leg in legs:
        visit(leg.name)
    return order


def plan(legs: list[Leg], enabled_tiers: set[str], run_disabled: bool = False) -> list[Leg]:
    """Topo-ordered legs whose tier is enabled and that are enabled, with all deps also in
    the plan (a leg whose dep is gated out is itself skipped, surfaced in the log)."""
    ordered = topo_order(legs)
    chosen: list[Leg] = []
    chosen_names: set[str] = set()
    for leg in ordered:
        if leg.tier not in enabled_tiers or (not leg.enabled and not run_disabled):
            continue
        if not all(d in chosen_names for d in leg.depends_on):
            log.info("skip %s: a dependency is gated out", leg.name)
            continue
        chosen.append(leg)
        chosen_names.add(leg.name)
    return chosen


def _skip_reasons(legs: list[Leg], chosen: list[Leg], tiers: set[str], run_disabled: bool) -> list[dict]:
    """Classify why each non-planned leg was skipped: tier gated, disabled, or dependency gated."""
    chosen_names = {l.name for l in chosen}
    out = []
    for leg in legs:
        if leg in chosen:
            continue
        if leg.tier not in tiers:
            reason = f"tier {leg.tier} not enabled"
        elif not leg.enabled and not run_disabled:
            reason = "disabled"
        elif not all(d in chosen_names for d in leg.depends_on):
            reason = "dependency gated out"
        else:
            reason = "not planned"
        out.append({"name": leg.name, "tier": leg.tier, "reason": reason})
    return out


def _next_commands(tiers: set[str]) -> list[str]:
    cmds = ["python scripts/run_queue.py --dry-run --tiers C   # resolve the Tier C plan"]
    if tiers == {"C"}:
        cmds += [
            "python scripts/run_queue.py --tiers C              # toy validation on this laptop",
            "python scripts/run_queue.py --tiers C --full       # full grids (Studio / Apple Silicon)",
        ]
    else:
        cmds.append("python scripts/run_queue.py --tiers C,E,R --full --run-disabled  # Studio + rented CUDA")
    return cmds


def run_queue(
    path: Path | None = None,
    dry_run: bool = True,
    enabled_tiers: set[str] | None = None,
    toy: bool = True,
    max_runs_per_leg: int | None = 1,
    run_disabled: bool = False,
    max_legs: int | None = None,
) -> dict:
    legs = load_queue(path)
    validate(legs)
    tiers = enabled_tiers or {"C"}
    chosen = plan(legs, tiers, run_disabled)
    if max_legs is not None:
        chosen = chosen[:max_legs]
    units = "toy_run_units" if toy else "run_units"
    scale = "toy" if toy else "full"
    totals = {
        "planned_legs": len(chosen),
        "run_units_this_scale": sum(getattr(l, units) for l in chosen),
        "run_units_full": sum(l.run_units for l in chosen),
        "run_units_toy": sum(l.toy_run_units for l in chosen),
    }
    summary = {
        "total_legs": len(legs),
        "enabled_tiers": sorted(tiers),
        "scale": scale,
        "planned": [
            {
                "name": l.name,
                "track": l.track,
                "experiment": l.experiment,
                "tier": l.tier,
                "depends_on": l.depends_on,
                "run_units_full": l.run_units,
                "run_units_toy": l.toy_run_units,
            }
            for l in chosen
        ],
        "skipped": _skip_reasons(legs, chosen, tiers, run_disabled),
        "gated_out": [l.name for l in legs if l not in chosen],  # names only (back-compat)
        "totals": totals,
        "next": _next_commands(tiers),
        "dry_run": dry_run,
    }
    if dry_run:
        log.info(
            "dry-run: %d/%d legs planned (tiers=%s, scale=%s, %d %s run-units)",
            len(chosen),
            len(legs),
            sorted(tiers),
            scale,
            totals["run_units_this_scale"],
            scale,
        )
        return summary
    results = {}
    for leg in chosen:
        log.info("running leg %s (%s, tier %s)", leg.name, leg.experiment, leg.tier)
        leg_cfg = load_leg(REPO_ROOT / leg.sweep)
        results[leg.name] = run_sweep(leg_cfg, toy=toy, max_runs=max_runs_per_leg)
    summary["results"] = {k: len(v) for k, v in results.items()}
    summary["raw"] = results
    return summary
