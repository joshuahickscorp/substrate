
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ..config import REPO_ROOT
from ..harness.queue import Leg, load_queue
from ..harness.sweep import full_run_units
from ..logging_utils import get_logger

log = get_logger("cost_projection")

TIER_FULL = {
    "C": {"unit_mult": 1, "assumed_per_unit_s": 0.0, "basis": "measured"},
    "E": {"unit_mult": 4, "assumed_per_unit_s": 1800.0, "basis": "assumed"},
    "R": {"unit_mult": 8, "assumed_per_unit_s": 7200.0, "basis": "assumed"},
}
ASSUMED_C_PER_UNIT_S = 30.0  # fallback when a C leg has no measured timing yet


def _leg_full_units(sweep_path: Path) -> int:
    p = sweep_path if sweep_path.is_absolute() else REPO_ROOT / sweep_path
    if not p.exists():
        log.info("sweep yaml missing for cost projection: %s (units=1)", p)
        return 1
    d = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    return full_run_units(d) if isinstance(d, dict) else 1


def _per_unit_measured(leg: Leg, timings: dict[str, float]) -> float | None:
    if leg.name in timings:
        return float(timings[leg.name])
    if leg.experiment in timings:
        return float(timings[leg.experiment])
    return None


def _run_units_full(leg: Leg, tier: dict, full_seed: int) -> int:
    base = _leg_full_units(Path(leg.sweep))
    return base if leg.tier == "C" else base * int(tier["unit_mult"])


def cost_projection(timings: dict, workers: int = 10, full_seed: int = 5) -> dict:
    workers = max(1, int(workers))
    full_seed = int(full_seed)
    legs = load_queue()

    per_leg: list = []
    totals_by_tier: dict[str, dict] = {}
    grand_serial = 0.0
    for leg in legs:
        tier = TIER_FULL.get(leg.tier, TIER_FULL["C"])
        units = _run_units_full(leg, tier, full_seed)
        measured = _per_unit_measured(leg, timings)
        if measured is not None and leg.tier == "C":
            per_unit, basis = measured, "measured"
        elif leg.tier == "C":
            per_unit, basis = ASSUMED_C_PER_UNIT_S, "assumed"  # C leg not yet timed
        else:
            per_unit, basis = float(tier["assumed_per_unit_s"]), "assumed"  # type: ignore[arg-type]  # E/R: never cpu-timed
        serial = units * per_unit
        leg_workers = min(workers, max(1, units))
        per_leg.append(
            {
                "name": leg.name,
                "tier": leg.tier,
                "enabled": leg.enabled,
                "run_units_full": units,
                "per_unit_s_measured_or_assumed": round(per_unit, 4),
                "serial_s": round(serial, 3),
                "parallel_wall_s": round(serial / leg_workers, 3),
                "basis": basis,
            }
        )
        t = totals_by_tier.setdefault(leg.tier, {"serial_s": 0.0, "run_units_full": 0, "legs": 0})
        t["serial_s"] += serial
        t["run_units_full"] += units
        t["legs"] += 1
        grand_serial += serial

    for t in totals_by_tier.values():
        t["serial_s"] = round(t["serial_s"], 3)
        t["parallel_wall_s"] = round(t["serial_s"] / workers, 3)

    notes = [
        "laptop-throttled: measured cpu times are thermal/shared-core throttle-limited and "
        "CONSERVATIVE vs the Studio; full-scale Studio wall-clock will be no worse, likely faster.",
        f"Tier C: run_units_full = each leg's full_axes x full_seeds (the SAME count run_queue "
        f"--full expands and the manifest declares); basis=measured where a per-unit timing was "
        f"supplied, else assumed at {ASSUMED_C_PER_UNIT_S}s/unit.",
        f"Tier E/R: cannot be cpu-timed (need env / rented CUDA); run_units scaled by a stated "
        f"cost-class multiplier (E x{TIER_FULL['E']['unit_mult']}, R x{TIER_FULL['R']['unit_mult']}) "
        f"with ASSUMPTION-BASED per-unit cost (E {TIER_FULL['E']['assumed_per_unit_s']}s, "
        f"R {TIER_FULL['R']['assumed_per_unit_s']}s). These are levers, not measurements.",
        f"parallel wall = serial_sum / effective_workers (workers={workers}, "
        f"per-leg capped by that leg's run_units).",
    ]
    return {
        "per_leg": per_leg,
        "totals_by_tier": totals_by_tier,
        "grand_total_serial_s": round(grand_serial, 3),
        "grand_total_parallel_wall_s": round(grand_serial / workers, 3),
        "workers": workers,
        "full_seed": full_seed,
        "notes": notes,
        "provenance": "provisional",  # timings are synthetic-latent measured / assumption-based
    }


def load_timings_from_checkpoint(ckpt_dir: Path) -> dict[str, float]:
    import json

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for p in Path(ckpt_dir).rglob("*.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        name = m.get("name")
        st, fin = m.get("started"), m.get("finished")
        if not name or st is None or fin is None:
            continue
        dt = float(fin) - float(st)
        if dt < 0:
            continue
        sums[name] = sums.get(name, 0.0) + dt
        counts[name] = counts.get(name, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def render_md(proj: dict) -> str:
    lines = [
        "# Studio cost projection (provisional)",
        "",
        f"Workers: {proj['workers']} | full_seed: {proj['full_seed']} | "
        f"grand serial: {proj['grand_total_serial_s']:.1f}s | "
        f"grand parallel wall: {proj['grand_total_parallel_wall_s']:.1f}s "
        f"({proj['grand_total_parallel_wall_s'] / 3600:.2f}h)",
        "",
        "## Notes",
    ]
    lines += [f"- {n}" for n in proj["notes"]]
    lines += ["", "## Totals by tier", ""]
    lines += [
        "| tier | legs | run_units_full | serial_s | parallel_wall_s |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tier, t in sorted(proj["totals_by_tier"].items()):
        lines.append(
            f"| {tier} | {t['legs']} | {t['run_units_full']} | "
            f"{t['serial_s']:.1f} | {t['parallel_wall_s']:.1f} |"
        )
    lines += ["", "## Per leg", ""]
    lines += [
        "| name | tier | en | run_units_full | per_unit_s | serial_s | parallel_wall_s | basis |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in proj["per_leg"]:
        lines.append(
            f"| {r['name']} | {r['tier']} | {'y' if r['enabled'] else 'n'} | {r['run_units_full']} | "
            f"{r['per_unit_s_measured_or_assumed']:.3f} | {r['serial_s']:.1f} | "
            f"{r['parallel_wall_s']:.1f} | {r['basis']} |"
        )
    return "\n".join(lines) + "\n"
