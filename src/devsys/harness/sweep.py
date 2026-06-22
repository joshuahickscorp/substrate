"""Sweep expansion. A leg declares axes (a dict of override-key -> list of values) and a
seed set; expand() takes the cartesian product x seeds into concrete override lists, each of
which is composed and run via run_experiment. Toy by default on this laptop; full scale on
the Studio is the same code with the toy_overrides dropped.
"""

from __future__ import annotations

from itertools import product

from omegaconf import OmegaConf

from ..config import compose


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


def expand(axes: dict[str, list], seeds: list[int], base: list[str]) -> list[list[str]]:
    keys = list(axes)
    grids = list(product(*[axes[k] for k in keys])) if keys else [()]
    out = []
    for s in seeds:
        for vals in grids:
            axis_ov = [f"{k}={_fmt(v)}" for k, v in zip(keys, vals, strict=False)]
            out.append(list(base) + [f"seed={s}"] + axis_ov)
    return out


def run_sweep(leg: dict, toy: bool = True, max_runs: int | None = None) -> list[dict]:
    """Run every (axis-combo x seed) in a leg. Returns [{overrides, metrics}].

    toy: the reduced axis subset + small overrides (the laptop validation grid).
    full: the genuine factorial (leg['full_axes'] if present, else 'axes') x full seed set
    (leg['full_seeds'] if present, else 'seeds'), with toy_overrides dropped. The Studio runs
    `run_queue.py --full`; the cost projection is computed against these full grids."""
    base = [f"experiment={leg['experiment']}"] + list(leg.get("base_overrides", []))
    if toy:
        base += list(leg.get("toy_overrides", []))
        axes, seeds = dict(leg.get("axes", {})), list(leg.get("seeds", [0]))
    else:
        axes = dict(leg.get("full_axes", leg.get("axes", {})))
        seeds = list(leg.get("full_seeds", leg.get("seeds", [0])))
    combos = expand(axes, seeds, base)
    if max_runs is not None:
        combos = combos[:max_runs]
    from .runner import run_experiment  # lazy: avoids sweep<->runner import cycle

    results = []
    for ov in combos:
        cfg = compose(ov)
        results.append({"overrides": ov, "metrics": run_experiment(cfg)})
    return results


def load_leg(path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _count(axes: dict, seeds: list) -> int:
    """Run-units = |seeds| x product of axis lengths. The ONE definition of a leg's size, shared
    by run_sweep (what actually runs), the run_queue manifest, and the cost projection."""
    n = len(seeds) if seeds else 1
    for v in axes.values():
        n *= max(1, len(v)) if isinstance(v, list) else 1
    return n


def toy_run_units(leg: dict) -> int:
    return _count(dict(leg.get("axes", {})), list(leg.get("seeds", [0])))


def full_run_units(leg: dict) -> int:
    """The canonical full-scale run-unit count: full_axes x full_seeds (falling back to axes/seeds
    if a leg declares no full grid). `run_queue --full` expands exactly this many units per leg,
    the run_queue manifest declares it, and cost_projection projects it. They agree by construction."""
    return _count(
        dict(leg.get("full_axes", leg.get("axes", {}))),
        list(leg.get("full_seeds", leg.get("seeds", [0]))),
    )
