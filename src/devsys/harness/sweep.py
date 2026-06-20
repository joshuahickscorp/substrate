"""Sweep expansion. A leg declares axes (a dict of override-key -> list of values) and a
seed set; expand() takes the cartesian product x seeds into concrete override lists, each of
which is composed and run via run_experiment. Toy by default on this laptop; full scale on
the Studio is the same code with the toy_overrides dropped.
"""

from __future__ import annotations

from itertools import product

from omegaconf import OmegaConf

from ..config import compose
from .runner import run_experiment


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
    """Run every (axis-combo x seed) in a leg. Returns [{overrides, metrics}]."""
    base = [f"experiment={leg['experiment']}"] + list(leg.get("base_overrides", []))
    if toy:
        base += list(leg.get("toy_overrides", []))
    combos = expand(dict(leg.get("axes", {})), list(leg.get("seeds", [0])), base)
    if max_runs is not None:
        combos = combos[:max_runs]
    results = []
    for ov in combos:
        cfg = compose(ov)
        results.append({"overrides": ov, "metrics": run_experiment(cfg)})
    return results


def load_leg(path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]
