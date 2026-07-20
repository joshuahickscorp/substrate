
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
    n = len(seeds) if seeds else 1
    for v in axes.values():
        n *= max(1, len(v)) if isinstance(v, list) else 1
    return n


def toy_run_units(leg: dict) -> int:
    return _count(dict(leg.get("axes", {})), list(leg.get("seeds", [0])))


def full_run_units(leg: dict) -> int:
    return _count(
        dict(leg.get("full_axes", leg.get("axes", {}))),
        list(leg.get("full_seeds", leg.get("seeds", [0]))),
    )
