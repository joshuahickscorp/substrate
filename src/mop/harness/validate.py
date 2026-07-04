"""Fail-fast validation. Catches the operator mistakes that would otherwise fail deep in a run
or, worse, silently do the wrong thing: a bad device kind, an unknown tier, a leg missing its
full grid, a toy axis the full grid lacks, a placeholder (unavailable) encoder asked to load
real weights, an experiment with no null. Raise early with a clear message, or collect every
problem at once for the Studio doctor.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..config import REPO_ROOT
from .sweep import full_run_units, load_leg

DEVICE_KINDS = {"cpu", "mps", "cuda"}
TIERS = {"C", "E", "R"}


class ConfigError(ValueError):
    """A configuration that cannot be trusted to run correctly."""


def validate_device(cfg: DictConfig) -> None:
    kind = str(OmegaConf.select(cfg, "device.kind", default=""))
    if kind not in DEVICE_KINDS:
        raise ConfigError(f"device.kind={kind!r} not in {sorted(DEVICE_KINDS)}")


def validate_experiment(cfg: DictConfig) -> None:
    e = cfg.get("experiment")
    if not e or not e.get("id"):
        raise ConfigError("experiment.id missing")
    if not str(e.get("null_hypothesis", "")).strip():
        raise ConfigError(f"experiment {e.get('id')} declares no null_hypothesis (doctrine contract)")


def validate_encoder(cfg: DictConfig) -> None:
    enc = cfg.get("encoder")
    if not enc:
        return
    if int(enc.get("embed_dim", 0)) <= 0:
        raise ConfigError(f"encoder {enc.get('name')} has non-positive embed_dim")
    # a placeholder (unavailable) encoder must NOT be asked to load real weights and pretend it is real
    if bool(enc.get("prefer_real", False)) and not bool(enc.get("available", True)):
        raise ConfigError(
            f"encoder {enc.get('name')} is available=false (weights not on HF) but prefer_real=true; "
            "it cannot load real weights. Use an available encoder or set prefer_real=false."
        )


def validate_config(cfg: DictConfig) -> None:
    validate_device(cfg)
    validate_experiment(cfg)
    validate_encoder(cfg)


def validate_leg(leg: dict, known_experiments: set[str] | None = None) -> list[str]:
    """Return a list of problems with a leg dict (empty == ok)."""
    probs = []
    if leg.get("tier") not in TIERS:
        probs.append(f"tier {leg.get('tier')!r} not in {sorted(TIERS)}")
    if "full_axes" not in leg or "full_seeds" not in leg:
        probs.append("missing full_axes/full_seeds (run_queue --full + cost projection need them)")
    toy = set(dict(leg.get("axes", {})))
    full = set(dict(leg.get("full_axes", leg.get("axes", {}))))
    if not toy <= full:
        probs.append(f"toy axes {toy - full} not in full_axes")
    if full_run_units(leg) < 1:
        probs.append("full_run_units < 1")
    if known_experiments is not None and leg.get("experiment") not in known_experiments:
        probs.append(f"unknown experiment {leg.get('experiment')!r}")
    return probs


def check_all() -> list[dict]:
    """Validate every encoder config, every experiment config, and every queued leg. Returns a
    flat list of {where, problem} (empty == clean). Never raises: this is the doctor's surface."""
    from ..experiments import REGISTRY
    from .queue import load_queue

    problems: list[dict] = []
    cdir = REPO_ROOT / "configs"
    for f in sorted((cdir / "encoder").glob("*.yaml")):
        enc = OmegaConf.load(f)
        try:
            validate_encoder(OmegaConf.create({"encoder": enc}))
        except ConfigError as e:
            problems.append({"where": f"encoder/{f.stem}", "problem": str(e)})
    for f in sorted((cdir / "experiment").glob("*.yaml")):
        cfg = OmegaConf.load(f)
        if f.name == "_mot_mirrors.yaml":  # collapsed MoT preregistration mirrors: check each entry
            for m in OmegaConf.select(cfg, "mirrors", default=[]):
                if not str(OmegaConf.select(m, "null_hypothesis", default="")).strip():
                    problems.append(
                        {"where": f"experiment/mirror/{m.get('id')}", "problem": "no null_hypothesis"}
                    )
            continue
        nh = OmegaConf.select(cfg, "null_hypothesis", default="")
        if not str(nh).strip():
            problems.append({"where": f"experiment/{f.stem}", "problem": "no null_hypothesis"})
    known = set(REGISTRY)
    for leg in load_queue():
        d = load_leg(Path(leg.sweep)) if Path(leg.sweep).is_absolute() else load_leg(REPO_ROOT / leg.sweep)
        for p in validate_leg(d, known):
            problems.append({"where": f"leg/{leg.name}", "problem": p})
        if leg.run_units != full_run_units(d):
            problems.append({"where": f"leg/{leg.name}", "problem": "manifest run_units != full grid"})
    return problems
