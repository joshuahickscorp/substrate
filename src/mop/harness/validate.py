from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from ..config import REPO_ROOT

DEVICE_KINDS = {"cpu", "mps", "cuda"}


class ConfigError(ValueError):
    pass


def _declared_null_contract(cfg: DictConfig) -> str:
    for path in ("null_hypothesis", "payload.strong_null", "payload.null"):
        value = OmegaConf.select(cfg, path, default="")
        if str(value).strip():
            return str(value)
    return ""


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
    if bool(enc.get("prefer_real", False)) and not bool(enc.get("available", True)):
        raise ConfigError(
            f"encoder {enc.get('name')} is available=false (weights not on HF) but prefer_real=true; "
            "it cannot load real weights. Use an available encoder or set prefer_real=false."
        )


def validate_config(cfg: DictConfig) -> None:
    validate_device(cfg)
    validate_experiment(cfg)
    validate_encoder(cfg)


def check_all() -> list[dict]:
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
        if not _declared_null_contract(cfg):
            problems.append({"where": f"experiment/{f.stem}", "problem": "no null_hypothesis"})
    return problems
