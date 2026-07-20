from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
GROUPS = ("device", "encoder", "shell", "experiment")


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig), f"{path} must be a mapping"
    return cfg


def compose(
    overrides: list[str] | None = None,
    config_dir: Path | None = None,
    base: str = "config.yaml",
) -> DictConfig:
    cdir = config_dir or CONFIG_DIR
    root = _load_yaml(cdir / base)
    defaults = dict(root.pop("defaults", {})) if "defaults" in root else {}

    group_over, dot_over = {}, []
    for o in overrides or []:
        o = o.lstrip("+")  # Hydra-style add-key prefix; OmegaConf merge adds keys anyway
        key, _, val = o.partition("=")
        if key in GROUPS:
            group_over[key] = val
        else:
            dot_over.append(o)
    defaults.update(group_over)

    merged: DictConfig = OmegaConf.create({})
    for group in GROUPS:
        name = defaults.get(group)
        if name in (None, "null", ""):
            continue
        gcfg = _load_yaml(cdir / group / f"{name}.yaml")
        merged = OmegaConf.merge(merged, OmegaConf.create({group: gcfg, f"{group}_name": name}))  # type: ignore[assignment]

    merged = OmegaConf.merge(merged, root)  # type: ignore[assignment]
    if dot_over:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(dot_over))  # type: ignore[assignment]
    OmegaConf.resolve(merged)
    assert isinstance(merged, DictConfig)
    return merged


def snapshot(cfg: DictConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return path


def get(cfg: DictConfig, dotted: str, default=None):
    return OmegaConf.select(cfg, dotted, default=default)
