"""Config system. OmegaConf with a small Hydra-style group composition layer.

Why not Hydra: the corpus pre-authorizes Hydra OR a lighter route; Hydra drags in
antlr+plugins and owns your entrypoint. We need exactly two things from it: group
selection (device=mps, encoder=..., experiment=...) and dotlist overrides. Both are
~40 lines on OmegaConf, which we already depend on. One source of truth, fewer deps.

config.yaml carries a `defaults:` map naming one file per group under configs/<group>/.
compose() loads each, nests it under the group key, overlays config.yaml's own keys,
then applies CLI-style dotlist overrides (e.g. shell.buffer.capacity=2048).
"""

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
    """Compose the full config tree from groups + dotlist overrides.

    overrides may include group switches (`experiment=e1_baseline`) and value dotlist
    overrides (`shell.buffer.capacity=4096`). Group switches are pulled out first.
    """
    cdir = config_dir or CONFIG_DIR
    root = _load_yaml(cdir / base)
    defaults = dict(root.pop("defaults", {})) if "defaults" in root else {}

    group_over, dot_over = {}, []
    for o in overrides or []:
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
    """Write a frozen yaml snapshot of the resolved config for a run. Reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return path


def get(cfg: DictConfig, dotted: str, default=None):
    return OmegaConf.select(cfg, dotted, default=default)
