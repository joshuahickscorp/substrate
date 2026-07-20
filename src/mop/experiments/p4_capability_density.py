
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ..devices import DeviceInfo
from ..substrate.p4_screen import P4_CELLS, run_p4_screen
from .base import Experiment


def _plain(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("P4 screen experiment config must resolve to a mapping")
    return cast(dict[str, Any], value)


class P4Screen(Experiment):
    id = "mop_p4_capability_density_screen"
    metric = ("heldout_combo_per_cell", "response_surface_coefficients")
    baseline = (
        "per-cell exact frozen initialization plus compute-matched random-target training, "
        "with a random-search configuration control at the screen level"
    )
    ablation = (
        "parameter budget 0.5x to 4x, dense vs recurrent vs sparse-gated block family, referent "
        "diversity, and temporal context on one shared tubelet-embedding substrate contract"
    )
    null_hypothesis = (
        "across twelve matched cells spanning 0.5x to 4x parameter budgets, three block families, "
        "two corpus densities, and two temporal extents, every registered response-surface "
        "coefficient on the predictive held-out combination score is bounded within the 0.10 SESOI "
        "band; capability density has no architecture/budget lever at this scale and follow-up "
        "cell selection is not better than uniform"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        config = _plain(cfg.experiment)
        profiles = config.pop("profiles", {})
        smoke = profiles.get("p4smoke", {"seeds": [0], "steps": 12, "checkpoint_every": 6})
        config["training"] = {**config.get("training", {}), **smoke}
        config["profile"] = "p4smoke-single-cell"
        cell = next(spec for spec in P4_CELLS if spec.cell_id == "C10")
        receipt = run_p4_screen(config, run_dir, device.kind, cells=[cell])
        cell_payload = receipt["cells"].get("C10", {})
        return {
            "heldout_combo_per_cell": {
                "C10": cell_payload.get("scores", {}).get("predictive", {}).get("mean")
            },
            "response_surface_coefficients": {},
            "complete": receipt.get("complete"),
            "null_supported": True,
            "scientific_scope": (
                "bounded single-cell execution and integrity smoke of the registered P4 codepath; "
                "the screen itself runs through scripts/p4_capability_density.py and no smoke "
                "result is a scientific verdict"
            ),
            "receipt": "p4_screen_receipt.json",
        }


__all__ = ["P4Screen"]
