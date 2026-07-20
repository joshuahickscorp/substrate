
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ..devices import DeviceInfo
from ..substrate.p5_context import P5CellSpec, run_p5_pilot
from .base import Experiment


def _plain(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("P5 context experiment config must resolve to a mapping")
    return cast(dict[str, Any], value)


class P5Context(Experiment):
    id = "mop_p5_context_capability"
    metric = ("heldout_combo_per_context", "context_response_curve")
    baseline = (
        "per-mechanism exact frozen initialization plus the exact-global reference arm at "
        "matched parameters and matched active FLOPs"
    )
    ablation = (
        "context mechanism (exact global attention vs windowed, recurrent, and hierarchical "
        "pooled factorizations) at 16, 32, and 64 frames on one shared substrate trunk"
    )
    null_hypothesis = (
        "at matched parameters and matched active FLOPs, windowed, recurrent, and hierarchical "
        "factorized context mechanisms match exact global attention on held-out factor "
        "combinations within the 0.10 SESOI at 32 and 64 frames while peaking at lower measured "
        "bytes; exact global interaction is not a live capability lever in this context range"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        config = _plain(cfg.experiment)
        profiles = config.pop("profiles", {})
        smoke = profiles.get("p5smoke", {"seeds": [0], "dense_steps": 12, "checkpoint_every": 6})
        config["training"] = {**config.get("training", {}), **smoke, "dense_steps": 2, "checkpoint_every": 1}
        config["profile"] = "p5smoke-two-cell"
        cells = [P5CellSpec(16, "exact_global"), P5CellSpec(16, "window_local")]
        receipt = run_p5_pilot(
            config,
            run_dir,
            device.kind,
            cells=cells,
            corpus_overrides={"replicates": 3},
            model_overrides={"dim": 32},
        )
        frame_payload = receipt.get("frames", {}).get("f16", {})
        scores = frame_payload.get("scores", {})
        return {
            "heldout_combo_per_context": {
                "f16": {mechanism: row.get("mean") for mechanism, row in scores.items()}
            },
            "context_response_curve": {},
            "complete": receipt.get("complete"),
            "null_supported": True,
            "scientific_scope": (
                "bounded two-cell execution and integrity smoke of the registered P5 codepath; "
                "the pilot itself runs through scripts/p5_context_capability.py and no smoke "
                "result is a scientific verdict"
            ),
            "receipt": "p5_context_receipt.json",
        }


__all__ = ["P5Context"]
