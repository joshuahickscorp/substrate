from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ..devices import DeviceInfo
from ..evidence import atomic_write_json
from ..substrate.custom_workbench import cm8_preflight, run_workbench
from .base import Experiment


def _plain(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("custom-substrate experiment config must resolve to a mapping")
    return cast(dict[str, Any], value)


class CM7(Experiment):
    id = "mop_cm7_min_objective_probe"
    metric = ("nuisance_invariance_per_objective", "held_out_combo_per_objective")
    baseline = "exact frozen random initialization plus compute-matched random-target training"
    ablation = "predictive vs invariance vs reconstruction objectives on one identical video-token encoder"
    null_hypothesis = (
        "at matched tiny capacity, matched data, matched 256px, both custom objectives tie random-init "
        "same-arch AND tie each other; objective is not a lever at this scale and the +0.31 was "
        "scale/data/architecture/resolution. A tie is a strong negative closing the custom-encoder line"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        receipt = run_workbench(_plain(cfg.experiment), run_dir=run_dir, device=device)
        aggregate = receipt.get("aggregate", {})
        invariance = {
            objective: row["referent_view_stability"]["mean"]
            for objective, row in aggregate.items()
            if "referent_view_stability" in row
        }
        heldout = {
            objective: row["heldout_combo_score"]["mean"]
            for objective, row in aggregate.items()
            if "heldout_combo_score" in row
        }
        return {
            "nuisance_invariance_per_objective": invariance,
            "held_out_combo_per_objective": heldout,
            "trainable_parameters": receipt["model"]["trainable_parameters"],
            "estimated_flops_by_objective": receipt["compute_match"].get("mean_flops", {}),
            "compute_matched": receipt["compute_match"]["all_ok"],
            "requirements_all_ok": receipt["requirements"]["all_ok"],
            "complete": receipt["complete"],
            "null_supported": not receipt["promotion"]["cm7_local_objective_lever_promotable"],
            "scientific_scope": "programmatic-video local objective probe",
            "receipt": "workbench_receipt.json",
        }


class CM8(Experiment):
    id = "mop_cm8_custom_jepa_pilot"
    metric = ("held_out_factoring_off_ceiling",)
    baseline = "best citable frozen teacher cache plus exact random-init same-architecture control"
    ablation = "teacher-free predictive objective vs optional same-referent distillation"
    null_hypothesis = (
        "the custom-JEPA pilot ties both a random-init same-arch ViT and the best frozen atlas "
        "substrate on held-out combinations at matched compute: the custom objective bought nothing"
    )
    tier = "env-later"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        del device
        receipt = cm8_preflight(_plain(cfg.experiment))
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / "cm8_preflight.json", receipt)
        return {
            "held_out_factoring_off_ceiling": 0.0,
            "runnable_local_preflight": receipt["runnable_local_preflight"],
            "scientific_execution_ready": receipt["scientific_execution_ready"],
            "scientific_promotion": False,
            "blockers": receipt["blockers"],
            "receipt": "cm8_preflight.json",
        }


__all__ = ["CM7", "CM8"]
