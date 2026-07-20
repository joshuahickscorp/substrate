from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ..devices import DeviceInfo
from ..evidence import atomic_write_json
from ..substrate.custom_workbench import cm8_preflight, run_workbench
from .base import Executor, Verifier


def _plain(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("custom-substrate experiment config must resolve to a mapping")
    return cast(dict[str, Any], value)


def _run_cm7(cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict[str, Any]:
    receipt = run_workbench(_plain(cfg.experiment), run_dir=run_dir, device=device)
    aggregate = receipt.get("aggregate", {})
    return {
        "nuisance_invariance_per_objective": {
            objective: row["referent_view_stability"]["mean"]
            for objective, row in aggregate.items()
            if "referent_view_stability" in row
        },
        "held_out_combo_per_objective": {
            objective: row["heldout_combo_score"]["mean"]
            for objective, row in aggregate.items()
            if "heldout_combo_score" in row
        },
        "trainable_parameters": receipt["model"]["trainable_parameters"],
        "estimated_flops_by_objective": receipt["compute_match"].get("mean_flops", {}),
        "compute_matched": receipt["compute_match"]["all_ok"],
        "requirements_all_ok": receipt["requirements"]["all_ok"],
        "complete": receipt["complete"],
        "null_supported": not receipt["promotion"]["cm7_local_objective_lever_promotable"],
        "scientific_scope": "programmatic-video local objective probe",
        "receipt": "workbench_receipt.json",
    }


def _verify_cm7(result: Mapping[str, Any], _record: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "verified": bool(result.get("compute_matched") and result.get("requirements_all_ok")),
        "scope": "local dispatch and receipt-chain prerequisites",
        "independent_scientific_confirmation": False,
    }


def _run_cm8(cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict[str, Any]:
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


def _verify_cm8(result: Mapping[str, Any], _record: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "verified": result.get("scientific_promotion") is False,
        "scope": "fail-closed preflight",
        "independent_scientific_confirmation": False,
    }


BINDINGS: dict[str, tuple[Executor, Verifier]] = {
    "mop_cm7_min_objective_probe": (_run_cm7, _verify_cm7),
    "mop_cm8_custom_jepa_pilot": (_run_cm8, _verify_cm8),
}

__all__ = ["BINDINGS"]
