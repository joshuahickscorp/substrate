#!/usr/bin/env python
"""Run or resume the local custom-substrate objective workbench.

Profiles are wall envelopes, not claim levels. ``local180`` expands the exact same CM7 contract to
five seeds, 256px inputs, and at most 180 minutes per invocation. A stopped run is resumable from
content-addressed per-arm checkpoints. CM8 is a read-only fail-closed preflight and never trains.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from mop.config import REPO_ROOT, compose
from mop.devices import resolve
from mop.substrate.custom_workbench import (
    attest_current_requirements,
    cm8_preflight,
    run_workbench,
)

PROFILES: dict[str, dict] = {
    "harness": {},
    "local30": {
        "data": {"replicates": 4},
        "training": {
            "seeds": [0, 1, 2],
            "steps": 12,
            "batch_size": 4,
            "eval_batch_size": 8,
            "checkpoint_every": 3,
            "wall_budget_seconds": 1800.0,
        },
    },
    "local180": {
        "data": {"replicates": 8},
        "training": {
            "seeds": [0, 1, 2, 3, 4],
            "steps": 1000,
            "batch_size": 4,
            "eval_batch_size": 8,
            "checkpoint_every": 25,
            "wall_budget_seconds": 10800.0,
        },
    },
}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _config(profile: str, overrides: list[str]) -> dict:
    cfg = compose(["experiment=mop_cm7_min_objective_probe", *overrides])
    experiment = OmegaConf.merge(cfg.experiment, OmegaConf.create(PROFILES[profile]))
    value = OmegaConf.to_container(experiment, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved workbench config is not a mapping")
    return value


def run_cm7(args: argparse.Namespace) -> int:
    config = _config(args.profile, args.override)
    run_dir = args.run_dir or REPO_ROOT / "runs/custom_substrate" / f"cm7_{args.profile}"
    receipt = run_workbench(config, run_dir=run_dir, device=resolve(args.device))
    receipt = attest_current_requirements(run_dir)
    proof = args.proof or REPO_ROOT / "proof/CUSTOM_SUBSTRATE_PILOT.json"
    _atomic_json(proof, receipt)
    print(
        json.dumps(
            {
                "complete": receipt["complete"],
                "resumable": receipt["resumable"],
                "wall_seconds": receipt["resource_telemetry"]["wall_seconds_this_invocation"],
                "parameters": receipt["model"]["trainable_parameters"],
                "best_objective": receipt["promotion"]["best_objective"],
                "cm7_promotable": receipt["promotion"]["cm7_local_objective_lever_promotable"],
                "proof": str(proof),
            },
            indent=2,
        )
    )
    return 0 if receipt["complete"] else 2


def run_cm8(args: argparse.Namespace) -> int:
    cfg = compose(["experiment=mop_cm8_custom_jepa_pilot"])
    value = OmegaConf.to_container(cfg.experiment, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved CM8 config is not a mapping")
    receipt = cm8_preflight(value)
    proof = args.proof or REPO_ROOT / "proof/CUSTOM_SUBSTRATE_CM8_PREFLIGHT.json"
    _atomic_json(proof, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def run_attest(args: argparse.Namespace) -> int:
    receipt = attest_current_requirements(args.run_dir)
    proof = args.proof or REPO_ROOT / "proof/CUSTOM_SUBSTRATE_PILOT.json"
    _atomic_json(proof, receipt)
    print(json.dumps(receipt["evidence_attestation"], indent=2))
    return 0 if receipt["evidence_attestation"]["scientifically_current"] else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cm7 = subparsers.add_parser("cm7", help="run or resume the matched local objective tournament")
    cm7.add_argument("--profile", choices=sorted(PROFILES), default="local180")
    cm7.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    cm7.add_argument("--run-dir", type=Path)
    cm7.add_argument("--proof", type=Path)
    cm7.add_argument(
        "--override",
        action="append",
        default=[],
        help="additional compose override, repeatable (for example experiment.training.steps=10)",
    )
    cm7.set_defaults(func=run_cm7)

    cm8 = subparsers.add_parser("cm8-preflight", help="audit CM8 upstream evidence without training")
    cm8.add_argument("--proof", type=Path)
    cm8.set_defaults(func=run_cm8)

    attest = subparsers.add_parser(
        "attest", help="rebind a completed run to current proof sources without retraining"
    )
    attest.add_argument("--run-dir", type=Path, required=True)
    attest.add_argument("--proof", type=Path)
    attest.set_defaults(func=run_attest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
