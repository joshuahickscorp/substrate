from __future__ import annotations

import json
import sys

CM7_PROFILES: dict[str, dict] = {
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
OPERATIONS = {"cm7", "cm8-preflight", "doctor", "profiles"}


def _workbench_config(profile: str, overrides: list[str]) -> dict:
    from omegaconf import OmegaConf

    from ..config import compose

    cfg = compose(["experiment=mop_cm7_min_objective_probe", *overrides])
    value = OmegaConf.to_container(
        OmegaConf.merge(cfg.experiment, OmegaConf.create(CM7_PROFILES[profile])), resolve=True
    )
    if not isinstance(value, dict):
        raise TypeError("resolved workbench config is not a mapping")
    return value


def _cm7(args) -> int:
    from ..config import REPO_ROOT
    from ..devices import resolve
    from ..evidence import atomic_write_json
    from ..substrate.custom_workbench import run_workbench

    run_dir = args.run_dir or REPO_ROOT / "runs/custom_substrate" / f"cm7_{args.profile}"
    receipt = run_workbench(
        _workbench_config(args.profile, args.override),
        run_dir=run_dir,
        device=resolve(args.device),
    )
    proof = args.proof or REPO_ROOT / "proof/CUSTOM_SUBSTRATE_PILOT.json"
    atomic_write_json(proof, receipt)
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


def _cm8(args) -> int:
    from typing import Any, cast

    from omegaconf import OmegaConf

    from ..config import REPO_ROOT, compose
    from ..evidence import atomic_write_json
    from ..substrate.custom_workbench import cm8_preflight

    value = OmegaConf.to_container(
        compose(["experiment=mop_cm8_custom_jepa_pilot"]).experiment,
        resolve=True,
    )
    if not isinstance(value, dict):
        raise TypeError("resolved CM8 config is not a mapping")
    receipt = cm8_preflight(cast(dict[str, Any], value))
    proof = args.proof or REPO_ROOT / "proof/CUSTOM_SUBSTRATE_CM8_PREFLIGHT.json"
    atomic_write_json(proof, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def _doctor(args) -> int:
    from ..config import REPO_ROOT
    from ..evidence import atomic_write_json
    from ..studio_doctor import doctor, render_md

    report = doctor(args.profile)
    markdown = REPO_ROOT / "runs/studio_doctor.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_md(report))
    if args.out:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        atomic_write_json(out, report)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_ok"] else 1


def _profiles(_args) -> int:
    from ..studio.profiles import list_profiles

    print(json.dumps(list_profiles(), indent=2))
    return 0


def _operation(argv: list[str]) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="mop",
        description="MOP operations",
        epilog="Other arguments run an experiment as configuration overrides.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cm7 = subparsers.add_parser("cm7", help="run or resume the matched local objective tournament")
    cm7.add_argument("--profile", choices=sorted(CM7_PROFILES), default="local180")
    cm7.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    cm7.add_argument("--run-dir", type=Path)
    cm7.add_argument("--proof", type=Path)
    cm7.add_argument(
        "--override",
        action="append",
        default=[],
        help="additional configuration override, repeatable",
    )
    cm7.set_defaults(func=_cm7)
    cm8 = subparsers.add_parser("cm8-preflight", help="audit CM8 upstream evidence without training")
    cm8.add_argument("--proof", type=Path)
    cm8.set_defaults(func=_cm8)
    doctor = subparsers.add_parser("doctor", help="run host readiness checks")
    doctor.add_argument("--profile")
    doctor.add_argument("--out", type=Path)
    doctor.set_defaults(func=_doctor)
    profiles = subparsers.add_parser("profiles", help="list resource profiles")
    profiles.set_defaults(func=_profiles)
    args = parser.parse_args(argv)
    return int(args.func(args))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in OPERATIONS | {"-h", "--help"}:
        return _operation(argv)
    from ..config import compose
    from .runner import run_experiment

    cfg = compose(argv)
    metrics = run_experiment(cfg)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
