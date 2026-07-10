#!/usr/bin/env python
"""Preflight or explicitly run official ViT-B dense cache/control tasks.

The default command is metadata-only preflight. Only the explicitly spelled ``encode`` command
constructs a model or reads checkpoint tensor bytes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from mop.substrate.vjepa21_dense_tasks import (
    DEFAULT_TASK_CONFIG,
    DenseTaskError,
    build_cache_plan,
    build_input_manifest,
    encode_dense_cache,
    no_heavy_preflight,
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _task_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema") != "mop-vjepa21-dense-task-config/v1":
        raise DenseTaskError(f"{path} is not a dense task config")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    preflight = sub.add_parser("preflight", help="no model, no forward, no checkpoint tensor read")
    preflight.add_argument("--config", type=Path, default=DEFAULT_TASK_CONFIG)
    preflight.add_argument("--input-manifest", type=Path)
    preflight.add_argument("--proof", type=Path, default=Path("proof/E6_VITB_DENSE_PREFLIGHT.json"))
    plan = sub.add_parser("plan", help="freeze a two-arm cache plan from an existing input manifest")
    plan.add_argument("--config", type=Path, default=DEFAULT_TASK_CONFIG)
    plan.add_argument("--input-manifest", type=Path)
    plan.add_argument("--learned-cache", type=Path)
    plan.add_argument("--random-cache", type=Path)
    plan.add_argument("--random-seed", type=int)
    plan.add_argument("--dtype", choices=("float16", "float32"))
    plan.add_argument("--out", type=Path, required=True)
    inputs = sub.add_parser("build-input", help="hash preprocessed tensors from records/source JSON")
    inputs.add_argument("--records", type=Path, required=True)
    inputs.add_argument("--source", type=Path, required=True)
    inputs.add_argument("--out", type=Path, required=True)
    encode = sub.add_parser("encode", help="HEAVY: construct one encoder and encode one declared arm")
    encode.add_argument("--config", type=Path, default=DEFAULT_TASK_CONFIG)
    encode.add_argument("--arm", choices=("learned", "random"), required=True)
    encode.add_argument("--input-manifest", type=Path)
    encode.add_argument("--cache", type=Path)
    encode.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    encode.add_argument("--dtype", choices=("float16", "float32"))
    encode.add_argument("--random-seed", type=int)
    encode.add_argument("--proof", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command or "preflight"
    try:
        if command == "preflight":
            receipt = no_heavy_preflight(
                task_config=getattr(args, "config", DEFAULT_TASK_CONFIG),
                input_manifest=getattr(args, "input_manifest", None),
            )
            proof = getattr(args, "proof", Path("proof/E6_VITB_DENSE_PREFLIGHT.json"))
            _atomic_json(proof, receipt)
            result_code = 0 if receipt["all_ok"] else 1
        elif command == "plan":
            config = _task_config(args.config)
            receipt = build_cache_plan(
                args.input_manifest or Path(config["input_manifest"]),
                learned_cache=args.learned_cache or Path(config["learned_cache"]),
                random_cache=args.random_cache or Path(config["random_cache"]),
                random_seed=(
                    args.random_seed if args.random_seed is not None else int(config["random_seed"])
                ),
                dtype=args.dtype or str(config["cache_dtype"]),
            )
            _atomic_json(args.out, receipt)
            result_code = 0
        elif command == "build-input":
            records = json.loads(args.records.read_text())
            source = json.loads(args.source.read_text())
            if not isinstance(records, list) or not isinstance(source, dict):
                raise DenseTaskError("records must be a list and source must be a mapping")
            receipt = build_input_manifest(records, source, output=args.out)
            result_code = 0
        elif command == "encode":
            config = _task_config(args.config)
            receipt = encode_dense_cache(
                args.input_manifest or Path(config["input_manifest"]),
                args.cache or Path(config[f"{args.arm}_cache"]),
                arm=args.arm,
                device=args.device,
                dtype=args.dtype or str(config["cache_dtype"]),
                random_seed=(
                    args.random_seed if args.random_seed is not None else int(config["random_seed"])
                ),
            )
            if args.proof:
                _atomic_json(args.proof, receipt)
            result_code = 0 if receipt["all_ok"] else 1
        else:  # pragma: no cover
            raise AssertionError(command)
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return result_code
    except (DenseTaskError, OSError, ValueError, json.JSONDecodeError) as exc:
        failure = {
            "schema": "mop-vjepa21-dense-task-failure/v1",
            "all_ok": False,
            "scientific_promotion": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
