#!/usr/bin/env python
"""Preflight, export, and offline-verify a portable custom-substrate artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mop.substrate.custom_artifact import (
    ArtifactRefused,
    export_artifact,
    load_portable_artifact,
    preflight_export,
    verifier_contract,
)


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _contract(_args: argparse.Namespace) -> int:
    _print(verifier_contract())
    return 0


def _preflight(args: argparse.Namespace) -> int:
    result = preflight_export(args.run_dir, args.verifier)
    _print(result)
    return 0 if result["eligible"] else 3


def _export(args: argparse.Namespace) -> int:
    _print(export_artifact(args.run_dir, args.verifier, args.output_root))
    return 0


def _verify(args: argparse.Namespace) -> int:
    loaded = load_portable_artifact(args.artifact_dir, device=args.device)
    report: dict = {
        "all_ok": True,
        "artifact_id": loaded.manifest["artifact_id"],
        "architecture": loaded.manifest["model"]["architecture"],
        "model_spec": loaded.manifest["model"]["spec"],
        "state_sha256": loaded.manifest["model"]["state_sha256"],
        "evidence_scope": loaded.manifest["evidence"]["scope"],
        "smoke_tested": False,
    }
    if args.smoke:
        spec = loaded.model.spec
        clips = torch.zeros(
            args.batch_size,
            3,
            args.frames or spec.max_frames,
            args.resolution or spec.max_resolution,
            args.resolution or spec.max_resolution,
            device=args.device,
        )
        with torch.inference_mode():
            output = loaded.model(clips)
        report.update(
            {
                "smoke_tested": True,
                "dense_shape": list(output.dense_spatiotemporal_tokens.shape),
                "pooled_shape": list(output.pooled_retrieval_key.shape),
                "finite": bool(
                    torch.isfinite(output.dense_spatiotemporal_tokens).all()
                    and torch.isfinite(output.pooled_retrieval_key).all()
                ),
            }
        )
    _print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract = subparsers.add_parser("contract", help="print the independent-verifier receipt contract")
    contract.set_defaults(func=_contract)

    preflight = subparsers.add_parser("preflight", help="audit a completed run without writing an artifact")
    preflight.add_argument("--run-dir", type=Path, required=True)
    preflight.add_argument("--verifier", type=Path, required=True)
    preflight.set_defaults(func=_preflight)

    export = subparsers.add_parser("export", help="create the deterministic content-addressed artifact")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--verifier", type=Path, required=True)
    export.add_argument("--output-root", type=Path, default=Path("artifacts/custom_substrate"))
    export.set_defaults(func=_export)

    verify = subparsers.add_parser("verify", help="offline-verify and load an exported artifact")
    verify.add_argument("--artifact-dir", type=Path, required=True)
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--smoke", action="store_true")
    verify.add_argument("--batch-size", type=int, default=1)
    verify.add_argument("--frames", type=int)
    verify.add_argument("--resolution", type=int)
    verify.set_defaults(func=_verify)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ArtifactRefused as exc:
        _print({"all_ok": False, "refused": True, "problem": str(exc)})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
