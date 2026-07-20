#!/usr/bin/env python
"""Stage, preflight, acquire, or probe the pinned official V-JEPA 2.1 ViT-B encoder.

The default command is metadata-only preflight. No command downloads weights unless ``download`` is
spelled explicitly, and no command constructs a model unless ``probe`` is spelled explicitly.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

from mop.substrate.vjepa21_official import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DOCTOR_RECEIPT,
    DEFAULT_PREFLIGHT_PROOF,
    DEFAULT_REPOSITORY_DIR,
    VITB,
    VJEPA21IntegrationError,
    build_preflight,
    checkpoint_receipt_path,
    download_vitb_checkpoint,
    expected_dense_tokens,
    load_vitb_encoder,
    sha256_file,
    stage_repository,
    validate_checkpoint_receipt,
    validate_repository,
    write_preflight,
)

PROBE_SCHEMA = "mop-vjepa21-official-probe/v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    preflight = sub.add_parser("preflight", help="HEAD/range/source/dependency/disk checks only")
    preflight.add_argument("--repo", type=Path, default=DEFAULT_REPOSITORY_DIR)
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument("--proof", type=Path, default=DEFAULT_PREFLIGHT_PROOF)
    preflight.add_argument("--doctor-receipt", type=Path, default=DEFAULT_DOCTOR_RECEIPT)
    preflight.add_argument("--timeout", type=float, default=30.0)
    preflight.add_argument("--skip-ranges", action="store_true")

    stage = sub.add_parser("stage-repo", help="clone only the pinned official source, never weights")
    stage.add_argument("--repo", type=Path, default=DEFAULT_REPOSITORY_DIR)

    download = sub.add_parser("download", help="explicit 1.664 GB ViT-B checkpoint acquisition")
    download.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    download.add_argument("--doctor-receipt", type=Path, default=DEFAULT_DOCTOR_RECEIPT)
    download.add_argument("--timeout", type=float, default=90.0)

    probe = sub.add_parser("probe", help="supervised strict load or one real tensor forward")
    probe.add_argument("--mode", choices=("load", "forward"), required=True)
    probe.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    probe.add_argument("--frames", type=int, default=8)
    probe.add_argument("--repo", type=Path, default=DEFAULT_REPOSITORY_DIR)
    probe.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    probe.add_argument("--timeout", type=float, default=1800.0)
    probe.add_argument("--proof", type=Path, default=None)

    child = sub.add_parser("_probe-child", help=argparse.SUPPRESS)
    child.add_argument("--mode", choices=("load", "forward"), required=True)
    child.add_argument("--device", choices=("cpu", "mps"), required=True)
    child.add_argument("--frames", type=int, required=True)
    child.add_argument("--repo", type=Path, required=True)
    child.add_argument("--checkpoint", type=Path, required=True)
    child.add_argument("--result", type=Path, required=True)
    return parser


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _probe_child(args: argparse.Namespace) -> int:
    import torch

    if args.frames <= 0 or args.frames % 2:
        raise VJEPA21IntegrationError("probe frames must be a positive even number")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise VJEPA21IntegrationError("MPS requested but unavailable")
    started = time.perf_counter()
    encoder = load_vitb_encoder(args.repo, args.checkpoint)
    loaded_at = time.perf_counter()
    parameters = sum(parameter.numel() for parameter in encoder.parameters())
    result = {
        "completed": True,
        "mode": args.mode,
        "device": args.device,
        "repository_commit": validate_repository(args.repo)["commit"],
        "checkpoint_sha256": validate_checkpoint_receipt(args.checkpoint, rehash=False)["receipt"]["sha256"],
        "parameters": parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "model_class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "load_seconds": loaded_at - started,
        "checkpoint_key": VITB["checkpoint_key"],
        "strict_load": True,
    }
    if args.mode == "forward":
        encoder = encoder.to(args.device)
        resolution = int(VITB["resolution"])
        clip = torch.zeros(
            (1, 3, args.frames, resolution, resolution),
            dtype=torch.float32,
            device=args.device,
        )
        clip[:, 0, :, ::16, :] = 0.5
        clip[:, 1, :, :, ::16] = -0.5
        forward_started = time.perf_counter()
        with torch.inference_mode():
            output = encoder(clip)
        if args.device == "mps":
            torch.mps.synchronize()
        expected_shape = [1, expected_dense_tokens(args.frames), int(VITB["embed_dim"])]
        result.update(
            {
                "input_shape": list(clip.shape),
                "input_layout": "B,C,T,H,W",
                "output_shape": list(output.shape),
                "expected_output_shape": expected_shape,
                "shape_matches": list(output.shape) == expected_shape,
                "output_finite": bool(torch.isfinite(output).all()),
                "output_mean": float(output.float().mean()),
                "output_std": float(output.float().std()),
                "forward_seconds": time.perf_counter() - forward_started,
                "mps_current_allocated_bytes": (
                    int(torch.mps.current_allocated_memory()) if args.device == "mps" else None
                ),
                "mps_driver_allocated_bytes": (
                    int(torch.mps.driver_allocated_memory()) if args.device == "mps" else None
                ),
            }
        )
        result["completed"] = bool(result["shape_matches"] and result["output_finite"])
    result["total_seconds"] = time.perf_counter() - started
    result["child_max_rss_bytes"] = _max_rss_bytes()
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["completed"] else 1


def _tree_rss(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        members = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    total = 0
    for member in members:
        try:
            total += int(member.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _probe(args: argparse.Namespace) -> dict:
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.frames <= 0 or args.frames % 2:
        raise ValueError("--frames must be a positive even number")
    repo = validate_repository(args.repo)
    checkpoint = validate_checkpoint_receipt(args.checkpoint, rehash=True)
    if not repo["all_ok"] or not checkpoint["all_ok"]:
        raise VJEPA21IntegrationError("probe source/checkpoint authority validation failed")
    started = time.perf_counter()
    max_rss = 0
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="mop-vjepa21-probe-") as temporary:
        child_result = Path(temporary) / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_probe-child",
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--frames",
            str(args.frames),
            "--repo",
            str(args.repo.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--result",
            str(child_result),
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = started + args.timeout
        while process.poll() is None:
            max_rss = max(max_rss, _tree_rss(process.pid))
            if time.perf_counter() >= deadline:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        payload = None
        if child_result.is_file():
            try:
                loaded = json.loads(child_result.read_text())
                payload = loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                payload = None
        return_code = int(process.returncode if process.returncode is not None else -999)
    status = "passed" if return_code == 0 and payload and payload.get("completed") else "failed"
    if timed_out:
        status = "timed-out"
    stderr_tail = stderr[-4000:]
    explicit_oom = any(
        marker in stderr_tail.lower()
        for marker in ("out of memory", "cannot allocate memory", "invalid buffer size")
    )
    return {
        "schema": PROBE_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "return_code": return_code,
        "timed_out": timed_out,
        "explicit_out_of_memory": explicit_oom,
        "hardware_limit_reached": explicit_oom,
        "wall_seconds": time.perf_counter() - started,
        "max_process_tree_rss_bytes": max_rss,
        "probe": {
            "mode": args.mode,
            "device": args.device,
            "frames": args.frames,
            "resolution": VITB["resolution"],
            "timeout_seconds": args.timeout,
            "command": command,
        },
        "authority": {
            "repository_commit": repo["commit"],
            "repository_validation_ok": repo["all_ok"],
            "checkpoint_receipt_path": str(checkpoint_receipt_path(args.checkpoint).resolve()),
            "checkpoint_receipt_sha256": sha256_file(checkpoint_receipt_path(args.checkpoint)),
            "checkpoint_sha256": checkpoint["receipt"]["sha256"],
        },
        "child": payload,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr_tail,
        "claim_boundary": {
            "e6_scientific_compatibility_proven": False,
            "dr14_scientific_compatibility_proven": False,
            "vitb_runtime_evidence_gate_passed": bool(status == "passed" and args.mode == "forward"),
            "model_scope": "official dense ViT-B only",
            "interpretation": (
                "a passing forward retires ViT-B runtime availability only; task validity, natural-video "
                "referents, matched controls, and cache manifests remain separate gates"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command or "preflight"
    try:
        if command == "stage-repo":
            receipt = stage_repository(args.repo)
        elif command == "download":
            receipt = download_vitb_checkpoint(
                args.checkpoint,
                timeout=args.timeout,
                doctor_receipt=args.doctor_receipt,
            )
        elif command == "probe":
            receipt = _probe(args)
            proof = args.proof or Path(
                "proof/VJEPA21_VITB_LOAD.json" if args.mode == "load" else "proof/VJEPA21_VITB_FORWARD.json"
            )
            write_preflight(proof, receipt)
        elif command == "_probe-child":
            return _probe_child(args)
        else:
            if args.command is None:
                args = _parser().parse_args(["preflight"])
            receipt = build_preflight(
                repository=args.repo,
                config=args.config,
                doctor_receipt=args.doctor_receipt,
                timeout=args.timeout,
                verify_ranges=not args.skip_ranges,
            )
            write_preflight(args.proof, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt.get("all_ok", receipt.get("status") == "passed") else 1
    except (VJEPA21IntegrationError, OSError, ValueError, json.JSONDecodeError) as exc:
        failure = {
            "schema": "mop-vjepa21-official-failure/v1",
            "created_at": datetime.now(UTC).isoformat(),
            "command": command,
            "all_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_loaded": False,
            "scientific_promotion": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
