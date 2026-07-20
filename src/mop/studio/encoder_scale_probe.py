"""Supervised, fail-closed probes for real frozen video encoders.

The parent process survives a child timeout or OS kill and still writes a durable receipt.  This is
important for scale-boundary work: a missing JSON file must never be interpreted as an out-of-memory
result, and a model that merely exists in the Hugging Face cache must never be called executable.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from omegaconf import OmegaConf

from ..substrate.encoder import load_encoder
from mop.substrate.events import sha256_file

SCHEMA = "mop-real-encoder-scale-probe/v1"
MODES = {"load", "forward"}
DEVICES = {"cpu", "mps"}


def _rss_tree_bytes(pid: int) -> int:
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


def _self_max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _child(config_path: Path, mode: str, device: str, result_path: Path) -> int:
    started = time.perf_counter()
    cfg = OmegaConf.load(config_path)
    cfg.random_init = False
    cfg.prefer_real = True
    cfg.require_real = True
    cfg.local_files_only = True
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    encoder = load_encoder(cfg).to(device)
    loaded_at = time.perf_counter()
    result: dict[str, Any] = {
        "backend": encoder.spec.backend,
        "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "model_class": encoder.model_class_name,
        "load_seconds": loaded_at - started,
    }
    if mode == "forward":
        frames = int(cfg.frames_per_clip)
        resolution = int(cfg.resolution)
        clip = torch.zeros((1, frames, 3, resolution, resolution), dtype=torch.float32, device=device)
        clip[:, :, 0, ::16, :] = 0.5
        clip[:, :, 1, :, ::16] = -0.5
        forward_started = time.perf_counter()
        with torch.inference_mode():
            output = encoder.encode(clip)
        if device == "mps":
            torch.mps.synchronize()
        result.update(
            {
                "forward_seconds": time.perf_counter() - forward_started,
                "input_shape": list(clip.shape),
                "input_layout": "B,T,C,H,W",
                "output_shape": list(output.shape),
                "output_finite": bool(torch.isfinite(output).all()),
                "output_mean": float(output.float().mean()),
                "output_std": float(output.float().std()),
                "mps_current_allocated_bytes": (
                    int(torch.mps.current_allocated_memory()) if device == "mps" else None
                ),
                "mps_driver_allocated_bytes": (
                    int(torch.mps.driver_allocated_memory()) if device == "mps" else None
                ),
            }
        )
    result.update(
        {
            "completed": True,
            "total_seconds": time.perf_counter() - started,
            "child_max_rss_bytes": _self_max_rss_bytes(),
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return 0


def run_supervised_probe(
    config_path: Path | str,
    *,
    mode: str,
    device: str = "cpu",
    timeout_seconds: float,
    poll_seconds: float = 0.1,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
    if device not in DEVICES:
        raise ValueError(f"device must be one of {sorted(DEVICES)}, got {device!r}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    cfg = OmegaConf.load(path)
    before = psutil.virtual_memory()
    command: list[str]
    max_rss = 0
    min_system_available = int(before.available)
    timed_out = False
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mop-encoder-probe-") as tmp:
        child_result = Path(tmp) / "child.json"
        command = [
            sys.executable,
            "-m",
            "mop.studio.encoder_scale_probe",
            "--child",
            "--config",
            str(path),
            "--mode",
            mode,
            "--device",
            device,
            "--child-result",
            str(child_result),
        ]
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        deadline = started + timeout_seconds
        while process.poll() is None:
            max_rss = max(max_rss, _rss_tree_bytes(process.pid))
            min_system_available = min(min_system_available, int(psutil.virtual_memory().available))
            if time.perf_counter() >= deadline:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(poll_seconds)
        stdout, stderr = process.communicate()
        max_rss = max(max_rss, _rss_tree_bytes(process.pid))
        child_payload: dict[str, Any] | None = None
        if child_result.exists():
            try:
                loaded = json.loads(child_result.read_text())
                child_payload = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                child_payload = None
        return_code = int(process.returncode if process.returncode is not None else -999)

    status = "passed" if return_code == 0 and child_payload and child_payload.get("completed") else "failed"
    if timed_out:
        status = "timed-out"
    stderr_tail = stderr[-4000:]
    lowered = stderr_tail.lower()
    explicit_oom = any(
        marker in lowered for marker in ("out of memory", "cannot allocate memory", "invalid buffer size")
    )
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    child_max_rss = int((child_payload or {}).get("child_max_rss_bytes", 0))
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "path": str(path),
            "sha256": sha256_file(path),
            "name": str(cfg.name),
            "model_id": str(cfg.hf_id),
            "revision": str(OmegaConf.select(cfg, "revision", default="")),
            "resolved": cfg_container,
        },
        "probe": {
            "mode": mode,
            "device": device,
            "timeout_seconds": timeout_seconds,
            "command": command,
            "offline_weights_required": True,
        },
        "status": status,
        "return_code": return_code,
        "timed_out": timed_out,
        "explicit_out_of_memory": explicit_oom,
        "hardware_limit_reached": explicit_oom,
        "wall_seconds": time.perf_counter() - started,
        "max_process_tree_rss_bytes": max_rss,
        "peak_rss_bytes": max(max_rss, child_max_rss),
        "system_memory": {
            "total_bytes": int(before.total),
            "available_before_bytes": int(before.available),
            "minimum_available_bytes": min_system_available,
        },
        "child": child_payload,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr_tail,
        "interpretation": (
            "A timeout is a bounded-attempt result, not by itself an out-of-memory claim. "
            "Only an explicit allocator failure sets hardware_limit_reached. This execution receipt "
            "pins model ID, revision, and config hash but does not independently hash the loaded shard; "
            "pair it with the cache's immutable encoder-weight receipt for content identity."
        ),
    }


def validate_probe_receipt(receipt: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema") != SCHEMA:
        problems.append("wrong schema")
    config = receipt.get("config")
    if (
        not isinstance(config, dict)
        or len(str(config.get("sha256") or "")) != 64
        or not config.get("revision")
        or not config.get("model_id")
    ):
        problems.append("config hash and pinned revision are required")
    probe = receipt.get("probe")
    if not isinstance(probe, dict) or probe.get("mode") not in MODES:
        problems.append("valid probe mode is required")
    if not isinstance(probe, dict) or probe.get("device") not in DEVICES:
        problems.append("valid probe device is required")
    if receipt.get("status") == "passed":
        child = receipt.get("child")
        if not isinstance(child, dict) or child.get("backend") != "vjepa_hf":
            problems.append("a passing receipt requires the real vjepa_hf backend")
        if not child or _receipt_int(child.get("parameters"), default=0) <= 0:
            problems.append("a passing receipt requires a positive model parameter count")
        if not child or _receipt_int(child.get("trainable_parameters"), default=-1) != 0:
            problems.append("a passing receipt requires a fully frozen model")
        if not child or not str(child.get("model_class") or "").strip():
            problems.append("a passing receipt requires the realized model class")
        if (
            probe
            and probe.get("mode") == "forward"
            and (not child or child.get("output_finite") is not True or not child.get("output_shape"))
        ):
            problems.append("a passing forward requires a finite, shaped output")
        if probe and probe.get("mode") == "forward" and child:
            input_shape = child.get("input_shape")
            if (
                child.get("input_layout") != "B,T,C,H,W"
                or not isinstance(input_shape, list)
                or len(input_shape) != 5
                or input_shape[0] != 1
                or input_shape[2] != 3
            ):
                problems.append("a passing forward requires the recorded B,T,C,H,W input contract")
        if receipt.get("hardware_limit_reached"):
            problems.append("a passing receipt cannot also claim a hardware limit")
    if receipt.get("hardware_limit_reached") and not receipt.get("explicit_out_of_memory"):
        problems.append("hardware limit cannot be inferred without an explicit allocator failure")
    return problems


def _receipt_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=sorted(MODES), default="forward")
    parser.add_argument("--device", choices=sorted(DEVICES), default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--out")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        if not args.child_result:
            parser.error("--child-result is required in child mode")
        return _child(Path(args.config), args.mode, args.device, Path(args.child_result))
    receipt = run_supervised_probe(
        args.config,
        mode=args.mode,
        device=args.device,
        timeout_seconds=args.timeout_seconds,
    )
    problems = validate_probe_receipt(receipt)
    receipt["validation"] = {"all_ok": not problems, "problems": problems}
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered)
    print(rendered, end="")
    return 0 if receipt["status"] == "passed" and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
