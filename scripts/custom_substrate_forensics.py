#!/usr/bin/env python
"""Replay one exact custom-substrate checkpoint step without applying an optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import torch

from mop.devices import resolve
from mop.substrate.custom_workbench import (
    CorpusSpec,
    ModelSpec,
    ProgrammaticVideoCorpus,
    TinyVideoSubstrate,
    _batch_for_step,
    _mask_for_step,
    _objective_target,
    build_referent_records,
    sha256_file,
    token_count,
)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def replay(run_dir: Path, *, seed: int, objective: str, device_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads((run_dir / "resolved_config.json").read_text())
    data_spec = CorpusSpec(**config["data"])
    model_spec = ModelSpec(**config["model"])
    training = config["training"]
    checkpoint_path = run_dir / "arms" / f"seed_{seed}" / objective / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    step = int(checkpoint["step"])
    device = resolve(device_name)
    records = build_referent_records(data_spec)
    corpus = ProgrammaticVideoCorpus(data_spec, records)
    model = TinyVideoSubstrate(model_spec)
    target = TinyVideoSubstrate(model_spec)
    model.load_state_dict(checkpoint["model"])
    target.load_state_dict(checkpoint["target"])
    target.requires_grad_(False)
    model.to(device.device)
    target.to(device.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    batch_size = int(training["batch_size"])
    train_indices = [row.index for row in records if row.split == "train"]
    batch_indices = _batch_for_step(train_indices, batch_size, seed, step)
    batch_records = [records[index] for index in batch_indices]
    view_a_cpu = corpus.batch(batch_indices, view=step * 2).permute(0, 2, 1, 3, 4)
    view_b_cpu = corpus.batch(batch_indices, view=step * 2 + 1).permute(0, 2, 1, 3, 4)
    mask_cpu = _mask_for_step(
        batch_size,
        token_count(data_spec, model_spec),
        float(training["mask_ratio"]),
        seed,
        step,
    )
    view_a = view_a_cpu.to(device.device)
    view_b = view_b_cpu.to(device.device)
    mask = mask_cpu.to(device.device)
    error = None
    loss_value = None
    prediction_finite = target_finite = gradients_finite = False
    try:
        optimizer.zero_grad(set_to_none=True)
        online_tokens = model.encode(view_a, mask)
        with torch.no_grad():
            target_tokens = target.encode(view_b)
        prediction = model.predictor(online_tokens)
        target_value = _objective_target(
            objective,
            model=model,
            target_tokens=target_tokens,
            online_clips=view_a,
            records=batch_records,
            teacher_targets=None,
            step=step,
        )
        prediction = torch.nn.functional.normalize(prediction, dim=-1)
        target_value = torch.nn.functional.normalize(target_value.detach(), dim=-1)
        prediction_loss = (1.0 - (prediction * target_value).sum(dim=-1))[mask].mean()
        feature_std = online_tokens.mean(dim=1).std(dim=0, unbiased=False)
        variance_loss = torch.nn.functional.relu(0.10 - feature_std).mean()
        loss = prediction_loss + float(training["variance_weight"]) * variance_loss
        loss_value = float(loss.detach().cpu())
        prediction_finite = bool(torch.isfinite(prediction).all())
        target_finite = bool(torch.isfinite(target_value).all())
        loss.backward()
        if device.kind == "mps":
            torch.mps.synchronize()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        gradients_finite = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
    except Exception as exc:  # forensic receipt records the exact backend failure
        error = f"{type(exc).__name__}: {exc}"
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if __import__("platform").system() != "Darwin":
        max_rss *= 1024
    return {
        "schema": "mop-custom-substrate-step-forensic/v1",
        "claim_scope": "single exact no-update replay; diagnostic only",
        "run_dir": str(run_dir),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": step,
            "objective": checkpoint["objective"],
            "seed": seed,
        },
        "replayed_step": step,
        "device": device.kind,
        "batch_indices": batch_indices,
        "batch_referents": [row.referent for row in batch_records],
        "input_sha256": _tensor_sha256(view_a_cpu),
        "target_view_sha256": _tensor_sha256(view_b_cpu),
        "mask_sha256": _tensor_sha256(mask_cpu),
        "pre_step_finiteness": {
            "model": _finite_tree(checkpoint["model"]),
            "target": _finite_tree(checkpoint["target"]),
            "optimizer": _finite_tree(checkpoint["optimizer"]),
            "input": bool(torch.isfinite(view_a_cpu).all()),
            "target_view": bool(torch.isfinite(view_b_cpu).all()),
        },
        "step_finiteness": {
            "prediction": prediction_finite,
            "target": target_finite,
            "loss": loss_value is not None and math.isfinite(loss_value),
            "gradients": gradients_finite,
        },
        "loss": loss_value,
        "error": error,
        "all_ok": error is None
        and loss_value is not None
        and math.isfinite(loss_value)
        and prediction_finite
        and target_finite
        and gradients_finite,
        "seconds": time.perf_counter() - started,
        "max_rss_bytes": max_rss,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", default="random_target")
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = replay(
        args.run_dir,
        seed=args.seed,
        objective=args.objective,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
