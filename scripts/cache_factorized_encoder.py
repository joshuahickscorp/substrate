#!/usr/bin/env python
"""Cache a FACTORIZED real-encoder latent store: structured synthetic video in which TWO independent
visual factors vary (color hue = factor A, motion/orientation = factor B), run through the REAL frozen
V-JEPA 2 encoder. The content is synthetic but the perceptual geometry is the real encoder's. Unlike the
single-factor cache_real_encoder.py (where one class index entangles freq/angle/motion/color together),
this gives two SEPARATELY-DECODABLE factors, which is what held-out-combination and compositionality
probes need: train on a subset of (A, B) pairs, test whether unseen (A, B) pairs decode above the
shuffled/frozen-random floor on real encoder features.

Storage: LatentStore has one label channel, so the two factors are encoded as a composite label
y = a*n_b + b, plus a factors.json sidecar recording n_a/n_b. mop.substrate.real_latent.factorized_arrays
reverses that into (x, y_a, y_b).

Usage: python scripts/cache_factorized_encoder.py [device=cpu] [+n_a=6] [+n_b=6] [+per=8] [+batch=1]
The encoder config supplies its native frame count and resolution. Use CPU unless a supervised MPS
probe for that exact encoder and shape has passed.

Matched architecture control: add `+random_init_control=true +encoder.random_init_seed=<seed>`.

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from mop.config import REPO_ROOT, compose
from mop.devices import resolve, safe_to
from mop.logging_utils import get_logger
from mop.substrate import LatentStore, load_encoder
from mop.substrate.cache_manifest import (
    ENCODER_RECEIPT_SCHEMA,
    RANDOM_INIT_RECEIPT_SCHEMA,
    write_cache_manifest,
)
from mop.substrate.encoder import module_state_sha256

log = get_logger("cache_factorized")
FRAMES, RES = 64, 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash shape, dtype, and exact CPU bytes for one generated encoder input."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(tensor.view(torch.uint8).reshape(-1).numpy()))
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _hue_tint(a: int, n_a: int) -> torch.Tensor:
    """Factor A: color hue, evenly spaced around the wheel. Independent of factor B."""
    h = a / max(1, n_a)
    return torch.tensor(
        [
            0.5 + 0.5 * math.cos(2 * math.pi * h),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 1 / 3)),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 2 / 3)),
        ]
    )


def make_factorized_clip(a: int, b: int, n_a: int, n_b: int, g: torch.Generator) -> torch.Tensor:
    """One [T,3,RES,RES] clip. Factor A (a) sets the COLOR HUE only; factor B (b) sets the grating
    ORIENTATION and DRIFT DIRECTION only. The spatial frequency is held fixed so the two factors are
    visually independent: any (hue, orientation) combination is realizable, which is what makes
    held-out-combination decoding a real test rather than an artifact of an entangled 1-D family."""
    lin = torch.linspace(0, 1, RES)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    theta = b * math.pi / max(1, n_b)  # factor B: orientation
    drift = 1.0 if (b % 2 == 0) else -1.0  # factor B: drift direction sign
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    freq = 5.0  # FIXED across both factors (not a hidden third factor)
    t = torch.arange(FRAMES).float() / FRAMES
    grating = torch.sin(2 * math.pi * (freq * proj[None] + drift * t[:, None, None]))
    grating = (grating + 1) / 2  # [T,RES,RES] in [0,1]
    tint = _hue_tint(a, n_a)  # factor A
    clip = grating[:, None, :, :] * tint[None, :, None, None]  # [T,3,RES,RES]
    clip = clip + 0.03 * torch.randn(clip.shape, generator=g)
    return clip.clamp(0, 1)


def main(argv: list[str] | None = None) -> int:
    global FRAMES, RES
    whole_t0 = time.perf_counter()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    cfg = compose(raw_args)
    random_control = bool(cfg.get("random_init_control", False))
    OmegaConf.update(cfg, "encoder.random_init", random_control, force_add=True)
    OmegaConf.update(cfg, "encoder.prefer_real", not random_control, force_add=True)
    OmegaConf.update(cfg, "encoder.require_real", not random_control, force_add=True)
    if random_control:
        # A matched random control only needs the pinned architecture config. It must never retrieve or
        # deserialize a pretrained shard as a side effect of cache construction.
        OmegaConf.update(cfg, "encoder.local_files_only", True, force_add=True)
    FRAMES = int(cfg.encoder.frames_per_clip)
    RES = int(cfg.encoder.resolution)
    n_a = int(cfg.get("n_a", 6))
    n_b = int(cfg.get("n_b", 6))
    per = int(cfg.get("per", 8))  # clips per (a, b) cell
    batch = int(cfg.get("batch", 1))
    if n_a < 2 or n_b < 2:
        raise ValueError("factorized caches require n_a >= 2 and n_b >= 2")
    if per < 1 or batch < 1:
        raise ValueError("per and batch must be positive")
    total = n_a * n_b * per
    name = str(cfg.get("cache_name", f"{cfg.encoder.name}_factorized"))
    target = REPO_ROOT / cfg.data_dir / "cache" / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing cache {target}; choose +cache_name=<new-name>")

    dev = resolve(str(cfg.device.kind))
    enc = load_encoder(cfg.encoder)
    expected_backend = "vjepa_hf_random_init" if random_control else "vjepa_hf"
    if enc.spec.backend != expected_backend:
        raise RuntimeError(
            f"requested backend {expected_backend}, got {enc.spec.backend}; refusing control laundering"
        )
    random_state_identity: dict[str, object] | None = None
    if random_control:
        random_state_identity = {
            "state_dict_sha256": module_state_sha256(enc),
            "state_dict_tensors": len(enc.state_dict()),
            "parameter_count": sum(parameter.numel() for parameter in enc.parameters()),
            "model_class": enc.model_class_name,
        }
    enc = enc.to(dev.device)

    g = torch.Generator().manual_seed(int(cfg.seed))
    store = LatentStore.create(
        REPO_ROOT / cfg.data_dir / "cache",
        name,
        feat_shape=(int(cfg.encoder.embed_dim),),
        capacity=total,
        key_dim=int(cfg.encoder.embed_dim),
        has_labels=True,
    )
    # composite label y = a*n_b + b, interleaved over cells then repeats
    cells = [(a, b) for a in range(n_a) for b in range(n_b)]
    order = [cells[i % len(cells)] for i in range(total)]
    labels = torch.tensor([a * n_b + b for (a, b) in order], dtype=torch.long)
    referents = [f"factorized:a{a}:b{b}:rep{index // len(cells)}" for index, (a, b) in enumerate(order)]
    stimulus_records: list[dict[str, object]] = []

    pos, t0 = 0, time.perf_counter()
    while pos < total:
        bs = min(batch, total - pos)
        clips = torch.stack(
            [make_factorized_clip(order[pos + j][0], order[pos + j][1], n_a, n_b, g) for j in range(bs)]
        )
        for j in range(bs):
            stimulus_records.append(
                {
                    "referent": referents[pos + j],
                    "shape": list(clips[j].shape),
                    "sha256": _tensor_sha256(clips[j]),
                }
            )
        z = enc.encode(safe_to(clips, dev.device)).reshape(bs, -1).float().cpu()
        store.write_batch(pos, z.numpy(), z.numpy(), labels[pos : pos + bs].numpy())
        pos += bs
        log.info("encoded %d/%d (%.1fs)", pos, total, time.perf_counter() - t0)
    store.finalize()
    encode_secs = time.perf_counter() - t0

    revision = str(cfg.encoder.get("revision", "")).strip()
    snapshot_patterns = (
        ["config.json", "video_preprocessor_config.json"]
        if random_control
        else ["config.json", "video_preprocessor_config.json", "model.safetensors"]
    )
    snapshot = Path(
        snapshot_download(
            str(cfg.encoder.hf_id),
            revision=revision or None,
            allow_patterns=snapshot_patterns,
            local_files_only=True,
        )
    )
    all_snapshot_files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if random_control:
        if random_state_identity is None:  # defensive, the branch above must construct it
            raise RuntimeError("random control is missing its realized-state identity")
        architecture_files = [
            path
            for path in all_snapshot_files
            if path.name in {"config.json", "video_preprocessor_config.json"}
        ]
        encoder_identity = {
            "schema": RANDOM_INIT_RECEIPT_SCHEMA,
            "weights_real": False,
            "model_id": str(cfg.encoder.hf_id),
            "revision": revision or snapshot.name,
            "backend": enc.spec.backend,
            "seed": int(cfg.encoder.random_init_seed),
            **random_state_identity,
            "architecture_files": [
                {
                    "path": str(path.relative_to(snapshot)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in architecture_files
            ],
        }
        (store.root / "initialization_receipt.json").write_text(
            json.dumps(encoder_identity, indent=2, sort_keys=True) + "\n"
        )
        encoder_receipt = None
        form_objective = "random-control"
    else:
        encoder_identity = {
            "schema": ENCODER_RECEIPT_SCHEMA,
            "weights_real": True,
            "model_id": str(cfg.encoder.hf_id),
            "revision": revision or snapshot.name,
            "backend": enc.spec.backend,
            "files": [
                {
                    "path": str(path.relative_to(snapshot)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in all_snapshot_files
            ],
        }
        encoder_receipt = encoder_identity
        form_objective = "inherited-frozen"
    split_generator = torch.Generator().manual_seed(int(cfg.seed) + 2203)
    split_order = torch.randperm(total, generator=split_generator).tolist()
    train_end = max(1, int(total * 0.6))
    val_end = max(train_end + 1, int(total * 0.8))
    raw_encoder_config = OmegaConf.to_container(cfg.encoder, resolve=True)
    if not isinstance(raw_encoder_config, dict):
        raise TypeError("encoder config did not resolve to a mapping")
    encoder_config: dict[str, Any] = {str(key): value for key, value in raw_encoder_config.items()}
    encoder_config.update(
        {
            "actual_backend": enc.spec.backend,
            "cache_script": "scripts/cache_factorized_encoder.py",
            "frames": FRAMES,
            "resolution": RES,
        }
    )
    write_cache_manifest(
        store.root,
        encoder_config=encoder_config,
        encoder_receipt=encoder_receipt,
        factors={
            "factor_a": [a for a, _ in order],
            "factor_b": [b for _, b in order],
            "composite_label": labels.tolist(),
        },
        factor_metadata={
            "n_a": n_a,
            "n_b": n_b,
            "factor_a_name": "hue",
            "factor_b_name": "orientation_drift",
            "seed": int(cfg.seed),
        },
        splits={
            "train": split_order[:train_end],
            "val": split_order[train_end:val_end],
            "test": split_order[val_end:],
        },
        referents=referents,
        form_kind="vision",
        form_objective=form_objective,
        referent_scheme="synthetic-factor-cell-repeat",
        full_hash_arrays=total <= 1024,
    )
    secs = encode_secs
    log.info(
        "cached %d factorized latents backend=%s in %.1fs (%.2fs/clip), n_a=%d n_b=%d",
        len(store),
        enc.spec.backend,
        secs,
        secs / total,
        n_a,
        n_b,
    )

    # immediate diagnostics: are BOTH factors independently decodable from the real latents?
    from mop.diagnostics import linear_probe
    from mop.substrate import factorized_arrays

    x, ya, yb = factorized_arrays(store)
    pa = linear_probe(x, ya, classification=True, epochs=300)
    pb = linear_probe(x, yb, classification=True, epochs=300)
    stimulus_set_digest = hashlib.sha256()
    for record in stimulus_records:
        stimulus_set_digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        stimulus_set_digest.update(b"\n")
    run_receipt = {
        "schema": (
            "mop-random-init-encoder-local-attempt/v1"
            if random_control
            else "mop-real-encoder-local-attempt/v1"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": (
            "factorized-random-architecture-control-cache"
            if random_control
            else "factorized-real-encoder-cache"
        ),
        "claim_scope": (
            "seeded random-architecture control over programmatic video; not learned evidence"
            if random_control
            else "real frozen encoder over programmatic video; not natural-video evidence"
        ),
        "cache": str(store.root),
        "host_device": dev.kind,
        "encoder": encoder_identity,
        "samples": total,
        "frames_per_sample": FRAMES,
        "resolution": RES,
        "encode_seconds": secs,
        "seconds_per_clip": secs / total,
        "total_process_seconds": time.perf_counter() - whole_t0,
        "max_rss_bytes": _max_rss_bytes(),
        "stimulus": {
            "schema": "mop-factorized-video-stimulus-receipt/v1",
            "generator": "make_factorized_clip",
            "generator_source_sha256": _sha256(Path(__file__).resolve()),
            "seed": int(cfg.seed),
            "native_resolution_render": True,
            "set_sha256": stimulus_set_digest.hexdigest(),
            "records": stimulus_records,
        },
        "probe": {
            "protocol": "diagnostic random holdout; atlas claims must use frozen manifest splits",
            "factor_a_accuracy": pa["score"],
            "factor_a_chance": pa["chance"],
            "factor_b_accuracy": pb["score"],
            "factor_b_chance": pb["chance"],
        },
        "completed": True,
        "hardware_limit_reached": False,
    }
    (store.root / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2, sort_keys=True) + "\n")
    # Refresh the manifest after the run receipt exists so the performance evidence is fingerprinted
    # alongside arrays, referents, factors, splits, and the immutable weight identity.
    write_cache_manifest(
        store.root,
        encoder_config=encoder_config,
        form_kind="vision",
        form_objective=form_objective,
        referent_scheme="synthetic-factor-cell-repeat",
        full_hash_arrays=total <= 1024,
    )
    proof_receipt = str(cfg.get("proof_receipt", "")).strip()
    if proof_receipt:
        proof_path = Path(proof_receipt)
        if not proof_path.is_absolute():
            proof_path = REPO_ROOT / proof_path
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        proof_path.write_text(json.dumps(run_receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"OK factorized cache: {len(store)} citable latents, "
        f"backend={enc.spec.backend}, {secs / total:.2f}s/clip"
    )
    print(f"   factor A (hue)   linear-probe acc={pa['score']:.3f} chance={pa['chance']:.3f}")
    print(f"   factor B (orient) linear-probe acc={pb['score']:.3f} chance={pb['chance']:.3f}")
    print(f"   (n={len(store)}, underpowered if small; both should decode if the encoder keeps both factors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
