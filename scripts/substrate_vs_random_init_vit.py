#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
from scripts.substrate_vs_random_features import (
    make_nuisance_clip,
    random_pixel_features,
)

from mop.config import compose
from mop.devices import resolve, safe_to
from mop.diagnostics import linear_probe
from mop.substrate import load_encoder
from mop.substrate.encoder import EncoderSpec, FrozenEncoder


def _random_init_vit(cfg) -> FrozenEncoder | None:
    try:
        from transformers import AutoConfig, AutoModel

        c = AutoConfig.from_pretrained(cfg.hf_id, trust_remote_code=True)
        with torch.no_grad():
            m = AutoModel.from_config(c, trust_remote_code=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        spec = EncoderSpec(
            name=cfg.name + "_randinit",
            embed_dim=int(cfg.embed_dim),
            dense=bool(cfg.dense),
            pool=str(cfg.pool),
            backend="random_init_vit",
        )
        return FrozenEncoder(spec, m)
    except Exception:
        return None


def _zscore(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, keepdim=True) + 1e-6
    return (x - mu) / sd


def run(n_shape: int, per: int, seed: int) -> dict:
    cfg = compose(
        [
            "encoder=vjepa2_vitl_fpc64_256",
            "device=cpu",
            "encoder.prefer_real=true",
            "+encoder.require_real=true",
        ]
    )
    dev = resolve("cpu")
    real = load_encoder(cfg.encoder).to(dev.device)
    real_backend = real.spec.backend
    rand = _random_init_vit(cfg.encoder)
    if rand is not None:
        rand = rand.to(dev.device)

    D = int(cfg.encoder.embed_dim)
    g = torch.Generator().manual_seed(seed)
    ds, tsub = 32, 8
    pin = tsub * 3 * ds * ds
    pg = torch.Generator().manual_seed(seed + 1)
    proj = torch.randn(pin, D, generator=pg) / (pin**0.5)

    labels = [s for s in range(n_shape) for _ in range(per)]
    Xreal, Xrand, Xpix = [], [], []
    t0 = time.perf_counter()
    for i, s in enumerate(labels):
        clip = make_nuisance_clip(s, g)
        cin = safe_to(clip.unsqueeze(0), dev.device)
        Xreal.append(real.encode(cin).reshape(-1).float().cpu())
        if rand is not None:
            Xrand.append(rand.encode(cin).reshape(-1).float().cpu())
        Xpix.append(random_pixel_features(clip, proj, ds, tsub))
        if (i + 1) % 4 == 0 or i + 1 == len(labels):
            print(f"encoded {i + 1}/{len(labels)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    y = torch.tensor(labels)

    def probe(feats: list[torch.Tensor]) -> dict:
        r = linear_probe(_zscore(torch.stack(feats)), y, classification=True, epochs=400, seed=seed)
        return {"acc": round(r["score"], 4), "chance": round(r["chance"], 4)}

    pv = probe(Xreal)
    pp = probe(Xpix)
    out = {
        "real_backend": real_backend,
        "random_init_available": rand is not None,
        "n_clips": len(labels),
        "n_shape": n_shape,
        "per_class": per,
        "chance": pv["chance"],
        "seconds": round(time.perf_counter() - t0, 1),
        "vjepa_shape_acc": pv["acc"],
        "random_pixel_shape_acc": pp["acc"],
        "vjepa_minus_random_pixel": round(pv["acc"] - pp["acc"], 4),
    }
    if rand is not None:
        pr = probe(Xrand)
        out["random_init_vit_shape_acc"] = pr["acc"]
        out["vjepa_minus_random_init_vit"] = round(pv["acc"] - pr["acc"], 4)

    delta_vit = out.get("vjepa_minus_random_init_vit")
    if real_backend != "vjepa_hf":
        out["verdict"] = "INVALID: real encoder ran as frozen_random, rerun with real weights"
    elif rand is None:
        out["verdict"] = (
            "INCOMPLETE: could not build random-init ViT offline; only random-pixel floor available"
        )
    elif delta_vit > 0.15:
        out["verdict"] = (
            "PRETRAINING IS THE DIFFERENCE: real V-JEPA decodes shape under nuisance far better than an "
            "identical-architecture random-init ViT at the same resolution. The substrate is special "
            "because of its learned weights, not its architecture or input size. The fork reopens toward "
            "keeping the frozen encoder; the earlier 'not special' read was an artifact of the vacuous "
            "latent-projection control."
        )
    elif delta_vit > 0.05:
        out["verdict"] = (
            "pretraining modestly helps: real V-JEPA beats the random-init ViT by a real but small margin"
        )
    elif pv["acc"] < pv["chance"] + 0.1:
        out["verdict"] = "TOO HARD: real V-JEPA itself near chance, nuisance overwhelmed even the substrate"
    else:
        out["verdict"] = (
            "pretraining does NOT help here: a random-init ViT of the same architecture decodes shape as "
            "well as the pretrained one, so the substrate advantage (if any) is architectural, not learned. "
            "Strong evidence pretrained weights add no invariance the architecture does not already give."
        )
    return out


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="real V-JEPA vs random-init same-arch ViT vs random-pixel")
    ap.add_argument("--n-shape", type=int, default=6)
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = run(a.n_shape, a.per, a.seed)
    text = json.dumps(result, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
