#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop_dr13_predictor_fidelity import (  # noqa: E402
    DEVICE,
    HF,
    T_START,
    context_up_to,
    encode_clip,
    load_clips,
    nmse,
    predict_slot,
    slot_indices,
)
from transformers import AutoConfig, AutoModel  # noqa: E402

from mop.diagnostics.riskcov import seed_ci  # noqa: E402

HORIZONS = [1, 2, 3, 4]
USABLE_FRACTION = 0.5  # adapted nmse must be below this * persistence to count as usable
RIDGE = 1e-2  # ridge on the adapter least-squares (stabilizes the D x D solve)
SEED_BUCKETS = 3


@torch.no_grad()
def in_context_pairs(predictor, seq, pps):
    ps, ys = [], []
    for s in range(1, T_START + 1):
        ctx = context_up_to(s, pps)
        tgt = slot_indices(s, pps)
        pred = predict_slot(predictor, seq, ctx, tgt)
        ps.append(pred)
        ys.append(seq[tgt])
    return torch.cat(ps), torch.cat(ys)


@torch.no_grad()
def rollout_future(predictor, seq, pps, gd):
    buf = seq.clone()
    ctx = context_up_to(T_START, pps)
    out = {}
    persist = seq[slot_indices(T_START, pps)]
    for step in range(1, max(HORIZONS) + 1):
        t = T_START + step
        if t >= gd:
            break
        tgt = slot_indices(t, pps)
        pred = predict_slot(predictor, buf, ctx, tgt)
        if step in HORIZONS:
            out[step] = (pred, seq[tgt], persist)
        buf = buf.clone()
        buf[tgt] = pred
        ctx = torch.cat([ctx, tgt])
    return out


def fit_adapter(P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    d = P.shape[1]
    ptp = P.T @ P
    lam = RIDGE * (ptp.diagonal().mean().item() + 1e-8)
    return torch.linalg.solve(ptp + lam * torch.eye(d), P.T @ Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=16)
    ap.add_argument("--clip-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(_ROOT / "runs" / "mot" / "dr13_readout_adapter.json"))
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(HF, local_files_only=True)
    gd = cfg.frames_per_clip // cfg.tubelet_size
    gs = cfg.crop_size // cfg.patch_size
    pps = gs * gs
    model = AutoModel.from_pretrained(HF, local_files_only=True, dtype=torch.float32).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    predictor = model.predictor

    seqs = [
        encode_clip(model, clip.to(DEVICE))
        for _name, clip in load_clips(args.clip_dir, args.n_clips, cfg.frames_per_clip, cfg.crop_size)
    ]
    n = len(seqs)
    ntr = max(2, n // 2)
    train, test = seqs[:ntr], seqs[ntr:] or seqs[:1]

    P = torch.cat([in_context_pairs(predictor, s, pps)[0] for s in train])
    Y = torch.cat([in_context_pairs(predictor, s, pps)[1] for s in train])
    in_ctx_floor = nmse(P, Y)
    adapted_in_ctx = nmse(P @ fit_adapter(P, Y), Y)
    A = fit_adapter(P, Y)

    per_h = {}
    for h in HORIZONS:
        raw_l, adp_l, per_l = [], [], []
        for s in test:
            fut = rollout_future(predictor, s, pps, gd)
            if h not in fut:
                continue
            pred, true, persist = fut[h]
            raw_l.append(nmse(pred, true))
            adp_l.append(nmse(pred @ A, true))
            per_l.append(nmse(persist, true))
        if not raw_l:
            continue

        def bucket(v):
            b = [[] for _ in range(SEED_BUCKETS)]
            for j, x in enumerate(v):
                b[j % SEED_BUCKETS].append(x)
            return [sum(z) / len(z) for z in b if z]

        raw_ci, adp_ci, per_ci = seed_ci(bucket(raw_l)), seed_ci(bucket(adp_l)), seed_ci(bucket(per_l))
        gap_closed = (raw_ci["mean"] - adp_ci["mean"]) / max(raw_ci["mean"] - in_ctx_floor, 1e-6)
        recovers = adp_ci["hi"] < raw_ci["lo"] and adp_ci["mean"] < USABLE_FRACTION * per_ci["mean"]
        per_h[h] = {
            "raw_nmse": raw_ci,
            "adapted_nmse": adp_ci,
            "persistence_nmse": per_ci,
            "gap_closed_fraction": round(gap_closed, 4),
            "adapter_beats_raw": bool(adp_ci["hi"] < raw_ci["lo"]),
            "adapter_recovers_usable": bool(recovers),
        }

    usable = any(v["adapter_recovers_usable"] for v in per_h.values())
    verdict = "adapter-recovers" if usable else "adapter-insufficient"
    report = {
        "facet": 12,
        "name": "readout-adapter rollout fidelity recovery (facet 12 licensed re-test)",
        "clip_source": args.clip_dir or "synthetic_bound_nuisance",
        "device": DEVICE,
        "preregistered": {
            "null": "the adapter does not bring adapted rollout nmse below both raw and 0.5*persistence "
            "at any horizon; a tie is a null",
            "usable_fraction": USABLE_FRACTION,
            "ridge": RIDGE,
            "horizons": HORIZONS,
            "adapter": "linear D x D, fit on visible-slot (pred, true) pairs on a train split, applied "
            "unchanged to a held-out split",
        },
        "in_context_gap_nmse": round(in_ctx_floor, 4),
        "in_context_gap_after_adapter": round(adapted_in_ctx, 4),
        "n_clips": n,
        "n_train": ntr,
        "per_horizon": per_h,
        "verdict": verdict,
        "caveat": "synthetic near-static clips confound the adapter (future ~ in-context); the licensed "
        "verdict needs real moving video via --clip-dir.",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
