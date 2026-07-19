#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))  # repo root, so `import scripts.xxx` resolves as a package

from transformers import AutoConfig, AutoModel  # noqa: E402
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Predictor  # noqa: E402

from mop.diagnostics.riskcov import seed_ci  # noqa: E402

HF = "facebook/vjepa2-vitl-fpc64-256"
HORIZONS = [1, 2, 3, 4, 6, 8]  # temporal-slot lookaheads
DEFAULT_N_CLIPS = 24  # tractable on CPU
SEED_BUCKETS = 3  # clips are bucketed into this many groups for a seed CI
T_START = 4  # context always includes slots [0..T_START] before the rollout
USABLE_FRACTION = 0.5  # real error must be < this * best-control error to count as usable
RANDINIT_SEED = 12345  # fixed seed for the fresh-weight predictor control
DEVICE = "cpu"  # MPS overflows 64-frame V-JEPA on the M3 Pro; CPU per goal brief


def slot_indices(t: int, pps: int) -> torch.Tensor:
    return torch.arange(t * pps, (t + 1) * pps, dtype=torch.long)


def context_up_to(t: int, pps: int) -> torch.Tensor:
    return torch.cat([slot_indices(i, pps) for i in range(t + 1)])


def nmse(p: torch.Tensor, y: torch.Tensor) -> float:
    return (((p - y) ** 2).sum(-1) / (y**2).sum(-1).clamp_min(1e-8)).mean().item()


def cosd(p: torch.Tensor, y: torch.Tensor) -> float:
    return (1 - F.cosine_similarity(p, y, dim=-1)).mean().item()


@torch.no_grad()
def encode_clip(model, clip: torch.Tensor) -> torch.Tensor:
    out = model(pixel_values_videos=clip.unsqueeze(0), skip_predictor=True)
    return out.last_hidden_state[0]


@torch.no_grad()
def predict_slot(
    predictor, ctx_buffer: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
) -> torch.Tensor:
    out = predictor(
        encoder_hidden_states=ctx_buffer.unsqueeze(0),
        context_mask=[ctx_idx.unsqueeze(0)],
        target_mask=[tgt_idx.unsqueeze(0)],
    )
    return out.last_hidden_state[0]


@torch.no_grad()
def rollout_errors(predictor, seq: torch.Tensor, seq_other: torch.Tensor, pps: int, gd: int):
    res: dict[int, dict[str, tuple[float, float]]] = {}
    buf = seq.clone()
    ctx_idx = context_up_to(T_START, pps)
    step_pred: dict[int, torch.Tensor] = {}
    step_true: dict[int, torch.Tensor] = {}
    for step in range(1, max(HORIZONS) + 1):
        t_tgt = T_START + step
        if t_tgt >= gd:
            break
        tgt_idx = slot_indices(t_tgt, pps)
        pred = predict_slot(predictor, buf, ctx_idx, tgt_idx)
        step_pred[t_tgt] = pred
        step_true[t_tgt] = seq[tgt_idx]
        buf = buf.clone()
        buf[tgt_idx] = pred
        ctx_idx = torch.cat([ctx_idx, tgt_idx])
    for h in HORIZONS:
        t_tgt = T_START + h
        if t_tgt not in step_pred:
            continue
        pred = step_pred[t_tgt]
        true = step_true[t_tgt]
        persist = seq[slot_indices(T_START, pps)]  # copy the last real context slot (no dynamics)
        shuf = seq_other[slot_indices(t_tgt, pps)] if seq_other is not None else true
        res[h] = {
            "real": (nmse(pred, true), cosd(pred, true)),
            "persist": (nmse(persist, true), cosd(persist, true)),
            "shuffled": (nmse(pred, shuf), cosd(pred, shuf)),
        }
    return res


def load_clips(clip_dir: str | None, n: int, frames: int, hw: int):
    if clip_dir:
        paths = sorted(Path(clip_dir).glob("*.pt"))[:n]
        if not paths:
            raise SystemExit(f"no .pt clips found under {clip_dir}")
        for p in paths:
            t = torch.load(p, map_location="cpu").float()
            if t.shape[-1] != hw or t.shape[0] != frames:
                t = F.interpolate(t, size=(hw, hw), mode="bilinear", align_corners=False)
            yield p.stem, t
        return
    from compositional_under_nuisance import make_bound_nuisance_clip

    for i in range(n):
        g = torch.Generator().manual_seed(1000 + i)
        yield f"synthetic_{i}", make_bound_nuisance_clip(i % 5, (i // 5) % 4, 4, g)


def run_predictor(predictor, seqs, tag, t0):
    acc = {h: {"real": [], "persist": [], "shuffled": []} for h in HORIZONS}
    accc = {h: {"real": [], "persist": [], "shuffled": []} for h in HORIZONS}
    for i, seq in enumerate(seqs):
        other = seqs[(i + 1) % len(seqs)]
        r = rollout_errors(predictor, seq, other, seqs.pps, seqs.gd)
        for h, d in r.items():
            for arm, (mn, mc) in d.items():
                acc[h][arm].append(mn)
                accc[h][arm].append(mc)
        if (i + 1) % 8 == 0:
            print(f"  [{tag}] {i + 1}/{len(seqs)} clips  ({time.time() - t0:.0f}s)", flush=True)
    return acc, accc


class SeqSet(list):

    def __init__(self, items, pps: int, gd: int):
        super().__init__(items)
        self.pps = pps
        self.gd = gd


def seed_bucket_means(per_clip):
    buckets: list[list[float]] = [[] for _ in range(SEED_BUCKETS)]
    for j, v in enumerate(per_clip):
        buckets[j % SEED_BUCKETS].append(v)
    return [sum(b) / len(b) for b in buckets if b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=DEFAULT_N_CLIPS)
    ap.add_argument(
        "--clip-dir",
        type=str,
        default=None,
        help="directory of .pt clip tensors [frames,3,H,W]; default = synthetic",
    )
    ap.add_argument("--out", type=str, default=str(_ROOT / "runs" / "mot" / "dr13_predictor_fidelity.json"))
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(HF, local_files_only=True)
    gd = cfg.frames_per_clip // cfg.tubelet_size
    gs = cfg.crop_size // cfg.patch_size
    pps = gs * gs

    model = AutoModel.from_pretrained(HF, local_files_only=True, dtype=torch.float32).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    real_predictor = model.predictor

    torch.manual_seed(RANDINIT_SEED)
    rand_predictor = VJEPA2Predictor(cfg).to(DEVICE)
    rand_predictor.eval()
    for p in rand_predictor.parameters():
        p.requires_grad_(False)

    seqs_list = []
    adj_dist = []
    for _name, clip in load_clips(args.clip_dir, args.n_clips, cfg.frames_per_clip, cfg.crop_size):
        seq = encode_clip(model, clip.to(DEVICE))
        seqs_list.append(seq)
        d = [
            nmse(seq[slot_indices(t + 1, pps)], seq[slot_indices(t, pps)])
            for t in range(T_START, min(T_START + max(HORIZONS), gd - 1))
        ]
        adj_dist.append(sum(d) / len(d))
    seqs = SeqSet(seqs_list, pps, gd)
    scale_nmse = sum(adj_dist) / len(adj_dist)
    print(
        f"encoded {len(seqs)} clips in {time.time() - t0:.0f}s; "
        f"encoder adjacent-slot nmse scale = {scale_nmse:.4f}",
        flush=True,
    )

    real_nmse, real_cosd = run_predictor(real_predictor, seqs, "real", t0)
    rand_nmse, _ = run_predictor(rand_predictor, seqs, "rand-init", t0)

    per_h = {}
    usable_h = 0
    for h in HORIZONS:
        if not real_nmse[h]["real"]:
            continue
        real_ci = seed_ci(seed_bucket_means(real_nmse[h]["real"]))
        persist_ci = seed_ci(seed_bucket_means(real_nmse[h]["persist"]))
        shuf_ci = seed_ci(seed_bucket_means(real_nmse[h]["shuffled"]))
        rand_ci = seed_ci(seed_bucket_means(rand_nmse[h]["real"]))
        controls = {"persistence": persist_ci, "random_init": rand_ci, "shuffled_target": shuf_ci}
        best_control = min(persist_ci["mean"], rand_ci["mean"], shuf_ci["mean"])
        beats = {name: (real_ci["hi"] < c["lo"]) for name, c in controls.items()}
        below_fraction = real_ci["mean"] < USABLE_FRACTION * best_control
        is_usable = all(beats.values()) and below_fraction
        per_h[h] = {
            "real_nmse": real_ci,
            "real_cosd": seed_ci(seed_bucket_means(real_cosd[h]["real"])),
            "controls_nmse": controls,
            "beats_by_seed_ci": beats,
            "best_control_nmse": round(best_control, 4),
            "real_over_best_control": round(real_ci["mean"] / max(best_control, 1e-8), 4),
            "below_half_of_best_control": below_fraction,
            "usable": is_usable,
        }
        if is_usable and h == usable_h + 1:
            usable_h = h

    beats_at_1 = per_h.get(1, {}).get("usable", False)
    if usable_h >= 2:
        verdict = "convert"
    elif beats_at_1:
        verdict = "wall-to-1-step"
    else:
        verdict = "null"

    report = {
        "facet": 12,
        "name": "world-model predictor rollout fidelity (DR13 horizon test on real V-JEPA 2)",
        "device": DEVICE,
        "model": HF,
        "clip_source": args.clip_dir or "synthetic_bound_nuisance",
        "grid": {"temporal_slots": gd, "spatial_side": gs, "patches_per_slot": pps},
        "preregistered": {
            "null": "real predictor never reaches the usability bar at any horizon; a tie is a null",
            "usable_fraction": USABLE_FRACTION,
            "usable_rule": "largest contiguous h where real_nmse.hi < every control_nmse.lo AND "
            "real_nmse.mean < USABLE_FRACTION * best_control_nmse",
            "controls": ["persistence", "random_init", "shuffled_target"],
            "horizons": HORIZONS,
            "n_clips": len(seqs),
            "seed_buckets": SEED_BUCKETS,
            "t_start": T_START,
        },
        "encoder_adjacent_slot_nmse_scale": round(scale_nmse, 4),
        "per_horizon": per_h,
        "usable_horizon": usable_h,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\n==== RESULT ====")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
