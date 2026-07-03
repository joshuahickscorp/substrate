#!/usr/bin/env python
"""DR13 on the REAL V-JEPA 2 PREDICTOR: world-model rollout fidelity (facet 12 acceptance gate).

The synthetic sibling `scripts/mop_dr13_horizon_limit.py` tests rollout-error compounding against
RANDOM transitions. This script is the real-predictor counterpart the Studio audit calls for
(STUDIO_POTENTIAL_AUDIT.md facet 12): it drives V-JEPA 2's actual PREDICTOR, the learned
latent-space simulator of video dynamics that the entire MoP corpus used the ENCODER of and threw
away, and measures how far its latent rollouts stay faithful before error compounds past the
trivial baselines. It gates the whole rollout lane (counterfactual and interventional abstraction,
the ex2 latent-planning precursor, DR7 latent chain-of-thought).

MECHANISM (from transformers/models/vjepa2/modeling_vjepa2.py): the predictor is a masked
spatiotemporal-patch predictor over one clip's temporal-slot x spatial patch grid. Given encoder
hidden states at CONTEXT patch indices plus TARGET patch indices, it forecasts the target patch
representations. The teacher is the encoder's OWN representation of those target patches, exactly
V-JEPA's training signal (validated bit-exact against the top-level VJEPA2Model path).

TEMPORAL ROLLOUT: context = all patches in temporal slots [0..t]; target = all patches in slot t+h.
h=1 single-step predicts slot t+1 from ground-truth context. h>=2 compounds by substituting the
PREDICTED slot reps back into the context buffer before predicting the next slot, a true open-loop
rollout through the model's own outputs.

PREREGISTERED NULL, THRESHOLD, CONTROLS (fixed in code before any number):
  usable horizon = the largest CONTIGUOUS h (from h=1) at which real_nmse.hi < every control_nmse.lo
  (non-overlapping seed CI, lower is better) AND real_nmse.mean < USABLE_FRACTION * best_control_nmse.
  A tie is a NULL. Controls (all non-vacuous):
    (a) persistence : predict slot t+h = copy the last ground-truth context slot (no dynamics).
    (b) random_init : the SAME predictor architecture, freshly initialized weights, identical pipe.
    (c) shuffled_tgt: the real predictor output scored against a DIFFERENT clip's true slot t+h
        (shares the predictor-vs-encoder representational gap, so beating it isolates genuine
        clip-specific dynamics net of that gap).
  Verdict: convert (usable horizon >= 2), wall-to-1-step (real cleanly usable only at h=1),
  null (real never reaches the usability bar at any horizon; a directional-but-sub-usable signal
  is still a null under this rule).

PROVENANCE (M3 Pro, 24 synthetic bound-nuisance clips, CPU, 2026-07-03): verdict NULL. The real
predictor beat all three controls by non-overlapping seed CI at every horizon 1..8, but by only
~5 to 7 percent (real ~0.93 of the best control), never near the 0.5 usability bar. A leakage probe
showed the predictor is lossy even on a VISIBLE slot (in-context nmse ~0.75), so most of the ~0.77
one-step error is a predictor-vs-encoder representational gap and the marginal one-step forecast
cost is small (~0.025 nmse above that floor); the encoder's own adjacent-slot nmse scale is ~0.95,
so slots are nearly decorrelated. This is a real-but-sub-usable world-model signal. The clips are
SYNTHETIC and out-of-distribution for a predictor trained on real video, and the whole-future-slot
masking is OOD for the training mask distribution, so the null is PROVISIONAL on this clipset; the
licensed real-scale verdict requires re-running with --clip-dir on the Studio's hosted real corpora
(facet 14 feeds facet 12). See docs/mixture_of_perspectives/ROLLOUT_LANE_RESULT.md.

Form (goal loop): no em dashes or en dashes. Preregister before running. A tie is a null. No score
is faked. A proven wall with a mechanism is success.

Usage:
  PYTHONPATH=<repo>/src:<repo>/scripts OMP_NUM_THREADS=4 \
    .venv/bin/python scripts/mop_dr13_predictor_fidelity.py [--n-clips 24] [--clip-dir DIR] [--out OUT]

  --clip-dir DIR points the gate at real clips: a directory of .pt tensors each shaped
  [frames, 3, H, W] (the Studio real-corpora re-run). Default is the synthetic reproduction.
"""

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

# ------------------------------------------------------------------------------------------------
# PREREGISTERED CONSTANTS (before any number exists)
# ------------------------------------------------------------------------------------------------
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
    """clip [frames,3,H,W] -> encoder patch states [N, D]."""
    out = model(pixel_values_videos=clip.unsqueeze(0), skip_predictor=True)
    return out.last_hidden_state[0]


@torch.no_grad()
def predict_slot(
    predictor, ctx_buffer: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
) -> torch.Tensor:
    """Run the predictor once. ctx_buffer [N,D] is the full patch buffer (real encoder states,
    with predicted slots substituted in for h>=2). Returns predicted target rows [len(tgt),D]."""
    out = predictor(
        encoder_hidden_states=ctx_buffer.unsqueeze(0),
        context_mask=[ctx_idx.unsqueeze(0)],
        target_mask=[tgt_idx.unsqueeze(0)],
    )
    return out.last_hidden_state[0]


@torch.no_grad()
def rollout_errors(predictor, seq: torch.Tensor, seq_other: torch.Tensor, pps: int, gd: int):
    """One clip. Returns {horizon -> {real,persist,shuffled}: (nmse, cosd)}. Compounding: predicted
    slots are fed back into the context buffer for the next step (true open-loop rollout)."""
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
    """Yield (name, clip[frames,3,hw,hw]). Synthetic by default; a directory of .pt clip tensors
    (each [frames,3,hw,hw]) points the gate at real corpora for the Studio real-scale re-run."""
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
    """A list of encoder sequences carrying the grid geometry for the rollout helpers."""

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
