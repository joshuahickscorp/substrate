#!/usr/bin/env python

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from compositional_under_nuisance import make_bound_nuisance_clip  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Predictor  # noqa: E402

HF = "facebook/vjepa2-vitl-fpc64-256"
REPORT = _ROOT / "runs" / "mot" / "dr13_predictor_fidelity.json"


def nmse(p, y):
    return (((p - y) ** 2).sum(-1) / (y**2).sum(-1).clamp_min(1e-8)).mean().item()


def main():
    cfg = AutoConfig.from_pretrained(HF, local_files_only=True)
    gs = cfg.crop_size // cfg.patch_size
    pps = gs * gs
    m = AutoModel.from_pretrained(HF, local_files_only=True, dtype=torch.float32)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    real_pred = m.predictor
    torch.manual_seed(999)  # DIFFERENT seed than harness rand-init
    rand_pred = VJEPA2Predictor(cfg).eval()
    for p in rand_pred.parameters():
        p.requires_grad_(False)

    checks = []

    dw = []
    rp = dict(rand_pred.named_parameters())
    for name, w in real_pred.named_parameters():
        if name in rp and rp[name].shape == w.shape:
            dw.append((rp[name] - w).abs().mean().item())
    mean_dw = sum(dw) / len(dw)
    ok = mean_dw > 1e-3
    checks.append(("V1a random-init != real weights", ok, f"mean|dw|={mean_dw:.4f}"))

    def slot(t):
        return torch.arange(t * pps, (t + 1) * pps, dtype=torch.long)

    def ctx_upto(t):
        return torch.cat([slot(i) for i in range(t + 1)])

    @torch.no_grad()
    def enc(clip):
        return m(pixel_values_videos=clip.unsqueeze(0), skip_predictor=True).last_hidden_state[0]

    @torch.no_grad()
    def pred_slot(predictor, buf, cidx, tidx):
        out = predictor(
            encoder_hidden_states=buf.unsqueeze(0),
            context_mask=[cidx.unsqueeze(0)],
            target_mask=[tidx.unsqueeze(0)],
        )
        return out.last_hidden_state[0]

    T = 4
    reals, persists, rands, shuf_ok = [], [], [], []
    seqs = []
    for i in range(6):
        g = torch.Generator().manual_seed(50000 + i)  # distinct seed space
        clip = make_bound_nuisance_clip(i % 5, (i // 5) % 4, 4, g)
        seqs.append(enc(clip))
    for i, seq in enumerate(seqs):
        cidx = ctx_upto(T)
        tidx = slot(T + 1)
        pr = pred_slot(real_pred, seq, cidx, tidx)
        rr = pred_slot(rand_pred, seq, cidx, tidx)
        true = seq[tidx]
        persist = seq[slot(T)]
        reals.append(nmse(pr, true))
        persists.append(nmse(persist, true))
        rands.append(nmse(rr, true))
        other = seqs[(i + 1) % len(seqs)][tidx]
        shuf_ok.append(nmse(true, other) > 1e-3)

    mr = sum(reals) / len(reals)
    mp = sum(persists) / len(persists)
    mrand = sum(rands) / len(rands)
    checks.append(
        ("V1b shuffled-target is a different clip", all(shuf_ok), f"all {len(shuf_ok)} clips true!=other")
    )
    checks.append(("V2a real < persistence (h=1)", mr < mp, f"real={mr:.4f} persist={mp:.4f}"))
    checks.append(("V2b real < random-init (h=1)", mr < mrand, f"real={mr:.4f} rand={mrand:.4f}"))

    with torch.no_grad():
        cidx = ctx_upto(T)
        in_ctx = pred_slot(real_pred, seqs[0], cidx, slot(T))  # slot in context
        e_in = nmse(in_ctx, seqs[0][slot(T)])
        fut = pred_slot(real_pred, seqs[0], cidx, slot(T + 1))  # future slot
        e_fut = nmse(fut, seqs[0][slot(T + 1)])
    checks.append(
        ("V3 future harder than in-context (no leak)", e_fut > e_in, f"in_ctx={e_in:.4f} future={e_fut:.4f}")
    )

    verdict = None
    if REPORT.exists():
        rep = json.loads(REPORT.read_text())
        verdict = rep.get("verdict")
        h1 = rep.get("per_horizon", {}).get("1", {})
        rep_real = h1.get("real_nmse", {}).get("mean")
        if rep_real is not None:
            close = abs(mr - rep_real) < 0.25
            checks.append(
                ("V4 harness h=1 real nmse reproduces", close, f"verify={mr:.4f} harness={rep_real}")
            )

    print("=== INDEPENDENT ADVERSARIAL VERIFICATION (facet 12) ===")
    allpass = True
    for name, ok, detail in checks:
        allpass &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    print(f"harness verdict = {verdict}")
    print(f"OVERALL: {'ALL PASS' if allpass else 'SOME FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
