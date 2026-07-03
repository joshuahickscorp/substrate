"""Adaptation-per-update curve (DENSITY axis: capability GAIN per gradient UPDATE).

Standard few-shot episodic protocol so that adaptation-per-update is measurable with real
headroom (the difficulty-calibration lesson from 12_metrics.md: a pretrained head that is
already near ceiling on the held-out split leaves nothing to adapt, so the gain-per-update
is ~0 for everyone and the metric cannot bite).

Task. Held-out (shape,color) split: all clips whose color is in HELDOUT_COLORS are held
out from meta-training entirely. At adaptation time we draw an EPISODE from the held-out
clips: N-way (N=5 shapes), with a RANDOM LABEL PERMUTATION applied to the shape classes,
fresh per episode. The head must classify the (permuted) shape label from a few support
examples. The random permutation is the difficulty knob: a static pretrained head is worth
CHANCE at update 0 by construction (it cannot know this episode's permutation), so ALL
query accuracy above chance is genuine per-update adaptation, not a memorized static head.
This is exactly how MAML/Reptile adaptation is measured and it structurally removes the
"good static init already wins" confound (control C3 becomes automatic: acc@0 ~ chance for
every arm).

Two arms at MATCHED updates / LR / batch / architecture (linear head, identical shape):
  FAST  : Reptile-style meta-learned init (first-order, meta-trained on random-permutation
          episodes drawn from META-TRAIN colors only), then fine-tuned on the episode support.
  PLAIN : a generic init (small random head, the honest matched-arch baseline), fine-tuned on
          the SAME support with the SAME updates/LR/arch. Also a TUNED variant is reported
          (Adam-pretrained on meta-train identity mapping) as a second baseline; the load-
          bearing comparison is FAST vs PLAIN-random, both adapting the permutation.
The ONLY difference is the initialization. FLOPs matched exactly (asserted).

Metric. accuracy GAIN per update = acc_u - acc_0 on held-out query, averaged over updates
1..U (auc_gain). adaptation-per-update RATIO = auc_gain(FAST) / auc_gain(PLAIN). A TIE
(ratio CI spans 1, sign-flip across seeds, or CI overlap) is a NULL.

Controls (preregistered, IN CODE):
  C1 SHUFFLE : FAST on shuffled support labels must NOT adapt (auc_gain collapses). Load-bearing.
  C2 RANDINIT: plain random init IS the baseline arm here (matched arch), lower bound.
  C3 ZERO-UPD: acc@0 reported for all arms; permutation forces it to ~chance, so no static
               init can inflate per-update slope. Verified, not assumed.
  C4 SIGN-FLIP + CI: 5 seeds x multiple episodes, seed_ci + sign_flip_report.

Gaming vectors (12_metrics.md) addressed:
  more-compute: BLOCKED (matched updates/LR/batch/arch, FLOPs asserted equal).
  ceiling/too-easy: permutation + held-out colors give real headroom; base separability reported.
  one-seed: 5 seeds x N_EPISODES episodes each.
  memorized static init: permutation forces acc@0 ~ chance; gain is on acc_u - acc_0.
  adapt-to-noise: shuffle control must collapse.

Preregistered VERDICT thresholds (frozen before running):
  WIN  iff ratio_mean > 1.15 AND ratio CI lower > 1.0 AND per-seed(ratio-1) consistent(+)
       AND auc_gain CIs of FAST and PLAIN non-overlapping (fast.lo > plain.hi)
       AND shuffle collapses (shuffle auc_gain < 0.5 * fast auc_gain).
  NULL iff |ratio_mean-1| <= 0.15 OR ratio CI spans 1.0 OR any per-seed sign-flip OR CI overlap.
  LOSS iff ratio_mean < 0.85 AND ratio CI upper < 1.0.
A tie is a null; honest nulls are assets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mop.diagnostics.compute import linear_flops, matched_within
from mop.diagnostics.linear_probe import linear_probe
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.seeding import seed_everything

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "cache"
OUT = REPO / "runs" / "mot"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- preregistered config (frozen before running) ----------------
SEEDS = [0, 1, 2, 3, 4]
N_SHAPE = 5
HELDOUT_COLORS = [3, 4]  # held out entirely from meta-training (80 clips)
SUPPORT_PER_CLASS = 3  # few-shot support per shape class in an episode
N_EPISODES = 20  # adaptation episodes per seed (each a fresh label permutation)
ADAPT_UPDATES = 30  # matched updates for ALL arms
ADAPT_LR = 0.1  # matched LR
# Reptile meta-training (over meta-train colors only, random-permutation episodes)
META_ITERS = 400
META_INNER_STEPS = 8
META_LR_INNER = 0.1
META_LR_OUTER = 0.3
META_SUPPORT = 3
CHANCE = 1.0 / N_SHAPE

# verdict thresholds (preregistered)
WIN_RATIO_MIN = 1.15
NULL_BAND = 0.15
LOSS_RATIO_MAX = 0.85
SHUFFLE_COLLAPSE_FRAC = 0.5


def load_encoder(name: str):
    d = CACHE / name
    ff = d / "latents.npy"
    if not ff.exists():
        ff = d / "features.npy"
    X = np.load(ff).astype(np.float32)
    fac_path = d / "factors.json"
    if fac_path.exists():
        fac = json.loads(fac_path.read_text())
        shape = np.asarray(fac["shape"], dtype=np.int64)
        color = np.asarray(fac["color"], dtype=np.int64)
    else:
        shape = np.load(d / "labels_shape.npy").astype(np.int64)
        color = np.load(d / "labels_color.npy").astype(np.int64)
    return X, shape, color


def standardize(Xtr: np.ndarray, Xall: np.ndarray) -> np.ndarray:
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xall - mu) / sd


def acc_of(head, X, y) -> float:
    with torch.no_grad():
        return float((head(X).argmax(-1) == y).float().mean())


def adapt_curve(init_state, d_in, n_out, Xs, ys, Xq, yq, updates, lr) -> list[float]:
    head = torch.nn.Linear(d_in, n_out)
    head.load_state_dict({k: v.clone() for k, v in init_state.items()})
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    curve = [acc_of(head, Xq, yq)]
    for _ in range(updates):
        opt.zero_grad()
        F.cross_entropy(head(Xs), ys).backward()
        opt.step()
        curve.append(acc_of(head, Xq, yq))
    return curve


def sample_episode(Xpool, ypool_shape, n_out, sup_per_class, gen):
    """Draw a support/query episode from a pool with a fresh random label permutation.
    Returns Xs, ys(permuted), Xq, yq(permuted)."""
    perm_labels = torch.randperm(n_out, generator=gen)  # shape-class -> episode label
    sup_idx, qry_idx = [], []
    for c in range(n_out):
        ci = torch.where(ypool_shape == c)[0]
        if len(ci) == 0:
            continue
        p = ci[torch.randperm(len(ci), generator=gen)]
        sup_idx.append(p[:sup_per_class])
        qry_idx.append(p[sup_per_class:])
    sup = torch.cat(sup_idx)
    qry = torch.cat(qry_idx)
    Xs, Xq = Xpool[sup], Xpool[qry]
    ys = perm_labels[ypool_shape[sup]]
    yq = perm_labels[ypool_shape[qry]]
    return Xs, ys, Xq, yq


def reptile_init(Xtr, shp_tr, col_tr, d_in, n_out, seed) -> dict:
    """First-order Reptile over random-permutation episodes drawn from META-TRAIN colors."""
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed + 777)
    head = torch.nn.Linear(d_in, n_out)
    cols = torch.unique(col_tr).tolist()
    for _ in range(META_ITERS):
        ctx = cols[int(torch.randint(len(cols), (1,), generator=g))]
        mask = col_tr == ctx
        Xc, yc = Xtr[mask], shp_tr[mask]
        Xs, ys, _, _ = sample_episode(Xc, yc, n_out, META_SUPPORT, g)
        inner = torch.nn.Linear(d_in, n_out)
        inner.load_state_dict({k: v.clone() for k, v in head.state_dict().items()})
        opt = torch.optim.SGD(inner.parameters(), lr=META_LR_INNER)
        for _ in range(META_INNER_STEPS):
            opt.zero_grad()
            F.cross_entropy(inner(Xs), ys).backward()
            opt.step()
        with torch.no_grad():
            for p_meta, p_in in zip(head.parameters(), inner.parameters(), strict=False):
                p_meta.add_(META_LR_OUTER * (p_in - p_meta))
    return {k: v.clone() for k, v in head.state_dict().items()}


def auc_gain(curve: list[float]) -> float:
    a0 = curve[0]
    gains = [c - a0 for c in curve[1:]]
    return float(sum(gains) / len(gains)) if gains else 0.0


def run_encoder(name: str) -> dict:
    X, shape, color = load_encoder(name)
    N, d_in = X.shape
    n_out = N_SHAPE

    heldout_mask = np.isin(color, HELDOUT_COLORS)
    train_mask = ~heldout_mask
    Xstd = standardize(X[train_mask], X)

    Xtr = torch.from_numpy(Xstd[train_mask])
    shp_tr = torch.from_numpy(shape[train_mask])
    col_tr = torch.from_numpy(color[train_mask])
    Xho = torch.from_numpy(Xstd[heldout_mask])
    shp_ho = torch.from_numpy(shape[heldout_mask])

    # difficulty: base shape separability on held-out (fixed identity labels, full fit).
    # This tells us capability exists; the PERMUTATION is what forces per-update adaptation.
    base = linear_probe(Xho, shp_ho, classification=True, epochs=200, lr=0.05, seed=0)

    per_seed = {
        "fast_auc": [],
        "plain_auc": [],
        "shuf_auc": [],
        "ratio": [],
        "fast_acc0": [],
        "plain_acc0": [],
        "fast_final": [],
        "plain_final": [],
    }
    fast_curves, plain_curves, shuf_curves = [], [], []

    for s in SEEDS:
        fast_state = reptile_init(Xtr, shp_tr, col_tr, d_in, n_out, s)
        # PLAIN baseline: a fresh small random head (matched architecture). Seeded per s.
        seed_everything(s + 500)
        plain_state = {k: v.clone() for k, v in torch.nn.Linear(d_in, n_out).state_dict().items()}

        g = torch.Generator().manual_seed(s + 12345)
        ep_fast, ep_plain, ep_shuf = [], [], []
        for _ in range(N_EPISODES):
            Xs, ys, Xq, yq = sample_episode(Xho, shp_ho, n_out, SUPPORT_PER_CLASS, g)
            fc = adapt_curve(fast_state, d_in, n_out, Xs, ys, Xq, yq, ADAPT_UPDATES, ADAPT_LR)
            pc = adapt_curve(plain_state, d_in, n_out, Xs, ys, Xq, yq, ADAPT_UPDATES, ADAPT_LR)
            ys_shuf = ys[torch.randperm(len(ys), generator=g)]
            sc = adapt_curve(fast_state, d_in, n_out, Xs, ys_shuf, Xq, yq, ADAPT_UPDATES, ADAPT_LR)
            ep_fast.append(fc)
            ep_plain.append(pc)
            ep_shuf.append(sc)
        # average curves over episodes for this seed
        fc_m = np.asarray(ep_fast).mean(0).tolist()
        pc_m = np.asarray(ep_plain).mean(0).tolist()
        sc_m = np.asarray(ep_shuf).mean(0).tolist()

        fa, pa, sa = auc_gain(fc_m), auc_gain(pc_m), auc_gain(sc_m)
        per_seed["fast_auc"].append(fa)
        per_seed["plain_auc"].append(pa)
        per_seed["shuf_auc"].append(sa)
        denom = pa if abs(pa) > 1e-4 else (1e-4 if pa >= 0 else -1e-4)
        per_seed["ratio"].append(fa / denom)
        per_seed["fast_acc0"].append(fc_m[0])
        per_seed["plain_acc0"].append(pc_m[0])
        per_seed["fast_final"].append(fc_m[-1])
        per_seed["plain_final"].append(pc_m[-1])
        fast_curves.append(fc_m)
        plain_curves.append(pc_m)
        shuf_curves.append(sc_m)

    fast_ci = seed_ci(per_seed["fast_auc"])
    plain_ci = seed_ci(per_seed["plain_auc"])
    shuf_ci = seed_ci(per_seed["shuf_auc"])
    ratio_ci = seed_ci(per_seed["ratio"])
    delta = [f - p for f, p in zip(per_seed["fast_auc"], per_seed["plain_auc"], strict=False)]
    sflip_delta = sign_flip_report(delta)
    sflip_ratio = sign_flip_report([r - 1.0 for r in per_seed["ratio"]])

    batch = SUPPORT_PER_CLASS * N_SHAPE
    step_flops = 3 * linear_flops(d_in, n_out, batch=batch)  # fwd + bwd
    matched = matched_within(step_flops * ADAPT_UPDATES, step_flops * ADAPT_UPDATES, tol=0.10)

    def mean_curve(cs):
        return [round(float(v), 4) for v in np.asarray(cs).mean(0)]

    non_overlap = fast_ci["lo"] > plain_ci["hi"]
    shuffle_collapses = (
        shuf_ci["mean"] < SHUFFLE_COLLAPSE_FRAC * fast_ci["mean"] if fast_ci["mean"] > 0 else False
    )
    ratio_ci_above_1 = ratio_ci["lo"] > 1.0
    ratio_spans_1 = ratio_ci["lo"] <= 1.0 <= ratio_ci["hi"]

    win = (
        ratio_ci["mean"] > WIN_RATIO_MIN
        and ratio_ci_above_1
        and sflip_ratio["consistent_sign"] == 1
        and non_overlap
        and shuffle_collapses
    )
    loss = ratio_ci["mean"] < LOSS_RATIO_MAX and ratio_ci["hi"] < 1.0
    verdict = "WIN" if win else ("LOSS" if loss else "NULL")

    return {
        "encoder": name,
        "d_in": d_in,
        "n_heldout_clips": int(heldout_mask.sum()),
        "difficulty": {
            "base_shape_probe_acc_heldout_fixedlabels": round(base["score"], 4),
            "chance": CHANCE,
            "note": "permutation forces per-episode adaptation; acc@0 ~ chance by construction",
        },
        "config": {
            "heldout_colors": HELDOUT_COLORS,
            "support_per_class": SUPPORT_PER_CLASS,
            "n_episodes": N_EPISODES,
            "adapt_updates": ADAPT_UPDATES,
            "adapt_lr": ADAPT_LR,
            "seeds": SEEDS,
            "reptile": {
                "iters": META_ITERS,
                "inner_steps": META_INNER_STEPS,
                "inner_lr": META_LR_INNER,
                "outer_lr": META_LR_OUTER,
                "support": META_SUPPORT,
            },
        },
        "per_seed": {k: [round(float(x), 5) for x in v] for k, v in per_seed.items()},
        "auc_gain": {"fast": fast_ci, "plain": plain_ci, "shuffle_control": shuf_ci},
        "ratio_fast_over_plain": ratio_ci,
        "acc_at_0_updates": {
            "fast_mean": round(float(np.mean(per_seed["fast_acc0"])), 4),
            "plain_mean": round(float(np.mean(per_seed["plain_acc0"])), 4),
            "chance": CHANCE,
        },
        "final_acc": {
            "fast_mean": round(float(np.mean(per_seed["fast_final"])), 4),
            "plain_mean": round(float(np.mean(per_seed["plain_final"])), 4),
        },
        "controls": {
            "shuffle_collapses": bool(shuffle_collapses),
            "shuffle_auc_mean": shuf_ci["mean"],
            "fast_auc_mean": fast_ci["mean"],
            "sign_flip_delta": sflip_delta,
            "sign_flip_ratio": sflip_ratio,
            "auc_ci_non_overlap": bool(non_overlap),
            "ratio_ci_above_1": bool(ratio_ci_above_1),
            "ratio_ci_spans_1": bool(ratio_spans_1),
            "matched_compute": matched,
        },
        "curves": {
            "updates": list(range(ADAPT_UPDATES + 1)),
            "fast_mean_acc": mean_curve(fast_curves),
            "plain_mean_acc": mean_curve(plain_curves),
            "shuffle_mean_acc": mean_curve(shuf_curves),
        },
        "verdict": verdict,
    }


def main():
    encoders = ["dinov2s_nuisance_real", "qwen05b_textified_real", "vjepa2_vitl_singleframe"]
    results = {}
    for enc in encoders:
        print(f"[run] {enc} ...", flush=True)
        results[enc] = run_encoder(enc)
        r = results[enc]
        print(
            f"  verdict={r['verdict']} ratio={r['ratio_fast_over_plain']['mean']} "
            f"[{r['ratio_fast_over_plain']['lo']},{r['ratio_fast_over_plain']['hi']}] "
            f"fast_auc={r['auc_gain']['fast']['mean']} plain_auc={r['auc_gain']['plain']['mean']} "
            f"shuf_auc={r['auc_gain']['shuffle_control']['mean']} "
            f"acc0(f/p)={r['acc_at_0_updates']['fast_mean']}/{r['acc_at_0_updates']['plain_mean']} "
            f"final(f/p)={r['final_acc']['fast_mean']}/{r['final_acc']['plain_mean']}",
            flush=True,
        )

    payload = {
        "experiment": "axis_density_adaptation_per_update",
        "axis": "adaptation_per_update (capability GAIN per gradient UPDATE)",
        "task": "few-shot episodic linear shape-readout adaptation to a held-out (shape,color) split with fresh per-episode label permutation",
        "arms": {
            "fast": "Reptile-style meta-learned init (random-permutation episodes over meta-train colors), then fine-tune",
            "plain": "generic random-init head, fine-tune on same support (matched updates/LR/arch)",
        },
        "ratio_definition": "auc_gain(FAST)/auc_gain(PLAIN); auc_gain = mean_u (acc_u - acc_0)",
        "preregistered_thresholds": {
            "win_ratio_min": WIN_RATIO_MIN,
            "null_band": NULL_BAND,
            "loss_ratio_max": LOSS_RATIO_MAX,
            "shuffle_collapse_frac": SHUFFLE_COLLAPSE_FRAC,
            "rule": "WIN needs ratio_mean>1.15 AND ratio CI>1 AND consistent(+) AND AUC CIs non-overlap AND shuffle collapses; TIE (ratio CI spans 1, sign-flip, CI overlap) is NULL",
        },
        "tie_is_null": True,
        "results": results,
    }
    outp = OUT / "adaptation_update.json"
    outp.write_text(json.dumps(payload, indent=2))
    print(f"[wrote] {outp}", flush=True)


if __name__ == "__main__":
    main()
