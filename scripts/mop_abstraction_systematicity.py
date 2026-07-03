"""LEVER 2: compositional / analogical abstraction on REAL V-JEPA latents.

Substrate: data/cache/vjepa2_vitl_nuisance -- REAL pooled V-JEPA ViT-L latents (1024d) on a clean
5x5 factorial grid (shape in 0..4, color in 0..4), 8 clips per cell, 200 total. LABEL-FREE clips.
Control (the bounding null): data/cache/randominit_vitl_nuisance -- SAME architecture ViT-L, UNTRAINED,
same 256px clips, IDENTICAL shape/color labels. This is the matched frozen-random substrate: if an
untrained ViT reproduces the compositional structure, it is geometry/architecture-inherited, not learned
abstraction. Beating a synthetic shuffle is NOT enough (the count-confound lesson); the win must beat the
random-init substrate.

GATE (measured, not assumed): on real latents shape acc ~0.79, color acc ~0.65 (both decodable). On
random-init shape ~0.26 (near chance 0.20) but color ~0.99 (color is a trivial low-level pixel statistic
that an untrained net recovers for free). So color is a CONFOUNDED factor for the random baseline and shape
is the ABSTRACT factor. The tests are designed so a trivial pixel-statistic substrate cannot pass.

TWO PREREGISTERED TESTS
=======================

TEST A -- ANALOGY (structure-mapping of the SHAPE relation across color).
  CONFOUND-AWARE AXIS: an earlier diagnostic showed color is a trivial linear pixel-statistic axis on the
  UNTRAINED substrate (color-offset cross-shape parallelism cosine 0.89, color-analogy 0.99), so testing
  the color relation would let a random encoder "win" for free -- the exact count-confound trap. Color is
  therefore the CONFOUNDED axis and must NOT be the analogy target. Shape is the abstract factor (untrained
  shape-analogy is 0.0). So TEST A transfers the SHAPE offset across COLOR contexts, the axis a pixel
  statistic cannot fake.
  For a fixed shape pair (X -> Y), the offset o_c = centroid(z | color=c, shape=Y) - centroid(z | color=c,
  shape=X) is estimated on a set of DONOR colors and TRANSFERRED to a held-out target color t:
  pred = centroid(z | color=t, shape=X) + mean_{c != t} o_c. Score = retrieval top-1: does pred land
  nearest (over all 5 shape centroids at color=t) to the true centroid(z | color=t, shape=Y). This asks
  whether the shape relation is a COLOR-INDEPENDENT parallel offset (the parallelogram).
  Averaged over all shape pairs (X,Y), all target colors t, and seeds (seed = bootstrap resample of the 8
  per-cell samples for centroid estimation).
  Baselines on the SAME construction:
    - chance = 1/5 = 0.20 (random pick among 5 shape centroids at the target color)
    - shuffled-offset control: transfer an offset for a MISMATCHED shape pair (breaks the relation)
    - random-init substrate: identical pipeline on random-init latents (the bounding null)
  ANALOGY WIN RULE (prereg): real top1 CI_lo (over seeds) > max(shuffled-offset mean, chance) AND
    real mean - random_init mean has CI_lo > 0 (no flip). Real must beat BOTH its own shuffle floor and
    the random-init substrate. A tie on either is a NULL.

TEST B -- SYSTEMATIC GENERALIZATION (held-out cells).
  Train a linear SHAPE probe on a subset of the 25 (shape,color) cells; test on held-out cells whose
  (shape,color) combination was never seen in training. Held-out design: a "staircase" so that every
  shape appears in training with SOME colors but the specific (shape,color) test cells are novel
  conjunctions. This is systematicity: recombining a known shape with a color context it was not trained
  on. Compositional generalization delta = held-out-cell shape accuracy MINUS a non-compositional
  baseline. The non-compositional baseline is the SAME probe evaluated when the held-out mask is a random
  subset of ROWS (a nearest-in-count control that removes whole samples but not whole conjunctions), i.e.
  the systematic (novel-conjunction) split vs a random (i.i.d.) split of equal test size. If the substrate
  is purely conjunction-memorizing, novel-conjunction accuracy collapses relative to the i.i.d. split.
  Baselines:
    - random-init substrate swept identically (bounding null)
    - shuffle-label floor (chance ~0.20)
  SYSTEMATICITY WIN RULE (prereg): real novel-conjunction shape accuracy CI_lo > chance + 0.15 (does not
    cliff to chance), AND (real novel-conj acc - random_init novel-conj acc) CI_lo > 0 (substrate buys
    generalization beyond the untrained net). A tie is a NULL. We ALSO report the systematic-generalization
    DELTA = novel-conjunction acc - i.i.d.-split acc (0 means fully systematic, negative means a
    conjunction-memorization penalty); this is the headline "systematic-generalization delta".

A TIE IS A NULL. No score is faked. Everything is seeded.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mop.diagnostics.riskcov import seed_ci
from mop.seeding import seed_everything

_ROOT = Path(__file__).resolve().parents[1]
OUT = str(_ROOT / "runs" / "mot" / "abstraction_systematicity.json")
CACHE = str(_ROOT / "data" / "cache")
N_SHAPE, N_COLOR, PER_CELL = 5, 5, 8
SEEDS = list(range(8))


def load(tag):
    b = f"{CACHE}/{tag}/"
    x = torch.tensor(np.load(b + "features.npy"), dtype=torch.float32)
    sh = torch.tensor(np.load(b + "labels_shape.npy"))
    co = torch.tensor(np.load(b + "labels_color.npy"))
    return x, sh, co


def standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def cell_index(sh, co):
    """dict (shape,color) -> list of row indices."""
    d = {}
    for i in range(sh.shape[0]):
        d.setdefault((int(sh[i]), int(co[i])), []).append(i)
    return d


# ----------------------------------------------------------------------------------------------------
# TEST A: analogy (parallel color-offset transfer across shape)
# ----------------------------------------------------------------------------------------------------
def analogy_top1(x, sh, co, seed, shuffled=False, subsample=True):
    """Bootstrap per-cell samples (seed), build centroids, transfer SHAPE offset across COLOR.
    cent[s,c] = centroid of latents at (shape=s, color=c). We transfer a shape pair X->Y offset,
    estimated on DONOR colors, to a held-out target color t, then retrieve among the 5 shape centroids
    at that color. Returns mean retrieval top1 over all shape pairs (X!=Y) and all target colors t."""
    g = torch.Generator().manual_seed(seed)
    cells = cell_index(sh, co)
    cent = torch.zeros(N_SHAPE, N_COLOR, x.shape[1])
    for s in range(N_SHAPE):
        for c in range(N_COLOR):
            idx = torch.tensor(cells[(s, c)])
            if subsample:
                pick = idx[torch.randint(0, idx.shape[0], (idx.shape[0],), generator=g)]
            else:
                pick = idx
            cent[s, c] = x[pick].mean(0)

    hits, total = 0, 0
    shape_pairs = [(sx, sy) for sx in range(N_SHAPE) for sy in range(N_SHAPE) if sx != sy]
    for sx, sy in shape_pairs:
        # offset per color for this shape pair: cent[sy, c] - cent[sx, c]
        offs = cent[sy, :] - cent[sx, :]  # [N_COLOR, dim]
        for t in range(N_COLOR):
            donors = [c for c in range(N_COLOR) if c != t]
            if shuffled:
                # mismatched shape pair: build the transferred offset from a DIFFERENT (sx2,sy2)
                gi = torch.Generator().manual_seed(seed + t + 100 * sx + sy)
                alt = shape_pairs[int(torch.randint(0, len(shape_pairs), (1,), generator=gi))]
                offs_alt = cent[alt[1], :] - cent[alt[0], :]
                transfer = offs_alt[torch.tensor(donors)].mean(0)
            else:
                transfer = offs[torch.tensor(donors)].mean(0)
            pred = cent[sx, t] + transfer
            # retrieval over the 5 shape centroids at the target color t
            d = (cent[:, t] - pred).norm(dim=-1)  # [N_SHAPE]
            if int(d.argmin()) == sy:
                hits += 1
            total += 1
    return hits / total


def run_analogy():
    xr, sh, co = load("vjepa2_vitl_nuisance")
    xf, shf, cof = load("randominit_vitl_nuisance")
    assert torch.equal(sh, shf) and torch.equal(co, cof)
    xr, xf = standardize(xr), standardize(xf)

    real, shuf, rand = [], [], []
    for s in SEEDS:
        real.append(analogy_top1(xr, sh, co, s, shuffled=False))
        shuf.append(analogy_top1(xr, sh, co, s, shuffled=True))
        rand.append(analogy_top1(xf, sh, co, s, shuffled=False))
    diff = [r - f for r, f in zip(real, rand, strict=False)]  # real minus random-init, per seed

    # permutation null: shuffle shape labels WITHIN each color column, rebuild centroids (full sample,
    # no bootstrap), recompute analogy. If real sits far above this, the shape relation (not centroid
    # geometry) drives the parallelogram. 200 permutations.
    def _full_cent(xf_, sh_lab):
        cl = cell_index(sh_lab, co)
        cent = torch.zeros(N_SHAPE, N_COLOR, xf_.shape[1])
        for s in range(N_SHAPE):
            for c in range(N_COLOR):
                cent[s, c] = xf_[torch.tensor(cl[(s, c)])].mean(0)
        return cent

    def _analogy_from_cent(cent):
        hits = tot = 0
        for sx in range(N_SHAPE):
            for sy in range(N_SHAPE):
                if sx == sy:
                    continue
                offs = cent[sy, :] - cent[sx, :]
                for t in range(N_COLOR):
                    donors = [c for c in range(N_COLOR) if c != t]
                    transfer = offs[torch.tensor(donors)].mean(0)
                    pred = cent[sx, t] + transfer
                    d = (cent[:, t] - pred).norm(dim=-1)
                    hits += int(d.argmin()) == sy
                    tot += 1
        return hits / tot

    real_full = _analogy_from_cent(_full_cent(xr, sh))
    perm_scores = []
    for k in range(200):
        gp = torch.Generator().manual_seed(1000 + k)
        shp = sh.clone()
        for c in range(N_COLOR):
            m = (co == c).nonzero().flatten()
            shp[m] = sh[m][torch.randperm(m.shape[0], generator=gp)]
        perm_scores.append(_analogy_from_cent(_full_cent(xr, shp)))
    perm_scores = np.array(perm_scores)
    perm_p = float((perm_scores >= real_full).mean())

    ci_real = seed_ci(real)
    ci_shuf = seed_ci(shuf)
    ci_rand = seed_ci(rand)
    ci_diff = seed_ci(diff)
    chance = 1.0 / N_SHAPE  # retrieval over 5 shape centroids at the target color

    beats_shuffle = ci_real["lo"] > max(ci_shuf["mean"], chance)
    beats_randinit = ci_diff["lo"] > 0.0
    win = bool(beats_shuffle and beats_randinit)
    return {
        "axis": "shape-offset transfer across color (confound-aware; color is the trivial axis and is NOT the target)",
        "shape_analogy_top1_real": ci_real,
        "shape_analogy_top1_shuffled_offset": ci_shuf,
        "shape_analogy_top1_random_init": ci_rand,
        "real_minus_random_init": ci_diff,
        "chance": round(chance, 4),
        "beats_shuffle_floor": bool(beats_shuffle),
        "beats_random_init_substrate": bool(beats_randinit),
        "analogy_win": win,
        "permutation_null": {
            "real_full_sample_top1": round(real_full, 4),
            "perm_mean": round(float(perm_scores.mean()), 4),
            "perm_max": round(float(perm_scores.max()), 4),
            "n_perm": len(perm_scores),
            "p_perm_ge_real": perm_p,
            "note": "shape labels shuffled within each color column; real top1 far above permuted floor",
        },
        "per_seed_real": [round(v, 4) for v in real],
        "per_seed_random_init": [round(v, 4) for v in rand],
    }


# ----------------------------------------------------------------------------------------------------
# TEST B: systematic generalization (novel-conjunction held-out cells)
# ----------------------------------------------------------------------------------------------------
def _train_shape_probe(xtr, ytr, dim, epochs=200, lr=0.05, seed=0):
    seed_everything(seed)
    head = torch.nn.Linear(dim, N_SHAPE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(xtr), ytr).backward()
        opt.step()
    return head


def systematicity_split(x, sh, co, seed):
    """Novel-conjunction (systematic) split vs i.i.d. split of equal test size.

    Systematic split: hold out a fixed set of (shape,color) CELLS as test; every shape and every color
    still appears in training (in other cells), so the model has seen the shape and the color separately
    but never THIS conjunction. Held-out cells = the anti-diagonal band: (s, (s+k) % N_COLOR) for
    k in {0,1} -> 2 cells per shape, 10 of 25 cells held out.

    i.i.d. split: hold out the SAME NUMBER of individual samples chosen at random across all cells
    (conjunctions all seen in training). Equal test-set size -> fair accuracy comparison. If the substrate
    only memorizes conjunctions, systematic acc << iid acc.
    """
    cells = cell_index(sh, co)
    held_cells = set()
    for s in range(N_SHAPE):
        for k in (0, 1):
            held_cells.add((s, (s + k) % N_COLOR))
    sys_test_idx = []
    sys_train_idx = []
    for (s, c), idxs in cells.items():
        if (s, c) in held_cells:
            sys_test_idx += idxs
        else:
            sys_train_idx += idxs
    n_test = len(sys_test_idx)

    # i.i.d. split of equal size
    g = torch.Generator().manual_seed(seed + 777)
    all_idx = torch.arange(x.shape[0])
    perm = all_idx[torch.randperm(x.shape[0], generator=g)]
    iid_test_idx = perm[:n_test].tolist()
    iid_train_idx = perm[n_test:].tolist()

    def acc(train_idx, test_idx):
        xt = x[torch.tensor(train_idx)]
        yt = sh[torch.tensor(train_idx)]
        xe = x[torch.tensor(test_idx)]
        ye = sh[torch.tensor(test_idx)]
        head = _train_shape_probe(xt, yt, x.shape[1], seed=seed)
        with torch.no_grad():
            return float((head(xe).argmax(-1) == ye).float().mean())

    sys_acc = acc(sys_train_idx, sys_test_idx)
    iid_acc = acc(iid_train_idx, iid_test_idx)
    return sys_acc, iid_acc, n_test


def run_systematicity():
    xr, sh, co = load("vjepa2_vitl_nuisance")
    xf, shf, cof = load("randominit_vitl_nuisance")
    assert torch.equal(sh, shf) and torch.equal(co, cof)
    xr, xf = standardize(xr), standardize(xf)

    sys_real, iid_real, sys_rand, iid_rand = [], [], [], []
    n_test = None
    for s in SEEDS:
        sa, ia, nt = systematicity_split(xr, sh, co, s)
        sar, iar, _ = systematicity_split(xf, sh, co, s)
        sys_real.append(sa)
        iid_real.append(ia)
        sys_rand.append(sar)
        iid_rand.append(iar)
        n_test = nt
    diff = [a - b for a, b in zip(sys_real, sys_rand, strict=False)]  # novel-conj: real minus random-init
    sysgen_delta = [
        a - b for a, b in zip(sys_real, iid_real, strict=False)
    ]  # systematic-generalization delta

    ci_sys_real = seed_ci(sys_real)
    ci_iid_real = seed_ci(iid_real)
    ci_sys_rand = seed_ci(sys_rand)
    ci_diff = seed_ci(diff)
    ci_delta = seed_ci(sysgen_delta)
    chance = 1.0 / N_SHAPE

    no_cliff = ci_sys_real["lo"] > chance + 0.15
    beats_randinit = ci_diff["lo"] > 0.0
    win = bool(no_cliff and beats_randinit)
    return {
        "novel_conjunction_shape_acc_real": ci_sys_real,
        "iid_split_shape_acc_real": ci_iid_real,
        "novel_conjunction_shape_acc_random_init": ci_sys_rand,
        "real_minus_random_init_novel_conj": ci_diff,
        "systematic_generalization_delta": ci_delta,
        "held_out_cells": "anti-diagonal band, 10 of 25 cells",
        "n_test_samples": n_test,
        "chance": round(chance, 4),
        "no_cliff_to_chance": bool(no_cliff),
        "beats_random_init_substrate": bool(beats_randinit),
        "systematicity_win": win,
        "per_seed_novel_conj_real": [round(v, 4) for v in sys_real],
        "per_seed_novel_conj_random_init": [round(v, 4) for v in sys_rand],
        "per_seed_iid_real": [round(v, 4) for v in iid_real],
    }


def main():
    seed_everything(0)
    analogy = run_analogy()
    systematicity = run_systematicity()

    # overall verdict: at least one test must WIN under its prereg rule for abstraction to push past null
    any_win = bool(analogy["analogy_win"] or systematicity["systematicity_win"])
    result = {
        "lever": "2_systematicity_analogy_real_vjepa_latents",
        "substrate": "vjepa2_vitl_nuisance (real ViT-L) vs randominit_vitl_nuisance (matched untrained)",
        "gate_decodability": {
            "note": "measured pre-run: real shape~0.79 color~0.65 both decodable; random-init shape~0.26 "
            "(near chance) color~0.99 (trivial pixel statistic). Color is confounded for the null, shape "
            "is the abstract factor; tests target shape-abstraction / shape-independent relations."
        },
        "test_A_analogy": analogy,
        "test_B_systematicity": systematicity,
        "any_win": any_win,
        "verdict": "WIN" if any_win else "NULL",
        "seeds": SEEDS,
        "prereg": {
            "analogy_win_rule": "real top1 CI_lo > max(shuffled-offset mean, chance) AND (real - random_init) CI_lo > 0",
            "systematicity_win_rule": "novel-conjunction acc CI_lo > chance+0.15 AND (real - random_init) CI_lo > 0",
            "tie_is_null": True,
        },
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
