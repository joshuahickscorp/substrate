"""AXIS 4 / ABSTRACTION, SUB-FORM 3: CROSS-SUBSTRATE analogy transfer.

Established WIN (not rehashed here): within V-JEPA, the SHAPE-offset parallelogram is systematic and
analogical (offset transfers across color, novel conjunctions generalize, an untrained ViT collapses).

THIS TEST (the stronger, new claim): is that analogical structure SUBSTRATE-INVARIANT? Does a shape
offset expressed in encoder A's latent space, carried through a SHARED linear map A -> B, predict the
correct shape analogy in a DIFFERENT real encoder B (different architecture, different pretraining)?
If yes beyond what two random-init encoders share through the identical pipeline, the abstraction is
encoder-independent, a stronger positive than the within-encoder win.

SUBSTRATES (all four share the IDENTICAL clip order, verified by checksum_first, so row i is the same
clip in every cache; this is what makes a shared row-paired alignment map legitimate):
  A_real = vjepa2_vitl_nuisance          (REAL V-JEPA ViT-L, 1024d)
  B_real = dinov2s_nuisance_real         (REAL DINOv2 ViT-S, 384d)  -- different arch AND pretraining
  A_rand = randominit_vitl_nuisance      (UNTRAINED ViT-L, 1024d, matched to A_real)
  B_rand = dinov2s_nuisance_randominit   (UNTRAINED DINOv2 ViT-S, 384d, matched to B_real)

The control is the RANDOM<->RANDOM pair through the same pipeline. Beating a within-space shuffle is NOT
enough (the count-confound lesson). The bounding null is the matched random-init substrate pair: if two
UNTRAINED encoders already share the cross-substrate shape analogy through the shared map, the apparent
transfer is geometry/architecture the alignment map buys for free, not learned abstraction.

CONFOUND GUARD (the systematicity lesson): color is a trivial low-level pixel statistic an untrained net
recovers for free, so we transfer the SHAPE offset ACROSS COLOR (shape is the abstract factor an untrained
net collapses on). Testing the color relation would let a random pair win for free.

CONSTRUCTION (preregistered in code before any result exists):
  1. Split the 200 rows into an ALIGN-TRAIN set and an EVAL set (row-paired across A and B; same rows
     used in both encoders since the clip order is shared). Standardize each encoder on its own
     align-train stats.
  2. Fit a SHARED linear map W: A -> B by ridge on the ALIGN-TRAIN rows (the al2 ridge_fit primitive),
     rank-truncated to a preregistered rank. This map is fit WITHOUT any shape/color labels; it only sees
     row-paired latents. It is the shared coordinate system.
  3. CROSS-ENCODER ANALOGY on the EVAL rows. Build per-cell centroids in A's space (centA[s,c]) and in
     B's space (centB[s,c]) from EVAL rows only, bootstrap-resampled. For a shape pair (X -> Y):
        offsetA_c   = centA[Y,c] - centA[X,c]                      (shape offset in A, per color)
        transferA   = mean over donor colors c != t of offsetA_c   (color-independent shape offset in A)
        predA       = centA[X,t] + transferA                       (predicted shape-Y centroid in A)
        predB       = standardize_and_map(predA) through W         (carried into B's space)
        retrieve    = argmin over the 5 shape centroids centB[:,t] of || centB[s,t] - predB ||
     Score = top-1 retrieval accuracy: does the A-space shape offset, mapped into B, land nearest to the
     TRUE shape-Y centroid in B at the held-out color t. Averaged over all shape pairs (X != Y), all
     target colors t, and seeds. This is the analogy expressed in one encoder and READ OUT in another.
  4. FLOORS on the SAME construction:
       chance          = 1/5 = 0.20 (random pick among 5 shape centroids in B at color t)
       shuffled-offset = transfer a MISMATCHED shape pair's A-offset through W (breaks the relation)
       random<->random = the identical pipeline with A_rand -> B_rand (the bounding null)
  We also record the SAME-ENCODER direction B -> A (dinov2 offset read out in v-jepa) as a symmetry check.

WIN RULE (preregistered, a TIE IS A NULL):
  real cross-encoder analogy top1 CI_lo (over seeds) > max(shuffled-offset mean, chance)
  AND (real mean - random<->random mean) CI_lo > 0 with no per-seed sign flip.
  Real must beat BOTH its own shuffle floor and the random-init<->random-init pair. A tie on either is a
  NULL. No score is faked. Everything is seeded; the align split and the centroid bootstrap both reseed.

No em dashes or en dashes (house rule).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))

from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402

CACHE = _ROOT / "data" / "cache"
OUT = _ROOT / "runs" / "mot" / "abstraction_cross_substrate.json"
N_SHAPE, N_COLOR, PER_CELL = 5, 5, 8
SEEDS = list(range(10))
RANK = 32  # preregistered alignment-map rank (primary rank in the al2 pilot)
RIDGE_LAMBDA = 1.0
ALIGN_TRAIN_FRAC = 0.6  # rows used to FIT the shared map; the rest are EVAL rows for the analogy
CHANCE = 1.0 / N_SHAPE

A_REAL, B_REAL = "vjepa2_vitl_nuisance", "dinov2s_nuisance_real"
A_RAND, B_RAND = "randominit_vitl_nuisance", "dinov2s_nuisance_randominit"


# ----------------------------------------------------------------------------------------------------
# loading (the two cache layouts: vjepa uses features/labels_*.npy, dino uses latents.npy + factors.json)
# ----------------------------------------------------------------------------------------------------
def load(tag):
    b = CACHE / tag
    if (b / "features.npy").exists():
        x = np.load(b / "features.npy")
        sh = np.load(b / "labels_shape.npy")
        co = np.load(b / "labels_color.npy")
    else:
        x = np.load(b / "latents.npy")
        with open(b / "factors.json") as _fh:
            fj = json.load(_fh)
        sh = np.array(fj["shape"])
        co = np.array(fj["color"])
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(np.asarray(sh)),
        torch.tensor(np.asarray(co)),
    )


def ridge_fit(xa, xb, lam=RIDGE_LAMBDA):
    """W = argmin ||xa W - xb||^2 + lam ||W||^2 (al2 primitive). [Da, Db]."""
    da = xa.shape[1]
    return torch.linalg.solve(xa.T @ xa + lam * torch.eye(da), xa.T @ xb)


def rank_truncate(w, k):
    if k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :k] @ torch.diag(s[:k]) @ vh[:k]


def cell_index(sh, co, rows):
    """dict (shape,color) -> tensor of row indices, restricted to `rows`."""
    d = {}
    for i in rows.tolist():
        d.setdefault((int(sh[i]), int(co[i])), []).append(i)
    return {k: torch.tensor(v) for k, v in d.items()}


def centroids(x, cells, seed, subsample=True):
    """cent[s,c] = (bootstrap) centroid of latents at (shape=s, color=c) over the given rows."""
    g = torch.Generator().manual_seed(seed)
    cent = torch.zeros(N_SHAPE, N_COLOR, x.shape[1])
    for s in range(N_SHAPE):
        for c in range(N_COLOR):
            idx = cells[(s, c)]
            if subsample:
                pick = idx[torch.randint(0, idx.shape[0], (idx.shape[0],), generator=g)]
            else:
                pick = idx
            cent[s, c] = x[pick].mean(0)
    return cent


def standardize_fit(x_train):
    return x_train.mean(0), x_train.std(0) + 1e-6


# ----------------------------------------------------------------------------------------------------
# one seed: fit shared map A->B on align-train rows, run cross-encoder analogy on eval rows
# ----------------------------------------------------------------------------------------------------
def cross_encoder_analogy(xa, xb, sh, co, seed, shuffled=False):
    """Fit W: A -> B on align-train rows, then express the SHAPE offset in A, map it into B, and
    retrieve among B's shape centroids. Returns mean top-1 over all shape pairs (X!=Y) and target
    colors t. `shuffled` transfers a MISMATCHED shape pair's A-offset (breaks the relation)."""
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    perm = torch.randperm(n, generator=g)
    cut = int(n * ALIGN_TRAIN_FRAC)
    align_rows, eval_rows = perm[:cut], perm[cut:]

    # standardize each encoder on its OWN align-train rows (map is fit in standardized space)
    mua, sda = standardize_fit(xa[align_rows])
    mub, sdb = standardize_fit(xb[align_rows])
    za = (xa - mua) / sda
    zb = (xb - mub) / sdb

    # shared linear map fit on row-paired align rows, label-free
    w = rank_truncate(ridge_fit(za[align_rows], zb[align_rows]), RANK)

    # centroids from EVAL rows only (bootstrap-resampled), in each encoder's standardized space
    cellsE = cell_index(sh, co, eval_rows)
    # guard: every (shape,color) cell must have >=1 eval sample; if not, this seed is skipped upstream
    centA = centroids(za, cellsE, seed)
    centB = centroids(zb, cellsE, seed)

    shape_pairs = [(sx, sy) for sx in range(N_SHAPE) for sy in range(N_SHAPE) if sx != sy]
    hits, total = 0, 0
    for sx, sy in shape_pairs:
        offsA = centA[sy, :] - centA[sx, :]  # [N_COLOR, Da] shape offset in A per color
        for t in range(N_COLOR):
            donors = torch.tensor([c for c in range(N_COLOR) if c != t])
            if shuffled:
                gi = torch.Generator().manual_seed(seed + t + 100 * sx + 7 * sy)
                alt = shape_pairs[int(torch.randint(0, len(shape_pairs), (1,), generator=gi))]
                offsA_use = centA[alt[1], :] - centA[alt[0], :]
            else:
                offsA_use = offsA
            transferA = offsA_use[donors].mean(0)  # color-independent shape offset in A
            predA = centA[sx, t] + transferA  # predicted shape-Y centroid in A's space
            predB = predA @ w  # carried into B's space through the shared map
            d = (centB[:, t] - predB).norm(dim=-1)  # retrieve among 5 shape centroids in B at color t
            if int(d.argmin()) == sy:
                hits += 1
            total += 1
    return hits / total


def eval_cells_complete(sh, co, seed):
    """Return True iff, for this seed's align/eval split, every (shape,color) cell has >=1 eval row in
    BOTH the A and B grids. The grids share row order so one check suffices. Prevents an empty-centroid
    seed from silently biasing the score."""
    g = torch.Generator().manual_seed(seed)
    n = sh.shape[0]
    perm = torch.randperm(n, generator=g)
    eval_rows = perm[int(n * ALIGN_TRAIN_FRAC) :]
    cells = cell_index(sh, co, eval_rows)
    return all((s, c) in cells for s in range(N_SHAPE) for c in range(N_COLOR))


def run_pair(a_tag, b_tag, label):
    xa, sh, co = load(a_tag)
    xb, sh_b, co_b = load(b_tag)
    assert torch.equal(sh, sh_b) and torch.equal(co, co_b), "grids not row-aligned across encoders"

    reals, shufs, _deltas_placeholder = [], [], []
    used_seeds = []
    for seed in SEEDS:
        if not eval_cells_complete(sh, co, seed):
            continue
        used_seeds.append(seed)
        reals.append(cross_encoder_analogy(xa, xb, sh, co, seed, shuffled=False))
        shufs.append(cross_encoder_analogy(xa, xb, sh, co, seed, shuffled=True))
    return {
        "label": label,
        "a": a_tag,
        "b": b_tag,
        "seeds_used": used_seeds,
        "real_per_seed": [round(v, 4) for v in reals],
        "shuffled_per_seed": [round(v, 4) for v in shufs],
        "real_ci": seed_ci(reals),
        "shuffled_ci": seed_ci(shufs),
    }


def main():
    print("cross-substrate analogy: real V-JEPA <-> real DINOv2 vs random <-> random")
    real_fwd = run_pair(A_REAL, B_REAL, "real: vjepa->dino")
    rand_fwd = run_pair(A_RAND, B_RAND, "rand: vjepa->dino")
    # symmetry / reverse direction as a secondary read (dino offset read out in v-jepa)
    real_rev = run_pair(B_REAL, A_REAL, "real: dino->vjepa")
    rand_rev = run_pair(B_RAND, A_RAND, "rand: dino->vjepa")

    # forward direction is the headline. delta vs random<->random, per matched seed.
    matched = sorted(set(real_fwd["seeds_used"]) & set(rand_fwd["seeds_used"]))
    rmap = dict(zip(real_fwd["seeds_used"], real_fwd["real_per_seed"], strict=False))
    cmap = dict(zip(rand_fwd["seeds_used"], rand_fwd["real_per_seed"], strict=False))
    deltas = [rmap[s] - cmap[s] for s in matched]
    delta_ci = seed_ci(deltas)
    flip = sign_flip_report(deltas)

    real_ci = real_fwd["real_ci"]
    shuf_mean = real_fwd["shuffled_ci"]["mean"]
    floor = max(shuf_mean, CHANCE)

    beats_own_floor = real_ci["lo"] > floor
    beats_random_pair = (delta_ci["lo"] > 0) and (not flip["any_flip"])
    win = bool(beats_own_floor and beats_random_pair)

    if win:
        verdict = "WIN: cross-substrate shape analogy transfers beyond the random<->random control"
    elif abs(delta_ci["mean"]) < 1e-9 or delta_ci["lo"] <= 0:
        verdict = "NULL: cross-substrate transfer does not beat the random-init<->random-init pair"
    else:
        verdict = "NULL"
    if not beats_own_floor:
        verdict = "NULL: real cross-encoder transfer does not clear its own shuffle/chance floor"

    result = {
        "experiment": "axis4_abstraction_subform3_cross_substrate_analogy",
        "claim": "the shape-offset analogy is substrate-invariant: a shape offset in encoder A, mapped "
        "through a shared label-free linear map, predicts the shape analogy in a different real encoder B, "
        "beyond what two matched random-init encoders share through the identical pipeline",
        "prereg": {
            "win_rule": "real forward top1 CI_lo > max(shuffled mean, chance=0.20) AND "
            "(real - random<->random) delta CI_lo > 0 with no sign flip; a tie is a NULL",
            "rank": RANK,
            "align_train_frac": ALIGN_TRAIN_FRAC,
            "seeds": SEEDS,
            "confound_guard": "transfer SHAPE offset across COLOR (shape is the abstract factor; color is "
            "the trivial pixel statistic a random net wins for free)",
        },
        "chance": CHANCE,
        "forward_real": real_fwd,
        "forward_random_control": rand_fwd,
        "reverse_real": real_rev,
        "reverse_random_control": rand_rev,
        "headline": {
            "real_forward_ci": real_ci,
            "real_forward_shuffled_mean": round(shuf_mean, 4),
            "own_floor_used": round(floor, 4),
            "beats_own_floor": bool(beats_own_floor),
            "random_control_forward_ci": rand_fwd["real_ci"],
            "delta_real_minus_random": delta_ci,
            "delta_sign_flip": flip,
            "matched_seeds": matched,
            "beats_random_pair": bool(beats_random_pair),
        },
        "win": win,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print(f"\nreal forward   : mean {real_ci['mean']}  CI [{real_ci['lo']}, {real_ci['hi']}]")
    print(f"real shuffled  : mean {shuf_mean}   own floor (max shuf,chance) {floor}")
    print(
        f"random control : mean {rand_fwd['real_ci']['mean']}  CI "
        f"[{rand_fwd['real_ci']['lo']}, {rand_fwd['real_ci']['hi']}]"
    )
    print(
        f"delta real-rand: mean {delta_ci['mean']}  CI [{delta_ci['lo']}, {delta_ci['hi']}]  "
        f"flip {flip['any_flip']}"
    )
    print(f"reverse real   : mean {real_rev['real_ci']['mean']}  reverse rand {rand_rev['real_ci']['mean']}")
    print(f"\nbeats own floor {beats_own_floor}  beats random pair {beats_random_pair}")
    print("VERDICT:", verdict)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
