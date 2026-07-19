"""Frontier temporal admission beds: E1 (relational temporal event formation) and C0 (trace stability).

These mechanisms are about sequence structure, not per-item selection, so they use a sequence-level harness:
per independent session the downstream utility of each arm is measured on an untouched stream, and the same
random-effects lower-95pct-CB decision rule as the battery is applied to mechanism-minus-best-simple-control.

Stream construct (externally ordered, not authored by the mechanism): real KMNIST images are emitted in runs
of the same class with geometric run lengths and intermittent observation noise. Ground-truth event boundaries
are the class-change indices, defined purely by the emission order. The downstream task is to classify the
current item from a context-pooled representation. Correct boundaries pool evidence within a class run and
average out noise; wrong boundaries pool across a change and corrupt the context. Independent units are
sessions (distinct class orders and seeds).

E1 must beat fixed windows, novelty thresholds, prediction-error thresholds, and a change-point detector using
a relational transition signal, and must beat shuffled and random boundaries. C0 (a stable persistent trace)
must beat direct state, last observation, EMA smoothing, and a matched-memory buffer.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, "/Users/scammermike/Downloads/mop/campaign2/lanes")
from n1_mnist_bed import SmallCNN  # noqa: E402

DATA = Path("/Users/scammermike/Downloads/mop/campaign2/data")
RUNS = Path("/Users/scammermike/Downloads/mop-scientific-frontier/runs/generation2/mop-generation2-scientific-frontier-v1")
RUNS.mkdir(parents=True, exist_ok=True)
_TF = transforms.ToTensor()
N_SESSIONS = 14
T_STEPS = 200


def _kmnist(train):
    return datasets.KMNIST(str(DATA), train=train, transform=_TF)


def _embed_pool(seed=1):
    """Train a small frozen CNN on KMNIST and cache per-class embedding pools + class centroids."""
    cache = RUNS / "temporal_embed_cache.npz"
    if cache.exists():
        z = np.load(cache)
        pools = {c: z[f"pool_{c}"] for c in range(10)}
        return pools, z["cents"]
    net = SmallCNN()
    ds = _kmnist(True)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3); net.train()
    for _ in range(2):
        for xb, yb in dl:
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval()
    test = _kmnist(False); targets = np.array(test.targets); rng = np.random.default_rng(seed)
    pools = {}
    with torch.no_grad():
        for c in range(10):
            idx = rng.choice(np.where(targets == c)[0], 400, replace=False)
            xb = torch.stack([test[i][0] for i in idx])
            pools[c] = net.features(xb).numpy().astype(np.float64)
    cents = np.stack([pools[c].mean(0) for c in range(10)])
    np.savez(cache, cents=cents, **{f"pool_{c}": pools[c] for c in range(10)})
    return pools, cents


def _make_session(pools, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(10)
    emb, labels, boundaries = [], [], []
    step = 0; oi = 0
    while step < T_STEPS:
        c = int(order[oi % 10]); oi += 1
        run = 1 + rng.geometric(0.28)
        boundaries.append(step)  # class change starts here
        for _ in range(run):
            if step >= T_STEPS:
                break
            v = pools[c][rng.integers(len(pools[c]))].copy()
            if rng.random() < 0.5:  # intermittent observation noise
                v = v + rng.normal(0, v.std() * 1.3, size=v.shape)
            emb.append(v); labels.append(c); step += 1
    return np.array(emb), np.array(labels), set(b for b in boundaries if 0 < b < T_STEPS)


def _classify_pooled(pooled, cents):
    d = np.linalg.norm(cents[None, :, :] - pooled[:, None, :], axis=2)
    return d.argmin(1)


# ---------- E1: event boundaries ----------
def _pool_by_boundaries(emb, bset):
    """Context pooling: at each step, mean of embeddings since the last boundary, concatenated with current."""
    T = len(emb); out = np.zeros_like(emb); start = 0
    for i in range(T):
        if i in bset:
            start = i
        out[i] = emb[start:i + 1].mean(0)
    return out


def _acc(pred, labels):
    return float((pred == labels).mean())


def e1_session_arms(emb, labels, true_b, seed, learned_thresh):
    """Return downstream accuracy per arm for one session."""
    T = len(emb)
    rng = np.random.default_rng(seed + 999)
    # relational signals
    trans = np.zeros(T)  # cosine drop between consecutive embeddings (relational transition surprise)
    for i in range(1, T):
        a, b = emb[i - 1], emb[i]
        trans[i] = 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    run_mean = np.zeros_like(emb); novelty = np.zeros(T); start = 0
    for i in range(T):
        run_mean[i] = emb[start:i + 1].mean(0)
        novelty[i] = np.linalg.norm(emb[i] - run_mean[i - 1]) if i > start else 0.0
        # context reset when local novelty high (used only to compute the novelty control's own segmentation)
    n_true = max(1, len(true_b))
    rate = n_true / T

    def thr_boundaries(sig, frac):
        k = max(1, int(round(rate * T)))
        idx = set(np.argsort(-sig)[:k].tolist()); idx.discard(0); return idx

    arms = {}
    # oracle
    arms["oracle"] = _acc(_classify_pooled(_pool_by_boundaries(emb, true_b), CENTS), labels)
    # none
    arms["none"] = _acc(_classify_pooled(_pool_by_boundaries(emb, set()), CENTS), labels)
    # fixed window (k tuned to mean run length)
    k = max(2, int(round(1 / rate)))
    arms["fixed_window"] = _acc(_classify_pooled(_pool_by_boundaries(emb, set(range(k, T, k))), CENTS), labels)
    # uniform + random rate-matched
    m = max(1, int(round(rate * T)))
    arms["uniform"] = _acc(_classify_pooled(_pool_by_boundaries(emb, set(np.linspace(1, T - 1, m).astype(int))), CENTS), labels)
    arms["random_rate_matched"] = _acc(_classify_pooled(_pool_by_boundaries(emb, set(rng.choice(range(1, T), m, replace=False).tolist())), CENTS), labels)
    # novelty threshold
    arms["novelty_threshold"] = _acc(_classify_pooled(_pool_by_boundaries(emb, thr_boundaries(novelty, rate)), CENTS), labels)
    # prediction-error threshold (predict next = current; error = distance)
    prederr = np.array([0.0] + [np.linalg.norm(emb[i] - emb[i - 1]) for i in range(1, T)])
    arms["prederr_threshold"] = _acc(_classify_pooled(_pool_by_boundaries(emb, thr_boundaries(prederr, rate)), CENTS), labels)
    # change-point (CUSUM on norm of running-mean shift)
    cusum = np.abs(np.gradient(np.linalg.norm(run_mean, axis=1)))
    arms["change_point"] = _acc(_classify_pooled(_pool_by_boundaries(emb, thr_boundaries(cusum, rate)), CENTS), labels)
    # E1 mechanism: relational transition surprise gated by context mismatch (learned threshold on trans*novelty)
    relscore = trans * (1 + novelty / (novelty.std() + 1e-9))
    e1_b = set(np.where(relscore > learned_thresh)[0].tolist()); e1_b.discard(0)
    if not e1_b:
        e1_b = thr_boundaries(relscore, rate)
    arms["E1"] = _acc(_classify_pooled(_pool_by_boundaries(emb, e1_b), CENTS), labels)
    # shuffled boundaries (same count, random times) and wrong-context handled at runner level
    arms["E1_shuffled"] = _acc(_classify_pooled(_pool_by_boundaries(emb, set(rng.choice(range(1, T), max(1, len(e1_b)), replace=False).tolist())), CENTS), labels)
    return arms


# ---------- C0: trace stability ----------
def c0_session_arms(emb, labels, seed):
    T = len(emb)
    rng = np.random.default_rng(seed + 555)
    arms = {}
    arms["oracle_stable_state"] = _acc(_classify_pooled(np.stack([CENTS[labels[i]] for i in range(T)]), CENTS), labels)
    arms["no_trace"] = _acc(_classify_pooled(emb, CENTS), labels)
    arms["direct_state"] = arms["no_trace"]
    lastobs = np.vstack([emb[0], emb[:-1]])
    arms["last_observation"] = _acc(_classify_pooled(lastobs, CENTS), labels)
    # EMA smoothing (simple trace)
    ema = np.zeros_like(emb); a = 0.3; ema[0] = emb[0]
    for i in range(1, T):
        ema[i] = a * emb[i] + (1 - a) * ema[i - 1]
    arms["ema_smoothing"] = _acc(_classify_pooled(ema, CENTS), labels)
    # matched-memory buffer (mean of last k)
    k = 5; buf = np.zeros_like(emb)
    for i in range(T):
        buf[i] = emb[max(0, i - k + 1):i + 1].mean(0)
    arms["matched_memory"] = _acc(_classify_pooled(buf, CENTS), labels)
    # random + shuffled traces
    arms["random_trace"] = _acc(_classify_pooled(rng.normal(size=emb.shape) * emb.std(), CENTS), labels)
    arms["shuffled_trace"] = _acc(_classify_pooled(emb[rng.permutation(T)], CENTS), labels)
    # C0 mechanism: confidence-weighted robust trace that resists outliers. Update trace only toward
    # observations close to the current trace (down-weight high-novelty/noisy frames), i.e., an attractor.
    trace = np.zeros_like(emb); trace[0] = emb[0]
    for i in range(1, T):
        prev = trace[i - 1]
        nov = np.linalg.norm(emb[i] - prev)
        w = float(np.exp(-nov / (np.linalg.norm(prev) + 1e-9)))  # trust observation inversely to novelty
        trace[i] = w * emb[i] + (1 - w) * prev
    arms["C0"] = _acc(_classify_pooled(trace, CENTS), labels)
    return arms


CENTS = None  # set by builder


def build_sessions():
    global CENTS
    pools, cents = _embed_pool()
    CENTS = cents
    sessions = [_make_session(pools, s) for s in range(N_SESSIONS)]
    return sessions
