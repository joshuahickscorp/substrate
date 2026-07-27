"""C1: P1R replay value as a soft SAMPLING PRIORITY over a fixed, independently-maintained buffer.

The buffer (GDumb class-balanced, or reservoir for the buffer-factor arm) is maintained independently of the
priority estimator; the priority estimator never influences which items enter or leave the buffer. The learned
P1R priority fits the validated Gen2 replay-value predictor over decision-time features [softmax probs, entropy,
feature-norm] (NOT raw loss, so it is distinct from the loss-priority control), applies the validated
toxic-value gate, and transforms predictions to a sealed priority distribution used to sample replay minibatches.

Sealed priority transform: priority_i = softmax(zscore(value_i) / TEMPERATURE) with a minimum-probability floor
MINP * uniform, renormalized; toxic-gated items get floor probability; recomputed once per task; cold-start
(empty buffer) does no replay; ties fall back to uniform. A tie is a null.

Two sources: EMNIST-balanced (image canary) and UCI HAR (subject-disjoint activity recognition, the non-image
principal source). House style: no dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, "/Users/scammermike/Downloads/mop/salvage/lanes2")
from estimators import select_and_fit  # noqa: E402

DATA_IMG = Path("/Users/scammermike/Downloads/mop/campaign2/data")
HAR = Path("/Users/scammermike/Downloads/mop-gen3/runs/generation3/mop-generation3-discovery-v1/data/UCI HAR Dataset")
_TF = transforms.ToTensor()

# sealed priority-transform parameters
TEMPERATURE = 0.5
MINP = 0.2          # floor = 0.2 * uniform
TOXIC_Q_FN, TOXIC_Q_R = 0.75, 0.2
MEM = 600
BATCH = 128
_CACHE = {}


# ---------------- models ----------------
class EmnistCNN(nn.Module):
    def __init__(self, n_out=47):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1); self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 128); self.head = nn.Linear(128, n_out)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


class HarMLP(nn.Module):
    def __init__(self, n_in=561, n_out=6):
        super().__init__()
        self.f1 = nn.Linear(n_in, 128); self.f2 = nn.Linear(128, 64); self.head = nn.Linear(64, n_out)

    def features(self, x):
        return F.relu(self.f2(F.relu(self.f1(x))))

    def forward(self, x):
        return self.head(self.features(x))


# ---------------- data ----------------
def prep_emnist(seed, n_tasks=9, per_class_tr=400, per_class_te=160, cpt=5):
    key = ("emnist", seed)
    if key in _CACHE:
        return _CACHE[key]
    tr = datasets.EMNIST(str(DATA_IMG), split="balanced", train=True, transform=_TF)
    te = datasets.EMNIST(str(DATA_IMG), split="balanced", train=False, transform=_TF)
    ytr = np.array(tr.targets); yte = np.array(te.targets); rng = np.random.default_rng(seed)
    classes = list(range(n_tasks * cpt)); tasks = [classes[i * cpt:(i + 1) * cpt] for i in range(n_tasks)]
    trp, tep = {}, {}
    for t, cs in enumerate(tasks):
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], per_class_tr, replace=False) for c in cs])
        xte = np.concatenate([rng.choice(np.where(yte == c)[0], per_class_te, replace=False) for c in cs])
        trp[t] = (torch.stack([tr[i][0] for i in idx]), torch.tensor([tr[i][1] for i in idx]))
        tep[t] = (torch.stack([te[i][0] for i in xte]), torch.tensor([te[i][1] for i in xte]))
    _CACHE[key] = (tasks, trp, tep, "img", 47); return _CACHE[key]


def prep_har(seed, cpt=2):
    key = ("har", seed)
    if key in _CACHE:
        return _CACHE[key]
    Xtr = np.loadtxt(HAR / "train/X_train.txt").astype(np.float32); ytr = np.loadtxt(HAR / "train/y_train.txt").astype(int) - 1
    Xte = np.loadtxt(HAR / "test/X_test.txt").astype(np.float32); yte = np.loadtxt(HAR / "test/y_test.txt").astype(int) - 1
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    rng = np.random.default_rng(seed)
    order = rng.permutation(6); tasks = [list(order[i * cpt:(i + 1) * cpt]) for i in range(6 // cpt)]
    trp, tep = {}, {}
    for t, cs in enumerate(tasks):
        itr = np.concatenate([np.where(ytr == c)[0] for c in cs]); ite = np.concatenate([np.where(yte == c)[0] for c in cs])
        trp[t] = (torch.tensor(Xtr[itr]), torch.tensor(ytr[itr]))
        tep[t] = (torch.tensor(Xte[ite]), torch.tensor(yte[ite]))
    _CACHE[key] = (tasks, trp, tep, "har", 6); return _CACHE[key]


# ---------------- buffer + priority ----------------
def gdumb_buffer(net, memx, memy, newx, newy, rng, kind):
    allx = torch.cat([memx, newx]) if len(memx) else newx
    ally = torch.cat([memy, newy]) if len(memy) else newy
    n = len(allx)
    if n <= MEM:
        return allx, ally
    classes = torch.unique(ally).tolist(); per = max(1, MEM // len(classes)); keep = []; an = ally.numpy()
    for c in classes:
        idx = np.where(an == c)[0]; keep.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
    keep = np.array(keep[:MEM]); return allx[keep], ally[keep]


def reservoir_buffer(net, memx, memy, newx, newy, rng, kind, seen):
    if len(memx) + len(newx) <= MEM:
        return (torch.cat([memx, newx]) if len(memx) else newx), (torch.cat([memy, newy]) if len(memy) else newy)
    bx = list(memx); by = list(memy.tolist()) if len(memy) else []
    for i in range(len(newx)):
        seen[0] += 1
        if len(bx) < MEM:
            bx.append(newx[i]); by.append(int(newy[i]))
        else:
            j = int(rng.integers(0, seen[0]))
            if j < MEM:
                bx[j] = newx[i]; by[j] = int(newy[i])
    return torch.stack(bx), torch.tensor(by)


@torch.no_grad()
def _feats(net, x):
    logits = torch.cat([net(x[s:s + 512]) for s in range(0, len(x), 512)])
    prob = F.softmax(logits, 1)
    ce = F.cross_entropy(logits, torch.zeros(len(x), dtype=torch.long), reduction="none")  # placeholder replaced below
    return logits, prob


def priority_over_buffer(net, memx, memy, policy, estimator, rng):
    """Return a sampling probability vector over the buffer per the policy (sealed transform)."""
    n = len(memx)
    if n == 0:
        return None
    with torch.no_grad():
        logits = torch.cat([net(memx[s:s + 512]) for s in range(0, n, 512)])
        prob = F.softmax(logits, 1)
        ce = F.cross_entropy(logits, memy, reduction="none").numpy()
        fnorm = torch.cat([net.features(memx[s:s + 512]) for s in range(0, n, 512)]).norm(dim=1).numpy()
    ent = -(prob * torch.log(prob + 1e-9)).sum(1).numpy()
    maxp = prob.max(1).values.numpy()
    uniform = np.ones(n) / n
    if policy == "uniform" or policy == "class_balanced_priority":
        return uniform
    if policy == "loss_priority":
        val = ce - ce.min()
    elif policy == "difficulty_priority":
        val = (1 - maxp)
    elif policy == "recency_priority":
        val = np.linspace(0.2, 1.0, n)
    elif policy == "random_rate_matched":
        val = rng.random(n)
    elif policy == "oracle_priority":
        val = ce - ce.min()
        for c in np.unique(memy.numpy()):
            m = memy.numpy() == c
            val[m] = val[m] / (val[m].sum() + 1e-9)
    elif policy.startswith("learned_p1r") or policy == "shuffled_p1r":
        feats = np.concatenate([prob.numpy(), ent[:, None], fnorm[:, None]], 1).astype(np.float64)
        r = (ce - ce.min()).astype(np.float64)
        tv = (fnorm > np.quantile(fnorm, TOXIC_Q_FN)) & (r < np.quantile(r, TOXIC_Q_R))
        m = min(n, 900); sub = rng.permutation(n)[:m]; half = m // 2
        pred = select_and_fit(estimator, feats[sub[:half]], r[sub[:half]], feats[sub[half:]], r[sub[half:]], feats)
        pred = np.asarray(pred, float); pred[tv] = pred.min()
        if policy == "shuffled_p1r":
            pred = pred[rng.permutation(n)]
        val = pred - pred.min()
    else:
        return uniform
    if val.std() < 1e-9:
        return uniform
    z = (val - val.mean()) / (val.std() + 1e-9)
    p = np.exp(z / TEMPERATURE); p = p / p.sum()
    floor = MINP / n
    p = np.maximum(p, floor); p = p / p.sum()
    return p


@torch.no_grad()
def _acc(net, x, y):
    out = torch.cat([net(x[s:s + 512]).argmax(1) for s in range(0, len(x), 512)])
    return float((out == y).float().mean())


def run_c1(source, buffer_policy, sampling_policy, estimator, seed, steps=120):
    torch.manual_seed(1000 + seed)
    tasks, trp, tep, kind, nout = (prep_har(seed) if source == "har" else prep_emnist(seed))
    net = (HarMLP() if kind == "har" else EmnistCNN(n_out=nout)); opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 100)
    if kind == "har":
        memx = torch.empty(0, 561)
    else:
        memx = torch.empty(0, 1, 28, 28)
    memy = torch.empty(0, dtype=torch.long); seen = [0]
    nT = len(tasks); acc = np.zeros((nT, nT)); prio = None
    for t in range(nT):
        newx, newy = trp[t]
        if len(memx):
            prio = priority_over_buffer(net, memx, memy, sampling_policy, estimator, rng)
        net.train()
        for _ in range(steps):
            bi = rng.choice(len(newx), min(BATCH, len(newx)), replace=False)
            xb, yb = newx[bi], newy[bi]
            if len(memx):
                mi = rng.choice(len(memx), min(BATCH, len(memx)), replace=False, p=prio)
                xb = torch.cat([xb, memx[mi]]); yb = torch.cat([yb, memy[mi]])
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
        if buffer_policy == "reservoir":
            memx, memy = reservoir_buffer(net, memx, memy, newx, newy, rng, kind, seen)
        else:
            memx, memy = gdumb_buffer(net, memx, memy, newx, newy, rng, kind)
        for j in range(t + 1):
            xt, yt = tep[j]; acc[t, j] = _acc(net, xt, yt)
    return float(acc[nT - 1, :].mean()), acc
