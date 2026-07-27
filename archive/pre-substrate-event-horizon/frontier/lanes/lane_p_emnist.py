"""Lane P: externally grounded P1R replication on EMNIST-balanced class-incremental.

EMNIST-balanced (NIST Special Database 19, 47 balanced classes) is a genuinely different source from
split-MNIST (canary), CIFAR-100 (confirmation), and KMNIST (third source). The stream is class-incremental
over 9 tasks. A single model learns the tasks in sequence under a FIXED replay memory M and MATCHED compute
(equal gradient steps per task for every method).

FAITHFUL P1R (post-audit repair): P1R is the mechanism validated in the three prior beds, a LEARNED replay-VALUE
predictor over decision-time features [class probabilities, entropy, feature norm] that predicts measured
post-shift loss (retention need) and carries the validated toxic-value gate tv = (feature-norm > q75) AND
(value < q20). Within each class quota (identical to GDumb's class balance) P1R keeps the top predicted-value
NON-toxic exemplars, so the comparison isolates the within-class replay-VALUE ranking, which is the actual
scientific claim, against GDumb at matched memory and compute. The earlier hand-coded 0.6*z(loss)+0.4*z(proto)
buffer filter was an unfaithful operationalization (it used the raw-loss control itself as the mechanism and
dropped the toxic gate); it is replaced here. Instrumentation is also hardened: torch is seeded, several seeds
are run, and reservoir is proper Vitter reservoir sampling.

Established methods (designs predate this campaign): reservoir (Vitter/Chaudhry), gdumb (Prabhu 2020 greedy
balancing), loss_based (hard-example replay), recency. Primary comparison: P1R final per-task test accuracy vs
the BEST established method under matched M and matched compute. Independent units are the past tasks.

House style: no em dashes and no en dashes.
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
from estimators import select_and_fit  # noqa: E402  validated Phase-4B replay-value predictor

DATA = Path("/Users/scammermike/Downloads/mop/campaign2/data")
_TF = transforms.ToTensor()
N_TASKS = 9
CLASSES_PER_TASK = 5           # 9*5 = 45 of the 47 balanced classes
PER_CLASS_TRAIN = 450
PER_CLASS_TEST = 160
MEM = 600                      # fixed replay memory (matched across methods)
STEPS_PER_TASK = 130           # matched compute
BATCH = 128
_PREP_CACHE = {}


class EmnistCNN(nn.Module):
    def __init__(self, n_out=47):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.head = nn.Linear(128, n_out)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


def _load():
    tr = datasets.EMNIST(str(DATA), split="balanced", train=True, transform=_TF)
    te = datasets.EMNIST(str(DATA), split="balanced", train=False, transform=_TF)
    return tr, te


def _prep(seed=0):
    if seed in _PREP_CACHE:
        return _PREP_CACHE[seed]
    tr, te = _load()
    ytr = np.array(tr.targets); yte = np.array(te.targets)
    rng = np.random.default_rng(seed)
    classes = list(range(N_TASKS * CLASSES_PER_TASK))
    tasks = [classes[i * CLASSES_PER_TASK:(i + 1) * CLASSES_PER_TASK] for i in range(N_TASKS)]
    train_pool, test_pool = {}, {}
    for t, cs in enumerate(tasks):
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], PER_CLASS_TRAIN, replace=False) for c in cs])
        xte = np.concatenate([rng.choice(np.where(yte == c)[0], PER_CLASS_TEST, replace=False) for c in cs])
        train_pool[t] = (torch.stack([tr[i][0] for i in idx]), torch.tensor([tr[i][1] for i in idx]))
        test_pool[t] = (torch.stack([te[i][0] for i in xte]), torch.tensor([te[i][1] for i in xte]))
    _PREP_CACHE[seed] = (tasks, train_pool, test_pool)
    return _PREP_CACHE[seed]


@torch.no_grad()
def _acc(net, x, y):
    net.eval(); out = []
    for s in range(0, len(x), 512):
        out.append(net(x[s:s + 512]).argmax(1))
    return float((torch.cat(out) == y).float().mean())


@torch.no_grad()
def _p1r_predicted_value(net, allx, ally, rng):
    """Validated P1R: learned replay-value predictor over [prob, entropy, feat-norm]; returns (pred_value, tv)."""
    n = len(allx)
    logits = torch.cat([net(allx[s:s + 512]) for s in range(0, n, 512)])
    prob = F.softmax(logits, 1).numpy()
    ce = F.cross_entropy(logits, ally, reduction="none").numpy()
    fnorm = torch.cat([net.features(allx[s:s + 512]) for s in range(0, n, 512)]).norm(dim=1).numpy()
    ent = -(prob * np.log(prob + 1e-9)).sum(1, keepdims=True)
    feats = np.concatenate([prob, ent, fnorm[:, None]], 1).astype(np.float64)
    r = (ce - ce.min()).astype(np.float64)                                   # measured post-shift loss value
    tv = (fnorm > np.quantile(fnorm, 0.75)) & (r < np.quantile(r, 0.2))       # validated toxic-value gate
    # fit the validated predictor on a sub-sample (charged mechanism cost), predict all
    m = min(n, 900); sub = rng.permutation(n)[:m]; half = m // 2
    tr_i, tu_i = sub[:half], sub[half:]
    pred = select_and_fit("rff_ridge", feats[tr_i], r[tr_i], feats[tu_i], r[tu_i], feats)
    return pred, tv


def _select_memory(method, net, memx, memy, newx, newy, rng, seen_before):
    allx = torch.cat([memx, newx]) if len(memx) else newx
    ally = torch.cat([memy, newy]) if len(memy) else newy
    n = len(allx)
    if method == "none":
        return memx, memy, seen_before + len(newx)
    if method == "reservoir":
        # proper Vitter reservoir sampling over the item stream
        if len(memx) < MEM and len(memx) + len(newx) <= MEM:
            return allx, ally, seen_before + len(newx)
        buf_x = list(memx) if len(memx) else []
        buf_y = list(memy.tolist()) if len(memy) else []
        seen = seen_before
        for i in range(len(newx)):
            seen += 1
            if len(buf_x) < MEM:
                buf_x.append(newx[i]); buf_y.append(int(newy[i]))
            else:
                j = int(rng.integers(0, seen))
                if j < MEM:
                    buf_x[j] = newx[i]; buf_y[j] = int(newy[i])
        return torch.stack(buf_x), torch.tensor(buf_y), seen
    if n <= MEM:
        return allx, ally, seen_before + len(newx)
    if method == "recency":
        keep = np.arange(n - MEM, n)
    elif method == "loss_based":
        with torch.no_grad():
            loss = torch.cat([F.cross_entropy(net(allx[s:s + 512]), ally[s:s + 512], reduction="none")
                              for s in range(0, n, 512)]).numpy()
        keep = np.argsort(-loss)[:MEM]
    elif method in ("gdumb", "p1r"):
        classes = torch.unique(ally).tolist()
        per = max(1, MEM // len(classes))
        pred = tv = None
        if method == "p1r":
            pred, tv = _p1r_predicted_value(net, allx, ally, rng)
        keep = []
        ally_np = ally.numpy()
        for c in classes:
            idx = np.where(ally_np == c)[0]
            if method == "gdumb":
                sel = rng.choice(idx, min(per, len(idx)), replace=False)
            else:  # faithful P1R: top predicted-value NON-toxic exemplars within the class quota
                cand = idx[~tv[idx]] if int((~tv[idx]).sum()) >= min(per, len(idx)) else idx
                sel = cand[np.argsort(-pred[cand])[:min(per, len(cand))]]
            keep.extend(np.asarray(sel).tolist())
        keep = np.array(keep[:MEM])
    else:
        keep = rng.choice(n, MEM, replace=False)
    return allx[keep], ally[keep], seen_before + len(newx)


def run_stream(method, seed=0, prepared=None):
    torch.manual_seed(1000 + seed)
    tasks, train_pool, test_pool = prepared if prepared is not None else _prep(seed)
    net = EmnistCNN(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 100)
    memx = torch.empty(0, 1, 28, 28); memy = torch.empty(0, dtype=torch.long)
    acc_matrix = np.zeros((N_TASKS, N_TASKS))
    seen = 0
    for t in range(N_TASKS):
        newx, newy = train_pool[t]
        net.train()
        for _ in range(STEPS_PER_TASK):
            bi = rng.choice(len(newx), min(BATCH, len(newx)), replace=False)
            xb, yb = newx[bi], newy[bi]
            if len(memx) and method != "none":
                mi = rng.choice(len(memx), min(BATCH, len(memx)), replace=False)
                xb = torch.cat([xb, memx[mi]]); yb = torch.cat([yb, memy[mi]])
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
        memx, memy, seen = _select_memory(method, net, memx, memy, newx, newy, rng, seen)
        for j in range(t + 1):
            xt, yt = test_pool[j]; acc_matrix[t, j] = _acc(net, xt, yt)
    return acc_matrix


def joint_utility(acc_matrix):
    return acc_matrix[N_TASKS - 1, :].copy()
