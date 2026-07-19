"""Lane P: externally grounded P1R replication on EMNIST-balanced class-incremental.

EMNIST-balanced (NIST Special Database 19, 47 balanced classes) is a genuinely different source from
split-MNIST (canary), CIFAR-100 (confirmation), and KMNIST (third source). The stream is class-incremental
over 9 tasks. A single model learns the tasks in sequence under a FIXED replay memory M and MATCHED compute
(equal gradient steps per task for every method). Each replay method decides which past experiences to keep
in the memory and replay.

Established methods, whose designs predate this campaign, are implemented here for comparison:
  reservoir      Vitter reservoir sampling (uniform-in-expectation memory), Chaudhry et al. tiny episodic memory.
  gdumb          GDumb greedy class balancing in memory (Prabhu et al. 2020).
  loss_based     keep the highest-loss past items (a standard hard-example replay heuristic).
  recency        keep the most recent items.
P1R is the mechanism under test: class-balanced quota (stability) filled by retention-need (high current loss)
and prototypicality (diversity), to hold stability and plasticity at matched cost.

Primary comparison: P1R final per-task test accuracy versus the BEST established method, under matched M and
matched compute. Independent units are the past tasks (retention measurements), consistent with the prior P1R
beds. A positive is classified same_team_external_method_positive because the established methods are
re-implemented in-house, not run from external code.

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

sys.path.insert(0, "/Users/scammermike/Downloads/mop/campaign2/lanes")

DATA = Path("/Users/scammermike/Downloads/mop/campaign2/data")
_TF = transforms.ToTensor()
N_TASKS = 9
CLASSES_PER_TASK = 5           # 9*5 = 45 of the 47 balanced classes
MEM = 600                      # fixed replay memory (matched across methods)
STEPS_PER_TASK = 250           # matched compute
BATCH = 128


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
    tr, te = _load()
    ytr = np.array(tr.targets); yte = np.array(te.targets)
    rng = np.random.default_rng(seed)
    classes = list(range(N_TASKS * CLASSES_PER_TASK))
    tasks = [classes[i * CLASSES_PER_TASK:(i + 1) * CLASSES_PER_TASK] for i in range(N_TASKS)]
    train_pool, test_pool = {}, {}
    for t, cs in enumerate(tasks):
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], 900, replace=False) for c in cs])
        xte = np.concatenate([rng.choice(np.where(yte == c)[0], 160, replace=False) for c in cs])
        train_pool[t] = (torch.stack([tr[i][0] for i in idx]), torch.tensor([tr[i][1] for i in idx]))
        test_pool[t] = (torch.stack([te[i][0] for i in xte]), torch.tensor([te[i][1] for i in xte]))
    return tasks, train_pool, test_pool


@torch.no_grad()
def _acc(net, x, y):
    net.eval(); out = []
    for s in range(0, len(x), 512):
        out.append(net(x[s:s + 512]).argmax(1))
    return float((torch.cat(out) == y).float().mean())


def _select_memory(method, net, memx, memy, newx, newy, rng):
    """Return updated (memx, memy) of size <= MEM after adding the new task, per the method."""
    allx = torch.cat([memx, newx]) if len(memx) else newx
    ally = torch.cat([memy, newy]) if len(memy) else newy
    n = len(allx)
    if n <= MEM:
        return allx, ally
    if method == "reservoir":
        keep = rng.choice(n, MEM, replace=False)
    elif method == "recency":
        keep = np.arange(n - MEM, n)
    elif method in ("gdumb", "p1r"):
        classes = torch.unique(ally).tolist()
        per = max(1, MEM // len(classes))
        keep = []
        with torch.no_grad():
            if method == "p1r":
                loss = torch.cat([F.cross_entropy(net(allx[s:s + 512]), ally[s:s + 512], reduction="none")
                                  for s in range(0, n, 512)]).numpy()
                feat = torch.cat([net.features(allx[s:s + 512]) for s in range(0, n, 512)]).numpy()
        for c in classes:
            idx = np.where(ally.numpy() == c)[0]
            if method == "gdumb":
                sel = rng.choice(idx, min(per, len(idx)), replace=False)
            else:  # p1r: retention-need (loss) balanced with prototypicality (near class centroid)
                cen = feat[idx].mean(0)
                proto = -np.linalg.norm(feat[idx] - cen, axis=1)
                score = 0.6 * (loss[idx] - loss[idx].mean()) / (loss[idx].std() + 1e-9) \
                    + 0.4 * (proto - proto.mean()) / (proto.std() + 1e-9)
                sel = idx[np.argsort(-score)[:min(per, len(idx))]]
            keep.extend(sel.tolist())
        keep = np.array(keep[:MEM])
    elif method == "loss_based":
        with torch.no_grad():
            loss = torch.cat([F.cross_entropy(net(allx[s:s + 512]), ally[s:s + 512], reduction="none")
                              for s in range(0, n, 512)]).numpy()
        keep = np.argsort(-loss)[:MEM]
    elif method == "none":
        keep = np.array([], dtype=int)
    else:
        keep = rng.choice(n, MEM, replace=False)
    return allx[keep], ally[keep]


def run_stream(method, seed=0):
    tasks, train_pool, test_pool = _prep(seed)
    net = EmnistCNN(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 100)
    memx = torch.empty(0, 1, 28, 28); memy = torch.empty(0, dtype=torch.long)
    acc_matrix = np.zeros((N_TASKS, N_TASKS))  # [after_task, on_task]
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
        memx, memy = _select_memory(method, net, memx, memy, newx, newy, rng)
        for j in range(t + 1):
            xt, yt = test_pool[j]; acc_matrix[t, j] = _acc(net, xt, yt)
    return acc_matrix


def joint_utility(acc_matrix):
    """Final per-task retention (accuracy on each task after the whole stream)."""
    return acc_matrix[N_TASKS - 1, :].copy()
