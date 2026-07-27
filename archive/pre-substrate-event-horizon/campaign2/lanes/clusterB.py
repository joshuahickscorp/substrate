"""Cluster B: R1 retrieval admission + P1R replication on KMNIST (third source), then the factorial cluster.

R1 retrieval bed: a base CNN is trained on KMNIST. Memory items each get a MEASURED retrieval value r: how
much a memory-augmented predictor's error on a held-out query set falls when that item is available to
retrieve. Decision-time features include a SIMILARITY signal (distance to the query-set centroid), so the
battery's simple controls already contain the strongest simple retrieval control (nearest-similarity). A
successful nearest-similarity lookup is not a novel mechanism, so R1 must beat it: the battery's incremental
clause is exactly mechanism-minus-best-simple. Independent units are KMNIST class-group tasks.

P1R replication bed: replay-value on KMNIST class-incremental tasks, same measured structure as the MNIST/CIFAR
P1R beds, to give a bounded third-source P1R replication if R1 fails admission.

Both plug into the calibrated Phase 4B battery. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from n1_mnist_bed import SmallCNN

DATA = Path("/Users/scammermike/Downloads/mop/campaign2/data")
RUNS = Path("/Users/scammermike/Downloads/mop/campaign2/runs")
torch.manual_seed(5)
np.random.seed(5)
_TF = transforms.ToTensor()


def _kmnist(train):
    return datasets.KMNIST(str(DATA), train=train, transform=_TF)


def _train(net, ds, epochs=2, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval(); return net


# ---------------- R1 retrieval admission ----------------
R1_TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # class-group tasks (independent units)


class R1RetrievalBed:
    def __init__(self):
        self.cache = RUNS / "r1_kmnist_cache.npz"
        self.n_families = len(R1_TASKS)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    def _build(self):
        base = _train(SmallCNN(), _kmnist(True), epochs=2, seed=1)
        with torch.no_grad():
            train = _kmnist(True); test = _kmnist(False)
            tr_targets = np.array(train.targets); te_targets = np.array(test.targets)
            rng = np.random.default_rng(6)
            self.data = {}
            for f, task in enumerate(R1_TASKS):
                # memory pool = candidate items from this task's classes
                midx = np.concatenate([np.where(tr_targets == c)[0] for c in task])
                midx = rng.choice(midx, 500, replace=False)
                mx = torch.stack([train[i][0] for i in midx]); my = torch.tensor([train[i][1] for i in midx])
                mfeat = base.features(mx)
                # query set = held-out test items from the same classes (the decisions retrieval must improve)
                qidx = np.concatenate([np.where(te_targets == c)[0] for c in task])
                qidx = rng.choice(qidx, 300, replace=False)
                qx = torch.stack([test[i][0] for i in qidx]); qy = torch.tensor([test[i][1] for i in qidx])
                qfeat = base.features(qx); qprob = F.softmax(base(qx), 1)
                base_wrong = (qprob.argmax(1) != qy)
                # MEASURED retrieval value r_m: for each memory item, how many currently-wrong queries would a
                # nearest-memory label vote FIX if m were its nearest same-features neighbour (memory-augmented fix)
                d = torch.cdist(qfeat, mfeat)  # queries x memory
                nn_idx = d.argmin(1)
                r = np.zeros(len(midx))
                for qi in range(len(qidx)):
                    if base_wrong[qi]:
                        m = nn_idx[qi].item()
                        if my[m] == qy[qi]:
                            r[m] += 1.0  # this memory item fixes a real error via retrieval
                r = r - r.min()
                # decision-time features of each memory item (retrieval-time info; NO query label)
                mprob = F.softmax(base(mx), 1)
                ment = -(mprob * torch.log(mprob + 1e-9)).sum(1, keepdim=True)
                mmargin = mprob.topk(2, 1).values[:, :1] - mprob.topk(2, 1).values[:, 1:2]
                sim_to_qcentroid = -torch.cdist(mfeat, qfeat.mean(0, keepdim=True))  # SIMILARITY signal (strong control)
                x = torch.cat([mprob, ment, mmargin, mfeat.norm(dim=1, keepdim=True), sim_to_qcentroid], 1).numpy().astype(np.float64)
                tv = (sim_to_qcentroid.numpy().ravel() > np.quantile(sim_to_qcentroid.numpy(), 0.75)) & (r < 0.5)
                perm = rng.permutation(len(r)); splits = {}
                for i, tag in enumerate(("train", "tune", "test")):
                    sl = perm[i * 150:(i + 1) * 150]
                    splits[tag] = {"x": x[sl], "r": r[sl].astype(np.float64),
                                   "t": rng.integers(0, 10, len(sl)), "tv": tv[sl]}
                self.data[f] = splits
            np.savez(self.cache, **{f"{f}_{t}_{k}": self.data[f][t][k]
                                    for f in self.data for t in self.data[f] for k in self.data[f][t]})

    def _load(self):
        z = np.load(self.cache)
        self.data = {f: {t: {k: z[f"{f}_{t}_{k}"] for k in ("x", "r", "t", "tv")}
                         for t in ("train", "tune", "test")} for f in range(self.n_families)}

    def family(self, f, seed=7):
        d = self.data[f]; return {"train": d["train"], "tune": d["tune"], "test": d["test"],
                                  "budget": int(0.2 * len(d["test"]["r"]))}


# ---------------- P1R replication on KMNIST ----------------
class P1RKmnistBed:
    def __init__(self):
        self.cache = RUNS / "p1r_kmnist_cache.npz"
        self.n_families = len(R1_TASKS)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    @torch.no_grad()
    def _feat(self, net, x):
        return net.features(x)

    def _build(self):
        net = SmallCNN(); opt = torch.optim.Adam(net.parameters(), 1e-3)
        train = _kmnist(True); targets = np.array(train.targets); rng = np.random.default_rng(7)
        for task in R1_TASKS:
            idx = np.concatenate([np.where(targets == c)[0] for c in task])
            idx = rng.choice(idx, 2000, replace=False)
            xb = torch.stack([train[i][0] for i in idx]); yb = torch.tensor([train[i][1] for i in idx])
            net.train()
            for _ in range(2):
                for s in range(0, len(xb), 256):
                    opt.zero_grad(); F.cross_entropy(net(xb[s:s + 256]), yb[s:s + 256]).backward(); opt.step()
        net.eval()
        self.data = {}
        for f, task in enumerate(R1_TASKS):
            idx = np.concatenate([np.where(targets == c)[0] for c in task])
            idx = rng.choice(idx, 900, replace=False)
            xs = torch.stack([train[i][0] for i in idx]); ys = torch.tensor([train[i][1] for i in idx])
            with torch.no_grad():
                logits = net(xs); prob = F.softmax(logits, 1)
                ce = F.cross_entropy(logits, ys, reduction="none").numpy(); feat = net.features(xs)
            r = ce - ce.min()
            ent = -(prob * torch.log(prob + 1e-9)).sum(1, keepdim=True)
            x = torch.cat([prob, ent, feat.norm(dim=1, keepdim=True)], 1).numpy().astype(np.float64)
            tv = (feat.norm(dim=1).numpy() > np.quantile(feat.norm(dim=1).numpy(), 0.75)) & (r < np.quantile(r, 0.2))
            perm = rng.permutation(len(r)); splits = {}
            for i, tag in enumerate(("train", "tune", "test")):
                sl = perm[i * 300:(i + 1) * 300]
                splits[tag] = {"x": x[sl], "r": r[sl].astype(np.float64),
                               "t": rng.integers(0, 10, len(sl)), "tv": tv[sl]}
            self.data[f] = splits
        np.savez(self.cache, **{f"{f}_{t}_{k}": self.data[f][t][k]
                                for f in self.data for t in self.data[f] for k in self.data[f][t]})

    def _load(self):
        z = np.load(self.cache)
        self.data = {f: {t: {k: z[f"{f}_{t}_{k}"] for k in ("x", "r", "t", "tv")}
                         for t in ("train", "tune", "test")} for f in range(self.n_families)}

    def family(self, f, seed=7):
        d = self.data[f]; return {"train": d["train"], "tune": d["tune"], "test": d["test"],
                                  "budget": int(0.2 * len(d["test"]["r"]))}
