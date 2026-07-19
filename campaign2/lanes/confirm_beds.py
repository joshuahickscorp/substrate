"""Independent-confirmation beds for N1 and P1R on CIFAR (leaves the MNIST family entirely).

N1 confirmation (CIFAR-10): new source (natural color images), new modality vs MNIST, and a MATERIALLY
DIFFERENT intervention -- additional inference via a SECOND, independently trained model (an ensemble
correction), not the MNIST test-time augmentation. The reducible value r is the MEASURED error reduction from
consulting the second model; the mechanism must predict, from decision-time features, which novel/uncertain
items that consultation actually fixes, while rejecting genuine OOD (CIFAR-100 images) it cannot fix.
Independent units are distribution regimes.

P1R confirmation (CIFAR-100): new source, new task family -- class-incremental over 10 tasks of 10 classes.
The reducible value r of an old-task experience is its post-sequence loss (retention need) as on split-MNIST,
but on a genuinely different natural-image source. Independent units are tasks.

Both plug into the calibrated Phase 4B battery. Implementations remain same-team across three capable
architectures, so a positive is classified same_team_cross_architecture, not full independent replication.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA = Path("/Users/scammermike/Downloads/mop/campaign2/data")
RUNS = Path("/Users/scammermike/Downloads/mop/campaign2/runs")
torch.manual_seed(2)
np.random.seed(2)
_TF = transforms.ToTensor()


class CifarCNN(nn.Module):
    def __init__(self, n_out=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 128)
        self.head = nn.Linear(128, n_out)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.relu(self.c2(x))
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


def _cifar(train, hundred=False):
    ds = datasets.CIFAR100 if hundred else datasets.CIFAR10
    return ds(str(DATA), train=train, transform=_TF, download=True)


def _train(net, ds, epochs=2, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    sub = torch.utils.data.Subset(ds, idx.tolist())
    dl = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval(); return net


def _corrupt(x, regime, rng):
    if regime == "clean":
        return x
    if regime == "noise":
        return torch.clamp(x + 0.4 * torch.randn_like(x), 0, 1)
    if regime == "blur":
        k = torch.ones(3, 1, 3, 3) / 9.0
        return F.conv2d(x, k, padding=1, groups=3)
    if regime == "bright":
        return torch.clamp(x * 1.5, 0, 1)
    if regime == "occlude":
        y = x.clone(); y[:, :, 12:20, 12:20] = 0; return y
    return x


N1_REGIMES = ("clean", "noise", "blur", "bright", "occlude", "cifar100_ood")


class N1CifarBed:
    def __init__(self):
        self.cache = RUNS / "n1_cifar_cache.npz"
        self.n_families = len(N1_REGIMES)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    @torch.no_grad()
    def _centroids(self, net):
        ds = _cifar(True)
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(8000))), batch_size=512)
        sums = torch.zeros(10, 128); cnt = torch.zeros(10)
        for xb, yb in dl:
            f = net.features(xb)
            for c in range(10):
                m = yb == c
                if m.any():
                    sums[c] += f[m].sum(0); cnt[c] += m.sum()
        return sums / cnt.clamp(min=1).unsqueeze(1)

    @torch.no_grad()
    def _items(self, base, second, cents, regime, seed):
        rng = np.random.default_rng(seed)
        src = _cifar(False, hundred=True) if regime == "cifar100_ood" else _cifar(False)
        idx = rng.choice(len(src), 700, replace=False)
        xs = torch.stack([src[i][0] for i in idx]); ys = torch.tensor([src[i][1] for i in idx])
        xc = _corrupt(xs, regime, rng)
        pb = F.softmax(base(xc), 1); ps = F.softmax(second(xc), 1)
        feat = base.features(xc)
        # INTERVENTION: additional inference via a second independent model (ensemble correction)
        ens = (pb + ps) / 2
        if regime == "cifar100_ood":
            base_c = -(pb * torch.log(pb + 1e-9)).sum(1); ens_c = -(ens * torch.log(ens + 1e-9)).sum(1)
        else:
            base_c = F.cross_entropy(torch.log(pb + 1e-9), ys, reduction="none")
            ens_c = F.cross_entropy(torch.log(ens + 1e-9), ys, reduction="none")
        r = (base_c - ens_c).clamp(min=0).numpy()
        ent = -(pb * torch.log(pb + 1e-9)).sum(1, keepdim=True)
        top2 = pb.topk(2, 1).values; margin = top2[:, :1] - top2[:, 1:2]; maxp = pb.max(1, keepdim=True).values
        dcent = torch.cdist(feat, cents).min(1, keepdim=True).values
        x = torch.cat([pb, ent, margin, maxp, feat.norm(dim=1, keepdim=True), dcent], 1).numpy().astype(np.float64)
        nov_hi = dcent.numpy().ravel() > np.quantile(dcent.numpy(), 0.75)
        tv = nov_hi & (r < 0.02)
        return {"x": x, "r": r.astype(np.float64), "t": rng.integers(0, 10, len(r)), "tv": tv}

    def _build(self):
        base = _train(CifarCNN(), _cifar(True), epochs=3, seed=1)
        second = _train(CifarCNN(), _cifar(True), epochs=3, seed=99)   # independently trained -> real ensemble
        cents = self._centroids(base)
        self.data = {f: {tag: self._items(base, second, cents, N1_REGIMES[f], f * 10 + i)
                         for i, tag in enumerate(("train", "tune", "test"))} for f in range(self.n_families)}
        np.savez(self.cache, **{f"{f}_{t}_{k}": self.data[f][t][k]
                                for f in self.data for t in self.data[f] for k in self.data[f][t]})

    def _load(self):
        z = np.load(self.cache)
        self.data = {f: {t: {k: z[f"{f}_{t}_{k}"] for k in ("x", "r", "t", "tv")}
                         for t in ("train", "tune", "test")} for f in range(self.n_families)}

    def family(self, f, seed=7):
        d = self.data[f]; return {"train": d["train"], "tune": d["tune"], "test": d["test"],
                                  "budget": int(0.2 * len(d["test"]["r"]))}


P1R_TASKS = [tuple(range(i * 10, i * 10 + 10)) for i in range(10)]  # 10 class-incremental tasks on CIFAR-100


class P1RCifar100Bed:
    def __init__(self):
        self.cache = RUNS / "p1r_cifar_cache.npz"
        self.n_families = len(P1R_TASKS)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    def _build(self):
        net = CifarCNN(n_out=100); opt = torch.optim.Adam(net.parameters(), 1e-3)
        train = _cifar(True, hundred=True)
        targets = np.array(train.targets)
        rng = np.random.default_rng(4)
        for task in P1R_TASKS:
            idx = np.concatenate([np.where(targets == c)[0] for c in task])
            idx = rng.choice(idx, min(2500, len(idx)), replace=False)
            xb = torch.stack([train[i][0] for i in idx]); yb = torch.tensor([train[i][1] for i in idx])
            net.train()
            for _ in range(2):
                for s in range(0, len(xb), 256):
                    opt.zero_grad(); F.cross_entropy(net(xb[s:s + 256]), yb[s:s + 256]).backward(); opt.step()
        net.eval()
        self.data = {}
        for f, task in enumerate(P1R_TASKS):
            idx = np.concatenate([np.where(targets == c)[0] for c in task])
            idx = rng.choice(idx, 900, replace=False)
            xs = torch.stack([train[i][0] for i in idx]); ys = torch.tensor([train[i][1] for i in idx])
            with torch.no_grad():
                logits = net(xs); prob = F.softmax(logits, 1)
                ce = F.cross_entropy(logits, ys, reduction="none").numpy(); feat = net.features(xs)
            r = ce - ce.min()
            ent = -(prob * torch.log(prob + 1e-9)).sum(1, keepdim=True)
            x = torch.cat([prob.topk(10, 1).values, ent, feat.norm(dim=1, keepdim=True)], 1).numpy().astype(np.float64)
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
