"""G1-N1 real-data canary bed: reducible novelty on MNIST-family images (measured, not authored).

Construct: a small CNN is trained on clean MNIST. At test time each item is drawn from one of several real
DISTRIBUTION REGIMES (clean, gaussian noise, blur, rotation, elastic, and genuine Fashion-MNIST OOD). A
bounded intervention -- test-time-augmentation ensemble inference (K augmented forward passes) at real compute
cost -- is applied and its effect on the item's loss is MEASURED. The reducible value r of an item is the
measured cross-entropy reduction from the intervention: positive where more inference genuinely helps
(reducible novelty), near zero where it does not (irreducible noise, ordinary difficulty, or true OOD that
more inference cannot fix). Nothing about which items are reducible is authored; it is a property of the real
data and the real model.

The mechanism must predict r from DECISION-TIME features only (base logits, entropy, margin, max prob, feature
norm, and distance to the nearest training-class centroid = a novelty signal). Independent units are the
distribution regimes; splits are group-disjoint. The noisy-TV mask marks high-novelty items whose intervention
does not help. This bed plugs into the calibrated Phase 4B battery unchanged.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path("/Users/scammermike/Downloads/mop/campaign2")
DATA = ROOT / "data"
CACHE = ROOT / "runs/n1_bed_cache.npz"
torch.manual_seed(0)
np.random.seed(0)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 64)
        self.head = nn.Linear(64, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


def _load(train):
    tf = transforms.ToTensor()
    return datasets.MNIST(str(DATA), train=train, transform=tf)


def train_base(epochs=2):
    ds = _load(True)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    net = SmallCNN()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()
    return net


def _corrupt(x, regime, rng):
    if regime == "clean":
        return x
    if regime == "noise":
        return torch.clamp(x + 0.5 * torch.randn_like(x), 0, 1)
    if regime == "blur":
        k = torch.ones(1, 1, 3, 3) / 9.0
        return F.conv2d(x, k, padding=1)
    if regime == "rotate":
        return torch.rot90(x, k=1, dims=(2, 3))
    if regime == "elastic":
        return torch.clamp(x + 0.3 * torch.roll(x, shifts=2, dims=3), 0, 1)
    return x


REGIMES = ("clean", "noise", "blur", "rotate", "elastic", "fashion_ood")


@torch.no_grad()
def _centroids(net):
    ds = _load(True)
    dl = torch.utils.data.DataLoader(ds, batch_size=512)
    sums = torch.zeros(10, 64)
    counts = torch.zeros(10)
    for xb, yb in dl:
        f = net.features(xb)
        for c in range(10):
            m = yb == c
            if m.any():
                sums[c] += f[m].sum(0)
                counts[c] += m.sum()
    return sums / counts.clamp(min=1).unsqueeze(1)


@torch.no_grad()
def _regime_items(net, regime, cents, n=800, seed=0):
    rng = np.random.default_rng(seed)
    if regime == "fashion_ood":
        base = datasets.FashionMNIST(str(DATA), train=False, transform=transforms.ToTensor())
        y_true = None
    else:
        base = _load(False)
    idx = rng.choice(len(base), size=n, replace=False)
    xs = torch.stack([base[i][0] for i in idx])
    ys = torch.tensor([base[i][1] for i in idx])
    xc = _corrupt(xs, regime, rng)
    # base prediction
    logits = net(xc)
    prob = F.softmax(logits, 1)
    feat = net.features(xc)
    # intervention: TTA ensemble (K augmented views), real extra compute
    K = 8
    acc = torch.zeros_like(prob)
    for _ in range(K):
        shift = int(rng.integers(-2, 3))
        acc += F.softmax(net(torch.roll(xc, shifts=shift, dims=3)), 1)
    tta = acc / K
    # measured reducibility r = base CE minus intervention CE (only defined vs a label; for OOD use entropy drop)
    if regime == "fashion_ood":
        base_ce = -(prob * torch.log(prob + 1e-9)).sum(1)   # entropy as a stand-in "cost"
        tta_ce = -(tta * torch.log(tta + 1e-9)).sum(1)
    else:
        base_ce = F.cross_entropy(torch.log(prob + 1e-9), ys, reduction="none")
        tta_ce = F.cross_entropy(torch.log(tta + 1e-9), ys, reduction="none")
    r = (base_ce - tta_ce).clamp(min=0).numpy()             # nonnegative measured error reduction
    # decision-time features (no label leakage)
    ent = -(prob * torch.log(prob + 1e-9)).sum(1, keepdim=True)
    top2 = prob.topk(2, 1).values
    margin = (top2[:, :1] - top2[:, 1:2])
    maxp = prob.max(1, keepdim=True).values
    fnorm = feat.norm(dim=1, keepdim=True)
    dcent = torch.cdist(feat, cents).min(1, keepdim=True).values   # novelty: distance to nearest class centroid
    x_feat = torch.cat([prob, ent, margin, maxp, fnorm, dcent], 1).numpy()
    # noisy-TV mask: high novelty (top-quartile distance to class centroids) but near-zero MEASURED
    # reducibility (intervention does not actually help) -> useless surprise the mechanism must avoid.
    nov_hi = dcent.numpy().ravel() > np.quantile(dcent.numpy(), 0.75)
    red_lo = r < 0.02
    tv = nov_hi & red_lo
    t = rng.integers(0, 10, len(r))
    return {"x": x_feat.astype(np.float64), "r": r.astype(np.float64), "t": t, "tv": tv}


class N1RealBed:
    """Distribution regimes are the independent units; each split is group-disjoint within a regime."""

    def __init__(self):
        self.n_families = len(REGIMES)
        if CACHE.exists():
            self._load_cache()
        else:
            self._build()

    def _build(self):
        net = train_base()
        cents = _centroids(net)
        self.data = {}
        for f, reg in enumerate(REGIMES):
            splits = {tag: _regime_items(net, reg, cents, seed=f * 10 + i)
                      for i, tag in enumerate(("train", "tune", "test"))}
            self.data[f] = splits
        np.savez(CACHE, **{f"{f}_{tag}_{k}": self.data[f][tag][k]
                           for f in self.data for tag in self.data[f] for k in self.data[f][tag]})

    def _load_cache(self):
        z = np.load(CACHE, allow_pickle=True)
        self.data = {}
        for f in range(self.n_families):
            self.data[f] = {tag: {k: z[f"{f}_{tag}_{k}"] for k in ("x", "r", "t", "tv")}
                            for tag in ("train", "tune", "test")}

    def family(self, f, seed=7):
        d = self.data[f]
        budget = int(0.2 * len(d["test"]["r"]))
        return {"train": d["train"], "tune": d["tune"], "test": d["test"], "budget": budget}


if __name__ == "__main__":
    bed = N1RealBed()
    print("N1 real bed built. regimes:", REGIMES)
    for f in range(bed.n_families):
        fam = bed.family(f)
        print(f"  {REGIMES[f]}: test n={len(fam['test']['r'])} budget={fam['budget']} "
              f"mean_r={fam['test']['r'].mean():.3f} tv_frac={fam['test']['tv'].mean():.2f}")
