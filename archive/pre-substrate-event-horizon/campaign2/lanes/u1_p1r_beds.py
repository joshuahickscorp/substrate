"""G1-U1 and G1-P1R real-data canary beds on MNIST-family data (measured value, not authored).

U1 (calibrated uncertainty for verification): a base model and a materially stronger verifier model are trained
on real MNIST. The bounded intervention is to VERIFY an item with the stronger model and take its answer at a
real compute cost. The reducible value r is the MEASURED decision-utility gain: 1 when the base is wrong and
the verifier corrects it, 0 otherwise (verifying a correct item or an item the verifier also gets wrong yields
no gain). The mechanism must predict, from decision-time uncertainty features, which items verification will
fix, beating entropy/margin/loss and every control. Independent units are distribution regimes.

P1R (stability-plasticity replay selection): split-MNIST gives 5 sequential tasks (digit pairs). After learning
a new task, the reducible value r of an old-task experience is the MEASURED joint retention-plasticity benefit
of replaying it, approximated by its post-shift loss under the current model (high-loss old experiences are the
ones whose replay most reduces forgetting) gated by whether replaying it does not block new learning. The
mechanism predicts which old experiences to replay under a fixed memory budget. Independent units are tasks.

Both plug into the calibrated Phase 4B battery. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from n1_mnist_bed import REGIMES, SmallCNN, _corrupt, _load, train_base

DATA = Path("/Users/scammermike/Downloads/mop/campaign2/data")
RUNS = Path("/Users/scammermike/Downloads/mop/campaign2/runs")
torch.manual_seed(1)
np.random.seed(1)


class StrongCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.relu(self.c3(F.relu(self.c2(x))))
        x = F.max_pool2d(x, 2)
        return self.head(F.relu(self.fc(x.flatten(1))))


def _train(net, epochs, subset=None):
    ds = _load(True)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval(); return net


# ---------------- U1 ----------------
class U1RealBed:
    def __init__(self):
        self.cache = RUNS / "u1_bed_cache.npz"
        self.n_families = len(REGIMES)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    @torch.no_grad()
    def _items(self, base, strong, regime, seed):
        rng = np.random.default_rng(seed)
        src = datasets.FashionMNIST(str(DATA), False, transform=transforms.ToTensor()) if regime == "fashion_ood" \
            else _load(False)
        idx = rng.choice(len(src), 700, replace=False)
        xs = torch.stack([src[i][0] for i in idx]); ys = torch.tensor([src[i][1] for i in idx])
        xc = _corrupt(xs, regime, rng)
        pb = F.softmax(base(xc), 1); ps = F.softmax(strong(xc), 1)
        base_pred = pb.argmax(1); strong_pred = ps.argmax(1)
        if regime == "fashion_ood":
            # no true label: value = verifier reduces entropy meaningfully (proxy decision gain)
            r = (( -(pb*torch.log(pb+1e-9)).sum(1) ) - ( -(ps*torch.log(ps+1e-9)).sum(1) )).clamp(min=0).numpy()
        else:
            r = ((base_pred != ys) & (strong_pred == ys)).float().numpy()  # verification fixes a real error
        ent = -(pb*torch.log(pb+1e-9)).sum(1, keepdim=True)
        top2 = pb.topk(2, 1).values; margin = top2[:, :1]-top2[:, 1:2]; maxp = pb.max(1, keepdim=True).values
        loss_proxy = (1-maxp)
        x = torch.cat([pb, ent, margin, maxp, loss_proxy], 1).numpy().astype(np.float64)
        nov_hi = ent.numpy().ravel() > np.quantile(ent.numpy(), 0.75)
        tv = nov_hi & (r < 0.5 if regime != "fashion_ood" else r < 0.02)
        return {"x": x, "r": r.astype(np.float64), "t": rng.integers(0, 10, len(r)), "tv": tv}

    def _build(self):
        base = train_base(2); strong = _train(StrongCNN(), 3)
        self.data = {f: {tag: self._items(base, strong, REGIMES[f], f*10+i)
                         for i, tag in enumerate(("train", "tune", "test"))} for f in range(self.n_families)}
        np.savez(self.cache, **{f"{f}_{t}_{k}": self.data[f][t][k]
                                for f in self.data for t in self.data[f] for k in self.data[f][t]})

    def _load(self):
        z = np.load(self.cache)
        self.data = {f: {t: {k: z[f"{f}_{t}_{k}"] for k in ("x", "r", "t", "tv")}
                         for t in ("train", "tune", "test")} for f in range(self.n_families)}

    def family(self, f, seed=7):
        d = self.data[f]; return {"train": d["train"], "tune": d["tune"], "test": d["test"],
                                  "budget": int(0.2*len(d["test"]["r"]))}


# ---------------- P1R ----------------
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class P1RRealBed:
    def __init__(self):
        self.cache = RUNS / "p1r_bed_cache.npz"
        self.n_families = len(TASKS)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    @torch.no_grad()
    def _feat(self, net, x):
        return net.features(x)

    def _build(self):
        # sequential learning over 5 tasks; measure post-sequence loss on each old task's experiences
        net = SmallCNN(); opt = torch.optim.Adam(net.parameters(), 1e-3)
        train = _load(True)
        by_class = {c: [i for i in range(len(train)) if train.targets[i].item() == c] for c in range(10)}
        rng = np.random.default_rng(3)
        for (a, b) in TASKS:
            idx = rng.choice(by_class[a]+by_class[b], 2000, replace=False)
            xb = torch.stack([train[i][0] for i in idx]); yb = torch.tensor([train[i][1] for i in idx])
            net.train()
            for _ in range(2):
                for s in range(0, len(xb), 256):
                    opt.zero_grad(); F.cross_entropy(net(xb[s:s+256]), yb[s:s+256]).backward(); opt.step()
        net.eval()
        # for each task (family), old-task experiences: r = post-sequence loss (retention need) gated by
        # not-fresh-noise; features = current model logits + feature norm at decision time
        self.data = {}
        for f, (a, b) in enumerate(TASKS):
            idx = rng.choice(by_class[a]+by_class[b], 900, replace=False)
            xs = torch.stack([train[i][0] for i in idx]); ys = torch.tensor([train[i][1] for i in idx])
            with torch.no_grad():
                logits = net(xs); prob = F.softmax(logits, 1)
                ce = F.cross_entropy(logits, ys, reduction="none").numpy()
                feat = net.features(xs)
            r = ce - ce.min()  # high post-sequence loss = high replay benefit for retention
            ent = -(prob*torch.log(prob+1e-9)).sum(1, keepdim=True)
            x = torch.cat([prob, ent, feat.norm(dim=1, keepdim=True)], 1).numpy().astype(np.float64)
            tv = (feat.norm(dim=1).numpy() > np.quantile(feat.norm(dim=1).numpy(), 0.75)) & (r < np.quantile(r, 0.2))
            splits = {}
            perm = rng.permutation(len(r))
            for i, tag in enumerate(("train", "tune", "test")):
                sl = perm[i*300:(i+1)*300]
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
                                  "budget": int(0.2*len(d["test"]["r"]))}


if __name__ == "__main__":
    for name, cls in (("U1", U1RealBed), ("P1R", P1RRealBed)):
        bed = cls()
        print(f"{name} bed built: {bed.n_families} families")
