"""Frontier selection-style admission beds: V1 (verification), K1 (contradiction repair), M1 (messaging).

All three are per-decision reducible-value beds that plug into the calibrated Phase 4B battery unchanged.
Each measures r = the real decision-utility gain of the mechanism's bounded intervention on an item, then
the battery tests whether a learned policy predicts r from decision-time features and beats the strongest
simple controls across independent groups, while resisting the noisy-TV trap that killed U1.

V1 verification: base model plus an independently trained stronger verifier on CIFAR-10 distribution regimes.
r = 1 when the base is wrong and the verifier corrects it, 0 otherwise. The strongest simple controls are
raw uncertainty, entropy, margin, and loss proxy (historical U1 is exactly the raw-uncertainty control, kept
as a frozen negative). Noisy-TV items are high-uncertainty items verification does not fix.

K1 contradiction repair: two independently trained base models on CIFAR-10. A contradiction is a disagreement.
The repair recomputes with a stronger third model. r = 1 when the base decision is wrong and repair fixes it,
0 when it does not, and the noisy-TV mask marks high-disagreement items where repair would flip an already
correct decision (false-repair harm the mechanism must avoid). Simple controls are the consistency check
(disagreement magnitude) and a majority rule.

M1 messaging: split-view MNIST. Component A sees the top half, component B sees the bottom half. r = 1 when A
alone is wrong and A plus B's message is right, 0 otherwise. The mechanism must predict message necessity from
A's own decision-time features and beat A's raw uncertainty; the centralized matched-capacity comparison is a
canary-stage architecture control, not an admission control.

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
from confirm_beds import CifarCNN, _cifar, _corrupt  # noqa: E402
from n1_mnist_bed import SmallCNN  # noqa: E402

DATA = Path("/Users/scammermike/Downloads/mop/campaign2/data")
RUNS = Path("/Users/scammermike/Downloads/mop-scientific-frontier/runs/generation2/mop-generation2-scientific-frontier-v1")
RUNS.mkdir(parents=True, exist_ok=True)
torch.manual_seed(11)
np.random.seed(11)
_TF = transforms.ToTensor()


def _train_cifar(net, epochs, seed):
    ds = _cifar(True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), 20000, replace=False)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval(); return net


class StrongCifarCNN(nn.Module):
    """Higher-capacity verifier: wider channels + extra block."""

    def __init__(self, n_out=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 128, 3, padding=1)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.head = nn.Linear(256, n_out)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.max_pool2d(F.relu(self.c4(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


REGIMES = ("clean", "noise", "blur", "bright", "occlude", "cifar100_ood")


class V1VerificationBed:
    """Selective verification value on CIFAR-10 regimes. Independent units are distribution regimes."""

    def __init__(self):
        self.cache = RUNS / "v1_bed_cache.npz"
        self.n_families = len(REGIMES)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    @torch.no_grad()
    def _items(self, base, strong, cents, regime, seed):
        rng = np.random.default_rng(seed)
        src = _cifar(False, hundred=True) if regime == "cifar100_ood" else _cifar(False)
        idx = rng.choice(len(src), 700, replace=False)
        xs = torch.stack([src[i][0] for i in idx]); ys = torch.tensor([src[i][1] for i in idx])
        xc = _corrupt(xs, regime, rng)
        pb = F.softmax(base(xc), 1); ps = F.softmax(strong(xc), 1)
        feat = base.features(xc)
        base_pred = pb.argmax(1); strong_pred = ps.argmax(1)
        if regime == "cifar100_ood":
            # no in-distribution label: verification value = entropy reduction the verifier provides
            r = ((-(pb * torch.log(pb + 1e-9)).sum(1)) - (-(ps * torch.log(ps + 1e-9)).sum(1))).clamp(min=0).numpy()
        else:
            r = ((base_pred != ys) & (strong_pred == ys)).float().numpy()  # verification fixes a real error
        ent = -(pb * torch.log(pb + 1e-9)).sum(1, keepdim=True)
        top2 = pb.topk(2, 1).values; margin = top2[:, :1] - top2[:, 1:2]; maxp = pb.max(1, keepdim=True).values
        loss_proxy = 1 - maxp
        dcent = torch.cdist(feat, cents).min(1, keepdim=True).values
        x = torch.cat([pb, ent, margin, maxp, loss_proxy, feat.norm(dim=1, keepdim=True), dcent], 1).numpy().astype(np.float64)
        nov_hi = ent.numpy().ravel() > np.quantile(ent.numpy(), 0.75)
        tv = nov_hi & (r < (0.5 if regime != "cifar100_ood" else 0.02))
        return {"x": x, "r": r.astype(np.float64), "t": rng.integers(0, 10, len(r)), "tv": tv}

    @torch.no_grad()
    def _centroids(self, net):
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(_cifar(True), list(range(8000))), batch_size=512)
        sums = torch.zeros(10, 128); cnt = torch.zeros(10)
        for xb, yb in dl:
            f = net.features(xb)
            for c in range(10):
                m = yb == c
                if m.any():
                    sums[c] += f[m].sum(0); cnt[c] += m.sum()
        return sums / cnt.clamp(min=1).unsqueeze(1)

    def _build(self):
        base = _train_cifar(CifarCNN(), 3, 1)
        strong = _train_cifar(StrongCifarCNN(), 5, 42)  # materially stronger verifier
        cents = self._centroids(base)
        self.data = {f: {tag: self._items(base, strong, cents, REGIMES[f], f * 10 + i)
                         for i, tag in enumerate(("train", "tune", "test"))} for f in range(self.n_families)}
        self._save()

    def _save(self):
        np.savez(self.cache, **{f"{f}_{t}_{k}": self.data[f][t][k]
                                for f in self.data for t in self.data[f] for k in self.data[f][t]})

    def _load(self):
        z = np.load(self.cache)
        self.data = {f: {t: {k: z[f"{f}_{t}_{k}"] for k in ("x", "r", "t", "tv")}
                         for t in ("train", "tune", "test")} for f in range(self.n_families)}

    def family(self, f, seed=7):
        d = self.data[f]; return {"train": d["train"], "tune": d["tune"], "test": d["test"],
                                  "budget": int(0.2 * len(d["test"]["r"]))}


class K1ContradictionBed:
    """Contradiction repair value on CIFAR-10 regimes. Two base models disagree; a stronger model repairs."""

    def __init__(self):
        self.cache = RUNS / "k1_bed_cache.npz"
        self.n_families = len(REGIMES)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    @torch.no_grad()
    def _items(self, m1, m2, strong, regime, seed):
        rng = np.random.default_rng(seed)
        src = _cifar(False, hundred=True) if regime == "cifar100_ood" else _cifar(False)
        idx = rng.choice(len(src), 700, replace=False)
        xs = torch.stack([src[i][0] for i in idx]); ys = torch.tensor([src[i][1] for i in idx])
        xc = _corrupt(xs, regime, rng)
        p1 = F.softmax(m1(xc), 1); p2 = F.softmax(m2(xc), 1); pss = F.softmax(strong(xc), 1)
        d1 = p1.argmax(1); d2 = p2.argmax(1); ds = pss.argmax(1)
        disagree = (d1 != d2).float()
        if regime == "cifar100_ood":
            # value = repair reduces committee entropy on genuine OOD (no label)
            r = ((-(p1 * torch.log(p1 + 1e-9)).sum(1)) - (-(pss * torch.log(pss + 1e-9)).sum(1))).clamp(min=0).numpy()
            false_repair = np.zeros(len(idx), dtype=bool)
        else:
            base_wrong = (d1 != ys)
            fixed = base_wrong & (ds == ys)
            broke = (~base_wrong) & (ds != ys)   # repair flipped a correct decision = false-repair harm
            r = fixed.float().numpy()
            false_repair = broke.numpy()
        jsd = (0.5 * (F.kl_div(torch.log(p1 + 1e-9), (p1 + p2) / 2, reduction="none").sum(1)
                      + F.kl_div(torch.log(p2 + 1e-9), (p1 + p2) / 2, reduction="none").sum(1)))
        ent1 = -(p1 * torch.log(p1 + 1e-9)).sum(1, keepdim=True)
        margin1 = p1.topk(2, 1).values[:, :1] - p1.topk(2, 1).values[:, 1:2]
        x = torch.cat([p1, disagree.unsqueeze(1), jsd.unsqueeze(1), ent1, margin1,
                       p1.max(1, keepdim=True).values], 1).numpy().astype(np.float64)
        # noisy-TV: high disagreement (top-signal) but no real repair value, or outright false-repair harm
        dis_hi = disagree.numpy() > 0.5
        tv = (dis_hi & (r < 0.5)) | false_repair
        return {"x": x, "r": r.astype(np.float64), "t": rng.integers(0, 10, len(r)), "tv": tv}

    def _build(self):
        m1 = _train_cifar(CifarCNN(), 3, 7)
        m2 = _train_cifar(CifarCNN(), 3, 8)          # independently trained second model
        strong = _train_cifar(StrongCifarCNN(), 5, 9)  # stronger repair model
        self.data = {f: {tag: self._items(m1, m2, strong, REGIMES[f], f * 10 + i)
                         for i, tag in enumerate(("train", "tune", "test"))} for f in range(self.n_families)}
        self._save()

    def _save(self):
        np.savez(self.cache, **{f"{f}_{t}_{k}": self.data[f][t][k]
                                for f in self.data for t in self.data[f] for k in self.data[f][t]})

    def _load(self):
        z = np.load(self.cache)
        self.data = {f: {t: {k: z[f"{f}_{t}_{k}"] for k in ("x", "r", "t", "tv")}
                         for t in ("train", "tune", "test")} for f in range(self.n_families)}

    def family(self, f, seed=7):
        d = self.data[f]; return {"train": d["train"], "tune": d["tune"], "test": d["test"],
                                  "budget": int(0.2 * len(d["test"]["r"]))}


MVIEWS = ("clean", "noise", "occlude_top", "shift", "elastic", "fashion_ood")


class M1MessagingBed:
    """Split-view MNIST. A sees the top half, B sees the bottom half; message value is measured per item."""

    def __init__(self):
        self.cache = RUNS / "m1_bed_cache.npz"
        self.n_families = len(MVIEWS)
        if self.cache.exists():
            self._load()
        else:
            self._build()

    def _mnist(self, train):
        return datasets.MNIST(str(DATA), train=train, transform=_TF)

    def _split(self, x):
        top = x.clone(); top[:, :, 14:, :] = 0     # A sees top half
        bot = x.clone(); bot[:, :, :14, :] = 0     # B sees bottom half
        return top, bot

    def _train_view(self, which, epochs=2):
        ds = self._mnist(True)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        net = SmallCNN(); opt = torch.optim.Adam(net.parameters(), 1e-3); net.train()
        for _ in range(epochs):
            for xb, yb in dl:
                top, bot = self._split(xb)
                v = top if which == "A" else bot
                opt.zero_grad(); F.cross_entropy(net(v), yb).backward(); opt.step()
        net.eval(); return net

    @torch.no_grad()
    def _corrupt_view(self, x, regime, rng):
        if regime == "noise":
            return torch.clamp(x + 0.5 * torch.randn_like(x), 0, 1)
        if regime == "occlude_top":
            y = x.clone(); y[:, :, 4:10, 8:20] = 0; return y
        if regime == "shift":
            return torch.roll(x, shifts=2, dims=3)
        if regime == "elastic":
            return torch.clamp(x + 0.3 * torch.roll(x, shifts=2, dims=3), 0, 1)
        return x

    @torch.no_grad()
    def _items(self, A, B, regime, seed):
        rng = np.random.default_rng(seed)
        src = datasets.FashionMNIST(str(DATA), False, transform=_TF) if regime == "fashion_ood" else self._mnist(False)
        idx = rng.choice(len(src), 700, replace=False)
        xs = torch.stack([src[i][0] for i in idx]); ys = torch.tensor([src[i][1] for i in idx])
        xc = self._corrupt_view(xs, regime, rng)
        top, bot = self._split(xc)
        pa = F.softmax(A(top), 1); pb = F.softmax(B(bot), 1)
        pboth = (pa + pb) / 2                      # A augmented by B's message
        da = pa.argmax(1); dboth = pboth.argmax(1)
        if regime == "fashion_ood":
            r = ((-(pa * torch.log(pa + 1e-9)).sum(1)) - (-(pboth * torch.log(pboth + 1e-9)).sum(1))).clamp(min=0).numpy()
        else:
            r = ((da != ys) & (dboth == ys)).float().numpy()  # message fixes a real A-error
        ent_a = -(pa * torch.log(pa + 1e-9)).sum(1, keepdim=True)
        margin_a = pa.topk(2, 1).values[:, :1] - pa.topk(2, 1).values[:, 1:2]
        x = torch.cat([pa, ent_a, margin_a, pa.max(1, keepdim=True).values], 1).numpy().astype(np.float64)
        # noisy-TV: A highly uncertain (top signal) but the message does not actually help
        au_hi = ent_a.numpy().ravel() > np.quantile(ent_a.numpy(), 0.75)
        tv = au_hi & (r < (0.5 if regime != "fashion_ood" else 0.02))
        return {"x": x, "r": r.astype(np.float64), "t": rng.integers(0, 10, len(r)), "tv": tv}

    def _build(self):
        A = self._train_view("A"); B = self._train_view("B")
        self.data = {f: {tag: self._items(A, B, MVIEWS[f], f * 10 + i)
                         for i, tag in enumerate(("train", "tune", "test"))} for f in range(self.n_families)}
        self._save()

    def _save(self):
        np.savez(self.cache, **{f"{f}_{t}_{k}": self.data[f][t][k]
                                for f in self.data for t in self.data[f] for k in self.data[f][t]})

    def _load(self):
        z = np.load(self.cache)
        self.data = {f: {t: {k: z[f"{f}_{t}_{k}"] for k in ("x", "r", "t", "tv")}
                         for t in ("train", "tune", "test")} for f in range(self.n_families)}

    def family(self, f, seed=7):
        d = self.data[f]; return {"train": d["train"], "tune": d["tune"], "test": d["test"],
                                  "budget": int(0.2 * len(d["test"]["r"]))}
