"""Domain providers: task sequences (train + untouched group-disjoint test) for the substrate engine.

Observation providers are declared: EMNIST/CIFAR flattened raw pixels (frozen identity provider), HAR 561-dim
sensor features. Every domain uses an owned trainable projection into substrate space, so a frozen provider is
never the only learned representation. House style: no dashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

DATA_IMG = Path("/Users/scammermike/Downloads/mop/campaign2/data")
HAR = Path("/Users/scammermike/Downloads/mop-gen3/runs/generation3/mop-generation3-discovery-v1/data/UCI HAR Dataset")
_TF = transforms.ToTensor()
_CACHE = {}


def _flat(ds, idx):
    return torch.stack([ds[i][0].flatten() for i in idx])


def emnist(seed, n_tasks=6, cpt=5, per_tr=500, per_te=160):
    key = ("emnist", seed, n_tasks)
    if key in _CACHE:
        return _CACHE[key]
    tr = datasets.EMNIST(str(DATA_IMG), split="balanced", train=True, transform=_TF)
    te = datasets.EMNIST(str(DATA_IMG), split="balanced", train=False, transform=_TF)
    ytr = np.array(tr.targets); yte = np.array(te.targets); rng = np.random.default_rng(seed)
    classes = list(range(n_tasks * cpt)); tasks = []
    for t in range(n_tasks):
        cs = classes[t * cpt:(t + 1) * cpt]
        itr = np.concatenate([rng.choice(np.where(ytr == c)[0], per_tr, replace=False) for c in cs])
        ite = np.concatenate([rng.choice(np.where(yte == c)[0], per_te, replace=False) for c in cs])
        tasks.append({"x": _flat(tr, itr), "y": torch.tensor(ytr[itr]), "ctx": torch.full((len(itr),), t),
                      "test": (_flat(te, ite), torch.tensor(yte[ite]), torch.full((len(ite),), t))})
    out = (tasks, 784, n_tasks * cpt); _CACHE[key] = out; return out


def cifar100(seed, n_tasks=6, cpt=5, per_tr=450, per_te=100):
    key = ("cifar", seed, n_tasks)
    if key in _CACHE:
        return _CACHE[key]
    tr = datasets.CIFAR100(str(DATA_IMG), train=True, transform=_TF, download=True)
    te = datasets.CIFAR100(str(DATA_IMG), train=False, transform=_TF, download=True)
    ytr = np.array(tr.targets); yte = np.array(te.targets); rng = np.random.default_rng(seed)
    classes = list(rng.permutation(100)[:n_tasks * cpt]); tasks = []
    for t in range(n_tasks):
        cs = classes[t * cpt:(t + 1) * cpt]
        itr = np.concatenate([rng.choice(np.where(ytr == c)[0], per_tr, replace=False) for c in cs])
        ite = np.concatenate([rng.choice(np.where(yte == c)[0], per_te, replace=False) for c in cs])
        # relabel to 0..K-1 for a compact head
        remap = {c: i for i, c in enumerate(classes)}
        ytr_r = np.array([remap[int(ytr[i])] for i in itr]); yte_r = np.array([remap[int(yte[i])] for i in ite])
        tasks.append({"x": _flat(tr, itr), "y": torch.tensor(ytr_r), "ctx": torch.full((len(itr),), t),
                      "test": (_flat(te, ite), torch.tensor(yte_r), torch.full((len(ite),), t))})
    out = (tasks, 3072, n_tasks * cpt); _CACHE[key] = out; return out


def har(seed, mode="class", n_tasks=3, cpt=2):
    """HAR class-incremental (mode=class) or subject-shift (mode=shift). 561-dim sensor features, non-image."""
    key = ("har", seed, mode)
    if key in _CACHE:
        return _CACHE[key]
    Xtr = np.loadtxt(HAR / "train/X_train.txt").astype(np.float32); ytr = np.loadtxt(HAR / "train/y_train.txt").astype(int) - 1
    str_ = np.loadtxt(HAR / "train/subject_train.txt").astype(int)
    Xte = np.loadtxt(HAR / "test/X_test.txt").astype(np.float32); yte = np.loadtxt(HAR / "test/y_test.txt").astype(int) - 1
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    rng = np.random.default_rng(seed); tasks = []
    if mode == "class":
        order = rng.permutation(6)
        for t in range(n_tasks):
            cs = order[t * cpt:(t + 1) * cpt]
            itr = np.concatenate([np.where(ytr == c)[0] for c in cs]); ite = np.concatenate([np.where(yte == c)[0] for c in cs])
            tasks.append({"x": torch.tensor(Xtr[itr]), "y": torch.tensor(ytr[itr]), "ctx": torch.full((len(itr),), t),
                          "test": (torch.tensor(Xte[ite]), torch.tensor(yte[ite]), torch.full((len(ite),), t))})
    else:  # subject-shift: same 6 classes, different disjoint subject groups per task (distribution shift + returning contexts)
        subs = np.unique(str_); groups = np.array_split(rng.permutation(subs), n_tasks)
        for t in range(n_tasks):
            m = np.isin(str_, groups[t]); itr = np.where(m)[0]
            tasks.append({"x": torch.tensor(Xtr[itr]), "y": torch.tensor(ytr[itr]), "ctx": torch.full((len(itr),), t),
                          "test": (torch.tensor(Xte), torch.tensor(yte), torch.full((len(yte),), t))})
    out = (tasks, 561, 6); _CACHE[key] = out; return out


PROVIDERS = {
    "emnist": lambda s: emnist(s),
    "cifar100": lambda s: cifar100(s),
    "har_class": lambda s: har(s, "class"),
    "har_shift": lambda s: har(s, "shift"),
}
IS_IMAGE = {"emnist": True, "cifar100": True, "har_class": False, "har_shift": False}
