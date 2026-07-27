"""Phase 4B: sufficiently capable, capacity-matched estimators for the architecture-independence test.

sklearn is unavailable, so the capable nonlinear estimators are implemented in numpy: k-nearest-neighbours
(local nonparametric), RBF kernel ridge (kernel method), and random-Fourier-feature ridge (a random-feature
network approximating a kernel). Linear least squares is kept ONLY as a simple baseline, never as the sole
independent architecture. Each estimator's hyperparameters are selected on a tuning partition (never on test)
so a failure is a genuine failure, not under-tuning. Architecture dependence is judged only across the
capable, independently implemented estimators under matched features, units, and budgets.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import numpy as np


def _standardize(xtr, xte):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    return (xtr - mu) / sd, (xte - mu) / sd


def linear(xtr, ytr, xte, **_):
    xtr, xte = _standardize(xtr, xte)
    xb = np.hstack([xtr, np.ones((len(xtr), 1))])
    w, *_ = np.linalg.lstsq(xb, ytr, rcond=None)
    return np.hstack([xte, np.ones((len(xte), 1))]) @ w


def knn(xtr, ytr, xte, k=15, **_):
    xtr, xte = _standardize(xtr, xte)
    out = np.empty(len(xte))
    for i in range(len(xte)):
        d = np.sum((xtr - xte[i]) ** 2, axis=1)
        nn = np.argpartition(d, min(k, len(xtr) - 1))[:k]
        out[i] = ytr[nn].mean()
    return out


def _rbf(a, b, gamma):
    a2 = np.sum(a ** 2, 1)[:, None]
    b2 = np.sum(b ** 2, 1)[None, :]
    return np.exp(-gamma * (a2 + b2 - 2 * a @ b.T))


def kernel_ridge(xtr, ytr, xte, gamma=0.1, lam=1.0, **_):
    xtr, xte = _standardize(xtr, xte)
    k = _rbf(xtr, xtr, gamma)
    alpha = np.linalg.solve(k + lam * np.eye(len(xtr)), ytr)
    return _rbf(xte, xtr, gamma) @ alpha


def rff_ridge(xtr, ytr, xte, n_feat=200, gamma=0.1, lam=1.0, seed=0, **_):
    xtr, xte = _standardize(xtr, xte)
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=np.sqrt(2 * gamma), size=(xtr.shape[1], n_feat))
    b = rng.uniform(0, 2 * np.pi, size=n_feat)
    ztr = np.sqrt(2.0 / n_feat) * np.cos(xtr @ w + b)
    zte = np.sqrt(2.0 / n_feat) * np.cos(xte @ w + b)
    a = np.linalg.solve(ztr.T @ ztr + lam * np.eye(n_feat), ztr.T @ ytr)
    return zte @ a


# capable nonlinear estimators (linear is a baseline, not in this set)
CAPABLE = {"knn": knn, "kernel_ridge": kernel_ridge, "rff_ridge": rff_ridge}
BASELINE = {"linear": linear}

GRIDS = {
    "knn": [{"k": k} for k in (5, 15, 30)],
    "kernel_ridge": [{"gamma": g, "lam": la} for g in (0.03, 0.1, 0.3) for la in (0.3, 1.0, 3.0)],
    "rff_ridge": [{"gamma": g, "lam": la, "n_feat": 200} for g in (0.03, 0.1, 0.3) for la in (0.3, 1.0, 3.0)],
}


def select_and_fit(name, xtr, ytr, xtune, ytune, xte):
    """Select hyperparameters on the tuning partition (never test), then predict on test. Fair capacity."""
    fn = CAPABLE.get(name) or BASELINE[name]
    if name not in GRIDS:
        return fn(xtr, ytr, xte)
    best, best_score = GRIDS[name][0], -np.inf
    for hp in GRIDS[name]:
        pred = fn(xtr, ytr, xtune, **hp)
        score = np.corrcoef(pred, ytune)[0, 1] if (np.std(pred) > 0 and np.std(ytune) > 0) else -np.inf
        if np.isfinite(score) and score > best_score:
            best_score, best = score, hp
    return fn(xtr, ytr, xte, **best)
