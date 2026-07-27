"""Phase 4 admission bed: budgeted allocation with a known reducible-value ground truth.

All three canary lanes are, at core, the same scientific question in different clothing: given a limited budget
(protection, verification, or compute) and only decision-time information, can a learned mechanism allocate the
budget to the items with genuine reducible downstream value, beating the strongest simple heuristics and every
required control, with real oracle headroom, across independent generator families and more than one
implementation? This module is a controlled bed with KNOWN ground truth so the oracle, the reducible/
irreducible structure, the timing, and the controls are exact. A controlled bed is the correct instrument for
an ADMISSION battery: it decides mechanism plausibility with the smallest valid compute BEFORE any real-data
canary. Passing it is not scientific confirmation; it only earns a lane its bounded canary.

The bed is not rigged: the mechanism sees only decision-time features and must decode reducible value that is
a genuine-but-noisy function of those features. Decodability is set per lane and the battery measures the
outcome; nothing forces a pass.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BedConfig:
    lane: str
    n_items: int = 400
    budget_frac: float = 0.2
    decodability: float = 0.6      # fraction of reducible-value variance explained by decision-time features
    noisy_tv_frac: float = 0.15    # items with high feature magnitude but zero reducible value
    n_features: int = 8
    right_time_window: int = 1


class AllocationBed:
    """One generator family (regime). Known ground truth: reducible value r, noise, right-time t."""

    def __init__(self, cfg: BedConfig, *, regime: int, seed: int):
        self.cfg = cfg
        self.regime = regime
        self.rng = np.random.default_rng(seed * 1000 + regime)
        self._generate()

    def _regime_map(self, x: np.ndarray) -> np.ndarray:
        """Independent generator families: materially different feature->value structures."""
        if self.regime == 0:      # linear
            w = self.rng.normal(size=self.cfg.n_features)
            return x @ w
        if self.regime == 1:      # sparse nonlinear
            idx = self.rng.choice(self.cfg.n_features, size=3, replace=False)
            return np.tanh(x[:, idx]).sum(axis=1) + 0.5 * (x[:, idx[0]] ** 2)
        # interaction
        return x[:, 0] * x[:, 1] - x[:, 2] * x[:, 3]

    def _generate(self) -> None:
        n, f = self.cfg.n_items, self.cfg.n_features
        x = self.rng.normal(size=(n, f))
        signal = self._regime_map(x)
        signal = (signal - signal.mean()) / (signal.std() + 1e-9)
        # reducible value r = decodable signal + independent noise (irreducible)
        d = self.cfg.decodability
        noise = self.rng.normal(size=n)
        r = np.sqrt(d) * signal + np.sqrt(max(1e-9, 1 - d)) * noise
        r = r - r.min()  # nonnegative gain if allocated at the right time
        # noisy-TV items: high feature magnitude, zero reducible value (useless surprise)
        tv = self.rng.random(n) < self.cfg.noisy_tv_frac
        x[tv] += 3.0 * self.rng.normal(size=(tv.sum(), f))  # inflate feature magnitude
        r[tv] = 0.0
        # right time per item
        t = self.rng.integers(0, 10, size=n)
        self.x, self.r, self.t, self.tv = x, r, t, tv
        self.budget = int(self.cfg.budget_frac * n)

    # ---- value model: gain realized only if budget spent on the item AT its right time ----
    def realized_value(self, chosen: np.ndarray, times: np.ndarray | None = None) -> float:
        chosen = np.asarray(chosen, dtype=int)
        if times is None:
            times = self.t[chosen]
        correct_time = np.abs(times - self.t[chosen]) <= self.cfg.right_time_window
        return float(np.sum(self.r[chosen] * correct_time))

    # ---- oracle: knows true r and t ----
    def oracle_value(self) -> float:
        order = np.argsort(-self.r)
        chosen = order[: self.budget]
        return self.realized_value(chosen, times=self.t[chosen])

    def random_value(self, seed: int) -> float:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(self.cfg.n_items, size=self.budget, replace=False)
        return self.realized_value(chosen, times=self.t[chosen])


# ---- mechanism implementations (two materially different estimators) ----
def _fit_linear(xtr, ytr):
    xb = np.hstack([xtr, np.ones((len(xtr), 1))])
    w, *_ = np.linalg.lstsq(xb, ytr, rcond=None)
    return lambda x: np.hstack([x, np.ones((len(x), 1))]) @ w


def _fit_knn(xtr, ytr, k=15):
    def predict(x):
        out = np.empty(len(x))
        for i, xi in enumerate(x):
            d = np.sum((xtr - xi) ** 2, axis=1)
            nn = np.argpartition(d, k)[:k]
            out[i] = ytr[nn].mean()
        return out
    return predict


MECHANISM_IMPLS = {"linear": _fit_linear, "knn": _fit_knn}


def mechanism_allocation(train: AllocationBed, test: AllocationBed, impl: str) -> np.ndarray:
    """Train a reducible-value estimator on decision-time features (train split) and allocate on test."""
    est = MECHANISM_IMPLS[impl](train.x, train.r)
    pred = est(test.x)
    return np.argsort(-pred)[: test.budget]


def heuristic_allocation(test: AllocationBed, kind: str) -> np.ndarray:
    """Strongest simple predictors: feature magnitude (entropy/margin proxy), a loss proxy, frequency, recency."""
    if kind == "feature_magnitude":   # high surprise -> exactly where noisy-TV lives
        score = np.linalg.norm(test.x, axis=1)
    elif kind == "first_feature":     # a single-feature loss/entropy proxy
        score = np.abs(test.x[:, 0])
    elif kind == "frequency":
        score = test.x[:, 1]
    elif kind == "recency":
        score = -test.t.astype(float)
    else:
        score = np.zeros(test.cfg.n_items)
    return np.argsort(-score)[: test.budget]
