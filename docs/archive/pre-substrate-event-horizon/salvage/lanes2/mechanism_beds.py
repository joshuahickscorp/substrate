"""Phase 4B mechanism-specific beds for P1R, U1, N1.

Each bed represents its scientific claim directly rather than as generic allocation, while sharing the
calibrated battery. The shared structure is a budgeted decision over independent generator families with a
known reducible-value ground truth, a real oracle, and a lane-specific NOISY-TV trap: a simple heuristic that
would misfire on high-signal-but-useless items, which the real mechanism must avoid.

  P1R: items are experiences; reducible value = joint retention-plasticity benefit of replaying the
       experience (near old-task boundaries and diverse). The recency heuristic is the trap. Irreducible:
       random experiences with no joint benefit.
  U1:  items are decisions; reducible value = decision-utility gain from verifying a wrong-but-fixable
       prediction. The entropy/feature-magnitude heuristic is the trap (high-entropy-but-correct items waste
       verification budget).
  N1:  items are inputs; reducible value = downstream gain from spending compute to reduce novelty, nonzero
       only for reducible-novel inputs. Raw novelty magnitude is the trap (irreducible noise and noisy-TV are
       high-novelty but yield no gain).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import numpy as np


class MechanismBed:
    """N independent generator families with a lane-specific reducible-value ground truth + noisy-TV trap."""

    def __init__(self, lane: str, *, n_families=12, n_items=300, n_features=8, budget_frac=0.2,
                 decodability=0.55, noisy_tv_frac=0.2):
        self.lane = lane
        self.n_families = n_families
        self.n = n_items
        self.f = n_features
        self.budget_frac = budget_frac
        self.dec = decodability
        self.tv = noisy_tv_frac

    def _signal(self, x, family):
        rng = np.random.default_rng(4000 + family)
        kind = family % 4  # materially different structures across families (independent units)
        if self.lane == "G1-P1R":
            # joint retention-plasticity: near-boundary (|x0| moderate) AND diverse (x1 x2 interaction)
            boundary = np.exp(-((x[:, 0]) ** 2))            # peaks at moderate distance from boundary
            diverse = np.tanh(x[:, 1] * x[:, 2])
            base = boundary * (1 + diverse)
        elif self.lane == "G1-U1":
            # wrong-but-fixable: a nonlinear pocket of genuine reducible uncertainty
            base = np.tanh(x[:, 0]) - 0.5 * x[:, 1] ** 2 + 0.3 * x[:, 2]
        else:  # G1-N1
            base = np.sin(x[:, 0] + x[:, 1]) + 0.5 * np.tanh(x[:, 2] * x[:, 3])
        if kind == 1:
            base = base + 0.4 * x[:, 3 % self.f]
        elif kind == 2:
            base = np.abs(base)
        elif kind == 3:
            base = base * (1 + 0.3 * np.tanh(x[:, 4 % self.f]))
        return base

    def _split(self, family, seed, tag):
        rng = np.random.default_rng((family + 1) * 100003 + seed * 17 + (hash(tag) % 997))
        x = rng.normal(size=(self.n, self.f))
        sig = self._signal(x, family)
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        d = self.dec
        r = np.sqrt(d) * sig + np.sqrt(max(1e-9, 1 - d)) * rng.normal(size=self.n)
        r = r - r.min()
        # lane-specific noisy-TV trap: items a naive heuristic loves but with zero reducible value
        tv = rng.random(self.n) < self.tv
        if self.lane == "G1-U1":
            x[tv, 0] += 3.5 * np.abs(rng.normal(size=tv.sum()))   # high apparent entropy, but useless
        elif self.lane == "G1-N1":
            x[tv] += 3.5 * rng.normal(size=(tv.sum(), self.f))    # high novelty magnitude, but useless
        else:  # P1R: spurious recency
            x[tv, -1] += 3.5 * np.abs(rng.normal(size=tv.sum()))
        r[tv] = 0.0
        t = rng.integers(0, 10, self.n)
        return {"x": x, "r": r, "t": t, "tv": tv}

    def family(self, f, seed):
        return {"train": self._split(f, seed, "train"), "tune": self._split(f, seed, "tune"),
                "test": self._split(f, seed, "test"), "budget": int(self.budget_frac * self.n)}
