"""Phase 4B battery calibration: positive and negative controls (construct validity).

Before the repaired battery is allowed to prune scientific lanes it must demonstrate it can tell known-good
from known-bad. This builds a generic families-based allocation bed with tunable decodability and validates:
  - a POSITIVE control (strongly decodable signal, capable mechanism) is admitted (passes A-J);
  - a NULL bed (signal independent of features) is pruned (fails decodability / incremental);
  - an ORACLE-FREE bed (no headroom over random) is classified invalid_bed;
  - random, noisy-TV, and shuffled selectors FAIL the value gates.
If any of these calibration expectations is violated, the battery is repaired before principal use.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repaired_battery import (  # noqa: E402
    SESOI,
    _family_norm_values,
    classify,
    heuristic_selector,
    mechanism_selector,
    random_selector,
    run_repaired_battery,
    shuffled_train_selector,
)


class AllocationBed:
    """N independent generator families; each has a materially different feature->reducible-value map."""

    def __init__(self, *, n_families=12, decodability=0.6, noisy_tv_frac=0.15, n_items=300, n_features=8,
                 budget_frac=0.2, oracle_free=False):
        self.n_families = n_families
        self.dec = decodability
        self.tv = noisy_tv_frac
        self.n = n_items
        self.f = n_features
        self.budget_frac = budget_frac
        self.oracle_free = oracle_free

    def _map(self, x, family):
        rng = np.random.default_rng(1000 + family)
        kind = family % 4
        if kind == 0:
            return x @ rng.normal(size=self.f)
        if kind == 1:
            idx = rng.choice(self.f, size=3, replace=False)
            return np.tanh(x[:, idx]).sum(1) + 0.5 * x[:, idx[0]] ** 2
        if kind == 2:
            return x[:, 0] * x[:, 1] - x[:, 2] * x[:, 3]
        w = rng.normal(size=self.f)
        return np.sin(x @ w)

    def _split(self, family, seed, tag):
        rng = np.random.default_rng((family + 1) * 100003 + seed * 17 + hash(tag) % 997)
        x = rng.normal(size=(self.n, self.f))
        if self.oracle_free:
            r = np.ones(self.n)  # exactly constant -> oracle == random, zero headroom
            t = rng.integers(0, 10, self.n)
            return {"x": x, "r": r, "t": t, "tv": np.zeros(self.n, bool)}
        sig = self._map(x, family)
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        d = self.dec
        r = np.sqrt(d) * sig + np.sqrt(max(1e-9, 1 - d)) * rng.normal(size=self.n)
        r = r - r.min()
        tv = rng.random(self.n) < self.tv
        x[tv] += 3.0 * rng.normal(size=(tv.sum(), self.f))
        r[tv] = 0.0
        t = rng.integers(0, 10, self.n)
        return {"x": x, "r": r, "t": t, "tv": tv}

    def family(self, f, seed):
        return {"train": self._split(f, seed, "train"), "tune": self._split(f, seed, "tune"),
                "test": self._split(f, seed, "test"), "budget": int(self.budget_frac * self.n)}


def main() -> int:
    import json
    results = {}

    # 1. positive control: strongly decodable -> must be admitted
    pos = AllocationBed(decodability=0.75, noisy_tv_frac=0.12)
    b_pos = run_repaired_battery(pos)
    results["positive_control"] = {**classify(b_pos), "expected": "admitted"}

    # 2. null bed: signal independent of features, NO noisy-TV structure -> truly null -> must be pruned
    null = AllocationBed(decodability=0.0, noisy_tv_frac=0.0)
    b_null = run_repaired_battery(null)
    results["null_bed"] = {**classify(b_null), "expected": "pruned_mechanism or invalid_bed"}

    # 3. oracle-free bed -> invalid_bed
    of = AllocationBed(oracle_free=True)
    b_of = run_repaired_battery(of)
    results["oracle_free_bed"] = {**classify(b_of), "expected": "invalid_bed"}

    # 4. negative selectors must fail the value gate on the positive bed
    rnd = _family_norm_values(pos, random_selector(0))
    sh = _family_norm_values(pos, shuffled_train_selector("kernel_ridge"))
    tvsel = _family_norm_values(pos, heuristic_selector("feature_magnitude"))  # chases noisy-TV surprise
    mech = _family_norm_values(pos, mechanism_selector("kernel_ridge"))
    results["negative_selectors"] = {
        "random_mean": round(float(np.mean(rnd)), 3),
        "shuffled_mean": round(float(np.mean(sh)), 3),
        "noisy_tv_heuristic_mean": round(float(np.mean(tvsel)), 3),
        "capable_mechanism_mean": round(float(np.mean(mech)), 3),
        "mechanism_beats_all_negatives": bool(np.mean(mech) > max(np.mean(rnd), np.mean(sh), np.mean(tvsel)) + SESOI),
    }

    calib_ok = (
        results["positive_control"]["classification"] == "admitted"
        and results["null_bed"]["classification"] in ("pruned_mechanism", "invalid_bed")
        and results["oracle_free_bed"]["classification"] == "invalid_bed"
        and results["negative_selectors"]["mechanism_beats_all_negatives"]
    )
    results["calibration_passed"] = calib_ok
    print(json.dumps(results, indent=2))
    print("\nBATTERY CALIBRATION:", "PASSED" if calib_ok else "FAILED (repair battery before use)")
    return 0 if calib_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
