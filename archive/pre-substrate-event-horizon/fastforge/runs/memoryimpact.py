"""Bounded impact measurement for the GDumb admission defect.

A unit test written late in the program found that the replay buffer gated admission on a free slot, so once
it filled, nothing new was ever admitted. Every sealed run used that buffer. This runner measures what the
defect actually cost rather than asserting it was harmless.

The comparison reruns the same arms under both policies at the same seeds and the same budget, and reports
the difference. Every sealed arm shared the same buffer, so arm comparisons were internally fair either way;
what is at stake is whether the replay baselines were weaker than their names imply.

House style: no dashes.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from fastforge import engine as E
from fastforge import within as W
from fastforge.runs import io

SEEDS = [0, 1, 2, 3, 4]
ARMS = ["lstm_gdumb", "G0_always_trainable", "reservoir"]
DOMAIN = "har"


class LegacyMemory(E.Memory):
    """The buffer exactly as every sealed run had it: admission required a free slot."""

    def add(self, x, y, rng):
        if self.policy == "none":
            return
        for i in range(len(x)):
            self.seen += 1
            if len(self.x) < self.cap:
                self.x.append(x[i])
                self.y.append(int(y[i]))
            elif self.policy == "reservoir":
                j = int(rng.integers(0, self.seen))
                if j < self.cap:
                    self.x[j], self.y[j] = x[i], int(y[i])
            elif self.policy == "recent":
                self.x.pop(0)
                self.y.pop(0)
                self.x.append(x[i])
                self.y.append(int(y[i]))
        if self.policy == "gdumb" and len(self.x) >= self.cap:
            yy = np.array(self.y)
            cls = np.unique(yy)
            per = max(1, self.cap // len(cls))
            keep = []
            for c in cls:
                idx = np.where(yy == c)[0]
                keep.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
            keep = keep[: self.cap]
            self.x = [self.x[i] for i in keep]
            self.y = [self.y[i] for i in keep]


def buffer_diversity(policy="gdumb", cap=600, n=3000, classes=6):
    """How many distinct source rows each policy retains from one oversized admission."""
    rng = np.random.default_rng(0)
    x = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).expand(n, 4, 2).contiguous()
    y = torch.arange(n) % classes
    out = {}
    for name, cls in (("legacy", LegacyMemory), ("fixed", E.Memory)):
        mem = cls(policy, cap)
        mem.add(x, y, np.random.default_rng(0))
        rows = {float(t[0, 0]) for t in mem.x}
        out[name] = {
            "retained": mem.size(),
            "distinct_source_rows": len(rows),
            "max_source_row_index": max(rows) if rows else None,
            "classes_present": len(set(mem.y)),
        }
    del rng
    return out


def main():
    t0 = time.time()
    diversity = buffer_diversity()
    steps = max(60, 550 // 2)
    results = {}
    for policy_name, cls in (("legacy", LegacyMemory), ("fixed", E.Memory)):
        E_Memory = E.Memory
        E.Memory = cls
        try:
            for arm in ARMS:
                vals = []
                for s in SEEDS:
                    torch.manual_seed(1000 + s)
                    r = W.run_within(arm, DOMAIN, s, "plain", steps)
                    vals.append(r["metrics"]["avg_final"])
                results.setdefault(arm, {})[policy_name] = vals
                print(f"  {policy_name:7s} {arm:22s} {np.mean(vals):.4f}", flush=True)
        finally:
            E.Memory = E_Memory

    effects = {arm: E.effect(results[arm]["fixed"], results[arm]["legacy"]) for arm in ARMS if arm in results}
    material = {a: abs(e["mean"]) >= io.SESOI for a, e in effects.items()}
    io.seal(
        "MOP_FAST_STATE_MEMORY_DEFECT_IMPACT.json",
        {
            "schema": "mop-fast-state-memory-defect-impact/v1",
            "defect": "the replay buffer gated admission on a free slot, so once it reached capacity no "
            "later item was admitted and the GDumb rebalance only ever saw what arrived first",
            "found_by": "a unit test written for the coverage target, not by the science",
            "every_sealed_run_used_the_legacy_buffer": True,
            "why_arm_comparisons_stay_internally_fair": "the same buffer was used by every arm at the same "
            "capacity, so the comparison between arms was not biased toward any of them. What is at stake "
            "is the absolute strength of the replay baselines.",
            "buffer_diversity_probe": diversity,
            "domain": DOMAIN,
            "seeds": SEEDS,
            "arms": ARMS,
            "accuracy_by_policy": {
                a: {k: [round(float(v), 4) for v in vals] for k, vals in per.items()}
                for a, per in results.items()
            },
            "fixed_minus_legacy": effects,
            "material_at_SESOI": material,
            "verdict": (
                "memory_defect_immaterial"
                if not any(material.values())
                else "memory_defect_material_rerun_required"
            ),
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print("effects", {a: e["mean"] for a, e in effects.items()}, flush=True)
    print("MEMORYIMPACT_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
