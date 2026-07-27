"""Temporal order validity gate for any registered domain.

A bed is only a temporal bed if order carries information. Two comparisons decide it, both on the unit
disjoint test split: a recurrent estimator against an order free bag of timesteps, and the same recurrent
estimator against itself on shuffled time. Failing either makes the bed invalid for substrate work, whatever
its accuracy.

House style: no dashes.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from fastforge import arch as A
from fastforge import data as D
from fastforge import engine as E
from fastforge import sequence as S
from fastforge.runs import io

SEEDS = [0, 1, 2, 3, 4]


def shuffle_time(x, rng):
    return x[:, torch.tensor(rng.permutation(x.shape[1])), :]


def gate(dname: str, steps: int, seeds=SEEDS):
    d = D.domain(dname)
    dom = {dname: (d["channels"], d["classes"])}
    res = {}
    for name, kind, shuffled in (("gru", "gru", False), ("gru_shuffled", "gru", True), ("bag", "bag", False)):
        accs = []
        for s in seeds:
            torch.manual_seed(1000 + s)
            m = A.build(kind, dom)
            rng = np.random.default_rng(s)
            x, xte = d["x"], d["xte"]
            if shuffled:
                x, xte = shuffle_time(x, rng), shuffle_time(xte, np.random.default_rng(s))
            E.fit(
                m,
                dname,
                x,
                d["y"],
                train_groups=list(m.param_groups),
                steps=steps,
                lr=S.lr_for(dname),
                rng=rng,
            )
            accs.append(E.evaluate(m, dname, xte, d["yte"]))
        res[name] = accs
    th = [g - b for g, b in zip(res["gru"], res["bag"], strict=True)]
    om = [g - s for g, s in zip(res["gru"], res["gru_shuffled"], strict=True)]
    th_lcb = E.lcb(th)
    verdict = (
        "temporal_headroom_present"
        if th_lcb >= 0.03 and float(np.mean(om)) > 0.02
        else "invalid_no_temporal_headroom"
    )
    return {
        "domain": dname,
        "steps": steps,
        "seeds": list(seeds),
        "n_train": int(len(d["x"])),
        "n_test": int(len(d["xte"])),
        "channels": d["channels"],
        "classes": d["classes"],
        "unit": d["unit"],
        "n_units_train": int(len(np.unique(d["u"]))),
        "n_units_test": int(len(np.unique(d["ute"]))),
        "gru": round(float(np.mean(res["gru"])), 4),
        "gru_shuffled": round(float(np.mean(res["gru_shuffled"])), 4),
        "bag_order_free": round(float(np.mean(res["bag"])), 4),
        "temporal_headroom_gru_vs_bag": round(float(np.mean(th)), 4),
        "temporal_headroom_lcb": round(th_lcb, 4),
        "order_matters_gru_vs_shuffled": round(float(np.mean(om)), 4),
        "verdict": verdict,
    }


def main():
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else "pamap2_transition"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    r = gate(which, steps)
    print(
        r["domain"],
        r["verdict"],
        "gru",
        r["gru"],
        "bag",
        r["bag_order_free"],
        "shuffled",
        r["gru_shuffled"],
        flush=True,
    )
    io.seal(
        f"MOP_DOMAIN_GATE_{which.upper()}.json",
        {"schema": "mop-domain-order-gate/v1", **r, "wall_seconds": round(time.time() - t0, 1)},
    )
    print("DOMAINGATE_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
