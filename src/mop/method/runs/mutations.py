"""Adversarial mutations against every positive.

Each mutation is an attempt to explain a positive away. A mutation is rejected when the result moves the way
the explanation says it must not: if the core effect is about temporal order, destroying the order must
destroy the effect, and if it is about capacity, tripling the control's capacity must remove it.

Nothing here is a robustness check. Every mutation is a hypothesis about what the effect really was.

House style: no dashes.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from fastforge import engine as E
from mop.method import io
from mop.method.runs import exp1, exp4
from mop.method.runs import factorial as Fx
from mop.method.runs import locus as L

SEEDS = (0, 1, 2)


def _fit_eval(sp, core, readout, seed, steps, hidden=None, x_transform=None, y_shuffle=False):
    torch.manual_seed(seed)
    hidden = hidden or Fx.match_hidden(sp["channels"], sp["classes"], readout)
    m = Fx.build(sp["channels"], sp["classes"], core, readout, hidden=hidden)
    X, Y = sp["main"]
    Xe, Ye = sp["tune"]
    rng = np.random.default_rng(seed)
    if x_transform is not None:
        X, Xe = x_transform(X, rng), x_transform(Xe, rng)
    if y_shuffle:
        Y = Y[torch.tensor(rng.permutation(len(Y)))]
    E.fit(m, None, X, Y, train_groups=["core", "readout"], steps=steps, lr=exp1.LR, rng=rng, batch=exp1.BATCH)
    return float(E.evaluate(m, None, Xe, Ye))


def _shuffle_time(x, rng):
    return x[:, torch.tensor(rng.permutation(x.shape[1]))]


# ---------------------------------------------------------------- E1


def e1_mutations(bedname: str, steps: int = 600) -> dict:
    sp = exp1.splits_with_units(bedname, 0)
    out = {}

    real = [_fit_eval(sp, "fast", "linear", s, steps) - _fit_eval(sp, "pooled", "linear", s, steps)
            for s in SEEDS]
    shuffled = [_fit_eval(sp, "fast", "linear", s, steps, x_transform=_shuffle_time)
                - _fit_eval(sp, "pooled", "linear", s, steps, x_transform=_shuffle_time) for s in SEEDS]
    out["destroying_temporal_order_destroys_the_core_effect"] = {
        "prediction": "if the core effect is about order, shuffling the time axis must remove most of it",
        "real_effect": round(float(np.mean(real)), 5),
        "mutated_effect": round(float(np.mean(shuffled)), 5),
        "rejected": float(np.mean(shuffled)) < 0.5 * float(np.mean(real)),
    }

    tripled = Fx.match_hidden(sp["channels"], sp["classes"], "linear") * 3
    big = [_fit_eval(sp, "fast", "linear", s, steps) - _fit_eval(sp, "pooled", "linear", s, steps, hidden=tripled)
           for s in SEEDS]
    out["tripling_the_control_capacity_does_not_remove_the_core_effect"] = {
        "prediction": "if the core effect is capacity, a three times wider order free control must close it",
        "real_effect": round(float(np.mean(real)), 5),
        "mutated_effect": round(float(np.mean(big)), 5),
        "control_hidden_width": tripled,
        "rejected": float(np.mean(big)) > 0.5 * float(np.mean(real)),
    }

    lab = [_fit_eval(sp, "fast", "linear", s, steps, y_shuffle=True)
           - _fit_eval(sp, "pooled", "linear", s, steps, y_shuffle=True) for s in SEEDS]
    out["shuffling_labels_destroys_the_core_effect"] = {
        "prediction": "with permuted labels there is nothing to learn, so no arm may separate",
        "real_effect": round(float(np.mean(real)), 5),
        "mutated_effect": round(float(np.mean(lab)), 5),
        "rejected": abs(float(np.mean(lab))) < 0.1 * abs(float(np.mean(real))),
    }
    out["all_rejected"] = all(v["rejected"] for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------- E4


def e4_mutations(bedname: str) -> dict:
    out = {}

    def gain(shift: bool, evaluate_on_old: bool = False):
        vals = []
        for s in SEEDS:
            ctx = exp4.contexts(bedname, s, shift=shift)
            m, _ = exp4.pretrain(ctx, s)
            snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
            Xe, Ye, _ = ctx["A_eval"] if evaluate_on_old else ctx["B_eval"]
            base = exp4.acc(m, Xe, Ye)
            exp4.adapt(m, "state_only", ctx, s)
            vals.append(exp4.acc(m, Xe, Ye) - base)
            m.load_state_dict(snap)
        return float(np.mean(vals))

    real = gain(shift=True)
    out["removing_the_covariate_shift_removes_the_state_only_gain"] = {
        "prediction": "the state adaptation exists to absorb a shift, so with no shift there is nothing to absorb",
        "real_effect": round(real, 5),
        "mutated_effect": round(gain(shift=False), 5),
    }
    out["removing_the_covariate_shift_removes_the_state_only_gain"]["rejected"] = (
        out["removing_the_covariate_shift_removes_the_state_only_gain"]["mutated_effect"] < 0.5 * real
    )

    out["adapting_to_the_new_context_does_not_help_the_old_one"] = {
        "prediction": "a state aimed at the new context must not improve the unshifted context",
        "real_effect": round(real, 5),
        "mutated_effect": round(gain(shift=True, evaluate_on_old=True), 5),
    }
    out["adapting_to_the_new_context_does_not_help_the_old_one"]["rejected"] = (
        out["adapting_to_the_new_context_does_not_help_the_old_one"]["mutated_effect"] < 0.25 * real
    )

    zero = []
    for s in SEEDS:
        ctx = exp4.contexts(bedname, s)
        m, _ = exp4.pretrain(ctx, s)
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        L.adapt_state(m, ctx["B_train"][0], np.random.default_rng(s), exp4.BATCH, exp4.ADAPT_STEPS)
        zero.append(max(float((p.detach() - before[n]).abs().max()) for n, p in m.named_parameters()))
    out["state_adaptation_changes_no_parameter"] = {
        "prediction": "the treatment claims zero parameter updates, so the largest parameter delta must be zero",
        "real_effect": 0.0,
        "mutated_effect": round(float(max(zero)), 12),
        "rejected": max(zero) == 0.0,
    }
    out["all_rejected"] = all(v["rejected"] for v in out.values() if isinstance(v, dict))
    return out


def main():
    t0 = time.time()
    e1 = {b: e1_mutations(b) for b in exp1.BEDS}
    e4 = {b: e4_mutations(b) for b in exp4.BEDS}
    doc = {
        "schema": "mop-positive-mutation-suite/v1",
        "rule": "a mutation is rejected when the result moves the way the alternative explanation says it must not",
        "seeds": list(SEEDS),
        "E1": e1,
        "E4": e4,
        "all_rejected": all(v["all_rejected"] for v in list(e1.values()) + list(e4.values())),
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_POSITIVE_MUTATION_SUITE.json", doc)
    print(f"mutations: all_rejected={doc['all_rejected']} in {doc['wall_seconds']}s", flush=True)
    for exp, per in (("E1", e1), ("E4", e4)):
        for b, r in per.items():
            for k, v in r.items():
                if isinstance(v, dict) and not v["rejected"]:
                    print(f"  SURVIVED {exp} {b} {k}: real {v['real_effect']} mutated {v['mutated_effect']}",
                          flush=True)
    print("MUTATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
