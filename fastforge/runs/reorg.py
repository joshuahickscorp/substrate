"""Functional reorganization and domain inference.

Two separate questions, deliberately kept apart.

Reorganization  does changing what updates, in response to measured interference, beat changing it according
                to the known domain label. Switching modules on a known label is a simple control, not
                self reorganization, so the label rule is the bar, not the baseline.

Context inference  when every context has the same input shape, can the active module be chosen without
                being told which context it is. This is run within a domain, because across domains the
                channel count gives the answer away and the question becomes trivial.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from fastforge import arch as A
from fastforge import engine as E
from fastforge import sequence as S
from fastforge.runs import crossdomain as CD
from fastforge.runs import io
from fastforge.within import _contexts

SEEDS = [0, 1, 2, 3, 4]
ROUTING = [
    "true_label",
    "simple_observation_classifier",
    "task_free_prototype",
    "task_free_confidence",
    "random",
    "wrong_context",
    "shuffled_context",
    "oracle_per_sample",
]

# how each cross domain arm decides what may change
POLICY_CLASS = {
    "G1_frozen_after_first": "fixed_domain_label_switching",
    "G2_reopened_at_return": "fixed_domain_label_switching",
    "H_task_label_freeze": "fixed_domain_label_switching",
    "G0_always_trainable": "fixed_schedule",
    "H_always_update": "fixed_schedule",
    "H_never_update": "fixed_schedule",
    "G3_reopened_on_drop": "measured_interference_response",
    "H_cosine_gate": "measured_interference_response",
    "H_probe_gate": "measured_interference_response",
    "H_drift_gate": "measured_interference_response",
    "H_perf_gate": "measured_interference_response",
    "H_random_gate": "random_switching",
    "H_shuffled_gate": "shuffled_switching",
    "H_wrong_domain_gate": "wrong_domain_switching",
    "G6_oracle_schedule": "oracle_switching",
    "H_oracle_schedule": "oracle_switching",
}


# ---------------------------------------------------------------- part one, reorganization


def reorganization():
    rows = {}
    for direction in CD.DIRECTIONS:
        dname = f"{direction[0]}->{direction[1]}"
        per = {}
        for s in CD.SEEDS:
            f = io.RUNS / "crossdomain" / f"{direction[0]}-{direction[1]}_{s}.json"
            if not f.is_file():
                return None
            per[s] = json.loads(f.read_text())
        rows[dname] = per
    out = {}
    for dname, per in rows.items():
        seeds = sorted(per)

        def util(arm, s, per=per):
            m = per[s][arm]["metrics"]
            return float(np.mean([m["second_acquisition"], m["return_recovery"], m["future_adaptation"]]))

        by_class = {}
        for arm, cls in POLICY_CLASS.items():
            if arm not in per[seeds[0]]:
                continue
            by_class.setdefault(cls, {})[arm] = [util(arm, s) for s in seeds]
        best = {cls: max(arms, key=lambda a: float(np.mean(arms[a]))) for cls, arms in by_class.items()}
        label_arm = best["fixed_domain_label_switching"]
        out[dname] = {
            "policy_class_best_arm": best,
            "policy_class_utility": {cls: round(float(np.mean(by_class[cls][best[cls]])), 4) for cls in best},
            "measured_interference_vs_fixed_label": E.effect(
                by_class["measured_interference_response"][best["measured_interference_response"]],
                by_class["fixed_domain_label_switching"][label_arm],
            ),
            "random_switching_vs_fixed_label": E.effect(
                by_class["random_switching"][best["random_switching"]],
                by_class["fixed_domain_label_switching"][label_arm],
            ),
            "wrong_domain_vs_fixed_label": E.effect(
                by_class["wrong_domain_switching"][best["wrong_domain_switching"]],
                by_class["fixed_domain_label_switching"][label_arm],
            ),
            "oracle_vs_fixed_label": E.effect(
                by_class["oracle_switching"][best["oracle_switching"]],
                by_class["fixed_domain_label_switching"][label_arm],
            ),
        }
        out[dname]["credited"] = out[dname]["measured_interference_vs_fixed_label"]["lower_95_cb"] >= io.SESOI
    return out


# ---------------------------------------------------------------- part two, context inference


def _train_contexts(dname, seed, steps):
    ctx, sp = _contexts(dname, seed)
    torch.manual_seed(1000 + seed)
    model = A.build("G", {c["name"]: (sp["channels"], sp["classes"]) for c in ctx})
    rng = np.random.default_rng(seed)
    for c in ctx:
        E.fit(
            model,
            c["name"],
            c["x"],
            c["y"],
            train_groups=S.local_groups(model, c["name"]) + ["fast_core"],
            steps=steps,
            lr=S.lr_for(dname),
            rng=rng,
        )
    return model, ctx


@torch.no_grad()
def _route(model, ctx, X, rule, seed, protos=None, clf=None):
    """Return the chosen context index per sample."""
    n, k = len(X), len(ctx)
    rng = np.random.default_rng(700 + seed)
    if rule == "random":
        return torch.tensor(rng.integers(0, k, n))
    if rule == "simple_observation_classifier":
        f = torch.cat([X.mean(1), X.std(1), X.amax(1)], 1)
        return clf(f).argmax(1)
    if rule == "task_free_prototype":
        # prototypes and queries must live in one space. Each context owns its own projection, so a
        # prototype built with context c's projection is not comparable to a query encoded with context 0's.
        # One reference encoder is used for both sides.
        h = model.features(X, ctx[0]["name"])[:, -1]
        d = torch.stack([(h - p).norm(dim=1) for p in protos])
        return d.argmin(0)
    if rule == "task_free_confidence":
        logits = torch.stack([torch.softmax(model(X, c["name"])[0], 1).amax(1) for c in ctx])
        return logits.argmax(0)
    return torch.zeros(n, dtype=torch.long)


def context_inference(dname, seed, steps):
    model, ctx = _train_contexts(dname, seed, steps)
    k = len(ctx)
    Xs = [c["xte"][:400] for c in ctx]
    Ys = [c["yte"][:400] for c in ctx]
    true = torch.cat([torch.full((len(x),), i, dtype=torch.long) for i, x in enumerate(Xs)])
    X = torch.cat(Xs)
    Y = torch.cat(Ys)

    # simple supervised observation classifier over context labels
    with torch.no_grad():
        trf = torch.cat(
            [torch.cat([c["x"][:400].mean(1), c["x"][:400].std(1), c["x"][:400].amax(1)], 1) for c in ctx]
        )
    trl = torch.cat([torch.full((min(400, len(c["x"])),), i, dtype=torch.long) for i, c in enumerate(ctx)])
    clf = torch.nn.Linear(trf.shape[1], k)
    opt = torch.optim.Adam(clf.parameters(), 3e-3)
    rng = np.random.default_rng(seed)
    for _ in range(300):
        bi = rng.choice(len(trf), min(256, len(trf)), replace=False)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(clf(trf[bi]), trl[bi]).backward()
        opt.step()
    with torch.no_grad():
        ref = ctx[0]["name"]  # one reference encoder for every prototype and every query
        protos = [model.features(c["x"][:400], ref)[:, -1].mean(0) for c in ctx]
        per_ctx_logits = torch.stack([model(X, c["name"])[0] for c in ctx])  # (k, n, classes)

    res = {}
    for rule in ROUTING:
        if rule == "true_label":
            choice = true
        elif rule == "wrong_context":
            choice = (true + 1) % k
        elif rule == "shuffled_context":
            choice = true[torch.tensor(np.random.default_rng(9 + seed).permutation(len(true)))]
        elif rule == "oracle_per_sample":
            correct = per_ctx_logits.argmax(2) == Y.unsqueeze(0)
            choice = torch.where(correct.any(0), correct.float().argmax(0), torch.zeros_like(true))
        else:
            choice = _route(model, ctx, X, rule, seed, protos, clf)
        pred = per_ctx_logits[choice, torch.arange(len(Y))].argmax(1)
        res[rule] = {
            "routing_accuracy": float((choice == true).float().mean()),
            "downstream_accuracy": float((pred == Y).float().mean()),
        }
    return res


def main():
    t0 = time.time()
    reorg = reorganization()
    if reorg is None:
        print("reorganization waiting on cross domain shards", flush=True)
    else:
        io.seal(
            "MOP_FUNCTIONAL_REORGANIZATION_REPORT.json",
            {
                "schema": "mop-functional-reorganization/v1",
                "bar": "a reorganization policy is credited only if it beats fixed domain label switching by "
                "SESOI at the lower confidence bound. Switching on a known label is a control.",
                "policy_classes": POLICY_CLASS,
                "per_direction": reorg,
                "verdict": (
                    "functional_reorganization_positive"
                    if all(v["credited"] for v in reorg.values())
                    else "functional_reorganization_null"
                ),
            },
        )
        print("reorganization:", {d: v["credited"] for d, v in reorg.items()}, flush=True)

    rows = {}
    for dname in ("har", "speech"):
        steps = max(60, S.budget(dname) // 2)
        per = [context_inference(dname, s, steps) for s in SEEDS]
        rows[dname] = {
            r: {k: round(float(np.mean([p[r][k] for p in per])), 4) for k in per[0][r]} for r in ROUTING
        }
        print(f"  context inference {dname} done", flush=True)
    task_free = ["task_free_prototype", "task_free_confidence"]
    beats = all(
        max(rows[d][r]["downstream_accuracy"] for r in task_free)
        > rows[d]["simple_observation_classifier"]["downstream_accuracy"] + io.SESOI
        for d in rows
    )
    shuffled_ok = all(
        max(rows[d][r]["downstream_accuracy"] for r in task_free)
        > rows[d]["shuffled_context"]["downstream_accuracy"] + io.SESOI
        for d in rows
    )
    io.seal(
        "MOP_TASK_FREE_CONTEXT_REPORT.json",
        {
            "schema": "mop-task-free-context/v1",
            "seeds": SEEDS,
            "why_within_domain": "across domains the channel count identifies the domain, so cross domain "
            "context inference is not a question. Within a domain every context shares an "
            "input shape, so the question is real.",
            "routing_rules": ROUTING,
            "results": rows,
            "beats_simple_classifier": beats,
            "shuffled_context_rejected": shuffled_ok,
            "verdict": "task_free_context_positive" if (beats and shuffled_ok) else "task_free_context_null",
            "note": "domain inference is not allowed to be the only adaptive component: the routing "
            "comparison "
            "holds the trained modules fixed and varies only the choice rule",
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print("task free context:", beats, shuffled_ok, flush=True)
    print("REORG_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
