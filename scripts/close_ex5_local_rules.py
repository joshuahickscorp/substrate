"""Closure control for ex5_local_rules_scale: is "local rules beat backprop" an Adam artifact?

Standing contention
-------------------
ex5_local_rules_scale found that on a domain-incremental continual stream the local rules
(feedback_alignment, predictive_coding) beat persistent backprop on BOTH final accuracy and
backward transfer (they forget less). The shipped table at the primary hidden width (64):

    backprop_adam        acc 0.336   bwt -0.723
    feedback_alignment   acc 0.497   bwt -0.554
    predictive_coding    acc 0.398   bwt -0.637

The standing suspicion: backprop in ex5 is trained with Adam (torch.optim.Adam, lr=0.05),
whose adaptive per-parameter normalization and momentum carry optimizer state ACROSS task
boundaries and take a large, roughly gradient-magnitude-independent effective step. The
local rules use a plain delta update W -= lr * grad (no state, no normalization) at the same
NOMINAL lr=0.05 but a much smaller EFFECTIVE per-step update magnitude. So backprop's extra
catastrophic forgetting may be an Adam artifact (big, state-compounding steps that overwrite
prior domains), not a property of "global backprop" per se.

Decisive control (this script)
------------------------------
Add a FOURTH arm, backprop_sgd: identical 2-layer architecture and identical schedule as the
backprop arm, but torch.optim.SGD (momentum=0, no Adam state) instead of Adam. Its learning
rate is not left at the nominal 0.05 (which, measured, gives a ~100x SMALLER per-step update
than Adam and a ~10-20x smaller update than the local rules); instead it is CALIBRATED per
(seed, hidden) so that backprop_sgd's measured mean per-step weight-update L2 magnitude matches
the mean per-step update L2 of the two local rules on that stream. That is exactly the
"matched effective step size" the contention asks for: same optimizer family as the local
rules (plain SGD, no state), same effective update magnitude, only the credit assignment
(exact global backprop gradient vs local/random-feedback surrogate) differs.

How the match is done (documented honestly)
-------------------------------------------
1. On task 0's train split we measure, over the epoch budget, the mean L2 norm of the applied
   weight update for feedback_alignment and for predictive_coding, and take their mean as the
   TARGET effective update magnitude.
2. We measure plain-SGD backprop's mean update L2 at a probe lr (the nominal 0.05). Because a
   plain-SGD update is exactly lr * grad, update magnitude is linear in lr, so the calibrated
   lr = probe_lr * (target / probe_update_magnitude). We clamp to a sane range and re-verify
   the achieved magnitude, reporting the calibrated lr and the achieved/target ratio.
This calibration is done ONCE per (seed, hidden) on task 0 only (cheap, no peeking at BWT),
then frozen for the whole stream, so the comparison stays matched-budget and honest.

Verdict logic
-------------
Compare final accuracy and BWT across backprop_adam, backprop_sgd, feedback_alignment,
predictive_coding at matched data/seed/budget over >=3 seeds.
  * If backprop_sgd closes most of the gap (forgets much less than backprop_adam and now ties
    or beats the local rules on accuracy AND bwt) => the ex5 finding was largely an Adam
    artifact (resolution: refuted/reframed).
  * If the local rules STILL beat backprop_sgd on accuracy and bwt => the finding is robust to
    matching the optimizer (resolution: survives).

Reuses ex5_local_rules_scale wholesale: _init_model, the three train_step/evaluate pairs,
the continual-stream + anchor-subset + backward_transfer harness shape, _anchor_indices,
ContinualResult, make_task_stream. Only a fourth arm (backprop with SGD) and the lr
calibration are added here; the shipped module is not modified.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only). No
sentience or agency language.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mop.experiments.ex5_local_rules_scale import (
    _anchor_indices,
    _backprop_evaluate,
    _backprop_train_step,
    _fa_evaluate,
    _fa_train_step,
    _init_model,
    _pc_evaluate,
    _pc_train_step,
)
from mop.metrics import ContinualResult
from mop.seeding import seed_everything
from mop.substrate.datasets import make_task_stream

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "runs" / "pre_studio" / "close_ex5_local_rules.json"
OUT_PATH = DEFAULT_OUT

N_TASKS = 40
DIM = 48
CLASSES_PER_TASK = 4
SAMPLES_PER_TASK = 80
SEPARATION = 0.6
HIDDEN = 64
EPOCHS_PER_TASK = 60
NOMINAL_LR = 0.05  # shipped ex5 lr; used as-is for backprop_adam, fa, pc, and as the SGD probe lr
N_ANCHORS = 8
SEEDS = [0, 1, 2, 3, 4]

LOCAL_ARMS = ("feedback_alignment", "predictive_coding")
TRAIN_STEP = {
    "backprop_adam": _backprop_train_step,
    "backprop_sgd": _backprop_train_step,
    "feedback_alignment": _fa_train_step,
    "predictive_coding": _pc_train_step,
}
EVALUATE = {
    "backprop_adam": _backprop_evaluate,
    "backprop_sgd": _backprop_evaluate,
    "feedback_alignment": _fa_evaluate,
    "predictive_coding": _pc_evaluate,
}


def _make_backprop_model(dim: int, hidden: int, n_classes: int, seed: int, optimizer: str, lr: float):
    model = _init_model("backprop", dim, hidden, n_classes, seed)
    if optimizer == "adam":
        for g_ in model.opt.param_groups:
            g_["lr"] = lr
    elif optimizer == "sgd":
        model.opt = torch.optim.SGD(model.params, lr=lr, momentum=0.0)
    else:
        raise ValueError(optimizer)
    return model


def _mean_update_l2(before: list[torch.Tensor], after: list[torch.Tensor]) -> float:
    delta = torch.cat([(a.detach() - b.detach()).flatten() for a, b in zip(after, before, strict=False)])
    return float(delta.norm())


def _measure_local_target(dim, hidden, n_classes, seed, xtr, ytr, epochs, lr) -> dict:
    out = {}
    for arm, train_step, init_name in (
        ("feedback_alignment", _fa_train_step, "feedback_alignment"),
        ("predictive_coding", _pc_train_step, "predictive_coding"),
    ):
        seed_everything(seed)
        m = _init_model(init_name, dim, hidden, n_classes, seed)
        norms = []
        for _ in range(epochs):
            before = [m.W1.clone(), m.b1.clone(), m.W2.clone(), m.b2.clone()]
            train_step(m, xtr, ytr, lr)
            after = [m.W1, m.b1, m.W2, m.b2]
            norms.append(_mean_update_l2(before, after))
        out[arm] = sum(norms) / len(norms)
    out["target"] = (out["feedback_alignment"] + out["predictive_coding"]) / 2.0
    return out


def _calibrate_sgd_lr(dim, hidden, n_classes, seed, xtr, ytr, epochs, probe_lr, target) -> dict:
    seed_everything(seed)
    m = _make_backprop_model(dim, hidden, n_classes, seed, "sgd", probe_lr)
    norms = []
    for _ in range(epochs):
        before = [p.detach().clone() for p in m.params]
        _backprop_train_step(m, xtr, ytr, probe_lr)
        norms.append(_mean_update_l2(before, m.params))
    probe_mag = sum(norms) / len(norms)
    raw_lr = probe_lr * (target / probe_mag) if probe_mag > 0 else probe_lr
    cal_lr = float(min(max(raw_lr, 1e-4), 5.0))
    seed_everything(seed)
    mv = _make_backprop_model(dim, hidden, n_classes, seed, "sgd", cal_lr)
    vnorms = []
    for _ in range(epochs):
        before = [p.detach().clone() for p in mv.params]
        _backprop_train_step(mv, xtr, ytr, cal_lr)
        vnorms.append(_mean_update_l2(before, mv.params))
    achieved = sum(vnorms) / len(vnorms)
    return {
        "probe_lr": probe_lr,
        "probe_update_l2": probe_mag,
        "target_update_l2": target,
        "calibrated_lr": cal_lr,
        "achieved_update_l2": achieved,
        "achieved_over_target": achieved / target if target > 0 else float("nan"),
    }


def _run_arm_on_stream(arm, tasks, dim, hidden, n_classes, epochs, seed, anchor_idx, lr, optimizer):
    seed_everything(seed)
    if arm in ("backprop_adam", "backprop_sgd"):
        model = _make_backprop_model(dim, hidden, n_classes, seed, optimizer, lr)
    else:
        model = _init_model(arm, dim, hidden, n_classes, seed)
    train_step = TRAIN_STEP[arm]
    evaluate = EVALUATE[arm]

    anchor_pos = {j: k for k, j in enumerate(anchor_idx)}
    K = len(anchor_idx)
    R: list[list[float]] = [[0.0] * K for _ in range(K)]

    def held_out(task):
        x, y = task.x, task.y
        cut = max(1, int(x.shape[0] * 0.8))
        xte, yte = x[cut:], y[cut:]
        return (xte, yte) if xte.shape[0] > 0 else (x[:cut], y[:cut])

    t0 = time.perf_counter()
    for i, task in enumerate(tasks):
        x, y = task.x, task.y
        cut = max(1, int(x.shape[0] * 0.8))
        xtr, ytr = x[:cut], y[:cut]
        for _ in range(epochs):
            train_step(model, xtr, ytr, lr)
        if i in anchor_pos:
            k = anchor_pos[i]
            xte, yte = held_out(task)
            R[k][k] = evaluate(model, xte, yte)
    for j in anchor_idx:
        k = anchor_pos[j]
        xte, yte = held_out(tasks[j])
        R[-1][k] = evaluate(model, xte, yte)
    elapsed = time.perf_counter() - t0
    return ContinualResult(R=R, chance=1.0 / n_classes), elapsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"receipt output path (default: {DEFAULT_OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_path: Path = args.out
    n_classes = CLASSES_PER_TASK  # domain-incremental: shared label space
    anchor_idx = _anchor_indices(N_TASKS, N_ANCHORS)
    arms = ("backprop_adam", "backprop_sgd", "feedback_alignment", "predictive_coding")

    per_seed = {arm: {"acc": [], "bwt": [], "sec": []} for arm in arms}
    calibrations = []

    for s in SEEDS:
        tasks = make_task_stream(
            n_tasks=N_TASKS,
            dim=DIM,
            classes_per_task=CLASSES_PER_TASK,
            samples_per_task=SAMPLES_PER_TASK,
            separation=SEPARATION,
            incremental="domain",
            seed=s,
        )
        t0 = tasks[0]
        cut = max(1, int(t0.x.shape[0] * 0.8))
        xtr, ytr = t0.x[:cut], t0.y[:cut]
        local_target = _measure_local_target(DIM, HIDDEN, n_classes, s, xtr, ytr, EPOCHS_PER_TASK, NOMINAL_LR)
        cal = _calibrate_sgd_lr(
            DIM, HIDDEN, n_classes, s, xtr, ytr, EPOCHS_PER_TASK, NOMINAL_LR, local_target["target"]
        )
        cal.update({"seed": s, "local_update_l2": {k: local_target[k] for k in LOCAL_ARMS}})
        calibrations.append(cal)

        lr_for = {
            "backprop_adam": NOMINAL_LR,
            "backprop_sgd": cal["calibrated_lr"],
            "feedback_alignment": NOMINAL_LR,
            "predictive_coding": NOMINAL_LR,
        }
        opt_for = {"backprop_adam": "adam", "backprop_sgd": "sgd"}

        for arm in arms:
            result, elapsed = _run_arm_on_stream(
                arm,
                tasks,
                DIM,
                HIDDEN,
                n_classes,
                EPOCHS_PER_TASK,
                s,
                anchor_idx,
                lr_for[arm],
                opt_for.get(arm, ""),
            )
            summ = result.summary()
            per_seed[arm]["acc"].append(summ["avg_accuracy"])
            per_seed[arm]["bwt"].append(summ["backward_transfer"])
            per_seed[arm]["sec"].append(elapsed)

    def mean(xs):
        return sum(xs) / len(xs)

    def std(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    table = {}
    for arm in arms:
        table[arm] = {
            "acc_mean": mean(per_seed[arm]["acc"]),
            "acc_std": std(per_seed[arm]["acc"]),
            "bwt_mean": mean(per_seed[arm]["bwt"]),
            "bwt_std": std(per_seed[arm]["bwt"]),
            "seconds": mean(per_seed[arm]["sec"]),
            "acc_per_seed": per_seed[arm]["acc"],
            "bwt_per_seed": per_seed[arm]["bwt"],
        }

    adam = table["backprop_adam"]
    sgd = table["backprop_sgd"]
    fa = table["feedback_alignment"]
    pc = table["predictive_coding"]
    best_local_acc = max(fa["acc_mean"], pc["acc_mean"])
    best_local_bwt = max(fa["bwt_mean"], pc["bwt_mean"])  # less-negative bwt = forgets less

    adam_local_acc_gap = best_local_acc - adam["acc_mean"]
    sgd_recovered_acc = sgd["acc_mean"] - adam["acc_mean"]
    acc_gap_closed_frac = (
        (sgd_recovered_acc / adam_local_acc_gap) if abs(adam_local_acc_gap) > 1e-9 else float("nan")
    )
    adam_local_bwt_gap = best_local_bwt - adam["bwt_mean"]
    sgd_recovered_bwt = sgd["bwt_mean"] - adam["bwt_mean"]
    bwt_gap_closed_frac = (
        (sgd_recovered_bwt / adam_local_bwt_gap) if abs(adam_local_bwt_gap) > 1e-9 else float("nan")
    )

    sgd_ties_or_beats_local_acc = sgd["acc_mean"] >= best_local_acc - 0.01
    sgd_ties_or_beats_local_bwt = sgd["bwt_mean"] >= best_local_bwt - 0.01
    sgd_forgets_much_less_than_adam = sgd["bwt_mean"] - adam["bwt_mean"] >= 0.05
    local_still_beat_sgd_acc = best_local_acc - sgd["acc_mean"] > 0.02
    local_still_beat_sgd_bwt = best_local_bwt - sgd["bwt_mean"] > 0.02

    if sgd_forgets_much_less_than_adam and sgd_ties_or_beats_local_acc and sgd_ties_or_beats_local_bwt:
        resolution = "refuted"
        verdict = (
            "Adam artifact confirmed: plain-SGD backprop at matched effective step forgets much "
            "less than Adam backprop and now ties or beats the local rules on accuracy and BWT."
        )
    elif sgd_forgets_much_less_than_adam and (local_still_beat_sgd_acc or local_still_beat_sgd_bwt):
        resolution = "reframed"
        verdict = (
            "Partly Adam artifact: matched-step plain-SGD backprop forgets much less than Adam, "
            "but the local rules still lead on at least one of accuracy or BWT, so the ex5 "
            "finding is real yet smaller than the Adam-vs-SGD comparison implied."
        )
    elif local_still_beat_sgd_acc and local_still_beat_sgd_bwt:
        resolution = "survives"
        verdict = (
            "Finding robust to matching the optimizer: even plain-SGD backprop at the local "
            "rules' effective step size still loses to the local rules on both accuracy and BWT."
        )
    else:
        resolution = "inconclusive"
        verdict = (
            "Mixed: matched-step plain-SGD backprop neither clearly recovers the gap nor clearly "
            "loses to the local rules on both axes."
        )

    out = {
        "id": "ex5_local_rules_scale",
        "control": "backprop_sgd (plain SGD, momentum=0) at lr calibrated to the local rules' "
        "effective per-step update magnitude, added as a fourth arm",
        "config": {
            "n_tasks": N_TASKS,
            "dim": DIM,
            "classes_per_task": CLASSES_PER_TASK,
            "samples_per_task": SAMPLES_PER_TASK,
            "separation": SEPARATION,
            "hidden": HIDDEN,
            "epochs_per_task": EPOCHS_PER_TASK,
            "nominal_lr": NOMINAL_LR,
            "n_anchors": N_ANCHORS,
            "seeds": SEEDS,
            "incremental": "domain",
        },
        "how_lr_matched": (
            "backprop_sgd uses plain torch.optim.SGD (momentum=0), whose applied update is "
            "exactly lr*grad and thus linear in lr. Per seed, on task 0 only, we measure the mean "
            "per-step applied-update L2 of feedback_alignment and predictive_coding, take their "
            "mean as the target, measure SGD's update L2 at the probe lr (0.05), then set "
            "calibrated_lr = probe_lr * (target / probe_magnitude) and re-verify. This matches "
            "backprop_sgd's EFFECTIVE step magnitude to the local rules, isolating credit "
            "assignment (exact backprop gradient vs local surrogate) from optimizer state/scale."
        ),
        "calibrations_per_seed": calibrations,
        "calibrated_sgd_lr_mean": mean([c["calibrated_lr"] for c in calibrations]),
        "table": table,
        "adam_vs_sgd_bwt_delta": sgd["bwt_mean"] - adam["bwt_mean"],
        "acc_gap_closed_fraction_by_sgd": acc_gap_closed_frac,
        "bwt_gap_closed_fraction_by_sgd": bwt_gap_closed_frac,
        "best_local_acc": best_local_acc,
        "best_local_bwt": best_local_bwt,
        "resolution": resolution,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("=== close_ex5_local_rules ===")
    print(f"seeds={SEEDS}  n_tasks={N_TASKS}  hidden={HIDDEN}  epochs/task={EPOCHS_PER_TASK}")
    print(f"calibrated SGD lr (mean over seeds): {out['calibrated_sgd_lr_mean']:.4f}")
    print(f"{'arm':<20} {'acc':>14} {'bwt':>16}")
    for arm in arms:
        r = table[arm]
        print(
            f"{arm:<20} {r['acc_mean']:.3f}+/-{r['acc_std']:.3f}   {r['bwt_mean']:+.3f}+/-{r['bwt_std']:.3f}"
        )
    print(f"adam->sgd bwt delta: {out['adam_vs_sgd_bwt_delta']:+.3f} (positive = SGD forgets less than Adam)")
    print(
        f"acc gap closed by SGD: {acc_gap_closed_frac:.2f}   bwt gap closed by SGD: {bwt_gap_closed_frac:.2f}"
    )
    print(f"resolution: {resolution}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
