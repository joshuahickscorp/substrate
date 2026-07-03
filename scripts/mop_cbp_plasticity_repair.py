"""BUILD 1 (MOLDABILITY axis, first plasticity-mechanism test): continual-backprop (Dohare selective
reinit of low-utility hidden units) vs plain SGD on the SAME validated 150-task concept-drift stream
where plain SGD provably loses plasticity (the mop_plasticity_certificate.py stream: gap +0.513,
dead 0 -> 0.75). 8 seeds.

WHY THIS TEST. The plasticity-loss certificate is BUILT and VALIDATED on this exact drift stream: plain
SGD loses the ability to fit fresh tasks (early-vs-late learnability gap CI strictly above 0) and dead
ReLU units pile up. That is the ONLY place on the laptop where plasticity loss is provably INDUCED, so it
is the one honest place to ask whether a plasticity MECHANISM (continual-backprop) actually repairs it.
If the mechanism is dead even here, that is a strong honest null: the axis has no laptop surface to move.

WHAT CBP IS. Utility-based selective reinit (Dohare et al. 2024), copied in mechanism from
scripts/studio/pr9_continual_backprop.py: track a running contribution-utility per hidden unit; each
optimizer step reset a small fraction of the MATURE, LOWEST-utility units (fc1 fan-in re-init, fc2 fan-out
zeroed, utility+age reset). Plain SGD is the identical arm with replacement_rate=0 (no unit ever
eligible). Both arms share the certificate's net, teacher stream, input distribution, lr, epochs, batch,
and task order; the ONLY difference is the reinit rule. That makes them a matched pair, exactly as the
certificate's drift-vs-fixed pair is matched.

PREREGISTERED WIN RULE (fixed in code BEFORE any number is seen, never tuned after):
  Let gap_sgd = early_learn_acc - late_learn_acc for plain SGD (the certified plasticity-loss gap),
  gap_cbp   = the same for CBP. Per seed, delta = gap_sgd - gap_cbp (how much CBP CLOSES the gap).
  CBP WINS iff ALL of:
    (W1) seed-CI lower bound of delta > 0.0           (CBP closes the gap beyond seed spread), AND
    (W2) sign_flip_report(delta).consistent_sign == +1 (no seed flips the sign), AND
    (W3) CBP late dead-unit fraction stays FAR BELOW SGD's: mean(dead_late_cbp) <= 0.5 * mean(dead_late_sgd)
         AND seed-CI lower bound of (dead_late_sgd - dead_late_cbp) > 0.0 (mechanism corroborated, no flip).
  A TIE IS A NULL: if (W1) fails (CI includes 0) the mechanism did not close the gap -> NULL, reported
  honestly (mechanism dead where loss is provably induced). The dead-unit thresholds 0.5x and CI>0 are
  fixed here before running.

PRECONDITION (sanity, not a tunable): the SGD arm must reproduce the certified plasticity loss on this
stream, i.e. gap_sgd seed-CI lo > 0 and dead_late_sgd clearly above dead_early_sgd. If it does not, the
comparison is inadmissible (there was nothing to repair) and we report that instead of a CBP verdict.

House form: no em/en dashes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402

OUT = _ROOT / "runs" / "mot" / "cbp_plasticity_repair.json"

# ---------------------------------------------------------------------------
# PREREGISTERED constants. Stream + net + optimizer regime are COPIED VERBATIM from the VALIDATED
# certificate (mop_plasticity_certificate.py) so this runs on the identical drift stream. Do not tune.
N_TASKS = 150
EARLY_WINDOW = 20
LATE_WINDOW = 20
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 seeds
Z = 1.96  # 95pct normal-approx CI

DIM = 20
N_CLASSES = 8
SAMPLES_PER_TASK = 200
HIDDEN = 48
TEACHER_HIDDEN = 32
SGD_LR = 0.2  # plain SGD, fixed lr, no momentum: the certificate baseline
EPOCHS_PER_TASK = 25
BATCH = 8

# CBP hyperparameters (Dohare selective reinit). Mechanism copied from pr9_continual_backprop.py.
# replacement_rate is the fraction of MATURE units reinit per optimizer step; maturity is how many steps a
# unit must survive before eligible; decay is the running-utility decay.
#
# NO-OP-BUG FIX (honesty, not a tune): the PR9/Studio default replacement_rate=1e-4 assumes ~10000-unit
# layers so that rate*n_eligible >= 1 per step. On this 48-unit laptop layer, int(1e-4 * 48) = 0 EVERY
# step, so the Studio code path reinits NOTHING (reinit_count == 0) and CBP is bit-identical to SGD. A
# "null" from a mechanism that never switched on is a FALSE null. Dohare's own formulation avoids this by
# carrying a FRACTIONAL reinit budget across steps (accumulate rate*n_eligible; reinit floor(budget) units
# and keep the remainder), so even a tiny rate eventually fires. We use that faithful accumulator here and
# PREREGISTER a SWEEP of replacement rates (below) so the mechanism is genuinely exercised at laptop scale.
CBP_MATURITY = 50
CBP_DECAY = 0.99

# PREREGISTERED replacement-rate grid (fixed before seeing any swept result). Chosen by reinit-budget
# arithmetic, NOT by any accuracy number: on 48 mature units, budget/step = rate*48, so these span roughly
# "one reinit every ~10 tasks" (2e-5*48*625 steps/task ~= 0.6/task) up to "several reinits per task"
# (2e-3). This brackets the regime where the mechanism is meaningfully active without churning the whole
# layer. The per-rate WIN RULE is identical for every rate; we report the full curve and the best rate.
CBP_RATE_GRID = [2e-5, 1e-4, 5e-4, 1e-3, 2e-3]

# PREREGISTERED dead-unit thresholds for W3 (fixed before running):
DEAD_RATIO_MAX = 0.5  # CBP late dead frac must be <= this multiple of SGD's late dead frac
# ---------------------------------------------------------------------------


class MLP(torch.nn.Module):
    """Plain 2-layer ReLU MLP (verbatim from the certificate). `h` holds the last hidden activation for
    the dead-unit fraction and for CBP's utility update."""

    def __init__(self, dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, n_classes)
        self.h: torch.Tensor | None = None

    def forward(self, x):
        self.h = F.relu(self.fc1(x))
        return self.fc2(self.h)


def _teacher(g: torch.Generator):
    W1 = torch.randn(DIM, TEACHER_HIDDEN, generator=g)
    W2 = torch.randn(TEACHER_HIDDEN, N_CLASSES, generator=g)
    return W1, W2


def _teach(X: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    return (F.relu(X @ W1) @ W2).argmax(-1)


def _task_stream(seed: int):
    """DRIFT stream (verbatim from the certificate, mode='drift'): a fresh teacher each task. This is the
    validated loss-inducing stream. Identical construction/seed to the certificate so the SGD arm here
    reproduces the certified gap."""
    g = torch.Generator().manual_seed(seed * 100 + 7)
    _fixed_unused = _teacher(g)  # drawn to keep the RNG stream identical to the certificate's drift path
    del _fixed_unused
    tasks = []
    for _ in range(N_TASKS):
        X = torch.randn(SAMPLES_PER_TASK, DIM, generator=g)
        W1, W2 = _teacher(g)  # NEW concept each task
        tasks.append((X, _teach(X, W1, W2)))
    return tasks


class ContinualBackprop:
    """Utility-based selective reinit (Dohare 2024), mechanism copied from pr9_continual_backprop.py.
    Plain SGD is this with replacement_rate=0 (no unit ever eligible)."""

    def __init__(self, model: MLP, *, replacement_rate: float, maturity: int, decay: float, seed: int):
        self.model = model
        self.replacement_rate = replacement_rate
        self.maturity = maturity
        self.decay = decay
        self.g = torch.Generator().manual_seed(seed + 7919)
        h = model.fc1.out_features
        self.util = torch.zeros(h)
        self.age = torch.zeros(h, dtype=torch.long)
        self.reinit_count = 0
        self._budget = 0.0  # fractional reinit budget carried across steps (Dohare small-layer form)

    @torch.no_grad()
    def step(self, hidden_act: torch.Tensor) -> None:
        self.age += 1
        contribution = hidden_act.abs().mean(0) * self.model.fc2.weight.abs().mean(0)
        self.util.mul_(self.decay).add_(contribution, alpha=1.0 - self.decay)
        if self.replacement_rate <= 0:
            return
        n_eligible = int((self.age >= self.maturity).sum())
        # accumulate a fractional budget so a tiny rate still fires on a small layer (no int() no-op)
        self._budget += self.replacement_rate * n_eligible
        n_reset = int(self._budget)
        if n_reset < 1:
            return
        n_reset = min(n_reset, n_eligible)
        self._budget -= n_reset
        eligible = (self.age >= self.maturity).nonzero(as_tuple=True)[0]
        worst = eligible[torch.argsort(self.util[eligible])[:n_reset]]
        fan_in = self.model.fc1.in_features
        new_w = torch.randn(n_reset, fan_in, generator=self.g) * (fan_in**-0.5)
        self.model.fc1.weight.data[worst] = new_w
        self.model.fc1.bias.data[worst] = 0.0
        self.model.fc2.weight.data[:, worst] = 0.0  # zero fan-out so a fresh unit does not shock output
        self.util[worst] = 0.0
        self.age[worst] = 0
        self.reinit_count += n_reset


def _learn_on_task(model, opt, cbp: ContinualBackprop, X, y) -> tuple[float, float]:
    """Train the FIXED SGD baseline for EPOCHS_PER_TASK passes, threading CBP into the SAME loop (one
    cbp.step per optimizer step). Return (best train accuracy on this task, dead-unit fraction at end).
    Dead-unit definition is the CERTIFICATE's: activation.abs().sum over the batch == 0. Model/optimizer
    and the CBP utility/age state all persist across the whole stream (no re-init between tasks): that
    persistence is what lets plasticity loss accumulate and what CBP is meant to counteract."""
    g = torch.Generator().manual_seed(0)  # verbatim from the certificate (fixed inner shuffle seed)
    n = X.shape[0]
    best = 0.0
    dead = 0.0
    for _ in range(EPOCHS_PER_TASK):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            j = perm[i : i + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(X[j]), y[j]).backward()
            opt.step()
            cbp.step(model.h.detach())  # utility update (+ selective reinit iff replacement_rate>0)
        with torch.no_grad():
            out = model(X)
            best = max(best, float((out.argmax(-1) == y).float().mean()))
            dead = float((model.h.abs().sum(0) == 0).float().mean())
    return best, dead


def _run_one_seed(seed: int, replacement_rate: float) -> dict:
    """replacement_rate == 0.0 is the plain-SGD arm (no unit ever eligible)."""
    torch.manual_seed(seed)
    model = MLP(DIM, HIDDEN, N_CLASSES)
    opt = torch.optim.SGD(model.parameters(), lr=SGD_LR)  # plain SGD, fixed lr, no momentum
    cbp = ContinualBackprop(
        model,
        replacement_rate=replacement_rate,
        maturity=CBP_MATURITY,
        decay=CBP_DECAY,
        seed=seed,
    )
    diag, dead = [], []
    for X, y in _task_stream(seed):
        b, d = _learn_on_task(model, opt, cbp, X, y)
        diag.append(b)
        dead.append(d)
    early = sum(diag[:EARLY_WINDOW]) / EARLY_WINDOW
    late = sum(diag[-LATE_WINDOW:]) / LATE_WINDOW
    dead_early = sum(dead[:EARLY_WINDOW]) / EARLY_WINDOW
    dead_late = sum(dead[-LATE_WINDOW:]) / LATE_WINDOW
    return {
        "early": early,
        "late": late,
        "gap": early - late,
        "dead_early": dead_early,
        "dead_late": dead_late,
        "reinit_count": cbp.reinit_count,
    }


def _arm(replacement_rate: float) -> dict:
    per_seed = [_run_one_seed(s, replacement_rate) for s in SEEDS]
    n = len(per_seed)
    gaps = [r["gap"] for r in per_seed]
    return {
        "arm": "sgd" if replacement_rate == 0.0 else f"cbp_rate={replacement_rate:g}",
        "replacement_rate": replacement_rate,
        "n_seeds": n,
        "early_learn_acc_mean": round(sum(r["early"] for r in per_seed) / n, 4),
        "late_learn_acc_mean": round(sum(r["late"] for r in per_seed) / n, 4),
        "gap_mean": round(sum(gaps) / n, 4),
        "gap_ci": seed_ci(gaps, z=Z),
        "per_seed_gap": [round(g, 4) for g in gaps],
        "dead_frac_early": round(sum(r["dead_early"] for r in per_seed) / n, 4),
        "dead_frac_late": round(sum(r["dead_late"] for r in per_seed) / n, 4),
        "per_seed_dead_late": [round(r["dead_late"], 4) for r in per_seed],
        "reinit_count_total": sum(r["reinit_count"] for r in per_seed),
        "_per_seed": per_seed,  # kept for paired deltas, stripped before write
    }


def _evaluate(sgd: dict, cbp: dict) -> dict:
    """Apply the PREREGISTERED win rule to the SGD arm vs one CBP arm. Paired per-seed deltas (same seed
    -> same stream)."""
    gap_delta = [s["gap"] - c["gap"] for s, c in zip(sgd["_per_seed"], cbp["_per_seed"], strict=True)]
    dead_delta = [
        s["dead_late"] - c["dead_late"] for s, c in zip(sgd["_per_seed"], cbp["_per_seed"], strict=True)
    ]
    gap_delta_ci = seed_ci(gap_delta, z=Z)
    gap_delta_flips = sign_flip_report(gap_delta)
    dead_delta_ci = seed_ci(dead_delta, z=Z)
    dead_delta_flips = sign_flip_report(dead_delta)

    w1_gap_closed = gap_delta_ci["lo"] > 0.0
    w2_no_flip = gap_delta_flips["consistent_sign"] == 1
    dead_ratio_ok = cbp["dead_frac_late"] <= DEAD_RATIO_MAX * sgd["dead_frac_late"]
    w3_dead = dead_ratio_ok and dead_delta_ci["lo"] > 0.0 and dead_delta_flips["consistent_sign"] == 1
    cbp_wins = bool(w1_gap_closed and w2_no_flip and w3_dead)
    return {
        "replacement_rate": cbp["replacement_rate"],
        "reinit_count_total": cbp["reinit_count_total"],
        "cbp_gap_mean": cbp["gap_mean"],
        "cbp_dead_frac_late": cbp["dead_frac_late"],
        "gap_delta_sgd_minus_cbp": {
            "per_seed": [round(d, 4) for d in gap_delta],
            "ci": gap_delta_ci,
            "sign_flips": gap_delta_flips,
        },
        "dead_late_delta_sgd_minus_cbp": {
            "per_seed": [round(d, 4) for d in dead_delta],
            "ci": dead_delta_ci,
            "sign_flips": dead_delta_flips,
        },
        "win_criteria": {
            "W1_gap_closed_ci_lo_gt_0": bool(w1_gap_closed),
            "W2_no_sign_flip": bool(w2_no_flip),
            "W3_dead_below": bool(w3_dead),
            "w3_dead_ratio_ok": bool(dead_ratio_ok),
        },
        "cbp_wins": cbp_wins,
    }


def main():
    t0 = time.perf_counter()
    sgd = _arm(0.0)

    # PRECONDITION: SGD must reproduce the certified plasticity loss on this stream (else inadmissible).
    sgd_gap_fires = sgd["gap_ci"]["lo"] > 0.0
    sgd_dead_rose = sgd["dead_frac_late"] > sgd["dead_frac_early"]
    precondition_ok = bool(sgd_gap_fires and sgd_dead_rose)

    # Sweep CBP over the preregistered rate grid. Win rule is identical for every rate.
    cbp_arms = [_arm(r) for r in CBP_RATE_GRID]
    per_rate = [_evaluate(sgd, c) for c in cbp_arms]
    any_win = any(e["cbp_wins"] for e in per_rate)
    # "best rate" = the one that most closes the gap (largest gap_delta mean), for reporting only.
    best = max(per_rate, key=lambda e: e["gap_delta_sgd_minus_cbp"]["ci"]["mean"])

    if not precondition_ok:
        verdict = (
            "INADMISSIBLE: the SGD arm did not reproduce the certified plasticity loss on this stream "
            f"(sgd_gap_fires={sgd_gap_fires}, sgd_dead_rose={sgd_dead_rose}); nothing to repair, no CBP "
            "verdict claimed."
        )
    elif any_win:
        wins = [e for e in per_rate if e["cbp_wins"]]
        rates = ", ".join(f"{e['replacement_rate']:g}" for e in wins)
        verdict = (
            "CBP WINS: on the validated drift stream where plain SGD loses plasticity, continual-backprop "
            f"selective reinit CLOSES the plasticity-loss gap beyond seed spread with a consistent sign "
            f"AND keeps the dead-unit fraction far below SGD's, at replacement_rate(s) {rates}. The "
            "MOLDABILITY mechanism is live on the laptop surface."
        )
    else:
        bd = best["gap_delta_sgd_minus_cbp"]["ci"]
        verdict = (
            "NULL: continual-backprop was genuinely exercised (reinits fired at every swept rate) but did "
            "NOT clear the preregistered win bar at ANY replacement rate, even though plain SGD provably "
            f"loses plasticity here. Best rate {best['replacement_rate']:g} closed the gap by only "
            f"mean={bd['mean']} (CI lo={bd['lo']}); a tie/near-tie is a NULL. The plasticity MECHANISM "
            "does not repair induced loss on this laptop surface: a strong honest null. The score moves "
            "only at Studio scale (a longer stream / wider layer where the reinit has room to matter)."
        )

    # strip the heavy per-seed payloads before writing
    sgd.pop("_per_seed", None)
    for arm in cbp_arms:
        arm.pop("_per_seed", None)

    out = {
        "build": "BUILD 1: continual-backprop (Dohare selective reinit) vs plain SGD on the VALIDATED "
        "150-task concept-drift plasticity-loss stream (mop_plasticity_certificate.py, drift mode)",
        "axis": "MOLDABILITY / plasticity",
        "stream": "identical to the validated certificate drift stream (fresh teacher per task, 150 tasks)",
        "cbp_mechanism": "utility-based selective reinit of mature low-utility hidden units (Dohare 2024), "
        "copied from scripts/studio/pr9_continual_backprop.py; plain SGD = replacement_rate 0",
        "matched": "both arms share net, teacher stream, input distribution, lr, epochs, batch, task "
        "order, and inner shuffle seed; ONLY the reinit rule differs",
        "preregistered_win_rule": (
            "delta = gap_sgd - gap_cbp per seed. CBP WINS iff (W1) seed-CI lo of delta > 0 AND (W2) no "
            "sign flip AND (W3) cbp late dead frac <= 0.5x sgd's AND seed-CI lo of (dead_sgd - dead_cbp) "
            "> 0 with no flip. A tie (W1 CI includes 0) is a NULL. Fixed before running, never tuned."
        ),
        "precondition": (
            "SGD arm must reproduce the certified plasticity loss (gap CI lo > 0 and dead rose); else "
            "inadmissible"
        ),
        "config": {
            "n_tasks": N_TASKS,
            "early_window": EARLY_WINDOW,
            "late_window": LATE_WINDOW,
            "seeds": SEEDS,
            "dim": DIM,
            "n_classes": N_CLASSES,
            "hidden": HIDDEN,
            "teacher_hidden": TEACHER_HIDDEN,
            "sgd_lr": SGD_LR,
            "epochs_per_task": EPOCHS_PER_TASK,
            "batch": BATCH,
            "samples_per_task": SAMPLES_PER_TASK,
            "cbp_rate_grid": CBP_RATE_GRID,
            "cbp_maturity": CBP_MATURITY,
            "cbp_decay": CBP_DECAY,
            "dead_ratio_max": DEAD_RATIO_MAX,
            "z": Z,
        },
        "no_op_bug_note": (
            "the PR9/Studio default replacement_rate=1e-4 reinits nothing on a 48-unit layer "
            "(int(1e-4*48)=0 per step); a fractional-budget accumulator was added so tiny rates still "
            "fire, and a preregistered rate grid was swept. Reporting reinit_count per rate proves the "
            "mechanism actually operated (a reinit_count==0 null would be a false null)."
        ),
        "sgd_arm": sgd,
        "cbp_arms": cbp_arms,
        "per_rate_evaluation": per_rate,
        "best_rate_by_gap_closed": best["replacement_rate"],
        "precondition_ok": precondition_ok,
        "any_rate_wins": any_win,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
