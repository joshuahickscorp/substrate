#!/usr/bin/env python
"""PR9 (Process B, Studio-only): continual backprop (utility-based selective reinitialization, Dohare
et al. 2024) on a LONG real-latent stream, gated by a plasticity-loss certificate. Does the CBP
utility-reinit maintain the plastic shell's plasticity over a long non-stationary stream where plain
SGD loses it, WITHOUT paying a retention cost, ON REAL V-JEPA features (not synthetic)?

WHY STUDIO (not the laptop): CBP only earns its verdict against a stream long enough that the PLAIN
(no-reinit) baseline demonstrably loses plasticity first -- otherwise there is nothing for the reinit to
restore, and any "win" is noise (exactly the EX15 caveat: no plasticity loss at this scale means nothing
to fix). A stream that long over the real bound-video / nuisance latent store is many thousands of
optimizer steps across dozens of task switches; it does not fit the laptop's 18GB pool or its patience.
The Studio box runs the full stream so the certificate gate below can actually fire. This script has a
hard free-RAM guard (>= 32GB) so it cannot run on the laptop by accident.

PLASTICITY-LOSS CERTIFICATE GATE (preregistered, evaluated BEFORE any CBP-vs-plain comparison is
reported): the comparison is only admissible if the PLAIN baseline itself exhibits loss of plasticity on
this stream, certified by BOTH of: (a) per-task adaptation accuracy trends DOWN over stream position
(late tasks learned worse than early tasks, slope CI below zero across seeds), and (b) the dead-unit
fraction trends UP over stream position. If the certificate does NOT fire, we REPORT NULL as
"no-plasticity-loss-to-restore" and do not compare arms -- a CBP "win" without a certified loss is not a
win, it is an unfalsifiable comparison.

PREREGISTERED NULL (fixed before any result exists): given a fired certificate, CBP does not restore
plasticity, OR it restores plasticity only at a retention cost. Formally the null is REJECTED only if,
across seeds, CBP's late-stream adaptation accuracy beats plain's with a seed-CI lower bound above zero
AND a consistent per-seed sign, AND CBP's retained accuracy on early tasks is NOT worse than plain's
beyond seed spread (no retention tax). Both arms run the IDENTICAL shell (same arch, init seed, LR,
steps, task order) so the only difference is the reinit rule; both log through one LRIntegralAccumulator
each and the run asserts the LR-integrals match before comparing (matched-compute precondition). The
honest floor is the plain-SGD arm on the SAME stream, never a fresh-init readout.

Consumes a real-latent store (default the DR1 bound-video store, else the WP-11 nuisance store); never
loads an encoder. Heavy queue class: run only when the encoder lane is free and on the Studio box.

Usage (Studio):
  python scripts/studio/pr9_continual_backprop.py --cache data/cache/vjepa2_vitl_bound_video --seeds 0-9

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mop.diagnostics.continual_metrics import LRIntegralAccumulator, adaptation_speed  # noqa: E402
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from mop.substrate import LatentStore  # noqa: E402

MIN_FREE_RAM_GB = 32.0
DEAD_UNIT_VAR_THRESHOLD = 1e-3

DEFAULTS = {
    "cache": "data/cache/vjepa2_vitl_bound_video",
    "seeds": tuple(range(10)),
    "hidden": 64,
    "lr": 1e-2,
    "steps_per_task": 200,
    "n_passes": 6,  # revisit the task set n_passes times -> a long non-stationary stream
    "test_frac": 0.25,
    "adapt_target_frac": 0.9,
    "cbp_replacement_rate": 1e-4,  # fraction of hidden units eligible for reinit per step
    "cbp_maturity": 50,  # steps a unit must survive before it is reinit-eligible
    "cbp_decay": 0.99,  # running-utility decay
}


def assert_studio_ram(min_gb: float = MIN_FREE_RAM_GB) -> float:
    """Hard guard: refuse to run unless >= min_gb of free RAM. Keeps the long stream off the laptop."""
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024**3)
    except Exception as e:
        raise SystemExit(
            f"cannot read free RAM ({e}); refusing to run without the >= {min_gb:.0f}GB safety check. "
            "Install psutil on the Studio box."
        ) from e
    if free_gb < min_gb:
        raise SystemExit(
            f"free RAM {free_gb:.1f}GB < required {min_gb:.0f}GB. Studio-only long stream; move it to "
            "the Studio box (the laptop pool is 18GB)."
        )
    return free_gb


def _zscore(train: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True) + 1e-6
    return (x - mu) / sd


def _class_split(y: torch.Tensor, test_frac: float, g: torch.Generator):
    tr, te = [], []
    for c in y.unique().tolist():
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        cut = max(1, int(len(idx) * test_frac))
        te.append(idx[:cut])
        tr.append(idx[cut:])
    return torch.cat(tr), torch.cat(te)


def _tasks_from_classes(n_classes: int) -> list[list[int]]:
    return [list(range(i, min(i + 2, n_classes))) for i in range(0, n_classes, 2)]


class Shell(nn.Module):
    """Linear-ReLU-Linear plastic head. Exposes hidden activations so utility and dead-unit readings
    are taken on the same hidden layer CBP reinitializes."""

    def __init__(self, dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.hidden(x))


class ContinualBackprop:
    """Utility-based selective reinit (Dohare 2024). Tracks a running contribution-utility per hidden
    unit; each step a small fraction of MATURE, LOWEST-utility units are reset (their fc1 fan-in
    re-initialized, their fc2 fan-out zeroed, their utility+age reset). Plain SGD is this with
    replacement_rate=0 (no unit is ever eligible)."""

    def __init__(self, shell: Shell, *, replacement_rate: float, maturity: int, decay: float, seed: int):
        self.shell = shell
        self.replacement_rate = replacement_rate
        self.maturity = maturity
        self.decay = decay
        self.g = torch.Generator().manual_seed(seed + 7919)
        h = shell.fc1.out_features
        self.util = torch.zeros(h)
        self.age = torch.zeros(h, dtype=torch.long)
        self.reinit_count = 0

    @torch.no_grad()
    def step(self, hidden_act: torch.Tensor) -> None:
        """Update utility from the batch's hidden activations, then reinit the eligible worst units."""
        self.age += 1
        contribution = hidden_act.abs().mean(0) * self.shell.fc2.weight.abs().mean(0)
        self.util.mul_(self.decay).add_(contribution, alpha=1.0 - self.decay)
        if self.replacement_rate <= 0:
            return
        n_eligible = int((self.age >= self.maturity).sum())
        n_reset = int(self.replacement_rate * n_eligible)
        if n_reset < 1:
            return
        eligible = (self.age >= self.maturity).nonzero(as_tuple=True)[0]
        worst = eligible[torch.argsort(self.util[eligible])[:n_reset]]
        fan_in = self.shell.fc1.in_features
        new_w = torch.randn(n_reset, fan_in, generator=self.g) * (fan_in**-0.5)
        self.shell.fc1.weight.data[worst] = new_w
        self.shell.fc1.bias.data[worst] = 0.0
        self.shell.fc2.weight.data[:, worst] = 0.0  # zero fan-out so a fresh unit does not shock output
        self.util[worst] = 0.0
        self.age[worst] = 0
        self.reinit_count += n_reset


@torch.no_grad()
def dead_unit_fraction(shell: Shell, x: torch.Tensor) -> float:
    h = shell.hidden(x)
    if h.shape[0] < 2:
        return 0.0
    var = h.var(dim=0, unbiased=False)
    return float((var < DEAD_UNIT_VAR_THRESHOLD).mean())


def run_stream(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    use_cbp: bool,
    hidden: int,
    lr: float,
    steps_per_task: int,
    n_passes: int,
    test_frac: float,
    adapt_target_frac: float,
    cbp_replacement_rate: float,
    cbp_maturity: int,
    cbp_decay: float,
) -> dict:
    """Run one shell (CBP or plain) through the long stream. Returns per-task adaptation accuracy,
    dead-unit fraction over stream position, early-task retained accuracy, and the LR-integral."""
    n_classes = int(y.max()) + 1
    tasks = _tasks_from_classes(n_classes)
    g = torch.Generator().manual_seed(seed)
    tr_idx, te_idx = _class_split(y, test_frac, g)
    xz = _zscore(x[tr_idx], x)
    torch.manual_seed(seed)
    shell = Shell(x.shape[1], hidden, n_classes)
    opt = torch.optim.SGD(shell.parameters(), lr=lr)
    lri = LRIntegralAccumulator()
    cbp = ContinualBackprop(
        shell,
        replacement_rate=cbp_replacement_rate if use_cbp else 0.0,
        maturity=cbp_maturity,
        decay=cbp_decay,
        seed=seed,
    )

    def task_indices(pool: torch.Tensor, classes: list[int]) -> torch.Tensor:
        m = torch.zeros(len(pool), dtype=torch.bool)
        for c in classes:
            m |= y[pool] == c
        return pool[m]

    @torch.no_grad()
    def acc_on(idx: torch.Tensor) -> float:
        return float((shell(xz[idx]).argmax(-1) == y[idx]).float().mean())

    te_task = [task_indices(te_idx, t) for t in tasks]
    early_te = te_task[0]
    per_task_adapt: list[float] = []
    dead_curve: list[float] = []
    stream_pos: list[int] = []
    pos = 0
    for _pass in range(n_passes):
        for ti, classes in enumerate(tasks):
            tr = task_indices(tr_idx, classes)
            curve: list[float] = []
            for _ in range(steps_per_task):
                opt.zero_grad()
                h = shell.hidden(xz[tr])
                logits = shell.fc2(h)
                F.cross_entropy(logits, y[tr]).backward()
                opt.step()
                lri.add(lr, steps=1)
                cbp.step(h.detach())
                curve.append(acc_on(te_task[ti]))
            per_task_adapt.append(float(curve[-1]))
            dead_curve.append(dead_unit_fraction(shell, xz[tr]))
            stream_pos.append(pos)
            pos += 1
    return {
        "per_task_adapt": per_task_adapt,
        "dead_curve": dead_curve,
        "stream_pos": stream_pos,
        "early_retained_acc": acc_on(early_te),
        "final_adapt_speed": adaptation_speed(curve, target_frac=adapt_target_frac)["steps"],
        "reinit_count": cbp.reinit_count,
        "lr_integral": lri.total(),
    }


def _slope(xs: list[int], ys: list[float]) -> float:
    """OLS slope of ys over xs (stream position). Used for the down/up trend certificate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (yv - my) for x, yv in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs) or 1e-12
    return num / den


def certificate(plain_runs: list[dict]) -> dict:
    """Plasticity-loss certificate on the PLAIN arm across seeds: (a) adaptation-accuracy slope over
    stream position is negative (CI upper bound below zero), (b) dead-unit slope is positive (CI lower
    bound above zero). Both must hold for the CBP comparison to be admissible."""
    adapt_slopes = [_slope(r["stream_pos"], r["per_task_adapt"]) for r in plain_runs]
    dead_slopes = [_slope(r["stream_pos"], r["dead_curve"]) for r in plain_runs]
    adapt_ci = seed_ci(adapt_slopes)
    dead_ci = seed_ci(dead_slopes)
    adapt_down = adapt_ci["hi"] < 0
    dead_up = dead_ci["lo"] > 0
    return {
        "adapt_slope_ci": adapt_ci,
        "dead_slope_ci": dead_ci,
        "adapt_trends_down": adapt_down,
        "dead_trends_up": dead_up,
        "fired": adapt_down and dead_up,
    }


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def run(cfg: dict) -> dict:
    store = LatentStore.open(Path(cfg["cache"]))
    y = store.labels()
    if y is None:
        raise SystemExit(f"cache {cfg['cache']} has no labels; PR9 needs a labeled real-latent store.")
    x = store.latents().float()
    kw = {
        k: cfg[k]
        for k in (
            "hidden",
            "lr",
            "steps_per_task",
            "n_passes",
            "test_frac",
            "adapt_target_frac",
            "cbp_replacement_rate",
            "cbp_maturity",
            "cbp_decay",
        )
    }
    seeds = list(cfg["seeds"])
    t0 = time.perf_counter()
    plain_runs, cbp_runs, per_seed = [], [], []
    late_frac = 0.5  # "late stream" = second half of the task sequence
    for s in seeds:
        plain = run_stream(x, y, seed=s, use_cbp=False, **kw)
        cbp = run_stream(x, y, seed=s, use_cbp=True, **kw)
        plain_runs.append(plain)
        cbp_runs.append(cbp)
        lri_matched = abs(plain["lr_integral"] - cbp["lr_integral"]) <= 0.02 * max(
            plain["lr_integral"], cbp["lr_integral"], 1e-12
        )
        cut = int(len(plain["per_task_adapt"]) * late_frac)
        late_plain = sum(plain["per_task_adapt"][cut:]) / max(1, len(plain["per_task_adapt"][cut:]))
        late_cbp = sum(cbp["per_task_adapt"][cut:]) / max(1, len(cbp["per_task_adapt"][cut:]))
        per_seed.append(
            {
                "seed": s,
                "late_adapt_plain": round(late_plain, 4),
                "late_adapt_cbp": round(late_cbp, 4),
                "late_adapt_delta": round(late_cbp - late_plain, 4),
                "retention_delta": round(cbp["early_retained_acc"] - plain["early_retained_acc"], 4),
                "reinit_count": cbp["reinit_count"],
                "lr_integral_matched": lri_matched,
            }
        )
    cert = certificate(plain_runs)
    late_deltas = [r["late_adapt_delta"] for r in per_seed]
    retention_deltas = [r["retention_delta"] for r in per_seed]
    late_ci = seed_ci(late_deltas)
    ret_ci = seed_ci(retention_deltas)
    late_flips = sign_flip_report(late_deltas)
    all_matched = all(r["lr_integral_matched"] for r in per_seed)
    plasticity_restored = late_ci["lo"] > 0 and late_flips["consistent_sign"] == 1
    no_retention_tax = ret_ci["lo"] >= 0 or ret_ci["hi"] >= 0  # not worse beyond seed spread
    retention_tax_paid = ret_ci["hi"] < 0  # CBP retention strictly worse across seeds
    if not cert["fired"]:
        null_supported = True
        verdict = (
            "NULL (no-plasticity-loss-to-restore): the plain arm did not certify loss of plasticity on "
            "this stream (adapt_down={} dead_up={}); the CBP comparison is inadmissible, no win claimed"
        ).format(cert["adapt_trends_down"], cert["dead_trends_up"])
    elif not all_matched:
        null_supported = True
        verdict = "NULL (compute-unmatched): LR-integrals differ between arms; comparison not honest"
    elif plasticity_restored and not retention_tax_paid:
        null_supported = False
        verdict = (
            "NULL REJECTED: with a fired plasticity-loss certificate, CBP utility-reinit restores "
            "late-stream plasticity beyond seed spread with a consistent sign and pays no retention tax"
        )
    else:
        null_supported = True
        verdict = (
            "NULL SUPPORTED: certificate fired but CBP does not restore plasticity beyond seed spread "
            f"(restored={plasticity_restored}) or pays a retention tax (tax={retention_tax_paid})"
        )
    return {
        "experiment": PR9_EXPERIMENT,
        "cache": cfg["cache"],
        "seeds": seeds,
        "certificate": cert,
        "per_seed": per_seed,
        "late_adapt_delta_ci": late_ci,
        "retention_delta_ci": ret_ci,
        "late_adapt_sign_flips": late_flips,
        "lr_integral_matched_all": all_matched,
        "plasticity_restored": plasticity_restored,
        "retention_tax_paid": retention_tax_paid,
        "no_retention_tax": no_retention_tax,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": verdict,
    }


PR9_EXPERIMENT = {
    "id": "pr9_continual_backprop",
    "metric": "late-stream adaptation-accuracy delta (CBP minus plain) gated by a plasticity certificate",
    "baseline": "identical plastic shell with plain SGD (replacement_rate=0) on the SAME long stream",
    "ablation": "utility-based selective reinit vs none, matched LR-integral",
    "null_hypothesis": (
        "given a fired plasticity-loss certificate, CBP does not restore plasticity beyond seed spread, "
        "or restores it only at a retention cost"
    ),
    "tier": "studio (long real-latent stream)",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR9 Studio: continual backprop on a long real-latent stream")
    ap.add_argument("--cache", default=DEFAULTS["cache"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--steps-per-task", type=int, default=DEFAULTS["steps_per_task"])
    ap.add_argument("--n-passes", type=int, default=DEFAULTS["n_passes"])
    ap.add_argument("--out", default="runs/mot/pr9_continual_backprop.json")
    a = ap.parse_args(argv)

    assert_studio_ram()  # Studio-only guard
    cfg = {
        **DEFAULTS,
        "cache": a.cache,
        "seeds": parse_seeds(a.seeds),
        "hidden": a.hidden,
        "lr": a.lr,
        "steps_per_task": a.steps_per_task,
        "n_passes": a.n_passes,
    }
    result = run(cfg)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, default=str))
    summary_keys = ("certificate", "late_adapt_delta_ci", "null_supported", "verdict")
    print(json.dumps({k: result[k] for k in summary_keys}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
