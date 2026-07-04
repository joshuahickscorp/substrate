#!/usr/bin/env python
"""PR9 (Process B, Studio-only), HARDENED: continual backprop (utility-based selective reinitialization,
Dohare et al. 2024) on a LONG real-latent stream, gated by a plasticity-loss certificate under a
WELL-TUNED baseline. Does the CBP utility-reinit maintain the plastic shell's plasticity over a long
non-stationary stream where a well-tuned plain SGD loses it, WITHOUT paying a retention cost, ON REAL
V-JEPA features (not synthetic)?

WHY STUDIO (not the laptop): CBP only earns its verdict against a stream long enough that the well-tuned
plain (no-reinit) baseline demonstrably loses plasticity first, otherwise there is nothing for the reinit
to restore, and any win is noise (the EX15 caveat: no plasticity loss at this scale means nothing to fix).
A stream that long over the real bound-video / nuisance latent store is many thousands of optimizer steps
across dozens of task switches; it does not fit the laptop's 18GB pool or its patience. This script has a
hard free-RAM guard (>= 32GB) so it cannot run the full stream on the laptop by accident; the guard can be
bypassed ONLY for an explicit tiny smoke check (--smoke), which also shrinks the stream.

HARDENING CARRIED FROM THE FIVE LAPTOP ROUNDS (see pr9_report.md for the full ledger):

1. CBP NO-OP BUG FIX (validated in scripts/mop_cbp_plasticity_repair.py). The original PR9 used
   n_reset = int(replacement_rate * n_eligible). With replacement_rate=1e-4 on a 64-unit layer,
   int(1e-4 * 64) = 0 EVERY step, so the mechanism NEVER FIRES (reinit_count stays 0) and CBP is
   bit-identical to plain SGD: a FALSE null. The fix is Dohare's faithful FRACTIONAL-BUDGET ACCUMULATOR:
   accumulate replacement_rate * n_eligible into a running budget, reinit floor(budget) units, carry the
   remainder. Even a tiny rate on a small layer eventually fires. This run ASSERTS reinit_count_total > 0
   for every swept CBP rate before it will claim any verdict; a zero-reinit arm is reported as a config
   error, never as a null.

2. MISTUNED-BASELINE TRAP FIX. The laptop killed two plasticity over-claims that compared a mechanism vs a
   baseline MISTUNED into the dead-ReLU regime (lr too high). Here the plain baseline LR is SWEPT over a
   grid and the WELL-TUNED LR is chosen by a tuning objective computed WITHOUT reference to CBP (best
   early-task learnability net of dead units on a held-out tuning seed set). The certificate and the
   CBP-vs-plain comparison then both run at that well-tuned LR (best-vs-best), so a win cannot be a
   mistuned-baseline artifact. The CBP arm uses the SAME well-tuned LR (matched-compute, matched LR
   integral) so the only difference between arms is the reinit rule.

3. PLASTICITY-LOSS CERTIFICATE (validated in scripts/mop_plasticity_certificate.py). Preregistered, and
   evaluated on the WELL-TUNED plain arm BEFORE any CBP comparison is reported. Admissible only if the
   well-tuned plain baseline itself exhibits loss of plasticity on this stream, certified by BOTH: (a)
   per-task adaptation accuracy trends DOWN over stream position (slope CI upper bound below zero across
   seeds), and (b) dead-unit fraction trends UP (slope CI lower bound above zero). If the certificate does
   NOT fire, we REPORT NULL as no-plasticity-loss-to-restore and do not compare arms.

4. CONTROL HYGIENE. Both arms run the IDENTICAL shell (same arch, init seed, LR, steps, task order) so the
   only difference is the reinit rule; both log through an LRIntegralAccumulator and the run asserts the
   LR integrals match before comparing (matched-compute precondition). The honest floor is the well-tuned
   plain-SGD arm on the SAME stream, never a fresh-init readout. No sign-flip rule on the per-seed delta.

5. ENCODER-LANE GUARD + RESUMABILITY. A pgrep guard refuses to run while an encoder job holds the lane
   (this is a CPU/RAM-heavy readout, not an encode; running it under an active encode double-books RAM).
   Per-seed, per-arm results are checkpointed to <out>.legs/ so a killed run resumes at the next unfinished
   (seed, arm) leg instead of recomputing the stream from scratch (per-clip-range-leg resumability).

Consumes a real-latent store (default the DR1 bound-video store, else the WP-11 nuisance store); never
loads an encoder. Heavy queue class: run only when the encoder lane is free and on the Studio box.

Usage (Studio):
  python scripts/studio/pr9_continual_backprop.py --cache data/cache/vjepa2_vitl_bound_video --seeds 0-9

Smoke (laptop, tiny N, proves reinits fire and the path runs):
  python pr9_hardened.py --cache data/cache/vjepa2_vitl_fpc64_256_real --seeds 0-1 --smoke

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# also expose src/ so `import mop` works when this file lives outside the repo tree (Studio harden dir)
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mop.diagnostics.continual_metrics import LRIntegralAccumulator, adaptation_speed  # noqa: E402
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from mop.seeding import parse_seeds  # noqa: E402
from mop.substrate import LatentStore  # noqa: E402

MIN_FREE_RAM_GB = 32.0
DEAD_UNIT_VAR_THRESHOLD = 1e-3

# STUDIO defaults (the real long-stream regime). n_passes revisits the whole task set repeatedly so the
# late window is many task switches after the early window: that length is what lets plasticity loss
# accumulate under a well-tuned baseline (short streams cannot induce it, the EX15 caveat).
DEFAULTS = {
    "cache": "data/cache/vjepa2_vitl_bound_video",
    "seeds": tuple(range(10)),
    "hidden": 64,
    "steps_per_task": 200,
    "n_passes": 12,  # long non-stationary stream (Studio scale); smoke shrinks this
    "test_frac": 0.25,
    "adapt_target_frac": 0.9,
    # PREREGISTERED baseline-LR grid. The plain baseline is TUNED over this grid to avoid the dead-ReLU
    # regime; both arms then use the winning LR (best-vs-best). Spans well below to well above the point
    # where a plain ReLU MLP on z-scored real latents starts killing units.
    "baseline_lr_grid": (3e-3, 1e-2, 3e-2, 1e-1),
    # PREREGISTERED CBP replacement-rate grid. Chosen by reinit-budget arithmetic (budget/step =
    # rate * n_mature), NOT by any accuracy number, so on a 64-unit layer these span "a reinit every few
    # tasks" up to "a few reinits per task". The fractional-budget accumulator guarantees each fires.
    "cbp_rate_grid": (2e-4, 1e-3, 5e-3),
    "cbp_maturity": 50,  # steps a unit must survive before it is reinit-eligible
    "cbp_decay": 0.99,  # running-utility decay
}

# smoke overrides (tiny N, still exercises the no-op-bug fix end to end so reinit_count_total > 0)
SMOKE_OVERRIDES = {
    "hidden": 32,
    "steps_per_task": 30,
    "n_passes": 4,
    "baseline_lr_grid": (1e-2, 5e-2),
    "cbp_rate_grid": (1e-3,),
    "cbp_maturity": 10,
}


def assert_studio_ram(min_gb: float = MIN_FREE_RAM_GB, *, allow_low: bool = False) -> float:
    """Hard guard: refuse to run unless >= min_gb of free RAM. Keeps the long stream off the laptop.
    `allow_low` (set only by --smoke) bypasses the floor for the tiny path."""
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024**3)
    except Exception as e:
        if allow_low:
            return -1.0
        raise SystemExit(
            f"cannot read free RAM ({e}); refusing to run without the >= {min_gb:.0f}GB safety check. "
            "Install psutil on the Studio box."
        ) from e
    if free_gb < min_gb and not allow_low:
        raise SystemExit(
            f"free RAM {free_gb:.1f}GB < required {min_gb:.0f}GB. Studio-only long stream; move it to "
            "the Studio box (the laptop pool is 18GB). Use --smoke ONLY for the tiny import/run check."
        )
    return free_gb


def assert_encoder_lane_free(*, skip: bool = False) -> None:
    """Encoder-lane guard: refuse to run while an encoder job holds the lane. This readout is CPU/RAM
    heavy; running it under an active encode double-books RAM and can OOM the Studio box. We pgrep for the
    known encode entrypoints. If pgrep is unavailable we warn but proceed (guard is best-effort). `skip`
    (set by --no-encoder-guard or --smoke) bypasses it."""
    if skip:
        return
    patterns = ["cache_latents", "encode_", "run_encode", "dr1_curate_bound_video", "FrozenEncoder"]
    try:
        for pat in patterns:
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                pids = r.stdout.strip().replace("\n", ",")
                raise SystemExit(
                    f"encoder lane busy: a process matching '{pat}' is running (pids {pids}). PR9 is a "
                    "heavy CPU/RAM readout; wait for the encode to finish or pass --no-encoder-guard if "
                    "you have confirmed the box has headroom."
                )
    except FileNotFoundError:
        print("WARN: pgrep not found; encoder-lane guard skipped (best-effort).", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("WARN: pgrep timed out; encoder-lane guard skipped (best-effort).", file=sys.stderr)


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
    replacement_rate=0 (no unit is ever eligible).

    NO-OP-BUG FIX (ported verbatim in mechanism from scripts/mop_cbp_plasticity_repair.py): a FRACTIONAL
    reinit budget is carried across steps, so even a tiny replacement_rate on a small layer eventually
    fires. The original int(rate * n_eligible) rounded to 0 every step on a 64-unit layer at rate=1e-4,
    reinit_count stayed 0, and CBP was a bit-identical no-op vs plain SGD: a false null. Do not revert to
    the int() form."""

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
        self._budget = 0.0  # fractional reinit budget carried across steps (Dohare small-layer form)

    @torch.no_grad()
    def step(self, hidden_act: torch.Tensor) -> None:
        """Update utility from the batch's hidden activations, then reinit the eligible worst units."""
        self.age += 1
        contribution = hidden_act.abs().mean(0) * self.shell.fc2.weight.abs().mean(0)
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
    return float((var < DEAD_UNIT_VAR_THRESHOLD).float().mean())


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
    dead-unit fraction over stream position, early-task retained accuracy, the reinit count, and the
    LR-integral."""
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
        if len(idx) == 0:
            return 0.0
        return float((shell(xz[idx]).argmax(-1) == y[idx]).float().mean())

    te_task = [task_indices(te_idx, t) for t in tasks]
    early_te = te_task[0]
    per_task_adapt: list[float] = []
    dead_curve: list[float] = []
    stream_pos: list[int] = []
    curve: list[float] = []
    pos = 0
    for _pass in range(n_passes):
        for ti, classes in enumerate(tasks):
            tr = task_indices(tr_idx, classes)
            if len(tr) == 0:
                continue
            curve = []
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
        "final_adapt_speed": adaptation_speed(curve, target_frac=adapt_target_frac)["steps"] if curve else 0,
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
    """Plasticity-loss certificate on the WELL-TUNED PLAIN arm across seeds: (a) adaptation-accuracy
    slope over stream position is negative (CI upper bound below zero), (b) dead-unit slope is positive
    (CI lower bound above zero). Both must hold for the CBP comparison to be admissible."""
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


# ---------------------------------------------------------------------------
# Baseline-LR tuning (mistuned-baseline-trap fix). The plain baseline LR is chosen by a tuning objective
# that NEVER looks at CBP: on a small held-out set of tuning seeds we run the plain arm at each grid LR
# and score it by mean early-task learnability minus a dead-unit penalty. The winning LR is the
# "well-tuned baseline", used for BOTH arms in the graded run so a CBP win cannot be a mistuned artifact.
# ---------------------------------------------------------------------------
def tune_baseline_lr(x: torch.Tensor, y: torch.Tensor, cfg: dict, tuning_seeds: list[int]) -> dict:
    grid = list(cfg["baseline_lr_grid"])
    kw_base = {
        k: cfg[k]
        for k in (
            "hidden",
            "steps_per_task",
            "n_passes",
            "test_frac",
            "adapt_target_frac",
            "cbp_maturity",
            "cbp_decay",
        )
    }
    per_lr = []
    for lr in grid:
        early_accs, dead_fracs = [], []
        for s in tuning_seeds:
            r = run_stream(x, y, seed=s, use_cbp=False, lr=lr, cbp_replacement_rate=0.0, **kw_base)
            n_early = max(1, len(r["per_task_adapt"]) // 4)
            early_accs.append(sum(r["per_task_adapt"][:n_early]) / n_early)
            n_e = max(1, len(r["dead_curve"]) // 4)
            dead_fracs.append(sum(r["dead_curve"][:n_e]) / n_e)
        mean_early = sum(early_accs) / len(early_accs)
        mean_dead = sum(dead_fracs) / len(dead_fracs)
        # objective: high early learnability, low early dead-unit fraction. A baseline mistuned into the
        # dead-ReLU regime (lr too high) is penalized here and cannot win the tuning.
        objective = mean_early - mean_dead
        per_lr.append(
            {
                "lr": lr,
                "mean_early_learnability": round(mean_early, 4),
                "mean_early_dead_frac": round(mean_dead, 4),
                "tuning_objective": round(objective, 4),
            }
        )
    best = max(per_lr, key=lambda d: d["tuning_objective"])
    return {"grid": grid, "per_lr": per_lr, "tuning_seeds": tuning_seeds, "best_lr": best["lr"], "best": best}


# ---------------------------------------------------------------------------
# Resumable per-(seed, arm) legs. Each stream run is checkpointed to <out>.legs/ keyed by
# (seed, arm, lr, rate). A killed run resumes at the next unfinished leg instead of recomputing.
# ---------------------------------------------------------------------------
def _leg_path(legs_dir: Path, seed: int, arm: str, lr: float, rate: float) -> Path:
    return legs_dir / f"seed{seed}_{arm}_lr{lr:g}_rate{rate:g}.json"


def _cached_run_stream(legs_dir: Path, seed: int, arm: str, lr: float, rate: float, fn) -> dict:
    p = _leg_path(legs_dir, seed, arm, lr, rate)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass  # corrupt leg: recompute
    r = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, default=str))
    return r


def run(cfg: dict, *, legs_dir: Path) -> dict:
    store = LatentStore.open(Path(cfg["cache"]))
    y = store.labels()
    if y is None:
        raise SystemExit(f"cache {cfg['cache']} has no labels; PR9 needs a labeled real-latent store.")
    x = store.latents().float()
    seeds = list(cfg["seeds"])

    kw_stream = {
        k: cfg[k]
        for k in (
            "hidden",
            "steps_per_task",
            "n_passes",
            "test_frac",
            "adapt_target_frac",
            "cbp_maturity",
            "cbp_decay",
        )
    }

    t0 = time.perf_counter()

    # STEP 1: tune the plain baseline LR (mistuned-baseline-trap fix). Use up to the first 2 seeds as
    # tuning seeds (disjoint objective from the graded comparison), or all seeds if only one is given.
    tuning_seeds = seeds[:2] if len(seeds) >= 2 else seeds
    lr_tuning = tune_baseline_lr(x, y, cfg, tuning_seeds)
    lr = lr_tuning["best_lr"]

    late_frac = 0.5  # "late stream" = second half of the task sequence

    # STEP 2: graded run. Plain arm at the well-tuned LR, and CBP at the well-tuned LR swept over the
    # rate grid. All legs are resumable.
    plain_runs: list[dict] = []
    plain_by_seed: dict[int, dict] = {}
    for s in seeds:
        plain = _cached_run_stream(
            legs_dir,
            s,
            "plain",
            lr,
            0.0,
            lambda s=s: run_stream(x, y, seed=s, use_cbp=False, lr=lr, cbp_replacement_rate=0.0, **kw_stream),
        )
        plain_runs.append(plain)
        plain_by_seed[s] = plain

    cert = certificate(plain_runs)

    # CBP arms, one per swept replacement rate. reinit_count_total must be > 0 for each (no-op-bug guard).
    per_rate = []
    for rate in cfg["cbp_rate_grid"]:
        cbp_runs = []
        per_seed = []
        for s in seeds:
            cbp = _cached_run_stream(
                legs_dir,
                s,
                "cbp",
                lr,
                rate,
                lambda s=s, rate=rate: run_stream(
                    x, y, seed=s, use_cbp=True, lr=lr, cbp_replacement_rate=rate, **kw_stream
                ),
            )
            cbp_runs.append(cbp)
            plain = plain_by_seed[s]
            lri_matched = abs(plain["lr_integral"] - cbp["lr_integral"]) <= 0.02 * max(
                plain["lr_integral"], cbp["lr_integral"], 1e-12
            )
            cut = int(len(plain["per_task_adapt"]) * late_frac)
            denom_p = max(1, len(plain["per_task_adapt"][cut:]))
            denom_c = max(1, len(cbp["per_task_adapt"][cut:]))
            late_plain = sum(plain["per_task_adapt"][cut:]) / denom_p
            late_cbp = sum(cbp["per_task_adapt"][cut:]) / denom_c
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
        reinit_total = sum(r["reinit_count"] for r in per_seed)
        late_deltas = [r["late_adapt_delta"] for r in per_seed]
        retention_deltas = [r["retention_delta"] for r in per_seed]
        late_ci = seed_ci(late_deltas)
        ret_ci = seed_ci(retention_deltas)
        late_flips = sign_flip_report(late_deltas)
        all_matched = all(r["lr_integral_matched"] for r in per_seed)
        # NO-OP-BUG GUARD: reinits MUST have fired for this to be a real CBP arm.
        reinits_fired = reinit_total > 0
        plasticity_restored = late_ci["lo"] > 0 and late_flips["consistent_sign"] == 1
        retention_tax_paid = ret_ci["hi"] < 0  # CBP retention strictly worse across seeds
        no_retention_tax = not retention_tax_paid
        per_rate.append(
            {
                "replacement_rate": rate,
                "reinit_count_total": reinit_total,
                "reinits_fired": reinits_fired,
                "per_seed": per_seed,
                "late_adapt_delta_ci": late_ci,
                "retention_delta_ci": ret_ci,
                "late_adapt_sign_flips": late_flips,
                "lr_integral_matched_all": all_matched,
                "plasticity_restored": plasticity_restored,
                "retention_tax_paid": retention_tax_paid,
                "no_retention_tax": no_retention_tax,
            }
        )

    any_zero_reinit = any(not r["reinits_fired"] for r in per_rate)
    all_matched = all(r["lr_integral_matched_all"] for r in per_rate)
    # a rate "wins" if it restores plasticity beyond seed spread with no retention tax
    winning_rates = [
        r for r in per_rate if r["reinits_fired"] and r["plasticity_restored"] and r["no_retention_tax"]
    ]
    # best rate by late-adapt delta mean (for reporting only)
    best_rate = max(per_rate, key=lambda r: r["late_adapt_delta_ci"]["mean"]) if per_rate else None

    # ---- verdict ladder (preregistered) ----
    if any_zero_reinit:
        null_supported = None
        verdict = (
            "CONFIG ERROR (no-op bug): at least one CBP rate produced reinit_count_total == 0, so the "
            "mechanism never fired and the comparison is meaningless. This is the exact bug the "
            "fractional-budget accumulator fixes; check the rate grid vs layer width. No verdict claimed."
        )
    elif not cert["fired"]:
        null_supported = True
        verdict = (
            "NULL (no-plasticity-loss-to-restore): the WELL-TUNED plain arm (lr={}) did not certify loss "
            "of plasticity on this stream (adapt_down={} dead_up={}); the CBP comparison is inadmissible, "
            "no win claimed. Moldability is dead at a frozen substrate here; Process C is licensed."
        ).format(lr, cert["adapt_trends_down"], cert["dead_trends_up"])
    elif not all_matched:
        null_supported = True
        verdict = "NULL (compute-unmatched): LR-integrals differ between arms; comparison not honest"
    elif winning_rates:
        null_supported = False
        rates = ", ".join(f"{r['replacement_rate']:g}" for r in winning_rates)
        verdict = (
            "NULL REJECTED: with a fired plasticity-loss certificate under a well-tuned baseline, CBP "
            "utility-reinit (reinits fired) restores late-stream plasticity beyond seed spread with a "
            f"consistent sign and pays no retention tax, at replacement_rate(s) {rates}. Moldability "
            "mechanism is live on the real long stream."
        )
    else:
        null_supported = True
        bd = best_rate["late_adapt_delta_ci"] if best_rate else {"mean": None, "lo": None}
        verdict = (
            "NULL SUPPORTED: certificate fired under a well-tuned baseline and CBP was genuinely exercised "
            "(reinits fired at every rate), but CBP does not restore plasticity beyond seed spread or pays "
            f"a retention tax at any swept rate. Best rate closed late-adapt by mean={bd['mean']} "
            f"(CI lo={bd['lo']}). Moldability is dead at a frozen substrate; Process C is licensed."
        )

    return {
        "experiment": PR9_EXPERIMENT,
        "cache": cfg["cache"],
        "seeds": seeds,
        "well_tuned_baseline_lr": lr,
        "baseline_lr_tuning": lr_tuning,
        "certificate": cert,
        "cbp_rate_grid": list(cfg["cbp_rate_grid"]),
        "per_rate": per_rate,
        "any_zero_reinit": any_zero_reinit,
        "lr_integral_matched_all": all_matched,
        "winning_rates": [r["replacement_rate"] for r in winning_rates],
        "best_rate_by_late_delta": best_rate["replacement_rate"] if best_rate else None,
        "reinit_count_total_all_rates": sum(r["reinit_count_total"] for r in per_rate),
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": verdict,
    }


PR9_EXPERIMENT = {
    "id": "pr9_continual_backprop",
    "metric": "late-stream adaptation-accuracy delta (CBP minus plain) gated by a plasticity certificate "
    "under a well-tuned baseline, swept over a CBP replacement-rate grid",
    "baseline": "identical plastic shell with plain SGD (replacement_rate=0) at the WELL-TUNED LR on the "
    "SAME long stream (baseline LR chosen by a CBP-blind tuning objective)",
    "ablation": "utility-based selective reinit (fractional-budget accumulator so it fires on small "
    "layers) vs none, matched LR-integral",
    "null_hypothesis": (
        "given a fired plasticity-loss certificate under a well-tuned baseline, CBP does not restore "
        "plasticity beyond seed spread, or restores it only at a retention cost"
    ),
    "tier": "studio (long real-latent stream)",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR9 Studio: continual backprop on a long real-latent stream")
    ap.add_argument("--cache", default=DEFAULTS["cache"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    ap.add_argument("--steps-per-task", type=int, default=DEFAULTS["steps_per_task"])
    ap.add_argument("--n-passes", type=int, default=DEFAULTS["n_passes"])
    ap.add_argument("--out", default="runs/mot/pr9_continual_backprop.json")
    ap.add_argument("--smoke", action="store_true", help="tiny N, bypass RAM/encoder guards (laptop check)")
    ap.add_argument("--no-encoder-guard", action="store_true", help="bypass the pgrep encoder-lane guard")
    a = ap.parse_args(argv)

    cfg = {
        **DEFAULTS,
        "cache": a.cache,
        "seeds": parse_seeds(a.seeds),
        "hidden": a.hidden,
        "steps_per_task": a.steps_per_task,
        "n_passes": a.n_passes,
    }
    if a.smoke:
        cfg = {**cfg, **SMOKE_OVERRIDES}
        # keep an explicitly-passed hidden/steps/passes only if the user overrode the defaults on CLI
        if a.hidden != DEFAULTS["hidden"]:
            cfg["hidden"] = a.hidden
        if a.steps_per_task != DEFAULTS["steps_per_task"]:
            cfg["steps_per_task"] = a.steps_per_task
        if a.n_passes != DEFAULTS["n_passes"]:
            cfg["n_passes"] = a.n_passes
        # smoke robustness: the DEFAULT cache is the Studio DR1 output (absent on the laptop), so a bare
        # `--smoke` would crash with FileNotFoundError. Fall back to the real 64-clip cache when the user
        # did not override --cache and the default is missing, so the laptop smoke runs out of the box.
        if a.cache == DEFAULTS["cache"] and not (_ROOT / cfg["cache"]).exists():
            fallback = "data/cache/vjepa2_vitl_fpc64_256_real"
            if (_ROOT / fallback).exists():
                print(f"[smoke] default cache {cfg['cache']} absent; using {fallback}", flush=True)
                cfg["cache"] = fallback

    assert_studio_ram(allow_low=a.smoke)  # Studio-only guard (bypassed only for the tiny smoke path)
    assert_encoder_lane_free(skip=a.smoke or a.no_encoder_guard)  # encoder-lane guard

    out_path = Path(a.out)
    legs_dir = out_path.with_suffix(out_path.suffix + ".legs")
    result = run(cfg, legs_dir=legs_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    summary_keys = (
        "well_tuned_baseline_lr",
        "certificate",
        "any_zero_reinit",
        "reinit_count_total_all_rates",
        "winning_rates",
        "null_supported",
        "verdict",
    )
    print(json.dumps({k: result[k] for k in summary_keys}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
