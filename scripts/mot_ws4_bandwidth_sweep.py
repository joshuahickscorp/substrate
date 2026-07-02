#!/usr/bin/env python
"""WS4 (WP-12): the broadcast-bottleneck bandwidth sweep. Does a NARROW shared global-workspace slot
with broadcast-back show a bandwidth benefit at FIXED total capacity, beating both unbottlenecked
matched-capacity fusion and generic regularization tuned to the same effective bandwidth? Slot width is
swept while total params are held constant by the capmatch fixed-total-params solver (widening the
per-source streams as the slot narrows), so narrow-vs-wide is bandwidth allocation, never raw capacity.

Arms per seed (all trained on the same fit split, tuned on the same validation split, scored on the
same test split, identical steps and optimizer):
  bottleneck sweep: BottleneckFusion at each slot width in the sweep, total params fixed.
  write_only: the classifier reads the slot alone at the validation-best slot width (isolates
  broadcast-back, the null's 'broadcast adds nothing over write-only' clause).
  unbottlenecked: the same fusion with slot = stream width (no narrowing) at the same total params.
  regularization: the unbottlenecked model with dropout rates and weight-decay values tuned on the
  validation split (the preregistered 'it is just regularization' null arm).

PREREGISTERED NULL (verbatim, registry WS4): the bottleneck's benefit is indistinguishable from generic
regularization at matched capacity, and broadcast-back adds nothing over write-only: narrow-vs-wide is
just a capacity effect.
PREREGISTERED VERDICT RULE (fixed before any result exists): the null is REJECTED only if the
bandwidth benefit (best-slot minus unbottlenecked accuracy) AND the regularization delta (best-slot
minus best-regularized accuracy) both have 5-seed 95 percent CIs excluding zero from below with no sign
flips, AND the validation-best slot is interior (narrower than the widest slot) in a majority of seeds.

Usage: python scripts/mot_ws4_bandwidth_sweep.py --seeds 0-4

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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.mot_ws1_agreement_vs_confidence import (  # noqa: E402
    load_dual_source,
    split_train_test,
    zscore_by,
)

from devsys.diagnostics.compute import param_count  # noqa: E402
from devsys.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from devsys.experiments.base import Experiment  # noqa: E402
from devsys.shell.capmatch import fixed_total_params_sweep  # noqa: E402

DEFAULTS = {
    "cache_a": "data/cache/vjepa2_vitl_nuisance",
    "cache_b": "data/cache/dinov2s_nuisance_real",
    "seeds": (0, 1, 2, 3, 4),
    "epochs": 300,
    "lr": 1e-2,
    "test_frac": 0.3,
    "val_frac": 0.25,
    "width": 64,
    "slots": (4, 8, 16, 32, 64),
    "dropout_grid": (0.1, 0.3, 0.5),
    "weight_decay_grid": (1e-4, 1e-3, 1e-2),
    "capmatch_tol": 0.02,
}


def _get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class BottleneckFusion(nn.Module):
    """Two source streams write into a narrow shared slot; the slot is broadcast back into both streams
    (write_only=False) or read out alone (write_only=True). Param count is monotone in `width` at fixed
    slot, so the capmatch solver can hold total params constant across the sweep."""

    def __init__(
        self,
        da: int,
        db: int,
        n_classes: int,
        width: int,
        slot: int,
        *,
        write_only: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.write_only = write_only
        self.pa, self.pb = nn.Linear(da, width), nn.Linear(db, width)
        self.write = nn.Linear(2 * width, slot)
        self.drop = nn.Dropout(dropout)
        if write_only:
            self.head = nn.Linear(slot, n_classes)
        else:
            self.ua, self.ub = nn.Linear(slot, width), nn.Linear(slot, width)
            self.head = nn.Linear(2 * width, n_classes)

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        ha, hb = self.pa(za), self.pb(zb)
        s = torch.tanh(self.write(self.drop(torch.cat([ha, hb], dim=-1))))
        if self.write_only:
            return self.head(s)
        ha, hb = F.relu(ha + self.ua(s)), F.relu(hb + self.ub(s))
        return self.head(self.drop(torch.cat([ha, hb], dim=-1)))


def _train(model: nn.Module, xa, xb, y, *, epochs: int, lr: float, seed: int, weight_decay: float = 0.0):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(xa, xb), y).backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def _acc(model: nn.Module, xa, xb, y) -> float:
    return float((model(xa, xb).argmax(-1) == y).float().mean())


def run(cfg=None, device=None, run_dir: Path | None = None) -> dict:
    za, zb, y = load_dual_source(
        _get(cfg, "cache_a", DEFAULTS["cache_a"]), _get(cfg, "cache_b", DEFAULTS["cache_b"])
    )
    seeds = list(_get(cfg, "seeds", DEFAULTS["seeds"]))
    epochs = int(_get(cfg, "epochs", DEFAULTS["epochs"]))
    lr = float(_get(cfg, "lr", DEFAULTS["lr"]))
    tol = float(_get(cfg, "capmatch_tol", DEFAULTS["capmatch_tol"]))
    slots = [int(s) for s in _get(cfg, "slots", DEFAULTS["slots"])]
    w0 = int(_get(cfg, "width", DEFAULTS["width"]))
    da, db, n_classes = za.shape[1], zb.shape[1], int(y.max()) + 1
    total = param_count(BottleneckFusion(da, db, n_classes, w0, w0))
    sweep = fixed_total_params_sweep(
        lambda slot, width: BottleneckFusion(da, db, n_classes, width, slot), total, slots, tol=tol
    )
    t0 = time.perf_counter()
    per_seed: list[dict] = []
    for s in seeds:
        tr_all, te = split_train_test(len(y), float(_get(cfg, "test_frac", DEFAULTS["test_frac"])), s)
        n_val = max(1, int(len(tr_all) * float(_get(cfg, "val_frac", DEFAULTS["val_frac"]))))
        fit, val = tr_all[n_val:], tr_all[:n_val]
        xa, xb = zscore_by(za[fit], za), zscore_by(zb[fit], zb)

        def fit_score(model, weight_decay=0.0, seed_off=0, s=s, fit=fit, val=val, te=te, xa=xa, xb=xb):
            _train(
                model,
                xa[fit],
                xb[fit],
                y[fit],
                epochs=epochs,
                lr=lr,
                seed=s + seed_off,
                weight_decay=weight_decay,
            )
            return _acc(model, xa[val], xb[val], y[val]), _acc(model, xa[te], xb[te], y[te])

        slot_scores = {
            n: fit_score(BottleneckFusion(da, db, n_classes, spec["width"], n)) for n, spec in sweep.items()
        }
        best_slot = max(slot_scores, key=lambda n: slot_scores[n][0])
        wo_model = BottleneckFusion(da, db, n_classes, sweep[best_slot]["width"], best_slot, write_only=True)
        wo_acc = fit_score(wo_model, seed_off=1)[1]
        unb_acc = fit_score(BottleneckFusion(da, db, n_classes, w0, w0), seed_off=2)[1]
        reg_scores = []
        for p in _get(cfg, "dropout_grid", DEFAULTS["dropout_grid"]):
            m = BottleneckFusion(da, db, n_classes, w0, w0, dropout=float(p))
            reg_scores.append((f"dropout_{p}", *fit_score(m, seed_off=3)))
        for wd in _get(cfg, "weight_decay_grid", DEFAULTS["weight_decay_grid"]):
            m = BottleneckFusion(da, db, n_classes, w0, w0)
            reg_scores.append((f"wd_{wd}", *fit_score(m, weight_decay=float(wd), seed_off=4)))
        best_reg = max(reg_scores, key=lambda r: r[1])
        slot_acc = slot_scores[best_slot][1]
        per_seed.append(
            {
                "seed": s,
                "slot_test_acc": {n: round(v[1], 4) for n, v in slot_scores.items()},
                "best_slot": best_slot,
                "best_slot_acc": round(slot_acc, 4),
                "write_only_acc": round(wo_acc, 4),
                "unbottlenecked_acc": round(unb_acc, 4),
                "best_reg": best_reg[0],
                "best_reg_acc": round(best_reg[2], 4),
                "bandwidth_benefit": slot_acc - unb_acc,
                "reg_delta": slot_acc - best_reg[2],
                "broadcast_minus_write_only": slot_acc - wo_acc,
                "interior_optimum": best_slot < max(slots),
            }
        )
    bb = [r["bandwidth_benefit"] for r in per_seed]
    rd = [r["reg_delta"] for r in per_seed]
    bb_ci, rd_ci = seed_ci(bb), seed_ci(rd)
    interior_majority = sum(r["interior_optimum"] for r in per_seed) > len(per_seed) / 2
    positive = bool(
        bb_ci["lo"] > 0
        and rd_ci["lo"] > 0
        and sign_flip_report(bb)["consistent_sign"] == 1
        and sign_flip_report(rd)["consistent_sign"] == 1
        and interior_majority
    )
    out = {
        "experiment": WS4Experiment.id,
        "contract": WS4Experiment().contract(),
        "seeds": seeds,
        "total_params": total,
        "sweep": {n: sweep[n] for n in sorted(sweep)},
        "per_seed": per_seed,
        "bandwidth_benefit_ci": bb_ci,
        "reg_delta_ci": rd_ci,
        "broadcast_minus_write_only_ci": seed_ci([r["broadcast_minus_write_only"] for r in per_seed]),
        "interior_majority": interior_majority,
        "null_supported": not positive,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": (
            "NULL REJECTED: a narrow broadcast slot beats both unbottlenecked fusion and tuned "
            "regularization at fixed total params, a specific GWT narrowness claim survives"
            if positive
            else "NULL SUPPORTED: the bottleneck's effect is indistinguishable from capacity or "
            "generic regularization (the preregistered corpus expectation)"
        ),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


class WS4Experiment(Experiment):
    id = "ws4_bandwidth_sweep"
    metric = ("bandwidth_benefit", "reg_delta", "broadcast_minus_write_only")
    baseline = "unbottlenecked matched-capacity fusion and dropout/weight-decay tuned on validation"
    ablation = "slot width swept at fixed total params; write-only removes broadcast-back"
    null_hypothesis = (
        "The bottleneck's benefit is indistinguishable from generic regularization at matched capacity, "
        "and broadcast-back adds nothing over write-only: narrow-vs-wide is just a capacity effect"
    )
    tier = "cpu-now"

    def run(self, cfg, device, run_dir: Path) -> dict:
        return run(cfg, device, run_dir)


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WS4 broadcast-bottleneck bandwidth sweep at fixed params")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache-a", default=DEFAULTS["cache_a"])
    ap.add_argument("--cache-b", default=DEFAULTS["cache_b"])
    ap.add_argument("--width", type=int, default=DEFAULTS["width"])
    ap.add_argument("--rerun", action="store_true", help="stage-4 rerun naming (_seeds10 suffix)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {"seeds": _parse_seeds(a.seeds), "cache_a": a.cache_a, "cache_b": a.cache_b, "width": a.width}
    out_path = a.out or (
        "runs/mot/ws4_bandwidth_sweep_seeds10.json" if a.rerun else "runs/mot/ws4_bandwidth_sweep.json"
    )
    result = run(cfg, "cpu", None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {k: result[k] for k in ("bandwidth_benefit_ci", "reg_delta_ci", "null_supported", "verdict")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
