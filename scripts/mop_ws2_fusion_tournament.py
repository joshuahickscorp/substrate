#!/usr/bin/env python
"""WS2 (WP-12): the matched-capacity fusion tournament. At STRICTLY matched trainable-param count
(solved by the capmatch helper, never eyeballed) and matched training protocol (identical steps,
optimizer, batch regime, so training FLOPs track params), does any STRUCTURED fusion of the two frozen
sources beat the unstructured concat-MLP floor on held-out accuracy or NLL?

Arms (all built at the concat-MLP reference's param count within capmatch tol):
  concat_mlp (the floor and reference), learned_linear (per-source projections summed, purely linear
  fusion), cross_attention (two source tokens, one multihead attention step), gwt_broadcast (narrow
  shared slot written by both sources and broadcast back).

PREREGISTERED NULL (verbatim, registry WS2): at matched capacity every structured fusion ties the
concat-MLP on held-out accuracy and NLL: structure buys nothing beyond param count.
PREREGISTERED VERDICT RULE (fixed before any result exists): the null is REJECTED only if at least one
structured arm's accuracy delta OR NLL improvement over concat-MLP has a 5-seed 95 percent CI excluding
zero from below with no per-seed sign flip. Param counts for every arm are recorded in the output; a
capmatch failure aborts the run rather than comparing mismatched capacity.

Usage: python scripts/mop_ws2_fusion_tournament.py --seeds 0-4

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

from scripts.mop_ws1_agreement_vs_confidence import (  # noqa: E402
    load_dual_source,
    split_train_test,
    zscore_by,
)

from mop.diagnostics.compute import param_count  # noqa: E402
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from mop.experiments.base import Experiment  # noqa: E402
from mop.seeding import parse_seeds as _parse_seeds  # noqa: E402
from mop.shell.capmatch import width_for_param_count  # noqa: E402

DEFAULTS = {
    "cache_a": "data/cache/vjepa2_vitl_nuisance",
    "cache_b": "data/cache/dinov2s_nuisance_real",
    "seeds": (0, 1, 2, 3, 4),
    "epochs": 300,
    "lr": 1e-2,
    "test_frac": 0.3,
    "width": 64,
    "capmatch_tol": 0.02,
}


def _get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ConcatMLP(nn.Module):
    """The unstructured floor: concat both sources, one hidden layer, classify."""

    def __init__(self, da: int, db: int, n_classes: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(da + db, width), nn.ReLU(), nn.Linear(width, n_classes))

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([za, zb], dim=-1))


class LearnedLinear(nn.Module):
    """Purely linear fusion: per-source projections summed, then a linear head. No nonlinearity, so any
    win over it by the others is structure or depth, and its win over concat would be parsimony."""

    def __init__(self, da: int, db: int, n_classes: int, width: int):
        super().__init__()
        self.pa, self.pb = nn.Linear(da, width), nn.Linear(db, width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        return self.head(self.pa(za) + self.pb(zb))


class CrossAttention(nn.Module):
    """Two source tokens, one multihead self-attention step, mean-pool, classify. The requested width
    is rounded UP to the nearest even number (num_heads=2 needs an even embed_dim) so the capmatch
    integer search can probe any width; param count stays non-decreasing in the requested width."""

    def __init__(self, da: int, db: int, n_classes: int, width: int):
        super().__init__()
        width = width + (width % 2)
        self.pa, self.pb = nn.Linear(da, width), nn.Linear(db, width)
        self.attn = nn.MultiheadAttention(width, num_heads=2, batch_first=True)
        self.head = nn.Linear(width, n_classes)

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([self.pa(za), self.pb(zb)], dim=1)
        out, _ = self.attn(tokens, tokens, tokens)
        return self.head(out.mean(dim=1))


class GWTBroadcast(nn.Module):
    """Global-workspace fusion: both sources write into a narrow shared slot (width // 4), the slot is
    broadcast back into each source's stream, then the streams are read out jointly."""

    def __init__(self, da: int, db: int, n_classes: int, width: int):
        super().__init__()
        slot = max(1, width // 4)
        self.pa, self.pb = nn.Linear(da, width), nn.Linear(db, width)
        self.write = nn.Linear(2 * width, slot)
        self.ua, self.ub = nn.Linear(slot, width), nn.Linear(slot, width)
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        ha, hb = self.pa(za), self.pb(zb)
        s = torch.tanh(self.write(torch.cat([ha, hb], dim=-1)))
        ha, hb = F.relu(ha + self.ua(s)), F.relu(hb + self.ub(s))
        return self.head(torch.cat([ha, hb], dim=-1))


ARM_MAKERS = {
    "learned_linear": LearnedLinear,
    "cross_attention": CrossAttention,
    "gwt_broadcast": GWTBroadcast,
}


def build_matched_arms(da: int, db: int, n_classes: int, width: int, tol: float) -> dict[str, dict]:
    """Solve each structured arm's width so its param count matches the concat-MLP reference within
    tol (H-CAPMATCH). Returns arm -> {make, width, params}; raises on any capmatch failure. The search
    upper bound is 8x the reference width: every structured arm grows at least as fast per unit width
    as the concat reference, so the solution lies far below that, and an unbounded search would probe
    the capmatch default hi (65536), transiently building a multi-GB attention module (RAM law)."""
    target = param_count(ConcatMLP(da, db, n_classes, width))
    arms: dict[str, dict] = {
        "concat_mlp": {"make": lambda: ConcatMLP(da, db, n_classes, width), "width": width, "params": target}
    }
    hi = max(64, 8 * width)
    for name, cls in ARM_MAKERS.items():
        w, p = width_for_param_count(lambda w, c=cls: c(da, db, n_classes, w), target, tol=tol, hi=hi)
        arms[name] = {"make": lambda w=w, c=cls: c(da, db, n_classes, w), "width": w, "params": p}
    return arms


def train_and_score(make, xa_tr, xb_tr, ytr, xa_te, xb_te, yte, *, epochs: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(xa_tr, xb_tr), ytr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(xa_te, xb_te)
        acc = float((logits.argmax(-1) == yte).float().mean())
        nll = float(F.cross_entropy(logits, yte))
    return {"acc": acc, "nll": nll}


def run(cfg=None, device=None, run_dir: Path | None = None) -> dict:
    za, zb, y = load_dual_source(
        _get(cfg, "cache_a", DEFAULTS["cache_a"]), _get(cfg, "cache_b", DEFAULTS["cache_b"])
    )
    seeds = list(_get(cfg, "seeds", DEFAULTS["seeds"]))
    epochs = int(_get(cfg, "epochs", DEFAULTS["epochs"]))
    lr = float(_get(cfg, "lr", DEFAULTS["lr"]))
    test_frac = float(_get(cfg, "test_frac", DEFAULTS["test_frac"]))
    n_classes = int(y.max()) + 1
    arms = build_matched_arms(
        za.shape[1],
        zb.shape[1],
        n_classes,
        int(_get(cfg, "width", DEFAULTS["width"])),
        float(_get(cfg, "capmatch_tol", DEFAULTS["capmatch_tol"])),
    )
    t0 = time.perf_counter()
    per_seed: list[dict] = []
    for s in seeds:
        tr, te = split_train_test(len(y), test_frac, s)
        xa, xb = zscore_by(za[tr], za), zscore_by(zb[tr], zb)
        scores = {
            name: train_and_score(
                spec["make"], xa[tr], xb[tr], y[tr], xa[te], xb[te], y[te], epochs=epochs, lr=lr, seed=s
            )
            for name, spec in arms.items()
        }
        per_seed.append({"seed": s, "scores": scores})
    structured = sorted(ARM_MAKERS)
    arm_reports: dict[str, dict] = {}
    any_win = False
    for name in structured:
        acc_d = [r["scores"][name]["acc"] - r["scores"]["concat_mlp"]["acc"] for r in per_seed]
        nll_d = [r["scores"]["concat_mlp"]["nll"] - r["scores"][name]["nll"] for r in per_seed]
        acc_ci, nll_ci = seed_ci(acc_d), seed_ci(nll_d)
        acc_win = acc_ci["lo"] > 0 and sign_flip_report(acc_d)["consistent_sign"] == 1
        nll_win = nll_ci["lo"] > 0 and sign_flip_report(nll_d)["consistent_sign"] == 1
        any_win = any_win or acc_win or nll_win
        arm_reports[name] = {
            "width": arms[name]["width"],
            "params": arms[name]["params"],
            "acc_delta_ci": acc_ci,
            "nll_delta_ci": nll_ci,
            "acc_win": acc_win,
            "nll_win": nll_win,
        }
    null_supported = not any_win
    out = {
        "experiment": WS2Experiment.id,
        "contract": WS2Experiment().contract(),
        "seeds": seeds,
        "reference_params": arms["concat_mlp"]["params"],
        "per_seed": per_seed,
        "arms": arm_reports,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": (
            "NULL SUPPORTED: every structured fusion ties the concat-MLP floor at matched capacity, "
            "structure is decoration (mirrors the corpus negatives)"
            if null_supported
            else "NULL REJECTED: at matched params a structured fusion beats concat-MLP, fusion "
            "structure is more than capacity"
        ),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


class WS2Experiment(Experiment):
    id = "ws2_fusion_tournament"
    metric = ("heldout_acc_delta_vs_concat", "nll_delta_vs_concat")
    baseline = "unstructured concat-MLP at the identical param count and training protocol"
    ablation = "fusion structure varies while capacity is held fixed by capmatch"
    null_hypothesis = (
        "At matched capacity every structured fusion ties the concat-MLP on held-out accuracy and NLL: "
        "structure buys nothing beyond param count"
    )
    tier = "cpu-now"

    def run(self, cfg, device, run_dir: Path) -> dict:
        return run(cfg, device, run_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WS2 matched-capacity fusion tournament")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache-a", default=DEFAULTS["cache_a"])
    ap.add_argument("--cache-b", default=DEFAULTS["cache_b"])
    ap.add_argument("--width", type=int, default=DEFAULTS["width"])
    ap.add_argument("--rerun", action="store_true", help="stage-4 rerun naming (_seeds10 suffix)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {"seeds": _parse_seeds(a.seeds), "cache_a": a.cache_a, "cache_b": a.cache_b, "width": a.width}
    out_path = a.out or (
        "runs/mot/ws2_fusion_tournament_seeds10.json" if a.rerun else "runs/mot/ws2_fusion_tournament.json"
    )
    result = run(cfg, "cpu", None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("arms", "null_supported", "verdict")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
