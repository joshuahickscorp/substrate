#!/usr/bin/env python
"""WS3 (WP-12, gated on a positive WS1): capacity-neutral arbitration. Given that agreement carries
correctness information (the WS1 gate), do fixed Bayesian-style rules beat naive combination with
essentially zero added parameters?

Arms (per-source heads are the ONLY trained modules, shared by every rule; the expensive path is one
small concat-MLP used by both routing arms identically):
  equal_weight: mean of the two source probability vectors.
  inverse_variance: per-sample precision weights 1 / (entropy_i + eps), the fixed Bayesian rule.
  disagreement_routed: the top route_frac samples by disagreement (1 - sum_k pA_k pB_k) get the
  expensive concat-MLP prediction, the rest get equal_weight.
  random_routed: the SAME expensive-path budget spent on a seeded random subset (the matched-rate
  control), plus single_best_source and concat_mlp reported for context.

GATE: this experiment runs only if `ws1_positive` is true in the WS1 verdict json; otherwise it writes
a SKIPPED record citing the WS1 verdict (Q3.7). PREREGISTERED NULL (verbatim, registry WS3): precision
weights are uninformative so inverse-variance ties equal-weight averaging, and disagreement is
uncorrelated with correctness so disagreement routing ties random routing at matched rate; OR
disagreement fires on aleatoric noise (fails noisy-TV).
PREREGISTERED VERDICT RULE (fixed before any result exists): the null is REJECTED only if BOTH the
inverse-variance delta over equal-weight AND the disagreement-routing delta over random routing have
5-seed 95 percent CIs excluding zero from below with no sign flips, AND noisy-TV passes on every seed.
A noisy-TV failure overrides any primary win.

Usage: python scripts/mop_ws3_arbitration.py --seeds 0-4

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.mop_ws1_agreement_vs_confidence import (  # noqa: E402
    load_dual_source,
    predict_probs,
    split_train_test,
    train_softmax_head,
    zscore_by,
)
from scripts.mop_ws2_fusion_tournament import ConcatMLP  # noqa: E402

from mop.diagnostics.noisy_tv import noisy_tv_diagnostic  # noqa: E402
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from mop.experiments.base import Experiment  # noqa: E402
from mop.seeding import parse_seeds as _parse_seeds  # noqa: E402

DEFAULTS = {
    "cache_a": "data/cache/vjepa2_vitl_nuisance",
    "cache_b": "data/cache/dinov2s_nuisance_real",
    "ws1_verdict": "runs/mot/ws1_agreement_vs_confidence.json",
    "seeds": (0, 1, 2, 3, 4),
    "epochs": 300,
    "lr": 1e-2,
    "test_frac": 0.3,
    "route_frac": 0.3,
    "entropy_eps": 0.05,
    "concat_width": 32,
    "noisy_tv_steps": 400,
    "noisy_tv_dim": 32,
}


def _get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p + 1e-12).log()).sum(-1)


def inverse_variance_fuse(pa: torch.Tensor, pb: torch.Tensor, eps: float) -> torch.Tensor:
    """Precision-weighted fusion with per-sample weights 1 / (predictive entropy + eps): the fixed
    zero-parameter Bayesian rule."""
    wa, wb = 1.0 / (_entropy(pa) + eps), 1.0 / (_entropy(pb) + eps)
    w = wa + wb
    return (wa / w).unsqueeze(-1) * pa + (wb / w).unsqueeze(-1) * pb


def routed_accuracy(
    fused: torch.Tensor, expensive: torch.Tensor, route_mask: torch.Tensor, y: torch.Tensor
) -> float:
    """Accuracy when routed samples take the expensive path and the rest take the cheap fusion."""
    preds = torch.where(route_mask, expensive.argmax(-1), fused.argmax(-1))
    return float((preds == y).float().mean())


def run_seed(za: torch.Tensor, zb: torch.Tensor, y: torch.Tensor, seed: int, cfg) -> dict:
    epochs = int(_get(cfg, "epochs", DEFAULTS["epochs"]))
    lr = float(_get(cfg, "lr", DEFAULTS["lr"]))
    eps = float(_get(cfg, "entropy_eps", DEFAULTS["entropy_eps"]))
    route_frac = float(_get(cfg, "route_frac", DEFAULTS["route_frac"]))
    n_classes = int(y.max()) + 1
    tr, te = split_train_test(len(y), float(_get(cfg, "test_frac", DEFAULTS["test_frac"])), seed)
    xa, xb = zscore_by(za[tr], za), zscore_by(zb[tr], zb)
    head_a = train_softmax_head(xa[tr], y[tr], n_classes, epochs=epochs, lr=lr, seed=seed)
    head_b = train_softmax_head(xb[tr], y[tr], n_classes, epochs=epochs, lr=lr, seed=seed + 1)
    concat = ConcatMLP(za.shape[1], zb.shape[1], n_classes, int(_get(cfg, "concat_width", 32)))
    torch.manual_seed(seed + 2)
    opt = torch.optim.Adam(concat.parameters(), lr=lr)
    concat.train()
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(concat(xa[tr], xb[tr]), y[tr]).backward()
        opt.step()
    concat.eval()
    pa, pb = predict_probs(head_a, xa[te]), predict_probs(head_b, xb[te])
    with torch.no_grad():
        p_exp = torch.softmax(concat(xa[te], xb[te]), dim=-1)
    yte = y[te]
    equal = (pa + pb) / 2
    invvar = inverse_variance_fuse(pa, pb, eps)
    acc = {
        "single_best": max(
            float((pa.argmax(-1) == yte).float().mean()), float((pb.argmax(-1) == yte).float().mean())
        ),
        "equal_weight": float((equal.argmax(-1) == yte).float().mean()),
        "inverse_variance": float((invvar.argmax(-1) == yte).float().mean()),
        "concat_mlp": float((p_exp.argmax(-1) == yte).float().mean()),
    }
    n_route = max(1, int(route_frac * len(te)))
    disagreement = 1.0 - (pa * pb).sum(-1)
    mask_d = torch.zeros(len(te), dtype=torch.bool)
    mask_d[disagreement.topk(n_route).indices] = True
    g = torch.Generator().manual_seed(seed + 5)
    mask_r = torch.zeros(len(te), dtype=torch.bool)
    mask_r[torch.randperm(len(te), generator=g)[:n_route]] = True
    acc["disagreement_routed"] = routed_accuracy(equal, p_exp, mask_d, yte)
    acc["random_routed"] = routed_accuracy(equal, p_exp, mask_r, yte)
    ntv = noisy_tv_diagnostic(
        dim=int(_get(cfg, "noisy_tv_dim", DEFAULTS["noisy_tv_dim"])),
        ensemble_size=3,
        steps=int(_get(cfg, "noisy_tv_steps", DEFAULTS["noisy_tv_steps"])),
        batch=64,
        seed=seed,
    )
    return {
        "seed": seed,
        "acc": {k: round(v, 4) for k, v in acc.items()},
        "invvar_delta": acc["inverse_variance"] - acc["equal_weight"],
        "routing_delta": acc["disagreement_routed"] - acc["random_routed"],
        "noisy_tv_pass": bool(
            ntv["noise_error_stays_high"]
            and ntv["epistemic_collapses_on_noise"]
            and ntv["learning_progress_separates"]
        ),
    }


def load_dual_source_d3(seed: int, n: int = 2000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A5 recal dual source: two distinct views of the D3 graded slot latent over the SAME samples, so
    per-sample disagreement is decision-relevant (the two views disagree exactly where corruption bites,
    the D3 hard bin). Source A sees the four slot blocks with independent view-noise; source B sees the
    same latent under a DIFFERENT independent view-noise draw plus a fixed random rotation, so the two
    heads land in different solutions and their agreement carries correctness information (the WS1
    premise that the base ws3 could not establish on the real caches). y is the D3 DSL label."""
    from mop.diagnostics.hardness import make_graded_slot_task

    task = make_graded_slot_task(n, noise=2.4, seed=seed)
    x = task.x
    g = torch.Generator().manual_seed(seed + 71)
    za = x + torch.randn_like(x) * 0.5
    rot = torch.linalg.qr(torch.randn(x.shape[1], x.shape[1], generator=g))[0]
    zb = (x + torch.randn_like(x) * 0.5) @ rot
    return za, zb, task.y


def _read_ws1(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"ws1_positive": False, "verdict": f"WS1 verdict json missing at {path}"}
    return json.loads(p.read_text())


def run(cfg=None, device=None, run_dir: Path | None = None) -> dict:
    ws1 = _read_ws1(_get(cfg, "ws1_verdict", DEFAULTS["ws1_verdict"]))
    base = {"experiment": WS3Experiment.id, "contract": WS3Experiment().contract()}
    if not ws1.get("ws1_positive", False):
        out = dict(base)
        out.update(
            {
                "skipped": True,
                "ws1_verdict": ws1.get("verdict", "unavailable"),
                "null_supported": True,
                "verdict": "SKIPPED: WS1 gate not positive, WS3 nulls stand unexamined by "
                "preregistration (arbitration is only worth testing once agreement carries information)",
            }
        )
        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
        return out
    za, zb, y = load_dual_source(
        _get(cfg, "cache_a", DEFAULTS["cache_a"]), _get(cfg, "cache_b", DEFAULTS["cache_b"])
    )
    seeds = list(_get(cfg, "seeds", DEFAULTS["seeds"]))
    t0 = time.perf_counter()
    per_seed = [run_seed(za, zb, y, s, cfg) for s in seeds]
    invvar_d = [r["invvar_delta"] for r in per_seed]
    routing_d = [r["routing_delta"] for r in per_seed]
    invvar_ci, routing_ci = seed_ci(invvar_d), seed_ci(routing_d)
    invvar_win = invvar_ci["lo"] > 0 and sign_flip_report(invvar_d)["consistent_sign"] == 1
    routing_win = routing_ci["lo"] > 0 and sign_flip_report(routing_d)["consistent_sign"] == 1
    noisy_pass = all(r["noisy_tv_pass"] for r in per_seed)
    positive = bool(invvar_win and routing_win and noisy_pass)
    out = dict(base)
    out.update(
        {
            "skipped": False,
            "seeds": seeds,
            "per_seed": per_seed,
            "invvar_delta_ci": invvar_ci,
            "routing_delta_ci": routing_ci,
            "invvar_win": invvar_win,
            "routing_win": routing_win,
            "noisy_tv_pass": noisy_pass,
            "null_supported": not positive,
            "seconds": round(time.perf_counter() - t0, 1),
            "verdict": (
                "NULL REJECTED: inverse-variance beats averaging AND disagreement routing beats random "
                "routing at matched rate with noisy-TV clean, arbitration is a capacity-neutral win"
                if positive
                else "NULL SUPPORTED: precision/disagreement signals uninformative (or noisy-TV "
                f"failed={not noisy_pass}), extends the e4 line"
            ),
        }
    )
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_recal(cfg=None, run_dir: Path | None = None) -> dict:
    """A5 recalibration: the base ws3 self-reported SKIPPED because it was gated on a positive WS1 that
    never materialized on the real caches (agreement carried no correctness information there). This
    driver UNGATES arbitration by running the identical arms (equal-weight, inverse-variance,
    disagreement-routed vs random-routed at matched rate, all controls and the noisy-TV guard verbatim)
    on the D3 graded slot task, where per-sample hardness IS decision-relevant so agreement/disagreement
    can in principle carry information. Verdict rule is unchanged: the null is REJECTED only if BOTH the
    inverse-variance delta and the disagreement-routing delta have CIs excluding zero from below with no
    sign flips AND noisy-TV passes every seed. A fresh D3 dual source is drawn per seed."""
    base = {"experiment": WS3Experiment.id, "contract": WS3Experiment().contract()}
    seeds = list(_get(cfg, "seeds", DEFAULTS["seeds"]))
    t0 = time.perf_counter()
    per_seed = []
    for s in seeds:
        za, zb, y = load_dual_source_d3(s)
        per_seed.append(run_seed(za, zb, y, s, cfg))
    invvar_d = [r["invvar_delta"] for r in per_seed]
    routing_d = [r["routing_delta"] for r in per_seed]
    invvar_ci, routing_ci = seed_ci(invvar_d), seed_ci(routing_d)
    invvar_win = invvar_ci["lo"] > 0 and sign_flip_report(invvar_d)["consistent_sign"] == 1
    routing_win = routing_ci["lo"] > 0 and sign_flip_report(routing_d)["consistent_sign"] == 1
    noisy_pass = all(r["noisy_tv_pass"] for r in per_seed)
    positive = bool(invvar_win and routing_win and noisy_pass)
    out = dict(base)
    out.update(
        {
            "skipped": False,
            "recal": "D3 graded slot task (diagnostics/hardness.make_graded_slot_task), WS1 gate removed",
            "seeds": seeds,
            "per_seed": per_seed,
            "invvar_delta_ci": invvar_ci,
            "routing_delta_ci": routing_ci,
            "invvar_win": invvar_win,
            "routing_win": routing_win,
            "noisy_tv_pass": noisy_pass,
            "null_supported": not positive,
            "seconds": round(time.perf_counter() - t0, 1),
            "verdict": (
                "NULL REJECTED: inverse-variance beats averaging AND disagreement routing beats random "
                "routing at matched rate with noisy-TV clean on the D3 regime, arbitration is a "
                "capacity-neutral win"
                if positive
                else "NULL SUPPORTED: precision/disagreement signals uninformative on the D3 regime (or "
                f"noisy-TV failed={not noisy_pass}), extends the e4 line even where hardness is "
                "decision-relevant"
            ),
        }
    )
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


class WS3Experiment(Experiment):
    id = "ws3_arbitration"
    metric = ("invvar_minus_equal_weight_acc", "disagreement_minus_random_routing_acc")
    baseline = "equal-weight averaging and random routing at the identical expensive-path rate"
    ablation = "precision weights and disagreement routing removed (equal weights, random routes)"
    null_hypothesis = (
        "Precision weights are uninformative so inverse-variance ties equal-weight averaging, and "
        "disagreement is uncorrelated with correctness so disagreement routing ties random routing at "
        "matched rate; OR disagreement fires on aleatoric noise (fails noisy-TV)"
    )
    tier = "cpu-now"

    def run(self, cfg, device, run_dir: Path) -> dict:
        return run(cfg, device, run_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WS3 uncertainty vs disagreement arbitration (WS1-gated)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache-a", default=DEFAULTS["cache_a"])
    ap.add_argument("--cache-b", default=DEFAULTS["cache_b"])
    ap.add_argument("--ws1-verdict", default=DEFAULTS["ws1_verdict"])
    ap.add_argument("--rerun", action="store_true", help="stage-4 rerun naming (_seeds10 suffix)")
    ap.add_argument("--recal", action="store_true", help="A5: ungate on D3 task, write _recal.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {
        "seeds": _parse_seeds(a.seeds),
        "cache_a": a.cache_a,
        "cache_b": a.cache_b,
        "ws1_verdict": a.ws1_verdict,
    }
    if a.recal:
        out_path = a.out or "runs/mot/ws3_arbitration_recal.json"
        result = run_recal(cfg, None)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2, default=str))
        print(
            json.dumps(
                {k: result[k] for k in ("invvar_delta_ci", "routing_delta_ci", "null_supported", "verdict")},
                indent=2,
            )
        )
        return 0
    out_path = a.out or (
        "runs/mot/ws3_arbitration_seeds10.json" if a.rerun else "runs/mot/ws3_arbitration.json"
    )
    result = run(cfg, "cpu", None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    keys = (
        ("null_supported", "verdict")
        if result.get("skipped")
        else ("invvar_delta_ci", "routing_delta_ci", "null_supported", "verdict")
    )
    print(json.dumps({k: result[k] for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
