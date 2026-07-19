#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mop.diagnostics.noisy_tv import noisy_tv_diagnostic
from mop.diagnostics.riskcov import auroc, ece_equal_mass, seed_ci, sign_flip_report
from mop.seeding import parse_seeds  # noqa: E402
from mop.shell.predictor import mlp

CONTRACT = {
    "id": "DR12",
    "metric": ("error_prediction_auroc_delta", "gated_accuracy_delta_at_matched_budget"),
    "baseline": "single-head confidence for AUROC; uniform-random gating at the same expert budget",
    "ablation": "confidence-gated arm, oracle-gated upper bound, ensemble-mean confidence variant",
    "null_hypothesis": (
        "Disagreement ties single-head confidence AND gating on it ties uniform compute, OR it "
        "chases irreducible noise (fails noisy-TV, the e4 pattern)"
    ),
    "tier": "cpu-now",
}

PARTS = ["blobs", "antipodal", "noise"]
PART_FRACS = (0.4, 0.3, 0.3)
DEGENERATE_HIGH = 0.97
DEGENERATE_LOW_MARGIN = 0.05
PR1_PATH = "runs/pre_studio/pr1_mode_error_disjointness.json"


def read_pr1_context(path: str | Path = PR1_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {
            "available": False,
            "green": False,
            "gate": "missing",
            "note": "PR1 verdict json missing; routing rows report as unverified context",
        }
    d = json.loads(p.read_text())
    verdict = str(d.get("verdict", ""))
    green = verdict.startswith("GREEN")
    cfg = d.get("config", {})
    return {
        "available": True,
        "green": green,
        "gate": "live" if green else "context-null",
        "verdict": verdict,
        "het_oracle_gain_mean": (d.get("het_oracle_gain") or {}).get("mean"),
        "hom_oracle_gain_mean": (d.get("hom_oracle_gain") or {}).get("mean"),
        "seed_sd_of_best_mode": d.get("seed_sd_of_best_mode"),
        "chosen_separation": (d.get("calibration") or {}).get("chosen_separation"),
        "config": {k: cfg.get(k) for k in ("dim", "n_classes", "separation", "n_train", "n_test")},
    }


def make_mixed_regime(
    seed: int, n_train: int, n_test: int, dim: int, n_classes: int, separation: float
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    blob_centers = torch.randn(n_classes, dim, generator=g)
    anti_centers = torch.randn(n_classes, dim, generator=g)

    def draw(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_blob = int(n * PART_FRACS[0])
        n_anti = int(n * PART_FRACS[1])
        n_noise = n - n_blob - n_anti
        yb = torch.randint(0, n_classes, (n_blob,), generator=g)
        xb = blob_centers[yb] * separation + 0.5 * torch.randn(n_blob, dim, generator=g)
        ya = torch.randint(0, n_classes, (n_anti,), generator=g)
        signs = (torch.randint(0, 2, (n_anti, 1), generator=g).float() * 2.0) - 1.0
        xa = anti_centers[ya] * separation * signs + 0.5 * torch.randn(n_anti, dim, generator=g)
        yn = torch.randint(0, n_classes, (n_noise,), generator=g)
        xn = 0.5 * torch.randn(n_noise, dim, generator=g)
        x = torch.cat([xb, xa, xn])
        y = torch.cat([yb, ya, yn])
        p = torch.cat([torch.zeros(n_blob), torch.ones(n_anti), torch.full((n_noise,), 2.0)]).long()
        perm = torch.randperm(n, generator=g)
        return x[perm], y[perm], p[perm]

    xtr, ytr, ptr = draw(n_train)
    xte, yte, pte = draw(n_test)
    return xtr, ytr, ptr, xte, yte, pte


def make_mixed_regime_d3(
    seed: int, n_train: int, n_test: int, dim: int, n_classes: int, separation: float
) -> tuple[torch.Tensor, ...]:
    from mop.diagnostics.hardness import make_graded_slot_task

    n_red_tr = n_train - int(n_train * PART_FRACS[2])
    n_red_te = n_test - int(n_test * PART_FRACS[2])
    task = make_graded_slot_task(n_red_tr + n_red_te, noise=2.8, seed=seed)
    xr, yr, hr = task.x, task.y, task.hard_mask.long()  # 0 easy bin, 1 hard bin (both reducible)

    def assemble(xr_s, yr_s, hr_s, n_noise: int, s: int):
        g = torch.Generator().manual_seed(s)
        xn = 0.5 * torch.randn(n_noise, task.dim, generator=g)
        yn = torch.randint(0, int(yr.max()) + 1, (n_noise,), generator=g)
        pn = torch.full((n_noise,), 2)
        x = torch.cat([xr_s, xn])
        y = torch.cat([yr_s, yn])
        p = torch.cat([hr_s, pn]).long()
        perm = torch.randperm(x.shape[0], generator=g)
        return x[perm], y[perm], p[perm]

    xtr, ytr, ptr = assemble(xr[:n_red_tr], yr[:n_red_tr], hr[:n_red_tr], n_train - n_red_tr, seed)
    xte, yte, pte = assemble(xr[n_red_tr:], yr[n_red_tr:], hr[n_red_tr:], n_test - n_red_te, seed + 1)
    return xtr, ytr, ptr, xte, yte, pte


def train_head(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, seed: int):
    g = torch.Generator().manual_seed(seed + 31)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, batch = x.shape[0], 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()


def run(
    seeds: list[int],
    dim: int = 192,
    n_train: int = 4000,
    n_test: int = 2000,
    n_classes: int = 6,
    separation: float = 0.35,
    ensemble: int = 5,
    hidden: int = 64,
    epochs: int = 25,
    expert_hidden: int = 256,
    expert_epochs: int = 40,
    budget_frac: float = 0.25,
    tv_kwargs: dict | None = None,
    pr1_path: str | Path = PR1_PATH,
    regime_builder=make_mixed_regime,
) -> dict:
    t0 = time.perf_counter()
    chance = 1.0 / n_classes
    pool_noise_frac = PART_FRACS[2]
    pr1 = read_pr1_context(pr1_path)
    rows, degenerate = [], []
    delta_auroc, delta_gate, dis_noise_shares = [], [], []
    for s in seeds:
        xtr, ytr, ptr, xte, yte, pte = regime_builder(s, n_train, n_test, dim, n_classes, separation)
        members = []
        for j in range(ensemble):
            torch.manual_seed(s * 100 + j)
            m = mlp(dim, n_classes, hidden=hidden, depth=1, ln=True)
            train_head(m, xtr, ytr, epochs, lr=1e-3, seed=s * 100 + j)
            members.append(m)
        with torch.no_grad():
            probs = torch.stack([F.softmax(m(xte), dim=-1) for m in members])  # [K, N, C]
        single = probs[0]
        pred_single = single.argmax(-1)
        err_single = (pred_single != yte).long()
        conf_single = single.max(-1).values
        disagreement = probs.var(0).mean(-1)  # the Ensemble.mean_and_disagreement convention
        mean_probs = probs.mean(0)
        err_mean = (mean_probs.argmax(-1) != yte).long()
        auroc_dis = auroc(disagreement, err_single)
        auroc_conf = auroc(1.0 - conf_single, err_single)
        auroc_dis_vs_mean = auroc(disagreement, err_mean)  # secondary variant, reported only
        red = pte < 2
        red_acc = float((pred_single[red] == yte[red]).float().mean())
        if red_acc > DEGENERATE_HIGH or red_acc < chance + DEGENERATE_LOW_MARGIN:
            degenerate.append({"seed": s, "reducible_acc": round(red_acc, 4)})
        torch.manual_seed(s * 100 + 77)
        expert = mlp(dim, n_classes, hidden=expert_hidden, depth=2, ln=True)
        train_head(expert, xtr, ytr, expert_epochs, lr=1e-3, seed=s * 100 + 77)
        with torch.no_grad():
            pred_expert = expert(xte).argmax(-1)
        b = int(budget_frac * n_test)
        gu = torch.Generator().manual_seed(s + 911)
        arm_scores = {
            "disagreement": disagreement,
            "confidence": 1.0 - conf_single,
            "uniform": torch.rand(n_test, generator=gu),
            "oracle": err_single.float(),
        }
        gate = {}
        for arm, score in arm_scores.items():
            idx = score.topk(b).indices
            pred = pred_single.clone()
            pred[idx] = pred_expert[idx]
            gate[arm] = {
                "acc": round(float((pred == yte).float().mean()), 4),
                "share_blobs": round(float((pte[idx] == 0).float().mean()), 4),
                "share_antipodal": round(float((pte[idx] == 1).float().mean()), 4),
                "share_noise": round(float((pte[idx] == 2).float().mean()), 4),
            }
        delta_auroc.append(auroc_dis - auroc_conf)
        delta_gate.append(gate["disagreement"]["acc"] - gate["uniform"]["acc"])
        dis_noise_shares.append(gate["disagreement"]["share_noise"])
        rows.append(
            {
                "seed": s,
                "single_head_acc": round(1.0 - float(err_single.float().mean()), 4),
                "reducible_acc": round(red_acc, 4),
                "auroc_disagreement": round(auroc_dis, 4),
                "auroc_confidence": round(auroc_conf, 4),
                "auroc_disagreement_vs_mean_errors": round(auroc_dis_vs_mean, 4),
                "ece_single_head": round(ece_equal_mass(conf_single, 1 - err_single), 4),
                "gate": gate,
            }
        )
        print(
            f"seed {s}: auroc dis={auroc_dis:.3f} conf={auroc_conf:.3f} "
            f"gate dis={gate['disagreement']['acc']:.3f} unif={gate['uniform']['acc']:.3f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    tv_kwargs = tv_kwargs or {}
    tv = noisy_tv_diagnostic(seed=seeds[0], **tv_kwargs)
    tv_bools = {
        k: bool(tv[k])
        for k in ("noise_error_stays_high", "epistemic_collapses_on_noise", "learning_progress_separates")
    }
    mean_dis_noise = sum(dis_noise_shares) / len(dis_noise_shares)
    chases_noise = mean_dis_noise > pool_noise_frac
    guard = {
        "noisy_tv": tv_bools,
        "noisy_tv_stats": {k: tv[k] for k in ("raw_error", "disagreement", "learning_progress")},
        "disagreement_gate_noise_share_mean": round(mean_dis_noise, 4),
        "pool_noise_frac": pool_noise_frac,
        "chases_noise": bool(chases_noise),
        "pass": bool(all(tv_bools.values()) and not chases_noise),
    }
    ci_auroc, ci_gate = seed_ci(delta_auroc), seed_ci(delta_gate)
    flips_auroc, flips_gate = sign_flip_report(delta_auroc), sign_flip_report(delta_gate)
    win_auroc = (
        ci_auroc["mean"] > max(ci_auroc["sd"], 0.0) and ci_auroc["mean"] > 0 and not flips_auroc["any_flip"]
    )
    win_gate = (
        ci_gate["mean"] > max(ci_gate["sd"], 0.0) and ci_gate["mean"] > 0 and not flips_gate["any_flip"]
    )
    if degenerate:
        verdict = "DEGENERATE: single-head reducible accuracy outside the calibrated band, recalibrate"
        null_supported = True
    elif not guard["pass"]:
        verdict = (
            "GUARD-FAIL: the noisy-TV contract or the noise-share check failed, disagreement chases "
            "irreducible noise, the e4 conflation stands, null supported"
        )
        null_supported = True
    elif win_auroc or win_gate:
        verdict = (
            "WIN: ensemble disagreement beats single-head confidence "
            f"(auroc win={win_auroc}, gated win={win_gate}) beyond the seed spread with no sign flip "
            "and the noisy-TV guard passes, epistemic disagreement is a usable compute-allocation signal"
        )
        null_supported = False
    else:
        verdict = (
            "NULL: disagreement ties single-head confidence and gating ties uniform at matched expert "
            "budget, the registry null holds"
        )
        null_supported = True
    return {
        "experiment": "mop_dr12_disagreement",
        "contract": CONTRACT,
        "config": {
            "seeds": seeds,
            "dim": dim,
            "n_train": n_train,
            "n_test": n_test,
            "n_classes": n_classes,
            "separation": separation,
            "ensemble": ensemble,
            "hidden": hidden,
            "epochs": epochs,
            "expert_hidden": expert_hidden,
            "expert_epochs": expert_epochs,
            "budget_frac": budget_frac,
            "part_fracs": list(PART_FRACS),
            "chance": chance,
        },
        "pr1_context": pr1,
        "per_seed": rows,
        "degenerate_seeds": degenerate,
        "delta_auroc": {**ci_auroc, "per_seed": [round(d, 4) for d in delta_auroc], "flips": flips_auroc},
        "delta_gate": {**ci_gate, "per_seed": [round(d, 4) for d in delta_gate], "flips": flips_gate},
        "guard": guard,
        "verdict_rule": (
            "WIN iff guard passes AND (delta_auroc or delta_gate has mean > seed SD, mean > 0, no "
            "sign flip); DEGENERATE iff reducible acc > 0.97 or < chance + 0.05 on any seed; guard "
            "failure overrides any primary win (thresholds fixed in code before running)"
        ),
        "verdict": verdict,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DR12: disagreement-as-uncertainty with noisy-TV guard")
    ap.add_argument("--seeds", default="0-4", help="seed spec, e.g. 0-4 or 0,2,5")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-classes", type=int, default=6)
    ap.add_argument("--separation", type=float, default=0.35)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--budget-frac", type=float, default=0.25)
    ap.add_argument("--pr1", default=PR1_PATH)
    ap.add_argument("--rerun", action="store_true", help="Stage 4 rerun, writes *_seeds10.json")
    ap.add_argument("--recal", action="store_true", help="A5: reducible partitions from D3 task")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.recal:
        result = run(
            parse_seeds(a.seeds),
            dim=64,
            n_train=a.n_train,
            n_test=a.n_test,
            n_classes=2,
            ensemble=a.ensemble,
            epochs=a.epochs,
            budget_frac=a.budget_frac,
            pr1_path=a.pr1,
            regime_builder=make_mixed_regime_d3,
        )
        result["recal"] = "D3 graded slot task (diagnostics/hardness.make_graded_slot_task)"
        out = a.out or "runs/mot/dr12_disagreement_recal.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps({k: result[k] for k in ("verdict", "null_supported", "seconds")}, indent=2))
        return 0
    result = run(
        parse_seeds(a.seeds),
        dim=a.dim,
        n_train=a.n_train,
        n_test=a.n_test,
        n_classes=a.n_classes,
        separation=a.separation,
        ensemble=a.ensemble,
        epochs=a.epochs,
        budget_frac=a.budget_frac,
        pr1_path=a.pr1,
    )
    out = a.out or f"runs/mot/dr12_disagreement{'_seeds10' if a.rerun else ''}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("verdict", "null_supported", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
