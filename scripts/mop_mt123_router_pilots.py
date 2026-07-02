#!/usr/bin/env python
"""MP1/MP2/MP3 router pilots (WP-08; registry docs/mixture_of_perspectives/11_experiment_registry.md,
MT full schema). Synthetic pilots today; the real answer stays blocked on DR1 (the manifest is
explicit), so this run scores accuracy-per-FLOP capability density only, the retention/byte axis is
the Studio's.

Three preregistered questions over ONE heterogeneous mode bank routed per episode:
  MP1: a per-episode router over {reactive readout, planner (refiner), sparse (kWTA) head} beats the
       single best mode on capability density (accuracy per FLOP) at matched compute
  MP2: ROUTING beats a uniform equal-weight blend of the same modes at matched TOTAL FLOPs (the
       matched blend shrinks the planner/sparse hidden widths so its per-sample cost equals the
       routed arm's, the honest matched-total-compute control; the full-cost blend is reported on
       density alongside)
  MP3: mode DIVERSITY is load-bearing: the heterogeneous bank beats k seed-copies of the best mode
       at matched params (capmatch, tol enforced) AND reported FLOPs, routed identically

PR1 reuse (the manifest rule, never rebuilt, never hardcoded): the calibrated difficulty
(separation), dim, and class count are READ from runs/pre_studio/pr1_mode_error_disjointness.json at
run time, and the task is PR1's preregistered three-subpopulation mixture imported from
scripts/pr1_mode_error_disjointness.py. The PR1 verdict gates interpretation: GREEN licenses these
as live tests; NULL demotes any win to PLAUSIBLE-BUT-UNVERIFIED reported against the PR1 context.

Episodes: each episode is `episode_size` samples drawn from ONE subpopulation of the mixture (the
natural per-episode structure the router must exploit); the router is a scalar-cheap selector (one
linear map on the episode mean latent, cheaper per sample than the cheapest mode) trained on a
router-train split disjoint from both the mode-train set and the eval episodes.

Mode bank note: the e7 sparse mode uses a local kWTA head defined in this script; WP-02 pass 2 owns
the shared shell/heads.py kWTA and this script should switch to it when it lands (deviation logged
in the lane report, the WP-02 stub API defers the heads extension).

Preregistered verdict logic per pilot (fixed in code before any run; manifest appendix):
  WIN iff the per-seed delta (routed minus baseline, on the pilot's preregistered metric) has
  mean > seed SD, mean > 0, and no sign flip, AND the recorded compute match holds on every seed
  (MP3: matched params AND FLOPs; MP2's matched blend is matched by construction; MP1's density
  metric embeds compute); a failed match reports UNMATCHED with null_supported=None (not evaluable),
  never a WIN and never a NULL; else NULL.
  MP1 metric: density delta vs the best single mode (selected on router-train, the tuned baseline).
  MP2 metric: accuracy delta vs the budget-matched blend when attainable (matched_within tol 0.10),
  else density delta vs the full blend (fallback logged).
  MP3 metric: density delta vs the param-matched homogeneous k-copy bank.
  A PR1 NULL context demotes every WIN string to PLAUSIBLE-BUT-UNVERIFIED (rows still run).

Usage: PYTHONPATH=. python scripts/mop_mt123_router_pilots.py --seeds 0-4 [--tuned]
Output: runs/mot/mt123_router_pilots.json (--tuned appends _tuned, --rerun appends _seeds10)

No em dashes or en dashes (BLACKHOLE.md). No encoder, no weights, latents are synthetic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scripts.mop_dr12_disagreement import parse_seeds, read_pr1_context
from scripts.pr1_mode_error_disjointness import SUBPOPS, RefinerHead, make_dataset
from torch import nn

from mop.diagnostics.compute import matched_within, mlp_flops, param_count, refiner_flops
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.shell.capmatch import width_for_param_count
from mop.shell.predictor import mlp

CONTRACT = {
    "id": "MP1/MP2/MP3",
    "metric": ("capability_density_acc_per_mflop", "accuracy_at_matched_total_flops"),
    "baseline": "best single mode (tuned on router-train); uniform blend; homogeneous k-copy bank",
    "ablation": "oracle per-episode router (upper bound), full-cost blend, per-mode accuracies",
    "null_hypothesis": (
        "MP1: routed density <= best single mode at matched compute; MP2: router <= uniform blend "
        "at matched total FLOPs; MP3: heterogeneous <= homogeneous k-copy at matched params and FLOPs"
    ),
    "tier": "cpu-now",
}

MODE_NAMES = ["reactive", "planner", "sparse"]
PR1_PATH = "runs/pre_studio/pr1_mode_error_disjointness.json"
BLEND_MATCH_TOL = 0.10
MIN_HIDDEN = 4
TUNE_GRID_LR = [1e-3, 3e-3, 1e-2]
TUNE_GRID_EPOCH_MULT = [1, 2]


class KWTAHead(nn.Module):
    """Local e7-style sparse mode: linear -> k-winners-take-all over hidden units -> linear. The
    shared shell/heads.py kWTA is a WP-02 pass 2 deliverable; switch to it when it lands."""

    def __init__(self, dim: int, hidden: int, n_classes: int, k_frac: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.k = max(1, int(round(k_frac * hidden)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        thresh = h.topk(self.k, dim=-1).values[..., -1:]
        return self.fc2(h * (h >= thresh).float())


def build_mode(name: str, dim: int, n_classes: int, hidden: int, steps: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if name == "reactive":
        return nn.Linear(dim, n_classes)
    if name == "planner":
        return RefinerHead(dim, n_classes, hidden=hidden, steps=steps)
    if name == "sparse":
        return KWTAHead(dim, hidden, n_classes)
    raise ValueError(f"unknown mode {name!r}")


def mode_flops(name: str, dim: int, n_classes: int, hidden: int, steps: int) -> int:
    """Per-sample forward FLOPs (kWTA masking and the refiner LayerNorm are rounding error, per the
    diagnostics/compute convention)."""
    if name == "reactive":
        return mlp_flops([dim, n_classes])
    if name == "planner":
        return refiner_flops(dim, hidden, steps) + mlp_flops([dim, n_classes])
    if name == "sparse":
        return mlp_flops([dim, hidden, n_classes])
    raise ValueError(f"unknown mode {name!r}")


def train_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, seed: int):
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


def fit_mode(
    name: str,
    build,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xval: torch.Tensor,
    yval: torch.Tensor,
    epochs: int,
    seed: int,
    tuned: bool,
) -> tuple[nn.Module, dict]:
    """Fit one mode; --tuned sweeps a small lr x epochs grid selected on the router-train split (the
    tuned-baseline doctrine), else uses the PR1-convention default lr per family."""
    default_lr = 1e-2 if name == "reactive" else 1e-3
    grid = (
        [(lr, epochs * m) for lr in TUNE_GRID_LR for m in TUNE_GRID_EPOCH_MULT]
        if tuned
        else [(default_lr, epochs)]
    )
    best, best_acc, best_cfg = None, -1.0, None
    for lr, ep in grid:
        model = build(seed)
        train_model(model, xtr, ytr, ep, lr, seed)
        with torch.no_grad():
            acc = float((model(xval).argmax(-1) == yval).float().mean())
        if acc > best_acc:
            best, best_acc, best_cfg = model, acc, {"lr": lr, "epochs": ep}
    return best, {"val_acc": round(best_acc, 4), **best_cfg, "tuned": tuned}


def make_episodes(
    sub: torch.Tensor, n_episodes: int, episode_size: int, seed: int
) -> list[tuple[torch.Tensor, int]]:
    """Episodes of `episode_size` indices, each drawn (with replacement) from ONE subpopulation,
    round-robin over subpopulations for balance."""
    g = torch.Generator().manual_seed(seed + 71)
    by_sub = [torch.nonzero(sub == k).flatten() for k in range(len(SUBPOPS))]
    episodes = []
    for e in range(n_episodes):
        pool = by_sub[e % len(SUBPOPS)]
        pick = pool[torch.randint(0, len(pool), (episode_size,), generator=g)]
        episodes.append((pick, e % len(SUBPOPS)))
    return episodes


def train_router(
    x: torch.Tensor,
    episodes: list[tuple[torch.Tensor, int]],
    correct: torch.Tensor,
    n_arms: int,
    seed: int,
    epochs: int = 200,
) -> nn.Module:
    """The scalar-cheap selector: one linear map on the episode mean latent, trained to predict the
    best arm per episode (label = argmax episode accuracy on the router-train split)."""
    feats = torch.stack([x[idx].mean(0) for idx, _ in episodes])
    labels = torch.stack([correct[:, idx].float().mean(1).argmax() for idx, _ in episodes])
    torch.manual_seed(seed + 13)
    router = nn.Linear(x.shape[1], n_arms)
    opt = torch.optim.Adam(router.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(router(feats), labels).backward()
        opt.step()
    return router


def routed_eval(
    x: torch.Tensor,
    y: torch.Tensor,
    episodes: list[tuple[torch.Tensor, int]],
    preds: torch.Tensor,
    router: nn.Module,
    arm_flops: list[int],
    router_flops: int,
    episode_size: int,
) -> dict:
    """Route each eval episode to one arm; also compute the per-episode oracle bound. Per-sample
    cost = selected arm cost + amortized router cost."""
    n_arms = preds.shape[0]
    total_correct, total, flops_sum, oracle_correct = 0.0, 0, 0.0, 0.0
    picks = []
    for idx, _ in episodes:
        with torch.no_grad():
            arm = int(router(x[idx].mean(0, keepdim=True)).argmax())
        picks.append(arm)
        ep_correct = (preds[:, idx] == y[idx]).float().mean(1)
        total_correct += float(ep_correct[arm]) * len(idx)
        oracle_correct += float(ep_correct.max()) * len(idx)
        total += len(idx)
        flops_sum += (arm_flops[arm] + router_flops / episode_size) * len(idx)
    counts = [picks.count(a) for a in range(n_arms)]
    return {
        "acc": total_correct / total,
        "oracle_acc": oracle_correct / total,
        "flops_per_sample": flops_sum / total,
        "selection_counts": counts,
    }


def density(acc: float, flops_per_sample: float) -> float:
    """Capability density: accuracy per MFLOP per sample (the MP1/MP2/MP3 pilot metric)."""
    return acc / (flops_per_sample / 1e6)


def verdict_block(
    deltas: list[float], metric: str, match: dict | None, pr1_green: bool, matched_ok: bool = True
) -> dict:
    """The preregistered WIN rule gated on the recorded compute match: a WIN can only fire when
    matched_ok is True (for MP3 that means params AND FLOPs matched on every seed, the manifest
    appendix wording). A failed match yields UNMATCHED with null_supported=None: the run is not
    evaluable at the preregistered matched compute, so neither the win nor the null branch fires."""
    ci, flips = seed_ci(deltas), sign_flip_report(deltas)
    cleared = ci["mean"] > max(ci["sd"], 0.0) and ci["mean"] > 0 and not flips["any_flip"]
    win = cleared and matched_ok
    if not matched_ok:
        return {
            "metric": metric,
            "delta": {**ci, "per_seed": [round(d, 6) for d in deltas], "flips": flips},
            "compute_match": match,
            "verdict": (
                f"UNMATCHED: the recorded compute match failed on at least one seed, so {metric} was "
                "not measured at the preregistered matched compute; no WIN and no NULL can fire"
            ),
            "null_supported": None,
        }
    if win and pr1_green:
        verdict = f"WIN: routed beats the baseline on {metric} beyond the seed spread with no sign flip"
    elif win:
        verdict = (
            f"PLAUSIBLE-BUT-UNVERIFIED: routed beats the baseline on {metric} beyond the seed spread, "
            "but the PR1 context is not GREEN, demoted pending the Studio regime (manifest appendix)"
        )
    else:
        verdict = f"NULL: routed does not beat the baseline on {metric} beyond the seed spread"
    return {
        "metric": metric,
        "delta": {**ci, "per_seed": [round(d, 6) for d in deltas], "flips": flips},
        "compute_match": match,
        "verdict": verdict,
        "null_supported": not win,
    }


def run(
    seeds: list[int],
    dim: int | None = None,
    n_classes: int | None = None,
    separation: float | None = None,
    n_train: int = 2000,
    n_test: int = 600,
    episode_size: int = 20,
    n_router_episodes: int = 90,
    n_eval_episodes: int = 90,
    epochs: int = 30,
    planner_hidden: int = 64,
    planner_steps: int = 3,
    sparse_hidden: int = 128,
    tuned: bool = False,
    cap_tol: float = 0.02,
    pr1_path: str | Path = PR1_PATH,
) -> dict:
    t0 = time.perf_counter()
    pr1 = read_pr1_context(pr1_path)
    cfg1 = pr1.get("config") or {}
    dim = dim or cfg1.get("dim") or 1024
    n_classes = n_classes or cfg1.get("n_classes") or 10
    separation = separation or pr1.get("chosen_separation") or cfg1.get("separation") or 0.12
    flops = [
        mode_flops(m, dim, n_classes, planner_hidden if m == "planner" else sparse_hidden, planner_steps)
        for m in MODE_NAMES
    ]
    router_flops = mlp_flops([dim, len(MODE_NAMES)])
    rows, d_mt1, d_mt2, d_mt3 = [], [], [], []
    mt2_fallbacks, mt2_matches, mt3_matches = 0, [], []
    for s in seeds:
        xtr, ytr, xte, yte, sub_te = make_dataset(s, n_train, n_test, n_classes, dim, separation)
        half = n_test // 2
        x_rt, y_rt, sub_rt = xte[:half], yte[:half], sub_te[:half]
        x_ev, y_ev, sub_ev = xte[half:], yte[half:], sub_te[half:]

        def builder(name, hid):
            return lambda seed_: build_mode(name, dim, n_classes, hid, planner_steps, seed_)

        modes, mode_meta = {}, {}
        for name in MODE_NAMES:
            hid = planner_hidden if name == "planner" else sparse_hidden
            model, meta = fit_mode(name, builder(name, hid), xtr, ytr, x_rt, y_rt, epochs, s, tuned)
            modes[name], mode_meta[name] = model, meta
        with torch.no_grad():
            preds_rt = torch.stack([modes[m](x_rt).argmax(-1) for m in MODE_NAMES])
            preds_ev = torch.stack([modes[m](x_ev).argmax(-1) for m in MODE_NAMES])
            probs_ev = torch.stack([F.softmax(modes[m](x_ev), dim=-1) for m in MODE_NAMES])
        ep_rt = make_episodes(sub_rt, n_router_episodes, episode_size, s)
        ep_ev = make_episodes(sub_ev, n_eval_episodes, episode_size, s + 500)
        correct_rt = (preds_rt == y_rt).long()
        router = train_router(x_rt, ep_rt, correct_rt, len(MODE_NAMES), s)
        routed = routed_eval(x_ev, y_ev, ep_ev, preds_ev, router, flops, router_flops, episode_size)

        # MP1: best single mode, SELECTED ON ROUTER-TRAIN (the tuned baseline), scored on eval episodes
        best_i = int(correct_rt.float().mean(1).argmax())
        ev_idx = torch.cat([idx for idx, _ in ep_ev])
        mode_ev_acc = [(preds_ev[i][ev_idx] == y_ev[ev_idx]).float().mean().item() for i in range(3)]
        best_density = density(mode_ev_acc[best_i], flops[best_i])
        routed_density = density(routed["acc"], routed["flops_per_sample"])
        d_mt1.append(routed_density - best_density)

        # MP2: uniform blend, full cost, plus the budget-matched blend at the routed arm's cost
        blend_pred = probs_ev.mean(0).argmax(-1)
        blend_acc = float((blend_pred[ev_idx] == y_ev[ev_idx]).float().mean())
        blend_flops = float(sum(flops))
        target = routed["flops_per_sample"]
        fixed = flops[0] + mlp_flops([dim, n_classes])  # reactive + planner head, width-independent
        # solve widths proportionally: keep the planner/sparse width ratio, total hidden units = scale;
        # cost(scale) = fixed + scale * (ratio * planner-per-unit + (1 - ratio) * sparse-per-unit)
        ratio = planner_hidden / (planner_hidden + sparse_hidden)
        var_per_unit = ratio * (4 * dim * planner_steps) + (1 - ratio) * 2 * (dim + n_classes)
        scale = (target - fixed) / var_per_unit
        hp = max(MIN_HIDDEN, int(round(scale * ratio))) if scale > 0 else 0
        hs = max(MIN_HIDDEN, int(round(scale * (1 - ratio)))) if scale > 0 else 0
        matched_blend = None
        if hp >= MIN_HIDDEN and hs >= MIN_HIDDEN:
            small_flops = (
                flops[0]
                + refiner_flops(dim, hp, planner_steps)
                + mlp_flops([dim, n_classes])
                + mlp_flops([dim, hs, n_classes])
            )
            match = matched_within(int(small_flops), int(target), tol=BLEND_MATCH_TOL)
            if match["matched"]:
                small = {
                    "reactive": modes["reactive"],
                    "planner": fit_mode(
                        "planner", builder("planner", hp), xtr, ytr, x_rt, y_rt, epochs, s, tuned
                    )[0],
                    "sparse": fit_mode(
                        "sparse", builder("sparse", hs), xtr, ytr, x_rt, y_rt, epochs, s, tuned
                    )[0],
                }
                with torch.no_grad():
                    sp = torch.stack([F.softmax(small[m](x_ev), dim=-1) for m in MODE_NAMES])
                m_acc = float((sp.mean(0).argmax(-1)[ev_idx] == y_ev[ev_idx]).float().mean())
                matched_blend = {
                    "acc": m_acc,
                    "flops_per_sample": small_flops,
                    "hp": hp,
                    "hs": hs,
                    "match": match,
                }
        if matched_blend is not None:
            d_mt2.append(routed["acc"] - matched_blend["acc"])
            mt2_matches.append(matched_blend["match"])
        else:
            d_mt2.append(routed_density - density(blend_acc, blend_flops))
            mt2_fallbacks += 1

        # MP3: homogeneous k-copy bank of the best mode at matched TOTAL params (capmatch)
        het_params = sum(param_count(modes[m]) for m in MODE_NAMES)
        best_name = MODE_NAMES[best_i]
        per_copy_target = het_params // len(MODE_NAMES)
        homo_note = ""
        if best_name == "reactive":
            # a linear head has no width knob; use depth-1 MLP copies at matched params, a strictly
            # at-least-as-capable homogeneous family (conservative against a false heterogeneous win)
            homo_note = "best mode is reactive (no width knob), homogeneous copies are depth-1 MLPs"

            def make_copy_arch(h):
                return mlp(dim, n_classes, hidden=h, depth=1, ln=False)

            copy_flops_of = lambda h: mlp_flops([dim, h, n_classes])  # noqa: E731
        elif best_name == "planner":

            def make_copy_arch(h):
                return RefinerHead(dim, n_classes, hidden=h, steps=planner_steps)

            copy_flops_of = lambda h: refiner_flops(dim, h, planner_steps) + mlp_flops([dim, n_classes])  # noqa: E731
        else:

            def make_copy_arch(h):
                return KWTAHead(dim, h, n_classes)

            copy_flops_of = lambda h: mlp_flops([dim, h, n_classes])  # noqa: E731
        width, copy_params = width_for_param_count(make_copy_arch, per_copy_target, tol=cap_tol)
        # each homogeneous copy goes through the SAME fit_mode path as the heterogeneous modes (same
        # tuned flag, same grid, selection on the router-train split), so the MP3 baseline is never
        # under-tuned relative to the routed arm; per-copy seeds stay decorrelated (s*1000+300+j)
        copies, copy_meta = [], []
        for j in range(len(MODE_NAMES)):
            cseed = s * 1000 + 300 + j

            def copy_builder(seed_, mk=make_copy_arch, w=width):
                torch.manual_seed(seed_)
                return mk(w)

            c, cmeta = fit_mode(best_name, copy_builder, xtr, ytr, x_rt, y_rt, epochs, cseed, tuned)
            copies.append(c)
            copy_meta.append(cmeta)
        with torch.no_grad():
            cp_rt = torch.stack([c(x_rt).argmax(-1) for c in copies])
            cp_ev = torch.stack([c(x_ev).argmax(-1) for c in copies])
        router_h = train_router(x_rt, ep_rt, (cp_rt == y_rt).long(), len(copies), s + 900)
        cflops = [copy_flops_of(width)] * len(copies)
        homo = routed_eval(x_ev, y_ev, ep_ev, cp_ev, router_h, cflops, router_flops, episode_size)
        homo_params = sum(param_count(c) for c in copies)
        mt3_match = {
            "params": matched_within(het_params, homo_params, tol=cap_tol * 2),
            "flops": matched_within(int(routed["flops_per_sample"]), int(homo["flops_per_sample"])),
            "copy_width": width,
            "copy_params": copy_params,
            "copy_fit": copy_meta,
            "note": homo_note,
        }
        mt3_matches.append(mt3_match)
        d_mt3.append(routed_density - density(homo["acc"], homo["flops_per_sample"]))

        rows.append(
            {
                "seed": s,
                "mode_meta": mode_meta,
                "mode_eval_acc": {m: round(a, 4) for m, a in zip(MODE_NAMES, mode_ev_acc, strict=True)},
                "best_mode": best_name,
                "routed": {k: round(v, 6) if isinstance(v, float) else v for k, v in routed.items()},
                "routed_density": round(routed_density, 6),
                "best_mode_density": round(best_density, 6),
                "blend_full": {"acc": round(blend_acc, 4), "flops_per_sample": blend_flops},
                "blend_matched": matched_blend,
                "homo": {k: round(v, 6) if isinstance(v, float) else v for k, v in homo.items()},
                "mt3_match": mt3_match,
            }
        )
        print(
            f"seed {s}: routed acc={routed['acc']:.3f} best={best_name} "
            f"({mode_ev_acc[best_i]:.3f}) blend={blend_acc:.3f} homo={homo['acc']:.3f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    mt2_metric = (
        "accuracy_vs_budget_matched_blend"
        if mt2_fallbacks == 0
        else f"density_vs_full_blend (matched blend unattainable on {mt2_fallbacks}/{len(seeds)} seeds)"
    )
    # MP3's appendix wording is "matched params AND FLOPs": both must hold on every seed or the
    # verdict is UNMATCHED. MP1 embeds compute in the density metric; MP2's matched blend is matched
    # by construction (only used when matched_within passes) and its density fallback is preregistered.
    mt3_matched_ok = all(m["params"]["matched"] and m["flops"]["matched"] for m in mt3_matches)
    return {
        "experiment": "mop_mt123_router_pilots",
        "contract": CONTRACT,
        "config": {
            "seeds": seeds,
            "dim": dim,
            "n_classes": n_classes,
            "separation": separation,
            "n_train": n_train,
            "n_test": n_test,
            "episode_size": episode_size,
            "n_router_episodes": n_router_episodes,
            "n_eval_episodes": n_eval_episodes,
            "epochs": epochs,
            "planner_hidden": planner_hidden,
            "planner_steps": planner_steps,
            "sparse_hidden": sparse_hidden,
            "tuned": tuned,
            "mode_flops_per_sample": {m: f for m, f in zip(MODE_NAMES, flops, strict=True)},
            "router_flops_per_episode": router_flops,
            "pilot_scope": "accuracy/FLOP density only; retention/byte is the DR1-scale Studio question",
        },
        "pr1_context": pr1,
        "pr1_gate": pr1["gate"],
        "per_seed": rows,
        "mt1_router_vs_best_mode": verdict_block(
            d_mt1,
            "capability_density_vs_best_single_mode",
            {"note": "matched compute is embedded in the density metric (accuracy per MFLOP per sample)"},
            pr1["green"],
        ),
        "mt2_router_vs_uniform_blend": verdict_block(
            d_mt2, mt2_metric, {"per_seed_matches": mt2_matches}, pr1["green"]
        ),
        "mt3_hetero_vs_homogeneous": verdict_block(
            d_mt3,
            "capability_density_vs_param_matched_k_copy_bank",
            {"per_seed": mt3_matches, "all_seeds_matched": mt3_matched_ok},
            pr1["green"],
            matched_ok=mt3_matched_ok,
        ),
        "verdict_rule": (
            "per pilot: WIN iff per-seed delta mean > seed SD, mean > 0, no sign flip, AND the "
            "recorded compute match holds on every seed (MP3: params and FLOPs; a failed match is "
            "UNMATCHED with null_supported=None, never a win or a null); PR1 NULL context demotes "
            "wins to PLAUSIBLE-BUT-UNVERIFIED; thresholds fixed in code before running"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MP1/MP2/MP3 router pilots (PR1-gated, synthetic)")
    ap.add_argument("--seeds", default="0-4", help="seed spec, e.g. 0-4 or 0,2,5")
    ap.add_argument("--dim", type=int, default=None, help="default: read from the PR1 json")
    ap.add_argument("--n-classes", type=int, default=None, help="default: read from the PR1 json")
    ap.add_argument("--separation", type=float, default=None, help="default: PR1 calibrated value")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--episode-size", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tuned", action="store_true", help="Stage 4 re-tuned single-mode baselines")
    ap.add_argument("--rerun", action="store_true", help="Stage 4 rerun, writes *_seeds10.json")
    ap.add_argument("--pr1", default=PR1_PATH)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = run(
        parse_seeds(a.seeds),
        dim=a.dim,
        n_classes=a.n_classes,
        separation=a.separation,
        n_train=a.n_train,
        n_test=a.n_test,
        episode_size=a.episode_size,
        epochs=a.epochs,
        tuned=a.tuned,
        pr1_path=a.pr1,
    )
    suffix = ("_tuned" if a.tuned else "") + ("_seeds10" if a.rerun else "")
    out = a.out or f"runs/mot/mt123_router_pilots{suffix}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, default=str))
    summary = {
        "pr1_gate": result["pr1_gate"],
        "mt1": result["mt1_router_vs_best_mode"]["verdict"],
        "mt2": result["mt2_router_vs_uniform_blend"]["verdict"],
        "mt3": result["mt3_hetero_vs_homogeneous"]["verdict"],
        "seconds": result["seconds"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
