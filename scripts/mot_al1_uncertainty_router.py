#!/usr/bin/env python
"""AL1: uncertainty router with noisy-TV guard (WP-08, H-ENSEMBLE consumer; registry
docs/mixture_of_thinking/11_experiment_registry.md, AL full schema).

Thesis: an uncertainty signal used ONLY as a ROUTER input (explicitly never a learning-rate gate,
the e4 corpus negative refuted that use 30/30) selects which episodes get the update budget and
beats random episode selection on adaptation-per-update, while passing the noisy-TV guard.

Regime (synthetic latents, no encoder, preregistered): a STREAM of candidate episodes with limited
lookahead (a sliding window of unconsumed episodes). Learnable episodes carry samples from a fixed
cluster task; noise episodes carry unlearnable content (random inputs, uniform random labels, fresh
every draw). Every arm gets the SAME update budget and the SAME fixed learning rate (uncertainty is
admitted only as a selection input); arms differ ONLY in which episode of the window they select:
  disagreement    : ensemble disagreement on the episode (epistemic, the tested arm)
  random          : uniform choice (the matched baseline)
  point_error     : current loss on the episode (the noisy-TV bait arm, expected to chase noise)
  disagreement_permuted : the tested arm on a PERMUTED stream order (the curriculum-permutation
                          control: the advantage must not be an ordering artifact)
All arms share the member initialization per seed, so selection policy is the only difference.

Metric: adaptation-per-update, scored as held-out learnable accuracy at the matched update budget
(equal updates for every arm makes final accuracy the per-update comparison); the accuracy curve and
adaptation_speed (diagnostics/continual_metrics) are reported alongside. Router scoring overhead
(no_grad forward passes on window episodes) is reported honestly; the preregistered match is on
gradient updates, the quantity the metric normalizes by.

Guard: the three-boolean contract from diagnostics/noisy_tv.py must pass jointly, AND the
disagreement router's selected-noise share must not exceed the pool noise fraction (selecting noise
above base rate = chasing aleatoric noise = e4 in router clothing). A guard failure overrides any
primary win (manifest appendix).

Preregistered verdict logic (fixed in code before any run):
  DEGENERATE if the random arm's final accuracy > 0.97 or < chance + 0.05 on any seed
  GUARD-FAIL if the noisy-TV booleans do not all pass or the disagreement router chases noise
  WIN  if guard passes AND delta(disagreement - random) has mean > seed SD, mean > 0, no sign flip,
       AND the permuted-stream disagreement arm also beats random on mean (curriculum control)
  NULL otherwise (the registry null: router matches random episode selection)

Usage: PYTHONPATH=. python scripts/mot_al1_uncertainty_router.py --seeds 0-4
Output: runs/mot/al1_uncertainty_router.json (--rerun writes al1_uncertainty_router_seeds10.json)

No em dashes or en dashes (BLACKHOLE.md). No encoder, no weights, latents are synthetic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scripts.mot_dr12_disagreement import parse_seeds

from devsys.diagnostics.continual_metrics import adaptation_speed
from devsys.diagnostics.noisy_tv import noisy_tv_diagnostic
from devsys.diagnostics.riskcov import seed_ci, sign_flip_report
from devsys.shell.predictor import mlp

CONTRACT = {
    "id": "AL1",
    "metric": ("heldout_accuracy_at_matched_update_budget", "noise_selection_share"),
    "baseline": "random episode selection at the same update budget and learning rate",
    "ablation": "point-error router (noisy-TV bait), permuted-stream disagreement (curriculum control)",
    "null_hypothesis": (
        "The uncertainty router matches random episode selection on adaptation-per-update, OR it "
        "chases irreducible aleatoric noise (fails the noisy-TV guard), replicating the e4 negative "
        "in router clothing"
    ),
    "tier": "cpu-now",
}

ARMS = ["disagreement", "random", "point_error", "disagreement_permuted"]
DEGENERATE_HIGH = 0.97
DEGENERATE_LOW_MARGIN = 0.05


def build_pool(
    seed: int,
    n_pool: int,
    episode_size: int,
    dim: int,
    n_classes: int,
    noise_frac: float,
    separation: float,
    n_eval: int,
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    """The episode pool plus a held-out learnable eval set. Learnable episodes draw from fixed
    cluster centers; noise episodes are random inputs with uniform random labels (irreducible)."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g)

    def learnable(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, n_classes, (n,), generator=g)
        return centers[y] * separation + 0.5 * torch.randn(n, dim, generator=g), y

    pool: list[dict] = []
    n_noise = int(round(n_pool * noise_frac))
    for i in range(n_pool):
        if i < n_noise:
            x = 0.5 * torch.randn(episode_size, dim, generator=g)
            y = torch.randint(0, n_classes, (episode_size,), generator=g)
            pool.append({"x": x, "y": y, "noise": True})
        else:
            x, y = learnable(episode_size)
            pool.append({"x": x, "y": y, "noise": False})
    order = torch.randperm(n_pool, generator=g).tolist()
    pool = [pool[i] for i in order]
    x_eval, y_eval = learnable(n_eval)
    return pool, x_eval, y_eval


def run_arm(
    arm: str,
    pool: list[dict],
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    seed: int,
    dim: int,
    n_classes: int,
    members: int,
    hidden: int,
    budget: int,
    window: int,
    updates_per_episode: int,
    lr: float,
    eval_every: int,
) -> dict:
    """One selection policy at the matched update budget. All arms rebuild the SAME initial members
    (same seed) so policy is the only difference; the LR is fixed (never gated by uncertainty)."""
    stream = list(range(len(pool)))
    if arm == "disagreement_permuted":
        gp = torch.Generator().manual_seed(seed + 4242)
        stream = [stream[i] for i in torch.randperm(len(stream), generator=gp).tolist()]
    torch.manual_seed(seed * 100 + 7)
    heads = [mlp(dim, n_classes, hidden=hidden, depth=1, ln=True) for _ in range(members)]
    opt = torch.optim.Adam([p for h in heads for p in h.parameters()], lr=lr)
    gr = torch.Generator().manual_seed(seed + 17)
    used: set[int] = set()
    noise_picks, acc_curve = 0, []
    router_overhead_fwd = 0

    @torch.no_grad()
    def probs(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([F.softmax(h(x), dim=-1) for h in heads])

    @torch.no_grad()
    def eval_acc() -> float:
        return float((probs(x_eval).mean(0).argmax(-1) == y_eval).float().mean())

    for step in range(budget):
        candidates = [i for i in stream if i not in used][:window]
        if not candidates:
            break
        if arm == "random":
            pick = candidates[int(torch.randint(0, len(candidates), (1,), generator=gr))]
        else:
            scores = []
            for i in candidates:
                p = probs(pool[i]["x"])
                router_overhead_fwd += len(heads)
                if arm == "point_error":
                    scores.append(float(F.nll_loss(p.mean(0).clamp_min(1e-8).log(), pool[i]["y"])))
                else:
                    scores.append(float(p.var(0).mean()))
            pick = candidates[max(range(len(candidates)), key=lambda k: scores[k])]
        used.add(pick)
        noise_picks += int(pool[pick]["noise"])
        # fresh labels for noise episodes on every visit (irreducible by construction)
        ep = pool[pick]
        y_ep = torch.randint(0, n_classes, ep["y"].shape, generator=gr) if ep["noise"] else ep["y"]
        for _ in range(updates_per_episode):
            opt.zero_grad()
            loss = torch.stack([F.cross_entropy(h(ep["x"]), y_ep) for h in heads]).sum()
            loss.backward()
            opt.step()
        if (step + 1) % eval_every == 0 or step + 1 == budget:
            acc_curve.append(round(eval_acc(), 4))
    return {
        "final_acc": acc_curve[-1] if acc_curve else 0.0,
        "acc_curve": acc_curve,
        "adaptation_speed": adaptation_speed(acc_curve, target_frac=0.9),
        "noise_share": round(noise_picks / max(1, len(used)), 4),
        "episodes_used": len(used),
        "router_overhead_forwards": router_overhead_fwd,
    }


def build_pool_d3(
    seed: int,
    n_pool: int,
    episode_size: int,
    dim: int,
    n_classes: int,
    noise_frac: float,
    separation: float,
    n_eval: int,
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    """A5 recal pool: learnable episodes and the held-out eval set are drawn from the D3 graded slot
    task (diagnostics/hardness), whose binary DSL label sits at a CALIBRATED, non-ceiling difficulty
    (~0.85 best-mode) so the random arm cannot saturate to 1.0 (the degenerate the base al1 hit). Noise
    episodes stay irreducible (random inputs, uniform random labels, fresh every visit). dim/n_classes
    are overridden to the D3 task's own (64, 2); the passed separation is ignored (D3 sets difficulty)."""
    from devsys.diagnostics.hardness import make_graded_slot_task

    n_noise = int(round(n_pool * noise_frac))
    n_learn_ep = n_pool - n_noise
    task = make_graded_slot_task(n_learn_ep * episode_size + n_eval, seed=seed)
    tx, ty = task.x, task.y
    g = torch.Generator().manual_seed(seed + 3)
    pool: list[dict] = []
    cur = 0
    for i in range(n_pool):
        if i < n_noise:
            x = 0.5 * torch.randn(episode_size, dim, generator=g)
            y = torch.randint(0, n_classes, (episode_size,), generator=g)
            pool.append({"x": x, "y": y, "noise": True})
        else:
            x = tx[cur : cur + episode_size]
            y = ty[cur : cur + episode_size]
            cur += episode_size
            pool.append({"x": x, "y": y, "noise": False})
    order = torch.randperm(n_pool, generator=g).tolist()
    pool = [pool[i] for i in order]
    x_eval, y_eval = tx[cur : cur + n_eval], ty[cur : cur + n_eval]
    return pool, x_eval, y_eval


def run(
    seeds: list[int],
    dim: int = 64,
    n_classes: int = 5,
    separation: float = 0.5,
    n_pool: int = 120,
    episode_size: int = 24,
    noise_frac: float = 0.4,
    budget: int = 48,
    window: int = 12,
    members: int = 4,
    hidden: int = 32,
    updates_per_episode: int = 2,
    lr: float = 5e-3,
    n_eval: int = 800,
    eval_every: int = 4,
    tv_kwargs: dict | None = None,
    pool_builder=build_pool,
) -> dict:
    t0 = time.perf_counter()
    chance = 1.0 / n_classes
    rows, degenerate = [], []
    deltas, perm_deltas, dis_noise_shares = [], [], []
    for s in seeds:
        pool, x_eval, y_eval = pool_builder(
            s, n_pool, episode_size, dim, n_classes, noise_frac, separation, n_eval
        )
        arms = {
            arm: run_arm(
                arm,
                pool,
                x_eval,
                y_eval,
                s,
                dim,
                n_classes,
                members,
                hidden,
                budget,
                window,
                updates_per_episode,
                lr,
                eval_every,
            )
            for arm in ARMS
        }
        rand_acc = arms["random"]["final_acc"]
        if rand_acc > DEGENERATE_HIGH or rand_acc < chance + DEGENERATE_LOW_MARGIN:
            degenerate.append({"seed": s, "random_final_acc": rand_acc})
        deltas.append(arms["disagreement"]["final_acc"] - rand_acc)
        perm_deltas.append(arms["disagreement_permuted"]["final_acc"] - rand_acc)
        dis_noise_shares.append(arms["disagreement"]["noise_share"])
        rows.append({"seed": s, "arms": arms})
        print(
            f"seed {s}: dis={arms['disagreement']['final_acc']:.3f} rand={rand_acc:.3f} "
            f"bait={arms['point_error']['final_acc']:.3f} "
            f"noise_share dis={arms['disagreement']['noise_share']:.2f} "
            f"bait={arms['point_error']['noise_share']:.2f} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    tv_kwargs = tv_kwargs or {}
    tv = noisy_tv_diagnostic(seed=seeds[0], **tv_kwargs)
    tv_bools = {
        k: bool(tv[k])
        for k in ("noise_error_stays_high", "epistemic_collapses_on_noise", "learning_progress_separates")
    }
    mean_dis_noise = sum(dis_noise_shares) / len(dis_noise_shares)
    chases_noise = mean_dis_noise > noise_frac
    guard = {
        "noisy_tv": tv_bools,
        "disagreement_noise_share_mean": round(mean_dis_noise, 4),
        "pool_noise_frac": noise_frac,
        "chases_noise": bool(chases_noise),
        "pass": bool(all(tv_bools.values()) and not chases_noise),
    }
    ci, flips = seed_ci(deltas), sign_flip_report(deltas)
    ci_perm = seed_ci(perm_deltas)
    win = (
        guard["pass"]
        and ci["mean"] > max(ci["sd"], 0.0)
        and ci["mean"] > 0
        and not flips["any_flip"]
        and ci_perm["mean"] > 0
    )
    if degenerate:
        verdict = "DEGENERATE: random-arm final accuracy outside the calibrated band, recalibrate"
        null_supported = True
    elif not guard["pass"]:
        verdict = (
            "GUARD-FAIL: noisy-TV contract or noise-share check failed, the router chases irreducible "
            "noise, e4 in router clothing, null supported"
        )
        null_supported = True
    elif win:
        verdict = (
            "WIN: the disagreement router beats random episode selection beyond the seed spread with "
            "no sign flip, survives the curriculum permutation, and passes the noisy-TV guard; "
            "uncertainty works as a router input where it failed as an LR gate"
        )
        null_supported = False
    else:
        verdict = (
            "NULL: the router ties random episode selection on adaptation-per-update, the registry null holds"
        )
        null_supported = True
    return {
        "experiment": "mot_al1_uncertainty_router",
        "contract": CONTRACT,
        "config": {
            "seeds": seeds,
            "dim": dim,
            "n_classes": n_classes,
            "separation": separation,
            "n_pool": n_pool,
            "episode_size": episode_size,
            "noise_frac": noise_frac,
            "budget": budget,
            "window": window,
            "members": members,
            "hidden": hidden,
            "updates_per_episode": updates_per_episode,
            "lr": lr,
            "lr_rule": "fixed for every arm, uncertainty is never an LR gate (the e4 doctrine)",
            "n_eval": n_eval,
            "chance": chance,
        },
        "per_seed": rows,
        "degenerate_seeds": degenerate,
        "delta_dis_minus_random": {**ci, "per_seed": [round(d, 4) for d in deltas], "flips": flips},
        "delta_permuted_minus_random": {**ci_perm, "per_seed": [round(d, 4) for d in perm_deltas]},
        "guard": guard,
        "verdict_rule": (
            "WIN iff guard passes AND delta(disagreement - random) mean > seed SD, mean > 0, no sign "
            "flip, AND permuted-stream delta mean > 0; guard failure overrides any primary win "
            "(thresholds fixed in code before running)"
        ),
        "verdict": verdict,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AL1: uncertainty router with noisy-TV guard")
    ap.add_argument("--seeds", default="0-4", help="seed spec, e.g. 0-4 or 0,2,5")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-classes", type=int, default=5)
    ap.add_argument("--separation", type=float, default=0.5)
    ap.add_argument("--n-pool", type=int, default=120)
    ap.add_argument("--episode-size", type=int, default=24)
    ap.add_argument("--noise-frac", type=float, default=0.4)
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--rerun", action="store_true", help="Stage 4 rerun, writes *_seeds10.json")
    ap.add_argument("--recal", action="store_true", help="A5: draw learnable episodes from D3 task")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.recal:
        # D3 task is binary and 64-dim; learnable content sits at a calibrated non-ceiling difficulty
        result = run(
            parse_seeds(a.seeds),
            dim=64,
            n_classes=2,
            n_pool=a.n_pool,
            episode_size=a.episode_size,
            noise_frac=a.noise_frac,
            budget=a.budget,
            pool_builder=build_pool_d3,
        )
        result["recal"] = "D3 graded slot task (diagnostics/hardness.make_graded_slot_task)"
        out = a.out or "runs/mot/al1_uncertainty_router_recal.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps({k: result[k] for k in ("verdict", "null_supported", "seconds")}, indent=2))
        return 0
    result = run(
        parse_seeds(a.seeds),
        dim=a.dim,
        n_classes=a.n_classes,
        separation=a.separation,
        n_pool=a.n_pool,
        episode_size=a.episode_size,
        noise_frac=a.noise_frac,
        budget=a.budget,
    )
    out = a.out or f"runs/mot/al1_uncertainty_router{'_seeds10' if a.rerun else ''}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("verdict", "null_supported", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
