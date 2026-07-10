"""Close the e7_sparse contention: is the sparse-beats-dense BWT advantage SUBSTRATE-SPECIFIC
(a property of the real synthetic domain-incremental geometry) or a GENERIC ARCHITECTURAL fact
that holds on any input geometry?

e7_sparse is the corpus's one provisional positive: sparse / gated heads (kWTA top-k, MoE) roughly
halve catastrophic forgetting (backward transfer, BWT) versus a parameter-matched dense head on a
domain-incremental latent stream. It survived a 30-run seed/axis sweep but never ran the standing
D2 control: the FROZEN-RANDOM-SUBSTRATE ablation (mop.diagnostics.substrate_ablation.
frozen_random_projection). Without it we cannot tell whether the advantage needs the real substrate
geometry or would appear under any fixed projection of the inputs.

The decisive comparison (this script): run the SAME dense vs kWTA vs MoE BWT comparison TWICE per
seed, matched compute / optimizer / stream:
  (1) REAL: the normal synthetic domain-incremental stream (make_task_stream), and
  (2) FROZEN-RANDOM: a fixed random linear projection (frozen_random_projection) applied to every
      task's x before any splitting or training (the 'would any projection do' baseline).
For each substrate we take the sparse-minus-dense BWT gain (best sparse arm BWT minus dense BWT),
averaged over multiple seeds, then compare the two gains.

Verdict logic (doctrine, honest, never tuned toward a positive):
  * gain_frozen_random roughly equal to gain_real (ratio near 1)  => GENERIC ARCHITECTURAL effect:
    sparse heads forget less regardless of substrate. Still real, but REFRAMED (not substrate
    specific).
  * gain_frozen_random much smaller or gone (ratio near 0, or the frozen-random gain no longer clears
    the null margin) => SUBSTRATE-SPECIFIC (the strongest possible result: the advantage needed the
    real geometry). This would let e7_sparse SURVIVE as a substrate-specific positive.

We reuse e7_sparse's own head builders (DenseHead, KWTAHead, MoEHead), its matched-expert-width
helper (E7._matched_expert_hidden), its 80/20 split (_split), and reproduce its exact per-arm
continual-training loop and ContinualResult.backward_transfer() BWT computation. We do NOT edit the
shipped module.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only); no sentience
language.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from mop.diagnostics.substrate_ablation import frozen_random_projection
from mop.experiments.e7_sparse import E7, DenseHead, KWTAHead, MoEHead, _split
from mop.metrics import ContinualResult
from mop.seeding import seed_everything
from mop.substrate.datasets import Task, make_task_stream

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "runs" / "pre_studio" / "close_e7_sparse.json"
OUT = DEFAULT_OUT

# Stream / head config, matched to configs/experiment/e7_sparse.yaml. Kept identical so the BWT
# numbers here are the same regime the corpus positive was measured in (only the substrate variant
# and the multi-seed loop are new).
CFG = dict(
    n_tasks=5,
    dim=128,
    classes_per_task=6,
    samples_per_task=256,
    separation=0.5,
    hidden=64,
    kwta_k=8,
    n_experts=4,
    epochs_per_task=60,
    lr=0.01,
    margin=0.02,  # e7_sparse's null margin: a sparse arm must beat dense BWT by this to count
)
SEEDS = [0, 1, 2, 3, 4]


def _apply_frozen_random(tasks: list[Task], seed: int) -> list[Task]:
    """Replace every task's x with a FIXED random linear projection of it (same projection W for the
    whole stream, so the cross-domain interference structure is a rotated / mixed view of the real
    one). This is the D2 'any projection' control at the input level. Labels and geometry-per-task
    ordering are untouched, so compute and the continual protocol are identical to the real arm."""
    return [
        Task(
            t.name,
            frozen_random_projection(t.x, seed=seed),
            t.y,
            xnext=t.xnext,
            n_classes=t.n_classes,
            task_id=t.task_id,
        )
        for t in tasks
    ]


def _build_arms(dim: int, n_classes: int) -> dict:
    """The three e7_sparse head builders at matched parameter count, using E7's own expert-width
    solver so the MoE stays param matched to the dense head (sparsity / routing is the only change)."""
    e7 = E7()
    hidden = CFG["hidden"]
    expert_hidden = e7._matched_expert_hidden(dim, hidden, n_classes, CFG["n_experts"])
    return {
        "dense": lambda: DenseHead(dim, hidden, n_classes),
        "kwta": lambda: KWTAHead(dim, hidden, n_classes, CFG["kwta_k"]),
        "moe": lambda: MoEHead(dim, n_classes, CFG["n_experts"], expert_hidden),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, test: list[Task]) -> list[float]:
    model.eval()
    accs = []
    for task in test:
        logits = model(task.x)
        accs.append(float((logits.argmax(-1) == task.y).float().mean()))
    return accs


def _run_arm(build, train: list[Task], test: list[Task], chance: float, seed: int) -> dict:
    """Reproduce E7._run_arm exactly: reseed, full-batch Adam, epochs_per_task steps per task,
    evaluate the whole test set after each task, then BWT from ContinualResult. CPU, tiny MLP."""
    seed_everything(seed)
    model = build()
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    R: list[list[float]] = []
    t0 = time.perf_counter()
    for task in train:
        model.train()
        for _ in range(CFG["epochs_per_task"]):
            opt.zero_grad()
            loss = loss_fn(model(task.x), task.y)
            loss.backward()
            opt.step()
        R.append(_evaluate(model, test))
    wall = time.perf_counter() - t0
    cr = ContinualResult(R=R, chance=chance)
    return {"bwt": cr.backward_transfer(), "n_params": n_params, "wall_clock_s": wall}


def _run_substrate(tasks: list[Task], dim: int, n_classes: int, chance: float, seed: int) -> dict:
    """Run all three arms on one substrate for one seed and return per-arm BWT plus the
    best-sparse-minus-dense gain (the e7_sparse effect size)."""
    train = [_split(t)[0] for t in tasks]
    test = [_split(t)[1] for t in tasks]
    arms = {name: _run_arm(b, train, test, chance, seed) for name, b in _build_arms(dim, n_classes).items()}
    dense_bwt = arms["dense"]["bwt"]
    gains = {n: arms[n]["bwt"] - dense_bwt for n in ("kwta", "moe")}
    best_sparse = max(gains, key=lambda n: gains[n])
    return {
        "bwt": {n: arms[n]["bwt"] for n in arms},
        "n_params": {n: arms[n]["n_params"] for n in arms},
        "best_sparse": best_sparse,
        "sparse_minus_dense_gain": gains[best_sparse],
        "gain_kwta": gains["kwta"],
        "gain_moe": gains["moe"],
    }


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
    dim = CFG["dim"]
    per_seed: list[dict] = []
    for seed in SEEDS:
        # One stream per seed; the frozen-random arm uses a projection of THAT SAME stream, so the
        # only difference between the two substrates is the fixed linear map (matched everything else).
        seed_everything(seed)
        raw = make_task_stream(
            n_tasks=CFG["n_tasks"],
            dim=dim,
            classes_per_task=CFG["classes_per_task"],
            samples_per_task=CFG["samples_per_task"],
            separation=CFG["separation"],
            incremental="domain",
            seed=seed,
        )
        n_classes = raw[0].n_classes
        chance = 1.0 / n_classes
        fr = _apply_frozen_random(raw, seed=seed)

        real = _run_substrate(raw, dim, n_classes, chance, seed)
        rand = _run_substrate(fr, dim, n_classes, chance, seed)
        per_seed.append({"seed": seed, "real": real, "frozen_random": rand})
        print(
            f"seed {seed}: real gain={real['sparse_minus_dense_gain']:+.4f} "
            f"(best={real['best_sparse']}, dense_bwt={real['bwt']['dense']:+.4f}) | "
            f"frozen_random gain={rand['sparse_minus_dense_gain']:+.4f} "
            f"(best={rand['best_sparse']}, dense_bwt={rand['bwt']['dense']:+.4f})",
            flush=True,
        )

    n = len(per_seed)
    mean_real = sum(s["real"]["sparse_minus_dense_gain"] for s in per_seed) / n
    mean_rand = sum(s["frozen_random"]["sparse_minus_dense_gain"] for s in per_seed) / n
    mean_dense_bwt_real = sum(s["real"]["bwt"]["dense"] for s in per_seed) / n
    mean_dense_bwt_rand = sum(s["frozen_random"]["bwt"]["dense"] for s in per_seed) / n
    # ratio of the frozen-random advantage to the real advantage. Near 1 => generic; near 0 (or
    # negative / below the margin) => substrate specific.
    ratio = (mean_rand / mean_real) if abs(mean_real) > 1e-9 else float("nan")

    margin = CFG["margin"]
    real_clears = mean_real > margin
    rand_clears = mean_rand > margin

    # Honest verdict. The corpus positive requires the real gain to clear the null margin; if it does
    # not at this scale we say so plainly rather than tuning.
    if not real_clears:
        resolution = "inconclusive"
        verdict = (
            f"Real sparse-minus-dense BWT gain ({mean_real:+.4f}) does not clear the e7_sparse null "
            f"margin ({margin}) at this scale, so the substrate-specificity question cannot be settled "
            f"here (frozen-random gain {mean_rand:+.4f})."
        )
    elif rand_clears and ratio >= 0.5:
        resolution = "reframed"
        verdict = (
            f"GENERIC ARCHITECTURAL effect: sparse heads forget less regardless of substrate. The "
            f"sparse-minus-dense BWT gain survives frozen-random projection nearly intact "
            f"({mean_rand:+.4f} vs real {mean_real:+.4f}, ratio {ratio:.2f}), so e7_sparse is real but "
            f"NOT substrate specific (reframed as an architectural fact, not a V-JEPA-geometry claim)."
        )
    else:
        resolution = "survives"
        verdict = (
            f"SUBSTRATE-SPECIFIC: the sparse-minus-dense BWT advantage collapses under frozen-random "
            f"projection ({mean_rand:+.4f} vs real {mean_real:+.4f}, ratio {ratio:.2f}"
            + (", below the null margin" if not rand_clears else "")
            + "), so the advantage needed the real stream geometry. e7_sparse survives as a "
            "substrate-specific positive."
        )

    out = {
        "id": "e7_sparse",
        "question": (
            "Is the sparse-beats-dense BWT advantage substrate-specific (needs the real domain-"
            "incremental geometry) or a generic architectural fact (holds under any fixed projection)?"
        ),
        "control": "frozen_random_projection applied to every task's x before training (D2 'any projection')",
        "config": CFG,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "mean_gain_real": round(mean_real, 5),
        "mean_gain_frozen_random": round(mean_rand, 5),
        "ratio_frozen_over_real": None if ratio != ratio else round(ratio, 4),
        "mean_dense_bwt_real": round(mean_dense_bwt_real, 5),
        "mean_dense_bwt_frozen_random": round(mean_dense_bwt_rand, 5),
        "null_margin": margin,
        "real_gain_clears_margin": bool(real_clears),
        "frozen_random_gain_clears_margin": bool(rand_clears),
        "resolution": resolution,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("\n" + verdict)
    print(f"resolution={resolution}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
