
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from mop.diagnostics.substrate_ablation import frozen_random_projection, rank_reduced_projection
from mop.experiments.b_biology import B5, _domain_stream, _mean, _spread
from mop.seeding import seed_everything
from mop.substrate.datasets import Task

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "pre_studio" / "close_b5_degeneracy.json"

CFG = dict(
    dim=48,
    n_classes=4,
    n_tasks=3,
    samples=200,
    separation=2.0,
    degenerate_widths=[32, 24, 40],
    epochs=60,
    lr=0.02,
)
SEEDS = [0, 1, 2, 3, 4]  # config default is [0, 1]; widened here for a real seed spread / margin
RATIO_SURVIVE = 0.5  # preregistered: projected/real gain at or above this counts as surviving


def _project_stream(tasks: list[Task], fn: Callable[[torch.Tensor], torch.Tensor]) -> list[Task]:
    return [Task(t.name, fn(t.x), t.y, t.xnext, t.n_classes, t.task_id) for t in tasks]


def _retention_arm(build: Callable[[], nn.ModuleList], stream: list[Task]) -> float:
    model = build()
    for t in stream:
        B5._train_vote(model, t.x, t.y, CFG["epochs"], CFG["lr"])
    return B5._vote_acc(model, stream[0].x, stream[0].y)


def _run_substrate(stream: list[Task]) -> dict:
    dim, nc = CFG["dim"], CFG["n_classes"]
    widths = CFG["degenerate_widths"]
    total = sum(sum(p.numel() for p in m.parameters()) for m in B5._build_degenerate(dim, nc, widths))
    sw = max(8, int(total / (dim + nc)))

    def build_single() -> nn.ModuleList:
        return nn.ModuleList([nn.Sequential(nn.Linear(dim, sw), nn.GELU(), nn.Linear(sw, nc))])

    def build_copies() -> nn.ModuleList:
        return nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, widths[0]), nn.GELU(), nn.Linear(widths[0], nc)) for _ in widths]
        )

    deg = _retention_arm(lambda: B5._build_degenerate(dim, nc, widths), stream)
    sin = _retention_arm(build_single, stream)
    cop = _retention_arm(build_copies, stream)
    best_baseline = max(sin, cop)
    return {
        "bwt": {"degenerate": deg, "single": sin, "copies": cop},
        "best_baseline": "single" if sin >= cop else "copies",
        "gain": deg - best_baseline,  # advantage over the harder (better) baseline
    }


def _mean_gain(per_seed: list[dict], key: str) -> float:
    return _mean([s[key]["gain"] for s in per_seed])


def _verdict(name: str, mean_proj: float, mean_real: float, margin: float) -> tuple[str, str]:
    ratio = (mean_proj / mean_real) if abs(mean_real) > 1e-9 else float("nan")
    real_clears = mean_real > margin
    proj_clears = mean_proj > margin
    if not real_clears:
        return "inconclusive", (
            f"Real degenerate-minus-best-baseline retention gain ({mean_real:+.4f}) does not clear the "
            f"seed-spread margin ({margin:.4f}) at this scale, so the {name} substrate-specificity "
            f"question cannot be settled here ({name} gain {mean_proj:+.4f})."
        )
    if proj_clears and ratio == ratio and ratio >= RATIO_SURVIVE:
        return "reframed", (
            f"GENERIC ARCHITECTURAL effect under {name}: the degenerate retention advantage survives "
            f"the projection ({mean_proj:+.4f} vs real {mean_real:+.4f}, ratio {ratio:.2f}), so B5 is "
            f"real but NOT substrate specific (an architectural fact about structural diversity, not a "
            f"geometry claim)."
        )
    ratio_str = f"{ratio:.2f}" if ratio == ratio else "nan"
    below = ", below the margin" if not proj_clears else ""
    return "survives", (
        f"SUBSTRATE-SPECIFIC under {name}: the degenerate retention advantage collapses "
        f"({mean_proj:+.4f} vs real {mean_real:+.4f}, ratio {ratio_str}{below}), so the advantage "
        f"needed the real stream geometry. B5 survives as a substrate-specific positive under {name}."
    )


def main() -> None:
    t0 = time.time()
    per_seed: list[dict] = []
    for seed in SEEDS:
        seed_everything(seed)
        raw = _domain_stream(
            CFG["dim"], CFG["n_tasks"], CFG["n_classes"], CFG["samples"], CFG["separation"], seed
        )
        fr = _project_stream(raw, lambda x, s=seed: frozen_random_projection(x, seed=s))
        rr = _project_stream(raw, lambda x, s=seed: rank_reduced_projection(x, seed=s))
        seed_everything(seed)
        real = _run_substrate(raw)
        seed_everything(seed)
        frozen = _run_substrate(fr)
        seed_everything(seed)
        reduced = _run_substrate(rr)
        per_seed.append({"seed": seed, "real": real, "frozen_random": frozen, "rank_reduced": reduced})
        print(
            f"seed {seed}: real gain={real['gain']:+.4f} "
            f"(deg={real['bwt']['degenerate']:.3f} best={real['best_baseline']}) | "
            f"frozen_random gain={frozen['gain']:+.4f} | rank_reduced gain={reduced['gain']:+.4f}",
            flush=True,
        )

    mean_real = _mean_gain(per_seed, "real")
    mean_fr = _mean_gain(per_seed, "frozen_random")
    mean_rr = _mean_gain(per_seed, "rank_reduced")
    margin = _spread([s["real"]["gain"] for s in per_seed])

    fr_res, fr_verdict = _verdict("frozen_random", mean_fr, mean_real, margin)
    rr_res, rr_verdict = _verdict("rank_reduced", mean_rr, mean_real, margin)

    out = {
        "id": "b5_degeneracy_robustness",
        "question": (
            "Is the degenerate-ensemble RETENTION (backward-transfer) advantage substrate-specific "
            "(needs the real domain-incremental geometry) or a generic architectural fact (holds under "
            "any fixed projection of the inputs)?"
        ),
        "control": (
            "frozen_random_projection (invertible, e7-comparable) and rank_reduced_projection "
            "(lossy, stronger) applied to every task's x before training"
        ),
        "config": CFG,
        "seeds": SEEDS,
        "ratio_survive_threshold": RATIO_SURVIVE,
        "per_seed": per_seed,
        "mean_gain_real": round(mean_real, 5),
        "mean_gain_frozen_random": round(mean_fr, 5),
        "mean_gain_rank_reduced": round(mean_rr, 5),
        "seed_spread_margin": round(margin, 5),
        "ratio_frozen_over_real": None if abs(mean_real) < 1e-9 else round(mean_fr / mean_real, 4),
        "ratio_rank_reduced_over_real": None if abs(mean_real) < 1e-9 else round(mean_rr / mean_real, 4),
        "real_gain_clears_margin": bool(mean_real > margin),
        "frozen_random_resolution": fr_res,
        "rank_reduced_resolution": rr_res,
        "verdict": fr_verdict + " | " + rr_verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("\n" + out["verdict"])
    print(f"frozen_random={fr_res} rank_reduced={rr_res}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
