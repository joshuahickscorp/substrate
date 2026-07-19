
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mop.diagnostics.continual_metrics import backward_transfer as bwt_matrix
from mop.diagnostics.riskcov import pareto_frontier, seed_ci
from mop.metrics import FrontierPoint, pareto_front
from mop.seeding import seed_everything
from mop.shell.buffer import ReplayBuffer

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "data" / "cache" / "vjepa2_vitl_nuisance"
OUT = _ROOT / "runs" / "mot" / "density_retention_byte.json"
DIM = 1024
BYTES_PER_FLOAT = 4  # float32 stored latents
K_PER_CLASS = [0, 2, 4, 8, 16, 32]  # >= 4 non-trivial budget points (plus K=0 no-replay floor)
SEEDS = [0, 1, 2, 3, 4]  # 5 seeds
EPOCHS = 40
LR = 0.05
N_CLASSES = 5

RETENTION_WIN_MARGIN = 0.05
KNEE_MARGINAL_EFF_FRAC = 0.10


def load_stream():
    X = torch.from_numpy(np.load(CACHE / "features.npy")).float()
    y = torch.from_numpy(np.load(CACHE / "labels_shape.npy")).long()
    tasks = []
    for c in range(N_CLASSES):
        m = y == c
        tasks.append((X[m], y[m]))
    return tasks


def run_stream(tasks, k_per_class: int, seed: int):
    seed_everything(seed)
    head = nn.Linear(DIM, N_CLASSES)
    opt = torch.optim.Adam(head.parameters(), lr=LR)

    bufs: dict[int, ReplayBuffer] = {}

    def all_replay(exclude=None):
        xs, ys = [], []
        for c, b in bufs.items():
            if c == exclude or len(b) == 0:
                continue
            s = b.sample(len(b))
            xs.append(s["x"])
            ys.append(s["y"])
        if not xs:
            return None
        return torch.cat(xs), torch.cat(ys)

    acc_matrix = [[0.0] * N_CLASSES for _ in range(N_CLASSES)]
    for ti, (xt, yt) in enumerate(tasks):
        for _ in range(EPOCHS):
            opt.zero_grad()
            xb, yb = xt, yt
            rep = all_replay()
            if rep is not None:
                xb = torch.cat([xt, rep[0]])
                yb = torch.cat([yt, rep[1]])
            F.cross_entropy(head(xb), yb).backward()
            opt.step()
        if k_per_class > 0:
            b = ReplayBuffer(k_per_class, DIM, prioritized=False, seed=seed + 100 * ti)
            b.add(xt, yt)
            bufs[ti] = b
        with torch.no_grad():
            for j in range(N_CLASSES):
                xj, yj = tasks[j]
                acc_matrix[ti][j] = float((head(xj).argmax(-1) == yj).float().mean())

    stored = sum(len(b) for b in bufs.values())
    return acc_matrix, stored


def retention_scores(acc_matrix):
    T = len(acc_matrix)
    bwt = bwt_matrix(acc_matrix)  # reuse the factored BWT core, not a hand-rolled loop
    final_mean = sum(acc_matrix[T - 1][j] for j in range(T)) / T
    final_first = acc_matrix[T - 1][0]  # retention on the EARLIEST class (most-forgotten)
    return {"bwt": bwt, "final_mean": final_mean, "final_first": final_first}


def main():
    tasks = load_stream()

    raw = {}  # k -> list over seeds of dict(bwt, final_mean, final_first, bytes, stored)
    for k in K_PER_CLASS:
        raw[k] = []
        for s in SEEDS:
            acc_matrix, stored = run_stream(tasks, k, s)
            sc = retention_scores(acc_matrix)
            sc["stored"] = stored
            sc["bytes"] = stored * DIM * BYTES_PER_FLOAT
            raw[k].append(sc)

    agg = {}
    for k in K_PER_CLASS:
        rows = raw[k]
        agg[k] = {
            "k_per_class": k,
            "stored_exemplars": rows[0]["stored"],  # deterministic across seeds
            "bytes": rows[0]["bytes"],
            "bwt": seed_ci([r["bwt"] for r in rows]),
            "final_mean": seed_ci([r["final_mean"] for r in rows]),
            "final_first": seed_ci([r["final_first"] for r in rows]),
        }

    floor_bwt = agg[0]["bwt"]["mean"]  # K=0 no-replay retention floor
    floor_final = agg[0]["final_mean"]["mean"]

    per_byte = {}
    ordered = [k for k in K_PER_CLASS if k > 0]
    prev_bytes, prev_bwt = 0.0, floor_bwt
    first_marginal_eff = None
    for k in ordered:
        b = agg[k]["bytes"]
        gain_bwt = agg[k]["bwt"]["mean"] - floor_bwt
        ratio = gain_bwt / b if b > 0 else 0.0
        d_bytes = b - prev_bytes
        d_bwt = agg[k]["bwt"]["mean"] - prev_bwt
        marginal_eff = d_bwt / d_bytes if d_bytes > 0 else 0.0
        if first_marginal_eff is None:
            first_marginal_eff = marginal_eff
        per_byte[k] = {
            "bytes": b,
            "bwt_gain_over_floor": round(gain_bwt, 4),
            "retention_per_byte_ratio": ratio,
            "marginal_retention_per_byte": marginal_eff,
            "marginal_eff_frac_of_first": (
                marginal_eff / first_marginal_eff if first_marginal_eff not in (None, 0.0) else 0.0
            ),
        }
        prev_bytes, prev_bwt = b, agg[k]["bwt"]["mean"]

    knee_k = None
    for k in ordered:
        if per_byte[k]["marginal_eff_frac_of_first"] >= KNEE_MARGINAL_EFF_FRAC:
            knee_k = k
    wins = {k: bool(agg[k]["bwt"]["mean"] - floor_bwt >= RETENTION_WIN_MARGIN) for k in ordered}

    pts_bwt = [(float(agg[k]["bytes"]), float(agg[k]["bwt"]["mean"])) for k in K_PER_CLASS]
    pts_final = [(float(agg[k]["bytes"]), float(agg[k]["final_mean"]["mean"])) for k in K_PER_CLASS]
    front_bwt = pareto_frontier(pts_bwt)  # keep point iff no cheaper point has >= retention
    front_final = pareto_frontier(pts_final)

    max_bytes = max(p[0] for p in pts_bwt) or 1.0
    fps = [
        FrontierPoint(
            name=f"K{k}",
            adaptation=1.0 - agg[k]["bytes"] / max_bytes,
            retention=max(0.0, 1.0 + agg[k]["bwt"]["mean"]),
        )
        for k in K_PER_CLASS
    ]
    fp_front = [p.name for p in pareto_front(fps)]

    result = {
        "task": "retention_per_byte_frontier",
        "input_cache": str(CACHE),
        "stream": "class_incremental_5_shape_classes",
        "dim": DIM,
        "bytes_accounting": "stored_exemplars * dim * 4 (float32 latents)",
        "prereg_thresholds": {
            "K_PER_CLASS": K_PER_CLASS,
            "SEEDS": SEEDS,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "RETENTION_WIN_MARGIN_bwt": RETENTION_WIN_MARGIN,
            "KNEE_MARGINAL_EFF_FRAC": KNEE_MARGINAL_EFF_FRAC,
        },
        "no_replay_floor": {"bwt": round(floor_bwt, 4), "final_mean": round(floor_final, 4)},
        "budgets": {
            str(k): {
                "k_per_class": agg[k]["k_per_class"],
                "stored_exemplars": agg[k]["stored_exemplars"],
                "bytes": agg[k]["bytes"],
                "bwt_mean": agg[k]["bwt"]["mean"],
                "bwt_ci_half": agg[k]["bwt"]["ci_half"],
                "bwt_lo": agg[k]["bwt"]["lo"],
                "bwt_hi": agg[k]["bwt"]["hi"],
                "final_mean": agg[k]["final_mean"]["mean"],
                "final_mean_ci_half": agg[k]["final_mean"]["ci_half"],
                "final_first_mean": agg[k]["final_first"]["mean"],
            }
            for k in K_PER_CLASS
        },
        "gaming_guard": {
            "rule": "bigger buffer trivially wins -> report ratio + knee, not raw retention",
            "retention_per_byte": {
                str(k): {
                    "bytes": per_byte[k]["bytes"],
                    "bwt_gain_over_floor": per_byte[k]["bwt_gain_over_floor"],
                    "retention_per_byte_ratio_1e6": round(per_byte[k]["retention_per_byte_ratio"] * 1e6, 6),
                    "marginal_retention_per_byte_1e6": round(
                        per_byte[k]["marginal_retention_per_byte"] * 1e6, 6
                    ),
                    "marginal_eff_frac_of_first": round(per_byte[k]["marginal_eff_frac_of_first"], 4),
                    "beats_floor_by_margin": wins[k],
                }
                for k in ordered
            },
            "knee_k_per_class": knee_k,
            "knee_bytes": agg[knee_k]["bytes"] if knee_k is not None else None,
            "note_ratio_units": "retention_per_byte_ratio_1e6 is BWT gain per MEGA-byte-equivalent (ratio * 1e6) for readability",
        },
        "pareto_frontier_bytes_vs_retention": {
            "bwt": [{"bytes": b, "retention_bwt": r} for b, r in front_bwt],
            "final_mean": [{"bytes": b, "retention_final_mean": r} for b, r in front_final],
            "frontierpoint_front_names": fp_front,
        },
    }

    winning_budgets = [k for k in ordered if wins[k]]
    floor_forgets = floor_bwt < -RETENTION_WIN_MARGIN  # did the no-replay arm actually forget?
    if not floor_forgets:
        verdict = (
            "NULL / uninformative regime: the no-replay (K=0) arm does NOT catastrophically "
            f"forget (BWT floor = {round(floor_bwt, 4)} >= -{RETENTION_WIN_MARGIN}). With little "
            "forgetting to undo, replay buys little retention, so the byte axis cannot bite here. "
            "This is the toy-scale trap flagged in 12_metrics.md B1(ii): report as honest null."
        )
    elif not winning_budgets:
        verdict = (
            f"NULL: the no-replay arm forgets (BWT floor = {round(floor_bwt, 4)}), but no replay "
            f"budget beats that floor by the preregistered {RETENTION_WIN_MARGIN} BWT margin. "
            "Replay does not measurably restore retention at pilot scale. Honest null."
        )
    else:
        verdict = (
            f"FRONTIER PRESENT: no-replay forgets (BWT floor = {round(floor_bwt, 4)}); replay "
            f"budgets {winning_budgets} (K per class) beat that floor by >= {RETENTION_WIN_MARGIN} "
            f"BWT. Knee at K={knee_k} ({agg[knee_k]['bytes'] if knee_k else None} bytes): past the "
            "knee, extra bytes buy negligible marginal retention. Report the knee + ratio, not the "
            "raw max-retention budget (which trivially wins by spending more memory)."
        )
    result["verdict"] = verdict
    result["verdict_inputs"] = {
        "no_replay_forgets": bool(floor_forgets),
        "winning_budgets_k": winning_budgets,
        "knee_k": knee_k,
    }

    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
