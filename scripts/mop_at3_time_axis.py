#!/usr/bin/env python
"""AT3 (WP-11): time-axis ablation. Which factors decode from full-clip V-JEPA but collapse at chance
from the single-frame (middle frame tiled to 64 frames, token/frame-matched) V-JEPA arm? A full-clip
advantage at MATCHED token count is a temporal currency, not a raw-information advantage.

Factors probed per arm (linear probe, matched split per seed):
  shape (static, from labels), color (static, from labels),
  motion_dir4 (temporal: quadrant bin of the clip's nuisance motion vector, recovered seed-for-seed by
  the cache scripts), speed2 (temporal: median split of motion magnitude).

PREREGISTERED NULL (verbatim, registry AT3): with token/frame count equalized, full-clip and
single-frame decodability are within seed spread for every factor: no factor genuinely requires the
temporal axis.

PREREGISTERED VERDICT RULE (fixed before any result exists): the null is REJECTED only if at least one
factor's full-minus-single delta has a 5-seed 95 percent CI excluding zero from below with no per-seed
sign flip. A factor is reported "needs_time" if additionally its single-frame accuracy is within 0.1 of
chance. Consumes the WP-11 caches; never loads an encoder.

Usage: python scripts/mop_at3_time_axis.py --seeds 0-4

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cache_randominit_vitl_features import NUISANCE_COLS, load_feature_cache  # noqa: E402

from mop.diagnostics.linear_probe import linear_probe  # noqa: E402
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from mop.experiments.base import Experiment  # noqa: E402
from mop.seeding import parse_seeds as _parse_seeds  # noqa: E402

DEFAULTS = {
    "full_cache": "data/cache/vjepa2_vitl_nuisance",
    "single_cache": "data/cache/vjepa2_vitl_singleframe",
    "seeds": (0, 1, 2, 3, 4),
    "probe_epochs": 300,
    "chance_margin": 0.1,
}


def _get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def temporal_labels(nuisance: torch.Tensor) -> dict[str, torch.Tensor]:
    """Derive the temporal factor labels from the recorded nuisance draws: motion_dir4 is the quadrant
    of (vx, vy), speed2 the median split of the motion magnitude."""
    vx = nuisance[:, NUISANCE_COLS.index("vx")]
    vy = nuisance[:, NUISANCE_COLS.index("vy")]
    ang = torch.atan2(vy, vx)  # [-pi, pi)
    motion_dir4 = torch.floor((ang + math.pi) / (math.pi / 2)).clamp(0, 3).to(torch.int64)
    speed = torch.sqrt(vx * vx + vy * vy)
    speed2 = (speed > speed.median()).to(torch.int64)
    return {"motion_dir4": motion_dir4, "speed2": speed2}


def run(cfg=None, device=None, run_dir: Path | None = None) -> dict:
    full = load_feature_cache(_get(cfg, "full_cache", DEFAULTS["full_cache"]))
    single = load_feature_cache(_get(cfg, "single_cache", DEFAULTS["single_cache"]))
    if single["labels_shape"] is None or not torch.equal(full["labels_shape"], single["labels_shape"]):
        raise ValueError("full and single-frame caches disagree on labels: clip identity rule violated")
    if full["nuisance"] is None:
        raise ValueError("full-clip cache has no nuisance.npy: temporal factors cannot be derived")
    factors: dict[str, torch.Tensor] = {"shape": full["labels_shape"]}
    if full["labels_color"] is not None:
        factors["color"] = full["labels_color"]
    factors.update(temporal_labels(full["nuisance"]))
    seeds = list(_get(cfg, "seeds", DEFAULTS["seeds"]))
    epochs = int(_get(cfg, "probe_epochs", DEFAULTS["probe_epochs"]))
    margin = float(_get(cfg, "chance_margin", DEFAULTS["chance_margin"]))
    t0 = time.perf_counter()
    per_factor: dict[str, dict] = {}
    any_win = False
    for name, y in factors.items():
        accs_f, accs_s, deltas = [], [], []
        chance = 1.0 / (int(y.max()) + 1)
        for s in seeds:
            rf = linear_probe(full["features"], y, epochs=epochs, seed=s)
            rs = linear_probe(single["features"], y, epochs=epochs, seed=s)
            accs_f.append(rf["score"])
            accs_s.append(rs["score"])
            deltas.append(rf["score"] - rs["score"])
        ci = seed_ci(deltas)
        flips = sign_flip_report(deltas)
        win = ci["lo"] > 0 and flips["consistent_sign"] == 1
        any_win = any_win or win
        single_mean = sum(accs_s) / len(accs_s)
        per_factor[name] = {
            "chance": round(chance, 4),
            "full_acc_mean": round(sum(accs_f) / len(accs_f), 4),
            "single_acc_mean": round(single_mean, 4),
            "delta_ci": ci,
            "sign_flips": flips,
            "delta_win": win,
            "needs_time": bool(win and single_mean <= chance + margin),
        }
    null_supported = not any_win
    needs_time = sorted(k for k, v in per_factor.items() if v["needs_time"])
    out = {
        "experiment": AT3Experiment.id,
        "contract": AT3Experiment().contract(),
        "seeds": seeds,
        "factors": per_factor,
        "needs_time_factors": needs_time,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": (
            "NULL SUPPORTED: full-clip ties single-frame on every factor at matched token count, no "
            "factor requires the temporal axis here"
            if null_supported
            else f"NULL REJECTED: temporal currency exists (needs_time={needs_time or 'delta-only'}), "
            "full-clip decodes what a token-matched static frame cannot"
        ),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


class AT3Experiment(Experiment):
    id = "at3_time_axis"
    metric = ("full_minus_single_decode_delta_per_factor",)
    baseline = "single-frame V-JEPA (middle frame tiled to 64 frames, token/frame-matched)"
    ablation = "temporal axis removed at matched token count"
    null_hypothesis = (
        "With token/frame count equalized, full-clip and single-frame decodability are within seed "
        "spread for every factor: no factor genuinely requires the temporal axis"
    )
    tier = "cpu-now"

    def run(self, cfg, device, run_dir: Path) -> dict:
        return run(cfg, device, run_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AT3 time-axis ablation, full-clip vs single-frame V-JEPA")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--full-cache", default=DEFAULTS["full_cache"])
    ap.add_argument("--single-cache", default=DEFAULTS["single_cache"])
    ap.add_argument("--probe-epochs", type=int, default=DEFAULTS["probe_epochs"])
    ap.add_argument("--rerun", action="store_true", help="stage-4 rerun naming (_seeds10 suffix)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {
        "seeds": _parse_seeds(a.seeds),
        "full_cache": a.full_cache,
        "single_cache": a.single_cache,
        "probe_epochs": a.probe_epochs,
    }
    out_path = a.out or ("runs/mot/at3_time_axis_seeds10.json" if a.rerun else "runs/mot/at3_time_axis.json")
    result = run(cfg, "cpu", None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("needs_time_factors", "null_supported", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
