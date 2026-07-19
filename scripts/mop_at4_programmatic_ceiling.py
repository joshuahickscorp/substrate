#!/usr/bin/env python

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

from scripts.featurize_programmatic import CACHE_ROOT, CLIPSET  # noqa: E402

from mop.diagnostics.cross_substrate import shuffled_label_null  # noqa: E402
from mop.diagnostics.linear_probe import linear_probe  # noqa: E402
from mop.diagnostics.riskcov import seed_ci  # noqa: E402
from mop.seeding import parse_seeds  # noqa: E402

CONTRACT = {
    "id": "AT4",
    "metric": "per-factor probe accuracy delta: programmatic ceiling minus best perceptual column",
    "baseline": "the perceptual encoder caches on the same clips and factors (same probes, same seeds)",
    "ablation": "handcrafted-descriptor reference column, shuffled-label floor per column",
    "null_hypothesis": (
        "The programmatic reference does not exceed the perceptual substrates on any atlas factor "
        "(trivially separable everywhere), so no perceptual tie can be attributed to a substrate bound"
    ),
    "tier": "cpu-now",
}

CEILING_COLUMN = "programmatic_reference"
HANDCRAFTED_COLUMN = "handcrafted_descriptors"
PERCEPTUAL_STORES = [
    "vjepa2_vitl_nuisance_real",
    "vjepa2_vitl_singleframe",
    "dinov2s_nuisance_real",
    "qwen05b_textified_real",
    "wav2vec2_sonified_real",
]
FACTORS = ["shape", "color"]
CEILING_MIN = 0.99  # the programmatic column carries the one-hot factor, anything less is a broken run


def load_column(root: Path) -> dict | None:
    if not (root / "meta.json").exists() or not (root / "factors.json").exists():
        return None
    from mop.substrate import LatentStore

    store = LatentStore.open(root)
    sidecar = json.loads((root / "factors.json").read_text())
    factors = {f: torch.tensor(sidecar[f], dtype=torch.long) for f in FACTORS if f in sidecar}
    if not factors:
        return None
    x = store.latents().float()
    if any(len(v) != x.shape[0] for v in factors.values()):
        return None
    return {"x": x, "factors": factors, "sidecar": sidecar}


def clip_identity_check(sidecars: dict[str, dict]) -> dict:
    fp = {
        t: (s.get("checksum_first"), s.get("checksum_last"))
        for t, s in sidecars.items()
        if s.get("checksum_first")
    }
    return {
        "verified_columns": sorted(fp),
        "unverified_columns": sorted(set(sidecars) - set(fp)),
        "identical": len(set(fp.values())) <= 1,
    }


def probe_accs(x: torch.Tensor, y: torch.Tensor, seeds: list[int], epochs: int) -> list[float]:
    return [round(linear_probe(x, y, classification=True, epochs=epochs, seed=s)["score"], 4) for s in seeds]


def factor_reading(prog_accs: list[float], perceptual: dict[str, list[float]], chance: float) -> dict:
    prog = seed_ci(prog_accs)
    if prog["mean"] < CEILING_MIN:
        return {
            "reading": "DEGENERATE",
            "programmatic_ci": prog,
            "detail": f"ceiling column decodes at {prog['mean']:.3f} < {CEILING_MIN}, pipeline broken",
        }
    if not perceptual:
        return {
            "reading": "NO-PERCEPTUAL-COLUMNS",
            "programmatic_ci": prog,
            "detail": "no perceptual cache on disk yet, ceiling reported, nothing compared",
        }
    best_tag = max(perceptual, key=lambda t: seed_ci(perceptual[t])["mean"])
    deltas = [p - b for p, b in zip(prog_accs, perceptual[best_tag], strict=True)]
    dci = seed_ci(deltas)
    headroom = dci["lo"] > 0
    return {
        "reading": "HEADROOM" if headroom else "TEST-TOO-EASY",
        "programmatic_ci": prog,
        "best_perceptual": best_tag,
        "best_perceptual_ci": seed_ci(perceptual[best_tag]),
        "delta_ci": dci,
        "delta_per_seed": [round(d, 4) for d in deltas],
        "chance": round(chance, 4),
        "detail": (
            "perceptual sits measurably below the ceiling, a tie here reads substrate-bounded"
            if headroom
            else "perceptual ties the ceiling, no tie on this factor is attributable to a substrate bound"
        ),
    }


def run(cache_root: Path, seeds: list[int], probe_epochs: int = 300) -> dict:
    t0 = time.perf_counter()
    columns: dict[str, dict] = {}
    skipped: list[str] = []
    for name in [CEILING_COLUMN, HANDCRAFTED_COLUMN, *PERCEPTUAL_STORES]:
        loaded = load_column(Path(cache_root) / name)
        if loaded is None:
            skipped.append(name)
            print(f"[skip] {name}: store or factors sidecar absent", flush=True)
        else:
            columns[name] = loaded
    if CEILING_COLUMN not in columns:
        return {
            "experiment": "mop_at4_programmatic_ceiling",
            "contract": CONTRACT,
            "clipset": CLIPSET,
            "seeds": seeds,
            "skipped_columns": skipped,
            "verdict": "NO CEILING: the programmatic_reference store is not on disk, run "
            "scripts/featurize_programmatic.py first",
            "null_supported": None,
            "seconds": round(time.perf_counter() - t0, 1),
        }

    table: dict[str, dict[str, dict]] = {}
    shuffled_floor: dict[str, dict[str, float]] = {}
    for tag, col in columns.items():
        table[tag] = {}
        shuffled_floor[tag] = {}
        for factor, y in col["factors"].items():
            accs = probe_accs(col["x"], y, seeds, probe_epochs)
            table[tag][factor] = {"per_seed": accs, "ci": seed_ci(accs)}
            shuffled_floor[tag][factor] = round(
                shuffled_label_null(col["x"], y, seed=seeds[0], probe_epochs=probe_epochs), 4
            )
            print(f"[{tag}] {factor}: acc={table[tag][factor]['ci']['mean']:.3f}", flush=True)

    readings: dict[str, dict] = {}
    hand_readings: dict[str, dict] = {}
    for factor in FACTORS:
        if factor not in columns[CEILING_COLUMN]["factors"]:
            continue
        y = columns[CEILING_COLUMN]["factors"][factor]
        chance = 1.0 / float(int(y.max()) + 1)
        perceptual = {
            t: table[t][factor]["per_seed"] for t in PERCEPTUAL_STORES if t in table and factor in table[t]
        }
        readings[factor] = factor_reading(table[CEILING_COLUMN][factor]["per_seed"], perceptual, chance)
        if HANDCRAFTED_COLUMN in table and factor in table[HANDCRAFTED_COLUMN]:
            hand = seed_ci(table[HANDCRAFTED_COLUMN][factor]["per_seed"])
            hand_readings[factor] = {
                "handcrafted_ci": hand,
                "note": "classical-vision reference, reported alongside; the null keys on the "
                "programmatic column only",
            }

    compared = [f for f, r in readings.items() if r["reading"] in ("HEADROOM", "TEST-TOO-EASY")]
    degenerate = [f for f, r in readings.items() if r["reading"] == "DEGENERATE"]
    headroom = [f for f in compared if readings[f]["reading"] == "HEADROOM"]
    if degenerate:
        verdict = f"DEGENERATE: ceiling column below {CEILING_MIN} on {degenerate}, pipeline broken"
        null_supported = None
    elif not compared:
        verdict = (
            "NO PERCEPTUAL COLUMNS: the ceiling is certified but no perceptual cache is on disk, "
            "nothing compared, the null is not evaluable yet"
        )
        null_supported = None
    elif headroom:
        verdict = (
            f"HEADROOM on {headroom}: the programmatic ceiling exceeds every perceptual column beyond "
            "the seed spread there, so a perceptual tie on those factors reads substrate-bounded (the "
            "custom-branch trigger is armed for them)"
        )
        null_supported = False
    else:
        verdict = (
            "NULL (test-too-easy everywhere): perceptual columns tie the programmatic ceiling on every "
            "comparable factor, no perceptual tie can be attributed to a substrate bound"
        )
        null_supported = True
    return {
        "experiment": "mop_at4_programmatic_ceiling",
        "contract": CONTRACT,
        "clipset": CLIPSET,
        "seeds": seeds,
        "probe_epochs": probe_epochs,
        "columns": sorted(columns),
        "skipped_columns": skipped,
        "clip_identity": clip_identity_check({t: c["sidecar"] for t, c in columns.items()}),
        "accuracy_table": table,
        "shuffled_label_floor": shuffled_floor,
        "per_factor_reading": readings,
        "handcrafted_reference": hand_readings,
        "verdict_rule": (
            f"per factor: DEGENERATE iff ceiling mean < {CEILING_MIN}; HEADROOM iff seed-CI lower "
            "bound of (programmatic minus best perceptual) > 0; else TEST-TOO-EASY; null supported "
            "iff at least one factor compared and none shows headroom; not-evaluable states "
            "(NO CEILING, NO PERCEPTUAL COLUMNS, DEGENERATE) report null_supported=None, never a "
            "rejected null (fixed in code before running)"
        ),
        "verdict": verdict,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AT4: programmatic ceiling reference over the atlas factors")
    ap.add_argument("--seeds", default="0-4", help="seed spec, e.g. 0-4 or 0,2,5")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--probe-epochs", type=int, default=300)
    ap.add_argument("--out", default="runs/mot/at4_programmatic_ceiling.json")
    ap.add_argument("--rerun", action="store_true", help="Stage 4 rerun marker, recorded in the output")
    a = ap.parse_args(argv)
    result = run(Path(a.cache_root), parse_seeds(a.seeds), probe_epochs=a.probe_epochs)
    result["rerun"] = a.rerun
    text = json.dumps(result, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(json.dumps({k: result[k] for k in ("verdict", "null_supported", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
