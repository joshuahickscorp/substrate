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

from mop.diagnostics.linear_probe import linear_probe  # noqa: E402
from mop.diagnostics.nonlinear_probe import nonlinear_probe  # noqa: E402
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from mop.seeding import parse_seeds  # noqa: E402
from mop.substrate import LatentStore  # noqa: E402

EXPERIMENT = {
    "id": "at1_grid_pilot",
    "metric": "per-substrate linear-probe delta over its OWN random-init control (shape factor)",
    "baseline": "matched-architecture from_config random-init per column (non-vacuous control)",
    "ablation": "capacity-capped MLP probe class (the probe-specific guard)",
    "null_hypothesis": "every substrate's delta over its own random-init is within seed spread "
    "(random-control-artifact): pretraining bought no nuisance invariance in any substrate",
    "tier": "cpu-now (pilot; the registered grid is studio)",
}

COLUMNS = [
    {
        "tag": "vjepa_fullclip",
        "modality": "video",
        "family": "vit_l_vjepa",
        "real": "vjepa2_vitl_nuisance_real",
        "randinit": "randominit_vitl_nuisance",
        "note": "real full-clip nuisance store is a lane-G product; skipped if absent",
    },
    {
        "tag": "vjepa_singleframe",
        "modality": "video_singleframe",
        "family": "vit_l_vjepa",
        "real": "vjepa2_vitl_singleframe",
        "randinit": "randominit_vitl_nuisance",
        "note": "control is the full-clip random-init ViT-L: matched arch and resolution, frame count "
        "differs (flagged confound, AT3 owns the clean time-axis claim)",
    },
    {
        "tag": "dinov2s",
        "modality": "image",
        "family": "vit_s_dinov2",
        "real": "dinov2s_nuisance_real",
        "randinit": "dinov2s_nuisance_randominit",
        "note": "",
    },
    {
        "tag": "qwen05b_textified",
        "modality": "text_rendered",
        "family": "qwen2_05b",
        "real": "qwen05b_textified_real",
        "randinit": "qwen05b_textified_randominit",
        "note": "label-free textification rendering",
    },
    {
        "tag": "wav2vec2_sonified",
        "modality": "audio_rendered",
        "family": "wav2vec2_base",
        "real": "wav2vec2_sonified_real",
        "randinit": "wav2vec2_sonified_randominit",
        "note": "preregistered sonification caveat applies: claim scope is within-rendering only",
    },
]
REFERENCE_COLUMNS = ["handcrafted_descriptors", "programmatic_reference"]  # AT4's, displayed not typed


def load_store(root: Path) -> tuple[torch.Tensor, torch.Tensor, dict | None] | None:
    if not (root / "meta.json").exists():
        return None
    store = LatentStore.open(root)
    y = store.labels()
    if y is None:
        return None
    sidecar = None
    if (root / "factors.json").exists():
        sidecar = json.loads((root / "factors.json").read_text())
    return store.latents().float(), y, sidecar


def clip_identity_check(sidecars: dict[str, dict | None]) -> dict:
    fingerprints = {
        t: (s["checksum_first"], s["checksum_last"]) for t, s in sidecars.items() if s is not None
    }
    distinct = set(fingerprints.values())
    return {
        "verified_columns": sorted(fingerprints),
        "unverified_columns": sorted(t for t, s in sidecars.items() if s is None),
        "identical": len(distinct) <= 1,
    }


def column_probe_deltas(
    x_real: torch.Tensor,
    x_rand: torch.Tensor,
    y: torch.Tensor,
    seeds: list[int],
    probe_epochs: int = 300,
    mlp_epochs: int = 150,
) -> dict:
    out: dict = {"linear": [], "mlp": [], "real_acc": [], "rand_acc": []}
    for s in seeds:
        pr = linear_probe(x_real, y, classification=True, epochs=probe_epochs, seed=s)
        pc = linear_probe(x_rand, y, classification=True, epochs=probe_epochs, seed=s)
        nr = nonlinear_probe(x_real, y, epochs=mlp_epochs, seed=s)
        nc = nonlinear_probe(x_rand, y, epochs=mlp_epochs, seed=s)
        out["real_acc"].append(round(pr["score"], 4))
        out["rand_acc"].append(round(pc["score"], 4))
        out["linear"].append(round(pr["score"] - pc["score"], 4))
        out["mlp"].append(round(nr["score"] - nc["score"], 4))
    return out


def classify_column(linear_deltas: list[float], mlp_deltas: list[float]) -> dict:
    lin = seed_ci(linear_deltas)
    mlp = seed_ci(mlp_deltas)
    flips = sign_flip_report(linear_deltas)
    if lin["lo"] <= 0:
        verdict = "random-control-artifact"
    elif mlp["lo"] <= 0:
        verdict = "probe-specific"
    elif flips["any_flip"]:
        verdict = "non-replicating"
    else:
        verdict = "genuine-substrate-signal"
    return {"verdict": verdict, "linear_ci": lin, "mlp_ci": mlp, "sign_flips": flips}


def grid_scope(columns: list[dict]) -> dict:
    if not columns:
        return {"scope": "no-columns-available", "survivors": []}
    survivors = [c for c in columns if c["verdict"] == "genuine-substrate-signal"]
    mods = sorted({c["modality"] for c in survivors})
    if not survivors:
        scope = "random-control-artifact (grid-wide): no available substrate beat its own random-init"
    elif len(mods) >= 2:
        scope = (
            f"pilot-universal across {mods} (available columns only; the registered universal claim "
            "needs the complete citable grid and matched random-init columns)"
        )
    else:
        scope = f"modality-specific ({mods[0]}) within the available columns (pilot scope)"
    return {"scope": scope, "survivors": [c["tag"] for c in survivors], "survivor_modalities": mods}


def evaluate_grid(
    cache_root: Path,
    seeds: list[int],
    columns: list[dict] | None = None,
    reference_columns: list[str] | None = None,
    probe_epochs: int = 300,
    mlp_epochs: int = 150,
) -> dict:
    columns = COLUMNS if columns is None else columns
    reference_columns = REFERENCE_COLUMNS if reference_columns is None else reference_columns
    t0 = time.perf_counter()
    grid, skipped, sidecars = [], [], {}
    for col in columns:
        real = load_store(cache_root / col["real"])
        rand = load_store(cache_root / col["randinit"])
        if real is None or rand is None:
            skipped.append(
                {
                    "tag": col["tag"],
                    "missing": [n for n, v in ((col["real"], real), (col["randinit"], rand)) if v is None],
                }
            )
            continue
        x_real, y_real, sc_real = real
        x_rand, y_rand, _ = rand
        if x_real.shape[0] != x_rand.shape[0] or not torch.equal(y_real, y_rand):
            skipped.append({"tag": col["tag"], "missing": ["label/count mismatch between arms"]})
            continue
        sidecars[col["tag"]] = sc_real
        deltas = column_probe_deltas(x_real, x_rand, y_real, seeds, probe_epochs, mlp_epochs)
        cls = classify_column(deltas["linear"], deltas["mlp"])
        grid.append({**{k: col[k] for k in ("tag", "modality", "family", "note")}, **deltas, **cls})
        print(f"[{col['tag']}] verdict={cls['verdict']} lin_ci={cls['linear_ci']}", flush=True)
    refs = {}
    for name in reference_columns:
        loaded = load_store(cache_root / name)
        if loaded is None:
            continue
        x, y, sc = loaded
        accs = [
            round(linear_probe(x, y, classification=True, epochs=probe_epochs, seed=s)["score"], 4)
            for s in seeds
        ]
        refs[name] = {"probe_acc": accs, "ci": seed_ci(accs)}
        sidecars[name] = sc
    scope = grid_scope([{k: c[k] for k in ("tag", "modality", "verdict")} for c in grid])
    if grid:
        verdict = scope["scope"]
        null_supported = all(c["verdict"] == "random-control-artifact" for c in grid)
    else:
        verdict = "NO COLUMNS: no real+random-init cache pair is on disk yet, nothing typed"
        null_supported = None  # not evaluable: nothing was tested, never a rejected null
    return {
        "experiment": EXPERIMENT,
        "clipset": CLIPSET,
        "seeds": seeds,
        "grid": grid,
        "reference_columns": refs,
        "skipped_columns": skipped,
        "clip_identity": clip_identity_check(sidecars),
        "scope": scope,
        "verdict": verdict,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AT1 cross-substrate nuisance grid pilot (nine verdicts)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--probe-epochs", type=int, default=300)
    ap.add_argument("--mlp-epochs", type=int, default=150)
    ap.add_argument("--out", default="runs/mot/at1_grid_pilot.json")
    ap.add_argument("--rerun", action="store_true", help="Stage 4 rerun marker, recorded in the output")
    a = ap.parse_args(argv)
    result = evaluate_grid(
        Path(a.cache_root), parse_seeds(a.seeds), probe_epochs=a.probe_epochs, mlp_epochs=a.mlp_epochs
    )
    result["rerun"] = a.rerun
    text = json.dumps(result, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
