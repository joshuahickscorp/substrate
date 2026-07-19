#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if (_ROOT / "scripts" / "featurize_programmatic.py").exists() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.featurize_programmatic import CACHE_ROOT  # noqa: E402
from scripts.mop_al2_alignment_pilot import evaluate_pairs  # noqa: E402
from scripts.mop_at1_grid_pilot import COLUMNS as AT1_PILOT_COLUMNS  # noqa: E402
from scripts.mop_at1_grid_pilot import REFERENCE_COLUMNS, evaluate_grid, parse_seeds  # noqa: E402

from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402

MIN_FREE_RAM_GB = 32.0

DEFAULT_N_SHAPE, DEFAULT_N_COLOR = 5, 5
ALIGN_TRAIN_FRAC = 0.6  # AB2 rows used to fit the shared map; rest are eval rows (verbatim from laptop)
RANK = 32  # AB2 preregistered shared-map rank (verbatim)
RIDGE_LAMBDA = 1.0  # AB2 ridge lambda (verbatim)

REGISTERED_COLUMNS = list(AT1_PILOT_COLUMNS) + [
    {
        "tag": "vjepa_bound_video",
        "modality": "video_bound",
        "family": "vit_l_vjepa",
        "real": "vjepa2_vitl_bound_video",
        "randinit": "randominit_vitl_nuisance",
        "note": "DR1 curated real bound-attribute video; control is the matched random-init ViT-L. Only "
        "admitted to the abstraction axes when its DR1 acceptance gate passed (dr1_gate_passed in the "
        "sidecar): target attribute label-free-recoverable from the paired caption, probe-verified above "
        "chance, BEFORE encode was spent.",
        "dr1_gated": True,
    },
    {
        "tag": "dinov2b",
        "modality": "image",
        "family": "vit_b_dinov2",
        "real": "dinov2b_nuisance_real",
        "randinit": "dinov2b_nuisance_randominit",
        "note": "at-scale larger DINOv2 column (Studio encode)",
    },
    {
        "tag": "vjepa21_dense_8192",
        "modality": "video_dense",
        "family": "vit_l_vjepa21_dense",
        "real": "vjepa21_vitl_dense8192_real",
        "randinit": "vjepa21_vitl_dense8192_randominit",
        "note": "STUDIO-TIME: dense V-JEPA 2.1 8192-token latents + matched random-init dense ViT. Absent "
        "on the laptop; excluded from the universal claim until the Studio dense encode writes both caches.",
    },
]

REGISTERED_ALIGNMENT_ARMS = [
    "vjepa2_vitl_nuisance",
    "vjepa2_vitl_singleframe",
    "vjepa2_vitl_bound_video",
    "dinov2s_nuisance_real",
    "dinov2b_nuisance_real",
    "qwen05b_textified_real",
    "wav2vec2_sonified_real",
]

RANDINIT_FOR = {
    "vjepa2_vitl_nuisance": "randominit_vitl_nuisance",
    "vjepa2_vitl_singleframe": "randominit_vitl_nuisance",
    "vjepa2_vitl_bound_video": "randominit_vitl_nuisance",
    "dinov2s_nuisance_real": "dinov2s_nuisance_randominit",
    "dinov2b_nuisance_real": "dinov2b_nuisance_randominit",
    "qwen05b_textified_real": "qwen05b_textified_randominit",
    "wav2vec2_sonified_real": "wav2vec2_sonified_randominit",
}

DR1_GATED_ARMS = {"vjepa2_vitl_bound_video"}

ARM_MODALITY = {
    "vjepa2_vitl_nuisance": "video",
    "vjepa2_vitl_singleframe": "video_singleframe",
    "vjepa2_vitl_bound_video": "video_bound",
    "dinov2s_nuisance_real": "image",
    "dinov2b_nuisance_real": "image",
    "qwen05b_textified_real": "text_rendered",
    "wav2vec2_sonified_real": "audio_rendered",
    "vjepa21_vitl_dense8192_real": "video_dense",
}


def assert_studio_ram(min_gb: float = MIN_FREE_RAM_GB) -> float:
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024**3)
    except Exception as e:
        raise SystemExit(
            f"cannot read free RAM ({e}); refusing to run without the >= {min_gb:.0f}GB safety check. "
            "Install psutil on the Studio box."
        ) from e
    if free_gb < min_gb:
        raise SystemExit(
            f"free RAM {free_gb:.1f}GB < required {min_gb:.0f}GB. This is the at-scale Studio grid; it "
            "will not run on the laptop pool (18GB)."
        )
    return free_gb


def _load_latents_factors(cache_root: Path, tag: str):
    b = cache_root / tag
    if not (b / "meta.json").exists():
        return None
    if (b / "features.npy").exists():
        x = np.load(b / "features.npy")
        sh = np.load(b / "labels_shape.npy")
        co = np.load(b / "labels_color.npy")
        sidecar = None
        if (b / "factors.json").exists():
            sidecar = json.loads((b / "factors.json").read_text())
    elif (b / "latents.npy").exists():
        x = np.load(b / "latents.npy")
        if not (b / "factors.json").exists():
            return None
        sidecar = json.loads((b / "factors.json").read_text())
        sh = np.asarray(sidecar["shape"])
        co = np.asarray(sidecar["color"])
    else:
        return None
    return (
        torch.tensor(np.asarray(x), dtype=torch.float32),
        torch.tensor(np.asarray(sh)).long(),
        torch.tensor(np.asarray(co)).long(),
        sidecar,
    )


def _grid_dims(sh: torch.Tensor, co: torch.Tensor, sidecar: dict | None) -> tuple[int, int]:
    if sidecar is not None and "n_shape" in sidecar and "n_color" in sidecar:
        return int(sidecar["n_shape"]), int(sidecar["n_color"])
    return int(sh.max()) + 1, int(co.max()) + 1


def _standardize_fit(x_train: torch.Tensor):
    return x_train.mean(0), x_train.std(0) + 1e-6


def _cell_index(sh: torch.Tensor, co: torch.Tensor, n_shape: int, n_color: int, rows=None):
    d: dict[tuple[int, int], list[int]] = {}
    it = range(sh.shape[0]) if rows is None else rows.tolist()
    for i in it:
        d.setdefault((int(sh[i]), int(co[i])), []).append(i)
    return {k: torch.tensor(v) for k, v in d.items()}


def _centroids(x, cells, n_shape, n_color, seed, subsample=True):
    g = torch.Generator().manual_seed(seed)
    cent = torch.zeros(n_shape, n_color, x.shape[1])
    for s in range(n_shape):
        for c in range(n_color):
            if (s, c) not in cells:
                return None
            idx = cells[(s, c)]
            pick = idx[torch.randint(0, idx.shape[0], (idx.shape[0],), generator=g)] if subsample else idx
            cent[s, c] = x[pick].mean(0)
    return cent


def _ridge_fit(xa, xb, lam=RIDGE_LAMBDA):
    da = xa.shape[1]
    return torch.linalg.solve(xa.T @ xa + lam * torch.eye(da), xa.T @ xb)


def _rank_truncate(w, k):
    if k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :k] @ torch.diag(s[:k]) @ vh[:k]


def _analogy_top1(x, sh, co, n_shape, n_color, seed, shuffled=False):
    cells = _cell_index(sh, co, n_shape, n_color)
    cent = _centroids(x, cells, n_shape, n_color, seed)
    if cent is None:
        return None
    shape_pairs = [(sx, sy) for sx in range(n_shape) for sy in range(n_shape) if sx != sy]
    hits, total = 0, 0
    for sx, sy in shape_pairs:
        offs = cent[sy, :] - cent[sx, :]  # [n_color, dim]
        for t in range(n_color):
            donors = [c for c in range(n_color) if c != t]
            if shuffled:
                gi = torch.Generator().manual_seed(seed + t + 100 * sx + sy)
                alt = shape_pairs[int(torch.randint(0, len(shape_pairs), (1,), generator=gi))]
                offs_alt = cent[alt[1], :] - cent[alt[0], :]
                transfer = offs_alt[torch.tensor(donors)].mean(0)
            else:
                transfer = offs[torch.tensor(donors)].mean(0)
            pred = cent[sx, t] + transfer
            d = (cent[:, t] - pred).norm(dim=-1)
            hits += int(int(d.argmin()) == sy)
            total += 1
    return hits / total


def ab1_column(x_real, x_rand, sh, co, n_shape, n_color, seeds):
    xr = (x_real - x_real.mean(0)) / (x_real.std(0) + 1e-6)
    xf = (x_rand - x_rand.mean(0)) / (x_rand.std(0) + 1e-6)
    chance = 1.0 / n_shape
    real, shuf, rand = [], [], []
    used = []
    for s in seeds:
        r = _analogy_top1(xr, sh, co, n_shape, n_color, s, shuffled=False)
        sf = _analogy_top1(xr, sh, co, n_shape, n_color, s, shuffled=True)
        rd = _analogy_top1(xf, sh, co, n_shape, n_color, s, shuffled=False)
        if r is None or sf is None or rd is None:
            continue
        real.append(r)
        shuf.append(sf)
        rand.append(rd)
        used.append(s)
    if not real:
        return {"verdict": "no-complete-seed", "seeds_used": []}
    diff = [a - b for a, b in zip(real, rand, strict=True)]
    ci_real, ci_shuf, ci_rand, ci_diff = seed_ci(real), seed_ci(shuf), seed_ci(rand), seed_ci(diff)
    flip = sign_flip_report(diff)
    beats_shuffle = ci_real["lo"] > max(ci_shuf["mean"], chance)
    beats_randinit = ci_diff["lo"] > 0.0 and not flip["any_flip"]
    win = bool(beats_shuffle and beats_randinit)
    verdict = "genuine-abstraction-signal" if win else "random-control-artifact"
    return {
        "verdict": verdict,
        "seeds_used": used,
        "chance": round(chance, 4),
        "analogy_top1_real": ci_real,
        "analogy_top1_shuffled_offset": ci_shuf,
        "analogy_top1_random_init": ci_rand,
        "real_minus_random_init": ci_diff,
        "sign_flips": flip,
        "beats_shuffle_floor": bool(beats_shuffle),
        "beats_random_init_substrate": bool(beats_randinit),
    }


def evaluate_ab1(cache_root: Path, seeds, columns) -> dict:
    grid = []
    skipped = []
    for col in columns:
        real = _load_latents_factors(cache_root, col["real"])
        rand = _load_latents_factors(cache_root, col["randinit"])
        if real is None or rand is None:
            skipped.append({"tag": col["tag"], "reason": "missing real or randinit cache"})
            continue
        xr, sh, co, sidecar = real
        xf, shf, cof, _ = rand
        if col["real"] in DR1_GATED_ARMS and not (sidecar or {}).get("dr1_gate_passed", False):
            skipped.append({"tag": col["tag"], "reason": "DR1 acceptance gate not passed in sidecar"})
            continue
        if xr.shape[0] != xf.shape[0] or not torch.equal(sh, shf) or not torch.equal(co, cof):
            skipped.append({"tag": col["tag"], "reason": "label/count mismatch real vs randinit"})
            continue
        n_shape, n_color = _grid_dims(sh, co, sidecar)
        rep = ab1_column(xr, xf, sh, co, n_shape, n_color, seeds)
        grid.append({**{k: col[k] for k in ("tag", "modality", "family")}, **rep})
        print(f"[AB1 {col['tag']}] verdict={rep.get('verdict')}", flush=True)
    survivors = [c for c in grid if c.get("verdict") == "genuine-abstraction-signal"]
    null_supported = (
        all(c.get("verdict") in ("random-control-artifact", "no-complete-seed") for c in grid)
        if grid
        else None
    )
    return {
        "axis": "AB1_within_encoder_systematicity_analogy",
        "null_hypothesis": "the shape-offset parallelogram does not beat the column's own matched "
        "random-init substrate (abstraction is architecture/geometry-inherited, not learned)",
        "grid": grid,
        "skipped": skipped,
        "survivors": [c["tag"] for c in survivors],
        "survivor_modalities": sorted({c["modality"] for c in survivors}),
        "null_supported": null_supported,
    }


def _cross_encoder_analogy(xa, xb, sh, co, n_shape, n_color, seed, shuffled=False):
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    perm = torch.randperm(n, generator=g)
    cut = int(n * ALIGN_TRAIN_FRAC)
    align_rows, eval_rows = perm[:cut], perm[cut:]

    mua, sda = _standardize_fit(xa[align_rows])
    mub, sdb = _standardize_fit(xb[align_rows])
    za = (xa - mua) / sda
    zb = (xb - mub) / sdb

    w = _rank_truncate(_ridge_fit(za[align_rows], zb[align_rows]), RANK)

    cellsE = _cell_index(sh, co, n_shape, n_color, rows=eval_rows)
    centA = _centroids(za, cellsE, n_shape, n_color, seed)
    centB = _centroids(zb, cellsE, n_shape, n_color, seed)
    if centA is None or centB is None:
        return None

    shape_pairs = [(sx, sy) for sx in range(n_shape) for sy in range(n_shape) if sx != sy]
    hits, total = 0, 0
    for sx, sy in shape_pairs:
        offsA = centA[sy, :] - centA[sx, :]
        for t in range(n_color):
            donors = torch.tensor([c for c in range(n_color) if c != t])
            if shuffled:
                gi = torch.Generator().manual_seed(seed + t + 100 * sx + 7 * sy)
                alt = shape_pairs[int(torch.randint(0, len(shape_pairs), (1,), generator=gi))]
                offsA_use = centA[alt[1], :] - centA[alt[0], :]
            else:
                offsA_use = offsA
            transferA = offsA_use[donors].mean(0)
            predA = centA[sx, t] + transferA
            predB = predA @ w
            d = (centB[:, t] - predB).norm(dim=-1)
            hits += int(int(d.argmin()) == sy)
            total += 1
    return hits / total


def _ab2_directed(xa, xb, sh, co, n_shape, n_color, seeds):
    reals, shufs, used = [], [], []
    for s in seeds:
        r = _cross_encoder_analogy(xa, xb, sh, co, n_shape, n_color, s, shuffled=False)
        sf = _cross_encoder_analogy(xa, xb, sh, co, n_shape, n_color, s, shuffled=True)
        if r is None or sf is None:
            continue
        reals.append(r)
        shufs.append(sf)
        used.append(s)
    return {"real": reals, "shuf": shufs, "seeds_used": used}


def ab2_pair(xa_real, xb_real, xa_rand, xb_rand, sh, co, n_shape, n_color, seeds):
    real = _ab2_directed(xa_real, xb_real, sh, co, n_shape, n_color, seeds)
    rand = _ab2_directed(xa_rand, xb_rand, sh, co, n_shape, n_color, seeds)
    chance = 1.0 / n_shape
    if not real["real"] or not rand["real"]:
        return {"verdict": "no-complete-seed", "seeds_used": []}
    real_ci = seed_ci(real["real"])
    shuf_mean = seed_ci(real["shuf"])["mean"]
    floor = max(shuf_mean, chance)
    matched = sorted(set(real["seeds_used"]) & set(rand["seeds_used"]))
    rmap = dict(zip(real["seeds_used"], real["real"], strict=True))
    cmap = dict(zip(rand["seeds_used"], rand["real"], strict=True))
    deltas = [rmap[s] - cmap[s] for s in matched]
    delta_ci = seed_ci(deltas)
    flip = sign_flip_report(deltas)
    beats_own_floor = real_ci["lo"] > floor
    beats_random_pair = (delta_ci["lo"] > 0) and (not flip["any_flip"])
    win = bool(beats_own_floor and beats_random_pair)
    verdict = "genuine-substrate-invariant-abstraction" if win else "random-control-artifact"
    return {
        "verdict": verdict,
        "chance": round(chance, 4),
        "seeds_used": matched,
        "real_top1": real_ci,
        "shuffled_mean": round(shuf_mean, 4),
        "own_floor": round(floor, 4),
        "random_control_top1": seed_ci(rand["real"]),
        "delta_real_minus_random": delta_ci,
        "sign_flips": flip,
        "beats_own_floor": bool(beats_own_floor),
        "beats_random_pair": bool(beats_random_pair),
    }


def evaluate_ab2(cache_root: Path, seeds, arms) -> dict:
    loaded: dict[str, tuple] = {}
    rand_loaded: dict[str, tuple] = {}
    excluded = []
    for arm in arms:
        got = _load_latents_factors(cache_root, arm)
        if got is None:
            excluded.append({"arm": arm, "reason": "real cache missing"})
            continue
        if arm in DR1_GATED_ARMS and not (got[3] or {}).get("dr1_gate_passed", False):
            excluded.append({"arm": arm, "reason": "DR1 gate not passed"})
            continue
        rname = RANDINIT_FOR.get(arm)
        rgot = _load_latents_factors(cache_root, rname) if rname else None
        if rgot is None:
            excluded.append({"arm": arm, "reason": f"matched random-init {rname} missing"})
            continue
        loaded[arm] = got
        rand_loaded[arm] = rgot

    pairs = {}
    names = sorted(loaded)
    for a in names:
        for b in names:
            if a == b:
                continue
            xa, sh_a, co_a, sc_a = loaded[a]
            xb, sh_b, co_b, _ = loaded[b]
            if xa.shape[0] != xb.shape[0] or not torch.equal(sh_a, sh_b) or not torch.equal(co_a, co_b):
                continue
            ra_name, rb_name = RANDINIT_FOR[a], RANDINIT_FOR[b]
            if ra_name == rb_name:
                pairs[f"{a} -> {b}"] = {
                    "verdict": "within-family-read-only",
                    "note": "same matched random-init substrate (degenerate random<->random control); "
                    "excluded from the win verdict per the three-way within-family rule",
                }
                continue
            xar, sh_ar, co_ar, _ = rand_loaded[a]
            xbr, _, _, _ = rand_loaded[b]
            n_shape, n_color = _grid_dims(sh_a, co_a, sc_a)
            rep = ab2_pair(xa, xb, xar, xbr, sh_a, co_a, n_shape, n_color, seeds)
            pairs[f"{a} -> {b}"] = rep
            print(f"[AB2 {a} -> {b}] verdict={rep.get('verdict')}", flush=True)

    def _mod(arm: str) -> str:
        return ARM_MODALITY.get(arm, "unknown")

    survivors = [
        name for name, p in pairs.items() if p.get("verdict") == "genuine-substrate-invariant-abstraction"
    ]
    survivor_mods = sorted({_mod(name.split(" -> ")[0]) for name in survivors})
    typed = [
        p for p in pairs.values() if p.get("verdict") not in ("within-family-read-only", "no-complete-seed")
    ]
    null_supported = all(p["verdict"] == "random-control-artifact" for p in typed) if typed else None
    return {
        "axis": "AB2_cross_substrate_transfer_substrate_invariance",
        "null_hypothesis": "the shape offset carried through the shared label-free map does not beat the "
        "matched random<->random pair (transfer is geometry the map buys for free, not learned "
        "substrate-invariant abstraction)",
        "arms_present": names,
        "arms_excluded": excluded,
        "pairs": pairs,
        "survivors": survivors,
        "survivor_modalities": survivor_mods,
        "null_supported": null_supported,
    }


def _present_columns(cache_root: Path, columns: list[dict]) -> tuple[list[dict], list[str]]:
    present, missing = [], []
    for col in columns:
        real_ok = (cache_root / col["real"] / "meta.json").exists()
        rand_ok = (cache_root / col["randinit"] / "meta.json").exists()
        if real_ok and rand_ok:
            present.append(col)
        else:
            missing.append(col["tag"])
    return present, missing


def _present_arms(cache_root: Path, arms: list[str]) -> tuple[list[str], list[str]]:
    present = [a for a in arms if (cache_root / a / "meta.json").exists()]
    missing = [a for a in arms if a not in present]
    return present, missing


def atlas_scope(at1, al2, ab1, ab2, full_grid, full_pairs) -> dict:
    at1_survivors = [c for c in at1.get("grid", []) if c.get("verdict") == "genuine-substrate-signal"]
    al2_survivors = [
        name for name, p in al2.get("pairs", {}).items() if p.get("verdict") == "genuine-shared-structure"
    ]
    ab1_survivors = ab1.get("survivors", [])
    ab2_survivors = ab2.get("survivors", [])

    mods = set(c["modality"] for c in at1_survivors)
    mods |= set(ab1.get("survivor_modalities", []))
    mods |= set(ab2.get("survivor_modalities", []))
    survivor_mods = sorted(mods)

    any_survivor = bool(at1_survivors or al2_survivors or ab1_survivors or ab2_survivors)
    if not any_survivor:
        scope = (
            "random-control-artifact (atlas-wide): no registered substrate beat its own control on any axis"
        )
    elif len(survivor_mods) >= 2 and full_grid and full_pairs:
        scope = (
            f"universal across {survivor_mods} on the FULL registered atlas (nuisance, alignment, "
            "within-encoder abstraction, and cross-substrate invariance axes; no missing cache narrowed "
            "the set)"
        )
    elif len(survivor_mods) >= 2:
        scope = (
            f"multi-modality ({survivor_mods}) but the registered set was INCOMPLETE; the universal "
            "claim is withheld until every registered cache is present"
        )
    else:
        mod = survivor_mods[0] if survivor_mods else "alignment-only"
        scope = f"modality-specific ({mod}) on the registered atlas"
    return {
        "scope": scope,
        "at1_survivors": [c["tag"] for c in at1_survivors],
        "al2_survivors": al2_survivors,
        "ab1_survivors": ab1_survivors,
        "ab2_survivors": ab2_survivors,
        "survivor_modalities": survivor_mods,
        "full_registered_grid": full_grid,
        "full_registered_pairs": full_pairs,
    }


def run(cache_root: Path, seeds, abstraction_seeds, allow_partial) -> dict:
    t0 = time.perf_counter()
    present_cols, missing_cols = _present_columns(cache_root, REGISTERED_COLUMNS)
    present_arms, missing_arms = _present_arms(cache_root, REGISTERED_ALIGNMENT_ARMS)
    full_grid = not missing_cols
    full_pairs = not missing_arms
    if (not full_grid or not full_pairs) and not allow_partial:
        raise SystemExit(
            f"registered atlas incomplete (missing columns={missing_cols} arms={missing_arms}); this "
            "at-scale grid refuses to run partial without --allow-partial, so a partial run cannot "
            "masquerade as the at-scale claim. Finish the Studio encodes or pass --allow-partial "
            "(verdict will withhold universal)."
        )
    at1 = evaluate_grid(cache_root, seeds, columns=present_cols, reference_columns=REFERENCE_COLUMNS)
    al2 = evaluate_pairs(cache_root, seeds, arms=present_arms)
    ab1 = evaluate_ab1(cache_root, abstraction_seeds, columns=present_cols)
    ab2 = evaluate_ab2(cache_root, abstraction_seeds, arms=present_arms)
    scope = atlas_scope(at1, al2, ab1, ab2, full_grid, full_pairs)

    nulls = [
        at1.get("null_supported"),
        al2.get("null_supported"),
        ab1.get("null_supported"),
        ab2.get("null_supported"),
    ]
    if all(n is None for n in nulls):
        null_supported = None
        verdict = "NO COLUMNS/PAIRS: nothing on disk to type on any axis"
    else:
        null_supported = all(n in (True, None) for n in nulls)
        verdict = scope["scope"]
    return {
        "experiment": {
            "id": "atlas_multi_encoder_grid",
            "metric": "at-scale AT1 nuisance grid + AL2 alignment grid + AB1 within-encoder abstraction "
            "+ AB2 cross-substrate invariance",
            "baseline": "each column's own matched random-init (AT1, AB1); permuted-pairing rank-k map "
            "(AL2); matched random<->random pair (AB2)",
            "null_hypothesis": (
                "no registered substrate beats its own random-init control (AT1, AB1), no real-encoder "
                "pair beats its permutation floor (AL2), and no real pair beats its random<->random "
                "transfer floor (AB2): the atlas is a random-control/alignment artifact on every axis"
            ),
            "tier": "studio (at-scale, full registered atlas, four axes)",
        },
        "seeds": seeds,
        "abstraction_seeds": abstraction_seeds,
        "registered_columns_present": [c["tag"] for c in present_cols],
        "registered_columns_missing": missing_cols,
        "registered_arms_present": present_arms,
        "registered_arms_missing": missing_arms,
        "full_registered_grid": full_grid,
        "full_registered_pairs": full_pairs,
        "at1_nuisance_grid": at1,
        "al2_alignment_grid": al2,
        "ab1_within_encoder_abstraction": ab1,
        "ab2_cross_substrate_invariance": ab2,
        "atlas_scope": scope,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": verdict,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Atlas at-scale AT1+AL2+AB1+AB2 multi-encoder grid (Studio)")
    ap.add_argument("--seeds", default="0-9", help="AT1/AL2 probe seeds")
    ap.add_argument(
        "--abstraction-seeds",
        default="0-9",
        help="AB1/AB2 abstraction-probe seeds (bootstrap/align-split reseeds)",
    )
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="run with a partial registered set (universal-scope claim is then withheld)",
    )
    ap.add_argument(
        "--skip-ram-guard",
        action="store_true",
        help="bypass the >= 32GB Studio RAM guard (SMOKE-CHECK ONLY; never for a real at-scale run)",
    )
    ap.add_argument("--out", default="runs/mot/atlas_multi_encoder_grid.json")
    a = ap.parse_args(argv)

    if not a.skip_ram_guard:
        assert_studio_ram()  # original full-grid policy guard; partial local runs bypass explicitly
    result = run(
        Path(a.cache_root),
        parse_seeds(a.seeds),
        parse_seeds(a.abstraction_seeds),
        a.allow_partial,
    )
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {k: result[k] for k in ("atlas_scope", "null_supported", "verdict")}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
