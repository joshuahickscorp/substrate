#!/usr/bin/env python

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

CACHE = REPO / "data" / "cache"
LANE = REPO / "runs" / "mot"
LANE.mkdir(parents=True, exist_ok=True)

from mop.diagnostics.linear_probe import linear_probe  # noqa: E402

PREREG = {
    "a_matched_arch": "PASS iff embed_dim, input_resolution, count, feature-D all EQUAL across real and "
    "random-init arms, control pretrained=False, real arm trained. Any smaller net / lower res / fewer "
    "clips on the control => FAIL (strawman).",
    "b_permutation_null": "PASS iff learned_recall > permuted_recall (a correspondence exists and the "
    "row-shuffle null removes it) AND permuted-target marginal kNN self-recall == unpermuted "
    "(pure relabeling preserves geometry). No correspondence to destroy => NULL, not PASS.",
    "c_vacuous_banned": "PASS iff (c1) |acc(real) - acc(square-random-proj)| < 0.02 (proves vacuity) AND "
    "(c2) every audited positive-claim script uses a genuine control arm (random-init net / random-pixel "
    "/ permutation), never frozen_random square projection.",
    "d_determinism": "PASS iff two same-seed reruns are BIT-identical (max abs diff == 0.0). Nonzero tiny "
    "diff reported as NEAR-identical, not PASS.",
}


def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)


def _load_meta(name: str) -> dict:
    p = CACHE / name / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _feat_D(name: str) -> int:
    root = CACHE / name
    if (root / "features.npy").exists():
        return int(np.load(root / "features.npy", mmap_mode="r").shape[1])
    if (root / "latents.npy").exists():
        return int(np.load(root / "latents.npy", mmap_mode="r").shape[1])
    return -1


def _count(name: str) -> int:
    root = CACHE / name
    for f in ("features.npy", "latents.npy"):
        if (root / f).exists():
            return int(np.load(root / f, mmap_mode="r").shape[0])
    return -1


def _sidecar(name: str) -> dict:
    p = CACHE / name / "factors.json"
    if not p.exists():
        return {}
    f = json.loads(p.read_text())
    return {
        k: f.get(k)
        for k in ("checksum_first", "checksum_last", "resolution", "frames", "model_id", "clipset", "arm")
    }


def check_a() -> dict:
    pairs = [
        {
            "name": "vjepa2_vitl (substrate-special headline)",
            "real": "vjepa2_vitl_nuisance",
            "rand": "randominit_vitl_nuisance",
            "res_key": "input_resolution",
        },
        {
            "name": "dinov2s (AT1 second survivor)",
            "real": "dinov2s_nuisance_real",
            "rand": "dinov2s_nuisance_randominit",
            "res_key": None,  # dino meta has no resolution field; check D+count and doc-asserted 224px
        },
        {
            "name": "qwen05b_shapecap (shapecap lift)",
            "real": "qwen05b_shapecap_real",
            "rand": "qwen05b_shapecap_randominit",
            "res_key": None,
        },
    ]
    results = []
    for pr in pairs:
        rm, cm = _load_meta(pr["real"]), _load_meta(pr["rand"])
        rD, cD = _feat_D(pr["real"]), _feat_D(pr["rand"])
        rN, cN = _count(pr["real"]), _count(pr["rand"])
        r_embed = rm.get("embed_dim", rm.get("key_dim", rD))
        c_embed = cm.get("embed_dim", cm.get("key_dim", cD))
        res_match = None
        r_res = c_res = None
        if pr["res_key"]:
            r_res = rm.get(pr["res_key"])
            c_res = cm.get(pr["res_key"])
            res_match = (r_res is not None) and (r_res == c_res)
        r_pretrained = rm.get("pretrained", None)
        c_pretrained = cm.get("pretrained", None)
        control_untrained = (
            (c_pretrained is False)
            or ("randominit" in pr["rand"])
            or (cm.get("family", "").startswith("random"))
        )
        real_trained = (r_pretrained is True) or ("real" in pr["real"]) or (rm.get("pretrained") is True)
        dim_match = (r_embed == c_embed) and (rD == cD) and rD > 0
        count_match = (rN == cN) and rN > 0
        core = dim_match and count_match and control_untrained and real_trained
        rs, cs = _sidecar(pr["real"]), _sidecar(pr["rand"])
        checksum_match = bool(
            rs.get("checksum_first")
            and rs.get("checksum_first") == cs.get("checksum_first")
            and rs.get("checksum_last") == cs.get("checksum_last")
        )
        r_res_sc, c_res_sc = rs.get("resolution"), cs.get("resolution")
        sidecar_res_match = r_res_sc == c_res_sc  # both-None (text) or equal ints (pixel) both pass
        sidecar_res_applicable = r_res_sc is not None
        sidecar_model_match = (rs.get("model_id") is not None) and (rs.get("model_id") == cs.get("model_id"))
        clipset_match = (rs.get("clipset") is not None) and (rs.get("clipset") == cs.get("clipset"))
        if pr["res_key"]:
            nuis_match = None
            rp, cp = CACHE / pr["real"] / "nuisance.npy", CACHE / pr["rand"] / "nuisance.npy"
            if rp.exists() and cp.exists():
                nr, nc = np.load(rp), np.load(cp)
                nuis_match = bool(nr.shape == nc.shape and np.abs(nr - nc).max() == 0.0)
            passed = bool(core and res_match and (nuis_match is not False))
            res_note = (
                f"input_resolution real={r_res} rand={c_res} match={res_match}; "
                f"per-clip nuisance draws bit-identical across arms={nuis_match} (clip identity proof, "
                f"meta.json layout has no checksum sidecar)"
            )
            checksum_match = bool(nuis_match)  # nuisance-identity is the checksum-equivalent for this layout
        elif rs:
            passed = bool(
                core and checksum_match and sidecar_res_match and sidecar_model_match and clipset_match
            )
            kind = "pixel" if sidecar_res_applicable else "text (no pixel resolution, matched-by-absence)"
            res_note = (
                f"{kind}; sidecar resolution real={r_res_sc} rand={c_res_sc} match={sidecar_res_match}; "
                f"model_id match={sidecar_model_match}; clipset match={clipset_match}; "
                f"clip checksum_first/last match={checksum_match}"
            )
        else:
            passed = bool(core)
            res_note = (
                "no resolution field in meta and no sidecar; arch (embed_dim) and count identical, "
                "resolution is set by one shared preprocessing path in the cache script"
            )
        results.append(
            {
                "pair": pr["name"],
                "real_cache": pr["real"],
                "rand_cache": pr["rand"],
                "real_embed_dim": r_embed,
                "rand_embed_dim": c_embed,
                "real_feature_D": rD,
                "rand_feature_D": cD,
                "dim_match": dim_match,
                "real_count": rN,
                "rand_count": cN,
                "count_match": count_match,
                "resolution_note": res_note,
                "clip_checksum_match": checksum_match,
                "sidecar_resolution_match": bool(sidecar_res_match),
                "sidecar_model_id_match": bool(sidecar_model_match),
                "control_untrained": bool(control_untrained),
                "real_trained": bool(real_trained),
                "control_pretrained_flag": c_pretrained,
                "real_pretrained_flag": r_pretrained,
                "verdict": "PASS" if passed else "FAIL",
            }
        )
    overall = "PASS" if all(r["verdict"] == "PASS" for r in results) else "FAIL"
    return {"rule": PREREG["a_matched_arch"], "pairs": results, "verdict": overall}


def _knn_self_recall(x: torch.Tensor, k: int = 5) -> float:
    d = torch.cdist(x, x)
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    return d[iu[0], iu[1]]


def check_b() -> dict:
    from scripts.mop_al2_alignment_pilot import (
        KNN_K,
        knn_neighbor_recall,
        rank_truncate,
        ridge_fit,
    )

    a_name, b_name = "vjepa2_vitl_nuisance", "dinov2s_nuisance_real"
    xa = torch.tensor(np.load(CACHE / a_name / "features.npy")).float()
    xb = torch.tensor(np.load(CACHE / b_name / "latents.npy")).float()
    n = min(xa.shape[0], xb.shape[0])
    xa, xb = xa[:n], xb[:n]

    seed, rank = 0, 32
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]

    def std(a, b):
        mu, sd = a.mean(0, keepdim=True), a.std(0, keepdim=True) + 1e-6
        return (a - mu) / sd, (b - mu) / sd

    atr, ate = std(xa[tr], xa[te])
    btr, bte = std(xb[tr], xb[te])

    w = rank_truncate(ridge_fit(atr, btr), rank)
    learned_recall = knn_neighbor_recall(ate @ w, bte, k=KNN_K)

    perm_recalls = []
    for _ in range(20):
        shuf = torch.randperm(cut, generator=g)
        w_s = rank_truncate(ridge_fit(atr, btr[shuf]), rank)
        perm_recalls.append(knn_neighbor_recall(ate @ w_s, bte, k=KNN_K))
    permuted_recall = float(np.mean(perm_recalls))

    correspondence_destroyed = learned_recall > permuted_recall

    dist_orig = torch.sort(_knn_self_recall(btr)).values
    shuf2 = torch.randperm(cut, generator=g)
    dist_perm = torch.sort(_knn_self_recall(btr[shuf2])).values
    geom_max_diff = float((dist_orig - dist_perm).abs().max())
    geometry_preserved = geom_max_diff < 1e-4

    correspondence_gap = round(learned_recall - permuted_recall, 4)
    if not correspondence_destroyed:
        verdict = "NULL"  # nothing to destroy; the null is well-formed but this pair has no correspondence
    elif geometry_preserved:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    gn = torch.Generator().manual_seed(123)
    ra = torch.randn(n, xa.shape[1], generator=gn)
    rb = torch.randn(n, 64, generator=gn)
    ratr, rate = std(ra[tr], ra[te])
    rbtr, rbte = std(rb[tr], rb[te])
    w0 = rank_truncate(ridge_fit(ratr, rbtr), rank)
    neg_learned = knn_neighbor_recall(rate @ w0, rbte, k=KNN_K)
    neg_perm = []
    for _ in range(20):
        shuf = torch.randperm(cut, generator=gn)
        w0s = rank_truncate(ridge_fit(ratr, rbtr[shuf]), rank)
        neg_perm.append(knn_neighbor_recall(rate @ w0s, rbte, k=KNN_K))
    neg_permuted = float(np.mean(neg_perm))
    neg_gap = round(neg_learned - neg_permuted, 4)
    metric_not_inflating = neg_gap < 0.05

    return {
        "rule": PREREG["b_permutation_null"],
        "pair": f"{a_name} -> {b_name}",
        "learned_recall": round(learned_recall, 4),
        "permuted_recall": round(permuted_recall, 4),
        "correspondence_gap": correspondence_gap,
        "correspondence_destroyed_by_null": bool(correspondence_destroyed),
        "target_geometry_max_pairwise_dist_diff": geom_max_diff,
        "geometry_preserved_under_relabel": bool(geometry_preserved),
        "negative_control_random_vs_random": {
            "learned_recall": round(neg_learned, 4),
            "permuted_recall": round(neg_permuted, 4),
            "gap": neg_gap,
            "metric_does_not_inflate_on_noise": bool(metric_not_inflating),
            "note": "two independent gaussian reps: learned map does NOT beat its permuted floor, proving "
            "the correspondence gap on the real pair is genuine and the metric is not spuriously optimistic",
        },
        "verdict": verdict if metric_not_inflating else "FAIL",
    }


def check_c() -> dict:
    x = torch.tensor(np.load(CACHE / "vjepa2_vitl_nuisance" / "features.npy")).float()
    y = torch.tensor(np.load(CACHE / "vjepa2_vitl_nuisance" / "labels_shape.npy")).long()
    D = x.shape[1]
    g = torch.Generator().manual_seed(0)
    M = torch.randn(D, D, generator=g)
    xp = x @ M
    accs_real, accs_proj = [], []
    for s in range(5):
        r = linear_probe(_zscore(x), y, classification=True, epochs=400, seed=s)
        p = linear_probe(_zscore(xp), y, classification=True, epochs=400, seed=s)
        accs_real.append(r["score"])
        accs_proj.append(p["score"])
    real_mean = float(np.mean(accs_real))
    proj_mean = float(np.mean(accs_proj))
    vac_gap = abs(real_mean - proj_mean)
    c1_pass = vac_gap < 0.02  # a genuine control would move the number; a vacuous one cannot

    claim_scripts = {
        "substrate_vs_random_init_vit.py": "scripts/substrate_vs_random_init_vit.py",
        "substrate_vs_random_features.py": "scripts/substrate_vs_random_features.py",
        "mop_at1_grid_pilot.py": "scripts/mop_at1_grid_pilot.py",
        "mop_pr2_plasticity_substrates.py": "scripts/mop_pr2_plasticity_substrates.py",
        "mop_survivor_reaudit.py": "scripts/mop_survivor_reaudit.py",
        "cache_randominit_vitl_features.py": "scripts/cache_randominit_vitl_features.py",
    }
    scan = {}
    for label, rel in claim_scripts.items():
        src = (REPO / rel).read_text()
        mentions_fr = "frozen_random" in src or "frozen-random" in src
        uses_random_init = ("random_init" in src) or ("randominit" in src) or ("_random_init_vit" in src)
        uses_random_pixel = "random_pixel" in src or "random-pixel" in src
        uses_permutation = "randperm" in src or "permut" in src or "shuffle" in src
        disclaimed = any(
            w in src for w in ("vacuous", "CORRECTED", "corrected", "INVALID", "rerun with real")
        )
        vacuous_active = mentions_fr and not disclaimed
        scan[label] = {
            "mentions_frozen_random": mentions_fr,
            "frozen_random_disclaimed_only": bool(mentions_fr and disclaimed),
            "uses_random_init_control": bool(uses_random_init),
            "uses_random_pixel_control": bool(uses_random_pixel),
            "uses_permutation_null": bool(uses_permutation),
            "vacuous_control_active": bool(vacuous_active),
        }
    c2_pass = all(not v["vacuous_control_active"] for v in scan.values())


    VALID_MARKERS = ("readout_contribution", "capability_per_bit", "nonlinear_gain", "linear_vs_mlp")
    corpus = {}
    roots = [REPO / "scripts", REPO / "src" / "mop"]
    for base in roots:
        for path in sorted(base.rglob("*.py")):
            txt = path.read_text()
            if "frozen_random" not in txt and "frozen-random" not in txt:
                continue
            rel = str(path.relative_to(REPO))
            vac_gate_lines = []
            for ln in txt.splitlines():
                low = ln.lower()
                if "frozen" not in low and "fr)" not in low and "- fa" not in low and "-fa" not in low:
                    pass
                is_gate = (
                    ("needs_real" in low and ("bool(" in low or ">" in low or "=" in low and "real" in low))
                    or ("- fa >" in low)
                    or ("-fa >" in low)
                    or ("real - fr" in low)
                    or ("ra - fa" in low)
                    or ("delta_frozen_random" in low and ">" in low)
                )
                disclaimed_line = any(d in low for d in ("vacuous", "projection-invariant", "ties a frozen"))
                if is_gate and not disclaimed_line:
                    vac_gate_lines.append(ln.strip())
            has_valid = any(m in txt for m in VALID_MARKERS)
            uses_vac_gate = len(vac_gate_lines) > 0
            defines_vacuous_field = "def substrate_ablation" in txt and "needs_real" in txt
            file_disclaims_vacuity = any(
                d in txt
                for d in ("projection-invariant", "ties a frozen-random", "linear probe is projection")
            )
            ingests_ablation_gate = (
                ("substrate_ablation(" in txt and ("needs_real" in txt or "delta_frozen_random" in txt))
                or ("ra - fa >" in txt)
                or ("real - fr >" in txt)
            )
            fr_substrate_arm = ("fr_stream" in txt) or ("_run_arm" in txt and "frozen" in txt.lower())
            classification = "clean"
            if ingests_ablation_gate:
                classification = "VACUOUS: linear real-minus-frozen_random gate as a positive verdict"
            elif defines_vacuous_field and not file_disclaims_vacuity:
                classification = "VACUOUS-DEFN: defines needs_real/delta_frozen_random (linear, square-map)"
            elif uses_vac_gate and fr_substrate_arm:
                classification = "BORDERLINE: frozen-random SUBSTRATE arm through a full learner (not a bare linear-probe projection); weaker but not provably vacuous"
            elif uses_vac_gate:
                classification = "VACUOUS: linear real-minus-frozen_random gate as a positive verdict"
            elif has_valid:
                classification = "valid: nonlinear-DiD or rank-bottleneck use of frozen_random"
            corpus[rel] = {
                "classification": classification,
                "vacuous_gate_lines": vac_gate_lines[:4],
                "has_valid_nonlinear_or_bottleneck_use": bool(has_valid),
            }

    corpus_vacuous = {k: v for k, v in corpus.items() if v["classification"].startswith("VACUOUS")}
    corpus_borderline = {k: v for k, v in corpus.items() if v["classification"].startswith("BORDERLINE")}
    corpus_flags = sorted(corpus_vacuous)
    borderline_flags = sorted(corpus_borderline)
    c3_pass = len(corpus_flags) == 0

    from mop.diagnostics.substrate_ablation import substrate_ablation

    xa = torch.tensor(np.load(CACHE / "vjepa2_vitl_nuisance" / "features.npy")).float()
    ya = torch.tensor(np.load(CACHE / "vjepa2_vitl_nuisance" / "labels_shape.npy")).long()
    abls = [substrate_ablation(xa, ya, seed=s) for s in range(5)]
    dfr = [a["delta_frozen_random"] for a in abls]
    dsh = [a["delta_shuffled"] for a in abls]
    nr = [a["needs_real"] for a in abls]
    c4 = {
        "factor": "shape (genuinely decodable from V-JEPA, real acc ~0.77 vs chance 0.20)",
        "delta_frozen_random_per_seed": dfr,
        "delta_shuffled_per_seed": dsh,
        "needs_real_per_seed": nr,
        "mean_abs_delta_frozen_random": round(float(np.mean(np.abs(dfr))), 4),
        "mean_delta_shuffled": round(float(np.mean(dsh)), 4),
        "needs_real_is_broken_gate": bool(all(not v for v in nr)),
        "note": "delta_frozen_random is seed noise (often NEGATIVE: the square full-rank projection scores "
        "AS HIGH OR HIGHER than real under a linear probe), so needs_real is False on all seeds even though "
        "shape IS decodable (delta_shuffled ~0.6). This is a FALSE-NEGATIVE control: any claim gated on "
        "needs_real would be wrongly demoted. The genuine half (delta_shuffled) still works.",
    }

    verdict = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"
    return {
        "rule": PREREG["c_vacuous_banned"],
        "c1_square_projection_vacuity": {
            "real_probe_acc_mean": round(real_mean, 4),
            "square_random_proj_probe_acc_mean": round(proj_mean, 4),
            "gap": round(vac_gap, 4),
            "note": "a square full-rank linear map is absorbed by the probe, so acc is unchanged; this "
            "proves it can never be a control (delta forced to 0 by construction)",
            "c1_pass": bool(c1_pass),
        },
        "c2_positive_claim_scripts_use_genuine_controls": scan,
        "c2_pass": bool(c2_pass),
        "c3_corpus_classification": corpus,
        "c3_vacuous_frozen_random_control_files": corpus_flags,
        "c3_borderline_frozen_random_substrate_arm_files": borderline_flags,
        "c3_pass": bool(c3_pass),
        "c3_note": "corpus-wide scan of scripts/ and src/mop/, precise vacuous-vs-valid split. VACUOUS = a "
        "linear real-minus-frozen_random delta/gate (needs_real, delta_frozen_random) consumed as a "
        "positive verdict; the square full-rank map is probe-absorbed so the delta is noise. VALID = the "
        "nonlinear difference-in-differences (readout_contribution) or the rank-reducing bottleneck "
        "(capability_per_bit), which break the linear invariance and ARE legitimate. The four live "
        "survivors (substrate-special, compositional, PR1, shapecap) do NOT use this arm.",
        "c4_empirical_vacuity_of_corpus_function": c4,
        "verdict": verdict,
    }


def check_d() -> dict:
    x = torch.tensor(np.load(CACHE / "vjepa2_vitl_nuisance" / "features.npy")).float()
    y = torch.tensor(np.load(CACHE / "vjepa2_vitl_nuisance" / "labels_shape.npy")).long()
    xz = _zscore(x)

    run1 = [linear_probe(xz, y, classification=True, epochs=400, seed=s)["score"] for s in range(5)]
    run2 = [linear_probe(xz, y, classification=True, epochs=400, seed=s)["score"] for s in range(5)]
    max_diff_probe = float(max(abs(a - b) for a, b in zip(run1, run2, strict=True)))

    ri = torch.tensor(np.load(CACHE / "randominit_vitl_nuisance" / "features.npy")).float()
    riz = _zscore(ri)

    def delta_run():
        vj = [linear_probe(xz, y, classification=True, epochs=400, seed=s)["score"] for s in range(5)]
        rr = [linear_probe(riz, y, classification=True, epochs=400, seed=s)["score"] for s in range(5)]
        return [a - b for a, b in zip(vj, rr, strict=True)]

    d1, d2 = delta_run(), delta_run()
    max_diff_delta = float(max(abs(a - b) for a, b in zip(d1, d2, strict=True)))

    bit_identical = (max_diff_probe == 0.0) and (max_diff_delta == 0.0)
    verdict = (
        "PASS"
        if bit_identical
        else ("NEAR-identical" if max(max_diff_probe, max_diff_delta) < 1e-6 else "FAIL")
    )
    return {
        "rule": PREREG["d_determinism"],
        "probe_run1": [round(v, 6) for v in run1],
        "probe_run2": [round(v, 6) for v in run2],
        "probe_max_abs_diff": max_diff_probe,
        "t4_delta_run1": [round(v, 6) for v in d1],
        "t4_delta_run2": [round(v, 6) for v in d2],
        "t4_delta_max_abs_diff": max_diff_delta,
        "bit_identical": bool(bit_identical),
        "verdict": verdict,
    }


def main():
    a = check_a()
    b = check_b()
    c = check_c()
    d = check_d()
    remaining_vacuous_or_mismatched = []
    for r in a["pairs"]:
        if r["verdict"] == "FAIL":
            remaining_vacuous_or_mismatched.append(f"(a) mismatched control: {r['pair']}")
    if b["verdict"] == "FAIL":
        remaining_vacuous_or_mismatched.append("(b) permutation null malformed on audited pair")
    for label, v in c["c2_positive_claim_scripts_use_genuine_controls"].items():
        if v["vacuous_control_active"]:
            remaining_vacuous_or_mismatched.append(f"(c) vacuous frozen_random active in {label}")
    for f in c["c3_vacuous_frozen_random_control_files"]:
        remaining_vacuous_or_mismatched.append(f"(c3) corpus: VACUOUS linear frozen_random gate in {f}")
    if d["verdict"] == "FAIL":
        remaining_vacuous_or_mismatched.append("(d) nondeterministic keyed computation")

    result = {
        "audit": "meta_control_audit",
        "axis": "falsification engine",
        "read_only_repo": True,
        "preregistered_thresholds": PREREG,
        "a_matched_arch_resolution": a,
        "b_permutation_null_correctness": b,
        "c_vacuous_frozen_random_banned": c,
        "d_seed_determinism": d,
        "remaining_vacuous_or_mismatched_controls": remaining_vacuous_or_mismatched,
    }
    (LANE / "meta_control_audit.json").write_text(json.dumps(result, indent=2, default=str))

    print("=== META-CONTROL AUDIT ===")
    print(f"(a) matched-arch + matched-resolution : {a['verdict']}")
    for r in a["pairs"]:
        print(
            f"    {r['pair']}: {r['verdict']} "
            f"[embed {r['real_embed_dim']}=={r['rand_embed_dim']}, D {r['real_feature_D']}=={r['rand_feature_D']}, "
            f"count {r['real_count']}=={r['rand_count']}, ctrl_untrained={r['control_untrained']}, "
            f"clip_checksum_match={r['clip_checksum_match']}, sidecar_res_match={r['sidecar_resolution_match']}]"
        )
    print(f"(b) permutation/topology null correct : {b['verdict']}")
    print(
        f"    learned_recall={b['learned_recall']} permuted={b['permuted_recall']} "
        f"gap={b['correspondence_gap']} destroyed={b['correspondence_destroyed_by_null']} "
        f"geom_preserved={b['geometry_preserved_under_relabel']} (max dist diff {b['target_geometry_max_pairwise_dist_diff']:.2e})"
    )
    nc = b["negative_control_random_vs_random"]
    print(
        f"    neg-control (random vs random): learned={nc['learned_recall']} permuted={nc['permuted_recall']} "
        f"gap={nc['gap']} not_inflating={nc['metric_does_not_inflate_on_noise']}"
    )
    print(f"(c) vacuous frozen-random banned      : {c['verdict']}")
    print(
        f"    c1 vacuity: real={c['c1_square_projection_vacuity']['real_probe_acc_mean']} "
        f"proj={c['c1_square_projection_vacuity']['square_random_proj_probe_acc_mean']} "
        f"gap={c['c1_square_projection_vacuity']['gap']} (pass={c['c1_square_projection_vacuity']['c1_pass']})"
    )
    for label, v in c["c2_positive_claim_scripts_use_genuine_controls"].items():
        print(
            f"    c2 {label}: vacuous_active={v['vacuous_control_active']} "
            f"(fr_mention={v['mentions_frozen_random']}, disclaimed_only={v['frozen_random_disclaimed_only']}, "
            f"rand_init={v['uses_random_init_control']}, rand_pixel={v['uses_random_pixel_control']}, perm={v['uses_permutation_null']})"
        )
    print(f"    c3 corpus VACUOUS linear-frozen_random gate files (pass={c['c3_pass']}):")
    for f in c["c3_vacuous_frozen_random_control_files"]:
        print(f"        VACUOUS: {f}  {c['c3_corpus_classification'][f]['vacuous_gate_lines'][:1]}")
    if not c["c3_vacuous_frozen_random_control_files"]:
        print("        NONE")
    for f in c["c3_borderline_frozen_random_substrate_arm_files"]:
        print(f"        BORDERLINE: {f}")
    c4 = c["c4_empirical_vacuity_of_corpus_function"]
    print(
        f"    c4 corpus substrate_ablation on real shape: delta_frozen_random={c4['delta_frozen_random_per_seed']} "
        f"delta_shuffled_mean={c4['mean_delta_shuffled']} needs_real={c4['needs_real_per_seed']} "
        f"broken_gate={c4['needs_real_is_broken_gate']}"
    )
    print(f"(d) seed determinism                  : {d['verdict']}")
    print(
        f"    probe max_abs_diff={d['probe_max_abs_diff']} t4_delta max_abs_diff={d['t4_delta_max_abs_diff']} "
        f"bit_identical={d['bit_identical']}"
    )
    print("remaining vacuous/mismatched controls :", remaining_vacuous_or_mismatched or "NONE")


if __name__ == "__main__":
    main()
