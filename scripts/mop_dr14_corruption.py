#!/usr/bin/env python
"""DR14: reasoning under corruption/compression, laptop arms (VQ/low-rank, 4-bit, noise), WP-13.

THESIS: an iterative reasoning primitive degrades with a FLATTER accuracy-vs-corruption slope than
a matched single-pass baseline: iteration recovers corrupted information a passive readout cannot.

NULL (preregistered): reasoning and single-pass degrade at the same rate under EVERY corruption
family (slope difference within SLOPE_MARGIN): iteration only processes what survives corruption.

Arms (both trained on CLEAN latents, both evaluated under IDENTICAL corrupted tensors, corrupted
once and shared, never re-drawn per arm):
  reasoning: IterativeRefiner (weight-tied residual refinement, the shell's canonical laptop
    reasoning primitive from the EX17 line) + linear head.
  single-pass baseline: untied residual blocks at depth_for_matched_flops (equal block count,
    equal forward FLOPs; parameter counts differ by construction and are reported honestly).
The dropped-channel arm is implemented cache-first over a strict citable dense-token cache. It
uses nested deterministic channel-group masks and the exact same materialized view for both arms.
When the natural task cache is absent it fails closed as a skipped environment/data gate, not a
Studio compute gate. Which OTHER primitives survived their own WP rows is read from runs/mot/*.json
at run time and recorded as context only (their
scripts are separate lanes; this script owns the corruption mechanics, not their harnesses).

Corruption families (severity normalized to [0, 1] per family, slopes fit on the degradation
acc_clean - acc vs severity):
  lowrank_vq: project onto the top-r principal components of the CLEAN TRAIN latents (r swept
    down), the low-rank/VQ compression proxy on pooled latents.
  quantize: per-feature uniform quantize_dequantize bit sweep (the shipped 4-bit control).
  noise: additive Gaussian at multiples of the latent std.

Guards (preregistered): noisy-TV guard on the noise arm: on PURE-NOISE inputs (signal destroyed,
marginals matched) BOTH arms must sit within GUARD_TOL of chance, otherwise the eval leaks and no
slope is interpretable. Shuffled-feature floor (cross-feature structure destroyed) reported per
arm. D3 difficulty calibration: the synthetic regime's separation is chosen so the clean linear
probe lands INSIDE [CAL_LO, CAL_HI] (no ceiling, no floor) before any arm is scored.

Data: the verdict is computed on the calibrated synthetic latent stream. The real pooled V-JEPA
cache (data/cache/vjepa2_vitl_fpc64_256_real, count 64) is swept identically and reported as a
pilot replication: 64 samples is below the preregistered MIN_REAL_N power floor, so the real arm
cannot set the verdict either way and is labeled as such.

Writes runs/mot/dr14_corruption.json.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).

Usage: PYTHONPATH=. .venv/bin/python scripts/mop_dr14_corruption.py --seeds 0-2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import depth_for_matched_flops, matched_within, param_count, refiner_flops
from mop.diagnostics.linear_probe import linear_probe
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.diagnostics.substrate_ablation import quantize_dequantize
from mop.experiments.base import Experiment
from mop.seeding import parse_seeds, seed_everything
from mop.shell.predictor import mlp
from mop.shell.refine import IterativeRefiner
from mop.substrate.vjepa21_dense_tasks import DenseTaskError, build_dr14_dense_views

# ------------------------------------------------------------------------------------------------
# preregistered thresholds (in code before any result exists)
# ------------------------------------------------------------------------------------------------
SLOPE_MARGIN = 0.05  # single-pass slope must exceed reasoning slope by this for "flatter"
GUARD_TOL = 0.08  # pure-noise accuracy must sit within this of chance for BOTH arms
CAL_LO, CAL_HI = 0.55, 0.90  # clean linear-probe band certifying a non-ceiling, non-floor regime
MIN_REAL_N = 200  # below this the real-cache sweep is a pilot, never verdict-setting
FLOP_TOL = 0.10
OUT_PATH = Path("runs/mot/dr14_corruption.json")


def default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "seeds": [0, 1, 2],
            "dim": 64,
            "n_classes": 6,
            "n_train": 1400,
            "n_test": 600,
            "sep_grid": [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5],
            "refiner_steps": 4,
            "hidden": 128,
            "epochs": 200,
            "lr": 1e-3,
            "rank_fracs": [1.0, 0.5, 0.25, 0.125, 0.0625],
            "bits": [32, 8, 6, 4, 3, 2],
            "noise_levels": [0.0, 0.25, 0.5, 1.0, 2.0],
            "use_real_cache": True,
            "real_cache_dir": "data/cache/vjepa2_vitl_fpc64_256_real",
            "use_dense_cache": True,
            "dense_cache_dir": "data/cache/vjepa21_vitb_e6_learned",
            "dense_drop_fractions": [0.0, 0.25, 0.5, 0.75],
            "dense_channel_group_width": 16,
            "allow_dense_fixture": False,
        }
    )


# ------------------------------------------------------------------------------------------------
# corruption operators (each applied ONCE per severity, the corrupted tensor shared by both arms)
# ------------------------------------------------------------------------------------------------
def corrupt_lowrank(x: torch.Tensor, train_ref: torch.Tensor, frac: float) -> torch.Tensor:
    """Project x onto the top-r principal directions of the CLEAN TRAIN latents (r = frac of the
    full rank, at least 1). frac >= 1.0 is the identity by construction."""
    if frac >= 1.0:
        return x.clone()
    mu = train_ref.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(train_ref - mu, full_matrices=False)
    r = max(1, int(round(frac * vh.shape[0])))
    v = vh[:r].T  # [D, r]
    return (x - mu) @ v @ v.T + mu


def corrupt_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    return x.clone() if bits >= 32 else quantize_dequantize(x, bits)


def corrupt_noise(x: torch.Tensor, level: float, seed: int) -> torch.Tensor:
    """Additive Gaussian at `level` times the latent std (deterministic given seed)."""
    if level <= 0.0:
        return x.clone()
    g = torch.Generator().manual_seed(seed)
    std = x.std().clamp(min=1e-8)
    return x + level * std * torch.randn(x.shape, generator=g)


def pure_noise_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Signal destroyed, marginal scale matched: the noisy-TV guard input."""
    g = torch.Generator().manual_seed(seed)
    return x.mean() + x.std() * torch.randn(x.shape, generator=g)


def shuffle_features(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Independently permute each feature column across rows (the destroyed-structure floor,
    latent_robustness's control convention)."""
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    for j in range(x.shape[1]):
        out[:, j] = x[torch.randperm(x.shape[0], generator=g), j]
    return out


# ------------------------------------------------------------------------------------------------
# arms
# ------------------------------------------------------------------------------------------------
class ReasoningArm(nn.Module):
    """Weight-tied iterative refinement + linear head: the reasoning primitive under test."""

    def __init__(self, dim: int, hidden: int, steps: int, n_classes: int):
        super().__init__()
        self.refiner = IterativeRefiner(dim, hidden=hidden, steps=steps)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.refiner(x)
        return self.head(z)


class SinglePassArm(nn.Module):
    """Untied residual blocks at matched FLOPs (equal block count, refine.py's stated control:
    iteration vs depth is exactly tied-vs-untied at equal blocks) + linear head."""

    def __init__(self, dim: int, hidden: int, blocks: int, n_classes: int):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(blocks))
        self.blocks = nn.ModuleList(mlp(dim, dim, hidden, depth=1, ln=True) for _ in range(blocks))
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for norm, block in zip(self.norms, self.blocks, strict=True):
            z = z + block(norm(z))
        return self.head(z)


def _train_arm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


@torch.no_grad()
def _acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((model(x).argmax(-1) == y).float().mean())


def fit_slope(severities: list[float], degradations: list[float]) -> float:
    """Least-squares slope of degradation (acc_clean - acc) on normalized severity."""
    s = torch.tensor(severities, dtype=torch.float64)
    d = torch.tensor(degradations, dtype=torch.float64)
    sc, dc = s - s.mean(), d - d.mean()
    denom = float((sc * sc).sum())
    return float((sc * dc).sum() / denom) if denom > 0 else 0.0


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def make_synthetic(
    cfg: DictConfig, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Gaussian class clusters with the separation CALIBRATED into the preregistered non-ceiling
    band [CAL_LO, CAL_HI] via the clean linear probe (D3): a slope on an at-ceiling or at-floor
    regime would be uninterpretable."""
    g = torch.Generator().manual_seed(seed)
    dim, k = int(cfg.dim), int(cfg.n_classes)
    n = int(cfg.n_train) + int(cfg.n_test)
    centers = torch.randn(k, dim, generator=g)
    y = torch.randint(0, k, (n,), generator=g)
    noise = torch.randn(n, dim, generator=g)
    chosen, probe_acc, in_band = None, None, False
    scored = []
    for sep in [float(s) for s in cfg.sep_grid]:
        x = centers[y] * sep + noise
        acc = linear_probe(x, y, epochs=100, seed=seed)["score"]
        scored.append((sep, round(acc, 4)))
        if CAL_LO <= acc <= CAL_HI:
            chosen, probe_acc, in_band = sep, acc, True
            break
    if chosen is None:  # fall back to the grid point closest to the band center, flagged
        chosen, probe_acc = min(scored, key=lambda t: abs(t[1] - (CAL_LO + CAL_HI) / 2))
    x = centers[y] * chosen + noise
    cut = int(cfg.n_train)
    cal = {"separation": chosen, "clean_probe_acc": probe_acc, "in_band": in_band, "grid": scored}
    return x[:cut], y[:cut], x[cut:], y[cut:], cal


def load_real_cache(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        import numpy as np

        x = torch.from_numpy(np.load(cache_dir / "latents.npy")).float()
        y = torch.from_numpy(np.load(cache_dir / "labels.npy")).long()
        return x, y
    except Exception:
        return None


def run_dense_drop_sweep(cfg: DictConfig, cache_dir: Path, seed: int) -> dict:
    """Run the registered dense dropped-channel arm from one strict shared-view receipt."""

    try:
        bundle = build_dr14_dense_views(
            cache_dir,
            fractions=tuple(float(value) for value in cfg.dense_drop_fractions),
            seed=seed,
            group_width=int(cfg.dense_channel_group_width),
            strict_run_identity=not bool(cfg.get("allow_dense_fixture", False)),
        )
    except (DenseTaskError, FileNotFoundError, OSError, ValueError) as exc:
        return {
            "skipped": str(exc),
            "cache": str(cache_dir),
            "verdict_setting": False,
            "scientific_promotion": False,
        }

    splits = bundle["splits"]
    train_indices = torch.tensor([int(index) for index in splits.get("train", [])])
    test_indices = torch.tensor([int(index) for index in splits.get("test", [])])
    if train_indices.numel() < 2 or test_indices.numel() < 2:
        return {
            "skipped": "dense cache train/test splits are too small",
            "cache": str(cache_dir),
            "verdict_setting": False,
            "scientific_promotion": False,
        }
    labels = torch.from_numpy(bundle["labels"]).long()
    ytr, yte = labels[train_indices], labels[test_indices]
    if labels.numel() == 0 or set(labels.tolist()) != set(range(int(labels.max()) + 1)):
        return {
            "skipped": "dense cache labels must be contiguous nonnegative integers",
            "cache": str(cache_dir),
            "verdict_setting": False,
            "scientific_promotion": False,
        }
    if not set(yte.tolist()) <= set(ytr.tolist()):
        return {
            "skipped": "dense cache test labels are absent from training",
            "cache": str(cache_dir),
            "verdict_setting": False,
            "scientific_promotion": False,
        }

    keys = [f"{float(value):.6f}" for value in bundle["fractions"]]
    clean = torch.from_numpy(bundle["views"][keys[0]]).float()
    xtr = clean[train_indices]
    dim = int(clean.shape[1])
    classes = int(labels.max()) + 1
    hidden, steps = int(cfg.hidden), int(cfg.refiner_steps)
    blocks = depth_for_matched_flops(dim, hidden, steps)
    seed_everything(seed)
    reasoning = ReasoningArm(dim, hidden, steps, classes)
    single = SinglePassArm(dim, hidden, blocks, classes)
    _train_arm(reasoning, xtr, ytr, int(cfg.epochs), float(cfg.lr))
    _train_arm(single, xtr, ytr, int(cfg.epochs), float(cfg.lr))

    acc_reasoning: list[float] = []
    acc_single: list[float] = []
    for key in keys:
        shared_test = torch.from_numpy(bundle["views"][key]).float()[test_indices]
        acc_reasoning.append(round(_acc(reasoning, shared_test, yte), 4))
        acc_single.append(round(_acc(single, shared_test, yte), 4))
    severities = [float(value) for value in bundle["fractions"]]
    clean_reasoning, clean_single = acc_reasoning[0], acc_single[0]
    slope_reasoning = fit_slope(severities, [clean_reasoning - value for value in acc_reasoning])
    slope_single = fit_slope(severities, [clean_single - value for value in acc_single])
    r_flops = refiner_flops(dim, hidden, steps)
    s_flops = refiner_flops(dim, hidden, blocks)
    view_receipt = {
        key: value
        for key, value in bundle.items()
        if key not in {"views", "labels", "splits", "source_row_sha256s"}
    }
    return {
        "cache": str(cache_dir),
        "n": int(bundle["count"]),
        "train_n": int(train_indices.numel()),
        "test_n": int(test_indices.numel()),
        "levels": severities,
        "severity": severities,
        "acc_reasoning": acc_reasoning,
        "acc_single_pass": acc_single,
        "slope_reasoning": round(slope_reasoning, 4),
        "slope_single_pass": round(slope_single, 4),
        "slope_diff": round(slope_single - slope_reasoning, 4),
        "reasoning_flatter": bool(slope_single - slope_reasoning > SLOPE_MARGIN),
        "shared_view_receipt": view_receipt,
        "compute": {
            "refiner_blocks": steps,
            "single_pass_blocks": blocks,
            "forward_flops": {"reasoning": r_flops, "single_pass": s_flops},
            "matched": matched_within(r_flops, s_flops, tol=FLOP_TOL),
            "params": {
                "reasoning": param_count(reasoning),
                "single_pass": param_count(single),
            },
        },
        "verdict_setting": False,
        "scientific_promotion": False,
        "claim_boundary": (
            "registered mechanics are runnable; a rights-clean powered natural task cache and "
            "independent statistical verification are still required"
        ),
    }


# ------------------------------------------------------------------------------------------------
# the sweep (one train/test split, both arms, all families, identical corrupted tensors)
# ------------------------------------------------------------------------------------------------
def run_sweep(
    cfg: DictConfig, xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor, yte: torch.Tensor, seed: int
) -> dict:
    seed_everything(seed)
    dim = xtr.shape[1]
    k = int(ytr.max()) + 1
    hidden, steps = int(cfg.hidden), int(cfg.refiner_steps)
    blocks = depth_for_matched_flops(dim, hidden, steps)

    reason = ReasoningArm(dim, hidden, steps, k)
    single = SinglePassArm(dim, hidden, blocks, k)
    _train_arm(reason, xtr, ytr, int(cfg.epochs), float(cfg.lr))
    _train_arm(single, xtr, ytr, int(cfg.epochs), float(cfg.lr))

    clean_r, clean_s = _acc(reason, xte, yte), _acc(single, xte, yte)
    chance = 1.0 / k

    families: dict[str, list[tuple[str, float | int]]] = {
        "lowrank_vq": [("frac", float(f)) for f in cfg.rank_fracs],
        "quantize": [("bits", int(b)) for b in cfg.bits],
        "noise": [("level", float(lv)) for lv in cfg.noise_levels],
    }
    out_fams = {}
    for fam, levels in families.items():
        sev, acc_r, acc_s, lvl_vals = [], [], [], []
        for i, (_, lv) in enumerate(levels):
            if fam == "lowrank_vq":
                xc = corrupt_lowrank(xte, xtr, float(lv))
            elif fam == "quantize":
                xc = corrupt_quantize(xte, int(lv))
            else:
                xc = corrupt_noise(xte, float(lv), seed=seed * 1000 + i)
            sev.append(i / max(1, len(levels) - 1))
            lvl_vals.append(lv)
            acc_r.append(round(_acc(reason, xc, yte), 4))  # the SAME xc scores both arms
            acc_s.append(round(_acc(single, xc, yte), 4))
        slope_r = fit_slope(sev, [clean_r - a for a in acc_r])
        slope_s = fit_slope(sev, [clean_s - a for a in acc_s])
        out_fams[fam] = {
            "levels": lvl_vals,
            "severity": [round(s, 4) for s in sev],
            "acc_reasoning": acc_r,
            "acc_single_pass": acc_s,
            "slope_reasoning": round(slope_r, 4),
            "slope_single_pass": round(slope_s, 4),
            "slope_diff": round(slope_s - slope_r, 4),  # positive = reasoning degrades flatter
            "reasoning_flatter": bool(slope_s - slope_r > SLOPE_MARGIN),
        }

    # noisy-TV guard (noise arm): pure-noise inputs must put BOTH arms at chance
    xn = pure_noise_like(xte, seed=seed + 77)
    guard_r, guard_s = _acc(reason, xn, yte), _acc(single, xn, yte)
    guard_ok = bool(abs(guard_r - chance) <= GUARD_TOL and abs(guard_s - chance) <= GUARD_TOL)
    # destroyed-structure floor (reported, the latent_robustness convention)
    xs = shuffle_features(xte, seed=seed + 78)
    floor_r, floor_s = _acc(reason, xs, yte), _acc(single, xs, yte)

    r_flops = refiner_flops(dim, hidden, steps)
    s_flops = refiner_flops(dim, hidden, blocks)
    return {
        "clean_acc": {"reasoning": round(clean_r, 4), "single_pass": round(clean_s, 4)},
        "chance": round(chance, 4),
        "families": out_fams,
        "noisy_tv_guard": {
            "pure_noise_acc": {"reasoning": round(guard_r, 4), "single_pass": round(guard_s, 4)},
            "guard_ok": guard_ok,
        },
        "shuffled_feature_floor": {"reasoning": round(floor_r, 4), "single_pass": round(floor_s, 4)},
        "compute": {
            "refiner_blocks": steps,
            "single_pass_blocks": blocks,
            "forward_flops": {"reasoning": r_flops, "single_pass": s_flops},
            "matched": matched_within(r_flops, s_flops, tol=FLOP_TOL),
            "params": {"reasoning": param_count(reason), "single_pass": param_count(single)},
            "param_note": "FLOP-matched by equal block count; untied blocks carry more params, "
            "reported, the refine.py iteration-vs-depth control convention",
        },
    }


def _upstream_context() -> dict:
    """Which reasoning-primitive rows have verdicts on disk (context only, never rebuilt here)."""
    ctx = {}
    mot = Path("runs/mot")
    if mot.exists():
        for p in sorted(mot.glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(d, dict) and "survives" in d:
                ctx[p.stem] = {"survives": d.get("survives"), "null_supported": d.get("null_supported")}
    return ctx


def _run_seed(cfg: DictConfig, seed: int) -> dict:
    xtr, ytr, xte, yte, cal = make_synthetic(cfg, seed)
    synth = run_sweep(cfg, xtr, ytr, xte, yte, seed)
    real: dict = {"skipped": "use_real_cache=False"}
    if bool(cfg.use_real_cache):
        loaded = load_real_cache(Path(str(cfg.real_cache_dir)))
        if loaded is None:
            real = {"skipped": f"cache not found at {cfg.real_cache_dir}"}
        else:
            x, y = loaded
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(x.shape[0], generator=g)
            cut = int(0.7 * x.shape[0])
            real = {
                "n": int(x.shape[0]),
                "verdict_setting": bool(x.shape[0] >= MIN_REAL_N),
                "note": "pilot replication, below MIN_REAL_N power floor"
                if x.shape[0] < MIN_REAL_N
                else "at power",
                "sweep": run_sweep(cfg, x[perm[:cut]], y[perm[:cut]], x[perm[cut:]], y[perm[cut:]], seed),
            }
    dense: dict = {"skipped": "use_dense_cache=False", "verdict_setting": False}
    if bool(cfg.use_dense_cache):
        dense = run_dense_drop_sweep(cfg, Path(str(cfg.dense_cache_dir)), seed)
    return {
        "seed": seed,
        "calibration": cal,
        "synthetic": synth,
        "real_cache_pilot": real,
        "dense_drop_pilot": dense,
    }


class DR14Corruption(Experiment):
    id = "mop_dr14_corruption"
    metric = (
        "accuracy_vs_corruption_slope",
        "dense_drop_slope",
        "slope_diff_per_family",
        "noisy_tv_guard",
        "scientific_promotion",
    )
    baseline = "single-pass untied residual net at matched forward FLOPs under IDENTICAL corruption"
    ablation = "corruption family sweep (low-rank/VQ, quantize, noise, and nested dense-channel drop)"
    null_hypothesis = (
        "Reasoning and single-pass degrade at the same rate under every corruption family (slope "
        "difference within SLOPE_MARGIN): iteration only processes what survives corruption."
    )
    tier = "env-later"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        t0 = time.time()
        per_seed = [_run_seed(cfg, int(s)) for s in cfg.seeds]
        n_seeds = len(per_seed)
        fams = list(per_seed[0]["synthetic"]["families"].keys())
        per_family = {}
        any_family_survives = False
        for fam in fams:
            diffs = [r["synthetic"]["families"][fam]["slope_diff"] for r in per_seed]
            n_flat = sum(1 for r in per_seed if r["synthetic"]["families"][fam]["reasoning_flatter"])
            fam_survives = n_flat == n_seeds
            any_family_survives = any_family_survives or fam_survives
            per_family[fam] = {
                "slope_diff_ci": seed_ci(diffs),
                "slope_diff_sign_flips": sign_flip_report(diffs),
                "n_seeds_reasoning_flatter": n_flat,
                "family_survives": fam_survives,
            }
        n_guard = sum(1 for r in per_seed if r["synthetic"]["noisy_tv_guard"]["guard_ok"])
        n_cal = sum(1 for r in per_seed if r["calibration"]["in_band"])
        survives = any_family_survives and n_guard == n_seeds and n_cal == n_seeds
        out = {
            "contract": self.contract(),
            "preregistered": {
                "slope_margin": SLOPE_MARGIN,
                "guard_tol": GUARD_TOL,
                "calibration_band": [CAL_LO, CAL_HI],
                "min_real_n": MIN_REAL_N,
                "flop_tol": FLOP_TOL,
                "survival_rule": "at least one corruption family shows the reasoning arm flatter by "
                "SLOPE_MARGIN on EVERY seed, the noisy-TV guard holds on every "
                "seed, and every seed's regime is calibrated in-band; the real "
                "cache and dense dropped-channel cache are pilots and never set the verdict",
            },
            "cfg": OmegaConf.to_container(cfg),
            "upstream_verdicts_present": _upstream_context(),
            "per_seed": per_seed,
            "per_family": per_family,
            "n_seeds_guard_ok": n_guard,
            "n_seeds_calibrated": n_cal,
            "n_seeds": n_seeds,
            "survives": survives,
            "null_supported": not survives,
            "scientific_promotion": False,
            "claim_boundary": {
                "synthetic_result_is_mechanics_only": True,
                "pooled_real_pilot_sets_verdict": False,
                "dense_drop_pilot_sets_verdict": False,
                "registered_dense_drop_interface_implemented": True,
            },
            "wall_clock_s": round(time.time() - t0, 3),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dr14_corruption.json").write_text(json.dumps(out, indent=2))
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description="DR14 corruption/compression laptop arms")
    ap.add_argument("--seeds", type=str, default="0-2")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    ap.add_argument("--dense-cache", type=str)
    args = ap.parse_args()
    cfg = default_cfg()
    cfg.seeds = parse_seeds(args.seeds)
    if args.dense_cache:
        cfg.use_dense_cache = True
        cfg.dense_cache_dir = args.dense_cache
    out_path = Path(args.out)
    result = DR14Corruption().run(cfg, resolve("cpu"), out_path.parent)
    written = out_path.parent / "dr14_corruption.json"
    if out_path != written:
        written.rename(out_path)
    print(f"[dr14] wrote {out_path}", file=sys.stderr)
    print(json.dumps({k: result[k] for k in ("survives", "null_supported", "per_family")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
