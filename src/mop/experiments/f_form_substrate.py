"""Series F: form-substrate experiments, one level above V-JEPA-shaped video latents.

These are cpu-now toy scaffolds for the user's "perfect substrate / any data" direction. They do not
train an encoder and do not claim a brain. They test whether arbitrary observation forms can be put behind
one referent-aligned interface, whether cross-form transfer needs real alignment, and whether the shared
form bottleneck itself is load-bearing.

F1 form alignment gate: can paired referents align one form to another better than raw or shuffled anchors.
F2 held-out form transfer: can a concept learned through several forms transfer to an unseen form.
F3 bottleneck capacity: does a too-small canonical form collapse capability relative to a wider form.
F5 cross-form memory binding: can memory retrieve the same referent through a different form.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.performance_density import density_block
from ..seeding import seed_everything
from ..substrate.form import (
    FormMeta,
    TensorFormAdapter,
    apply_affine_alignment,
    build_form_matrix,
    fit_affine_alignment,
    form_audit,
)
from .base import Experiment, _mean

FORM_KINDS = ("vision", "audio", "symbolic", "timeseries")


def _int(e: DictConfig, name: str, default: int) -> int:
    return int(getattr(e, name, default))


def _float(e: DictConfig, name: str, default: float) -> float:
    return float(getattr(e, name, default))


def _str(e: DictConfig, name: str, default: str) -> str:
    return str(getattr(e, name, default))


def _balanced_world(
    *,
    samples: int,
    classes: int,
    world_dim: int,
    separation: float,
    noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(samples) % classes
    y = y[torch.randperm(samples, generator=g)]
    centers = torch.randn(classes, world_dim, generator=g) * separation
    z = centers[y] + noise * torch.randn(samples, world_dim, generator=g)
    return z.float(), y.long()


def _form_features(
    z: torch.Tensor,
    *,
    feature_dim: int,
    kinds: tuple[str, ...] = FORM_KINDS,
    noise: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for i, kind in enumerate(kinds):
        g = torch.Generator().manual_seed(seed + 1009 * (i + 1))
        w = torch.randn(z.shape[1], feature_dim, generator=g) / math.sqrt(z.shape[1])
        bias = 0.15 * torch.randn(feature_dim, generator=g)
        form_noise = noise * torch.randn(z.shape[0], feature_dim, generator=g)
        x = z @ w + bias + form_noise
        if kind == "symbolic":
            x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1.0e-6)
        elif kind == "timeseries":
            x = torch.roll(x, shifts=1, dims=1)
        out[kind] = x.float()
    return out


def _referents(n: int) -> list[str]:
    return [f"r{i:04d}" for i in range(n)]


def _matrix(features: dict[str, torch.Tensor], y: torch.Tensor) -> dict:
    refs = _referents(y.shape[0])
    adapters = []
    for kind, x in sorted(features.items()):
        meta = FormMeta(
            tag=kind,
            kind=kind,
            feature_dim=int(x.shape[1]),
            source="synthetic-form-world",
            objective="handcrafted",
        )
        adapters.append(TensorFormAdapter(meta, x, refs, factors={"class": y}))
    return form_audit(build_form_matrix(adapters), require_controls=False)


def _split(n: int, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + 811)
    perm = torch.randperm(n, generator=g)
    cut = int(n * train_frac)
    return perm[:cut], perm[cut:]


def _fit_head(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    classes: int,
    epochs: int,
    lr: float,
    seed: int,
) -> nn.Linear:
    seed_everything(seed)
    head = nn.Linear(x.shape[1], classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(x), y).backward()
        opt.step()
    return head


def _acc(head: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((head(x).argmax(-1) == y).float().mean())


def _aligned_forms(
    features: dict[str, torch.Tensor],
    train_idx: torch.Tensor,
    *,
    reference: str = "vision",
) -> dict[str, torch.Tensor]:
    ref = features[reference]
    aligned = {reference: ref}
    for kind, x in features.items():
        if kind == reference:
            continue
        w = fit_affine_alignment(x[train_idx], ref[train_idx])
        aligned[kind] = apply_affine_alignment(x, w)
    return aligned


class F1(Experiment):
    id = "f1_form_alignment_gate"
    metric = ("aligned_transfer", "raw_transfer", "shuffled_anchor_transfer")
    baseline = "source-form head tested directly on the target form without alignment"
    ablation = "paired-referent affine alignment vs shuffled-referent alignment"
    null_hypothesis = (
        "paired referent alignment ties raw transfer or shuffled-anchor alignment, so the form interface "
        "is just a coordinate relabeling and not a usable cross-form bridge"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        aligned, raw, shuffled, source = [], [], [], []
        head_params = 0
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 220),
                classes=_int(e, "classes", 4),
                world_dim=_int(e, "world_dim", 24),
                separation=_float(e, "separation", 2.0),
                noise=_float(e, "world_noise", 0.6),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 32),
                noise=_float(e, "form_noise", 0.08),
                seed=s,
            )
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.55), s)
            src, tgt = _str(e, "source_form", "vision"), _str(e, "target_form", "audio")
            head = _fit_head(
                forms[src][tr],
                y[tr],
                classes=_int(e, "classes", 4),
                epochs=_int(e, "epochs", 80),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            head_params = sum(p.numel() for p in head.parameters())
            source.append(_acc(head, forms[src][te], y[te]))
            raw.append(_acc(head, forms[tgt][te], y[te]))
            w = fit_affine_alignment(forms[tgt][tr], forms[src][tr])
            aligned.append(_acc(head, apply_affine_alignment(forms[tgt][te], w), y[te]))
            g = torch.Generator().manual_seed(s + 515)
            shuffled_src = forms[src][tr[torch.randperm(tr.shape[0], generator=g)]]
            w_shuf = fit_affine_alignment(forms[tgt][tr], shuffled_src)
            shuffled.append(_acc(head, apply_affine_alignment(forms[tgt][te], w_shuf), y[te]))
        best_control = max(_mean(raw), _mean(shuffled))
        gain = _mean(aligned) - best_control
        return {
            "source_form": _str(e, "source_form", "vision"),
            "target_form": _str(e, "target_form", "audio"),
            "source_acc": round(_mean(source), 4),
            "raw_transfer": round(_mean(raw), 4),
            "aligned_transfer": round(_mean(aligned), 4),
            "shuffled_anchor_transfer": round(_mean(shuffled), 4),
            "aligned_gain_over_best_control": round(gain, 4),
            "seeds": seeds,
            "null_supported": bool(gain <= _float(e, "margin", 0.05)),
            "density": density_block(
                {"aligned_transfer": _mean(aligned)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


class F2(Experiment):
    id = "f2_heldout_form_transfer"
    metric = ("heldout_form_acc", "single_form_baseline", "multi_form_gain")
    baseline = "single reference-form head, with the held-out form aligned by unlabeled referents"
    ablation = "train the head on several aligned forms vs only the reference form"
    null_hypothesis = (
        "multi-form training ties the single-form baseline on a held-out observation family, or the "
        "held-out family stays near chance after alignment"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        heldout = _str(e, "heldout_form", "timeseries")
        multi_acc, single_acc, audits = [], [], []
        head_params = 0
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 240),
                classes=_int(e, "classes", 4),
                world_dim=_int(e, "world_dim", 28),
                separation=_float(e, "separation", 1.8),
                noise=_float(e, "world_noise", 0.7),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 36),
                noise=_float(e, "form_noise", 0.12),
                seed=s,
            )
            audits.append(_matrix(forms, y))
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.55), s)
            aligned = _aligned_forms(forms, tr)
            train_forms = [k for k in FORM_KINDS if k != heldout]
            x_multi = torch.cat([aligned[k][tr] for k in train_forms], dim=0)
            y_multi = torch.cat([y[tr] for _ in train_forms], dim=0)
            head_multi = _fit_head(
                x_multi,
                y_multi,
                classes=_int(e, "classes", 4),
                epochs=_int(e, "epochs", 80),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            head_single = _fit_head(
                aligned["vision"][tr],
                y[tr],
                classes=_int(e, "classes", 4),
                epochs=_int(e, "epochs", 80),
                lr=_float(e, "lr", 0.03),
                seed=s + 91,
            )
            head_params = sum(p.numel() for p in head_multi.parameters())
            multi_acc.append(_acc(head_multi, aligned[heldout][te], y[te]))
            single_acc.append(_acc(head_single, aligned[heldout][te], y[te]))
        chance = 1.0 / _int(e, "classes", 4)
        gain = _mean(multi_acc) - _mean(single_acc)
        return {
            "heldout_form": heldout,
            "heldout_form_acc": round(_mean(multi_acc), 4),
            "single_form_baseline": round(_mean(single_acc), 4),
            "multi_form_gain": round(gain, 4),
            "chance": round(chance, 4),
            "audit_all_ok": bool(all(a["all_ok"] for a in audits)),
            "seeds": seeds,
            "null_supported": bool(gain <= _float(e, "margin", 0.03) or _mean(multi_acc) <= chance + 0.1),
            "density": density_block(
                {"heldout_form_acc": _mean(multi_acc)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


class F3(Experiment):
    id = "f3_form_bottleneck_capacity"
    metric = ("wide_form_acc", "small_form_acc", "wide_minus_small")
    baseline = "small canonical bottleneck and shuffled-label floor"
    ablation = "wide canonical form bottleneck vs small bottleneck at matched data and head"
    null_hypothesis = (
        "the wide canonical bottleneck ties the small bottleneck, so interface width is not the bound, "
        "or both sit near the shuffled-label floor"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        wide_acc, small_acc, shuffle_acc = [], [], []
        head_params = 0
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 260),
                classes=_int(e, "classes", 6),
                world_dim=_int(e, "world_dim", 32),
                separation=_float(e, "separation", 1.5),
                noise=_float(e, "world_noise", 0.9),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 40),
                noise=_float(e, "form_noise", 0.1),
                seed=s,
            )
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.55), s)
            aligned = _aligned_forms(forms, tr)
            x_train = torch.cat([aligned[k][tr] for k in FORM_KINDS], dim=0)
            y_train = torch.cat([y[tr] for _ in FORM_KINDS], dim=0)
            x_test = torch.cat([aligned[k][te] for k in FORM_KINDS], dim=0)
            y_test = torch.cat([y[te] for _ in FORM_KINDS], dim=0)

            def _project(
                width: int,
                offset: int,
                *,
                seed: int = s,
                train: torch.Tensor = x_train,
                test: torch.Tensor = x_test,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                g = torch.Generator().manual_seed(seed + offset)
                p = torch.randn(train.shape[1], width, generator=g) / math.sqrt(train.shape[1])
                return train @ p, test @ p

            xw, xw_te = _project(_int(e, "wide_dim", 18), 700)
            xs, xs_te = _project(_int(e, "small_dim", 2), 701)
            hw = _fit_head(
                xw,
                y_train,
                classes=_int(e, "classes", 6),
                epochs=_int(e, "epochs", 90),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            hs = _fit_head(
                xs,
                y_train,
                classes=_int(e, "classes", 6),
                epochs=_int(e, "epochs", 90),
                lr=_float(e, "lr", 0.03),
                seed=s + 1,
            )
            g = torch.Generator().manual_seed(s + 912)
            y_shuf = y_train[torch.randperm(y_train.shape[0], generator=g)]
            hf = _fit_head(
                xw,
                y_shuf,
                classes=_int(e, "classes", 6),
                epochs=_int(e, "epochs", 90),
                lr=_float(e, "lr", 0.03),
                seed=s + 2,
            )
            head_params = sum(p.numel() for p in hw.parameters())
            wide_acc.append(_acc(hw, xw_te, y_test))
            small_acc.append(_acc(hs, xs_te, y_test))
            shuffle_acc.append(_acc(hf, xw_te, y_test))
        gain = _mean(wide_acc) - _mean(small_acc)
        chance = 1.0 / _int(e, "classes", 6)
        return {
            "wide_dim": _int(e, "wide_dim", 18),
            "small_dim": _int(e, "small_dim", 2),
            "wide_form_acc": round(_mean(wide_acc), 4),
            "small_form_acc": round(_mean(small_acc), 4),
            "shuffle_floor_acc": round(_mean(shuffle_acc), 4),
            "wide_minus_small": round(gain, 4),
            "chance": round(chance, 4),
            "seeds": seeds,
            "null_supported": bool(
                gain <= _float(e, "margin", 0.05) or _mean(wide_acc) <= max(chance, _mean(shuffle_acc)) + 0.1
            ),
            "density": density_block(
                {"wide_form_acc": _mean(wide_acc)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


class F5(Experiment):
    id = "f5_cross_form_memory_binding"
    metric = ("cross_form_recall_at_1", "raw_recall_at_1", "shuffled_anchor_recall_at_1")
    baseline = "same memory queried through the target form without alignment"
    ablation = "paired-referent alignment vs shuffled-referent alignment before nearest-neighbor retrieval"
    null_hypothesis = (
        "cross-form retrieval ties per-form nearest neighbor or shuffled referents, so memory is "
        "form-local rather than referent-bound"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        aligned, raw, shuffled, local = [], [], [], []
        store_form = _str(e, "store_form", "vision")
        query_form = _str(e, "query_form", "audio")
        store_bytes = 0
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 220),
                classes=_int(e, "classes", 5),
                world_dim=_int(e, "world_dim", 28),
                separation=_float(e, "separation", 1.7),
                noise=_float(e, "world_noise", 0.7),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 36),
                noise=_float(e, "form_noise", 0.08),
                seed=s,
            )
            tr, te = _split(y.shape[0], _float(e, "anchor_frac", 0.45), s)
            store = forms[store_form]
            query = forms[query_form]

            def _recall(
                q: torch.Tensor,
                *,
                test_idx: torch.Tensor = te,
                store_memory: torch.Tensor = store,
            ) -> float:
                pred = torch.cdist(q[test_idx], store_memory).argmin(dim=1)
                return float((pred == test_idx).float().mean())

            store_bytes = store.element_size() * store.nelement()
            raw.append(_recall(query))
            w = fit_affine_alignment(query[tr], store[tr])
            aligned.append(_recall(apply_affine_alignment(query, w)))
            g = torch.Generator().manual_seed(s + 619)
            shuf_store = store[tr[torch.randperm(tr.shape[0], generator=g)]]
            w_shuf = fit_affine_alignment(query[tr], shuf_store)
            shuffled.append(_recall(apply_affine_alignment(query, w_shuf)))
            local.append(_recall(store))
        best_control = max(_mean(raw), _mean(shuffled))
        gain = _mean(aligned) - best_control
        return {
            "store_form": store_form,
            "query_form": query_form,
            "cross_form_recall_at_1": round(_mean(aligned), 4),
            "raw_recall_at_1": round(_mean(raw), 4),
            "shuffled_anchor_recall_at_1": round(_mean(shuffled), 4),
            "same_form_local_recall_at_1": round(_mean(local), 4),
            "aligned_gain_over_best_control": round(gain, 4),
            "seeds": seeds,
            "null_supported": bool(gain <= _float(e, "margin", 0.05)),
            "density": density_block(
                {"cross_form_recall_at_1": _mean(aligned)},
                seconds=time.perf_counter() - t0,
                bytes=float(store_bytes),
            ),
        }


def _hetero_payloads(
    z: torch.Tensor,
    *,
    dims: dict[str, int],
    noise: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Per-form RAW payloads of DIFFERENT dimensionality (heterogeneous ad hoc featurizers).

    Unlike `_form_features` (a common width), each form gets its own raw dim and its own random
    geometry, so a naive flatten cannot line them up. This is the bed for F4: the question is whether
    a canonical, referent-aligned token layer beats these raw payloads and handcrafted per-form stats.
    """
    out: dict[str, torch.Tensor] = {}
    for i, (kind, d) in enumerate(sorted(dims.items())):
        g = torch.Generator().manual_seed(seed + 2003 * (i + 1))
        w = torch.randn(z.shape[1], d, generator=g) / math.sqrt(z.shape[1])
        bias = 0.15 * torch.randn(d, generator=g)
        x = z @ w + bias + noise * torch.randn(z.shape[0], d, generator=g)
        if kind == "symbolic":
            x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1.0e-6)
        out[kind] = x.float()
    return out


def _to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad with zeros or truncate a raw payload to a fixed width (the honest raw-featurizer control)."""
    n, d = x.shape
    if d == dim:
        return x
    if d > dim:
        return x[:, :dim]
    return torch.cat([x, torch.zeros(n, dim - d)], dim=1)


def _handcrafted(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-form pooled statistics (mean, std, min, max) tiled to a fixed width."""
    stats = torch.stack([x.mean(1), x.std(1), x.amin(1), x.amax(1)], dim=1)  # [N, 4]
    reps = (dim + 3) // 4
    return stats.repeat(1, reps)[:, :dim]


class F4(Experiment):
    id = "f4_raw_payload_vs_form_tokens"
    metric = ("cross_form_transfer_per_dim", "retention_per_dim", "control_delta")
    baseline = "raw zero-padded payloads and handcrafted per-form statistics at matched dimension"
    ablation = "canonical referent-aligned form tokens vs raw and handcrafted featurizers"
    null_hypothesis = (
        "canonical form tokens tie raw flattened or handcrafted per-form features on cross-form "
        "transfer, so the form-token layer is ceremony over arbitrary tensors"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 4)
        dim = _int(e, "matched_dim", 16)
        reference = _str(e, "reference_form", "vision")
        dims = {
            "vision": _int(e, "vision_dim", 20),
            "audio": _int(e, "audio_dim", 12),
            "symbolic": _int(e, "symbolic_dim", 28),
            "timeseries": _int(e, "timeseries_dim", 8),
        }
        canon_cross, raw_cross, hand_cross, canon_same = [], [], [], []
        t0 = time.perf_counter()
        head_params = 0
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 240),
                classes=classes,
                world_dim=_int(e, "world_dim", 24),
                separation=_float(e, "separation", 1.8),
                noise=_float(e, "world_noise", 0.7),
                seed=s,
            )
            payloads = _hetero_payloads(z, dims=dims, noise=_float(e, "form_noise", 0.1), seed=s)
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.55), s)

            # three encoders to the SAME matched dim, evaluated on the same head-fit protocol
            def _encode(
                kind: str,
                mode: str,
                *,
                pay: dict[str, torch.Tensor] = payloads,
                train_idx: torch.Tensor = tr,
            ) -> torch.Tensor:
                x = pay[kind]
                if mode == "raw":
                    return _to_dim(x, dim)
                if mode == "handcrafted":
                    return _handcrafted(x, dim)
                # canonical: affine-align each form into the reference form's dim on paired referents
                ref = _to_dim(pay[reference], dim)
                if kind == reference:
                    return ref
                w = fit_affine_alignment(_to_dim(x, dim)[train_idx], ref[train_idx])
                return apply_affine_alignment(_to_dim(x, dim), w)

            others = [k for k in dims if k != reference]
            per_mode = {}
            for mode in ("canonical", "raw", "handcrafted"):
                ref_repr = _encode(reference, mode)
                head = _fit_head(
                    ref_repr[tr],
                    y[tr],
                    classes=classes,
                    epochs=_int(e, "epochs", 80),
                    lr=_float(e, "lr", 0.03),
                    seed=s,
                )
                head_params = sum(p.numel() for p in head.parameters())
                cross = _mean([_acc(head, _encode(k, mode)[te], y[te]) for k in others])
                per_mode[mode] = (cross, _acc(head, ref_repr[te], y[te]))
            canon_cross.append(per_mode["canonical"][0])
            raw_cross.append(per_mode["raw"][0])
            hand_cross.append(per_mode["handcrafted"][0])
            canon_same.append(per_mode["canonical"][1])
        best_control = max(_mean(raw_cross), _mean(hand_cross))
        delta = _mean(canon_cross) - best_control
        return {
            "reference_form": reference,
            "matched_dim": dim,
            "canonical_cross_form_acc": round(_mean(canon_cross), 4),
            "raw_cross_form_acc": round(_mean(raw_cross), 4),
            "handcrafted_cross_form_acc": round(_mean(hand_cross), 4),
            "cross_form_transfer_per_dim": round(_mean(canon_cross) / dim, 6),
            "retention_per_dim": round(_mean(canon_same) / dim, 6),
            "control_delta": round(delta, 4),
            "seeds": seeds,
            "null_supported": bool(delta <= _float(e, "margin", 0.05)),
            "density": density_block(
                {"canonical_cross_form_acc": _mean(canon_cross)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


class F17(Experiment):
    id = "f17_missing_form_recovery"
    metric = ("recovery_acc", "best_remaining_form_acc", "absence_ece", "recovery_per_extra_flop")
    baseline = "best remaining single form and impute-by-mean when one form is absent at test time"
    ablation = "mean-fuse the remaining aligned forms vs substitute the missing form with its train mean"
    null_hypothesis = (
        "recovery ties the best remaining single form, or confidence does not predict correctness "
        "under a missing form, so the forms are redundant channels and the monitor is uninformative"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.compute import mlp_flops
        from ..diagnostics.operational_awareness import (
            confidence_calibration,
            missing_form_detection,
        )

        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 4)
        recov, best_rem, impute, conf_full, conf_absent = [], [], [], [], []
        det_scores, det_absent, cal_conf, cal_correct = [], [], [], []
        head_params, feat_dim = 0, 0
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 260),
                classes=classes,
                world_dim=_int(e, "world_dim", 26),
                separation=_float(e, "separation", 1.8),
                noise=_float(e, "world_noise", 0.7),
                seed=s,
            )
            forms = _form_features(
                z, feature_dim=_int(e, "feature_dim", 32), noise=_float(e, "form_noise", 0.1), seed=s
            )
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.55), s)
            aligned = _aligned_forms(forms, tr)  # all arms in the reference form's space
            kinds = list(FORM_KINDS)
            feat_dim = aligned[kinds[0]].shape[1]
            train_means = {k: aligned[k][tr].mean(0, keepdim=True) for k in kinds}

            def _fuse(mats: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(mats, 0).mean(0)

            head = _fit_head(
                _fuse([aligned[k][tr] for k in kinds]),
                y[tr],
                classes=classes,
                epochs=_int(e, "epochs", 90),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            head_params = sum(p.numel() for p in head.parameters())

            def _conf(x: torch.Tensor, *, h: nn.Module = head) -> torch.Tensor:
                with torch.no_grad():
                    return torch.softmax(h(x), -1).max(-1).values

            full_test = _fuse([aligned[k][te] for k in kinds])
            conf_full.append(float(_conf(full_test).mean()))

            for missing in kinds:  # drop each form in turn
                remaining = [k for k in kinds if k != missing]
                recov_test = _fuse([aligned[k][te] for k in remaining])
                recov.append(_acc(head, recov_test, y[te]))
                best_rem.append(max(_acc(head, aligned[k][te], y[te]) for k in remaining))
                imp = _fuse(
                    [aligned[k][te] for k in remaining] + [train_means[missing].expand(te.shape[0], -1)]
                )
                impute.append(_acc(head, imp, y[te]))
                c = _conf(recov_test)
                conf_absent.append(float(c.mean()))
                cal_conf.extend(c.tolist())
                with torch.no_grad():
                    cal_correct.extend((head(recov_test).argmax(-1) == y[te]).float().tolist())

            # OA1 missing-form detection: corrupt one form per test sample, rank it by consensus residual
            g = torch.Generator().manual_seed(s + 4201)
            corrupt = torch.randint(0, len(kinds), (te.shape[0],), generator=g)
            stacked = torch.stack([aligned[k][te] for k in kinds], 1)  # [N, K, D]
            for j in range(len(kinds)):
                mask = corrupt == j
                stacked[mask, j] += 3.0 * torch.randn(int(mask.sum()), feat_dim, generator=g)
            consensus = stacked.mean(1, keepdim=True)
            residual = (stacked - consensus).pow(2).mean(-1)  # [N, K]
            for j in range(len(kinds)):
                det_scores.extend(residual[:, j].tolist())
                det_absent.extend((corrupt == j).float().tolist())

        recovery_acc = _mean(recov)
        head_flops = max(1, mlp_flops([feat_dim, classes]))
        oa1 = missing_form_detection(det_scores, det_absent)
        oa2 = confidence_calibration(cal_conf, cal_correct)
        # OA2: confidence is informative iff it PREDICTS correctness under absence (AUROC over chance).
        # Raw confidence magnitude is a poor proxy here (a head trained on 4-form fusion sees a
        # different scale under 3 forms), so the null clause tests calibration, not a magnitude drop.
        confidence_informative = oa2["auroc"] > 0.5 + _float(e, "cal_margin", 0.03)
        recovery_gain = recovery_acc - _mean(best_rem)
        return {
            "recovery_acc": round(recovery_acc, 4),
            "best_remaining_form_acc": round(_mean(best_rem), 4),
            "impute_by_mean_acc": round(_mean(impute), 4),
            "absence_ece": round(oa2["ece"], 4),
            "recovery_per_extra_flop": round(recovery_acc / head_flops, 9),
            "recovery_gain_over_best_remaining": round(recovery_gain, 4),
            "confidence_full": round(_mean(conf_full), 4),
            "confidence_under_absence": round(_mean(conf_absent), 4),
            "confidence_predicts_correctness": bool(confidence_informative),
            "oa1_missing_form_auroc": round(oa1["auroc"], 4),
            "oa2_calibration_auroc": round(oa2["auroc"], 4),
            "seeds": seeds,
            "null_supported": bool(recovery_gain <= _float(e, "margin", 0.02) or not confidence_informative),
            "density": density_block(
                {"recovery_acc": recovery_acc},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


def _two_factor_forms(
    *,
    samples: int,
    n_a: int,
    n_b: int,
    world_dim: int,
    separation: float,
    noise: float,
    form_noise: float,
    cross_leak: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two independent factors a and b, exposed asymmetrically across two forms.

    Form A encodes a strongly and b weakly (attenuated by `cross_leak`); form B encodes b strongly and
    a weakly. Compositional binding across forms means reading a from form A and b from form B over the
    same referent, so a held-out (a, b) combination is decodable even though the pair was never trained.
    Returns (form_a, form_b, a_labels, b_labels).
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, n_a, (samples,), generator=g)
    b = torch.randint(0, n_b, (samples,), generator=g)
    ca = torch.randn(n_a, world_dim, generator=g) * separation
    cb = torch.randn(n_b, world_dim, generator=g) * separation
    za = ca[a] + noise * torch.randn(samples, world_dim, generator=g)
    zb = cb[b] + noise * torch.randn(samples, world_dim, generator=g)
    wa = torch.randn(2 * world_dim, world_dim, generator=g) / math.sqrt(2 * world_dim)
    wb = torch.randn(2 * world_dim, world_dim, generator=g) / math.sqrt(2 * world_dim)
    form_a = torch.cat([za, cross_leak * zb], 1) @ wa + form_noise * torch.randn(
        samples, world_dim, generator=g
    )
    form_b = torch.cat([cross_leak * za, zb], 1) @ wb + form_noise * torch.randn(
        samples, world_dim, generator=g
    )
    return form_a.float(), form_b.float(), a.long(), b.long()


class F9(Experiment):
    id = "f9_cross_form_compositional_binding"
    metric = ("heldout_combo_acc", "seen_combo_acc", "heldout_seen_gap")
    baseline = "shuffled-label floor and a composite-label head that can only memorize seen conjunctions"
    ablation = "factored two-head decode (a from form A, b from form B) vs composite-conjunction head"
    null_hypothesis = (
        "held-out cross-form combinations collapse toward the shuffle floor while seen pairs stay high, "
        "so the system memorized form-specific conjunctions instead of binding factors across forms"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_a, n_b = _int(e, "n_a", 4), _int(e, "n_b", 4)
        heldout_acc, seen_acc, shuffle_acc, composite_held = [], [], [], []
        head_params = 0
        t0 = time.perf_counter()
        for s in seeds:
            fa, fb, a, b = _two_factor_forms(
                samples=_int(e, "samples", 600),
                n_a=n_a,
                n_b=n_b,
                world_dim=_int(e, "world_dim", 16),
                separation=_float(e, "separation", 1.6),
                noise=_float(e, "world_noise", 0.5),
                form_noise=_float(e, "form_noise", 0.1),
                cross_leak=_float(e, "cross_leak", 0.15),
                seed=s,
            )
            # hold out the "diagonal" combos a==b (each a-value and each b-value still appears elsewhere)
            held_pairs = {(i, i % n_b) for i in range(min(n_a, n_b))}
            is_held = torch.tensor([(int(a[i]), int(b[i])) in held_pairs for i in range(a.shape[0])])
            fused = torch.cat([fa, fb], 1)  # bind the two forms over the shared referent
            tr = (~is_held).nonzero(as_tuple=True)[0]
            te_held = is_held.nonzero(as_tuple=True)[0]
            # seen-combo test split: hold out a fraction of the trained (off-diagonal) rows
            g = torch.Generator().manual_seed(s + 77)
            perm = tr[torch.randperm(tr.shape[0], generator=g)]
            cut = int(perm.shape[0] * 0.85)
            fit_idx, seen_te = perm[:cut], perm[cut:]

            def _compose_acc(ha, hb, idx, *, x=fused, ya=a, yb=b):
                with torch.no_grad():
                    pa = ha(x[idx]).argmax(-1)
                    pb = hb(x[idx]).argmax(-1)
                    return float(((pa == ya[idx]) & (pb == yb[idx])).float().mean())

            head_a = _fit_head(
                fused[fit_idx],
                a[fit_idx],
                classes=n_a,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            head_b = _fit_head(
                fused[fit_idx],
                b[fit_idx],
                classes=n_b,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s + 5,
            )
            head_params = sum(p.numel() for p in head_a.parameters()) + sum(
                p.numel() for p in head_b.parameters()
            )
            heldout_acc.append(_compose_acc(head_a, head_b, te_held))
            seen_acc.append(_compose_acc(head_a, head_b, seen_te))

            # control 1: composite-conjunction head (n_a*n_b classes) cannot reach unseen combos
            comp_y = a * n_b + b
            comp_head = _fit_head(
                fused[fit_idx],
                comp_y[fit_idx],
                classes=n_a * n_b,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s + 9,
            )
            composite_held.append(_acc(comp_head, fused[te_held], comp_y[te_held]))

            # control 2: shuffled-label floor for the factored arm
            gg = torch.Generator().manual_seed(s + 313)
            a_sh = a[fit_idx][torch.randperm(fit_idx.shape[0], generator=gg)]
            b_sh = b[fit_idx][torch.randperm(fit_idx.shape[0], generator=gg)]
            ha_sh = _fit_head(
                fused[fit_idx],
                a_sh,
                classes=n_a,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s + 11,
            )
            hb_sh = _fit_head(
                fused[fit_idx],
                b_sh,
                classes=n_b,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s + 13,
            )
            shuffle_acc.append(_compose_acc(ha_sh, hb_sh, te_held))
        chance = 1.0 / (n_a * n_b)
        gap = _mean(seen_acc) - _mean(heldout_acc)
        floor = max(chance, _mean(shuffle_acc))
        return {
            "n_a": n_a,
            "n_b": n_b,
            "heldout_combo_acc": round(_mean(heldout_acc), 4),
            "seen_combo_acc": round(_mean(seen_acc), 4),
            "heldout_seen_gap": round(gap, 4),
            "composite_head_heldout_acc": round(_mean(composite_held), 4),
            "shuffle_floor_acc": round(_mean(shuffle_acc), 4),
            "chance": round(chance, 4),
            "seeds": seeds,
            "null_supported": bool(
                gap > _float(e, "gap_margin", 0.15) or _mean(heldout_acc) <= floor + _float(e, "margin", 0.1)
            ),
            "density": density_block(
                {"heldout_combo_acc": _mean(heldout_acc)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


def _run_form_scheduler(
    forms: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    policy: str,
    rounds: int,
    steps: int,
    classes: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """Train per-form heads under a form-selection policy and return (final test acc, visit counts).

    Each form supplies (x_tr, y_tr, x_te, y_te). A round trains the chosen form's head for `steps`
    Adam steps. Policies: uniform (round robin), error (highest current loss, chases the noisy form),
    novelty (least visited), learning_progress (largest recent test-accuracy gain, the only signal
    that should ignore the unlearnable noisy form).
    """
    seed_everything(seed)
    tags = sorted(forms)
    heads = {t: nn.Linear(forms[t][0].shape[1], classes) for t in tags}
    opts = {t: torch.optim.Adam(heads[t].parameters(), lr=lr) for t in tags}
    visits = dict.fromkeys(tags, 0)
    acc = {t: _acc(heads[t], forms[t][2], forms[t][3]) for t in tags}
    progress = dict.fromkeys(tags, 0.1)

    def _loss(t: str) -> float:
        with torch.no_grad():
            return float(F.cross_entropy(heads[t](forms[t][0]), forms[t][1]))

    for r in range(rounds):
        if policy == "uniform":
            pick = tags[r % len(tags)]
        elif policy == "novelty":
            pick = min(tags, key=lambda t: visits[t])
        elif policy == "error":
            pick = max(tags, key=_loss)
        else:  # learning_progress
            pick = max(tags, key=lambda t: progress[t] + 0.02 / (1 + visits[t]))
        xt, yt = forms[pick][0], forms[pick][1]
        for _ in range(steps):
            opts[pick].zero_grad()
            F.cross_entropy(heads[pick](xt), yt).backward()
            opts[pick].step()
        new = _acc(heads[pick], forms[pick][2], forms[pick][3])
        progress[pick] = max(0.0, new - acc[pick])
        acc[pick] = new
        visits[pick] += 1
    return acc, visits


class F10(Experiment):
    id = "f10_intrinsic_form_curriculum"
    metric = ("coverage_per_update", "noisy_form_timeshare", "transfer_gain")
    baseline = "uniform round-robin form selection and prediction-error selection"
    ablation = "learning-progress form selection vs uniform, prediction-error, and novelty"
    null_hypothesis = (
        "learning-progress form selection ties uniform coverage or spends as much time on the "
        "unlearnable noisy form as uniform, so the curriculum is not form-aware"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 4)
        rounds = _int(e, "rounds", 40)
        real_kinds = ("vision", "audio", "symbolic")
        lp_cov, uni_cov, err_cov = [], [], []
        lp_noisy, uni_noisy, err_noisy = [], [], []
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 320),
                classes=classes,
                world_dim=_int(e, "world_dim", 20),
                separation=_float(e, "separation", 1.8),
                noise=_float(e, "world_noise", 0.6),
                seed=s,
            )
            feats = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 28),
                kinds=real_kinds,
                noise=_float(e, "form_noise", 0.1),
                seed=s,
            )
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.6), s)
            forms: dict[str, tuple] = {}
            for k in real_kinds:
                forms[k] = (feats[k][tr], y[tr], feats[k][te], y[te])
            # several noisy-TV forms: pure-noise features with random labels, all unlearnable (chance).
            # With many uninformative candidate forms, a form-blind uniform policy wastes most of its
            # budget, and only a learning-progress scheduler concentrates on the few learnable forms.
            fd = _int(e, "feature_dim", 28)
            noisy_tags = []
            for j in range(_int(e, "n_noisy", 3)):
                gn = torch.Generator().manual_seed(s + 999 + 37 * j)
                tag = f"noisy_tv{j}"
                noisy_tags.append(tag)
                forms[tag] = (
                    torch.randn(tr.shape[0], fd, generator=gn),
                    torch.randint(0, classes, (tr.shape[0],), generator=gn),
                    torch.randn(te.shape[0], fd, generator=gn),
                    torch.randint(0, classes, (te.shape[0],), generator=gn),
                )

            def _cov(policy: str, *, seed: int = s, frm: dict = forms, ntags: list = noisy_tags):
                acc, visits = _run_form_scheduler(
                    frm,
                    policy=policy,
                    rounds=rounds,
                    steps=_int(e, "steps", 5),
                    classes=classes,
                    lr=_float(e, "lr", 0.05),
                    seed=seed,
                )
                coverage = _mean([acc[k] for k in real_kinds])
                timeshare = sum(visits[t] for t in ntags) / max(1, sum(visits.values()))
                return coverage, timeshare

            c_lp, n_lp = _cov("learning_progress")
            c_uni, n_uni = _cov("uniform")
            c_err, n_err = _cov("error")
            lp_cov.append(c_lp)
            uni_cov.append(c_uni)
            err_cov.append(c_err)
            lp_noisy.append(n_lp)
            uni_noisy.append(n_uni)
            err_noisy.append(n_err)
        transfer_gain = _mean(lp_cov) - _mean(uni_cov)
        return {
            "coverage_per_update": round(_mean(lp_cov) / rounds, 6),
            "lp_coverage": round(_mean(lp_cov), 4),
            "uniform_coverage": round(_mean(uni_cov), 4),
            "error_coverage": round(_mean(err_cov), 4),
            "noisy_form_timeshare": round(_mean(lp_noisy), 4),
            "uniform_noisy_timeshare": round(_mean(uni_noisy), 4),
            "error_noisy_timeshare": round(_mean(err_noisy), 4),
            "transfer_gain": round(transfer_gain, 4),
            "seeds": seeds,
            "null_supported": bool(
                transfer_gain <= _float(e, "margin", 0.02)
                or _mean(lp_noisy) >= _mean(uni_noisy) - _float(e, "noisy_margin", 0.02)
            ),
            "density": density_block(
                {"lp_coverage": _mean(lp_cov)},
                seconds=time.perf_counter() - t0,
                updates=float(rounds * _int(e, "steps", 5)),
            ),
        }


def _kmeans_codes(x: torch.Tensor, k: int, *, seed: int, iters: int = 25) -> torch.Tensor:
    """A tiny Lloyd k-means (no new dependency). Returns hard code assignments [N] in 0..k-1.

    The codebook init is seeded, so two seeds that recover the SAME partition means the code is a
    stable language over the data, not a per-run idiolect (the Wittgenstein private-language question
    at the form layer). Well-separated form clusters give init-stable codes; overlapping ones do not.
    """
    g = torch.Generator().manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    codes = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, centers)
        codes = d.argmin(1)
        for c in range(k):
            m = codes == c
            if bool(m.any()):
                centers[c] = x[m].mean(0)
    return codes


class F12(Experiment):
    id = "f12_private_form_language_stability"
    metric = ("cross_seed_code_transfer", "code_agreement", "random_code_floor")
    baseline = "random-codebook agreement floor and chance cross-seed probe transfer"
    ablation = "seeded k-means form codes vs random codebooks; cross-seed Hungarian-matched transfer"
    null_hypothesis = (
        "cross-seed code agreement sits at or below the random-codebook floor and cross-seed probe "
        "transfer is at chance, so the form codes are private idiolects rather than a shared language"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.seed_consistency import code_stability

        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 5)
        k = _int(e, "codebook_k", 5)
        # one shared set of referents (fixed), re-observed per seed with independent form noise, so
        # code assignments are comparable point-by-point across seeds.
        z, y = _balanced_world(
            samples=_int(e, "samples", 300),
            classes=classes,
            world_dim=_int(e, "world_dim", 20),
            separation=_float(e, "separation", 1.6),
            noise=_float(e, "world_noise", 0.5),
            seed=_int(e, "base_seed", 999),
        )
        code_lists, rand_lists = [], []
        t0 = time.perf_counter()
        for s in seeds:
            feats = _form_features(
                z, feature_dim=_int(e, "feature_dim", 24), noise=_float(e, "form_noise", 0.3), seed=s
            )
            fused = torch.cat([feats[kd] for kd in FORM_KINDS], 1)
            code_lists.append(_kmeans_codes(fused, k, seed=s))
            gr = torch.Generator().manual_seed(s + 4242)
            rand_lists.append(torch.randint(0, k, (fused.shape[0],), generator=gr))
        agreement = code_stability(code_lists, k)
        rand_floor = code_stability(rand_lists, k)

        # cross-seed probe transfer: fit a probe on seed-0 codes -> class, test on seed-1 codes
        # after Hungarian relabeling into seed-0's code space
        def _onehot(c: torch.Tensor) -> torch.Tensor:
            return F.one_hot(c, k).float()

        tr, te = _split(y.shape[0], _float(e, "train_frac", 0.6), 0)
        probe = _fit_head(
            _onehot(code_lists[0])[tr],
            y[tr],
            classes=classes,
            epochs=_int(e, "epochs", 120),
            lr=_float(e, "lr", 0.05),
            seed=0,
        )
        transfer = []
        for j in range(1, len(code_lists)):
            # map seed-j codes onto seed-0 codes via the best Hungarian assignment
            conf = torch.zeros(k, k)
            for a in range(k):
                for b in range(k):
                    conf[a, b] = float(((code_lists[0] == a) & (code_lists[j] == b)).sum())
            remap = conf.argmax(0)  # seed-j code b -> seed-0 code remap[b]
            mapped = remap[code_lists[j]]
            transfer.append(_acc(probe, _onehot(mapped)[te], y[te]))
        chance = 1.0 / classes
        cross_transfer = _mean(transfer) if transfer else 0.0
        codes_are_language = agreement["mean_agreement"] > rand_floor["mean_agreement"] + _float(
            e, "margin", 0.15
        ) and cross_transfer > chance + _float(e, "transfer_margin", 0.1)
        return {
            "codebook_k": k,
            "code_agreement": round(agreement["mean_agreement"], 4),
            "random_code_floor": round(rand_floor["mean_agreement"], 4),
            "cross_seed_code_transfer": round(cross_transfer, 4),
            "chance": round(chance, 4),
            "codes_recur_across_seeds": bool(codes_are_language),
            "seeds": seeds,
            "null_supported": bool(not codes_are_language),
            "density": density_block(
                {"code_agreement": agreement["mean_agreement"]},
                seconds=time.perf_counter() - t0,
                params=float(k * (_int(e, "feature_dim", 24) * len(FORM_KINDS))),
            ),
        }


class F19(Experiment):
    id = "f19_cross_scale_referent_binding"
    metric = ("cross_scale_recall_at_k", "flat_memory_recall_at_k", "recall_per_byte")
    baseline = "a flat exemplar store of the same byte budget, matched on stored vectors"
    ablation = "a multi-scale store (episode centroids plus object exemplars) vs a flat exemplar store"
    null_hypothesis = (
        "the multi-scale store ties the flat exemplar store at matched bytes, so allocating memory "
        "across referent scales buys no cross-scale retrieval and memory stays single-scale"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_ep = _int(e, "episodes", 12)
        o_per = _int(e, "objects_per_episode", 10)
        dim = _int(e, "world_dim", 16)
        budget = _int(e, "store_vectors", 60)  # both memories occupy this many vectors (matched bytes)
        hier_recall, flat_recall = [], []
        t0 = time.perf_counter()
        for s in seeds:
            g = torch.Generator().manual_seed(s)
            centers = torch.randn(n_ep, dim, generator=g) * _float(e, "separation", 1.4)
            noise = _float(e, "object_noise", 1.1)
            # stored objects: o_per per episode
            ep_ids = torch.arange(n_ep).repeat_interleave(o_per)
            store_obj = centers[ep_ids] + noise * torch.randn(n_ep * o_per, dim, generator=g)
            # fresh queries at OBJECT scale, retrieve the EPISODE referent (cross-scale)
            q_ep = torch.randint(0, n_ep, (_int(e, "queries", 240),), generator=g)
            queries = centers[q_ep] + noise * torch.randn(q_ep.shape[0], dim, generator=g)

            # hierarchical store: n_ep episode centroids (denoised) + remaining budget as exemplars.
            # episode retrieval matches the coarse centroids.
            centroids = torch.stack([store_obj[ep_ids == c].mean(0) for c in range(n_ep)])
            pred_h_ep = torch.cdist(queries, centroids).argmin(1)  # nearest coarse centroid
            hier_recall.append(float((pred_h_ep == q_ep).float().mean()))

            # flat store: the SAME byte budget spent entirely on raw object exemplars, no centroids.
            perm = torch.randperm(store_obj.shape[0], generator=g)[:budget]
            flat_vecs, flat_ep = store_obj[perm], ep_ids[perm]
            pred_f_ep = flat_ep[torch.cdist(queries, flat_vecs).argmin(1)]
            flat_recall.append(float((pred_f_ep == q_ep).float().mean()))
        store_bytes = budget * dim * 4
        gain = _mean(hier_recall) - _mean(flat_recall)
        return {
            "episodes": n_ep,
            "cross_scale_recall_at_k": round(_mean(hier_recall), 4),
            "flat_memory_recall_at_k": round(_mean(flat_recall), 4),
            "hier_minus_flat": round(gain, 4),
            "recall_per_byte": round(_mean(hier_recall) / store_bytes, 9),
            "store_vectors": budget,
            "seeds": seeds,
            "null_supported": bool(gain <= _float(e, "margin", 0.05)),
            "density": density_block(
                {"cross_scale_recall_at_k": _mean(hier_recall)},
                seconds=time.perf_counter() - t0,
                bytes=float(store_bytes),
            ),
        }
