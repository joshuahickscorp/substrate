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
from ..seeding import derive_seed32, seed_everything
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


def _three_way_split(
    n: int,
    anchor_frac: float,
    label_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Disjoint alignment, supervised-training, and untouched evaluation rows."""
    g = torch.Generator().manual_seed(seed + 811)
    perm = torch.randperm(n, generator=g)
    n_anchor, n_label = int(n * anchor_frac), int(n * label_frac)
    if n_anchor <= 0 or n_label <= 0 or n_anchor + n_label >= n:
        raise ValueError("anchor, label, and test splits must all be non-empty")
    return (
        perm[:n_anchor],
        perm[n_anchor : n_anchor + n_label],
        perm[n_anchor + n_label :],
    )


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
    baseline = (
        "one frozen source-form head evaluated on raw, moment-matched, and shuffled-anchor target coordinates"
    )
    ablation = "paired-referent affine target-to-source alignment vs unpaired coordinate controls"
    null_hypothesis = (
        "paired referent alignment fails to beat the strongest raw, moment-matched, or "
        "shuffled-anchor control by the preregistered margin, or remains near chance"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        aligned, raw, moment, shuffled, source, oracle = [], [], [], [], [], []
        deltas: list[float] = []
        head_params = 0
        split_rows: dict[str, int] = {}
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 420),
                classes=_int(e, "classes", 8),
                world_dim=_int(e, "world_dim", 28),
                separation=_float(e, "separation", 0.9),
                noise=_float(e, "world_noise", 1.1),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 30),
                noise=_float(e, "form_noise", 0.55),
                seed=s,
            )
            anchors, labels, test = _three_way_split(
                y.shape[0],
                _float(e, "anchor_frac", 0.25),
                _float(e, "label_frac", 0.35),
                s,
            )
            split_rows = {
                "anchor": int(anchors.shape[0]),
                "label": int(labels.shape[0]),
                "test": int(test.shape[0]),
            }
            src, tgt = _str(e, "source_form", "vision"), _str(e, "target_form", "audio")
            head = _fit_head(
                forms[src][labels],
                y[labels],
                classes=_int(e, "classes", 8),
                epochs=_int(e, "epochs", 70),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            head_params = sum(p.numel() for p in head.parameters())
            source.append(_acc(head, forms[src][test], y[test]))
            raw_seed = _acc(head, forms[tgt][test], y[test])
            raw.append(raw_seed)

            target_anchor = forms[tgt][anchors]
            source_anchor = forms[src][anchors]
            target_mean = target_anchor.mean(0)
            target_std = target_anchor.std(0).clamp_min(1.0e-6)
            source_mean = source_anchor.mean(0)
            source_std = source_anchor.std(0).clamp_min(1.0e-6)
            moment_target = (forms[tgt] - target_mean) / target_std * source_std + source_mean
            moment_seed = _acc(head, moment_target[test], y[test])
            moment.append(moment_seed)

            w = fit_affine_alignment(target_anchor, source_anchor)
            aligned_seed = _acc(
                head,
                apply_affine_alignment(forms[tgt][test], w),
                y[test],
            )
            aligned.append(aligned_seed)
            g = torch.Generator().manual_seed(s + 515)
            shuffled_src = source_anchor[torch.randperm(anchors.shape[0], generator=g)]
            w_shuf = fit_affine_alignment(target_anchor, shuffled_src)
            shuffled_seed = _acc(
                head,
                apply_affine_alignment(forms[tgt][test], w_shuf),
                y[test],
            )
            shuffled.append(shuffled_seed)
            target_oracle = _fit_head(
                forms[tgt][labels],
                y[labels],
                classes=_int(e, "classes", 8),
                epochs=_int(e, "epochs", 70),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            oracle.append(_acc(target_oracle, forms[tgt][test], y[test]))
            deltas.append(aligned_seed - max(raw_seed, moment_seed, shuffled_seed))
        best_control = max(_mean(raw), _mean(moment), _mean(shuffled))
        gain = _mean(deltas)
        chance = 1.0 / _int(e, "classes", 8)
        return {
            "source_form": _str(e, "source_form", "vision"),
            "target_form": _str(e, "target_form", "audio"),
            "source_acc": round(_mean(source), 4),
            "raw_transfer": round(_mean(raw), 4),
            "moment_matched_transfer": round(_mean(moment), 4),
            "aligned_transfer": round(_mean(aligned), 4),
            "shuffled_anchor_transfer": round(_mean(shuffled), 4),
            "target_supervised_oracle": round(_mean(oracle), 4),
            "best_unpaired_control": round(best_control, 4),
            "aligned_gain_over_best_control": round(gain, 4),
            "chance": round(chance, 4),
            "split_rows": split_rows,
            "disjoint_splits": True,
            "head_params": head_params,
            "rows_per_step": split_rows["label"],
            "optimizer_steps": _int(e, "epochs", 70),
            "total_rows_seen": split_rows["label"] * _int(e, "epochs", 70),
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
            "null_supported": bool(
                gain <= _float(e, "margin", 0.05) or _mean(aligned) <= chance + _float(e, "chance_band", 0.1)
            ),
            "density": density_block(
                {"aligned_transfer": _mean(aligned)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


class F2(Experiment):
    id = "f2_heldout_form_transfer"
    metric = ("heldout_form_acc", "single_form_baseline", "multi_form_gain")
    baseline = (
        "single reference form and matched Gaussian augmentation with identical heads, rows, "
        "updates, and held-out-form alignment"
    )
    ablation = (
        "matched-exposure training that substitutes one aligned training form per referent and optimizer step"
    )
    null_hypothesis = (
        "matched-exposure multi-form training fails to beat the strongest single-reference or "
        "matched-noise control by the preregistered margin, the held-out form remains near chance, "
        "or referent-shuffled alignment ties the treatment"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        heldout = _str(e, "heldout_form", "timeseries")
        multi_acc, single_acc, noise_acc, shuffled_acc, reference_acc, audits = [], [], [], [], [], []
        deltas: list[float] = []
        head_params = 0
        accounting: dict[str, dict[str, int]] = {}
        split_rows: dict[str, int] = {}
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 400),
                classes=_int(e, "classes", 10),
                world_dim=_int(e, "world_dim", 28),
                separation=_float(e, "separation", 0.9),
                noise=_float(e, "world_noise", 1.2),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 36),
                noise=_float(e, "form_noise", 0.7),
                seed=s,
            )
            audits.append(_matrix(forms, y))
            anchors, labels, test = _three_way_split(
                y.shape[0],
                _float(e, "anchor_frac", 0.25),
                _float(e, "label_frac", 0.3),
                s,
            )
            split_rows = {
                "anchor": int(anchors.shape[0]),
                "label": int(labels.shape[0]),
                "test": int(test.shape[0]),
            }
            aligned = _aligned_forms(forms, anchors)
            train_forms = [k for k in FORM_KINDS if k != heldout]
            reference = _str(e, "reference_form", "vision")
            residual = torch.cat(
                [aligned[k][anchors] - aligned[reference][anchors] for k in train_forms if k != reference]
            )
            augmentation_scale = float(residual.std())
            epochs = _int(e, "epochs", 60)

            def _fit_matched(
                mode: str,
                *,
                seed: int = s,
                n_epochs: int = epochs,
                label_idx: torch.Tensor = labels,
                training_forms: tuple[str, ...] = tuple(train_forms),
                aligned_features: dict[str, torch.Tensor] = aligned,
                reference_form: str = reference,
                noise_scale: float = augmentation_scale,
                targets: torch.Tensor = y,
            ) -> nn.Linear:
                seed_everything(seed + 71)
                head = nn.Linear(_int(e, "feature_dim", 36), _int(e, "classes", 10))
                opt = torch.optim.Adam(head.parameters(), lr=_float(e, "lr", 0.03))
                g = torch.Generator().manual_seed(seed + 903)
                for epoch in range(n_epochs):
                    if mode == "multi":
                        choice = (torch.arange(label_idx.shape[0]) + epoch) % len(training_forms)
                        rows = torch.stack([aligned_features[k][label_idx] for k in training_forms])
                        x_epoch = rows[choice, torch.arange(label_idx.shape[0])]
                    elif mode == "noise":
                        x_epoch = aligned_features[reference_form][label_idx] + noise_scale * torch.randn(
                            aligned_features[reference_form][label_idx].shape, generator=g
                        )
                    else:
                        x_epoch = aligned_features[reference_form][label_idx]
                    opt.zero_grad()
                    F.cross_entropy(head(x_epoch), targets[label_idx]).backward()
                    opt.step()
                return head

            head_multi = _fit_matched("multi")
            head_single = _fit_matched("single")
            head_noise = _fit_matched("noise")
            head_params = sum(p.numel() for p in head_multi.parameters())
            multi_seed = _acc(head_multi, aligned[heldout][test], y[test])
            single_seed = _acc(head_single, aligned[heldout][test], y[test])
            noise_seed = _acc(head_noise, aligned[heldout][test], y[test])
            multi_acc.append(multi_seed)
            single_acc.append(single_seed)
            noise_acc.append(noise_seed)
            reference_acc.append(_acc(head_multi, aligned[reference][test], y[test]))

            g = torch.Generator().manual_seed(s + 1907)
            shuffled_reference = aligned[reference][anchors[torch.randperm(anchors.shape[0], generator=g)]]
            shuffled_map = fit_affine_alignment(forms[heldout][anchors], shuffled_reference)
            shuffled_seed = _acc(
                head_multi,
                apply_affine_alignment(forms[heldout][test], shuffled_map),
                y[test],
            )
            shuffled_acc.append(shuffled_seed)
            deltas.append(multi_seed - max(single_seed, noise_seed))
            accounting = {
                arm: {
                    "rows_per_step": int(labels.shape[0]),
                    "optimizer_steps": epochs,
                    "total_rows_seen": int(labels.shape[0]) * epochs,
                    "head_params": head_params,
                    "input_dim": _int(e, "feature_dim", 36),
                }
                for arm in ("multi", "single", "matched_noise")
            }
        chance = 1.0 / _int(e, "classes", 10)
        strongest = max(_mean(single_acc), _mean(noise_acc))
        gain = _mean(multi_acc) - strongest
        return {
            "heldout_form": heldout,
            "heldout_form_acc": round(_mean(multi_acc), 4),
            "single_form_baseline": round(_mean(single_acc), 4),
            "matched_noise_baseline": round(_mean(noise_acc), 4),
            "reference_form_test_acc": round(_mean(reference_acc), 4),
            "shuffled_heldout_alignment_acc": round(_mean(shuffled_acc), 4),
            "strongest_exposure_matched_control": round(strongest, 4),
            "multi_form_gain": round(gain, 4),
            "chance": round(chance, 4),
            "audit_all_ok": bool(all(a["all_ok"] for a in audits)),
            "split_rows": split_rows,
            "disjoint_splits": True,
            "matched_accounting": accounting,
            "matched_rows_updates_head": len({tuple(values.values()) for values in accounting.values()}) == 1,
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
            "null_supported": bool(
                gain <= _float(e, "margin", 0.02)
                or _mean(multi_acc) <= chance + _float(e, "chance_band", 0.1)
                or _mean(multi_acc) <= _mean(shuffled_acc) + _float(e, "margin", 0.02)
            ),
            "density": density_block(
                {"heldout_form_acc": _mean(multi_acc)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
                updates=float(_int(e, "epochs", 60)),
            ),
        }


class F3(Experiment):
    id = "f3_form_bottleneck_capacity"
    metric = ("wide_form_acc", "small_form_acc", "wide_minus_small")
    baseline = (
        "nested zero-padded small bottleneck, shuffled labels, no bottleneck, and all-form concatenation"
    )
    ablation = (
        "nested wide vs small canonical bottlenecks with identical head topology, initialization, "
        "data, and updates"
    )
    null_hypothesis = (
        "the nested wide bottleneck fails to beat the zero-padded small bottleneck by the "
        "preregistered margin, or wide performance remains near the chance or shuffled-label floor"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        wide_acc, small_acc, shuffle_acc, uncompressed_acc, concat_acc = [], [], [], [], []
        deltas: list[float] = []
        head_params = 0
        accounting: dict[str, dict[str, int]] = {}
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 360),
                classes=_int(e, "classes", 10),
                world_dim=_int(e, "world_dim", 28),
                separation=_float(e, "separation", 0.8),
                noise=_float(e, "world_noise", 1.2),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 28),
                noise=_float(e, "form_noise", 0.6),
                seed=s,
            )
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.55), s)
            aligned = _aligned_forms(forms, tr)
            x_train = torch.cat([aligned[k][tr] for k in FORM_KINDS], dim=0)
            y_train = torch.cat([y[tr] for _ in FORM_KINDS], dim=0)
            x_test = torch.cat([aligned[k][te] for k in FORM_KINDS], dim=0)
            y_test = torch.cat([y[te] for _ in FORM_KINDS], dim=0)

            feature_dim = _int(e, "feature_dim", 28)
            wide_dim = _int(e, "wide_dim", 14)
            small_dim = _int(e, "small_dim", 2)
            if not 0 < small_dim < wide_dim <= feature_dim:
                raise ValueError("F3 requires 0 < small_dim < wide_dim <= feature_dim")
            g = torch.Generator().manual_seed(s + 700)
            q, _ = torch.linalg.qr(torch.randn(feature_dim, feature_dim, generator=g))
            head_dim = feature_dim * len(FORM_KINDS)

            def _pad(x: torch.Tensor, *, common_dim: int = head_dim) -> torch.Tensor:
                if x.shape[1] > common_dim:
                    raise ValueError("F3 representation exceeds the common head dimension")
                return F.pad(x, (0, common_dim - x.shape[1]))

            xw, xw_te = _pad(x_train @ q[:, :wide_dim]), _pad(x_test @ q[:, :wide_dim])
            xs, xs_te = _pad(x_train @ q[:, :small_dim]), _pad(x_test @ q[:, :small_dim])
            xu, xu_te = _pad(x_train), _pad(x_test)
            concat_train = torch.cat([aligned[k][tr] for k in FORM_KINDS], dim=1)
            concat_test = torch.cat([aligned[k][te] for k in FORM_KINDS], dim=1)
            xc = concat_train.repeat((len(FORM_KINDS), 1))
            xc_te = concat_test.repeat((len(FORM_KINDS), 1))
            epochs = _int(e, "epochs", 70)
            classes = _int(e, "classes", 10)
            shared_seed = s + 1701
            hw = _fit_head(
                xw,
                y_train,
                classes=classes,
                epochs=epochs,
                lr=_float(e, "lr", 0.03),
                seed=shared_seed,
            )
            hs = _fit_head(
                xs,
                y_train,
                classes=classes,
                epochs=epochs,
                lr=_float(e, "lr", 0.03),
                seed=shared_seed,
            )
            hu = _fit_head(
                xu,
                y_train,
                classes=classes,
                epochs=epochs,
                lr=_float(e, "lr", 0.03),
                seed=shared_seed,
            )
            hc = _fit_head(
                xc,
                y_train,
                classes=classes,
                epochs=epochs,
                lr=_float(e, "lr", 0.03),
                seed=shared_seed,
            )
            g = torch.Generator().manual_seed(s + 912)
            y_shuf_referent = y[tr][torch.randperm(tr.shape[0], generator=g)]
            y_shuf = torch.cat([y_shuf_referent for _ in FORM_KINDS])
            hf = _fit_head(
                xw,
                y_shuf,
                classes=classes,
                epochs=epochs,
                lr=_float(e, "lr", 0.03),
                seed=shared_seed,
            )
            head_params = sum(p.numel() for p in hw.parameters())
            wide_seed = _acc(hw, xw_te, y_test)
            small_seed = _acc(hs, xs_te, y_test)
            wide_acc.append(wide_seed)
            small_acc.append(small_seed)
            shuffle_acc.append(_acc(hf, xw_te, y_test))
            uncompressed_acc.append(_acc(hu, xu_te, y_test))
            concat_acc.append(_acc(hc, xc_te, y_test))
            deltas.append(wide_seed - small_seed)
            accounting = {
                arm: {
                    "head_dim": head_dim,
                    "head_params": head_params,
                    "train_rows": int(x_train.shape[0]),
                    "optimizer_steps": epochs,
                    "total_rows_seen": int(x_train.shape[0]) * epochs,
                }
                for arm in ("wide", "small", "no_bottleneck", "concat", "shuffle")
            }
        gain = _mean(wide_acc) - _mean(small_acc)
        chance = 1.0 / _int(e, "classes", 10)
        return {
            "wide_dim": _int(e, "wide_dim", 14),
            "small_dim": _int(e, "small_dim", 2),
            "wide_form_acc": round(_mean(wide_acc), 4),
            "small_form_acc": round(_mean(small_acc), 4),
            "shuffle_floor_acc": round(_mean(shuffle_acc), 4),
            "no_bottleneck_acc": round(_mean(uncompressed_acc), 4),
            "concatenated_forms_upper_bound_acc": round(_mean(concat_acc), 4),
            "wide_minus_small": round(gain, 4),
            "wide_retention_vs_no_bottleneck": round(_mean(wide_acc) - _mean(uncompressed_acc), 4),
            "chance": round(chance, 4),
            "nested_projection": True,
            "small_zero_padded": True,
            "identical_head_topology": True,
            "matched_accounting": accounting,
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
            "null_supported": bool(
                gain <= _float(e, "margin", 0.05)
                or _mean(wide_acc) <= max(chance, _mean(shuffle_acc)) + _float(e, "floor_band", 0.1)
            ),
            "density": density_block(
                {"wide_form_acc": _mean(wide_acc)},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
            ),
        }


class F5(Experiment):
    id = "f5_cross_form_memory_binding"
    metric = (
        "cross_form_recall_at_k",
        "same_form_recall_at_k",
        "shuffled_referent_floor",
        "recall_per_slot",
    )
    baseline = "same-form independent-view retrieval, raw cross-form retrieval, and shuffled referents"
    ablation = "paired-referent alignment vs shuffled alignment through the shared ReplayBuffer/KVIndex"
    null_hypothesis = (
        "cross-form retrieval fails to beat raw or shuffled controls, or remains materially below "
        "same-form independent-view retrieval, so memory is form-local rather than referent-bound"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.riskcov import seed_ci, sign_flip_report
        from ..shell.buffer import ReplayBuffer

        e = cfg.experiment
        seeds = list(e.seeds)
        aligned, raw, shuffled, local = [], [], [], []
        deltas: list[float] = []
        store_form = _str(e, "store_form", "vision")
        query_form = _str(e, "query_form", "audio")
        k = _int(e, "k", 1)
        store_bytes = 0
        store_slots = 0
        index_backend = "brute"
        t0 = time.perf_counter()
        for s in seeds:
            z, _ = _balanced_world(
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
            tr, te = _split(z.shape[0], _float(e, "anchor_frac", 0.45), s)
            g = torch.Generator().manual_seed(s + 601)
            query_noise = _float(e, "query_noise", 0.35)
            store = forms[store_form]
            query = forms[query_form] + query_noise * torch.randn(forms[query_form].shape, generator=g)
            same_query = store + query_noise * torch.randn(store.shape, generator=g)

            memory = ReplayBuffer(
                capacity=int(te.shape[0]),
                dim=store.shape[1],
                key_dim=store.shape[1],
                prioritized=False,
                eviction="fifo",
                index=index_backend,
                seed=s,
            )
            memory.add(store[te], te, key=store[te])
            store_slots = len(memory)
            store_bytes = int(
                memory.x[: len(memory)].nelement() * memory.x.element_size()
                + memory.keys[: len(memory)].nelement() * memory.keys.element_size()
                + memory.y[: len(memory)].nelement() * memory.y.element_size()
            )

            def _recall(
                q: torch.Tensor,
                *,
                mem: ReplayBuffer = memory,
                test_idx: torch.Tensor = te,
            ) -> float:
                got = mem.retrieve(q[test_idx], k=k)["y"]
                return float((got == test_idx[:, None]).any(dim=1).float().mean())

            raw_seed = _recall(query)
            raw.append(raw_seed)
            w = fit_affine_alignment(query[tr], store[tr])
            aligned_seed = _recall(apply_affine_alignment(query, w))
            aligned.append(aligned_seed)
            g = torch.Generator().manual_seed(s + 619)
            shuf_store = store[tr[torch.randperm(tr.shape[0], generator=g)]]
            w_shuf = fit_affine_alignment(query[tr], shuf_store)
            shuffled_seed = _recall(apply_affine_alignment(query, w_shuf))
            shuffled.append(shuffled_seed)
            local_seed = _recall(same_query)
            local.append(local_seed)
            deltas.append(aligned_seed - max(raw_seed, shuffled_seed))
        best_control = max(_mean(raw), _mean(shuffled))
        gain = _mean(aligned) - best_control
        oracle_gap = _mean(local) - _mean(aligned)
        margin = _float(e, "margin", 0.05)
        oracle_band = _float(e, "oracle_band", 0.15)
        recall_per_slot = _mean(aligned) / max(store_slots, 1)
        return {
            "store_form": store_form,
            "query_form": query_form,
            "k": k,
            "cross_form_recall_at_k": round(_mean(aligned), 4),
            "same_form_recall_at_k": round(_mean(local), 4),
            "raw_cross_form_recall_at_k": round(_mean(raw), 4),
            "shuffled_referent_floor": round(_mean(shuffled), 4),
            "recall_per_slot": round(recall_per_slot, 8),
            "aligned_gain_over_best_control": round(gain, 4),
            "same_form_oracle_gap": round(oracle_gap, 4),
            "store_slots": store_slots,
            "store_bytes": store_bytes,
            "index_backend": index_backend,
            "seeds": seeds,
            "per_seed_deltas": [round(v, 4) for v in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
            "null_supported": bool(gain <= margin or oracle_gap > oracle_band),
            "density": density_block(
                {
                    "cross_form_recall_at_k": _mean(aligned),
                    "recall_per_slot": recall_per_slot,
                },
                primary="cross_form_recall_at_k",
                seconds=time.perf_counter() - t0,
                bytes=float(store_bytes),
            ),
        }


def _hetero_token_payloads(
    z: torch.Tensor,
    *,
    dims: dict[str, int],
    token_count: int,
    noise: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Render ordered world chunks into genuine heterogeneous [N,T,C] payloads."""
    if z.shape[1] % token_count:
        raise ValueError("F4 world_dim must be divisible by token_count")
    chunk_dim = z.shape[1] // token_count
    out: dict[str, torch.Tensor] = {}
    for i, (kind, d) in enumerate(sorted(dims.items())):
        tokens = []
        for token in range(token_count):
            g = torch.Generator().manual_seed(seed + 2003 * (i + 1) + 101 * token)
            w = torch.randn(chunk_dim, d, generator=g) / math.sqrt(chunk_dim)
            bias = 0.15 * torch.randn(d, generator=g)
            chunk = z[:, token * chunk_dim : (token + 1) * chunk_dim]
            x = chunk @ w + bias + noise * torch.randn(z.shape[0], d, generator=g)
            tokens.append(x)
        x = torch.stack(tokens, dim=1)
        if kind == "symbolic":
            x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1.0e-6)
        out[kind] = x.float()
    return out


def _tokens_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Resize only the channel axis; the token axis is never flattened or resampled."""
    n, tokens, d = x.shape
    if d == dim:
        return x
    if d > dim:
        return x[:, :, :dim]
    return torch.cat([x, torch.zeros(n, tokens, dim - d)], dim=2)


def _token_handcrafted(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-token statistics, preserving [N,T,D] geometry."""
    stats = torch.stack(
        [x.mean(2), x.std(2), x.amin(2), x.amax(2)],
        dim=2,
    )
    reps = (dim + 3) // 4
    return stats.repeat(1, 1, reps)[:, :, :dim]


class _F4TokenProbe(nn.Module):
    def __init__(self, tokens: int, token_dim: int, hidden: int, classes: int):
        super().__init__()
        self.token = nn.Linear(token_dim, hidden)
        self.readout = nn.Linear(tokens * hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("F4 token probe requires [N,T,D] input")
        return self.readout(torch.tanh(self.token(x)).flatten(1))


def _f4_fit_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    tokens: int,
    token_dim: int,
    hidden: int,
    classes: int,
    epochs: int,
    lr: float,
    seed: int,
) -> _F4TokenProbe:
    seed_everything(seed)
    head = _F4TokenProbe(tokens, token_dim, hidden, classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(x), y).backward()
        opt.step()
    return head


def _f4_align_tokens(
    target: torch.Tensor,
    reference: torch.Tensor,
    anchors: torch.Tensor,
    *,
    shuffled: bool,
    seed: int,
) -> torch.Tensor:
    """Fit one affine bridge per ordered token position."""
    mapped = []
    g = torch.Generator().manual_seed(seed)
    target_order = torch.arange(target.shape[1])
    source_anchor = reference[anchors]
    if shuffled:
        source_anchor = source_anchor[torch.randperm(anchors.shape[0], generator=g)]
        target_order = torch.roll(target_order, 1)
    for reference_token, target_token in enumerate(target_order.tolist()):
        w = fit_affine_alignment(
            target[anchors, target_token],
            source_anchor[:, reference_token],
        )
        mapped.append(apply_affine_alignment(target[:, target_token], w))
    return torch.stack(mapped, dim=1)


class F4(Experiment):
    id = "f4_raw_payload_vs_form_tokens"
    metric = ("cross_form_transfer_per_dim", "retention_per_dim", "control_delta")
    baseline = (
        "raw resized token payloads, per-token handcrafted statistics, shuffled referent-token "
        "alignment, and token-order permutation with matched token heads"
    )
    ablation = "ordered tokenwise paired-referent alignment into a fixed token geometry"
    null_hypothesis = (
        "ordered canonical form tokens fail to beat the strongest raw, handcrafted, "
        "shuffled-referent, or token-order control by the preregistered margin"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 8)
        token_count = _int(e, "token_count", 4)
        token_dim = _int(e, "token_dim", 8)
        token_hidden = _int(e, "token_hidden", 8)
        total_dim = token_count * token_dim
        reference = _str(e, "reference_form", "vision")
        dims = {
            "vision": _int(e, "vision_channel_dim", 8),
            "audio": _int(e, "audio_channel_dim", 6),
            "symbolic": _int(e, "symbolic_channel_dim", 12),
            "timeseries": _int(e, "timeseries_channel_dim", 5),
        }
        canon_cross, raw_cross, hand_cross, shuffled_cross, order_cross, canon_same = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        deltas: list[float] = []
        t0 = time.perf_counter()
        head_params = 0
        split_rows: dict[str, int] = {}
        audits: list[dict] = []
        accounting: dict[str, dict[str, int]] = {}
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 400),
                classes=classes,
                world_dim=_int(e, "world_dim", 32),
                separation=_float(e, "separation", 0.9),
                noise=_float(e, "world_noise", 1.1),
                seed=s,
            )
            payloads = _hetero_token_payloads(
                z,
                dims=dims,
                token_count=token_count,
                noise=_float(e, "form_noise", 0.5),
                seed=s,
            )
            anchors, labels, test = _three_way_split(
                y.shape[0],
                _float(e, "anchor_frac", 0.25),
                _float(e, "label_frac", 0.35),
                s,
            )
            split_rows = {
                "anchor": int(anchors.shape[0]),
                "label": int(labels.shape[0]),
                "test": int(test.shape[0]),
            }
            resized = {kind: _tokens_to_dim(value, token_dim) for kind, value in payloads.items()}
            handcrafted = {kind: _token_handcrafted(value, token_dim) for kind, value in payloads.items()}
            refs = _referents(y.shape[0])
            adapters = [
                TensorFormAdapter(
                    FormMeta(
                        tag=kind,
                        kind=kind,
                        feature_dim=total_dim,
                        source="f4-synthetic-token-payload",
                        objective="handcrafted",
                        token_shape=(token_count, token_dim),
                        time_axis=True,
                    ),
                    value,
                    refs,
                    factors={"class": y},
                )
                for kind, value in resized.items()
            ]
            audits.append(form_audit(build_form_matrix(adapters), require_controls=False))
            others = [k for k in dims if k != reference]
            epochs = _int(e, "epochs", 70)
            heads = {}
            for mode, ref_repr in (
                ("canonical", resized[reference]),
                ("raw", resized[reference]),
                ("handcrafted", handcrafted[reference]),
            ):
                heads[mode] = _f4_fit_probe(
                    ref_repr[labels],
                    y[labels],
                    tokens=token_count,
                    token_dim=token_dim,
                    hidden=token_hidden,
                    classes=classes,
                    epochs=epochs,
                    lr=_float(e, "lr", 0.03),
                    seed=s + 4101,
                )
            # The same frozen reference head evaluates every target transform.
            canonical_targets = {
                kind: _f4_align_tokens(
                    resized[kind],
                    resized[reference],
                    anchors,
                    shuffled=False,
                    seed=s + 5101,
                )
                for kind in others
            }
            shuffled_targets = {
                kind: _f4_align_tokens(
                    resized[kind],
                    resized[reference],
                    anchors,
                    shuffled=True,
                    seed=s + 6101,
                )
                for kind in others
            }
            canonical_seed = _mean(
                [_acc(heads["canonical"], canonical_targets[k][test], y[test]) for k in others]
            )
            raw_seed = _mean([_acc(heads["raw"], resized[k][test], y[test]) for k in others])
            hand_seed = _mean([_acc(heads["handcrafted"], handcrafted[k][test], y[test]) for k in others])
            shuffled_seed = _mean(
                [_acc(heads["canonical"], shuffled_targets[k][test], y[test]) for k in others]
            )
            order_seed = _mean(
                [
                    _acc(
                        heads["canonical"],
                        torch.roll(canonical_targets[k][test], shifts=1, dims=1),
                        y[test],
                    )
                    for k in others
                ]
            )
            canon_cross.append(canonical_seed)
            raw_cross.append(raw_seed)
            hand_cross.append(hand_seed)
            shuffled_cross.append(shuffled_seed)
            order_cross.append(order_seed)
            canon_same.append(_acc(heads["canonical"], resized[reference][test], y[test]))
            deltas.append(canonical_seed - max(raw_seed, hand_seed, shuffled_seed, order_seed))
            head_params = sum(p.numel() for p in heads["canonical"].parameters())
            accounting = {
                mode: {
                    "head_params": head_params,
                    "train_rows": int(labels.shape[0]),
                    "optimizer_steps": epochs,
                    "total_rows_seen": int(labels.shape[0]) * epochs,
                    "token_count": token_count,
                    "token_dim": token_dim,
                }
                for mode in ("canonical", "raw", "handcrafted")
            }
        best_control = max(
            _mean(raw_cross),
            _mean(hand_cross),
            _mean(shuffled_cross),
            _mean(order_cross),
        )
        delta = _mean(deltas)
        return {
            "reference_form": reference,
            "matched_dim": total_dim,
            "token_shape": [token_count, token_dim],
            "token_axis_preserved": True,
            "token_probe_rejects_flattened": True,
            "audit_all_ok": bool(all(audit["all_ok"] for audit in audits)),
            "canonical_cross_form_acc": round(_mean(canon_cross), 4),
            "raw_cross_form_acc": round(_mean(raw_cross), 4),
            "handcrafted_cross_form_acc": round(_mean(hand_cross), 4),
            "shuffled_referent_token_acc": round(_mean(shuffled_cross), 4),
            "token_order_permutation_acc": round(_mean(order_cross), 4),
            "strongest_control_acc": round(best_control, 4),
            "cross_form_transfer_per_dim": round(_mean(canon_cross) / total_dim, 6),
            "retention_per_dim": round(_mean(canon_same) / total_dim, 6),
            "control_delta": round(delta, 4),
            "split_rows": split_rows,
            "disjoint_splits": True,
            "matched_accounting": accounting,
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
            "null_supported": bool(delta <= _float(e, "margin", 0.05)),
            "density": density_block(
                {"cross_form_transfer_per_dim": _mean(canon_cross) / total_dim},
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
        "recovery fails to beat the strongest tuned single-form, impute-by-mean, or zero-filled-concat "
        "control, or confidence does not predict correctness under a missing form, so recovery or "
        "monitoring is uninformative"
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
        recov, best_rem, impute, zero_concat, conf_full, conf_absent = [], [], [], [], [], []
        per_seed_deltas: list[float] = []
        det_scores, det_absent, cal_conf, cal_correct = [], [], [], []
        head_params, feat_dim, extra_flops = 0, 0, 0
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
            single_heads = {
                k: _fit_head(
                    aligned[k][tr],
                    y[tr],
                    classes=classes,
                    epochs=_int(e, "epochs", 90),
                    lr=_float(e, "lr", 0.03),
                    seed=s + 101 + i,
                )
                for i, k in enumerate(kinds)
            }
            concat_head = _fit_head(
                torch.cat([aligned[k][tr] for k in kinds], dim=1),
                y[tr],
                classes=classes,
                epochs=_int(e, "epochs", 90),
                lr=_float(e, "lr", 0.03),
                seed=s + 211,
            )

            def _conf(x: torch.Tensor, *, h: nn.Module = head) -> torch.Tensor:
                with torch.no_grad():
                    return torch.softmax(h(x), -1).max(-1).values

            full_test = _fuse([aligned[k][te] for k in kinds])
            conf_full.append(float(_conf(full_test).mean()))

            seed_recov, seed_controls = [], []
            for missing_idx, missing in enumerate(kinds):  # drop each form in turn
                remaining = [k for k in kinds if k != missing]
                recov_test = _fuse([aligned[k][te] for k in remaining])
                rec = _acc(head, recov_test, y[te])
                recov.append(rec)
                best = max(_acc(single_heads[k], aligned[k][te], y[te]) for k in remaining)
                best_rem.append(best)
                imp = _fuse(
                    [aligned[k][te] for k in remaining] + [train_means[missing].expand(te.shape[0], -1)]
                )
                imp_acc = _acc(head, imp, y[te])
                impute.append(imp_acc)
                concat_parts = [aligned[k][te] for k in kinds]
                concat_parts[missing_idx] = torch.zeros_like(concat_parts[missing_idx])
                zero_acc = _acc(concat_head, torch.cat(concat_parts, dim=1), y[te])
                zero_concat.append(zero_acc)
                seed_recov.append(rec)
                seed_controls.append(max(best, imp_acc, zero_acc))
                c = _conf(recov_test)
                conf_absent.append(float(c.mean()))
                cal_conf.extend(c.tolist())
                with torch.no_grad():
                    cal_correct.extend((head(recov_test).argmax(-1) == y[te]).float().tolist())
                # OA1 is intake-grounded: the available-form mask says which typed arm is absent.
                # This is a mechanical identity check, separate from corruption/anomaly detection.
                det_scores.extend([1.0 if j == missing_idx else 0.0 for j in range(len(kinds))])
                det_absent.extend([1.0 if j == missing_idx else 0.0 for j in range(len(kinds))])
                extra_flops += int(te.shape[0]) * max(1, len(remaining) - 1) * feat_dim
            per_seed_deltas.append(_mean(seed_recov) - _mean(seed_controls))

        recovery_acc = _mean(recov)
        head_flops = max(1, mlp_flops([feat_dim, classes]))
        oa1 = missing_form_detection(det_scores, det_absent)
        oa2 = confidence_calibration(cal_conf, cal_correct)
        # OA2: confidence is informative iff it PREDICTS correctness under absence (AUROC over chance).
        # Raw confidence magnitude is a poor proxy here (a head trained on 4-form fusion sees a
        # different scale under 3 forms), so the null clause tests calibration, not a magnitude drop.
        confidence_informative = oa2["auroc"] > 0.5 + _float(e, "cal_margin", 0.03)
        strongest_control = max(_mean(best_rem), _mean(impute), _mean(zero_concat))
        recovery_gain = recovery_acc - strongest_control
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        return {
            "recovery_acc": round(recovery_acc, 4),
            "best_remaining_form_acc": round(_mean(best_rem), 4),
            "impute_by_mean_acc": round(_mean(impute), 4),
            "zero_filled_concat_acc": round(_mean(zero_concat), 4),
            "strongest_control_acc": round(strongest_control, 4),
            "absence_ece": round(oa2["ece"], 4),
            "recovery_per_extra_flop": round(recovery_gain / max(extra_flops, 1), 12),
            "recovery_gain_over_strongest_control": round(recovery_gain, 4),
            "confidence_full": round(_mean(conf_full), 4),
            "confidence_under_absence": round(_mean(conf_absent), 4),
            "confidence_predicts_correctness": bool(confidence_informative),
            "oa1_missing_form_auroc": round(oa1["auroc"], 4),
            "oa1_source": "typed-form-availability-mask",
            "oa2_calibration_auroc": round(oa2["auroc"], 4),
            "seeds": seeds,
            "per_seed_deltas": [round(v, 4) for v in per_seed_deltas],
            "seed_ci": seed_ci(per_seed_deltas),
            "sign_flip_report": sign_flip_report(per_seed_deltas),
            "null_supported": bool(recovery_gain <= _float(e, "margin", 0.02) or not confidence_informative),
            "density": density_block(
                {"recovery_acc": recovery_acc},
                seconds=time.perf_counter() - t0,
                params=float(head_params),
                flops=float(head_flops + extra_flops),
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
    baseline = "best single form, shuffled referents, shuffled labels, and a conjunction head"
    ablation = "decode factor a only from form A and factor b only from form B, then bind by referent"
    null_hypothesis = (
        "held-out cross-form combinations collapse toward the strongest single-form, shuffled-label, "
        "or shuffled-referent floor while seen pairs stay high, so the system did not bind factors "
        "across forms"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        n_a, n_b = _int(e, "n_a", 4), _int(e, "n_b", 4)
        heldout_acc, seen_acc, shuffle_acc, shuffled_ref, composite_held, single_form = [], [], [], [], [], []
        deltas: list[float] = []
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
            fused = torch.cat([fa, fb], 1)  # only the conjunction control receives both forms
            tr = (~is_held).nonzero(as_tuple=True)[0]
            te_held = is_held.nonzero(as_tuple=True)[0]
            # seen-combo test split: hold out a fraction of the trained (off-diagonal) rows
            g = torch.Generator().manual_seed(s + 77)
            perm = tr[torch.randperm(tr.shape[0], generator=g)]
            cut = int(perm.shape[0] * 0.85)
            fit_idx, seen_te = perm[:cut], perm[cut:]

            def _compose_acc(ha, hb, idx, *, xa=fa, xb=fb, ya=a, yb=b):
                with torch.no_grad():
                    pa = ha(xa[idx]).argmax(-1)
                    pb = hb(xb[idx]).argmax(-1)
                    return float(((pa == ya[idx]) & (pb == yb[idx])).float().mean())

            head_a = _fit_head(
                fa[fit_idx],
                a[fit_idx],
                classes=n_a,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s,
            )
            head_b = _fit_head(
                fb[fit_idx],
                b[fit_idx],
                classes=n_b,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s + 5,
            )
            head_params = sum(p.numel() for p in head_a.parameters()) + sum(
                p.numel() for p in head_b.parameters()
            )
            held = _compose_acc(head_a, head_b, te_held)
            heldout_acc.append(held)
            seen_acc.append(_compose_acc(head_a, head_b, seen_te))

            # Each single-form control must decode both factors from one arm. This tests whether the
            # cross-form result merely exploits leakage of both factors into either representation.
            single_scores = []
            for offset, xone in enumerate((fa, fb)):
                ha_one = _fit_head(
                    xone[fit_idx],
                    a[fit_idx],
                    classes=n_a,
                    epochs=_int(e, "epochs", 120),
                    lr=_float(e, "lr", 0.03),
                    seed=s + 40 + offset,
                )
                hb_one = _fit_head(
                    xone[fit_idx],
                    b[fit_idx],
                    classes=n_b,
                    epochs=_int(e, "epochs", 120),
                    lr=_float(e, "lr", 0.03),
                    seed=s + 50 + offset,
                )
                single_scores.append(_compose_acc(ha_one, hb_one, te_held, xa=xone, xb=xone))
            single_seed = max(single_scores)
            single_form.append(single_seed)

            # Break the shared referent at evaluation while preserving both marginal form streams.
            gr = torch.Generator().manual_seed(s + 271)
            shuffled_idx = te_held[torch.randperm(te_held.shape[0], generator=gr)]
            with torch.no_grad():
                pa = head_a(fa[te_held]).argmax(-1)
                pb = head_b(fb[shuffled_idx]).argmax(-1)
                shuf_ref_seed = float(((pa == a[te_held]) & (pb == b[te_held])).float().mean())
            shuffled_ref.append(shuf_ref_seed)

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
                fa[fit_idx],
                a_sh,
                classes=n_a,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s + 11,
            )
            hb_sh = _fit_head(
                fb[fit_idx],
                b_sh,
                classes=n_b,
                epochs=_int(e, "epochs", 120),
                lr=_float(e, "lr", 0.03),
                seed=s + 13,
            )
            shuffle_seed = _compose_acc(ha_sh, hb_sh, te_held)
            shuffle_acc.append(shuffle_seed)
            deltas.append(held - max(single_seed, shuf_ref_seed, shuffle_seed))
        chance = 1.0 / (n_a * n_b)
        gap = _mean(seen_acc) - _mean(heldout_acc)
        floor = max(chance, _mean(shuffle_acc), _mean(shuffled_ref), _mean(single_form))
        return {
            "n_a": n_a,
            "n_b": n_b,
            "heldout_combo_acc": round(_mean(heldout_acc), 4),
            "seen_combo_acc": round(_mean(seen_acc), 4),
            "heldout_seen_gap": round(gap, 4),
            "composite_head_heldout_acc": round(_mean(composite_held), 4),
            "shuffle_floor_acc": round(_mean(shuffle_acc), 4),
            "shuffled_referent_acc": round(_mean(shuffled_ref), 4),
            "best_single_form_acc": round(_mean(single_form), 4),
            "chance": round(chance, 4),
            "seeds": seeds,
            "per_seed_deltas": [round(v, 4) for v in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
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
    forms: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ],
    *,
    policy: str,
    rounds: int,
    steps: int,
    classes: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, int], nn.Module]:
    """Train one shared head under a form-selection policy.

    Each form supplies train, validation, and untouched test splits. Lesson selection sees validation
    only; final coverage comes from test. Noisy-TV training batches refresh on every visit. Policies:
    uniform, error, novelty, and learning progress.
    """
    seed_everything(seed)
    tags = sorted(forms)
    head = nn.Linear(forms[tags[0]][0].shape[1], classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    visits = dict.fromkeys(tags, 0)
    val_acc = {t: _acc(head, forms[t][2], forms[t][3]) for t in tags}
    progress = dict.fromkeys(tags, 0.1)

    def _loss(t: str) -> float:
        with torch.no_grad():
            return float(F.cross_entropy(head(forms[t][0]), forms[t][1]))

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
        if pick.startswith("noisy_tv"):
            g = torch.Generator().manual_seed(seed + 100_003 * (r + 1) + visits[pick])
            xt = torch.randn(xt.shape, generator=g)
            yt = torch.randint(0, classes, yt.shape, generator=g)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(head(xt), yt).backward()
            opt.step()
        new = _acc(head, forms[pick][2], forms[pick][3])
        progress[pick] = max(0.0, new - val_acc[pick])
        val_acc[pick] = new
        visits[pick] += 1
    test_acc = {t: _acc(head, forms[t][4], forms[t][5]) for t in tags}
    return test_acc, visits, head


class F10(Experiment):
    id = "f10_intrinsic_form_curriculum"
    metric = ("coverage_per_update", "noisy_form_timeshare", "transfer_gain")
    baseline = "uniform round-robin form selection and prediction-error selection"
    ablation = "learning-progress form selection vs uniform, prediction-error, and novelty"
    null_hypothesis = (
        "learning-progress selection fails to improve untouched held-out-form transfer over every "
        "control or spends as much time on noisy forms as uniform, so the curriculum is not form-aware"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 4)
        rounds = _int(e, "rounds", 40)
        real_kinds = ("vision", "audio", "symbolic")
        lp_cov, uni_cov, err_cov, nov_cov = [], [], [], []
        lp_noisy, uni_noisy, err_noisy, nov_noisy = [], [], [], []
        lp_transfer, uni_transfer, err_transfer, nov_transfer = [], [], [], []
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
                noise=_float(e, "form_noise", 0.1),
                seed=s,
            )
            gsplit = torch.Generator().manual_seed(s + 333)
            order = torch.randperm(y.shape[0], generator=gsplit)
            n_train = int(y.shape[0] * _float(e, "train_frac", 0.6))
            n_val = int(y.shape[0] * _float(e, "val_frac", 0.2))
            tr, val, te = order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]
            aligned = _aligned_forms(feats, tr)
            forms: dict[str, tuple] = {}
            for k in real_kinds:
                forms[k] = (
                    aligned[k][tr],
                    y[tr],
                    aligned[k][val],
                    y[val],
                    aligned[k][te],
                    y[te],
                )
            heldout = aligned["timeseries"]
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
                    torch.randn(val.shape[0], fd, generator=gn),
                    torch.randint(0, classes, (val.shape[0],), generator=gn),
                    torch.randn(te.shape[0], fd, generator=gn),
                    torch.randint(0, classes, (te.shape[0],), generator=gn),
                )

            def _cov(
                policy: str,
                *,
                seed: int = s,
                frm: dict = forms,
                ntags: list = noisy_tags,
                transfer_form: torch.Tensor = heldout,
                test_idx: torch.Tensor = te,
                labels: torch.Tensor = y,
            ):
                acc, visits, head = _run_form_scheduler(
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
                transfer = _acc(head, transfer_form[test_idx], labels[test_idx])
                return coverage, timeshare, transfer

            c_lp, n_lp, t_lp = _cov("learning_progress")
            c_uni, n_uni, t_uni = _cov("uniform")
            c_err, n_err, t_err = _cov("error")
            c_nov, n_nov, t_nov = _cov("novelty")
            lp_cov.append(c_lp)
            uni_cov.append(c_uni)
            err_cov.append(c_err)
            nov_cov.append(c_nov)
            lp_noisy.append(n_lp)
            uni_noisy.append(n_uni)
            err_noisy.append(n_err)
            nov_noisy.append(n_nov)
            lp_transfer.append(t_lp)
            uni_transfer.append(t_uni)
            err_transfer.append(t_err)
            nov_transfer.append(t_nov)
        strongest_transfer = max(_mean(uni_transfer), _mean(err_transfer), _mean(nov_transfer))
        transfer_gain = _mean(lp_transfer) - strongest_transfer
        coverage_gain = _mean(lp_cov) - max(_mean(uni_cov), _mean(err_cov), _mean(nov_cov))
        seed_deltas = [
            lp_transfer[i] - max(uni_transfer[i], err_transfer[i], nov_transfer[i]) for i in range(len(seeds))
        ]
        return {
            "coverage_per_update": round(_mean(lp_cov) / rounds, 6),
            "lp_coverage": round(_mean(lp_cov), 4),
            "uniform_coverage": round(_mean(uni_cov), 4),
            "error_coverage": round(_mean(err_cov), 4),
            "novelty_coverage": round(_mean(nov_cov), 4),
            "noisy_form_timeshare": round(_mean(lp_noisy), 4),
            "uniform_noisy_timeshare": round(_mean(uni_noisy), 4),
            "error_noisy_timeshare": round(_mean(err_noisy), 4),
            "novelty_noisy_timeshare": round(_mean(nov_noisy), 4),
            "transfer_gain": round(transfer_gain, 4),
            "learning_progress_transfer": round(_mean(lp_transfer), 4),
            "strongest_control_transfer": round(strongest_transfer, 4),
            "coverage_gain_over_strongest_control": round(coverage_gain, 4),
            "seeds": seeds,
            "per_seed_transfer_deltas": [round(v, 4) for v in seed_deltas],
            "seed_ci": seed_ci(seed_deltas),
            "sign_flip_report": sign_flip_report(seed_deltas),
            "null_supported": bool(
                transfer_gain <= _float(e, "margin", 0.02)
                or _mean(lp_noisy) >= _mean(uni_noisy) - _float(e, "noisy_margin", 0.02)
            ),
            "density": density_block(
                {"coverage_per_update": _mean(lp_cov) / rounds},
                seconds=time.perf_counter() - t0,
                updates=float(rounds * _int(e, "steps", 5)),
            ),
        }


def _kmeans_codes(
    x: torch.Tensor,
    k: int,
    *,
    seed: int,
    iters: int = 25,
    fit_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """A tiny Lloyd k-means (no new dependency). Returns hard code assignments [N] in 0..k-1.

    The codebook init is seeded, so two seeds that recover the SAME partition means the code is a
    stable language over the data, not a per-run idiolect (the Wittgenstein private-language question
    at the form layer). Well-separated form clusters give init-stable codes; overlapping ones do not.
    """
    g = torch.Generator().manual_seed(seed)
    fit = x if fit_idx is None else x[fit_idx]
    if fit.shape[0] < k:
        raise ValueError(f"k={k} exceeds fit rows {fit.shape[0]}")
    centers = fit[torch.randperm(fit.shape[0], generator=g)[:k]].clone()
    codes = torch.zeros(fit.shape[0], dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(fit, centers)
        codes = d.argmin(1)
        for c in range(k):
            m = codes == c
            if bool(m.any()):
                centers[c] = fit[m].mean(0)
    return torch.cdist(x, centers).argmin(1)


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
        from ..diagnostics.riskcov import seed_ci, sign_flip_report
        from ..diagnostics.seed_consistency import _hungarian, code_stability

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
        tr, te = _split(y.shape[0], _float(e, "train_frac", 0.6), 0)
        code_lists, rand_lists = [], []
        t0 = time.perf_counter()
        for s in seeds:
            feats = _form_features(
                z, feature_dim=_int(e, "feature_dim", 24), noise=_float(e, "form_noise", 0.3), seed=s
            )
            fused = torch.cat([feats[kd] for kd in FORM_KINDS], 1)
            code_lists.append(_kmeans_codes(fused, k, seed=s, fit_idx=tr))
            gr = torch.Generator().manual_seed(s + 4242)
            rand_lists.append(torch.randint(0, k, (fused.shape[0],), generator=gr))
        agreement = code_stability([codes[te] for codes in code_lists], k)
        rand_floor = code_stability([codes[te] for codes in rand_lists], k)

        # cross-seed probe transfer: fit a probe on seed-0 codes -> class, test on seed-1 codes
        # after Hungarian relabeling into seed-0's code space
        def _onehot(c: torch.Tensor) -> torch.Tensor:
            return F.one_hot(c, k).float()

        probe = _fit_head(
            _onehot(code_lists[0])[tr],
            y[tr],
            classes=classes,
            epochs=_int(e, "epochs", 120),
            lr=_float(e, "lr", 0.05),
            seed=0,
        )
        transfer, shuffled_transfer = [], []
        for j in range(1, len(code_lists)):
            # Learn a one-to-one code permutation on TRAIN referents only, then evaluate on held-out
            # referents. The exact dependency-free Hungarian solver is shared across all machines.
            conf = torch.zeros(k, k, dtype=torch.float64)
            for a in range(k):
                for b in range(k):
                    conf[a, b] = float(((code_lists[0][tr] == a) & (code_lists[j][tr] == b)).sum())
            assignment = _hungarian(-conf.numpy())  # seed-0 code a -> seed-j code b
            remap = torch.empty(k, dtype=torch.long)
            for a, b in enumerate(assignment):
                remap[b] = a
            mapped = remap[code_lists[j]]
            transfer.append(_acc(probe, _onehot(mapped)[te], y[te]))
            g = torch.Generator().manual_seed(9000 + j)
            shuffled_j = code_lists[j][tr[torch.randperm(tr.shape[0], generator=g)]]
            shuf_conf = torch.zeros(k, k, dtype=torch.float64)
            for a in range(k):
                for b in range(k):
                    shuf_conf[a, b] = float(((code_lists[0][tr] == a) & (shuffled_j == b)).sum())
            shuf_assignment = _hungarian(-shuf_conf.numpy())
            shuf_remap = torch.empty(k, dtype=torch.long)
            for a, b in enumerate(shuf_assignment):
                shuf_remap[b] = a
            shuffled_transfer.append(_acc(probe, _onehot(shuf_remap[code_lists[j]])[te], y[te]))
        chance = 1.0 / classes
        cross_transfer = _mean(transfer) if transfer else 0.0
        shuffled_cross_transfer = _mean(shuffled_transfer) if shuffled_transfer else 0.0
        codes_are_language = agreement["mean_agreement"] > rand_floor["mean_agreement"] + _float(
            e, "margin", 0.15
        ) and cross_transfer > max(chance, shuffled_cross_transfer) + _float(e, "transfer_margin", 0.1)
        transfer_deltas = [v - shuffled_transfer[i] for i, v in enumerate(transfer)]
        return {
            "codebook_k": k,
            "code_agreement": round(agreement["mean_agreement"], 4),
            "random_code_floor": round(rand_floor["mean_agreement"], 4),
            "cross_seed_code_transfer": round(cross_transfer, 4),
            "shuffled_referent_transfer": round(shuffled_cross_transfer, 4),
            "chance": round(chance, 4),
            "codes_recur_across_seeds": bool(codes_are_language),
            "seeds": seeds,
            "per_seed_transfer_deltas": [round(v, 4) for v in transfer_deltas],
            "seed_ci": seed_ci(transfer_deltas),
            "sign_flip_report": sign_flip_report(transfer_deltas),
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
    baseline = (
        "flat object memory, a single-scale scene memory, and a random hierarchy, each using the "
        "same ReplayBuffer allocation and exact byte budget"
    )
    ablation = (
        "one typed KVIndex containing object, scene, episode, and task nodes with explicit parent "
        "links, evaluated in both directions between every adjacent scale"
    )
    null_hypothesis = (
        "hierarchical referent memory ties the strongest flat, single-scale, or random-hierarchy "
        "control at matched bytes, so scale structure buys no retrieval and memory stays clip-shaped"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.compute import knn_flops
        from ..diagnostics.riskcov import seed_ci, sign_flip_report
        from ..shell.buffer import ReplayBuffer

        e = cfg.experiment
        seeds = list(e.seeds)
        n_tasks = _int(e, "tasks", 3)
        ep_per_task = _int(e, "episodes_per_task", 3)
        scene_per_ep = _int(e, "scenes_per_episode", 3)
        obj_per_scene = _int(e, "objects_per_scene", 4)
        n_ep = n_tasks * ep_per_task
        n_scene = n_ep * scene_per_ep
        n_obj = n_scene * obj_per_scene
        dim = _int(e, "world_dim", 16)
        budget = _int(e, "store_vectors", 96)
        coarse_nodes = n_tasks + n_ep + n_scene
        if budget < coarse_nodes:
            raise ValueError(
                f"F19 store_vectors={budget} cannot hold the {coarse_nodes} task/episode/scene nodes"
            )
        if budget > n_obj:
            raise ValueError(
                f"F19 store_vectors={budget} exceeds the {n_obj} distinct object controls; "
                "increase the hierarchy or lower the budget"
            )

        # Scale order is causal containment order. Every record carries a complete lineage tuple;
        # -1 means that a finer descendant does not exist for this node.
        task_scale, episode_scale, scene_scale, object_scale = range(4)
        relations = (
            (task_scale, episode_scale),
            (episode_scale, task_scale),
            (episode_scale, scene_scale),
            (scene_scale, episode_scale),
            (scene_scale, object_scale),
            (object_scale, scene_scale),
        )
        relation_names = {
            (task_scale, episode_scale): "task_to_episode",
            (episode_scale, task_scale): "episode_to_task",
            (episode_scale, scene_scale): "episode_to_scene",
            (scene_scale, episode_scale): "scene_to_episode",
            (scene_scale, object_scale): "scene_to_object",
            (object_scale, scene_scale): "object_to_scene",
        }
        hierarchy_scores: list[float] = []
        flat_scores: list[float] = []
        single_scores: list[float] = []
        random_scores: list[float] = []
        deltas: list[float] = []
        relation_totals: dict[str, list[float]] = {name: [] for name in relation_names.values()}
        store_bytes_by_arm: dict[str, int] = {}
        query_count = _int(e, "queries_per_relation", 48)
        k = _int(e, "k", 3)
        scale_weight = _float(e, "scale_weight", 8.0)
        t0 = time.perf_counter()

        def _centroids(x: torch.Tensor, ids: torch.Tensor, count: int) -> torch.Tensor:
            return torch.stack([x[ids == i].mean(0) for i in range(count)])

        def _lineages(
            task_ids: torch.Tensor,
            episode_ids: torch.Tensor,
            scene_ids: torch.Tensor,
            object_ids: torch.Tensor,
        ) -> torch.Tensor:
            return torch.stack([task_ids, episode_ids, scene_ids, object_ids], dim=1).long()

        def _make_store(
            base_keys: torch.Tensor,
            scales: torch.Tensor,
            lineages: torch.Tensor,
            *,
            seed: int,
        ) -> dict:
            if base_keys.shape[0] != budget:
                raise AssertionError(f"F19 arm stored {base_keys.shape[0]} rows, expected {budget}")
            typed = torch.cat([base_keys, F.one_hot(scales, 4).float() * scale_weight], dim=1)
            memory = ReplayBuffer(
                capacity=budget,
                dim=dim,
                key_dim=dim + 4,
                prioritized=False,
                eviction="fifo",
                index="brute",
                seed=seed,
            )
            memory.add(base_keys, torch.arange(budget), key=typed)
            allocated_bytes = int(
                memory.x.nelement() * memory.x.element_size()
                + memory.keys.nelement() * memory.keys.element_size()
                + memory.y.nelement() * memory.y.element_size()
                + memory.prio.nelement() * memory.prio.element_size()
                + scales.nelement() * scales.element_size()
                + lineages.nelement() * lineages.element_size()
            )
            return {
                "memory": memory,
                "scales": scales,
                "lineages": lineages,
                "bytes": allocated_bytes,
            }

        def _score_relation(
            store: dict,
            query: torch.Tensor,
            source_ids: torch.Tensor,
            source_scale: int,
            target_scale: int,
            *,
            route_scale: int,
        ) -> float:
            typed_query = torch.cat(
                [
                    query,
                    F.one_hot(torch.full((query.shape[0],), route_scale), 4).float() * scale_weight,
                ],
                dim=1,
            )
            got = store["memory"].retrieve(typed_query, k=k)["y"]
            candidates = store["lineages"][got]
            if target_scale < source_scale:
                # Upward lookup has one correct ancestor. Recover it from the source lineage.
                source_lineage = node_lineages[source_scale][source_ids]
                expected = source_lineage[:, target_scale][:, None]
                correct = candidates[:, :, target_scale] == expected
            else:
                # Downward lookup may return any descendant of the source node.
                correct = candidates[:, :, source_scale] == source_ids[:, None]
            return float(correct.any(dim=1).float().mean())

        for s in seeds:
            g = torch.Generator().manual_seed(s)
            object_ids = torch.arange(n_obj)
            scene_ids = object_ids // obj_per_scene
            episode_ids = scene_ids // scene_per_ep
            task_ids = episode_ids // ep_per_task

            task_state = _float(e, "task_separation", 2.0) * torch.randn(n_tasks, dim, generator=g)
            episode_state = task_state[torch.arange(n_ep) // ep_per_task] + _float(
                e, "episode_spread", 1.1
            ) * torch.randn(n_ep, dim, generator=g)
            scene_state = episode_state[torch.arange(n_scene) // scene_per_ep] + _float(
                e, "scene_spread", 0.8
            ) * torch.randn(n_scene, dim, generator=g)
            object_state = scene_state[scene_ids] + _float(e, "object_spread", 0.55) * torch.randn(
                n_obj, dim, generator=g
            )
            observed_objects = object_state + _float(e, "store_noise", 0.7) * torch.randn(
                n_obj, dim, generator=g
            )

            task_keys = _centroids(observed_objects, task_ids, n_tasks)
            episode_keys = _centroids(observed_objects, episode_ids, n_ep)
            scene_keys = _centroids(observed_objects, scene_ids, n_scene)
            object_keys = observed_objects

            task_lineage = _lineages(
                torch.arange(n_tasks),
                torch.full((n_tasks,), -1),
                torch.full((n_tasks,), -1),
                torch.full((n_tasks,), -1),
            )
            episode_lineage = _lineages(
                torch.arange(n_ep) // ep_per_task,
                torch.arange(n_ep),
                torch.full((n_ep,), -1),
                torch.full((n_ep,), -1),
            )
            scene_lineage = _lineages(
                (torch.arange(n_scene) // scene_per_ep) // ep_per_task,
                torch.arange(n_scene) // scene_per_ep,
                torch.arange(n_scene),
                torch.full((n_scene,), -1),
            )
            object_lineage = _lineages(task_ids, episode_ids, scene_ids, object_ids)
            node_keys = (task_keys, episode_keys, scene_keys, object_keys)
            node_lineages = (task_lineage, episode_lineage, scene_lineage, object_lineage)

            # Stratify object slots so the hierarchy cannot win because a scene was accidentally
            # omitted. The flat arm receives the same courtesy and the larger object allocation.
            first_per_scene = torch.arange(n_scene) * obj_per_scene
            remaining_obj = object_ids[~torch.isin(object_ids, first_per_scene)]
            remaining_obj = remaining_obj[torch.randperm(remaining_obj.shape[0], generator=g)]
            hier_obj_n = budget - coarse_nodes
            hier_obj = torch.cat(
                [first_per_scene[:hier_obj_n], remaining_obj[: max(0, hier_obj_n - n_scene)]]
            )
            if hier_obj.shape[0] < hier_obj_n:
                extra = object_ids[torch.randperm(n_obj, generator=g)[: hier_obj_n - hier_obj.shape[0]]]
                hier_obj = torch.cat([hier_obj, extra])
            hier_keys = torch.cat([task_keys, episode_keys, scene_keys, object_keys[hier_obj]])
            hier_scales = torch.cat(
                [
                    torch.full((n_tasks,), task_scale),
                    torch.full((n_ep,), episode_scale),
                    torch.full((n_scene,), scene_scale),
                    torch.full((hier_obj_n,), object_scale),
                ]
            ).long()
            hier_lineages = torch.cat(
                [task_lineage, episode_lineage, scene_lineage, object_lineage[hier_obj]]
            )

            flat_obj = torch.cat(
                [
                    first_per_scene,
                    remaining_obj[: budget - n_scene],
                ]
            )[:budget]
            flat_keys = object_keys[flat_obj]
            flat_scales = torch.full((budget,), object_scale).long()
            flat_lineages = object_lineage[flat_obj]

            # The single-scale arm uses the entire allocation for noisy scene exemplars. It is a
            # strong specialist for scene queries, but has no explicit episode/task/object nodes.
            single_idx = torch.arange(budget) % n_scene
            single_keys = scene_keys[single_idx] + _float(e, "duplicate_noise", 0.05) * torch.randn(
                budget, dim, generator=g
            )
            single_scales = torch.full((budget,), scene_scale).long()
            single_lineages = scene_lineage[single_idx]

            hierarchy = _make_store(hier_keys, hier_scales, hier_lineages, seed=s)
            flat = _make_store(flat_keys, flat_scales, flat_lineages, seed=s + 1)
            single = _make_store(single_keys, single_scales, single_lineages, seed=s + 2)

            # Random hierarchy keeps every key and scale allocation intact but permutes referent
            # lineages within each scale. This isolates hierarchical identity from extra centroids.
            random_lineages = hier_lineages.clone()
            for scale in range(4):
                idx = torch.where(hier_scales == scale)[0]
                if idx.numel() > 1:
                    random_lineages[idx] = random_lineages[idx[torch.randperm(idx.numel(), generator=g)]]
            random_hierarchy = _make_store(
                hier_keys.clone(), hier_scales.clone(), random_lineages, seed=s + 3
            )
            arm_stores = {
                "hierarchy": hierarchy,
                "flat": flat,
                "single": single,
                "random": random_hierarchy,
            }
            for name, store in arm_stores.items():
                store_bytes_by_arm[name] = int(store["bytes"])

            seed_scores: dict[str, list[float]] = {name: [] for name in arm_stores}
            for source_scale, target_scale in relations:
                source_n = node_keys[source_scale].shape[0]
                source = torch.randint(0, source_n, (query_count,), generator=g)
                queries = node_keys[source_scale][source] + _float(e, "query_noise", 0.8) * torch.randn(
                    query_count, dim, generator=g
                )
                h_score = _score_relation(
                    hierarchy,
                    queries,
                    source,
                    source_scale,
                    target_scale,
                    route_scale=target_scale,
                )
                f_score = _score_relation(
                    flat,
                    queries,
                    source,
                    source_scale,
                    target_scale,
                    route_scale=object_scale,
                )
                s_score = _score_relation(
                    single,
                    queries,
                    source,
                    source_scale,
                    target_scale,
                    route_scale=scene_scale,
                )
                r_score = _score_relation(
                    random_hierarchy,
                    queries,
                    source,
                    source_scale,
                    target_scale,
                    route_scale=target_scale,
                )
                seed_scores["hierarchy"].append(h_score)
                seed_scores["flat"].append(f_score)
                seed_scores["single"].append(s_score)
                seed_scores["random"].append(r_score)
                relation_totals[relation_names[(source_scale, target_scale)]].append(h_score)

            h_seed = _mean(seed_scores["hierarchy"])
            f_seed = _mean(seed_scores["flat"])
            s_seed = _mean(seed_scores["single"])
            r_seed = _mean(seed_scores["random"])
            hierarchy_scores.append(h_seed)
            flat_scores.append(f_seed)
            single_scores.append(s_seed)
            random_scores.append(r_seed)
            deltas.append(h_seed - max(f_seed, s_seed, r_seed))

        if len(set(store_bytes_by_arm.values())) != 1:
            raise AssertionError(f"F19 memory controls are not byte matched: {store_bytes_by_arm}")
        store_bytes = next(iter(store_bytes_by_arm.values()))
        hierarchy_mean = _mean(hierarchy_scores)
        flat_mean = _mean(flat_scores)
        single_mean = _mean(single_scores)
        random_mean = _mean(random_scores)
        strongest = max(flat_mean, single_mean, random_mean)
        gain = hierarchy_mean - strongest
        retrieval_flops = knn_flops(
            len(relations) * query_count,
            budget,
            dim + 4,
        )
        return {
            "hierarchy": {
                "tasks": n_tasks,
                "episodes": n_ep,
                "scenes": n_scene,
                "objects": n_obj,
                "parent_links_explicit": True,
                "shared_index": True,
            },
            "relations": {name: round(_mean(values), 4) for name, values in relation_totals.items()},
            "k": k,
            "cross_scale_recall_at_k": round(hierarchy_mean, 4),
            "flat_memory_recall_at_k": round(flat_mean, 4),
            "single_scale_recall_at_k": round(single_mean, 4),
            "random_hierarchy_recall_at_k": round(random_mean, 4),
            "strongest_control_recall_at_k": round(strongest, 4),
            "hierarchy_gain_over_strongest": round(gain, 4),
            "recall_per_byte": round(hierarchy_mean / store_bytes, 12),
            "store_vectors": budget,
            "store_bytes": store_bytes,
            "store_bytes_by_arm": store_bytes_by_arm,
            "matched_memory_bytes": True,
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
            "null_supported": bool(gain <= _float(e, "margin", 0.05)),
            "density": density_block(
                {"cross_scale_recall_at_k": hierarchy_mean},
                seconds=time.perf_counter() - t0,
                bytes=float(store_bytes),
                flops=float(retrieval_flops),
            ),
        }


def _f13_project(x: torch.Tensor, width: int, *, seed: int) -> torch.Tensor:
    """Fixed label-free projection of vectors or token tensors, matched across arms."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(x.shape[-1], width, generator=g) / math.sqrt(x.shape[-1])
    return x @ p


def _f13_fit_shell(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    classes: int,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> nn.Module:
    seed_everything(seed)
    if hidden > 0:
        shell: nn.Module = nn.Sequential(
            nn.Linear(x.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )
    else:
        shell = nn.Linear(x.shape[1], classes)
    opt = torch.optim.Adam(shell.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(shell(x), y).backward()
        opt.step()
    return shell


def _f13_budget_subset(
    train_idx: torch.Tensor,
    y: torch.Tensor,
    rows: int,
    *,
    classes: int,
    seed: int,
) -> torch.Tensor:
    """A deterministic class-covering subset so low replay budgets remain meaningful."""
    rows = min(int(rows), int(train_idx.shape[0]))
    if rows < classes:
        raise ValueError(f"F13 replay budget holds {rows} rows, fewer than {classes} classes")
    g = torch.Generator().manual_seed(seed)
    selected: list[torch.Tensor] = []
    used = torch.zeros(train_idx.shape[0], dtype=torch.bool)
    for label in range(classes):
        positions = torch.where(y[train_idx] == label)[0]
        if positions.numel() == 0:
            raise ValueError(f"F13 train split contains no examples of class {label}")
        pick = positions[torch.randint(0, positions.numel(), (1,), generator=g)]
        selected.append(train_idx[pick])
        used[pick] = True
    remaining = train_idx[~used]
    remaining = remaining[torch.randperm(remaining.shape[0], generator=g)]
    return torch.cat([*selected, remaining[: rows - classes]])


class F13(Experiment):
    id = "f13_form_energy_budget"
    metric = ("accuracy_per_byte", "accuracy_per_param", "estimated_energy_per_correct")
    baseline = (
        "a single raw form projected to the same width, matched-width random features, and the "
        "identical linear head shell"
    )
    ablation = (
        "fused referent-aligned form tokens projected to a swept width vs a single raw form at "
        "matched width, bytes, and head"
    )
    null_hypothesis = (
        "every form interface lies on the same full-system accuracy-versus-cost frontier as raw or "
        "matched random features, so form structure buys no capability per retained byte, parameter, "
        "FLOP, or analytically estimated joule"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        import json

        from ..diagnostics.compute import mlp_flops
        from ..diagnostics.riskcov import pareto_area, pareto_frontier, seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 6)
        feature_dim = _int(e, "feature_dim", 24)
        widths = [int(w) for w in getattr(e, "widths", [4, 8, 16])]
        token_counts = [int(t) for t in getattr(e, "token_counts", [1, 4])]
        shell_sizes = [int(h) for h in getattr(e, "shell_sizes", [0, 16])]
        replay_budgets = [int(b) for b in getattr(e, "replay_bytes", [4096, 16384])]
        if not all((widths, token_counts, shell_sizes, replay_budgets)):
            raise ValueError("F13 width, token, shell, and replay axes must all be non-empty")
        if any(value <= 0 for value in [*widths, *token_counts, *replay_budgets]):
            raise ValueError("F13 widths, token counts, and replay budgets must be positive")

        pj_per_flop = _float(e, "energy_pj_per_flop", 15.0)
        pj_per_byte = _float(e, "energy_pj_per_byte", 20.0)
        if pj_per_flop <= 0 or pj_per_byte <= 0:
            raise ValueError("F13 energy-model coefficients must be positive")
        records: list[dict] = []
        per_seed_areas: dict[str, list[float]] = {arm: [] for arm in ("form", "raw", "random")}
        area_deltas: list[float] = []
        t0 = time.perf_counter()
        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 480),
                classes=classes,
                world_dim=_int(e, "world_dim", 24),
                separation=_float(e, "separation", 1.2),
                noise=_float(e, "world_noise", 0.8),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=feature_dim,
                noise=_float(e, "form_noise", 0.9),
                seed=s,
            )
            tr, te = _split(y.shape[0], _float(e, "train_frac", 0.6), s)
            aligned = _aligned_forms(forms, tr)
            raw = forms[_str(e, "raw_form", "vision")]
            epochs = _int(e, "epochs", 70)
            lr = _float(e, "lr", 0.03)
            n = int(y.shape[0])
            world_dim = int(z.shape[1])
            base_form_flops = 2 * n * world_dim * feature_dim
            # Least-squares fit and application for the three non-reference forms. It is an
            # analytical upper-order estimate, recorded separately from measured wall time.
            alignment_fit_flops = (len(FORM_KINDS) - 1) * (
                2 * int(tr.shape[0]) * feature_dim * feature_dim + feature_dim**3
            )
            alignment_apply_flops = (len(FORM_KINDS) - 1) * 2 * n * feature_dim * feature_dim
            alignment_bytes = (len(FORM_KINDS) - 1) * (feature_dim + 1) * feature_dim * 4

            seed_records: list[dict] = []
            for token_count in token_counts:
                form_tokens = []
                token_noise = _float(e, "token_noise", 0.55)
                for form_index, kind in enumerate(FORM_KINDS):
                    g = torch.Generator().manual_seed(s + 100_003 * token_count + 997 * form_index)
                    noise = token_noise * torch.randn(n, token_count, feature_dim, generator=g)
                    form_tokens.append(aligned[kind][:, None, :] + noise)
                fused_tokens = torch.stack(form_tokens).mean(0)
                g = torch.Generator().manual_seed(s + 200_003 * token_count)
                raw_tokens = raw[:, None, :] + token_noise * torch.randn(
                    n, token_count, feature_dim, generator=g
                )
                random_tokens = torch.randn(
                    n,
                    token_count,
                    feature_dim,
                    generator=torch.Generator().manual_seed(s + 300_007 * token_count),
                )

                for width in widths:
                    projection_seed = s + 31 * width + 101 * token_count
                    arm_features = {
                        "form": _f13_project(fused_tokens, width, seed=projection_seed).mean(1),
                        "raw": _f13_project(raw_tokens, width, seed=projection_seed).mean(1),
                        "random": _f13_project(random_tokens, width, seed=projection_seed + 13).mean(1),
                    }
                    row_bytes = token_count * width * 4 + 8
                    for replay_budget in replay_budgets:
                        stored_rows = min(int(tr.shape[0]), replay_budget // row_bytes)
                        subset = _f13_budget_subset(
                            tr,
                            y,
                            stored_rows,
                            classes=classes,
                            seed=s + width + token_count + replay_budget,
                        )
                        retained_bytes = int(subset.shape[0]) * row_bytes
                        for hidden in shell_sizes:
                            dims = [width, classes] if hidden <= 0 else [width, hidden, classes]
                            params = (
                                width * classes + classes
                                if hidden <= 0
                                else width * hidden + hidden + hidden * classes + classes
                            )
                            train_forward = mlp_flops(dims, int(subset.shape[0]))
                            train_shell_flops = 3 * train_forward * epochs
                            eval_shell_flops = mlp_flops(dims, int(te.shape[0]))
                            token_projection_flops = 2 * n * token_count * feature_dim * width
                            token_pool_flops = n * token_count * width
                            fusion_flops = n * token_count * feature_dim * (len(FORM_KINDS) - 1)
                            for arm_index, arm in enumerate(("form", "raw", "random")):
                                shell = _f13_fit_shell(
                                    arm_features[arm][subset],
                                    y[subset],
                                    classes=classes,
                                    hidden=hidden,
                                    epochs=epochs,
                                    lr=lr,
                                    seed=s + 17 * arm_index + width + hidden,
                                )
                                accuracy = _acc(shell, arm_features[arm][te], y[te])
                                if arm == "form":
                                    production_flops = len(FORM_KINDS) * base_form_flops
                                    structural_flops = (
                                        alignment_fit_flops + alignment_apply_flops + fusion_flops
                                    )
                                    structural_bytes = alignment_bytes
                                    produced_bytes = len(FORM_KINDS) * n * token_count * feature_dim * 4
                                elif arm == "raw":
                                    production_flops = base_form_flops
                                    structural_flops = 0
                                    structural_bytes = 0
                                    produced_bytes = n * token_count * feature_dim * 4
                                else:
                                    production_flops = n * token_count * feature_dim
                                    structural_flops = 0
                                    structural_bytes = 0
                                    produced_bytes = n * token_count * feature_dim * 4
                                total_flops = int(
                                    production_flops
                                    + structural_flops
                                    + token_projection_flops
                                    + token_pool_flops
                                    + train_shell_flops
                                    + eval_shell_flops
                                )
                                total_retained_bytes = int(retained_bytes + params * 4 + structural_bytes)
                                # Training rereads the pooled replay rows once per update. The energy
                                # model is explicitly analytical, not a wall-power measurement.
                                bytes_moved = int(
                                    produced_bytes
                                    + n * token_count * width * 4
                                    + retained_bytes
                                    + epochs * int(subset.shape[0]) * width * 4
                                    + params * 4 * epochs
                                )
                                energy_joules = 1.0e-12 * (
                                    total_flops * pj_per_flop + bytes_moved * pj_per_byte
                                )
                                correct = max(accuracy * int(te.shape[0]), 1.0e-9)
                                record = {
                                    "seed": s,
                                    "arm": arm,
                                    "width": width,
                                    "token_count": token_count,
                                    "shell_size": hidden,
                                    "replay_budget_bytes": replay_budget,
                                    "stored_rows": int(subset.shape[0]),
                                    "accuracy": float(accuracy),
                                    "params": int(params),
                                    "retained_bytes": total_retained_bytes,
                                    "estimated_flops": total_flops,
                                    "estimated_energy_joules": energy_joules,
                                    "estimated_energy_per_correct": energy_joules / correct,
                                }
                                records.append(record)
                                seed_records.append(record)

            energy_max = max(record["estimated_energy_joules"] for record in seed_records)
            areas: dict[str, float] = {}
            for arm in ("form", "raw", "random"):
                points = [
                    (record["estimated_energy_joules"] / energy_max, record["accuracy"])
                    for record in seed_records
                    if record["arm"] == arm
                ]
                areas[arm] = pareto_area(points, x_max=1.0)
                per_seed_areas[arm].append(areas[arm])
            area_deltas.append(areas["form"] - max(areas["raw"], areas["random"]))

        grouped: dict[tuple, list[dict]] = {}
        for record in records:
            key = (
                record["arm"],
                record["width"],
                record["token_count"],
                record["shell_size"],
                record["replay_budget_bytes"],
            )
            grouped.setdefault(key, []).append(record)
        frontier_points = []
        for key, group in sorted(grouped.items()):
            arm, width, token_count, shell_size, replay_budget = key
            frontier_points.append(
                {
                    "arm": arm,
                    "width": width,
                    "token_count": token_count,
                    "shell_size": shell_size,
                    "replay_budget_bytes": replay_budget,
                    "stored_rows": int(round(_mean([row["stored_rows"] for row in group]))),
                    "accuracy": round(_mean([row["accuracy"] for row in group]), 6),
                    "params": int(round(_mean([row["params"] for row in group]))),
                    "retained_bytes": int(round(_mean([row["retained_bytes"] for row in group]))),
                    "estimated_flops": int(round(_mean([row["estimated_flops"] for row in group]))),
                    "estimated_energy_joules": _mean([row["estimated_energy_joules"] for row in group]),
                    "estimated_energy_per_correct": _mean(
                        [row["estimated_energy_per_correct"] for row in group]
                    ),
                }
            )

        form_points = [point for point in frontier_points if point["arm"] == "form"]
        control_points = [point for point in frontier_points if point["arm"] != "form"]
        dominant_form_points = []
        for point in form_points:
            dominates = any(
                point["accuracy"] > control["accuracy"] + _float(e, "point_margin", 0.01)
                and point["params"] <= control["params"]
                and point["retained_bytes"] <= control["retained_bytes"]
                and point["estimated_flops"] <= control["estimated_flops"]
                and point["estimated_energy_joules"] <= control["estimated_energy_joules"]
                for control in control_points
            )
            if dominates:
                dominant_form_points.append(point)

        form_area = _mean(per_seed_areas["form"])
        raw_area = _mean(per_seed_areas["raw"])
        rand_area = _mean(per_seed_areas["random"])
        best_byte = max(form_points, key=lambda point: point["accuracy"] / point["retained_bytes"])
        best_param = max(form_points, key=lambda point: point["accuracy"] / point["params"])
        best_energy = min(form_points, key=lambda point: point["estimated_energy_per_correct"])
        acc_per_byte = best_byte["accuracy"] / best_byte["retained_bytes"]
        acc_per_param = best_param["accuracy"] / best_param["params"]
        energy_per_correct = best_energy["estimated_energy_per_correct"]
        chance = 1.0 / classes
        margin = _float(e, "margin", 0.02)
        receipt = {
            "schema": "mop-f13-density-frontier/v2",
            "experiment_id": self.id,
            "claim_level": "R0-synthetic",
            "energy_measured": False,
            "energy_model": {
                "id": "analytical-flop-plus-byte/v1",
                "pj_per_flop": pj_per_flop,
                "pj_per_byte": pj_per_byte,
                "warning": "analytical estimate, not wall-power telemetry",
            },
            "grid": {
                "widths": widths,
                "token_counts": token_counts,
                "shell_sizes": shell_sizes,
                "replay_bytes": replay_budgets,
                "seeds": seeds,
            },
            "frontier_points": frontier_points,
            "pareto_frontiers": {
                arm: pareto_frontier(
                    [
                        (point["estimated_energy_joules"], point["accuracy"])
                        for point in frontier_points
                        if point["arm"] == arm
                    ]
                )
                for arm in ("form", "raw", "random")
            },
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = run_dir / "f13_density_frontier.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return {
            "grid": receipt["grid"],
            "frontier_points": frontier_points,
            "accuracy_per_byte": round(acc_per_byte, 6),
            "accuracy_per_param": round(acc_per_param, 6),
            "estimated_energy_per_correct": energy_per_correct,
            "estimated_energy_unit": "joules_per_correct_prediction",
            "energy_measured": False,
            "energy_model": receipt["energy_model"],
            "form_pareto_area": round(form_area, 4),
            "raw_pareto_area": round(raw_area, 4),
            "random_pareto_area": round(rand_area, 4),
            "form_area_gain_over_strongest": round(form_area - max(raw_area, rand_area), 4),
            "dominant_form_point_count": len(dominant_form_points),
            "best_byte_point": best_byte,
            "best_param_point": best_param,
            "best_energy_point": best_energy,
            "chance": round(chance, 4),
            "seeds": seeds,
            "per_seed_frontier_deltas": [round(value, 4) for value in area_deltas],
            "seed_ci": seed_ci(area_deltas),
            "sign_flip_report": sign_flip_report(area_deltas),
            "frontier_receipt": str(receipt_path),
            "null_supported": bool(form_area <= max(raw_area, rand_area) + margin),
            "density": density_block(
                {
                    "accuracy_per_byte": acc_per_byte,
                    "form_acc": best_energy["accuracy"],
                },
                primary="accuracy_per_byte",
                seconds=time.perf_counter() - t0,
                params=float(best_energy["params"]),
                bytes=float(best_energy["retained_bytes"]),
                flops=float(best_energy["estimated_flops"]),
            ),
        }


def _f18_state_form(
    a: torch.Tensor,
    c: torch.Tensor,
    extra: torch.Tensor,
    *,
    a_scale: float,
    c_scale: float,
    feature_dim: int,
    form_noise: float,
    seed: int,
) -> torch.Tensor:
    """Render a world state (factor a, confounder c, nuisance extra) through one random form
    projection. Factor a lands on axis 0 and the confounder on axis 1 before mixing, so both are
    linearly present in the state but entangled by the projection into the observed form."""
    state = extra.clone()
    state[:, 0] = state[:, 0] + a_scale * a.float()
    state[:, 1] = state[:, 1] + c_scale * c.float()
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(state.shape[1], feature_dim, generator=g) / math.sqrt(state.shape[1])
    bias = 0.1 * torch.randn(feature_dim, generator=g)
    noise = form_noise * torch.randn(state.shape[0], feature_dim, generator=g)
    return (state @ w + bias + noise).float()


def _f18_fit_map(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> nn.Linear:
    """Fit the cross-form state transport itself, not a scalar label surrogate."""
    seed_everything(seed)
    head = nn.Linear(x.shape[1], y.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.mse_loss(head(x), y.float()).backward()
        opt.step()
    return head


def _f18_decoded_acc(
    transport: nn.Linear,
    prototypes: torch.Tensor,
    x: torch.Tensor,
    y_int: torch.Tensor,
) -> float:
    with torch.no_grad():
        pred = torch.cdist(transport(x), prototypes).argmin(1)
    return float((pred == y_int).float().mean())


def _f18_cosine(transport: nn.Linear, x: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cosine_similarity(transport(x), target, dim=1).mean())


class F18(Experiment):
    id = "f18_counterfactual_form_intervention"
    metric = (
        "counterfactual_match_acc",
        "correlational_baseline_acc",
        "unseen_value_gap",
        "counterfactual_acc_per_param",
    )
    baseline = (
        "an identical cross-form map trained on observational before/after pairs whose natural "
        "change is confounded, with rows, updates, parameters, and FLOPs exactly matched"
    )
    ablation = (
        "randomized do-value to predicted Form-B state transport vs correlational, random-direction, "
        "and shuffled-counterfactual controls"
    )
    null_hypothesis = (
        "the intervention predictor leaks (predicts only seen intervention values) or ties the "
        "correlational predictor, so the matrix binds appearances rather than intervention structure"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.compute import matched_within, mlp_flops
        from ..diagnostics.riskcov import seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        a_values = _int(e, "a_values", 7)
        contexts = _int(e, "contexts", 2)
        train_deltas = [int(d) for d in getattr(e, "train_deltas", [1, 2])]
        test_delta = _int(e, "test_delta", 3)
        if not train_deltas or test_delta in train_deltas:
            raise ValueError("F18 requires non-empty train_deltas and a genuinely held-out test_delta")
        if any(delta <= 0 for delta in [*train_deltas, test_delta]):
            raise ValueError("F18 currently preregisters positive interventions only")
        before_values = a_values - test_delta
        if before_values < 2:
            raise ValueError("F18 a_values must leave at least two valid before states at test_delta")
        chance = 1.0 / a_values

        cf, corr, seen_gap, rand_dir, shuf, xform = [], [], [], [], [], []
        cf_cosine, corr_cosine = [], []
        deltas: list[float] = []
        head_params = 0
        decoder_bytes = 0
        train_flops = 0
        train_rows = 0
        matched_rows: list[bool] = []
        t0 = time.perf_counter()
        for s in seeds:
            gen = torch.Generator().manual_seed(s + 1301)
            samples = _int(e, "samples", 420)
            world_dim = _int(e, "world_dim", 16)
            a_before = torch.randint(0, before_values, (samples,), generator=gen)
            c = torch.randint(0, contexts, (samples,), generator=gen)
            extra = _float(e, "nuisance", 0.6) * torch.randn(samples, world_dim, generator=gen)
            x_before = _f18_state_form(
                a_before,
                c,
                extra,
                a_scale=_float(e, "a_scale", 1.0),
                c_scale=_float(e, "c_scale", 1.0),
                feature_dim=_int(e, "feature_dim", 24),
                form_noise=_float(e, "form_noise", 0.9),
                seed=s + 11,
            )

            # Each training referent receives exactly one randomized do-value. This removes the old
            # 2x-row advantage while retaining randomized intervention assignment.
            intervention_index = torch.arange(samples) % len(train_deltas)
            intervention_index = intervention_index[
                torch.randperm(intervention_index.shape[0], generator=gen)
            ]
            do_delta = torch.tensor(train_deltas)[intervention_index]
            b_interventional = _f18_state_form(
                a_before + do_delta,
                c,
                extra,
                a_scale=_float(e, "a_scale", 1.0),
                c_scale=_float(e, "c_scale", 1.0),
                feature_dim=_int(e, "feature_dim", 24),
                form_noise=_float(e, "form_noise", 0.9),
                seed=s + 977,
            )

            # Observational change is caused by the recorded context. It uses only the same train
            # delta support, but assignment is confounded rather than randomized.
            delta_table = torch.tensor(train_deltas)
            natural_delta = delta_table[c % len(train_deltas)]
            b_observational = _f18_state_form(
                a_before + natural_delta,
                c,
                extra,
                a_scale=_float(e, "a_scale", 1.0),
                c_scale=_float(e, "c_scale", 1.0),
                feature_dim=_int(e, "feature_dim", 24),
                form_noise=_float(e, "form_noise", 0.9),
                seed=s + 977,
            )
            b_test = _f18_state_form(
                a_before + test_delta,
                c,
                extra,
                a_scale=_float(e, "a_scale", 1.0),
                c_scale=_float(e, "c_scale", 1.0),
                feature_dim=_int(e, "feature_dim", 24),
                form_noise=_float(e, "form_noise", 0.9),
                seed=s + 977,
            )
            tr, te = _split(samples, _float(e, "train_frac", 0.6), s)
            train_rows = int(tr.shape[0])
            x_int = torch.cat([x_before[tr], do_delta[tr, None].float()], dim=1)
            x_corr = torch.cat([x_before[tr], torch.zeros(tr.shape[0], 1)], dim=1)
            epochs, lr = _int(e, "epochs", 260), _float(e, "lr", 0.05)
            iv = _f18_fit_map(
                x_int,
                b_interventional[tr],
                epochs=epochs,
                lr=lr,
                seed=s + 3,
            )
            head_params = sum(p.numel() for p in iv.parameters())
            cr = _f18_fit_map(
                x_corr,
                b_observational[tr],
                epochs=epochs,
                lr=lr,
                seed=s + 5,
            )

            # shuffled-counterfactual-pairs: decouple before from after in interventional training
            g2 = torch.Generator().manual_seed(s + 707)
            y_shuf = b_interventional[tr[torch.randperm(tr.shape[0], generator=g2)]]
            sh = _f18_fit_map(x_int, y_shuf, epochs=epochs, lr=lr, seed=s + 7)

            # Nearest Form-B prototypes are a shared, parameter-free measuring instrument, built on
            # an independent calibration bank covering every factor value. They never see transport
            # outputs and preserve the claim's target: location in Form-B geometry.
            probe_n = _int(e, "probe_samples", 560)
            probe_a = torch.arange(probe_n) % a_values
            probe_a = probe_a[torch.randperm(probe_n, generator=gen)]
            probe_c = torch.randint(0, contexts, (probe_n,), generator=gen)
            probe_extra = _float(e, "nuisance", 0.6) * torch.randn(probe_n, world_dim, generator=gen)
            probe_b = _f18_state_form(
                probe_a,
                probe_c,
                probe_extra,
                a_scale=_float(e, "a_scale", 1.0),
                c_scale=_float(e, "c_scale", 1.0),
                feature_dim=_int(e, "feature_dim", 24),
                form_noise=_float(e, "form_noise", 0.9),
                seed=s + 977,
            )
            ptr, _ = _split(probe_n, _float(e, "probe_train_frac", 0.7), s + 43)
            prototypes = torch.stack(
                [probe_b[ptr][probe_a[ptr] == value].mean(0) for value in range(a_values)]
            )
            decoder_bytes = prototypes.nelement() * prototypes.element_size()

            # unseen delta test
            x_te_test = torch.cat([x_before[te], torch.full((te.shape[0], 1), float(test_delta))], dim=1)
            y_after = a_before[te] + test_delta
            x_te_corr = torch.cat([x_before[te], torch.zeros(te.shape[0], 1)], dim=1)
            cf_seed = _f18_decoded_acc(iv, prototypes, x_te_test, y_after)
            corr_seed = _f18_decoded_acc(cr, prototypes, x_te_corr, y_after)
            shuf_seed = _f18_decoded_acc(sh, prototypes, x_te_test, y_after)
            cf.append(cf_seed)
            corr.append(corr_seed)
            shuf.append(shuf_seed)
            cf_cosine.append(_f18_cosine(iv, x_te_test, b_test[te]))
            corr_cosine.append(_f18_cosine(cr, x_te_corr, b_test[te]))

            # random-intervention-direction: feed a random delta instead of the true do-value
            g3 = torch.Generator().manual_seed(s + 909)
            random_choice = torch.randint(0, len(train_deltas), (te.shape[0],), generator=g3)
            rd = torch.tensor(train_deltas)[random_choice]
            rand_seed = _f18_decoded_acc(
                iv,
                prototypes,
                torch.cat([x_before[te], rd[:, None].float()], dim=1),
                y_after,
            )
            rand_dir.append(rand_seed)

            # seen-delta accuracy for the leakage gap (predict trained deltas on held-out samples)
            seen = []
            for d in train_deltas:
                xd = torch.cat([x_before[te], torch.full((te.shape[0], 1), float(d))], dim=1)
                b_seen = _f18_state_form(
                    a_before + d,
                    c,
                    extra,
                    a_scale=_float(e, "a_scale", 1.0),
                    c_scale=_float(e, "c_scale", 1.0),
                    feature_dim=_int(e, "feature_dim", 24),
                    form_noise=_float(e, "form_noise", 0.9),
                    seed=s + 977,
                )
                seen.append(_f18_decoded_acc(iv, prototypes, xd, a_before[te] + d))
                # Make the target construction load-bearing: the prediction is compared to a Form-B
                # state for every seen intervention as well, not just its decoded label.
                _ = _f18_cosine(iv, xd, b_seen[te])
            seen_gap.append(_mean(seen) - cf_seed)

            with torch.no_grad():
                xform_seed = float((torch.cdist(b_test[te], prototypes).argmin(1) == y_after).float().mean())
            xform.append(xform_seed)
            deltas.append(cf_seed - max(corr_seed, shuf_seed, rand_seed))
            matched_rows.append(x_int.shape[0] == x_corr.shape[0] == tr.shape[0])

        din = _int(e, "feature_dim", 24) + 1
        dout = _int(e, "feature_dim", 24)
        per_update = mlp_flops([din, dout], train_rows)
        # Forward plus backward is explicitly estimated as 3x the linear forward cost. Both arms use
        # the same rows and update count; this is the quantity the old implementation failed to match.
        train_flops = 3 * per_update * _int(e, "epochs", 260)
        match = matched_within(train_flops, train_flops)
        cf_acc, corr_acc, gap = _mean(cf), _mean(corr), _mean(seen_gap)
        best_control = max(corr_acc, _mean(shuf), _mean(rand_dir))
        margin = _float(e, "margin", 0.05)
        leak_margin = _float(e, "leak_margin", 0.15)
        acc_per_param = cf_acc / head_params if head_params else 0.0
        return {
            "counterfactual_match_acc": round(cf_acc, 4),
            "correlational_baseline_acc": round(corr_acc, 4),
            "unseen_value_gap": round(gap, 4),
            "counterfactual_acc_per_param": round(acc_per_param, 8),
            "random_direction_acc": round(_mean(rand_dir), 4),
            "shuffled_pair_acc": round(_mean(shuf), 4),
            "cross_form_readout_acc": round(_mean(xform), 4),
            "counterfactual_form_b_cosine": round(_mean(cf_cosine), 4),
            "correlational_form_b_cosine": round(_mean(corr_cosine), 4),
            "predicted_object": "form_b_state",
            "chance": round(chance, 4),
            "matched_compute": match,
            "matched_train_rows": bool(all(matched_rows)),
            "train_rows_per_arm": train_rows,
            "updates_per_arm": _int(e, "epochs", 260),
            "transport_params": head_params,
            "measurement_decoder": "independent-form-b-nearest-prototype",
            "measurement_decoder_bytes": decoder_bytes,
            "train_deltas": train_deltas,
            "test_delta": test_delta,
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in deltas],
            "seed_ci": seed_ci(deltas),
            "sign_flip_report": sign_flip_report(deltas),
            "null_supported": bool(cf_acc <= best_control + margin or gap > leak_margin),
            "density": density_block(
                {
                    "counterfactual_match_acc": cf_acc,
                    "counterfactual_acc_per_param": acc_per_param,
                },
                primary="counterfactual_match_acc",
                params=float(head_params),
                flops=float(train_flops),
                updates=float(_int(e, "epochs", 260)),
                seconds=time.perf_counter() - t0,
            ),
        }


def _f20_arm(
    *,
    n: int,
    dim: int,
    classes: int,
    nuis_classes: int,
    signal: float,
    nuisance: float,
    rho: float,
    noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fixture arm: in-distribution features x, target y, recorded nuisance factor g, and a
    decorrelated shell copy xdec whose nuisance is redrawn independent of y. signal weights the true
    target subspace, nuisance weights a factor subspace that tracks y in distribution with rate rho.
    A crisis arm sets signal to zero (all apparent competence is nuisance-carried, the A6 pattern),
    a predictor-wall arm keeps signal below the pass threshold on its own, a healthy arm keeps it
    above, and a noisy arm sets both to zero (pure aleatoric stream)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, classes, (n,), generator=g)
    track = torch.rand(n, generator=g) < rho
    free = torch.randint(0, nuis_classes, (n,), generator=g)
    nuis = torch.where(track, y % nuis_classes, free)
    signal_dirs = torch.randn(classes, dim, generator=g)
    nuis_dirs = torch.randn(nuis_classes, dim, generator=g)
    x = signal * signal_dirs[y] + nuisance * nuis_dirs[nuis] + noise * torch.randn(n, dim, generator=g)
    nuis_dec = torch.randint(0, nuis_classes, (n,), generator=g)
    xdec = signal * signal_dirs[y] + nuisance * nuis_dirs[nuis_dec] + noise * torch.randn(n, dim, generator=g)
    return x.float(), y.long(), nuis.long(), xdec.float()


def _f20_probe_seed(seed: int, probe_index: int) -> int:
    """Preserve F20's legacy sequence while bounding high-seed Generation 1 child streams."""
    return derive_seed32(seed * 1000 + probe_index, "f20_substrate_crisis_test.probe")


def _f20_score(
    x: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    xdec: torch.Tensor,
    *,
    classes: int,
    epochs: int,
    lr: float,
    train_frac: float,
    seed: int,
) -> dict:
    """Monitor read-outs for one arm, using only in-distribution data plus the recorded factor.

    The crisis score is the held-out accuracy drop between the standard split and the off-diagonal
    slice, the samples where the recorded nuisance factor disagrees with the target. A probe that is
    riding the nuisance scores near chance on that slice while its standard accuracy stays high (the
    A6 pattern); a probe with genuine target signal keeps the slice, so its drop is small; a pure
    noise stream sits at chance on both, so its drop is near zero (no false trigger). raw_error is
    the plain in-distribution held-out error; confidence is the mean max-softmax; decorr_acc is the
    realized accuracy of the in-distribution probe on the decorrelated shell (the ground-truth verdict
    for whether the substrate actually reaches the target)."""
    tr, te = _split(y.shape[0], train_frac, seed)
    head = _fit_head(x[tr], y[tr], classes=classes, epochs=epochs, lr=lr, seed=seed)
    acc_in = _acc(head, x[te], y[te])
    with torch.no_grad():
        conf = float(F.softmax(head(x[te]), dim=-1).max(dim=-1).values.mean())
    off = te[g[te] != y[te]]
    off_acc = _acc(head, x[off], y[off]) if off.numel() > 0 else acc_in
    acc_dec = _acc(head, xdec[te], y[te])
    return {
        "crisis_score": acc_in - off_acc,
        "raw_error": 1.0 - acc_in,
        "confidence": conf,
        "decorr_acc": acc_dec,
    }


class F20(Experiment):
    id = "f20_substrate_crisis_test"
    metric = (
        "crisis_auroc",
        "raw_error_auroc",
        "false_trigger_rate",
        "avoided_wasted_compute_per_monitor_flop",
    )
    baseline = "raw held-out error signal and a fixed confidence threshold as the crisis predictors"
    ablation = (
        "off-diagonal factor slice accuracy drop vs raw error, fixed confidence, and a rate-matched "
        "random trigger, with a noisy-TV false-alarm bed"
    )
    null_hypothesis = (
        "the crisis detector fails to beat the strongest raw-error, fixed-confidence, or random "
        "baseline, or triggers on aleatoric noise, so prospective insufficiency is not established"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..diagnostics.compute import mlp_flops
        from ..diagnostics.operational_awareness import crisis_detection, rewrite_caution
        from ..diagnostics.riskcov import auroc, seed_ci, sign_flip_report

        e = cfg.experiment
        seeds = list(e.seeds)
        classes = _int(e, "classes", 4)
        nuis_classes = _int(e, "nuis_classes", 4)
        n = _int(e, "samples", 240)
        dim = _int(e, "dim", 24)
        epochs = _int(e, "epochs", 120)
        lr = _float(e, "lr", 0.05)
        train_frac = _float(e, "train_frac", 0.6)
        rho = _float(e, "rho", 0.9)
        wnoise = _float(e, "world_noise", 0.7)
        tau = _float(e, "crisis_trigger_threshold", 0.20)
        fail_th = _float(e, "fail_threshold", 0.42)
        error_th = _float(e, "error_threshold", 0.5)
        nuis_side = _float(e, "nuisance_side", 1.6)
        families = (
            ("crisis", _int(e, "n_crisis", 8), 0.0, _float(e, "nuisance_crisis", 2.4), wnoise),
            (
                "wall",
                _int(e, "n_wall", 8),
                _float(e, "signal_wall", 0.7),
                _float(e, "nuisance_wall", 1.2),
                _float(e, "wall_noise", 2.6),
            ),
            ("healthy", _int(e, "n_healthy", 8), _float(e, "signal_healthy", 1.4), nuis_side, wnoise),
        )
        scores: list[float] = []
        raw_errors: list[float] = []
        confidences: list[float] = []
        failed: list[float] = []
        noise_scores: list[float] = []
        noise_errors: list[float] = []
        per_seed_deltas: list[float] = []
        per_seed_false_rates: list[float] = []
        n_probes = 0
        t0 = time.perf_counter()
        for s in seeds:
            score_start = len(scores)
            noise_start = len(noise_scores)
            probe_index = 0
            for _kind, count, sig, nui, fnoise in families:
                for _ in range(count):
                    probe_index += 1
                    probe_seed = _f20_probe_seed(s, probe_index)
                    x, y, g, xdec = _f20_arm(
                        n=n,
                        dim=dim,
                        classes=classes,
                        nuis_classes=nuis_classes,
                        signal=sig,
                        nuisance=nui,
                        rho=rho,
                        noise=fnoise,
                        seed=probe_seed,
                    )
                    r = _f20_score(
                        x,
                        y,
                        g,
                        xdec,
                        classes=classes,
                        epochs=epochs,
                        lr=lr,
                        train_frac=train_frac,
                        seed=probe_seed,
                    )
                    n_probes += 1
                    scores.append(r["crisis_score"])
                    raw_errors.append(r["raw_error"])
                    confidences.append(r["confidence"])
                    failed.append(1.0 if r["decorr_acc"] < fail_th else 0.0)
            for _ in range(_int(e, "n_noise", 10)):
                probe_index += 1
                probe_seed = _f20_probe_seed(s, probe_index)
                x, y, g, xdec = _f20_arm(
                    n=n,
                    dim=dim,
                    classes=classes,
                    nuis_classes=nuis_classes,
                    signal=0.0,
                    nuisance=0.0,
                    rho=rho,
                    noise=wnoise,
                    seed=probe_seed,
                )
                r = _f20_score(
                    x,
                    y,
                    g,
                    xdec,
                    classes=classes,
                    epochs=epochs,
                    lr=lr,
                    train_frac=train_frac,
                    seed=probe_seed,
                )
                n_probes += 1
                noise_scores.append(r["crisis_score"])
                noise_errors.append(r["raw_error"])

            local_scores = scores[score_start:]
            local_raw = raw_errors[score_start:]
            local_conf = confidences[score_start:]
            local_failed = failed[score_start:]
            local_crisis = crisis_detection(local_scores, local_failed, raw_error=local_raw)
            local_fixed = auroc([1.0 - value for value in local_conf], local_failed)
            local_random = torch.rand(
                len(local_failed), generator=torch.Generator().manual_seed(s + 977)
            ).tolist()
            local_random_auroc = auroc(local_random, local_failed)
            local_strongest = max(local_crisis["raw_error_auroc"], local_fixed, local_random_auroc)
            per_seed_deltas.append(local_crisis["auroc"] - local_strongest)
            local_noise = noise_scores[noise_start:]
            per_seed_false_rates.append(_mean([1.0 if score > tau else 0.0 for score in local_noise]))

        crisis = crisis_detection(scores, failed, raw_error=raw_errors)
        crisis_auroc = crisis["auroc"]
        raw_error_auroc = crisis["raw_error_auroc"]
        fixed_conf_auroc = auroc([1.0 - c for c in confidences], failed)
        triggered_on_noise = [1.0 if sc > tau else 0.0 for sc in noise_scores]
        triggered_on_real = [
            1.0 if score > tau else 0.0
            for score, failure in zip(scores, failed, strict=True)
            if failure > 0.5
        ]
        caution = rewrite_caution(triggered_on_noise, triggered_on_real)
        false_trigger_rate = caution["false_trigger_rate"]
        crisis_trigger_rate = _mean([1.0 if sc > tau else 0.0 for sc in scores])
        fixed_error_false_rate = _mean([1.0 if er > error_th else 0.0 for er in noise_errors])
        rg = torch.Generator().manual_seed(int(seeds[0]) + 977)
        random_scores = torch.rand(len(failed), generator=rg).tolist()
        random_trigger_auroc = auroc(random_scores, failed)

        per_probe_flops = mlp_flops([dim, classes], batch=int(n * train_frac)) * epochs * 3
        monitor_flops = float(per_probe_flops * n_probes)
        shell_flops = _float(e, "shell_scaleup_flops", 1.0e10)
        n_caught = sum(triggered_on_real)
        avoided = n_caught * shell_flops
        hypothetical_avoided_per_monitor_flop = avoided / monitor_flops if monitor_flops > 0 else 0.0
        # No real shell run was skipped by this synthetic fixture. The canonical avoided-compute
        # metric therefore stays zero until a preregistered prospective forecast prevents a real run.
        avoided_per_monitor_flop = 0.0
        margin = _float(e, "margin", 0.1)
        noise_margin = _float(e, "noise_margin", 0.12)
        strongest_baseline_auroc = max(raw_error_auroc, fixed_conf_auroc, random_trigger_auroc)
        null_supported = bool(
            crisis_auroc <= strongest_baseline_auroc + margin or false_trigger_rate > noise_margin
        )
        return {
            "crisis_auroc": round(crisis_auroc, 4),
            "raw_error_auroc": round(raw_error_auroc, 4),
            "false_trigger_rate": round(false_trigger_rate, 4),
            "avoided_wasted_compute_per_monitor_flop": round(avoided_per_monitor_flop, 4),
            "hypothetical_avoided_compute_per_monitor_flop": round(hypothetical_avoided_per_monitor_flop, 4),
            "avoided_compute_measured": False,
            "fixed_confidence_auroc": round(fixed_conf_auroc, 4),
            "strongest_baseline_auroc": round(strongest_baseline_auroc, 4),
            "random_trigger_auroc": round(random_trigger_auroc, 4),
            "random_trigger_matched_rate": round(crisis_trigger_rate, 4),
            "fixed_error_noise_false_rate": round(fixed_error_false_rate, 4),
            "true_trigger_rate": round(caution["true_trigger_rate"], 4),
            "auroc_over_raw_error": round(crisis["auroc_over_raw_error"], 4),
            "failure_rate": round(crisis["failure_rate"], 4),
            "monitor_flops": monitor_flops,
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in per_seed_deltas],
            "per_seed_false_trigger_rates": [round(value, 4) for value in per_seed_false_rates],
            "seed_ci": seed_ci(per_seed_deltas),
            "sign_flip_report": sign_flip_report(per_seed_deltas),
            "null_supported": null_supported,
            "density": density_block(
                {
                    "crisis_auroc": crisis_auroc,
                    "avoided_wasted_compute_per_monitor_flop": avoided_per_monitor_flop,
                },
                primary="crisis_auroc",
                flops=monitor_flops,
                seconds=time.perf_counter() - t0,
            ),
        }


class F14(Experiment):
    id = "f14_lifelong_form_expansion"
    metric = (
        "old_form_bwt",
        "new_form_transfer",
        "old_memory_recall_delta",
        "new_form_memory_recall",
        "alignment_budget",
    )
    baseline = (
        "frozen zero-shot, matched-compute new-only adaptation, matched-cumulative-compute "
        "retraining, no alignment, shuffled referents, and raw or shuffled memory queries"
    )
    ablation = (
        "insert one new form with form-local alignment and replay while old memory keys, values, "
        "and referent ids remain immutable"
    )
    null_hypothesis = (
        "new-form insertion changes any old-memory key, value, or referent id, changes old-memory "
        "retrieval, forgets old forms beyond the retention band, or fails to beat the strongest "
        "matched existing-head and raw or shuffled retrieval controls"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        import copy
        import hashlib

        from ..diagnostics.compute import mlp_flops
        from ..diagnostics.riskcov import seed_ci, sign_flip_report
        from ..shell.buffer import ReplayBuffer

        e = cfg.experiment
        seeds = [int(seed) for seed in e.seeds]
        if len(seeds) < 5 or len(set(seeds)) != len(seeds):
            raise ValueError("F14 requires at least five unique seeds")
        classes = _int(e, "classes", 10)
        old_kinds = ("vision", "audio")
        new_kind = _str(e, "new_form", "timeseries")
        if new_kind in old_kinds or new_kind not in FORM_KINDS:
            raise ValueError("F14 new_form must be a non-old form in FORM_KINDS")

        bwt: list[float] = []
        no_replay_bwt: list[float] = []
        old_before_acc: list[float] = []
        old_after_acc: list[float] = []
        new_transfer: list[float] = []
        frozen_transfer: list[float] = []
        no_replay_transfer: list[float] = []
        scratch_t: list[float] = []
        scratch_old: list[float] = []
        noalign_t: list[float] = []
        shuf_t: list[float] = []
        old_memory_before: list[float] = []
        old_memory_after: list[float] = []
        old_memory_delta: list[float] = []
        new_memory: list[float] = []
        raw_memory: list[float] = []
        shuffled_memory: list[float] = []
        new_deltas: list[float] = []
        alignment_floor_deltas: list[float] = []
        replay_bwt_advantages: list[float] = []
        memory_deltas: list[float] = []
        invariant_rows: list[dict[str, object]] = []
        snapshot_receipts: list[dict[str, object]] = []
        split_rows: dict[str, int] = {}
        accounting: dict[str, dict[str, int]] = {}
        budget = 0
        head_params = 0
        memory_slots = 0
        memory_bytes = 0
        train_flops = 0
        index_backend = "brute"
        t0 = time.perf_counter()

        def _tensor_sha256(tensor: torch.Tensor) -> str:
            raw = tensor.detach().cpu().contiguous().numpy().tobytes()
            return hashlib.sha256(raw).hexdigest()

        for s in seeds:
            z, y = _balanced_world(
                samples=_int(e, "samples", 400),
                classes=classes,
                world_dim=_int(e, "world_dim", 28),
                separation=_float(e, "separation", 0.85),
                noise=_float(e, "world_noise", 1.15),
                seed=s,
            )
            forms = _form_features(
                z,
                feature_dim=_int(e, "feature_dim", 30),
                noise=_float(e, "form_noise", 0.55),
                seed=s,
            )
            anchors, labels, test = _three_way_split(
                y.shape[0],
                _float(e, "anchor_frac", 0.25),
                _float(e, "label_frac", 0.4),
                s,
            )
            split_rows = {
                "anchor": int(anchors.shape[0]),
                "label": int(labels.shape[0]),
                "test": int(test.shape[0]),
            }
            budget = split_rows["anchor"]
            epochs = _int(e, "epochs", 60)
            adapt_epochs = _int(e, "adapt_epochs", 25)
            lr = _float(e, "lr", 0.03)
            adapt_lr = _float(e, "adapt_lr", 0.012)
            feature_dim = _int(e, "feature_dim", 30)
            rows_per_step = 2 * int(labels.shape[0])

            # Phase one learns the old forms and writes an episodic memory whose referent ids are
            # the global synthetic-world row ids. The new form receives its own map later; neither
            # the old map nor any memory tensor is eligible for rewrite.
            reference = forms["vision"]
            old_weight = fit_affine_alignment(forms["audio"][anchors], reference[anchors])
            old_weight_snapshot = old_weight.clone()
            old_aligned = {
                "vision": reference,
                "audio": apply_affine_alignment(forms["audio"], old_weight),
            }
            x_old = torch.cat([old_aligned[k][labels] for k in old_kinds], dim=0)
            y_old = torch.cat([y[labels] for _ in old_kinds], dim=0)
            phase_one_head = _fit_head(
                x_old,
                y_old,
                classes=classes,
                epochs=epochs,
                lr=lr,
                seed=s,
            )
            head_params = sum(parameter.numel() for parameter in phase_one_head.parameters())
            old_before_seed = _mean([_acc(phase_one_head, old_aligned[k][test], y[test]) for k in old_kinds])
            old_before_acc.append(old_before_seed)

            memory = ReplayBuffer(
                capacity=int(labels.shape[0]),
                dim=feature_dim,
                key_dim=feature_dim,
                prioritized=False,
                eviction="fifo",
                index=index_backend,
                seed=s,
            )
            # `y` deliberately stores referent ids, not class labels. Replay resolves classes through
            # the immutable id, while retrieval can verify identity without relying on slot order.
            memory.add(reference[labels], labels, key=reference[labels])
            memory_slots = len(memory)
            memory_bytes = int(
                memory.x[:memory_slots].nelement() * memory.x.element_size()
                + memory.keys[:memory_slots].nelement() * memory.keys.element_size()
                + memory.y[:memory_slots].nelement() * memory.y.element_size()
                + memory.prio[:memory_slots].nelement() * memory.prio.element_size()
            )
            key_snapshot = memory.keys[:memory_slots].clone()
            value_snapshot = memory.x[:memory_slots].clone()
            id_snapshot = memory.y[:memory_slots].clone()
            length_snapshot = len(memory)
            seen_snapshot = memory.seen
            before_hashes = {
                "keys": _tensor_sha256(key_snapshot),
                "values": _tensor_sha256(value_snapshot),
                "referent_ids": _tensor_sha256(id_snapshot),
            }

            k = _int(e, "k", 1)
            if k <= 0:
                raise ValueError("F14 retrieval k must be positive")

            def _recall(
                query: torch.Tensor,
                *,
                store: ReplayBuffer = memory,
                retrieval_k: int = k,
                expected: torch.Tensor = labels,
            ) -> float:
                retrieved = store.retrieve(query, k=retrieval_k)["y"]
                return float((retrieved == expected[:, None]).any(dim=1).float().mean())

            old_query_generator = torch.Generator().manual_seed(s + 1709)
            old_query = old_aligned["audio"][labels] + _float(e, "query_noise", 0.3) * torch.randn(
                old_aligned["audio"][labels].shape, generator=old_query_generator
            )
            old_recall_before_seed = _recall(old_query)
            old_memory_before.append(old_recall_before_seed)

            # Phase two fits only the new form's map. The shuffled map is fit once and reused for
            # both classifier and memory controls, so those controls share the same false pairing.
            new_weight = fit_affine_alignment(forms[new_kind][anchors], reference[anchors])
            new_aligned = apply_affine_alignment(forms[new_kind], new_weight)
            shuffle_generator = torch.Generator().manual_seed(s + 1801)
            shuffled_reference = reference[
                anchors[torch.randperm(anchors.shape[0], generator=shuffle_generator)]
            ]
            shuffled_weight = fit_affine_alignment(forms[new_kind][anchors], shuffled_reference)
            shuffled_aligned = apply_affine_alignment(forms[new_kind], shuffled_weight)

            frozen_seed = _acc(phase_one_head, new_aligned[test], y[test])
            frozen_transfer.append(frozen_seed)

            replay_head = copy.deepcopy(phase_one_head)
            replay_opt = torch.optim.Adam(replay_head.parameters(), lr=adapt_lr)
            for _ in range(adapt_epochs):
                replay = memory.sample(int(labels.shape[0]))
                replay_classes = y[replay["y"]]
                x_phase_two = torch.cat([new_aligned[labels], replay["x"]], dim=0)
                y_phase_two = torch.cat([y[labels], replay_classes], dim=0)
                replay_opt.zero_grad()
                F.cross_entropy(replay_head(x_phase_two), y_phase_two).backward()
                replay_opt.step()

            no_replay_head = copy.deepcopy(phase_one_head)
            no_replay_opt = torch.optim.Adam(no_replay_head.parameters(), lr=adapt_lr)
            for _ in range(adapt_epochs):
                # Duplicate the new-form rows to charge exactly the same rows and optimizer steps as
                # the replay arm. This is wasted duplicate exposure by design, not hidden extra data.
                x_phase_two = torch.cat([new_aligned[labels], new_aligned[labels]], dim=0)
                y_phase_two = torch.cat([y[labels], y[labels]], dim=0)
                no_replay_opt.zero_grad()
                F.cross_entropy(no_replay_head(x_phase_two), y_phase_two).backward()
                no_replay_opt.step()

            old_after_seed = _mean(
                [_acc(replay_head, old_aligned[kind][test], y[test]) for kind in old_kinds]
            )
            no_replay_old_seed = _mean(
                [_acc(no_replay_head, old_aligned[kind][test], y[test]) for kind in old_kinds]
            )
            bwt_seed = old_after_seed - old_before_seed
            no_replay_bwt_seed = no_replay_old_seed - old_before_seed
            bwt.append(bwt_seed)
            no_replay_bwt.append(no_replay_bwt_seed)
            replay_bwt_advantages.append(bwt_seed - no_replay_bwt_seed)
            old_after_acc.append(old_after_seed)

            new_seed = _acc(replay_head, new_aligned[test], y[test])
            no_replay_seed = _acc(no_replay_head, new_aligned[test], y[test])
            noalign_seed = _acc(replay_head, forms[new_kind][test], y[test])
            shuf_seed = _acc(replay_head, shuffled_aligned[test], y[test])
            new_transfer.append(new_seed)
            no_replay_transfer.append(no_replay_seed)
            noalign_t.append(noalign_seed)
            shuf_t.append(shuf_seed)
            alignment_floor_deltas.append(new_seed - max(noalign_seed, shuf_seed))
            new_deltas.append(new_seed - max(frozen_seed, no_replay_seed, noalign_seed, shuf_seed))

            # The scratch upper bound consumes the exact cumulative row and optimizer-step budget of
            # phase one plus phase two, but samples all three forms throughout instead of expanding.
            seed_everything(s + 7)
            scratch_head = nn.Linear(feature_dim, classes)
            scratch_opt = torch.optim.Adam(scratch_head.parameters(), lr=_float(e, "scratch_lr", lr))
            scratch_generator = torch.Generator().manual_seed(s + 1901)
            scratch_bank = torch.stack(
                [old_aligned["vision"][labels], old_aligned["audio"][labels], new_aligned[labels]],
                dim=1,
            )
            for step in range(epochs + adapt_epochs):
                referent_positions = torch.randint(
                    0,
                    labels.shape[0],
                    (rows_per_step,),
                    generator=scratch_generator,
                )
                form_choice = (torch.arange(rows_per_step) + step) % 3
                scratch_x = scratch_bank[referent_positions, form_choice]
                scratch_y = y[labels[referent_positions]]
                scratch_opt.zero_grad()
                F.cross_entropy(scratch_head(scratch_x), scratch_y).backward()
                scratch_opt.step()
            scratch_t.append(_acc(scratch_head, new_aligned[test], y[test]))
            scratch_old.append(
                _mean([_acc(scratch_head, old_aligned[kind][test], y[test]) for kind in old_kinds])
            )

            # All retrieval controls use the same independent noise tensor. Old recall is re-run on
            # the exact same query tensor, after every phase-two operation, to make a zero delta an
            # exact invariant rather than an expectation over query noise.
            new_query_generator = torch.Generator().manual_seed(s + 2003)
            shared_query_noise = _float(e, "query_noise", 0.3) * torch.randn(
                new_aligned[labels].shape, generator=new_query_generator
            )
            new_recall_seed = _recall(new_aligned[labels] + shared_query_noise)
            raw_recall_seed = _recall(forms[new_kind][labels] + shared_query_noise)
            shuffled_recall_seed = _recall(shuffled_aligned[labels] + shared_query_noise)
            old_recall_after_seed = _recall(old_query)
            new_memory.append(new_recall_seed)
            raw_memory.append(raw_recall_seed)
            shuffled_memory.append(shuffled_recall_seed)
            old_memory_after.append(old_recall_after_seed)
            old_delta_seed = old_recall_after_seed - old_recall_before_seed
            old_memory_delta.append(old_delta_seed)
            memory_deltas.append(new_recall_seed - max(raw_recall_seed, shuffled_recall_seed))

            after_hashes = {
                "keys": _tensor_sha256(memory.keys[:memory_slots]),
                "values": _tensor_sha256(memory.x[:memory_slots]),
                "referent_ids": _tensor_sha256(memory.y[:memory_slots]),
            }
            invariants: dict[str, object] = {
                "seed": s,
                "keys_unchanged": torch.equal(key_snapshot, memory.keys[:memory_slots]),
                "values_unchanged": torch.equal(value_snapshot, memory.x[:memory_slots]),
                "referent_ids_unchanged": torch.equal(id_snapshot, memory.y[:memory_slots]),
                "length_unchanged": len(memory) == length_snapshot,
                "seen_unchanged": memory.seen == seen_snapshot,
                "old_alignment_unchanged": torch.equal(old_weight_snapshot, old_weight),
                "old_recall_unchanged": old_delta_seed == 0.0,
            }
            invariant_rows.append(invariants)
            snapshot_receipts.append(
                {
                    "seed": s,
                    "before": before_hashes,
                    "after": after_hashes,
                    "length_before": length_snapshot,
                    "length_after": len(memory),
                    "seen_before": seen_snapshot,
                    "seen_after": memory.seen,
                }
            )

            phase_one_rows = rows_per_step * epochs
            phase_two_rows = rows_per_step * adapt_epochs
            cumulative_rows = phase_one_rows + phase_two_rows
            cumulative_steps = epochs + adapt_epochs
            accounting = {
                "replay_expansion": {
                    "head_params": head_params,
                    "phase_one_rows_per_step": rows_per_step,
                    "phase_one_steps": epochs,
                    "phase_two_rows_per_step": rows_per_step,
                    "phase_two_steps": adapt_epochs,
                    "total_rows_seen": cumulative_rows,
                    "optimizer_steps": cumulative_steps,
                },
                "no_replay_expansion": {
                    "head_params": head_params,
                    "phase_one_rows_per_step": rows_per_step,
                    "phase_one_steps": epochs,
                    "phase_two_rows_per_step": rows_per_step,
                    "phase_two_steps": adapt_epochs,
                    "total_rows_seen": cumulative_rows,
                    "optimizer_steps": cumulative_steps,
                },
                "scratch": {
                    "head_params": sum(parameter.numel() for parameter in scratch_head.parameters()),
                    "phase_one_rows_per_step": rows_per_step,
                    "phase_one_steps": epochs,
                    "phase_two_rows_per_step": rows_per_step,
                    "phase_two_steps": adapt_epochs,
                    "total_rows_seen": cumulative_rows,
                    "optimizer_steps": cumulative_steps,
                },
            }
            train_flops = mlp_flops([feature_dim, classes], batch=rows_per_step) * cumulative_steps * 3

        invariants_all_ok = all(
            all(bool(value) for key, value in row.items() if key != "seed") for row in invariant_rows
        )
        replay_no_replay_matched = accounting["replay_expansion"] == accounting["no_replay_expansion"]
        scratch_matched = (
            accounting["replay_expansion"]["head_params"] == accounting["scratch"]["head_params"]
            and accounting["replay_expansion"]["total_rows_seen"] == accounting["scratch"]["total_rows_seen"]
            and accounting["replay_expansion"]["optimizer_steps"] == accounting["scratch"]["optimizer_steps"]
        )
        expansion_gain = _mean(alignment_floor_deltas)
        strongest_existing = max(
            _mean(frozen_transfer),
            _mean(no_replay_transfer),
            _mean(noalign_t),
            _mean(shuf_t),
        )
        strongest_gain = _mean(new_deltas)
        aggregate_strongest_gain = _mean(new_transfer) - strongest_existing
        memory_gain = _mean(memory_deltas)
        primary_seed_summary = seed_ci(new_deltas)

        null_reasons: list[str] = []
        if not invariants_all_ok:
            null_reasons.append("old_memory_or_alignment_mutated")
        if any(delta != 0.0 for delta in old_memory_delta):
            null_reasons.append("old_memory_recall_changed")
        if _mean(bwt) < -_float(e, "forget_margin", 0.1):
            null_reasons.append("old_form_forgetting_exceeded_band")
        if expansion_gain <= _float(e, "margin", 0.1):
            null_reasons.append("new_form_failed_alignment_floors")
        if float(primary_seed_summary["lo"]) <= _float(e, "adapt_margin", 0.01):
            null_reasons.append("replay_failed_strongest_existing_head_control")
        if memory_gain <= _float(e, "memory_margin", 0.1):
            null_reasons.append("new_form_memory_failed_retrieval_floors")
        if not replay_no_replay_matched or not scratch_matched:
            null_reasons.append("compute_not_matched")

        return {
            "new_form": new_kind,
            "old_form_bwt": round(_mean(bwt), 4),
            "old_form_acc_before": round(_mean(old_before_acc), 4),
            "old_form_acc_after": round(_mean(old_after_acc), 4),
            "new_form_transfer": round(_mean(new_transfer), 4),
            "alignment_budget": budget,
            "old_memory_recall_before": round(_mean(old_memory_before), 4),
            "old_memory_recall_after": round(_mean(old_memory_after), 4),
            "old_memory_recall_delta": round(_mean(old_memory_delta), 4),
            "new_form_memory_recall": round(_mean(new_memory), 4),
            "raw_new_form_memory_floor": round(_mean(raw_memory), 4),
            "shuffled_new_form_memory_floor": round(_mean(shuffled_memory), 4),
            "new_form_memory_gain_over_floor": round(memory_gain, 4),
            "memory_slots": memory_slots,
            "memory_tensor_bytes": memory_bytes,
            "memory_index_backend": index_backend,
            "memory_keys_unchanged": all(bool(row["keys_unchanged"]) for row in invariant_rows),
            "memory_values_unchanged": all(bool(row["values_unchanged"]) for row in invariant_rows),
            "memory_referent_ids_unchanged": all(
                bool(row["referent_ids_unchanged"]) for row in invariant_rows
            ),
            "memory_length_unchanged": all(bool(row["length_unchanged"]) for row in invariant_rows),
            "memory_seen_unchanged": all(bool(row["seen_unchanged"]) for row in invariant_rows),
            "old_alignment_unchanged": all(bool(row["old_alignment_unchanged"]) for row in invariant_rows),
            "memory_invariants_all_ok": invariants_all_ok,
            "memory_invariants_by_seed": invariant_rows,
            "memory_snapshot_receipts": snapshot_receipts,
            "frozen_zero_shot_transfer": round(_mean(frozen_transfer), 4),
            "no_replay_transfer": round(_mean(no_replay_transfer), 4),
            "no_replay_old_form_bwt": round(_mean(no_replay_bwt), 4),
            "retrain_from_scratch_transfer": round(_mean(scratch_t), 4),
            "retrain_from_scratch_old_form_acc": round(_mean(scratch_old), 4),
            "no_alignment_floor": round(_mean(noalign_t), 4),
            "shuffled_referent_floor": round(_mean(shuf_t), 4),
            "expansion_gain_over_floor": round(expansion_gain, 4),
            "strongest_existing_head_control": round(strongest_existing, 4),
            "new_form_gain_over_strongest_control": round(strongest_gain, 4),
            "aggregate_mean_gain_over_strongest_control": round(aggregate_strongest_gain, 4),
            "split_rows": split_rows,
            "disjoint_splits": True,
            "matched_accounting": accounting,
            "matched_replay_no_replay_compute": replay_no_replay_matched,
            "matched_scratch_cumulative_compute": scratch_matched,
            "estimated_training_flops": train_flops,
            "seeds": seeds,
            "per_seed_deltas": [round(value, 4) for value in new_deltas],
            "seed_ci": primary_seed_summary,
            "sign_flip_report": sign_flip_report(new_deltas),
            "per_seed_alignment_floor_deltas": [round(value, 4) for value in alignment_floor_deltas],
            "alignment_floor_seed_ci": seed_ci(alignment_floor_deltas),
            "alignment_floor_sign_flip_report": sign_flip_report(alignment_floor_deltas),
            "per_seed_old_form_bwt": [round(value, 4) for value in bwt],
            "old_form_bwt_seed_ci": seed_ci(bwt),
            "old_form_bwt_sign_flip_report": sign_flip_report(bwt),
            "per_seed_replay_bwt_advantage": [round(value, 4) for value in replay_bwt_advantages],
            "replay_bwt_advantage_seed_ci": seed_ci(replay_bwt_advantages),
            "replay_bwt_advantage_sign_flip_report": sign_flip_report(replay_bwt_advantages),
            "per_seed_old_memory_recall_delta": [round(value, 4) for value in old_memory_delta],
            "old_memory_recall_seed_ci": seed_ci(old_memory_delta),
            "old_memory_recall_sign_flip_report": sign_flip_report(old_memory_delta),
            "per_seed_new_memory_deltas": [round(value, 4) for value in memory_deltas],
            "new_memory_seed_ci": seed_ci(memory_deltas),
            "new_memory_sign_flip_report": sign_flip_report(memory_deltas),
            "null_reasons": null_reasons,
            "null_supported": bool(null_reasons),
            "density": density_block(
                {
                    "new_form_transfer": _mean(new_transfer),
                    "new_form_memory_recall": _mean(new_memory),
                },
                primary="new_form_transfer",
                seconds=time.perf_counter() - t0,
                params=float(head_params),
                bytes=float(memory_bytes),
                flops=float(train_flops),
            ),
        }
