
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, refiner_flops
from ..diagnostics.held_out_combo import factorized_latents, held_out_combination
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from ..shell.modulation import Chunking
from ..shell.refine import IterativeRefiner
from .base import Experiment, _mean


class C1(Experiment):
    id = "c1_held_out_combination"
    metric = ("heldout_minus_seen_gap", "per_factor_heldout_vs_shuffle_floor", "heldout_vs_frozen_random")
    baseline = "shuffle-label floor (chance) and a frozen-random projection arm on the same factor grid"
    ablation = "real factorized latents vs frozen-random projection; held-out diagonal vs seen cells"
    null_hypothesis = (
        "held-out-combination accuracy collapses to chance while seen-combination accuracy is high "
        "(pooled latent encodes whole-clip conjunctions, not separable factors), or both factors are "
        "jointly undecodable so the test is uninterpretable"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_a, n_b = int(e.n_a), int(e.n_b)
        dim, per_cell = int(e.dim), int(e.per_cell)
        interaction = float(e.interaction)

        real_ho, real_seen, real_gap, fr_ho, shuf_ho = [], [], [], [], []
        a_decode, b_decode = [], []
        for s in seeds:
            seed_everything(s)
            x, ya, yb = factorized_latents(
                n_a=n_a, n_b=n_b, per_cell=per_cell, dim=dim, interaction=interaction, seed=s
            )
            a_decode.append(linear_probe(x, ya, seed=s)["score"])
            b_decode.append(linear_probe(x, yb, seed=s)["score"])
            r = held_out_combination(x, ya, yb, seed=s)
            real_ho.append(r["heldout_acc"])
            real_seen.append(r["seen_acc"])
            real_gap.append(r["gap"])
            fr = held_out_combination(frozen_random_projection(x, s), ya, yb, seed=s)
            fr_ho.append(fr["heldout_acc"])
            g = torch.Generator().manual_seed(s + 7)
            yshuf = ya[torch.randperm(ya.shape[0], generator=g)]
            sh = held_out_combination(x, yshuf, yb, seed=s)
            shuf_ho.append(sh["heldout_acc"])

        chance = 1.0 / n_a
        ho, seen = _mean(real_ho), _mean(real_seen)
        frm, shm = _mean(fr_ho), _mean(shuf_ho)
        jointly_decodable = bool(_mean(a_decode) > chance + 0.1 and _mean(b_decode) > chance + 0.1)
        compositional = bool(ho > shm + 0.15 and ho > frm + 0.1)
        null_supported = bool((not jointly_decodable) or (ho <= chance + 0.15 and seen > chance + 0.2))
        return {
            "heldout_acc": round(ho, 4),
            "seen_acc": round(seen, 4),
            "heldout_minus_seen_gap": round(seen - ho, 4),
            "shuffle_floor_heldout": round(shm, 4),
            "frozen_random_heldout": round(frm, 4),
            "factor_a_decodability": round(_mean(a_decode), 4),
            "factor_b_decodability": round(_mean(b_decode), 4),
            "chance": round(chance, 4),
            "jointly_decodable": jointly_decodable,
            "compositional_above_floor": compositional,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


def _analogy_quads(n_content: int, dim: int, sep: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    contents = torch.randn(n_content, dim, generator=g) * sep
    r = torch.randn(dim, generator=g)  # the shared relational transform offset
    src = contents + 0.3 * torch.randn(n_content, dim, generator=g)
    tgt = contents + r + 0.3 * torch.randn(n_content, dim, generator=g)
    return src, tgt


class C2(Experiment):
    id = "c2_latent_analogy"
    metric = ("analogy_retrieval_top1", "offset_cosine_consistency", "delta_over_shuffled_offset")
    baseline = "shuffled-offset control (offset estimated from a mismatched content pair) and chance rank"
    ablation = "structure-mapped offset transfer vs shuffled-offset; real latents vs frozen-random"
    null_hypothesis = (
        "offset vectors do not transfer across object contents (the transform is entangled with "
        "identity), so analogy retrieval ties the shuffled-offset control, or the transform factor is "
        "not decodable at all"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_content, dim, sep = int(e.n_content), int(e.dim), float(e.separation)

        real_top1, shuf_top1, fr_top1, cos_cons = [], [], [], []
        for s in seeds:
            seed_everything(s)
            src, tgt = _analogy_quads(n_content, dim, sep, s)

            def _retrieve(src_, tgt_, shuffled: bool, sd=s):
                hits = 0
                cosines = []
                offsets = tgt_ - src_  # per-content offset estimate
                for i in range(src_.shape[0]):
                    others = [j for j in range(src_.shape[0]) if j != i]
                    if shuffled:
                        g = torch.Generator().manual_seed(sd + i + 100)
                        donor = others[int(torch.randint(0, len(others), (1,), generator=g))]
                        off = offsets[donor]
                    else:
                        off = offsets[torch.tensor(others)].mean(0)  # transfer from held-out contents
                    pred = src_[i] + off
                    d = (tgt_ - pred).norm(dim=-1)
                    if int(d.argmin()) == i:
                        hits += 1
                    ref = offsets[torch.tensor(others)].mean(0)
                    cosines.append(float(F.cosine_similarity(off.unsqueeze(0), ref.unsqueeze(0)).item()))
                return hits / src_.shape[0], _mean(cosines)

            rt, rc = _retrieve(src, tgt, shuffled=False)
            st, _ = _retrieve(src, tgt, shuffled=True)
            fsrc = frozen_random_projection(torch.cat([src, tgt]), s)
            ft, _ = _retrieve(fsrc[:n_content], fsrc[n_content:], shuffled=False)
            real_top1.append(rt)
            shuf_top1.append(st)
            fr_top1.append(ft)
            cos_cons.append(rc)

        rt, st, ft = _mean(real_top1), _mean(shuf_top1), _mean(fr_top1)
        chance = 1.0 / n_content
        delta = rt - st
        null_supported = bool(delta <= 0.1 or rt <= chance + 0.1)
        return {
            "analogy_retrieval_top1": round(rt, 4),
            "shuffled_offset_top1": round(st, 4),
            "frozen_random_top1": round(ft, 4),
            "offset_cosine_consistency": round(_mean(cos_cons), 4),
            "delta_over_shuffled_offset": round(delta, 4),
            "chance": round(chance, 4),
            "transfers_across_content": bool(delta > 0.1 and rt > chance + 0.1),
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


def _category_latents(n_cat: int, dim: int, sep: float, per: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_cat, dim, generator=g) * sep
    xs, ys = [], []
    for c in range(n_cat):
        xs.append(centers[c].unsqueeze(0) + 0.7 * torch.randn(per, dim, generator=g))
        ys += [c] * per
    return torch.cat(xs), torch.tensor(ys), centers


def _silhouette(x: torch.Tensor, y: torch.Tensor) -> float:
    d = torch.cdist(x, x)
    n = x.shape[0]
    vals = []
    cats = y.unique().tolist()
    for i in range(n):
        same = (y == y[i]) & (torch.arange(n) != i)
        a = float(d[i][same].mean()) if same.any() else 0.0
        bs = []
        for c in cats:
            if c == int(y[i]):
                continue
            m = y == c
            if m.any():
                bs.append(float(d[i][m].mean()))
        b = min(bs) if bs else 0.0
        denom = max(a, b)
        vals.append((b - a) / denom if denom > 0 else 0.0)
    return _mean(vals)


class C3(Experiment):
    id = "c3_prototype_typicality"
    metric = ("typicality_correlation", "prototype_vs_exemplar_gap", "silhouette_vs_frozen_random")
    baseline = "exemplar k-NN classifier and a frozen-random projection on the same categories"
    ablation = "nearest-centroid prototype vs exemplar k-NN; real latents vs frozen-random geometry"
    null_hypothesis = (
        "no typicality gradient (latents uniformly distributed within category, an exemplar bag), or a "
        "frozen-random projection shows the same cluster geometry so the structure is not "
        "V-JEPA-specific"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_cat, dim, sep, per = int(e.n_cat), int(e.dim), float(e.separation), int(e.per)

        typ_corr, proto_acc, exem_acc, sil_real, sil_fr = [], [], [], [], []
        for s in seeds:
            seed_everything(s)
            x, y, _ = _category_latents(n_cat, dim, sep, per, s)
            cut = int(x.shape[0] * 0.7)
            perm = torch.randperm(x.shape[0])
            x, y = x[perm], y[perm]
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            cents = torch.stack([xtr[ytr == c].mean(0) for c in range(n_cat)])
            pd = torch.cdist(xte, cents)
            pacc = float((pd.argmin(1) == yte).float().mean())
            ed = torch.cdist(xte, xtr)
            eacc = float((ytr[ed.argmin(1)] == yte).float().mean())
            margins = []
            dists = []
            head = torch.nn.Linear(dim, n_cat)
            opt = torch.optim.Adam(head.parameters(), lr=0.05)
            for _ in range(int(e.probe_epochs)):
                opt.zero_grad()
                F.cross_entropy(head(xtr), ytr).backward()
                opt.step()
            with torch.no_grad():
                logits = head(xte)
                for i in range(xte.shape[0]):
                    true = int(yte[i])
                    lo = logits[i].clone()
                    correct = lo[true].item()
                    lo[true] = -1e9
                    margins.append(correct - float(lo.max()))
                    dists.append(float((xte[i] - cents[true]).norm()))
            mt = torch.tensor(margins)
            dt = torch.tensor(dists)
            mt = mt - mt.mean()
            dt = dt - dt.mean()
            denom = (mt.norm() * dt.norm()).clamp(min=1e-8)
            typ_corr.append(float((mt * dt).sum() / denom))  # expect negative (far -> low margin)
            proto_acc.append(pacc)
            exem_acc.append(eacc)
            sil_real.append(_silhouette(xte, yte))
            sil_fr.append(_silhouette(frozen_random_projection(xte, s), yte))

        corr = _mean(typ_corr)
        pa, ea = _mean(proto_acc), _mean(exem_acc)
        sr, sf = _mean(sil_real), _mean(sil_fr)
        prototype_works = bool(abs(pa - ea) < 0.1 and pa > 1.0 / n_cat + 0.1)
        has_typicality = bool(abs(corr) > 0.1)
        null_supported = bool((not has_typicality) or (sr - sf) <= 0.05)
        return {
            "typicality_correlation": round(corr, 4),
            "prototype_accuracy": round(pa, 4),
            "exemplar_accuracy": round(ea, 4),
            "prototype_vs_exemplar_gap": round(pa - ea, 4),
            "silhouette_real": round(sr, 4),
            "silhouette_frozen_random": round(sf, 4),
            "silhouette_delta": round(sr - sf, 4),
            "prototype_ties_exemplar": prototype_works,
            "has_typicality_gradient": has_typicality,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


def _event_sequence(n_events: int, frames_per_event: int, dim: int, sep: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    cents = torch.randn(n_events, dim, generator=g) * sep
    frames, truth = [], []
    for ev in range(n_events):
        for f in range(frames_per_event):
            frames.append(cents[ev] + 0.2 * torch.randn(dim, generator=g))
            truth.append(1 if (f == 0 and ev > 0) else 0)
    return torch.stack(frames).unsqueeze(0), torch.tensor(truth)


def _boundary_f1(pred: torch.Tensor, truth: torch.Tensor) -> float:
    tp = float(((pred == 1) & (truth == 1)).sum())
    fp = float(((pred == 1) & (truth == 0)).sum())
    fn = float(((pred == 0) & (truth == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


class C4(Experiment):
    id = "c4_event_segmentation"
    metric = ("boundary_f1", "delta_over_uniform_rate", "delta_over_shuffled_latent")
    baseline = "uniform-rate segmenter (fixed period) and a shuffled-latent control on the same sequence"
    ablation = "surprise-boundary Chunking vs uniform-rate vs shuffled-latent; threshold PR sweep"
    null_hypothesis = (
        "surprise boundaries do not align with true transitions better than a uniform-rate segmenter "
        "(pooled clip latents too coarse temporally, taxonomy 5), or the substrate gives no usable "
        "surprise signal"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_events, fpe = int(e.n_events), int(e.frames_per_event)
        dim, sep = int(e.dim), float(e.separation)

        surf1, unif1, shuf1, best_thr = [], [], [], []
        for s in seeds:
            seed_everything(s)
            seq, truth = _event_sequence(n_events, fpe, dim, sep, s)
            gap_truth = truth[1:]  # gap t corresponds to transition into frame t+1
            d = (seq[0, 1:] - seq[0, :-1]).norm(dim=-1)  # [T-1] successive-latent distance
            thrs = torch.linspace(d.min(), d.max(), int(e.n_thresholds))
            best, bthr = 0.0, float(thrs[0])
            for thr in thrs:
                ch = Chunking(threshold=float(thr))
                pred = ch.boundaries(seq)[0]  # [T-1]
                f1 = _boundary_f1(pred.long(), gap_truth.long())
                if f1 > best:
                    best, bthr = f1, float(thr)
            surf1.append(best)
            best_thr.append(bthr)
            unif = torch.zeros_like(gap_truth)
            unif[(fpe - 1) :: fpe] = 1
            unif1.append(_boundary_f1(unif.long(), gap_truth.long()))
            g = torch.Generator().manual_seed(s + 3)
            sseq = seq[:, torch.randperm(seq.shape[1], generator=g)]
            sd = (sseq[0, 1:] - sseq[0, :-1]).norm(dim=-1)
            sbest = 0.0
            for thr in torch.linspace(sd.min(), sd.max(), int(e.n_thresholds)):
                pred = (sd > thr).long()
                sbest = max(sbest, _boundary_f1(pred, gap_truth.long()))
            shuf1.append(sbest)

        sf, uf, hf = _mean(surf1), _mean(unif1), _mean(shuf1)
        beats_uniform = bool(sf > uf + 0.1)
        beats_shuffle = bool(sf > hf + 0.1)
        null_supported = bool((not beats_uniform) or (not beats_shuffle))
        return {
            "boundary_f1": round(sf, 4),
            "uniform_rate_f1": round(uf, 4),
            "shuffled_latent_f1": round(hf, 4),
            "delta_over_uniform_rate": round(sf - uf, 4),
            "delta_over_shuffled_latent": round(sf - hf, 4),
            "mean_best_threshold": round(_mean(best_thr), 4),
            "beats_uniform": beats_uniform,
            "beats_shuffled": beats_shuffle,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


def _relational_family(family: int, n_rel: int, dim: int, sep: float, per: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1000 * family)
    rel_axis = torch.randn(dim, generator=torch.Generator().manual_seed(99))  # SHARED across families
    surface = torch.randn(dim, generator=g) * sep  # family-specific surface offset
    xs, ys = [], []
    for r in range(n_rel):
        base = surface + (r - (n_rel - 1) / 2) * 2.0 * rel_axis
        xs.append(base.unsqueeze(0) + 0.6 * torch.randn(per, dim, generator=g))
        ys += [r] * per
    return torch.cat(xs), torch.tensor(ys)


class C5(Experiment):
    id = "c5_transfer_matrix"
    metric = ("off_diagonal_transfer_accuracy", "asymmetry_index", "delta_over_shuffle_floor")
    baseline = "shuffle-label chance floor and a frozen-random projection arm on the same families"
    ablation = "cross-family probe transfer (train i test j) vs within-family; real vs frozen-random"
    null_hypothesis = (
        "off-diagonal transfer is at chance (no shared abstract relation, each family is its own "
        "surface code), or transfer is symmetric and content-driven not relation-driven"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_fam, n_rel = int(e.n_families), int(e.n_relations)
        dim, sep, per = int(e.dim), float(e.separation), int(e.per)

        off_real, off_fr, off_shuf, asym = [], [], [], []
        for s in seeds:
            seed_everything(s)
            fams = [_relational_family(f, n_rel, dim, sep, per, s) for f in range(n_fam)]

            def _matrix(get):
                M = torch.zeros(n_fam, n_fam)
                heads = []
                for i in range(n_fam):
                    xi, yi = get(i)
                    head = torch.nn.Linear(dim, n_rel)
                    opt = torch.optim.Adam(head.parameters(), lr=0.05)
                    for _ in range(int(e.probe_epochs)):
                        opt.zero_grad()
                        F.cross_entropy(head(xi), yi).backward()
                        opt.step()
                    heads.append(head)
                for i in range(n_fam):
                    for j in range(n_fam):
                        xj, yj = get(j)
                        with torch.no_grad():
                            M[i, j] = (heads[i](xj).argmax(-1) == yj).float().mean()
                return M

            M = _matrix(lambda i, src=fams: src[i])
            offmask = ~torch.eye(n_fam, dtype=torch.bool)
            off_real.append(float(M[offmask].mean()))
            asym.append(float((M - M.T).abs()[offmask].mean()))
            frfams = [(frozen_random_projection(fams[f][0], s), fams[f][1]) for f in range(n_fam)]
            Mfr = _matrix(lambda i, src=frfams: src[i])
            off_fr.append(float(Mfr[offmask].mean()))
            shfams = []
            for f in range(n_fam):
                g = torch.Generator().manual_seed(s + f + 50)
                yy = fams[f][1][torch.randperm(fams[f][1].shape[0], generator=g)]
                shfams.append((fams[f][0], yy))
            Msh = _matrix(lambda i, src=shfams: src[i])
            off_shuf.append(float(Msh[offmask].mean()))

        orr, ofr, osh = _mean(off_real), _mean(off_fr), _mean(off_shuf)
        chance = 1.0 / n_rel
        transfers = bool(orr > max(osh, chance) + 0.1 and orr > ofr + 0.05)
        null_supported = bool(orr <= osh + 0.1 or orr <= chance + 0.1)
        return {
            "off_diagonal_transfer_accuracy": round(orr, 4),
            "frozen_random_off_diagonal": round(ofr, 4),
            "shuffle_floor_off_diagonal": round(osh, 4),
            "asymmetry_index": round(_mean(asym), 4),
            "delta_over_shuffle_floor": round(orr - osh, 4),
            "chance": round(chance, 4),
            "relation_transfers": transfers,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


class C6(Experiment):
    id = "c6_dual_process_halting"
    metric = ("halting_vs_difficulty_correlation", "accuracy_at_matched_compute", "gain_per_compute")
    baseline = "single-pass head at matched average FLOPs (diagnostics.compute)"
    ablation = "adaptive-halting refiner vs single-pass; halting step correlated with item difficulty"
    null_hypothesis = (
        "halting step is uncorrelated with difficulty (refiner halts on a content-irrelevant signal), "
        "or at matched average compute the adaptive policy ties the single-pass head (iteration was "
        "just depth)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        epochs, lr = int(e.epochs), float(e.lr)

        corrs, ref_acc, single_acc = [], [], []
        for s in seeds:
            seed_everything(s)
            easy = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.easy_sep),
                seed=s,
            )[0]
            hard = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.hard_sep),
                seed=s + 11,
            )[0]
            x = torch.cat([easy.x, hard.x])
            y = torch.cat([easy.y, hard.y])
            difficulty = torch.cat([torch.zeros(easy.x.shape[0]), torch.ones(hard.x.shape[0])])
            perm = torch.randperm(x.shape[0])
            x, y, difficulty = x[perm], y[perm], difficulty[perm]
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte, dte = x[:cut], y[:cut], x[cut:], y[cut:], difficulty[cut:]
            nc = int(y.max()) + 1

            seed_everything(s)
            refiner = IterativeRefiner(dim, hidden, steps, halt=True)
            rhead = torch.nn.Linear(dim, nc)
            opt = torch.optim.Adam([*refiner.parameters(), *rhead.parameters()], lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                z, _ = refiner(xtr)
                F.cross_entropy(rhead(z), ytr).backward()
                opt.step()
            with torch.no_grad():
                z, used = refiner(xte)
                ref_acc.append(float((rhead(z).argmax(-1) == yte).float().mean()))
            u = used - used.mean()
            dd = dte - dte.mean()
            denom = (u.norm() * dd.norm()).clamp(min=1e-8)
            corrs.append(float((u * dd).sum() / denom))
            avg_steps = float(used.float().mean())

            seed_everything(s)
            depth = max(1, round(avg_steps))
            single = IterativeRefiner(dim, hidden, depth, halt=False)
            shead = torch.nn.Linear(dim, nc)
            opt2 = torch.optim.Adam([*single.parameters(), *shead.parameters()], lr=lr)
            for _ in range(epochs):
                opt2.zero_grad()
                z, _ = single(xtr)
                F.cross_entropy(shead(z), ytr).backward()
                opt2.step()
            with torch.no_grad():
                z, _ = single(xte)
                single_acc.append(float((shead(z).argmax(-1) == yte).float().mean()))

        corr = _mean(corrs)
        ra, sa = _mean(ref_acc), _mean(single_acc)
        flops_ref = refiner_flops(dim, hidden, max(1, round(steps)))
        flops_single = refiner_flops(dim, hidden, max(1, round(steps)))
        compute = matched_within(flops_ref, flops_single)
        halts_with_difficulty = bool(abs(corr) > 0.1)
        gain = ra - sa
        null_supported = bool((not halts_with_difficulty) or abs(gain) <= float(e.margin))
        return {
            "halting_vs_difficulty_correlation": round(corr, 4),
            "refiner_accuracy": round(ra, 4),
            "single_pass_accuracy": round(sa, 4),
            "accuracy_at_matched_compute_gain": round(gain, 4),
            "margin": float(e.margin),
            "compute_matched": compute,
            "halts_more_on_hard": halts_with_difficulty,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


class C7(Experiment):
    id = "c7_chunking_expertise"
    metric = ("steps_to_competence_ratio", "final_accuracy_gap", "delta_over_random_boundaries")
    baseline = "uniform-window summaries and a random-boundary chunking control at matched capacity"
    ablation = "surprise-bounded chunk summaries vs uniform windows vs random boundaries"
    null_hypothesis = (
        "chunked input ties uniform-window input at matched capacity (chunking is a re-binning with no "
        "acquisition benefit), or the C4 boundaries are not reliable enough to help"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_events, fpe = int(e.n_events), int(e.frames_per_event)
        dim, sep = int(e.dim), float(e.separation)
        target_thr = float(e.competence_threshold)

        chunk_steps, unif_steps, rand_steps = [], [], []
        chunk_final, unif_final = [], []
        for s in seeds:
            seed_everything(s)
            n_seq = int(e.n_sequences)
            xs_chunk, xs_unif, xs_rand, ys = [], [], [], []
            for q in range(n_seq):
                seq, truth = _event_sequence(n_events, fpe, dim, sep, s * 100 + q)
                frames = seq[0]
                d = (frames[1:] - frames[:-1]).norm(dim=-1)
                thr = float(d.mean() + d.std())
                bnds = (d > thr).nonzero().flatten().tolist()
                first_chunk_end = (bnds[0] + 1) if bnds else fpe
                xs_chunk.append(frames[:first_chunk_end].mean(0))
                xs_unif.append(frames[:fpe].mean(0))
                g = torch.Generator().manual_seed(s + q + 5)
                rb = int(torch.randint(2, frames.shape[0], (1,), generator=g))
                xs_rand.append(frames[:rb].mean(0))
                ys.append(int(truth.sum() % int(e.n_classes)))  # a learnable label of the sequence
            yt = torch.tensor(ys)
            xc = torch.stack(xs_chunk)
            xu = torch.stack(xs_unif)
            xr = torch.stack(xs_rand)
            nc = max(2, int(yt.max()) + 1)

            def _steps_to_competence(xin, yt=yt, nc=nc):
                cut = int(xin.shape[0] * 0.6)
                xtr, ytr, xte, yte = xin[:cut], yt[:cut], xin[cut:], yt[cut:]
                head = torch.nn.Linear(dim, nc)
                opt = torch.optim.Adam(head.parameters(), lr=0.05)
                reached, final = int(e.max_steps), 0.0
                for step in range(int(e.max_steps)):
                    opt.zero_grad()
                    F.cross_entropy(head(xtr), ytr).backward()
                    opt.step()
                    with torch.no_grad():
                        acc = float((head(xte).argmax(-1) == yte).float().mean())
                    final = acc
                    if acc >= target_thr and reached == int(e.max_steps):
                        reached = step + 1
                return reached, final

            cs, cf = _steps_to_competence(xc)
            us, uf = _steps_to_competence(xu)
            rs, _ = _steps_to_competence(xr)
            chunk_steps.append(cs)
            unif_steps.append(us)
            rand_steps.append(rs)
            chunk_final.append(cf)
            unif_final.append(uf)

        cs, us, rs = _mean(chunk_steps), _mean(unif_steps), _mean(rand_steps)
        ratio = cs / max(1e-6, us)
        chunk_helps = bool(cs < us - 0.5 and cs < rs - 0.5)
        null_supported = bool(ratio >= 0.9 or cs >= rs - 0.5)
        return {
            "steps_to_competence_chunked": round(cs, 4),
            "steps_to_competence_uniform": round(us, 4),
            "steps_to_competence_random_boundaries": round(rs, 4),
            "steps_to_competence_ratio": round(ratio, 4),
            "final_accuracy_gap": round(_mean(chunk_final) - _mean(unif_final), 4),
            "delta_over_random_boundaries": round(rs - cs, 4),
            "chunking_speeds_acquisition": chunk_helps,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


class C8(Experiment):
    id = "c8_concept_blending"
    metric = ("midpoint_decodability", "off_manifold_distance_ratio", "midpoint_vs_frozen_random")
    baseline = "interpolation in a frozen-random projection and the real-latent nearest-neighbour scale"
    ablation = "linear blend of two prototype centroids vs endpoints; real vs frozen-random manifold"
    null_hypothesis = (
        "midpoint blends are off-manifold and undecodable (the latent space is not convex for "
        "concepts, blends are nonsense), or frozen-random shows identical interpolation behavior"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_cat, dim, sep, per = int(e.n_cat), int(e.dim), float(e.separation), int(e.per)

        mid_dec, off_ratio, fr_mid_dec = [], [], []
        for s in seeds:
            seed_everything(s)
            x, y, cents = _category_latents(n_cat, dim, sep, per, s)
            head = torch.nn.Linear(dim, n_cat)
            opt = torch.optim.Adam(head.parameters(), lr=0.05)
            for _ in range(int(e.probe_epochs)):
                opt.zero_grad()
                F.cross_entropy(head(x), y).backward()
                opt.step()
            cA, cB = cents[0], cents[1]
            mid = 0.5 * cA + 0.5 * cB
            with torch.no_grad():
                probs = torch.softmax(head(mid.unsqueeze(0)), dim=-1)[0]
            mid_dec.append(float(probs[0] + probs[1]))  # mass on the two parents
            dmid = float((x - mid).norm(dim=-1).min())
            dnn = torch.cdist(x[:50], x[:50])
            dnn.fill_diagonal_(float("inf"))
            typ = float(dnn.min(1).values.mean())
            off_ratio.append(dmid / max(1e-6, typ))
            xf = frozen_random_projection(x, s)
            centsf = torch.stack([xf[y == c].mean(0) for c in range(n_cat)])
            headf = torch.nn.Linear(dim, n_cat)
            optf = torch.optim.Adam(headf.parameters(), lr=0.05)
            for _ in range(int(e.probe_epochs)):
                optf.zero_grad()
                F.cross_entropy(headf(xf), y).backward()
                optf.step()
            midf = 0.5 * centsf[0] + 0.5 * centsf[1]
            with torch.no_grad():
                pf = torch.softmax(headf(midf.unsqueeze(0)), dim=-1)[0]
            fr_mid_dec.append(float(pf[0] + pf[1]))

        md, orr, frd = _mean(mid_dec), _mean(off_ratio), _mean(fr_mid_dec)
        coherent = bool(md > 0.6 and orr < float(e.off_manifold_ratio))
        differs_from_fr = bool(abs(md - frd) > 0.1)
        null_supported = bool((not coherent) or (not differs_from_fr))
        return {
            "midpoint_decodability": round(md, 4),
            "off_manifold_distance_ratio": round(orr, 4),
            "frozen_random_midpoint_decodability": round(frd, 4),
            "midpoint_vs_frozen_random": round(md - frd, 4),
            "blend_is_coherent": coherent,
            "differs_from_frozen_random": differs_from_fr,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }


class C9(Experiment):
    id = "c9_systematicity_sweep"
    metric = ("systematicity_curve", "cliff_location", "real_minus_frozen_random_auc")
    baseline = "frozen-random projection arm swept identically; shuffle-label chance floor"
    ablation = "held-out fraction sweep on the factor grid; real latents vs frozen-random"
    null_hypothesis = (
        "accuracy cliffs immediately once any combination is held out (pure conjunction memorization, "
        "no systematicity), or the curve is identical to frozen-random (substrate does not buy "
        "systematicity)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        n_a, n_b = int(e.n_a), int(e.n_b)
        dim, per_cell = int(e.dim), int(e.per_cell)
        interaction = float(e.interaction)
        widths = list(e.held_out_widths)  # number of held-out b-cells per a (the held-out depth)

        curve_real: dict[int, list[float]] = {w: [] for w in widths}
        curve_fr: dict[int, list[float]] = {w: [] for w in widths}
        for s in seeds:
            seed_everything(s)
            x, ya, yb = factorized_latents(
                n_a=n_a, n_b=n_b, per_cell=per_cell, dim=dim, interaction=interaction, seed=s
            )
            xf = frozen_random_projection(x, s)
            for w in widths:
                held = {a: [(a + k) % n_b for k in range(w)] for a in range(n_a)}

                def _ho(xin, held=held, ya=ya, yb=yb):
                    is_held = torch.tensor([int(b) in held[int(a)] for a, b in zip(ya, yb, strict=True)])
                    xtr, ytr = xin[~is_held], ya[~is_held]
                    xte, yte = xin[is_held], ya[is_held]
                    if xte.shape[0] < 2 or xtr.shape[0] < 4:
                        return 1.0 / n_a
                    head = torch.nn.Linear(dim, n_a)
                    opt = torch.optim.Adam(head.parameters(), lr=0.05)
                    for _ in range(int(e.probe_epochs)):
                        opt.zero_grad()
                        F.cross_entropy(head(xtr.float()), ytr).backward()
                        opt.step()
                    with torch.no_grad():
                        return float((head(xte.float()).argmax(-1) == yte).float().mean())

                curve_real[w].append(_ho(x))
                curve_fr[w].append(_ho(xf))

        chance = 1.0 / n_a
        real_curve = {w: round(_mean(curve_real[w]), 4) for w in widths}
        fr_curve = {w: round(_mean(curve_fr[w]), 4) for w in widths}
        cliff = None
        for w in widths:
            if real_curve[w] <= chance + 0.15:
                cliff = w
                break
        auc_gap = _mean([real_curve[w] - fr_curve[w] for w in widths])
        graceful = bool(cliff is None or cliff > widths[0])
        beats_fr = bool(auc_gap > 0.05)
        null_supported = bool((cliff is not None and cliff == widths[0]) or (not beats_fr))
        return {
            "systematicity_curve_real": real_curve,
            "systematicity_curve_frozen_random": fr_curve,
            "held_out_widths": list(widths),
            "cliff_location": cliff,
            "chance": round(chance, 4),
            "real_minus_frozen_random_auc": round(auc_gap, 4),
            "graceful_degradation": graceful,
            "beats_frozen_random": beats_fr,
            "seeds": list(seeds),
            "null_supported": null_supported,
        }
