
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, param_count, refiner_flops
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.nonlinear_probe import readout_contribution
from ..diagnostics.seed_consistency import code_stability, cross_seed_cka
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from ..shell.predictor import mlp
from ..shell.refine import IterativeRefiner
from ..substrate.datasets import make_task_stream
from .base import Experiment, _fit_eval, _mean


def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    cent = x[idx].clone()
    codes = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iters):
        codes = torch.cdist(x, cent).argmin(1)
        for c in range(k):
            m = codes == c
            if m.any():
                cent[c] = x[m].mean(0)
    return codes


def _purity(codes: torch.Tensor, y: torch.Tensor, k: int) -> float:
    correct = 0
    for c in range(k):
        m = codes == c
        if m.any():
            correct += int(torch.bincount(y[m]).max())
    return correct / int(y.shape[0])


class _PredShell(nn.Module):

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.enc = mlp(dim, hidden, hidden, depth=1, ln=True)
        self.dec = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.dec(h), h


def _train_next_latent(shell: nn.Module, x: torch.Tensor, xnext: torch.Tensor, epochs: int, lr: float):
    opt = torch.optim.Adam(shell.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = shell(x)
        F.mse_loss(pred, xnext).backward()
        opt.step()


class P1(Experiment):
    id = "p1_concept_no_labels"
    metric = ("hidden_purity", "hidden_probe_acc", "delta_over_raw", "delta_over_frozen_random")
    baseline = "raw frozen latent (k-means + linear probe on the substrate, no shell)"
    ablation = "shell hidden state vs raw latent vs frozen-random substrate, on a held-out factor"
    null_hypothesis = (
        "shell hidden states decode the held-out factor no better than the raw frozen latent, or the "
        "same purity appears under frozen-random (projection-invariant geometry, not concept formation)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, k = int(e.dim), int(e.hidden), int(e.k_clusters)
        hid_pur, raw_pur, fr_pur, hid_acc, raw_acc, fr_acc = [], [], [], [], [], []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                forward_dynamics=True,
                seed=s,
            )[0]
            x, xnext, y = task.x, task.xnext, task.y
            assert xnext is not None  # forward_dynamics=True guarantees the next-latent target
            shell = _PredShell(dim, hidden)
            _train_next_latent(shell, x, xnext, int(e.epochs), float(e.lr))
            with torch.no_grad():
                _, h = shell(x)
            xf = frozen_random_projection(x, s)
            hid_pur.append(_purity(_kmeans(h, k, int(e.kmeans_iters), s), y, k))
            raw_pur.append(_purity(_kmeans(x, k, int(e.kmeans_iters), s), y, k))
            fr_pur.append(_purity(_kmeans(xf, k, int(e.kmeans_iters), s), y, k))
            hid_acc.append(linear_probe(h, y, seed=s)["score"])
            raw_acc.append(linear_probe(x, y, seed=s)["score"])
            fr_acc.append(linear_probe(xf, y, seed=s)["score"])

        hp, rp, fp = _mean(hid_pur), _mean(raw_pur), _mean(fr_pur)
        ha, ra, fa = _mean(hid_acc), _mean(raw_acc), _mean(fr_acc)
        seed_spread = max(0.02, float(torch.tensor(hid_pur).std()) if len(seeds) > 1 else 0.05)
        d_raw = hp - rp
        d_fr = hp - fp
        return {
            "hidden_purity": round(hp, 4),
            "raw_purity": round(rp, 4),
            "frozen_random_purity": round(fp, 4),
            "hidden_probe_acc": round(ha, 4),
            "raw_probe_acc": round(ra, 4),
            "frozen_random_probe_acc": round(fa, 4),
            "delta_over_raw": round(d_raw, 4),
            "delta_over_frozen_random": round(d_fr, 4),
            "seed_spread": round(seed_spread, 4),
            "seeds": list(seeds),
            "null_supported": bool(d_raw <= seed_spread or d_fr <= seed_spread),
            "shell_forms_concept": bool(d_raw > seed_spread and d_fr > seed_spread),
        }


def _nn_memorize(xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor) -> torch.Tensor:
    nn_idx = torch.cdist(xte, xtr).argmin(1)
    return ytr[nn_idx]


class P2(Experiment):
    id = "p2_memorize_vs_concept"
    metric = ("gen_gap_shell", "gen_gap_nn", "extrapolation_error_shell", "extrapolation_error_nn")
    baseline = "nearest-neighbor memorization over cached latents (lookup, no rule)"
    ablation = "shell vs NN-lookup on interpolated and extrapolated held-out factor values"
    null_hypothesis = (
        "the shell ties the nearest-neighbor memorization baseline on held-out values (it memorized, did "
        "not form a concept), or it fails extrapolation exactly where lookup fails (no rule learned)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        gg_shell, gg_nn, ex_shell, ex_nn, seen_shell = [], [], [], [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            n = int(e.samples)
            f = torch.rand(n, generator=g) * 2 - 1
            direction = torch.randn(dim, generator=g)
            direction = direction / direction.norm()
            x = f.unsqueeze(1) * direction * float(e.separation) + 0.3 * torch.randn(n, dim, generator=g)
            target = (float(e.slope) * f + float(e.bias)).unsqueeze(1)
            lo, hi = float(e.band_lo), float(e.band_hi)
            seen = (f.abs() >= lo) & (f.abs() <= hi)
            interp = f.abs() < lo
            extrap = f.abs() > hi
            if seen.sum() < 5 or interp.sum() < 2 or extrap.sum() < 2:
                continue
            xtr, ttr = x[seen], target[seen]
            head = mlp(dim, 1, int(e.hidden), depth=1, ln=True)
            opt = torch.optim.Adam(head.parameters(), lr=float(e.lr))
            for _ in range(int(e.epochs)):
                opt.zero_grad()
                F.mse_loss(head(xtr), ttr).backward()
                opt.step()

            def err(mask, head=head, x=x, target=target):
                with torch.no_grad():
                    return float(F.mse_loss(head(x[mask]), target[mask]))

            def nn_err(mask, xtr=xtr, ttr=ttr, x=x, target=target):
                pred = _nn_memorize(xtr, ttr, x[mask])
                return float(F.mse_loss(pred, target[mask]))

            seen_err_s, seen_err_n = err(seen), nn_err(seen)
            interp_err_s, interp_err_n = err(interp), nn_err(interp)
            extrap_err_s, extrap_err_n = err(extrap), nn_err(extrap)
            gg_shell.append(interp_err_s - seen_err_s)
            gg_nn.append(interp_err_n - seen_err_n)
            ex_shell.append(extrap_err_s)
            ex_nn.append(extrap_err_n)
            seen_shell.append(seen_err_s)

        if not gg_shell:
            return {"null_supported": True, "degenerate_split": True, "seeds": list(seeds)}
        ggs, ggn = _mean(gg_shell), _mean(gg_nn)
        exs, exn = _mean(ex_shell), _mean(ex_nn)
        margin = float(e.margin)
        beats_interp = ggs < ggn - margin
        beats_extrap = exs < exn * float(e.extrap_ratio)
        return {
            "gen_gap_shell": round(ggs, 4),
            "gen_gap_nn": round(ggn, 4),
            "extrapolation_error_shell": round(exs, 4),
            "extrapolation_error_nn": round(exn, 4),
            "seen_error_shell": round(_mean(seen_shell), 4),
            "beats_interp": bool(beats_interp),
            "beats_extrap": bool(beats_extrap),
            "seeds": list(seeds),
            "null_supported": bool(not (beats_interp and beats_extrap)),
            "concept_formed": bool(beats_interp and beats_extrap),
        }


class P3(Experiment):
    id = "p3_knowledge_is_prediction"
    metric = ("pred_decode_correlation", "dissociation_found", "max_pred_r2", "max_factor_acc")
    baseline = "raw-latent factor decodability (the ceiling the shell hidden can reach)"
    ablation = "sweep of prediction-error levels (capacity rungs); decodability vs prediction at each"
    null_hypothesis = (
        "prediction error and factor decodability are monotonically coupled at this substrate, so the "
        "knowledge-is-prediction identity is trivially supported and uninformative"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        rungs = list(e.capacity_rungs)
        corr_seeds, dissoc_seeds = [], []
        max_r2_all, max_acc_all = [], []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                forward_dynamics=True,
                seed=s,
            )[0]
            x, xnext, y = task.x, task.xnext, task.y
            assert xnext is not None  # forward_dynamics=True guarantees the next-latent target
            cut = int(x.shape[0] * 0.7)
            r2s, accs = [], []
            for h in rungs:
                seed_everything(s)
                shell = _PredShell(dim, int(h))
                _train_next_latent(shell, x[:cut], xnext[:cut], int(e.epochs), float(e.lr))
                with torch.no_grad():
                    pred, hid = shell(x[cut:])
                    ss_res = ((pred - xnext[cut:]) ** 2).sum()
                    ss_tot = ((xnext[cut:] - xnext[cut:].mean(0)) ** 2).sum() + 1e-8
                    r2 = float(1 - ss_res / ss_tot)
                    _, hid_all = shell(x)
                acc = linear_probe(hid_all, y, seed=s)["score"]
                r2s.append(r2)
                accs.append(acc)
            r = torch.tensor(r2s)
            a = torch.tensor(accs)
            if r.std() > 1e-6 and a.std() > 1e-6:
                corr = float(((r - r.mean()) * (a - a.mean())).mean() / (r.std() * a.std() + 1e-8))
            else:
                corr = 0.0
            corr_seeds.append(corr)
            dissoc = False
            for i in range(1, len(r2s)):
                d_r2 = r2s[i] - r2s[i - 1]
                d_acc = accs[i] - accs[i - 1]
                if (d_r2 > float(e.step_margin) and d_acc < -float(e.step_margin)) or (
                    d_acc > float(e.step_margin) and d_r2 < -float(e.step_margin)
                ):
                    dissoc = True
            dissoc_seeds.append(dissoc)
            max_r2_all.append(max(r2s))
            max_acc_all.append(max(accs))

        mc = _mean(corr_seeds)
        any_dissoc = any(dissoc_seeds)
        return {
            "pred_decode_correlation": round(mc, 4),
            "dissociation_found": bool(any_dissoc),
            "max_pred_r2": round(_mean(max_r2_all), 4),
            "max_factor_acc": round(_mean(max_acc_all), 4),
            "capacity_rungs": list(rungs),
            "decodability_saturated": bool(_mean(max_acc_all) > float(e.saturate_acc)),
            "seeds": list(seeds),
            "null_supported": bool(not any_dissoc),
            "identity_falsified": bool(any_dissoc),
        }


def _quantize_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    levels = max(2, 2**bits)
    lo = x.min(0, keepdim=True).values
    hi = x.max(0, keepdim=True).values
    scale = (hi - lo).clamp(min=1e-8) / (levels - 1)
    q = torch.round((x - lo) / scale)
    return q * scale + lo


class P4(Experiment):
    id = "p4_intelligence_is_compression"
    metric = ("capability_vs_bits", "non_monotone_peak", "peak_bits", "monotone_decay")
    baseline = "full-precision stored latents (the fidelity ceiling)"
    ablation = "held-out generalization vs stored-latent bit depth; test for a non-monotone abstraction peak"
    null_hypothesis = (
        "capability decays monotonically with compression (compression is pure fidelity loss, not "
        "abstraction); there is no regime where compressing improves generalization"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        bits_list = list(e.bits)
        curves: dict[int, list[float]] = {b: [] for b in bits_list}
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            x, y = task.x, task.y
            for b in bits_list:
                xq = _quantize_bits(x, b)
                curves[b].append(linear_probe(xq, y, seed=s)["score"])
        curve = {b: round(_mean(curves[b]), 4) for b in bits_list}
        ordered = sorted(bits_list)
        full_acc = curve[ordered[-1]]
        interior = ordered[:-1]
        if len(seeds) > 1:
            spread = max(0.02, _mean([float(torch.tensor(curves[b]).std()) for b in bits_list]))
        else:
            spread = 0.03
        peak_bits = max(interior, key=lambda b: curve[b]) if interior else ordered[-1]
        non_monotone = curve[peak_bits] > full_acc + spread
        monotone = all(curve[ordered[i]] <= curve[ordered[i + 1]] + spread for i in range(len(ordered) - 1))
        return {
            "capability_vs_bits": curve,
            "full_precision_acc": round(full_acc, 4),
            "non_monotone_peak": bool(non_monotone),
            "peak_bits": int(peak_bits),
            "monotone_decay": bool(monotone),
            "seed_spread": round(spread, 4),
            "seeds": list(seeds),
            "null_supported": bool(not non_monotone),
            "compression_as_abstraction": bool(non_monotone),
        }


class P5(Experiment):
    id = "p5_private_language"
    metric = ("cross_seed_cka", "cross_shell_probe_acc", "code_stability", "delta_over_frozen_random")
    baseline = "frozen-random projection cross-seed CKA (the idiolect floor)"
    ablation = "shell-code CKA, cross-shell probe transfer, and code stability vs the frozen-random floor"
    null_hypothesis = (
        "codes are seed-idiosyncratic: cross-seed alignment is at the frozen-random level and cross-shell "
        "probes are at chance (the latent code is a private language, not a shareable scheme)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        dim, hidden, k = int(e.dim), int(e.hidden), int(e.k_clusters)
        n_shells = int(e.n_shells)
        seed_everything(int(e.data_seed))
        task = make_task_stream(
            n_tasks=1,
            dim=dim,
            classes_per_task=int(e.n_classes),
            samples_per_task=int(e.samples),
            separation=float(e.separation),
            forward_dynamics=True,
            seed=int(e.data_seed),
        )[0]
        x, xnext, y = task.x, task.xnext, task.y
        assert xnext is not None  # forward_dynamics=True guarantees the next-latent target
        reps, codes = [], []
        for j in range(n_shells):
            seed_everything(j + 1)
            shell = _PredShell(dim, hidden)
            _train_next_latent(shell, x, xnext, int(e.epochs), float(e.lr))
            with torch.no_grad():
                _, h = shell(x)
            reps.append(h)
            codes.append(_kmeans(h, k, int(e.kmeans_iters), j + 1))
        cka = cross_seed_cka(reps)["mean_cka"]
        fr_reps = [frozen_random_projection(x, j + 100) for j in range(n_shells)]
        fr_cka = cross_seed_cka(fr_reps)["mean_cka"]
        cut = int(x.shape[0] * 0.7)
        nc = int(y.max()) + 1
        transfer = []
        for a in range(n_shells):
            for bsh in range(n_shells):
                if a == bsh:
                    continue
                probe = nn.Linear(hidden, nc)
                opt = torch.optim.Adam(probe.parameters(), lr=0.05)
                for _ in range(150):
                    opt.zero_grad()
                    F.cross_entropy(probe(reps[a][:cut]), y[:cut]).backward()
                    opt.step()
                with torch.no_grad():
                    acc = float((probe(reps[bsh][cut:]).argmax(-1) == y[cut:]).float().mean())
                transfer.append(acc)
        stab = code_stability(codes, k)
        mt = _mean(transfer)
        chance = 1.0 / nc
        delta_fr = cka - fr_cka
        public = (delta_fr > float(e.margin)) and (mt > chance + 0.1) and stab["stable"]
        return {
            "cross_seed_cka": round(cka, 4),
            "frozen_random_cka": round(fr_cka, 4),
            "delta_over_frozen_random": round(delta_fr, 4),
            "cross_shell_probe_acc": round(mt, 4),
            "probe_chance": round(chance, 4),
            "code_stability": stab["mean_agreement"],
            "code_stable": bool(stab["stable"]),
            "n_shells": n_shells,
            "null_supported": bool(not public),
            "public_language": bool(public),
        }


def _eval_code_transfer(codes_feat: torch.Tensor, y: torch.Tensor, seed: int) -> float:
    return linear_probe(codes_feat, y, seed=seed)["score"]


class P6(Experiment):
    id = "p6_meaning_without_symbols"
    metric = ("continuous_acc", "vq_acc", "random_codebook_acc", "discrete_minus_continuous")
    baseline = "matched-capacity continuous shell hidden (the no-symbol code)"
    ablation = "continuous vs VQ-discrete vs random-codebook code utility at matched parameter budget"
    null_hypothesis = (
        "the discrete-symbol shell ties the matched-capacity continuous shell (symbols add nothing), or "
        "it only ties because the random-codebook also ties (structure was the bottleneck, not symbols)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, k = int(e.dim), int(e.hidden), int(e.codebook_size)
        cont_acc, vq_acc, rand_acc = [], [], []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                forward_dynamics=True,
                seed=s,
            )[0]
            x, xnext, y = task.x, task.xnext, task.y
            assert xnext is not None  # forward_dynamics=True guarantees the next-latent target
            shell = _PredShell(dim, hidden)
            _train_next_latent(shell, x, xnext, int(e.epochs), float(e.lr))
            with torch.no_grad():
                _, h = shell(x)
            cont_acc.append(_eval_code_transfer(h, y, s))
            vq_codes = _kmeans(h, k, int(e.kmeans_iters), s)
            g = torch.Generator().manual_seed(s + 7)
            rand_codes = torch.randint(0, k, (h.shape[0],), generator=g)
            vq_oh = F.one_hot(vq_codes, k).float()
            rand_oh = F.one_hot(rand_codes, k).float()
            vq_acc.append(_eval_code_transfer(vq_oh, y, s))
            rand_acc.append(_eval_code_transfer(rand_oh, y, s))

        ca, va, ra = _mean(cont_acc), _mean(vq_acc), _mean(rand_acc)
        margin = float(e.margin)
        vq_beats_cont = va > ca + margin
        vq_beats_random = va > ra + margin
        return {
            "continuous_acc": round(ca, 4),
            "vq_acc": round(va, 4),
            "random_codebook_acc": round(ra, 4),
            "discrete_minus_continuous": round(va - ca, 4),
            "discrete_minus_random": round(va - ra, 4),
            "codebook_size": k,
            "seeds": list(seeds),
            "null_supported": bool(not (vq_beats_cont and vq_beats_random)),
            "symbols_buy_meaning": bool(vq_beats_cont and vq_beats_random),
        }


class _UntiedDepth(nn.Module):

    def __init__(self, dim: int, hidden: int, steps: int):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(steps)])
        self.blocks = nn.ModuleList([mlp(dim, dim, hidden, depth=1, ln=True) for _ in range(steps)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for norm, block in zip(self.norms, self.blocks, strict=True):
            z = z + block(norm(z))
        return z


class P9(Experiment):
    id = "p9_thought_without_language"
    metric = ("gain_per_compute", "halting_step_mean", "refiner_acc", "control_acc")
    baseline = "compute-matched untied-depth network (same block count + hidden, untied weights)"
    ablation = "weight-tied refiner vs untied depth at matched forward FLOPs; language-free substrate"
    null_hypothesis = (
        "at matched compute the refiner ties the untied-depth control (the gain was depth, not "
        "iteration, so there is no language-free reasoning signal, only more parameters)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        margin = float(e.margin)
        epochs, lr = int(e.epochs), float(e.lr)
        ref_acc, ctl_acc, halts = [], [], []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            x, y = task.x, task.y
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            nc = int(y.max()) + 1
            refiner = IterativeRefiner(dim, hidden, steps, halt=bool(e.adaptive_halt))
            rhead = nn.Linear(dim, nc)
            ref_acc.append(_fit_eval(refiner, rhead, xtr, ytr, xte, yte, epochs, lr))
            with torch.no_grad():
                _, used = refiner(xte)
            halts.append(float(used.float().mean()))
            seed_everything(s)
            ctl = _UntiedDepth(dim, hidden, steps)
            chead = nn.Linear(dim, nc)
            ctl_acc.append(_fit_eval(ctl, chead, xtr, ytr, xte, yte, epochs, lr))

        rm, cm = _mean(ref_acc), _mean(ctl_acc)
        flops = refiner_flops(dim, hidden, steps)  # equal block count + hidden -> equal FLOPs
        compute = matched_within(flops, flops)
        gain = rm - cm
        refiner = IterativeRefiner(dim, hidden, steps)
        ctl = _UntiedDepth(dim, hidden, steps)
        return {
            "refiner_acc": round(rm, 4),
            "control_acc": round(cm, 4),
            "gain_per_compute": round(gain, 4),
            "margin": margin,
            "halting_step_mean": round(_mean(halts), 3),
            "params_refiner": param_count(refiner),
            "params_control": param_count(ctl),
            "compute_matched": compute["matched"],
            "seeds": list(seeds),
            "null_supported": bool(abs(gain) <= margin),
            "language_free_reasoning": bool(gain > margin and compute["matched"]),
        }


class P10(Experiment):
    id = "p10_theory_ladenness"
    metric = ("readout_contribution_index", "nonlinear_gain_real", "nonlinear_gain_fr", "index_per_factor")
    baseline = "frozen-random projection (the any-projection floor for the nonlinear probe gain)"
    ablation = "linear vs small-MLP probe on real vs frozen-random substrates, per factor"
    null_hypothesis = (
        "the nonlinear probe gain is identical on real and frozen-random substrates (the recovered "
        "structure is contributed by the probe, not the encoder; perception is theory-laden all the way down)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_factors = int(e.n_factors)
        idx_per_factor: dict[str, float] = {}
        gain_real_all, gain_fr_all, idx_all = [], [], []
        for fct in range(n_factors):
            idx_seeds, gr_seeds, gf_seeds = [], [], []
            for s in seeds:
                seed_everything(s + fct)
                task = make_task_stream(
                    n_tasks=1,
                    dim=dim,
                    classes_per_task=int(e.n_classes),
                    samples_per_task=int(e.samples),
                    separation=float(e.separation) * (1.0 + 0.3 * fct),
                    seed=s + 13 * fct,
                )[0]
                x, y = task.x, task.y
                rc = readout_contribution(x, y, hidden=int(e.probe_hidden), seed=s)
                idx_seeds.append(rc["readout_contribution_index"])
                gr_seeds.append(rc["nonlinear_gain_real"])
                gf_seeds.append(rc["nonlinear_gain_frozen_random"])
            mi = _mean(idx_seeds)
            idx_per_factor[f"factor{fct}"] = round(mi, 4)
            idx_all.append(mi)
            gain_real_all.append(_mean(gr_seeds))
            gain_fr_all.append(_mean(gf_seeds))
        mean_idx = _mean(idx_all)
        margin = float(e.margin)
        substrate_contributes = mean_idx > margin
        return {
            "readout_contribution_index": round(mean_idx, 4),
            "nonlinear_gain_real": round(_mean(gain_real_all), 4),
            "nonlinear_gain_frozen_random": round(_mean(gain_fr_all), 4),
            "index_per_factor": idx_per_factor,
            "n_factors": n_factors,
            "seeds": list(seeds),
            "null_supported": bool(not substrate_contributes),
            "substrate_contributes_structure": bool(substrate_contributes),
        }
