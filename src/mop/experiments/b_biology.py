
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, mlp_flops, param_count
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from ..shell.consolidation import EWC
from ..shell.predictor import mlp
from ..substrate.datasets import make_task_stream
from .base import Experiment, _mean, _spread


def _domain_stream(dim: int, n_tasks: int, classes: int, samples: int, sep: float, seed: int):
    return make_task_stream(
        n_tasks=n_tasks,
        dim=dim,
        classes_per_task=classes,
        samples_per_task=samples,
        separation=sep,
        incremental="domain",
        seed=seed,
    )


def _eval_acc(head: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((head(x).argmax(-1) == y).float().mean())


class B1(Experiment):
    id = "b1_clonal_selection"
    metric = ("backward_transfer", "retention_at_matched_slots", "mutation_effect")
    baseline = "PER (priority) eviction and random eviction at the SAME buffer slots"
    ablation = "clonal affinity eviction, with and without somatic mutation, vs PER vs random"
    null_hypothesis = (
        "at matched slots clonal-selection eviction ties PER and random on backward transfer "
        "within the seed spread, and the somatic-mutation arm does not beat the no-mutation arm"
    )
    tier = "cpu-now"

    @staticmethod
    def _train_with_buffer(
        stream, dim, nc, cap, policy, mutate, epochs, lr, replay_b, seed
    ) -> tuple[float, list[float]]:
        seed_everything(seed)
        head = mlp(dim, nc, 64, depth=1)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        bx = torch.zeros(0, dim)
        by = torch.zeros(0, dtype=torch.long)
        baff = torch.zeros(0)  # affinity score per stored exemplar
        g = torch.Generator().manual_seed(seed + 7)
        acc_after_each: list[float] = []
        for task in stream:
            x, y = task.x, task.y
            for _ in range(epochs):
                opt.zero_grad()
                loss = F.cross_entropy(head(x), y)
                if len(bx):
                    loss = loss + F.cross_entropy(head(bx), by)
                loss.backward()
                opt.step()
            with torch.no_grad():
                per_loss = F.cross_entropy(head(x), y, reduction="none")
            cand_aff = per_loss  # higher loss => higher learning affinity (immune analogy)
            bx = torch.cat([bx, x])
            by = torch.cat([by, y])
            baff = torch.cat([baff, cand_aff])
            if len(bx) > cap:
                if policy == "clonal":
                    keep = baff.topk(cap).indices
                elif policy == "per":
                    keep = baff.topk(cap).indices  # PER also keeps high-priority (high error)
                else:  # random reservoir
                    keep = torch.randperm(len(bx), generator=g)[:cap]
                bx, by, baff = bx[keep], by[keep], baff[keep]
                if policy == "clonal" and mutate:
                    m = int(0.25 * len(bx))
                    idx = torch.randperm(len(bx), generator=g)[:m]
                    bx[idx] = bx[idx] + 0.05 * torch.randn(m, dim, generator=g)
            acc_after_each.append(_eval_acc(head, stream[0].x, stream[0].y))
        bwt = _eval_acc(head, stream[0].x, stream[0].y)
        return bwt, acc_after_each

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        cap = int(e.capacity)
        clon, clon_nomut, per, rnd = [], [], [], []
        for s in seeds:
            stream = _domain_stream(dim, int(e.n_tasks), nc, int(e.samples), float(e.separation), s)
            clon.append(
                self._train_with_buffer(
                    stream, dim, nc, cap, "clonal", True, int(e.epochs), float(e.lr), int(e.replay_batch), s
                )[0]
            )
            clon_nomut.append(
                self._train_with_buffer(
                    stream, dim, nc, cap, "clonal", False, int(e.epochs), float(e.lr), int(e.replay_batch), s
                )[0]
            )
            per.append(
                self._train_with_buffer(
                    stream, dim, nc, cap, "per", False, int(e.epochs), float(e.lr), int(e.replay_batch), s
                )[0]
            )
            rnd.append(
                self._train_with_buffer(
                    stream, dim, nc, cap, "random", False, int(e.epochs), float(e.lr), int(e.replay_batch), s
                )[0]
            )

        cm, pm, rm, nm = _mean(clon), _mean(per), _mean(rnd), _mean(clon_nomut)
        spread = max(_spread(clon), _spread(per), _spread(rnd))
        gain_vs_per = cm - pm
        gain_vs_random = cm - rm
        mutation_effect = cm - nm  # clonal+mutate minus clonal-no-mutate
        beats = (gain_vs_per > spread) and (gain_vs_random > spread)
        mutation_helps = mutation_effect > spread
        return {
            "bwt_clonal": round(cm, 4),
            "bwt_clonal_no_mutation": round(nm, 4),
            "bwt_per": round(pm, 4),
            "bwt_random": round(rm, 4),
            "gain_vs_per": round(gain_vs_per, 4),
            "gain_vs_random": round(gain_vs_random, 4),
            "mutation_effect": round(mutation_effect, 4),
            "seed_spread": round(spread, 4),
            "capacity_slots": cap,
            "seeds": list(seeds),
            "clonal_beats_baselines": bool(beats),
            "mutation_helps": bool(mutation_helps),
            "null_supported": bool((not beats) and (not mutation_helps)),
        }


class B2(Experiment):
    id = "b2_baldwin_meta_init"
    metric = ("bwt_naive_from_meta", "bwt_naive_from_generic", "assimilation_gain")
    baseline = "a naive learner (no replay, no EWC) from a GENERIC init on the same stream"
    ablation = "naive-from-meta-init vs naive-from-generic-init; meta-init trained WITH protection"
    null_hypothesis = (
        "the meta-init confers no retention advantage to a naive learner over a generic init "
        "(nothing assimilated; protection lives only in the online mechanism)"
    )
    tier = "cpu-now"

    @staticmethod
    def _naive_bwt(stream, init_state, dim, nc, epochs, lr, seed) -> float:
        seed_everything(seed)
        head = mlp(dim, nc, 48, depth=1)
        if init_state is not None:
            head.load_state_dict(init_state)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for task in stream:
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(head(task.x), task.y).backward()
                opt.step()
        return _eval_acc(head, stream[0].x, stream[0].y)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        from_meta, from_generic = [], []
        for s in seeds:
            seed_everything(s)
            meta = mlp(dim, nc, 48, depth=1)
            meta_lr = float(e.meta_lr)
            for it in range(int(e.outer_iters)):
                stream = _domain_stream(
                    dim, int(e.n_tasks), nc, int(e.samples), float(e.separation), s * 100 + it
                )
                inner = mlp(dim, nc, 48, depth=1)
                inner.load_state_dict(meta.state_dict())
                opt = torch.optim.Adam(inner.parameters(), lr=float(e.lr))
                buf_x = torch.zeros(0, dim)
                buf_y = torch.zeros(0, dtype=torch.long)
                for task in stream:  # inner loop with replay protection active
                    for _ in range(int(e.inner_epochs)):
                        opt.zero_grad()
                        loss = F.cross_entropy(inner(task.x), task.y)
                        if len(buf_x):
                            loss = loss + F.cross_entropy(inner(buf_x), buf_y)
                        loss.backward()
                        opt.step()
                    buf_x = torch.cat([buf_x, task.x])
                    buf_y = torch.cat([buf_y, task.y])
                with torch.no_grad():
                    for pm_, pi in zip(meta.parameters(), inner.parameters(), strict=True):
                        pm_.add_(meta_lr * (pi.detach() - pm_))
            meta_state = {k: v.clone() for k, v in meta.state_dict().items()}
            eval_stream = _domain_stream(
                dim, int(e.n_tasks), nc, int(e.samples), float(e.separation), s + 9999
            )
            from_meta.append(
                self._naive_bwt(eval_stream, meta_state, dim, nc, int(e.eval_epochs), float(e.lr), s)
            )
            from_generic.append(
                self._naive_bwt(eval_stream, None, dim, nc, int(e.eval_epochs), float(e.lr), s)
            )

        mm, gm = _mean(from_meta), _mean(from_generic)
        spread = max(_spread(from_meta), _spread(from_generic))
        assimilation = mm - gm
        assimilated = assimilation > spread
        return {
            "bwt_naive_from_meta": round(mm, 4),
            "bwt_naive_from_generic": round(gm, 4),
            "assimilation_gain": round(assimilation, 4),
            "seed_spread": round(spread, 4),
            "seeds": list(seeds),
            "protection_assimilated": bool(assimilated),
            "null_supported": bool(not assimilated),
        }


class B3(Experiment):
    id = "b3_stigmergic_curriculum"
    metric = ("learnable_coverage", "noisy_tv_time_share", "stigmergic_vs_lp")
    baseline = "explicit learning-progress (LP) selector over the same regions"
    ablation = "stigmergic pheromone field (deposit=LP, evaporation) vs explicit LP vs raw-error deposit"
    null_hypothesis = (
        "the stigmergic field ties explicit LP selection (it is a reparameterization of the same "
        "bandit), and a raw-error deposit chases the noisy-TV unless the deposit uses LP"
    )
    tier = "cpu-now"

    @staticmethod
    def _samplers(dim, noise_scale, batch, seed):
        g = torch.Generator().manual_seed(seed + 100)
        centers = torch.randn(2, dim, generator=g)
        rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
        shift = 0.3

        def learnable_a():
            yy = torch.randint(0, 2, (batch,), generator=g)
            x = centers[yy] * 2.0 + 0.5 * torch.randn(batch, dim, generator=g)
            return x, x @ rot

        def learnable_b():
            yy = torch.randint(0, 2, (batch,), generator=g)
            x = centers[yy] * 2.0 + shift + 0.5 * torch.randn(batch, dim, generator=g)
            return x, (x - shift) @ rot + shift

        def noise():
            x = torch.randn(batch, dim, generator=g)
            return x, torch.randn(batch, dim, generator=g) * noise_scale  # irreducible target

        return [learnable_a, learnable_b, noise]

    @staticmethod
    def _select_run(deposit_signal, samplers, dim, steps, decay, seed, temp=8.0) -> dict:
        seed_everything(seed)
        nreg = len(samplers)
        phero = torch.zeros(nreg)  # leaky integral of the deposit signal
        preds = [mlp(dim, dim, 48, depth=1) for _ in samplers]
        opts = [torch.optim.Adam(p.parameters(), lr=1e-2) for p in preds]
        prev_loss: list[float | None] = [None] * nreg
        visits = torch.zeros(nreg)
        g = torch.Generator().manual_seed(seed + 3)
        for _ in range(steps):
            probs = torch.softmax(temp * phero, dim=0)
            r = int(torch.multinomial(probs, 1, generator=g))
            visits[r] += 1
            xr, tr = samplers[r]()  # fresh batch each visit
            opts[r].zero_grad()
            loss = F.mse_loss(preds[r](xr), tr)
            loss.backward()
            opts[r].step()
            with torch.no_grad():
                losses = []
                for _ in range(4):
                    xe, te = samplers[r]()
                    losses.append(float(F.mse_loss(preds[r](xe), te)))
                new_loss = sum(losses) / len(losses)
            pl = prev_loss[r]
            lp = 0.0 if pl is None else max(0.0, pl - new_loss)
            prev_loss[r] = new_loss
            dep = lp if deposit_signal == "lp" else new_loss / dim
            phero = phero * decay  # evaporation everywhere
            phero[r] = phero[r] + (1 - decay) * dep  # deposit at the visited region
        share = (visits / visits.sum()).tolist()
        return {"share": share, "phero": phero.tolist()}

    @staticmethod
    def _explicit_lp(samplers, dim, steps, seed) -> list[float]:
        seed_everything(seed)
        nreg = len(samplers)
        preds = [mlp(dim, dim, 48, depth=1) for _ in samplers]
        opts = [torch.optim.Adam(p.parameters(), lr=1e-2) for p in preds]
        lp_est = [1.0] * nreg
        prev: list[float | None] = [None] * nreg
        visits = torch.zeros(nreg)
        g = torch.Generator().manual_seed(seed + 5)
        for _ in range(steps):
            if float(torch.rand(1, generator=g)) < 0.15:
                r = int(torch.randint(0, nreg, (1,), generator=g))
            else:
                r = int(torch.tensor(lp_est).argmax())
            visits[r] += 1
            xr, tr = samplers[r]()
            opts[r].zero_grad()
            F.mse_loss(preds[r](xr), tr).backward()
            opts[r].step()
            with torch.no_grad():
                losses = []
                for _ in range(4):
                    xe, te = samplers[r]()
                    losses.append(float(F.mse_loss(preds[r](xe), te)))
                nl = sum(losses) / len(losses)
            pv = prev[r]
            lp_est[r] = 0.0 if pv is None else max(0.0, pv - nl)
            prev[r] = nl
        return (visits / visits.sum()).tolist()

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, steps, decay = int(e.dim), int(e.steps), float(e.decay)
        batch = int(e.batch)
        stig_cov, lp_cov, stig_noise, rawerr_noise = [], [], [], []
        for s in seeds:
            samplers = self._samplers(dim, float(e.noise_scale), batch, s)
            noise_idx = 2
            stig = self._select_run("lp", samplers, dim, steps, decay, s)
            stig_raw = self._select_run("raw_error", samplers, dim, steps, decay, s)
            lp = self._explicit_lp(samplers, dim, steps, s)
            # learnable coverage = share spent on the two learnable regions
            stig_cov.append(stig["share"][0] + stig["share"][1])
            lp_cov.append(lp[0] + lp[1])
            stig_noise.append(stig["share"][noise_idx])
            rawerr_noise.append(stig_raw["share"][noise_idx])

        sc, lc = _mean(stig_cov), _mean(lp_cov)
        sn, rn = _mean(stig_noise), _mean(rawerr_noise)
        spread = max(_spread(stig_cov), _spread(lp_cov))
        coverage_gap = abs(sc - lc)
        ties_lp = coverage_gap <= spread + 0.05
        stig_rejects_noise = sn < 0.34  # below uniform third over three regions
        rawerr_chases = rn >= sn  # raw-error deposit spends at least as much on noise as LP-stigmergy
        return {
            "stigmergic_learnable_coverage": round(sc, 4),
            "lp_learnable_coverage": round(lc, 4),
            "coverage_gap": round(coverage_gap, 4),
            "stigmergic_noise_share": round(sn, 4),
            "rawerror_noise_share": round(rn, 4),
            "seed_spread": round(spread, 4),
            "seeds": list(seeds),
            "stigmergy_ties_lp": bool(ties_lp),
            "stigmergy_rejects_noisy_tv": bool(stig_rejects_noise),
            "rawerror_chases_noisy_tv": bool(rawerr_chases),
            "null_supported": bool(ties_lp and rawerr_chases),
        }


class B4(Experiment):
    id = "b4_homeostatic_scaling"
    metric = ("backward_transfer", "homeostasis_effect", "homeostasis_on_top_of_ewc")
    baseline = "naive sequential (no protection) and EWC-only on the domain stream"
    ablation = "none, homeostasis-only, EWC-only, EWC+homeostasis; frozen-random substrate arm"
    null_hypothesis = (
        "homeostatic scaling alone does not beat naive on backward transfer (a normalization trick, "
        "not protection), and it adds nothing on top of EWC (same protected directions)"
    )
    tier = "cpu-now"

    @staticmethod
    def _scale_to_target(layer: nn.Linear, x: torch.Tensor, target: float) -> None:
        with torch.no_grad():
            act = layer(x).abs().mean().clamp_min(1e-6)
            layer.weight.mul_(target / act)
            if layer.bias is not None:
                layer.bias.mul_(target / act)

    @staticmethod
    def _run_arm(stream, dim, nc, use_ewc, use_homeo, epochs, lr, target, seed) -> float:
        seed_everything(seed)
        body = nn.Linear(dim, 48)
        act = nn.GELU()
        head = nn.Linear(48, nc)
        net = nn.Sequential(body, act, head)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        ewc = EWC(lam=100.0) if use_ewc else None
        for task in stream:
            for _ in range(epochs):
                opt.zero_grad()
                loss = F.cross_entropy(net(task.x), task.y)
                if ewc is not None and ewc.fisher:
                    loss = loss + ewc.penalty(net)
                loss.backward()
                opt.step()
            if use_homeo:
                B4._scale_to_target(body, task.x, target)
            if ewc is not None:
                ewc.estimate(net, [task], lambda b: F.cross_entropy(net(b.x), b.y), samples=1)
        return _eval_acc(net, stream[0].x, stream[0].y)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        epochs, lr, target = int(e.epochs), float(e.lr), float(e.act_target)
        naive, homeo, ewc, ewc_homeo, fr_homeo = [], [], [], [], []
        for s in seeds:
            stream = _domain_stream(dim, int(e.n_tasks), nc, int(e.samples), float(e.separation), s)
            naive.append(self._run_arm(stream, dim, nc, False, False, epochs, lr, target, s))
            homeo.append(self._run_arm(stream, dim, nc, False, True, epochs, lr, target, s))
            ewc.append(self._run_arm(stream, dim, nc, True, False, epochs, lr, target, s))
            ewc_homeo.append(self._run_arm(stream, dim, nc, True, True, epochs, lr, target, s))
            fr_stream = [
                type(t)(t.name, frozen_random_projection(t.x, s), t.y, None, t.n_classes, t.task_id)
                for t in stream
            ]
            fr_homeo.append(self._run_arm(fr_stream, dim, nc, False, True, epochs, lr, target, s))

        nv, hm, ew, eh, frh = _mean(naive), _mean(homeo), _mean(ewc), _mean(ewc_homeo), _mean(fr_homeo)
        spread = max(_spread(naive), _spread(homeo), _spread(ewc), _spread(ewc_homeo))
        homeo_effect = hm - nv
        on_top = eh - ew
        homeo_helps = homeo_effect > spread
        adds_on_top = on_top > spread
        needs_real = (homeo_effect - (frh - nv)) > spread  # effect bigger on real than frozen-random
        return {
            "bwt_naive": round(nv, 4),
            "bwt_homeostasis": round(hm, 4),
            "bwt_ewc": round(ew, 4),
            "bwt_ewc_homeostasis": round(eh, 4),
            "bwt_frozen_random_homeostasis": round(frh, 4),
            "homeostasis_effect": round(homeo_effect, 4),
            "homeostasis_on_top_of_ewc": round(on_top, 4),
            "seed_spread": round(spread, 4),
            "seeds": list(seeds),
            "homeostasis_helps_alone": bool(homeo_helps),
            "homeostasis_adds_on_top_of_ewc": bool(adds_on_top),
            "effect_needs_real_substrate": bool(needs_real),
            "null_supported": bool((not homeo_helps) and (not adds_on_top)),
        }


class B5(Experiment):
    id = "b5_degeneracy_robustness"
    metric = ("degradation_auc", "backward_transfer", "degeneracy_vs_baselines")
    baseline = "matched-param single predictor and K identical copies (pure redundancy)"
    ablation = "structurally-distinct sub-predictors (degenerate) vs single vs copies, at matched compute"
    null_hypothesis = (
        "degeneracy ties the matched-param single predictor and ties pure redundancy on both "
        "perturbation robustness and retention (any win was capacity, removed by matched compute)"
    )
    tier = "cpu-now"

    @staticmethod
    def _build_degenerate(dim, nc, widths):
        acts = [nn.GELU(), nn.ReLU(), nn.Tanh()]
        mods = []
        for i, w in enumerate(widths):
            mods.append(nn.Sequential(nn.Linear(dim, w), acts[i % len(acts)], nn.Linear(w, nc)))
        return nn.ModuleList(mods)

    @staticmethod
    def _train_vote(members, x, y, epochs, lr) -> None:
        opt = torch.optim.Adam([p for m in members for p in m.parameters()], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            logits = torch.stack([m(x) for m in members], 0).mean(0)
            F.cross_entropy(logits, y).backward()
            opt.step()

    @staticmethod
    def _vote_acc(members, x, y) -> float:
        with torch.no_grad():
            logits = torch.stack([m(x) for m in members], 0).mean(0)
            return float((logits.argmax(-1) == y).float().mean())

    @staticmethod
    def _degradation_auc(members, x, y, strengths, seed) -> float:
        g = torch.Generator().manual_seed(seed + 11)
        accs = []
        for st in strengths:
            xp = x + st * torch.randn(x.shape, generator=g)
            accs.append(B5._vote_acc(members, xp, y))
        return _mean(accs)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        widths = [int(w) for w in e.degenerate_widths]
        strengths = [float(x) for x in e.perturb_strengths]
        deg_auc, sin_auc, cop_auc = [], [], []
        deg_bwt: list[float] = []
        sin_bwt: list[float] = []
        cop_bwt: list[float] = []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            x, y = task.x, task.y
            cut = int(0.7 * len(x))
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            stream = _domain_stream(dim, int(e.n_tasks), nc, int(e.samples), float(e.separation), s)

            deg = self._build_degenerate(dim, nc, widths)
            self._train_vote(deg, xtr, ytr, int(e.epochs), float(e.lr))
            total = sum(param_count(m) for m in deg)
            sw = max(8, int(total / (dim + nc)))
            single = nn.ModuleList([nn.Sequential(nn.Linear(dim, sw), nn.GELU(), nn.Linear(sw, nc))])
            self._train_vote(single, xtr, ytr, int(e.epochs), float(e.lr))
            copies = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(dim, widths[0]), nn.GELU(), nn.Linear(widths[0], nc))
                    for _ in widths
                ]
            )
            self._train_vote(copies, xtr, ytr, int(e.epochs), float(e.lr))

            deg_auc.append(self._degradation_auc(deg, xte, yte, strengths, s))
            sin_auc.append(self._degradation_auc(single, xte, yte, strengths, s))
            cop_auc.append(self._degradation_auc(copies, xte, yte, strengths, s))

            for m, store in (
                (self._build_degenerate(dim, nc, widths), deg_bwt),
                (nn.ModuleList([nn.Sequential(nn.Linear(dim, sw), nn.GELU(), nn.Linear(sw, nc))]), sin_bwt),
                (
                    nn.ModuleList(
                        [
                            nn.Sequential(nn.Linear(dim, widths[0]), nn.GELU(), nn.Linear(widths[0], nc))
                            for _ in widths
                        ]
                    ),
                    cop_bwt,
                ),
            ):
                for t in stream:
                    self._train_vote(m, t.x, t.y, int(e.epochs), float(e.lr))
                store.append(self._vote_acc(m, stream[0].x, stream[0].y))

        da, sa, ca = _mean(deg_auc), _mean(sin_auc), _mean(cop_auc)
        db, sb, cb = _mean(deg_bwt), _mean(sin_bwt), _mean(cop_bwt)
        spread = max(_spread(deg_auc), _spread(sin_auc), _spread(cop_auc))
        flatter = (da - sa > spread) and (da - ca > spread)
        retains = (db - sb > spread) and (db - cb > spread)
        flops_deg = sum(mlp_flops([dim, w, nc]) for w in widths)
        total_params = sum(param_count(m) for m in self._build_degenerate(dim, nc, widths))
        sw = max(8, int(total_params / (dim + nc)))
        flops_single = mlp_flops([dim, sw, nc])
        compute = matched_within(flops_deg, flops_single)
        flatter = flatter and compute["matched"]
        retains = retains and compute["matched"]
        return {
            "degradation_auc_degenerate": round(da, 4),
            "degradation_auc_single": round(sa, 4),
            "degradation_auc_copies": round(ca, 4),
            "bwt_degenerate": round(db, 4),
            "bwt_single": round(sb, 4),
            "bwt_copies": round(cb, 4),
            "seed_spread": round(spread, 4),
            "degenerate_flops": int(flops_deg),
            "single_flops": int(flops_single),
            "compute_matched": compute,
            "seeds": list(seeds),
            "degenerate_more_robust": bool(flatter),
            "degenerate_retains_better": bool(retains),
            "null_supported": bool((not flatter) and (not retains)),
        }


class B6(Experiment):
    id = "b6_offline_consolidation"
    metric = ("bwt_online", "bwt_offline", "bwt_dreaming")
    baseline = "online replay at a fixed total replay budget"
    ablation = "online vs offline-interleaved vs offline+generative-dreaming, at matched replay budget"
    null_hypothesis = (
        "offline interleaving ties online replay at matched replay budget (scheduling does not "
        "matter, only sample count), and the dreaming arm fails to match stored replay"
    )
    tier = "cpu-now"

    @staticmethod
    def _run(stream, dim, nc, mode, budget, epochs, lr, seed) -> float:
        seed_everything(seed)
        head = mlp(dim, nc, 48, depth=1)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        bx = torch.zeros(0, dim)
        by = torch.zeros(0, dtype=torch.long)
        g = torch.Generator().manual_seed(seed + 13)
        per_task_budget = max(1, budget // max(1, len(stream)))
        for ti, task in enumerate(stream):
            for _ in range(epochs):
                opt.zero_grad()
                loss = F.cross_entropy(head(task.x), task.y)
                if mode == "online" and len(bx):
                    k = min(per_task_budget, len(bx))
                    idx = torch.randperm(len(bx), generator=g)[:k]
                    loss = loss + F.cross_entropy(head(bx[idx]), by[idx])
                loss.backward()
                opt.step()
            bx = torch.cat([bx, task.x])
            by = torch.cat([by, task.y])
            if mode in ("offline", "dreaming") and ti > 0:
                off_opt = torch.optim.Adam(head.parameters(), lr=lr * 0.5)
                k = min(per_task_budget, len(bx))
                for _ in range(epochs):
                    idx = torch.randperm(len(bx), generator=g)[:k]
                    rx, ry = bx[idx], by[idx]
                    if mode == "dreaming":
                        rx = rx + 0.1 * torch.randn(rx.shape, generator=g)
                    off_opt.zero_grad()
                    F.cross_entropy(head(rx), ry).backward()
                    off_opt.step()
        return _eval_acc(head, stream[0].x, stream[0].y)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc, budget = int(e.dim), int(e.n_classes), int(e.replay_budget)
        on, off, dream = [], [], []
        for s in seeds:
            stream = _domain_stream(dim, int(e.n_tasks), nc, int(e.samples), float(e.separation), s)
            on.append(self._run(stream, dim, nc, "online", budget, int(e.epochs), float(e.lr), s))
            off.append(self._run(stream, dim, nc, "offline", budget, int(e.epochs), float(e.lr), s))
            dream.append(self._run(stream, dim, nc, "dreaming", budget, int(e.epochs), float(e.lr), s))

        om, ofm, dm = _mean(on), _mean(off), _mean(dream)
        spread = max(_spread(on), _spread(off), _spread(dream))
        offline_beats = (ofm - om) > spread
        dream_matches_stored = abs(dm - ofm) <= spread
        return {
            "bwt_online": round(om, 4),
            "bwt_offline": round(ofm, 4),
            "bwt_dreaming": round(dm, 4),
            "offline_minus_online": round(ofm - om, 4),
            "dreaming_minus_offline": round(dm - ofm, 4),
            "seed_spread": round(spread, 4),
            "replay_budget": budget,
            "seeds": list(seeds),
            "offline_beats_online": bool(offline_beats),
            "dreaming_matches_stored": bool(dream_matches_stored),
            "null_supported": bool(not offline_beats),
        }


class B7(Experiment):
    id = "b7_developmental_curriculum"
    metric = ("final_competence", "ordered_vs_shuffled", "ordered_vs_hardfirst")
    baseline = "the same data presented in shuffled order at matched total samples"
    ablation = "ordered (easy-to-hard) vs shuffled vs hard-first (anti-curriculum); D3 resolvability check"
    null_hypothesis = (
        "ordered curriculum ties shuffled at matched samples (order does not matter on a frozen pooled "
        "substrate; the stream is too uniform or short for staging to bite)"
    )
    tier = "cpu-now"

    @staticmethod
    def _make_blocks(dim, nc, samples, seps, seed):
        blocks = []
        for i, sep in enumerate(seps):
            t = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=samples,
                separation=sep,
                seed=seed * 50 + i,
            )[0]
            blocks.append((t.x, t.y, sep))
        return blocks

    @staticmethod
    def _train_order(blocks, order, dim, nc, epochs, lr, seed) -> float:
        seed_everything(seed)
        head = mlp(dim, nc, 48, depth=1)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for bi in order:
            x, y, _ = blocks[bi]
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(head(x), y).backward()
                opt.step()
        hardest = min(range(len(blocks)), key=lambda i: blocks[i][2])
        hx, hy, _ = blocks[hardest]
        return _eval_acc(head, hx, hy)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        seps = [float(x) for x in e.separations]  # ascending difficulty knob, listed easy-to-hard
        ordered, shuffled, hardfirst = [], [], []
        resolvable_flags = []
        for s in seeds:
            blocks = self._make_blocks(dim, nc, int(e.samples), sorted(seps, reverse=True), s)
            easy_order = list(range(len(blocks)))  # ascending index == easy-to-hard
            g = torch.Generator().manual_seed(s + 17)
            shuf_order = torch.randperm(len(blocks), generator=g).tolist()
            hard_order = list(reversed(easy_order))
            ordered.append(self._train_order(blocks, easy_order, dim, nc, int(e.epochs), float(e.lr), s))
            shuffled.append(self._train_order(blocks, shuf_order, dim, nc, int(e.epochs), float(e.lr), s))
            hardfirst.append(self._train_order(blocks, hard_order, dim, nc, int(e.epochs), float(e.lr), s))
            solo_easy = self._train_order([blocks[0]], [0], dim, nc, int(e.epochs), float(e.lr), s)
            solo_hard = self._train_order([blocks[-1]], [0], dim, nc, int(e.epochs), float(e.lr), s)
            resolvable_flags.append(abs(solo_easy - solo_hard) > 0.05)

        om, sm, hm = _mean(ordered), _mean(shuffled), _mean(hardfirst)
        spread = max(_spread(ordered), _spread(shuffled), _spread(hardfirst))
        resolvable = sum(resolvable_flags) >= max(1, len(seeds) // 2 + len(seeds) % 2)
        ordered_beats = (om - sm > spread) and (om - hm > spread)
        return {
            "final_competence_ordered": round(om, 4),
            "final_competence_shuffled": round(sm, 4),
            "final_competence_hardfirst": round(hm, 4),
            "ordered_minus_shuffled": round(om - sm, 4),
            "ordered_minus_hardfirst": round(om - hm, 4),
            "seed_spread": round(spread, 4),
            "regime_resolvable_d3": bool(resolvable),
            "seeds": list(seeds),
            "ordered_beats_both": bool(ordered_beats and resolvable),
            "null_supported": bool(not (ordered_beats and resolvable)),
        }


class B9(Experiment):
    id = "b9_cerebellar_forward_model"
    metric = ("accuracy_flat", "accuracy_forward_model", "rollout_r2")
    baseline = "a flat predictor at MATCHED compute (no forward-model correction loop)"
    ablation = "flat vs +forward-model-error-as-feature vs +forward-model-with-stop-gradient"
    null_hypothesis = (
        "the forward-model head ties the flat predictor at matched compute (the correction loop is "
        "just extra depth), or rollout R2 is below the D6 floor so there is nothing to correct against"
    )
    tier = "cpu-now"

    @staticmethod
    def _rollout_r2(fwd: nn.Module, x: torch.Tensor, xnext: torch.Tensor) -> float:
        with torch.no_grad():
            pred = fwd(x)
            ss_res = ((pred - xnext) ** 2).sum()
            ss_tot = ((xnext - xnext.mean(0)) ** 2).sum() + 1e-8
            return float(1 - ss_res / ss_tot)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc, hidden = int(e.dim), int(e.n_classes), int(e.hidden)
        epochs, lr = int(e.epochs), float(e.lr)
        flat_acc: list[float] = []
        fm_acc: list[float] = []
        sg_acc: list[float] = []
        r2s: list[float] = []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                forward_dynamics=True,
                seed=s,
            )[0]
            x, y, xn = task.x, task.y, task.xnext
            assert xn is not None  # forward_dynamics=True guarantees the next-latent target
            cut = int(0.7 * len(x))
            xtr, ytr, xntr = x[:cut], y[:cut], xn[:cut]
            xte, yte = x[cut:], y[cut:]

            seed_everything(s)
            flat = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, nc)
            )
            fo = torch.optim.Adam(flat.parameters(), lr=lr)
            for _ in range(epochs):
                fo.zero_grad()
                F.cross_entropy(flat(xtr), ytr).backward()
                fo.step()
            flat_acc.append(_eval_acc(flat, xte, yte))

            for stop_grad, store in ((False, fm_acc), (True, sg_acc)):
                seed_everything(s)
                fwd = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
                clf = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, nc))
                opt = torch.optim.Adam([*fwd.parameters(), *clf.parameters()], lr=lr)
                for _ in range(epochs):
                    opt.zero_grad()
                    pred_next = fwd(xtr)
                    err = pred_next - xntr  # the cerebellar correction signal
                    if stop_grad:
                        err = err.detach()
                    feat = torch.cat([xtr, err], dim=-1)
                    loss = F.cross_entropy(clf(feat), ytr) + F.mse_loss(pred_next, xntr)
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    pn = fwd(xte)
                    err_te = pn - pn.mean(0, keepdim=True)
                    feat_te = torch.cat([x[cut:], err_te], dim=-1)
                    store.append(float((clf(feat_te).argmax(-1) == yte).float().mean()))
                if not stop_grad:
                    r2s.append(self._rollout_r2(fwd, xtr, xntr))

        fa, ma, sa = _mean(flat_acc), _mean(fm_acc), _mean(sg_acc)
        r2 = _mean(r2s)
        spread = max(_spread(flat_acc), _spread(fm_acc), _spread(sg_acc))
        flops_flat = mlp_flops([dim, hidden, hidden, nc])
        flops_fm = mlp_flops([dim, hidden, dim]) + mlp_flops([dim * 2, hidden, nc])
        compute = matched_within(flops_flat, flops_fm)
        d6_floor = float(e.rollout_r2_floor)
        rollout_ok = r2 > d6_floor
        fm_helps = (max(ma, sa) - fa) > spread
        return {
            "accuracy_flat": round(fa, 4),
            "accuracy_forward_model": round(ma, 4),
            "accuracy_forward_model_stopgrad": round(sa, 4),
            "rollout_r2": round(r2, 4),
            "d6_rollout_floor": d6_floor,
            "rollout_above_floor": bool(rollout_ok),
            "fm_minus_flat": round(max(ma, sa) - fa, 4),
            "seed_spread": round(spread, 4),
            "compute_matched": compute,
            "seeds": list(seeds),
            "forward_model_helps": bool(fm_helps and rollout_ok),
            "null_supported": bool((not fm_helps) or (not rollout_ok)),
        }


class B10(Experiment):
    id = "b10_energy_budget"
    metric = ("capability_per_flop", "activation_sparsity", "frontier_beats_width_sweep")
    baseline = "the dense shell and a matched width sweep tracing the same accuracy-FLOP line"
    ablation = "activation-energy (L1) penalty swept vs dense vs a plain width sweep"
    null_hypothesis = (
        "the energy penalty only moves accuracy and FLOPs along the same line a width sweep already "
        "traces (no special frontier), or sparsity collapses accuracy with no usable operating point"
    )
    tier = "cpu-now"

    @staticmethod
    def _train_energy(x, y, xte, yte, dim, nc, hidden, lam, epochs, lr, seed) -> tuple[float, float]:
        seed_everything(seed)
        lin1 = nn.Linear(dim, hidden)
        lin2 = nn.Linear(hidden, nc)
        opt = torch.optim.Adam([*lin1.parameters(), *lin2.parameters()], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            h = F.relu(lin1(x))
            logits = lin2(h)
            loss = F.cross_entropy(logits, y) + lam * h.abs().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            hte = F.relu(lin1(xte))
            acc = float((lin2(hte).argmax(-1) == yte).float().mean())
            sparsity = float((hte <= 1e-6).float().mean())  # fraction of zero (non-firing) units
        return acc, sparsity

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc, hidden = int(e.dim), int(e.n_classes), int(e.hidden)
        lams = [float(x) for x in e.energy_lambdas]
        widths = [int(w) for w in e.sweep_widths]
        epochs, lr = int(e.epochs), float(e.lr)

        energy_cpf, width_cpf, dense_cpf = [], [], []
        best_sparsity = []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            x, y = task.x, task.y
            cut = int(0.7 * len(x))
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            dense_flops = mlp_flops([dim, hidden, nc])

            e_pts = []
            for lam in lams:
                acc, sp = self._train_energy(xtr, ytr, xte, yte, dim, nc, hidden, lam, epochs, lr, s)
                eff = dense_flops * (1 - sp * (mlp_flops([dim, hidden]) / dense_flops))
                e_pts.append((acc, max(1.0, eff)))
            best_sparsity.append(
                max(self._train_energy(xtr, ytr, xte, yte, dim, nc, hidden, max(lams), epochs, lr, s)[1], 0.0)
            )
            energy_cpf.append(max(a / f for a, f in e_pts))

            w_pts = []
            for w in widths:
                acc, _ = self._train_energy(xtr, ytr, xte, yte, dim, nc, w, 0.0, epochs, lr, s)
                w_pts.append((acc, max(1.0, mlp_flops([dim, w, nc]))))
            width_cpf.append(max(a / f for a, f in w_pts))

            acc_dense, _ = self._train_energy(xtr, ytr, xte, yte, dim, nc, hidden, 0.0, epochs, lr, s)
            dense_cpf.append(acc_dense / max(1.0, dense_flops))

        ecpf, wcpf, dcpf = _mean(energy_cpf), _mean(width_cpf), _mean(dense_cpf)
        sp = _mean(best_sparsity)
        spread = max(_spread(energy_cpf), _spread(width_cpf), _spread(dense_cpf))
        beats_width = (ecpf - wcpf) > spread
        beats_dense = (ecpf - dcpf) > spread
        usable = sp < 0.98  # not a total collapse to all-dead units
        return {
            "capability_per_flop_energy": round(ecpf * 1e6, 6),  # scaled to readable units (acc per MFLOP)
            "capability_per_flop_width": round(wcpf * 1e6, 6),
            "capability_per_flop_dense": round(dcpf * 1e6, 6),
            "activation_sparsity": round(sp, 4),
            "energy_minus_width_cpf": round((ecpf - wcpf) * 1e6, 6),
            "seed_spread": round(spread * 1e6, 6),
            "usable_operating_point": bool(usable),
            "seeds": list(seeds),
            "energy_beats_width_sweep": bool(beats_width and beats_dense and usable),
            "null_supported": bool((not (beats_width and beats_dense)) or (not usable)),
        }
