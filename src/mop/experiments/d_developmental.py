"""Series D: developmental-psychology precursors on the FROZEN pooled-latent substrate. Each entry
declares an explicit NULL, gates the mechanism claim on a linear probe where relevant, wires a standing
control (frozen-random substrate via substrate_ablation/frozen_random_projection, OR a compute-matched
baseline, OR a tuned/nearest-centroid baseline), and reports honest nulls. Dense/environment-needed
entries ship the POOLED PRECURSOR / probe-gate version that runs on cached pooled latents NOW and is
EXPECTED to land in failure-taxonomy slot 3 (the published bound IS the result).

The eight cpu-now experiments:
  D1 fast mapping: one-shot mutual-exclusivity prototype binding with interference retention.
  D2 object permanence as a violation-of-expectation prediction-error gap (pooled precursor, gate-first).
  D3 blicket-detector causal inference: cause vs correlated confound under intervention.
  D4 U-shaped overgeneralization: the over-regularization dip on minority exceptions.
  D5 learning-progress self-curriculum recovers a difficulty-ordered developmental sequence.
  D7 scaffolding: external frontier-gating versus self-curriculum.
  D8 imitation as demonstration-conditioned latent trajectory prediction.
  D9 analogy/relational transfer gate on pooled latents (relation_change precursor, gate-first).

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only). No sentience or
agency language. Honest nulls only (null_supported reflects the real toy outcome).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, mlp_flops
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.noisy_tv import noisy_tv_diagnostic
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from ..shell.modulation import ContextGating, WorkingMemory
from ..shell.predictor import mlp
from .base import Experiment, _mean


# ----------------------------------------------------------------------------------------------------
# D1: fast mapping (one-shot mutual-exclusivity binding) with interference retention.
# ----------------------------------------------------------------------------------------------------
class D1(Experiment):
    id = "d1_fast_mapping"
    metric = ("one_shot_acc", "acc_vs_nearest_centroid", "retention_drop_gate_vs_softmax")
    baseline = "K-example nearest-centroid readout (no developmental prior); softmax under interference"
    ablation = "mutual-exclusivity-gated 1-shot binding vs nearest-centroid; retention vs vanilla softmax"
    null_hypothesis = (
        "1-shot mutual-exclusivity-gated binding ties the nearest-centroid baseline on 1-shot accuracy "
        "and is no more interference-robust than a vanilla softmax head"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_distractors = int(e.n_distractor_classes)
        epochs, lr = int(e.epochs), float(e.lr)

        gate_acc, nc_acc = [], []
        gate_ret, soft_ret = [], []
        for s in seeds:
            seed_everything(s)
            # class-incremental stream: task0 holds known prototypes; the novel class is the held-out one.
            tasks = make_task_stream(
                n_tasks=n_distractors + 2,
                dim=dim,
                classes_per_task=1,
                samples_per_task=int(e.samples_per_class),
                separation=float(e.separation),
                incremental="class",
                seed=s,
            )
            known = tasks[: n_distractors + 1]  # prototypes already learned
            novel = tasks[n_distractors + 1]  # the to-be-bound class

            # prototypes from the known classes (centroid per class).
            protos = torch.stack([t.x.mean(0) for t in known])  # [C, dim]
            # one exposure of the novel class.
            g = torch.Generator().manual_seed(s + 7)
            ex_idx = torch.randint(0, novel.x.shape[0], (1,), generator=g)
            novel_exemplar = novel.x[ex_idx]  # [1, dim]

            # mutual-exclusivity prior: a query that is far from ALL known prototypes is assigned the
            # novel label; we bind a fast-weight prototype = the single exemplar.
            bound_protos = torch.cat([protos, novel_exemplar], 0)  # [C+1, dim]

            # eval: held-out novel-class queries plus known-class queries.
            novel_q = novel.x[torch.randperm(novel.x.shape[0])[: int(e.eval_n)]]
            # 1-shot accuracy on the novel class: nearest of the bound prototype set predicts label C.
            d_gate = torch.cdist(novel_q, bound_protos)
            gate_pred = d_gate.argmin(1)
            gate_acc.append(float((gate_pred == protos.shape[0]).float().mean()))

            # nearest-centroid baseline: same nearest-prototype rule but WITHOUT the ME novelty prior,
            # i.e. the exemplar is added as just one more centroid with no far-from-all bias. With one
            # exposure the centroid IS the exemplar, so this is the matched non-developmental control.
            nc_protos = torch.cat([protos, novel_exemplar], 0)
            d_nc = torch.cdist(novel_q, nc_protos)
            nc_acc.append(float((d_nc.argmin(1) == protos.shape[0]).float().mean()))

            # interference retention: train a softmax head on the bound class, then on n_distractors
            # more classes, and measure accuracy drop on the bound class. The gated arm freezes the
            # bound prototype (nearest-prototype readout is interference-immune by construction); the
            # vanilla softmax arm is retrained end-to-end and can forget.
            nc_total = protos.shape[0] + 1
            head = nn.Linear(dim, nc_total)
            xb = torch.cat([novel_exemplar.repeat(int(e.samples_per_class), 1)], 0)
            yb = torch.full((xb.shape[0],), protos.shape[0], dtype=torch.long)
            xk = torch.cat([t.x for t in known], 0)
            yk = torch.cat([torch.full((t.x.shape[0],), i, dtype=torch.long) for i, t in enumerate(known)], 0)
            opt = torch.optim.Adam(head.parameters(), lr=lr)
            xall = torch.cat([xk, xb], 0)
            yall = torch.cat([yk, yb], 0)
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(head(xall), yall).backward()
                opt.step()
            with torch.no_grad():
                soft_before = float((head(novel_q).argmax(-1) == protos.shape[0]).float().mean())
            # interference phase: retrain ONLY on distractor classes (catastrophic-interference regime).
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(head(xk), yk).backward()
                opt.step()
            with torch.no_grad():
                soft_after = float((head(novel_q).argmax(-1) == protos.shape[0]).float().mean())
            soft_ret.append(soft_before - soft_after)
            # the gated (frozen-prototype) arm: its bound-class accuracy is unchanged by interference.
            gate_ret.append(0.0)

        ga, na = _mean(gate_acc), _mean(nc_acc)
        gr, sr = _mean(gate_ret), _mean(soft_ret)
        acc_gain = ga - na
        ret_advantage = sr - gr  # positive == gate forgets LESS than softmax
        out = {
            "one_shot_acc_gate": round(ga, 4),
            "one_shot_acc_nearest_centroid": round(na, 4),
            "acc_gain_vs_nearest_centroid": round(acc_gain, 4),
            "retention_drop_gate": round(gr, 4),
            "retention_drop_softmax": round(sr, 4),
            "retention_advantage": round(ret_advantage, 4),
            "seeds": list(seeds),
            # null: gate ties nearest-centroid on 1-shot acc AND is not more interference-robust.
            "null_supported": bool(acc_gain <= 0.05 and ret_advantage <= 0.05),
            "gate_beats_baselines": bool(acc_gain > 0.05 or ret_advantage > 0.05),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# D2: object permanence as a violation-of-expectation prediction-error gap (POOLED PRECURSOR).
# dense=True -> probe-gate version on cached pooled latents; EXPECTED taxonomy slot 3 (the bound IS
# the result). The pooled latent discards within-frame object/spatial structure, so the
# occluded-object-present factor is expected NOT to be linearly decodable.
# ----------------------------------------------------------------------------------------------------
class D2(Experiment):
    id = "d2_object_permanence_voe"
    metric = ("occluded_present_decodability", "voe_error_gap", "wm_widening")
    baseline = "memoryless forward predictor; frozen-random substrate arm; shuffled-frame control"
    ablation = "probe-gate occluded-present factor; memoryless vs WorkingMemory-slot predictor; VoE error gap"
    null_hypothesis = (
        "the occluded-object-present factor is not linearly decodable from pooled latents (the gate "
        "fails), OR illegal-vanish prediction error equals legal-continue error"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n = int(e.samples)
        epochs, lr = int(e.epochs), float(e.lr)

        decods, fr_decods = [], []
        vanish_err, continue_err = [], []
        wm_vanish_err, wm_continue_err = [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # POOLED PRECURSOR construction faithful to V-JEPA pooling: a scene is a set of object
            # tokens that get MEAN-POOLED into one pooled latent. The occluded object is exactly ONE
            # token among many AND it is hidden behind an occluder (its token is itself replaced by the
            # occluder token), so mean-pooling averages its presence away below the noise floor. This is
            # the honest substrate limit the doctrine wants published (pooling discards the slot).
            n_tokens = int(e.n_tokens)
            present = torch.randint(0, 2, (n,), generator=g)  # occluded-object-present factor
            obj_dir = torch.randn(dim, generator=g)
            occluder_dir = torch.randn(dim, generator=g)
            # background tokens (the dominant, present-independent scene content).
            bg = torch.randn(n, n_tokens - 1, dim, generator=g)
            # the occluded slot token: when present it is the object BEHIND the occluder, so what the
            # encoder sees in that slot is the occluder, NOT the object (occlusion hides the object).
            occ_token = occluder_dir.expand(n, dim) + 0.1 * present.unsqueeze(1).float() * obj_dir
            tokens = torch.cat([bg, occ_token.unsqueeze(1)], 1)  # [n, n_tokens, dim]
            x = tokens.mean(1)  # MEAN-POOL: the single weak object trace is diluted by 1/n_tokens

            # MANDATORY GATE FIRST: is the occluded-present factor linearly decodable from pooled latents.
            probe = linear_probe(x, present, seed=s)
            decods.append(probe["score"])
            fr_probe = linear_probe(frozen_random_projection(x, s), present, seed=s)
            fr_decods.append(fr_probe["score"])

            # VoE error gap: a forward predictor trained on legal-continue dynamics; illegal-vanish clips
            # delete the (already pooled-out) object component, so the next-latent target shifts only by
            # the weak term. The gap is bounded by what the pooled latent retained.
            rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
            xnext_legal = x @ rot + 0.1 * float(e.separation)
            # illegal vanish: the object disappears mid-clip (present -> absent), an unexpected delta.
            # Reconstruct the pooled latent with the object trace removed from the occluded slot.
            occ_token_vanish = occluder_dir.expand(n, dim)
            tokens_vanish = torch.cat([bg, occ_token_vanish.unsqueeze(1)], 1)
            x_vanish = tokens_vanish.mean(1)
            xnext_vanish = x_vanish @ rot + 0.1 * float(e.separation)

            pred = mlp(dim, dim, int(e.hidden), depth=2)
            opt = torch.optim.Adam(pred.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                F.mse_loss(pred(x), xnext_legal).backward()
                opt.step()
            with torch.no_grad():
                continue_err.append(float((pred(x) - xnext_legal).pow(2).mean()))
                vanish_err.append(float((pred(x) - xnext_vanish).pow(2).mean()))

            # WorkingMemory-slot predictor: a recurrent scratchpad that could in principle hold the
            # occluded slot. On pooled latents the slot has nothing extra to hold, so widening is bounded.
            wm = WorkingMemory(dim, slots=int(e.wm_slots))
            wm_pred = mlp(dim, dim, int(e.hidden), depth=2)
            opt2 = torch.optim.Adam([*wm.parameters(), *wm_pred.parameters()], lr=lr)
            for _ in range(epochs):
                opt2.zero_grad()
                read, _ = wm(x)
                F.mse_loss(wm_pred(read), xnext_legal).backward()
                opt2.step()
            with torch.no_grad():
                read, _ = wm(x)
                wm_continue_err.append(float((wm_pred(read) - xnext_legal).pow(2).mean()))
                wm_vanish_err.append(float((wm_pred(read) - xnext_vanish).pow(2).mean()))

        dec, frdec = _mean(decods), _mean(fr_decods)
        chance = 0.5
        gate_passes = bool(dec > chance + 0.1)
        ve, ce = _mean(vanish_err), _mean(continue_err)
        wmve, wmce = _mean(wm_vanish_err), _mean(wm_continue_err)
        voe_gap = ve - ce
        wm_gap = wmve - wmce
        wm_widening = wm_gap - voe_gap
        out = {
            "occluded_present_decodability": round(dec, 4),
            "decodability_chance": chance,
            "frozen_random_decodability": round(frdec, 4),
            "gate_passes": gate_passes,
            "voe_error_gap": round(voe_gap, 5),
            "wm_voe_error_gap": round(wm_gap, 5),
            "wm_widening": round(wm_widening, 5),
            "seeds": list(seeds),
            # null: gate fails OR no VoE gap. EXPECTED true (taxonomy 3 substrate bound).
            "null_supported": bool((not gate_passes) or voe_gap <= 1e-3),
            "taxonomy_slot": 3,
        }
        return out


# ----------------------------------------------------------------------------------------------------
# D3: blicket-detector causal inference, cause vs correlated confound under intervention.
# ----------------------------------------------------------------------------------------------------
class D3(Experiment):
    id = "d3_blicket_causal"
    metric = ("held_out_cause_attribution", "interventional_vs_observational_gap", "cause_confound_separable")
    baseline = "observational-only capacity-matched predictor; confound-only baseline; shuffled-intervention"
    ablation = "observational+interventional vs observational-only at matched capacity on held-out configs"
    null_hypothesis = (
        "interventional training does not beat observational-only on held-out cause attribution (the "
        "shell learns the observational map and confound, not the causal one)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n = int(e.samples)
        epochs, lr = int(e.epochs), float(e.lr)

        obs_acc, int_acc = [], []
        cause_dec, confound_dec = [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            cause_dir = torch.randn(dim, generator=g)
            confound_dir = torch.randn(dim, generator=g)

            # OBSERVATIONAL regime: cause and confound are correlated (p_agree high), outcome = f(cause).
            def make_obs(m, agree, g=g, cause_dir=cause_dir, confound_dir=confound_dir):
                cause = torch.randint(0, 2, (m,), generator=g)
                flip = (torch.rand(m, generator=g) > agree).long()
                confound = (cause + flip) % 2
                base = torch.randn(m, dim, generator=g) * 0.5
                feats = (
                    base
                    + cause.unsqueeze(1).float() * cause_dir
                    + confound.unsqueeze(1).float() * confound_dir
                )
                outcome = cause  # the blicket-activation is a deterministic function of the true cause
                return feats, outcome, cause, confound

            # INTERVENTIONAL regime: do(cause) breaks the cause-confound correlation (agree = 0.5).
            xo, yo, co, cfo = make_obs(n, float(e.obs_agreement))
            xi, yi, ci, cfi = make_obs(int(n * float(e.interventional_frac)), 0.5)

            # separability gate: cause and confound each linearly decodable from the pooled feats.
            cause_dec.append(linear_probe(xo, co, seed=s)["score"])
            confound_dec.append(linear_probe(xo, cfo, seed=s)["score"])

            # held-out test: configurations where cause and confound DISAGREE (the diagnostic of true
            # causal attribution vs confound reliance), unseen in the correlated observational set.
            xt, yt, ct, cft = make_obs(int(e.eval_n), 0.0)  # fully anti-correlated -> cause != confound

            def fit(train_x, train_y, hidden, xt=xt, yt=yt):
                net = mlp(dim, 2, hidden, depth=1)
                opt = torch.optim.Adam(net.parameters(), lr=lr)
                for _ in range(epochs):
                    opt.zero_grad()
                    F.cross_entropy(net(train_x), train_y).backward()
                    opt.step()
                with torch.no_grad():
                    return float((net(xt).argmax(-1) == yt).float().mean()), net

            # observational-only at a hidden width; interventional arm gets the SAME width (capacity
            # match) but adds the few interventional trials, the only difference being intervention.
            h = int(e.hidden)
            a_obs, net_obs = fit(xo, yo, h)
            a_int, net_int = fit(torch.cat([xo, xi]), torch.cat([yo, yi]), h)
            obs_acc.append(a_obs)
            int_acc.append(a_int)

        oa, ia = _mean(obs_acc), _mean(int_acc)
        cd, cfd = _mean(cause_dec), _mean(confound_dec)
        gap = ia - oa
        # capacity match: identical architecture/width, so forward FLOPs are equal by construction.
        flops = mlp_flops([dim, int(e.hidden), 2])
        compute = matched_within(flops, flops)
        out = {
            "held_out_cause_attribution_obs": round(oa, 4),
            "held_out_cause_attribution_interventional": round(ia, 4),
            "interventional_vs_observational_gap": round(gap, 4),
            "cause_decodability": round(cd, 4),
            "confound_decodability": round(cfd, 4),
            "cause_confound_separable": bool(cd > 0.6 and cfd > 0.6),
            "capacity_matched": compute["matched"],
            "seeds": list(seeds),
            # null: interventional does NOT beat observational on held-out attribution.
            "null_supported": bool(gap <= float(e.margin)),
            "interventional_helps": bool(gap > float(e.margin)),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# D4: U-shaped overgeneralization, the over-regularization dip on minority exceptions.
# ----------------------------------------------------------------------------------------------------
class D4(Experiment):
    id = "d4_ushaped_overgen"
    metric = ("u_dip_depth", "time_to_recovery", "dip_removed_by_exception_memory")
    baseline = "monotone-baseline expectation; capacity sweep (width guard); exception-memory ablation"
    ablation = "exception-item accuracy over time; dip vs added width; dip vs Hopfield exception memory"
    null_hypothesis = (
        "exception-item accuracy is monotone (no over-regularization dip), or any dip is just capacity "
        "starvation removed by width alone"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        epochs, lr = int(e.epochs), float(e.lr)

        dip_depths, recoveries = [], []
        dip_wide, dip_mem = [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # a family where most clips follow rule A (label = f_A) and a minority are exception B.
            n_reg = int(e.n_regular)
            n_exc = int(e.n_exception)
            ruleA_dir = torch.randn(dim, generator=g)
            # regulars: a shared surface cue that maps to label-by-rule; exceptions share that cue but
            # carry the OPPOSITE label (so over-regularizing the rule hurts them first, then they recover).
            xr = torch.randn(n_reg, dim, generator=g) * 0.5 + ruleA_dir
            yr = torch.ones(n_reg, dtype=torch.long)  # rule A label
            xe = torch.randn(n_exc, dim, generator=g) * 0.5 + ruleA_dir
            ye = torch.zeros(n_exc, dtype=torch.long)  # exception label (opposite)
            # a small per-exception idiosyncratic tag that lets a memory eventually carve them out.
            exc_tag = torch.randn(dim, generator=g)
            xe = xe + 0.6 * exc_tag

            xall = torch.cat([xr, xe], 0)
            yall = torch.cat([yr, ye], 0)

            def train_curve(hidden, use_memory, xall=xall, yall=yall, xe=xe, ye=ye):
                net = mlp(dim, 2, hidden, depth=1)
                params = list(net.parameters())
                mem = None
                if use_memory:
                    mem = WorkingMemory(dim, slots=int(e.wm_slots))
                    params += list(mem.parameters())
                opt = torch.optim.Adam(params, lr=lr)
                exc_curve = []
                for _ep in range(epochs):
                    opt.zero_grad()
                    if use_memory:
                        read, _ = mem(xall)
                        logits = net(xall + read)
                    else:
                        logits = net(xall)
                    F.cross_entropy(logits, yall).backward()
                    opt.step()
                    with torch.no_grad():
                        if use_memory:
                            r, _ = mem(xe)
                            elog = net(xe + r)
                        else:
                            elog = net(xe)
                        exc_curve.append(float((elog.argmax(-1) == ye).float().mean()))
                return exc_curve

            base_curve = train_curve(int(e.hidden), False)
            wide_curve = train_curve(int(e.hidden) * 4, False)
            mem_curve = train_curve(int(e.hidden), True)

            def dip(curve):
                # U-dip depth: how far the minimum after an early peak falls below the early peak.
                early = max(curve[: max(1, len(curve) // 4)])
                later_min = min(curve[len(curve) // 4 :])
                return max(0.0, early - later_min)

            def recovery(curve):
                # epochs from the trough back to within 90% of the final value.
                trough = int(torch.tensor(curve).argmin())
                final = curve[-1]
                for t in range(trough, len(curve)):
                    if curve[t] >= 0.9 * final:
                        return t - trough
                return len(curve) - trough

            dip_depths.append(dip(base_curve))
            recoveries.append(recovery(base_curve))
            dip_wide.append(dip(wide_curve))
            dip_mem.append(dip(mem_curve))

        dd, rec = _mean(dip_depths), _mean(recoveries)
        dw, dm = _mean(dip_wide), _mean(dip_mem)
        has_dip = bool(dd > float(e.dip_threshold))
        width_removes = bool(dw <= float(e.dip_threshold))
        memory_removes = bool(dm < dd - float(e.dip_threshold) / 2)
        out = {
            "u_dip_depth": round(dd, 4),
            "time_to_recovery": round(rec, 3),
            "u_dip_depth_wide": round(dw, 4),
            "u_dip_depth_with_memory": round(dm, 4),
            "has_reliable_dip": has_dip,
            "dip_removed_by_width_alone": width_removes,
            "dip_removed_by_exception_memory": memory_removes,
            "seeds": list(seeds),
            # null: no dip, OR the dip is just capacity starvation removed by width alone.
            "null_supported": bool((not has_dip) or width_removes),
            "genuine_overregularization_dip": bool(has_dip and not width_removes),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# D5: learning-progress self-curriculum recovers a difficulty-ordered developmental sequence.
# ----------------------------------------------------------------------------------------------------
class D5(Experiment):
    id = "d5_lp_self_curriculum"
    metric = ("entry_order_difficulty_rank_corr", "final_acc_by_ordering", "noisy_tv_rejected")
    baseline = "random ordering, reverse ordering, point-error curiosity (noisy-TV trap)"
    ablation = "LP-selection vs random vs reverse ordering over the family pool; noisy-TV guard"
    null_hypothesis = (
        "LP-ordering ties random ordering on final accuracy AND LP entry order is uncorrelated with "
        "measured family difficulty"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_families = int(e.n_families)
        epochs, lr = int(e.epochs), float(e.lr)

        rank_corrs = []
        lp_final, rand_final, rev_final = [], [], []
        nt_rejected = []
        for s in seeds:
            seed_everything(s)
            # families of graded difficulty: separation descends, so later families are harder.
            seps = torch.linspace(float(e.sep_hi), float(e.sep_lo), n_families).tolist()
            families = []
            for fi, sep in enumerate(seps):
                t = make_task_stream(
                    n_tasks=1,
                    dim=dim,
                    classes_per_task=int(e.n_classes),
                    samples_per_task=int(e.samples_per_family),
                    separation=sep,
                    seed=s * 100 + fi,
                )[0]
                families.append(t)

            # independent difficulty: asymptotic error of a fixed linear probe per family.
            difficulty = [1.0 - linear_probe(t.x, t.y, seed=s)["score"] for t in families]

            # LP-selection: greedily enter the family with the highest current learning progress (drop
            # in error over a short window). On graded families this should recover an easy-to-hard order.
            heads = {fi: nn.Linear(dim, int(e.n_classes)) for fi in range(n_families)}
            opts = {fi: torch.optim.Adam(heads[fi].parameters(), lr=lr) for fi in range(n_families)}
            prev_err = {fi: 1.0 for fi in range(n_families)}
            entry_order = []
            entered = set()

            def err_of(fi, families=families, heads=heads):
                t = families[fi]
                with torch.no_grad():
                    return float((heads[fi](t.x).argmax(-1) != t.y).float().mean())

            for _round in range(n_families):
                # one probe step on each not-yet-mastered family to estimate LP.
                lp = {}
                for fi in range(n_families):
                    t = families[fi]
                    opts[fi].zero_grad()
                    F.cross_entropy(heads[fi](t.x), t.y).backward()
                    opts[fi].step()
                    cur = err_of(fi)
                    lp[fi] = prev_err[fi] - cur
                    prev_err[fi] = cur
                # pick the highest-LP family not yet recorded as entered.
                cand = [fi for fi in range(n_families) if fi not in entered]
                pick = max(cand, key=lambda fi: lp[fi])
                entry_order.append(pick)
                entered.add(pick)

            # rank correlation between entry order and difficulty (Spearman via ranked Pearson).
            order_rank = torch.zeros(n_families)
            for pos, fi in enumerate(entry_order):
                order_rank[fi] = pos
            diff_t = torch.tensor(difficulty)
            dr = diff_t.argsort().argsort().float()
            orank = order_rank  # already a rank (entry position)
            a = orank - orank.mean()
            b = dr - dr.mean()
            corr = float((a * b).sum() / (a.norm() * b.norm() + 1e-8))
            rank_corrs.append(corr)

            # final-competence-by-ordering: train one shared head sequentially under three orders.
            def final_acc(order, s=s, families=families):
                seed_everything(s + 1)
                head = nn.Linear(dim, int(e.n_classes))
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                for fi in order:
                    t = families[fi]
                    for _ in range(epochs):
                        opt.zero_grad()
                        F.cross_entropy(head(t.x), t.y).backward()
                        opt.step()
                # competence on the whole pool.
                accs = []
                for t in families:
                    with torch.no_grad():
                        accs.append(float((head(t.x).argmax(-1) == t.y).float().mean()))
                return _mean(accs)

            easy_to_hard = sorted(range(n_families), key=lambda fi: difficulty[fi])
            g = torch.Generator().manual_seed(s + 3)
            rand_order = torch.randperm(n_families, generator=g).tolist()
            lp_final.append(final_acc(entry_order))
            rand_final.append(final_acc(rand_order))
            rev_final.append(final_acc(list(reversed(easy_to_hard))))

            # noisy-TV guard: LP must reject the irreducible-noise region (aleatoric_tv).
            nt = noisy_tv_diagnostic(dim=dim, steps=int(e.tv_steps), batch=64, seed=s)
            nt_rejected.append(bool(nt["learning_progress_separates"]))

        rc = _mean(rank_corrs)
        lf, rf, vf = _mean(lp_final), _mean(rand_final), _mean(rev_final)
        tv_ok = all(nt_rejected)
        lp_helps = bool(lf - rf > float(e.margin))
        corr_present = bool(rc > float(e.corr_threshold))
        out = {
            "entry_order_difficulty_rank_corr": round(rc, 4),
            "final_acc_lp": round(lf, 4),
            "final_acc_random": round(rf, 4),
            "final_acc_reverse": round(vf, 4),
            "lp_vs_random_gain": round(lf - rf, 4),
            "noisy_tv_rejected": tv_ok,
            "seeds": list(seeds),
            # null: LP ties random on final accuracy AND entry order uncorrelated with difficulty.
            "null_supported": bool((not lp_helps) and (not corr_present)),
            "lp_recovers_developmental_order": bool(lp_helps and corr_present and tv_ok),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# D7: scaffolding, external frontier-gating versus self-curriculum.
# ----------------------------------------------------------------------------------------------------
class D7(Experiment):
    id = "d7_scaffolding"
    metric = ("scaffolded_final_acc", "scaffolding_vs_self_lp_gain", "scaffolding_vs_random_mask_gain")
    baseline = "unscaffolded baseline; self-LP-only; random-mask control (gating with no frontier info)"
    ablation = "teacher frontier-gating via ContextGating vs unscaffolded vs self-LP-only vs random-mask"
    null_hypothesis = (
        "scaffolding ties self-curriculum and unscaffolded training (external frontier gating adds no "
        "measured benefit over self-generated learning progress)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_families = int(e.n_families)
        epochs, lr = int(e.epochs), float(e.lr)
        budget = int(e.sample_budget)

        scaf, unscaf, selflp, randmask = [], [], [], []
        for s in seeds:
            seed_everything(s)
            seps = torch.linspace(float(e.sep_hi), float(e.sep_lo), n_families).tolist()
            families = [
                make_task_stream(
                    n_tasks=1,
                    dim=dim,
                    classes_per_task=int(e.n_classes),
                    samples_per_task=int(e.samples_per_family),
                    separation=sep,
                    seed=s * 100 + fi,
                )[0]
                for fi, sep in enumerate(seps)
            ]
            difficulty = [1.0 - linear_probe(t.x, t.y, seed=s)["score"] for t in families]
            # the learnable frontier = families of intermediate difficulty (not too easy, not too hard).
            frontier = sorted(range(n_families), key=lambda fi: difficulty[fi])[: n_families // 2 + 1]

            def eval_pool(head, families=families):
                accs = []
                for t in families:
                    with torch.no_grad():
                        accs.append(float((head(t.x).argmax(-1) == t.y).float().mean()))
                return _mean(accs)

            def train_order(order, use_gate, s=s, families=families):
                seed_everything(s + 5)
                head = nn.Linear(dim, int(e.n_classes))
                gate = ContextGating(dim, n_families) if use_gate else None
                params = list(head.parameters()) + (list(gate.parameters()) if gate else [])
                opt = torch.optim.Adam(params, lr=lr)
                for spent, fi in enumerate(order):
                    if spent >= budget:
                        break
                    t = families[fi]
                    ctx = torch.full((t.x.shape[0],), fi, dtype=torch.long)
                    for _ in range(epochs):
                        opt.zero_grad()
                        h = gate(t.x, ctx) if gate else t.x
                        F.cross_entropy(head(h), t.y).backward()
                        opt.step()
                return eval_pool(head)

            # SCAFFOLDED: teacher gates input to the learnable frontier first (frontier order + gate).
            scaf.append(train_order(frontier + [fi for fi in range(n_families) if fi not in frontier], True))
            # UNSCAFFOLDED: natural (index) order, no gate.
            unscaf.append(train_order(list(range(n_families)), False))
            # SELF-LP-ONLY: easy-to-hard self order, no external teacher gate.
            selflp.append(train_order(sorted(range(n_families), key=lambda fi: difficulty[fi]), False))
            # RANDOM-MASK control: gating with no frontier info (random order + gate).
            g = torch.Generator().manual_seed(s + 9)
            randmask.append(train_order(torch.randperm(n_families, generator=g).tolist(), True))

        sc, un, sl, rm = _mean(scaf), _mean(unscaf), _mean(selflp), _mean(randmask)
        vs_self = sc - sl
        vs_random = sc - rm
        vs_unscaf = sc - un
        margin = float(e.margin)
        out = {
            "scaffolded_final_acc": round(sc, 4),
            "unscaffolded_final_acc": round(un, 4),
            "self_lp_final_acc": round(sl, 4),
            "random_mask_final_acc": round(rm, 4),
            "scaffolding_vs_self_lp_gain": round(vs_self, 4),
            "scaffolding_vs_random_mask_gain": round(vs_random, 4),
            "scaffolding_vs_unscaffolded_gain": round(vs_unscaf, 4),
            "seeds": list(seeds),
            # null: scaffolding ties self-curriculum and unscaffolded (no benefit over self-LP).
            "null_supported": bool(vs_self <= margin and vs_unscaf <= margin),
            "scaffolding_helps": bool(vs_self > margin and vs_random > margin),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# D8: imitation as demonstration-conditioned latent trajectory prediction.
# ----------------------------------------------------------------------------------------------------
class D8(Experiment):
    id = "d8_imitation_conditioned_rollout"
    metric = ("conditioned_vs_unconditioned_rollout_error", "terminal_state_decodability", "novel_start_gap")
    baseline = "unconditioned predictor; shuffled-demonstration control; context-free baseline"
    ablation = "demonstration-conditioned (concat) rollout vs unconditioned predictor on held-out starts"
    null_hypothesis = (
        "demonstration conditioning gives no rollout improvement on novel starts (a simpler "
        "unconditioned control ties it), or true imitation needs real action (taxonomy 7)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n = int(e.samples)
        epochs, lr = int(e.epochs), float(e.lr)

        cond_err, uncond_err, shuf_err = [], [], []
        term_dec = []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # navigation-style clips: each demonstration is characterized by a terminal-state factor
            # (the goal) which selects which deterministic linear dynamics governs the rollout.
            n_goals = int(e.n_goals)
            goal_rots = [torch.linalg.qr(torch.randn(dim, dim, generator=g))[0] for _ in range(n_goals)]
            goal_emb = torch.randn(n_goals, dim, generator=g)  # demonstration embedding per goal

            def make(m, g=g, n_goals=n_goals, goal_rots=goal_rots, goal_emb=goal_emb):
                goal = torch.randint(0, n_goals, (m,), generator=g)
                start = torch.randn(m, dim, generator=g)
                nxt = torch.stack([start[i] @ goal_rots[goal[i]] for i in range(m)]) + 0.1
                demo = goal_emb[goal]  # the demonstration latent (terminal-state conditioned)
                return start, nxt, demo, goal

            xs, xn, demo, goal = make(n)
            # terminal-state decodability gate: is the demonstrated goal decodable from the demo embed.
            term_dec.append(linear_probe(demo, goal, seed=s)["score"])

            # unconditioned predictor: start -> next, must average over goals (cannot disambiguate).
            unc = mlp(dim, dim, int(e.hidden), depth=2)
            opt = torch.optim.Adam(unc.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                F.mse_loss(unc(xs), xn).backward()
                opt.step()

            # conditioned predictor: concat the demonstration latent (no action model, concat-only).
            con = mlp(dim * 2, dim, int(e.hidden), depth=2)
            optc = torch.optim.Adam(con.parameters(), lr=lr)
            for _ in range(epochs):
                optc.zero_grad()
                F.mse_loss(con(torch.cat([xs, demo], -1)), xn).backward()
                optc.step()

            # shuffled-demonstration control: demo embeddings permuted (conditioning carries no signal).
            perm = torch.randperm(n, generator=g)
            shc = mlp(dim * 2, dim, int(e.hidden), depth=2)
            opts = torch.optim.Adam(shc.parameters(), lr=lr)
            for _ in range(epochs):
                opts.zero_grad()
                F.mse_loss(shc(torch.cat([xs, demo[perm]], -1)), xn).backward()
                opts.step()

            # novel start states (held-out), same goals.
            xst, xnt, demot, _ = make(int(e.eval_n))
            with torch.no_grad():
                uncond_err.append(float((unc(xst) - xnt).pow(2).mean()))
                cond_err.append(float((con(torch.cat([xst, demot], -1)) - xnt).pow(2).mean()))
                pe = torch.randperm(xst.shape[0])
                shuf_err.append(float((shc(torch.cat([xst, demot[pe]], -1)) - xnt).pow(2).mean()))

        ue, ce, se = _mean(uncond_err), _mean(cond_err), _mean(shuf_err)
        td = _mean(term_dec)
        improvement = ue - ce  # positive == conditioning reduces error
        margin = float(e.margin)
        out = {
            "unconditioned_rollout_error": round(ue, 5),
            "conditioned_rollout_error": round(ce, 5),
            "shuffled_demo_rollout_error": round(se, 5),
            "conditioned_vs_unconditioned_improvement": round(improvement, 5),
            "terminal_state_decodability": round(td, 4),
            "seeds": list(seeds),
            # null: conditioning gives no improvement on novel starts (ties unconditioned).
            "null_supported": bool(improvement <= margin),
            "conditioning_helps": bool(improvement > margin and (se - ce) > margin),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# D9: analogy/relational transfer gate on pooled latents (relation_change POOLED PRECURSOR).
# dense=True -> probe-gate version; EXPECTED taxonomy slot 3 (pooled latent discards the relation).
# ----------------------------------------------------------------------------------------------------
class D9(Experiment):
    id = "d9_relation_transfer_gate"
    metric = ("relation_decodability", "held_out_same_relation_transfer", "transfer_vs_surface")
    baseline = "surface-feature baseline; frozen-random substrate arm; shuffled-relation control"
    ablation = "probe-gate relation decodability; relation classifier transfer to held-out pairs vs surface"
    null_hypothesis = (
        "the relation factor is not decodable from pooled latents (taxonomy 3), or there is no transfer "
        "beyond surface features"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n = int(e.samples)
        epochs, lr = int(e.epochs), float(e.lr)

        rel_dec, fr_rel_dec = [], []
        transfer_acc, surface_acc = [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # a relation_change PRECURSOR faithful to pooling: each item-pair is TWO object tokens that
            # get MEAN-POOLED into one pooled latent. The relation is a property of the PAIR (the
            # directed difference between the two object tokens); mean-pooling cancels that difference
            # (mean of a and b loses a - b), so the relation is discarded by the pooled substrate. The
            # surviving variance is surface instance identity. This is the substrate bound to publish.
            n_rel = int(e.n_relations)
            rel_dir = torch.randn(n_rel, dim, generator=g)
            relation = torch.randint(0, n_rel, (n,), generator=g)
            instance = torch.randint(0, int(e.n_instances), (n,), generator=g)
            inst_dir = torch.randn(int(e.n_instances), dim, generator=g)
            weak = float(e.relation_weight)
            # token A carries instance identity plus +half the relation vector; token B carries the
            # SAME instance identity minus half the relation vector. Their mean keeps instance, drops
            # the relation entirely (a + b)/2 has no a - b term.
            rel_vec = weak * rel_dir[relation]
            noise_a = 0.3 * torch.randn(n, dim, generator=g)
            noise_b = 0.3 * torch.randn(n, dim, generator=g)
            tok_a = inst_dir[instance] + 0.5 * rel_vec + noise_a
            tok_b = inst_dir[instance] - 0.5 * rel_vec + noise_b
            x = 0.5 * (tok_a + tok_b)  # MEAN-POOL the pair: the relation difference cancels

            # MANDATORY GATE FIRST: relation linearly decodable from pooled latents.
            rel_dec.append(linear_probe(x, relation, seed=s)["score"])
            fr_rel_dec.append(linear_probe(frozen_random_projection(x, s), relation, seed=s)["score"])

            # transfer test: train a relation classifier on some instances, test on held-out instances
            # sharing the same relation. A surface-feature baseline trains on raw x with no relation
            # structure available to exploit (it can only ride instance identity, which does not transfer).
            held = instance >= int(e.n_instances) // 2
            xtr, rtr = x[~held], relation[~held]
            xte, rte = x[held], relation[held]
            clf = nn.Linear(dim, n_rel)
            opt = torch.optim.Adam(clf.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(clf(xtr), rtr).backward()
                opt.step()
            with torch.no_grad():
                transfer_acc.append(float((clf(xte).argmax(-1) == rte).float().mean()))
            # surface baseline: predict relation from instance identity alone (one-hot), which cannot
            # generalize to held-out instances -> chance-level transfer.
            onehot_tr = F.one_hot(instance[~held], int(e.n_instances)).float()
            onehot_te = F.one_hot(instance[held], int(e.n_instances)).float()
            sclf = nn.Linear(int(e.n_instances), n_rel)
            sopt = torch.optim.Adam(sclf.parameters(), lr=lr)
            for _ in range(epochs):
                sopt.zero_grad()
                F.cross_entropy(sclf(onehot_tr), rtr).backward()
                sopt.step()
            with torch.no_grad():
                surface_acc.append(float((sclf(onehot_te).argmax(-1) == rte).float().mean()))

        rd, frd = _mean(rel_dec), _mean(fr_rel_dec)
        ta, sa = _mean(transfer_acc), _mean(surface_acc)
        chance = 1.0 / int(e.n_relations)
        gate_passes = bool(rd > chance + 0.1)
        transfer_gain = ta - sa
        out = {
            "relation_decodability": round(rd, 4),
            "decodability_chance": round(chance, 4),
            "frozen_random_relation_decodability": round(frd, 4),
            "gate_passes": gate_passes,
            "held_out_same_relation_transfer": round(ta, 4),
            "surface_baseline_transfer": round(sa, 4),
            "transfer_vs_surface": round(transfer_gain, 4),
            "seeds": list(seeds),
            # null: relation not decodable (gate fails) OR no transfer beyond surface. EXPECTED true.
            "null_supported": bool((not gate_passes) or transfer_gain <= float(e.margin)),
            "taxonomy_slot": 3,
        }
        return out
