"""Series A: perception / ecological + animal cognition on the FROZEN pooled-latent substrate.

Eight cpu-now precursor experiments. Each one runs on cached pooled latents NOW (synthetic latent
streams stand in for the real cached families), declares an explicit NULL, and wires a standing control
(frozen-random via substrate_ablation/frozen_random_projection, matched-compute via diagnostics.compute,
or a tuned baseline). Pooled latents discard within-frame object/spatial structure, so several of these
are EXPECTED to land in taxonomy slot 3 (the frozen latent lacks the factor) and the published bound IS
the result. Honest nulls only: null_supported reflects the real toy outcome, never tuned toward positive.

A1 affordance decodability  (tax 3): is action-relevance linearly decodable, and does it need the real
   substrate or would a frozen-random projection do.
A2 viewpoint/motion invariance (tax 3): does event identity transfer across a viewpoint transform beyond
   a pixel-difference baseline and beyond frozen-random.
A3 what-where-when episodic memory (tax 4): does conjunctive KV retrieval beat independent-feature
   retrieve-then-intersect (Clayton-Dickinson binding test).
A4 cognitive-map / latent navigation (tax 10): does a successor representation over latents predict
   held-out shortcut reachability beyond a transition-frequency baseline.
A5 perception-action loop necessity (tax 9): does an action-conditioned next-latent predictor beat an
   action-blind one at matched compute, and collapse under action-shuffle.
A6 object permanence under occlusion (tax 3): does the mid-occlusion latent carry a decodable trace of
   the hidden object beyond a frame-only baseline and beyond frozen-random.
A7 symbolic-communication channel (tax 3): can a tiny VQ code carry decodable navigation content beyond
   a random code at a bottleneck (waggle-dance analog).
A8 affordance-driven vs stimulus-driven curiosity (tax 8): does affordance-weighted selection beat
   learning-progress-only without re-chasing the noisy-TV, and beyond a static affordance-prior-only arm.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, mlp_flops
from ..diagnostics.geometry import linear_cka, neighborhood_overlap
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.noisy_tv import noisy_tv_diagnostic
from ..diagnostics.substrate_ablation import frozen_random_projection, substrate_ablation
from ..seeding import seed_everything
from ..shell.buffer import ReplayBuffer
from ..shell.predictor import mlp
from .base import Experiment


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


# ---------------------------------------------------------------------------------------------------
# A1: affordance decodability probe (action-relevance from a pooled latent)
# ---------------------------------------------------------------------------------------------------
class A1(Experiment):
    id = "a1_affordance_decode"
    metric = ("probe_acc_above_shuffle_floor", "delta_vs_frozen_random", "linear_vs_mlp_gap")
    baseline = "shuffle-label chance floor and a frozen-random-substrate projection at matched probe"
    ablation = "real encoder vs frozen-random; linear probe vs small-MLP probe per affordance contrast"
    null_hypothesis = (
        "action-relevance labels do not exceed the shuffle-label chance floor, OR the real encoder ties "
        "a frozen-random projection (linear decodability is projection-invariant), so any affordance "
        "mechanism would read capacity not a substrate affordance"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        # three action-relevance contrasts standing in for containment / occlusion_reveal / navigation
        contrasts = ["into_container_vs_contact", "reachable_vs_occluded", "passage_vs_blocked"]
        rows: dict[str, dict] = {}
        for ci, cname in enumerate(contrasts):
            real_acc, fr_acc, shuf_acc, mlp_acc = [], [], [], []
            for s in seeds:
                seed_everything(s)
                # each contrast is a 2-class action-relevance relabel of a control family
                task = make_task_stream(
                    n_tasks=1,
                    dim=dim,
                    classes_per_task=2,
                    samples_per_task=int(e.samples),
                    separation=float(e.separation),
                    seed=s + 17 * ci,
                )[0]
                x, y = task.x, task.y
                abl = substrate_ablation(x, y, seed=s)
                real_acc.append(abl["real"])
                fr_acc.append(abl["frozen_random"])
                shuf_acc.append(abl["shuffled"])
                # small-MLP probe to read the linear-vs-nonlinear capacity gap
                mlp_acc.append(_mlp_probe(x, y, hidden=int(e.mlp_hidden), epochs=int(e.epochs), seed=s))
            ra, fa = _mean(real_acc), _mean(fr_acc)
            sa, ma = _mean(shuf_acc), _mean(mlp_acc)
            rows[cname] = {
                "real_acc": round(ra, 4),
                "frozen_random_acc": round(fa, 4),
                "shuffle_floor_acc": round(sa, 4),
                "mlp_acc": round(ma, 4),
                "above_shuffle_floor": bool(ra - sa > 0.05),
                "delta_vs_frozen_random": round(ra - fa, 4),
                "linear_vs_mlp_gap": round(ma - ra, 4),
                "needs_real_substrate": bool(ra - fa > 0.05 and ra - sa > 0.05),
            }
        any_above = any(r["above_shuffle_floor"] for r in rows.values())
        any_needs_real = any(r["needs_real_substrate"] for r in rows.values())
        return {
            "contrasts": rows,
            "seeds": list(seeds),
            "any_contrast_above_floor": bool(any_above),
            "any_contrast_needs_real_substrate": bool(any_needs_real),
            # null: nothing clears the shuffle floor while also beating frozen-random
            "null_supported": bool(not any_needs_real),
        }


# ---------------------------------------------------------------------------------------------------
# A2: viewpoint and motion invariance battery (world stability under transformation)
# ---------------------------------------------------------------------------------------------------
class A2(Experiment):
    id = "a2_viewpoint_invariance"
    metric = ("cross_viewpoint_transfer", "cka_across_viewpoints", "transfer_gap_vs_frozen_random")
    baseline = "pixel-difference baseline and a frozen-random-substrate arm"
    ablation = "train probe on viewpoint A, test on held-out viewpoint B; real vs frozen-random"
    null_hypothesis = (
        "event identity does not transfer across viewpoint conditions beyond a pixel-difference "
        "baseline, OR the real encoder invariance ties frozen-random (pooling plus random already "
        "smears viewpoint, so invariance is trivial averaging not structure)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, k = int(e.dim), int(e.n_events)
        real_t, fr_t, pix_t, ckas = [], [], [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # k event identities with a shared latent direction; a viewpoint transform is a fixed
            # rotation plus a small viewpoint-specific shift (the camera move). The encoder is modeled
            # as a near-invariant view: identity survives the transform up to noise.
            centers = torch.randn(k, dim, generator=g)
            rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
            shift = torch.randn(dim, generator=g) * float(e.view_shift)
            n = int(e.samples)
            y = torch.randint(0, k, (n,), generator=g)
            xa = centers[y] * float(e.separation) + 0.5 * torch.randn(n, dim, generator=g)
            # viewpoint B: rotate the SAME event content and add a viewpoint shift + fresh sensor noise
            xb = xa @ rot + shift + 0.5 * torch.randn(n, dim, generator=g)
            # real arm: train identity probe on A, test transfer to B
            real_t.append(_transfer_probe(xa, y, xb, y, epochs=int(e.epochs), seed=s))
            # frozen-random arm: same probe-transfer through a fixed random projection of both views
            far = frozen_random_projection(xa, seed=s)
            fbr = frozen_random_projection(xb, seed=s)
            fr_t.append(_transfer_probe(far, y, fbr, y, epochs=int(e.epochs), seed=s))
            # pixel-difference baseline: nearest-event by raw L2 in the (pre-encoder) input proxy, which
            # is the un-rotated content; transfer collapses because B is rotated away from A prototypes.
            proto = torch.stack([xa[y == c].mean(0) for c in range(k)])
            pix_pred = torch.cdist(xb, proto).argmin(1)
            pix_t.append(float((pix_pred == y).float().mean()))
            ckas.append(linear_cka(xa, xb))
        rt, ft, pt, ck = _mean(real_t), _mean(fr_t), _mean(pix_t), _mean(ckas)
        beats_pixel = rt - pt > 0.05
        beats_fr = rt - ft > 0.05
        return {
            "cross_viewpoint_transfer_real": round(rt, 4),
            "cross_viewpoint_transfer_frozen_random": round(ft, 4),
            "pixel_baseline_transfer": round(pt, 4),
            "cka_across_viewpoints": round(ck, 4),
            "transfer_gap_vs_pixel": round(rt - pt, 4),
            "transfer_gap_vs_frozen_random": round(rt - ft, 4),
            "seeds": list(seeds),
            "invariance_beats_pixel": bool(beats_pixel),
            "invariance_beats_frozen_random": bool(beats_fr),
            # null: no transfer over pixel baseline OR ties frozen-random (trivial smearing)
            "null_supported": bool(not (beats_pixel and beats_fr)),
        }


# ---------------------------------------------------------------------------------------------------
# A3: episodic-like what-where-when memory (Clayton-Dickinson test)
# ---------------------------------------------------------------------------------------------------
class A3(Experiment):
    id = "a3_what_where_when"
    metric = ("integration_gap", "recall_at_k", "where_when_decodable")
    baseline = "independent-feature retrieve-then-intersect and a recency-only baseline"
    ablation = "conjunctive KV retrieval vs intersect-independent-features vs recency-only"
    null_hypothesis = (
        "integrated what-where-when retrieval does not beat independent single-feature retrieval "
        "intersected post hoc (no binding, just three lookups), OR the where and when tags are not "
        "decodable from the stored latent at all (taxonomy 3)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_what, n_where = int(e.n_what), int(e.n_where)
        n = int(e.samples)
        conj_acc, indep_acc, recency_acc = [], [], []
        where_dec, when_dec = [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # a stored latent is the concatenation-in-superposition of what/where/when subspaces; the
            # binding question is whether one record carries all three jointly (it does here by
            # construction, so the honest test is whether conjunctive retrieval USES the binding).
            what_c = torch.randn(n_what, dim, generator=g)
            where_c = torch.randn(n_where, dim, generator=g)
            wy = torch.randint(0, n_what, (n,), generator=g)
            ly = torch.randint(0, n_where, (n,), generator=g)
            ty = torch.arange(n)  # when-index = store order
            when_vec = torch.linspace(-1.0, 1.0, n).unsqueeze(1) * torch.randn(1, dim, generator=g)
            x = (
                what_c[wy] * float(e.separation)
                + where_c[ly] * float(e.separation)
                + when_vec
                + 0.3 * torch.randn(n, dim, generator=g)
            )
            buf = ReplayBuffer(capacity=n, dim=dim, prioritized=False, seed=s)
            buf.add(x, wy)
            # decodability gate (taxonomy-3 check): are where and when readable from the stored latent
            where_dec.append(linear_probe(x, ly, seed=s)["score"])
            when_dec.append(linear_probe(x, (ty * n_where // n).long(), seed=s)["score"])
            # build conjunctive queries: pick a (where=L, when in window T) and ask which what was there
            kq = int(e.n_queries)
            qg = torch.Generator().manual_seed(s + 99)
            c_hits, i_hits, r_hits = 0, 0, 0
            for _ in range(kq):
                li = int(torch.randint(0, n_where, (1,), generator=qg))
                ti = int(torch.randint(0, n, (1,), generator=qg))
                tw = int(e.when_window)
                # ground truth: the stored record at index ti IF it has where==li, else nearest in window
                cand = [j for j in range(max(0, ti - tw), min(n, ti + tw + 1)) if int(ly[j]) == li]
                if not cand:
                    continue
                gt_j = min(cand, key=lambda j: abs(j - ti))
                gt_what = int(wy[gt_j])
                # conjunctive retrieval: a query latent built from BOTH where and when cues, one lookup
                qvec = where_c[li] * float(e.separation) + when_vec[ti]
                r = buf.retrieve(qvec.unsqueeze(0), k=1)
                if int(r["y"][0, 0]) == gt_what:
                    c_hits += 1
                # independent retrieve-then-intersect: top-m by where, top-m by when, intersect
                m = int(e.intersect_m)
                rw = buf.retrieve((where_c[li] * float(e.separation)).unsqueeze(0), k=m)["idx"][0]
                rt = buf.retrieve(when_vec[ti].unsqueeze(0), k=m)["idx"][0]
                inter = [int(j) for j in rw.tolist() if j in set(rt.tolist())]
                pred_i = int(wy[inter[0]]) if inter else int(wy[int(rw[0])])
                if pred_i == gt_what:
                    i_hits += 1
                # recency-only: just return the most recent record in the when-window
                pred_r = int(wy[gt_j])  # recency baseline ignores where, takes nearest-in-time overall
                near_t = min(range(max(0, ti - tw), min(n, ti + tw + 1)), key=lambda j: abs(j - ti))
                pred_r = int(wy[near_t])
                if pred_r == gt_what:
                    r_hits += 1
            denom = max(1, kq)
            conj_acc.append(c_hits / denom)
            indep_acc.append(i_hits / denom)
            recency_acc.append(r_hits / denom)
        ca, ia, ra = _mean(conj_acc), _mean(indep_acc), _mean(recency_acc)
        wd, td = _mean(where_dec), _mean(when_dec)
        decodable = wd > 1.0 / n_where + 0.1 and td > 1.0 / max(2, n_where) + 0.05
        gap = ca - max(ia, ra)
        return {
            "conjunctive_acc": round(ca, 4),
            "independent_intersect_acc": round(ia, 4),
            "recency_only_acc": round(ra, 4),
            "integration_gap": round(gap, 4),
            "where_decodable_acc": round(wd, 4),
            "when_decodable_acc": round(td, 4),
            "where_when_decodable": bool(decodable),
            "seeds": list(seeds),
            "binding_beats_independent": bool(gap > 0.05 and decodable),
            # null: conjunctive ties independent/recency OR tags not decodable
            "null_supported": bool(not (gap > 0.05 and decodable)),
        }


# ---------------------------------------------------------------------------------------------------
# A4: cognitive-map / latent navigation structure (place-and-grid analog, no environment)
# ---------------------------------------------------------------------------------------------------
class A4(Experiment):
    id = "a4_cognitive_map"
    metric = ("position_decode_r2", "neighborhood_arena_corr", "shortcut_acc_vs_transition_freq")
    baseline = "transition-frequency baseline and a frozen-random-substrate arm"
    ablation = "successor representation over latents vs transition-frequency; real vs frozen-random"
    null_hypothesis = (
        "latents support only observed-transition prediction; held-out shortcut reachability ties a "
        "transition-frequency baseline (no map, just a chain), OR position is not decodable above floor "
        "(taxonomy 3)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        grid = int(e.grid)  # grid x grid arena
        r2s, corrs, sr_acc, tf_acc = [], [], [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # each arena cell maps to a latent via a smooth random embedding of its (row,col), so latent
            # neighborhood SHOULD track arena distance if the embedding is metric-preserving.
            coords = torch.stack([torch.tensor([r, c]) for r in range(grid) for c in range(grid)]).float()
            n_cells = coords.shape[0]
            basis = torch.randn(2, dim, generator=g)
            cell_latent = coords @ basis + 0.2 * torch.randn(n_cells, dim, generator=g)
            # position decodability (place-analog): regress (row,col) from the latent
            r2 = linear_probe(cell_latent, coords, classification=False, seed=s)["score"]
            r2s.append(r2)
            # neighborhood-vs-arena-distance correlation via neighborhood_overlap with the true coords
            corrs.append(neighborhood_overlap(cell_latent, coords, k=int(e.knn)))
            # successor representation over observed transitions on a grid random walk
            adj = _grid_adjacency(grid)
            T = _walk_transition_counts(adj, steps=int(e.walk_steps), seed=s)
            P = T / T.sum(1, keepdim=True).clamp(min=1e-8)
            gamma = float(e.gamma)
            SR = torch.linalg.inv(torch.eye(n_cells) - gamma * P)
            # held-out shortcut pairs: start-goal not adjacent in training; reachable = SR mass over a
            # tuned threshold. transition-frequency baseline uses raw one-step counts only.
            pairs, labels = _shortcut_pairs(adj, n_cells, n_pairs=int(e.n_pairs), seed=s)
            sr_pred = torch.tensor([SR[i, j] for (i, j) in pairs])
            tf_pred = torch.tensor([T[i, j] for (i, j) in pairs])
            sr_acc.append(_threshold_acc(sr_pred, labels))
            tf_acc.append(_threshold_acc(tf_pred, labels))
        r2m, cm = _mean(r2s), _mean(corrs)
        sm, tm = _mean(sr_acc), _mean(tf_acc)
        pos_decodable = r2m > 0.1
        sr_beats_tf = sm - tm > 0.05
        return {
            "position_decode_r2": round(r2m, 4),
            "neighborhood_arena_corr": round(cm, 4),
            "shortcut_acc_sr": round(sm, 4),
            "shortcut_acc_transition_freq": round(tm, 4),
            "sr_minus_transition_freq": round(sm - tm, 4),
            "position_decodable": bool(pos_decodable),
            "seeds": list(seeds),
            "map_supports_shortcut": bool(pos_decodable and sr_beats_tf),
            # null: shortcut ties transition-freq OR position not decodable
            "null_supported": bool(not (pos_decodable and sr_beats_tf)),
        }


# ---------------------------------------------------------------------------------------------------
# A5: perception-action loop necessity test (matched-compute action conditioning)
# ---------------------------------------------------------------------------------------------------
class A5(Experiment):
    id = "a5_action_loop"
    metric = ("action_information_gap", "shuffle_collapse", "compute_matched")
    baseline = "action-blind next-latent predictor at matched compute and an action-shuffle control"
    ablation = "action-conditioned vs action-blind vs action-shuffled at matched FLOPs"
    null_hypothesis = (
        "action-conditioning does not reduce next-latent error beyond the action-blind predictor at "
        "matched compute, OR the gain survives action-shuffling (so it was added capacity, taxonomy 9, "
        "not action information)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, adim = int(e.dim), int(e.action_dim)
        hidden, epochs = int(e.hidden), int(e.epochs)
        cond_err, blind_err, shuf_err = [], [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            n = int(e.samples)
            x = torch.randn(n, dim, generator=g)
            a = torch.randn(n, adim, generator=g)
            # forward dynamics where the next latent genuinely depends on the action: distinct rotation
            # selected by a discrete action channel, so action carries real predictive information.
            rots = [torch.linalg.qr(torch.randn(dim, dim, generator=g))[0] for _ in range(adim)]
            sel = a.argmax(1)
            xnext = torch.stack([x[i] @ rots[int(sel[i])] for i in range(n)]) + 0.1
            cut = int(n * 0.7)
            # action-conditioned: input dim+adim. blind: input dim (must marginalize over actions).
            # shuffled: action token permuted across samples (same FLOPs, action made uninformative).
            # blind gets a wider hidden so FLOPs are matched to the conditioned arm.
            h_blind = _hidden_for_matched(dim, adim, hidden)
            cm = matched_within(mlp_flops([dim + adim, hidden, dim]), mlp_flops([dim, h_blind, dim]))
            cond_err.append(
                _fit_dynamics(torch.cat([x, a], 1), xnext, cut, dim + adim, hidden, dim, epochs, s)
            )
            blind_err.append(_fit_dynamics(x, xnext, cut, dim, h_blind, dim, epochs, s))
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(s + 5))
            shuf_err.append(
                _fit_dynamics(torch.cat([x, a[perm]], 1), xnext, cut, dim + adim, hidden, dim, epochs, s)
            )
        ce, be, se = _mean(cond_err), _mean(blind_err), _mean(shuf_err)
        info_gap = be - ce  # positive if conditioning helps
        # honest action test: the conditioned gain must COLLAPSE under shuffle (shuf ~ blind, not cond)
        shuffle_collapse = (se - ce) > 0.5 * info_gap if info_gap > 0 else False
        conditioning_helps = info_gap > float(e.margin)
        return {
            "conditioned_error": round(ce, 4),
            "blind_error": round(be, 4),
            "shuffled_error": round(se, 4),
            "action_information_gap": round(info_gap, 4),
            "shuffle_minus_conditioned": round(se - ce, 4),
            "shuffle_collapse": bool(shuffle_collapse),
            "compute_matched": bool(cm["matched"]),
            "compute_ratio": cm["ratio"],
            "seeds": list(seeds),
            "action_carries_information": bool(conditioning_helps and shuffle_collapse and cm["matched"]),
            # null: no gain at matched compute OR gain survives shuffle (it was capacity)
            "null_supported": bool(not (conditioning_helps and shuffle_collapse and cm["matched"])),
        }


# ---------------------------------------------------------------------------------------------------
# A6: object permanence as latent persistence under occlusion
# ---------------------------------------------------------------------------------------------------
class A6(Experiment):
    id = "a6_object_permanence"
    metric = ("mid_occlusion_separation", "persistence_vs_frame_only", "real_vs_frozen_random")
    baseline = "single-frame-only probe and a frozen-random-substrate arm"
    ablation = "temporal-window shell carrying state vs frame-only probe; real vs frozen-random"
    null_hypothesis = (
        "mid-occlusion the latent carries no decodable trace of the hidden object beyond the current "
        "(empty) frame, so permanence and vanish are indistinguishable until reappearance (taxonomy 3), "
        "OR a single-frame baseline already separates them"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        temporal_acc, frame_acc, fr_acc = [], [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            n = int(e.samples)
            # two classes: permanence-holds (object reappears) vs vanishes (never returns). Pre-occlusion
            # frames carry the object identity; the occluded (current) frame is near-empty for both, so
            # only a temporal shell carrying pre-occlusion state can separate the classes mid-occlusion.
            y = torch.randint(0, 2, (n,), generator=g)
            obj_dir = torch.randn(dim, generator=g)
            pre = y.float().unsqueeze(1) * 0.0 + obj_dir  # object present pre-occlusion for both
            pre = obj_dir.unsqueeze(0) + 0.3 * torch.randn(n, dim, generator=g)
            # mid-occlusion CURRENT frame: empty scene plus tiny class-correlated residual leak. With a
            # weak leak the frame-only probe is near chance; the temporal shell adds the pre frame state.
            leak = float(e.permanence_leak)
            mid_frame = leak * (2 * y.float() - 1).unsqueeze(1) * obj_dir + 0.3 * torch.randn(
                n, dim, generator=g
            )
            # temporal shell: a window pooling pre + mid (carries persistence). frame-only: mid only.
            window = torch.cat([pre, mid_frame], dim=1)
            temporal_acc.append(linear_probe(window, y, seed=s)["score"])
            frame_acc.append(linear_probe(mid_frame, y, seed=s)["score"])
            fr_acc.append(linear_probe(frozen_random_projection(window, seed=s), y, seed=s)["score"])
        ta, fa, fra = _mean(temporal_acc), _mean(frame_acc), _mean(fr_acc)
        sep_above_frame = ta - fa > 0.05
        sep_above_fr = ta - fra > 0.05
        return {
            "mid_occlusion_temporal_acc": round(ta, 4),
            "frame_only_acc": round(fa, 4),
            "frozen_random_acc": round(fra, 4),
            "persistence_vs_frame_only": round(ta - fa, 4),
            "real_vs_frozen_random": round(ta - fra, 4),
            "seeds": list(seeds),
            "permanence_separable": bool(sep_above_frame and sep_above_fr),
            # null: no separation over frame-only OR over frozen-random (no maintained occluded state)
            "null_supported": bool(not (sep_above_frame and sep_above_fr)),
        }


# ---------------------------------------------------------------------------------------------------
# A7: symbolic-communication channel test (waggle-dance analog)
# ---------------------------------------------------------------------------------------------------
class A7(Experiment):
    id = "a7_comm_channel"
    metric = ("receiver_acc_vs_random_code", "bits_at_knee", "shuffle_collapse")
    baseline = "random-codebook of the same size and a raw-latent-passthrough ceiling"
    ablation = "learned VQ code vs random code across a codebook-size sweep; sender-receiver shuffle"
    null_hypothesis = (
        "the code transmits no more target information than a random code of the same size (no "
        "referential structure), OR target direction/distance is not decodable from the pooled latent "
        "at all (taxonomy 3), so there is nothing to encode"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_target = int(e.n_targets)  # target direction/distance bins the receiver must recover
        sweep = list(e.codebook_sizes)
        per_k: dict[int, dict] = {}
        ceiling_all = []
        for k in sweep:
            learned, random_, shuf = [], [], []
            for s in seeds:
                seed_everything(s)
                g = torch.Generator().manual_seed(s + k)
                n = int(e.samples)
                tcenters = torch.randn(n_target, dim, generator=g)
                ty = torch.randint(0, n_target, (n,), generator=g)
                x = tcenters[ty] * float(e.separation) + 0.5 * torch.randn(n, dim, generator=g)
                # sender emits a k-symbol code by VQ (k-means) over the latent; receiver decodes target
                codes = _kmeans_codes(x, k, int(e.kmeans_iters), s)
                learned.append(_code_receiver_acc(codes, ty, k, seed=s))
                rg = torch.Generator().manual_seed(s + 1234)
                rand_codes = torch.randint(0, k, (n,), generator=rg)
                random_.append(_code_receiver_acc(rand_codes, ty, k, seed=s))
                # sender-receiver shuffle: code from a DIFFERENT clip (codes permuted across samples)
                pg = torch.Generator().manual_seed(s + 77)
                perm = torch.randperm(n, generator=pg)
                shuf.append(_code_receiver_acc(codes[perm], ty, k, seed=s))
                ceiling_all.append(linear_probe(x, ty, seed=s)["score"])  # raw-latent passthrough
            la, raa, sa = _mean(learned), _mean(random_), _mean(shuf)
            per_k[k] = {
                "learned_acc": round(la, 4),
                "random_code_acc": round(raa, 4),
                "shuffle_acc": round(sa, 4),
                "beats_random": bool(la - raa > 0.05),
                "shuffle_collapse": bool(la - sa > 0.05),
                "bits": round(float(torch.log2(torch.tensor(float(k)))), 3),
            }
        ceiling = _mean(ceiling_all)
        referential_ks = [k for k, r in per_k.items() if r["beats_random"] and r["shuffle_collapse"]]
        bits_at_knee = per_k[min(referential_ks)]["bits"] if referential_ks else None
        return {
            "per_codebook_size": per_k,
            "raw_latent_ceiling": round(ceiling, 4),
            "referential_codebook_sizes": referential_ks,
            "bits_at_knee": bits_at_knee,
            "seeds": list(seeds),
            "channel_is_referential": bool(len(referential_ks) > 0),
            # null: no code beats a random code (with shuffle collapse) at any bottleneck
            "null_supported": bool(len(referential_ks) == 0),
        }


# ---------------------------------------------------------------------------------------------------
# A8: affordance-driven vs stimulus-driven curiosity
# ---------------------------------------------------------------------------------------------------
class A8(Experiment):
    id = "a8_affordance_curiosity"
    metric = ("affordance_coverage_gain", "noisy_tv_time_share", "vs_prior_only")
    baseline = "learning-progress-only, affordance-prior-only (LP off), uniform; plus noisy-TV guards"
    ablation = "affordance-times-LP selection vs LP-only vs prior-only vs uniform"
    null_hypothesis = (
        "affordance weighting ties pure learning-progress on coverage, OR it helps only because of the "
        "static prior (the prior-only arm matches it), OR it re-chases the noisy-TV (the affordance "
        "probe fires on noise); any of these is taxonomy 8 or 2"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        aff_cov, lp_cov, prior_cov, unif_cov = [], [], [], []
        aff_tv, lp_tv = [], []
        for s in seeds:
            # noisy-TV diagnostic gives the per-region learning-progress and raw error (the curiosity
            # signals). The affordance relevance is a probe score on an action-relevance contrast.
            nt = noisy_tv_diagnostic(
                dim=dim,
                ensemble_size=int(e.ensemble_size),
                steps=int(e.steps),
                batch=int(e.batch),
                noise_scale=float(e.noise_scale),
                seed=s,
            )
            lp_l = nt["learning_progress"]["learnable"]
            lp_n = nt["learning_progress"]["noise"]
            # affordance relevance: how action-decodable each region is. The learnable region carries a
            # decodable action-relevance contrast; pure noise does not (probe near chance).
            aff_l, aff_n = _affordance_relevance(dim, float(e.separation), s)
            # selection scores per arm on the two regions
            arms = {
                "affordance_lp": (aff_l * lp_l, aff_n * lp_n),
                "lp_only": (lp_l, lp_n),
                "prior_only": (aff_l, aff_n),  # LP held constant -> just the static affordance prior
                "uniform": (1.0, 1.0),
            }

            # coverage of the action-relevant (learnable) region = its selection share
            def share(pair):
                a, b = max(pair[0], 0.0), max(pair[1], 0.0)
                return a / (a + b + 1e-8)

            aff_cov.append(share(arms["affordance_lp"]))
            lp_cov.append(share(arms["lp_only"]))
            prior_cov.append(share(arms["prior_only"]))
            unif_cov.append(share(arms["uniform"]))
            # noisy-TV time-share = selection share spent on the NOISE region (must stay near zero)
            aff_tv.append(1.0 - share(arms["affordance_lp"]))
            lp_tv.append(1.0 - share(arms["lp_only"]))
        am, lm, pm, um = _mean(aff_cov), _mean(lp_cov), _mean(prior_cov), _mean(unif_cov)
        atv, ltv = _mean(aff_tv), _mean(lp_tv)
        beats_lp = am - lm > 0.02
        beats_prior = am - pm > 0.02
        tv_low = atv < float(e.tv_threshold)
        return {
            "affordance_coverage": round(am, 4),
            "lp_only_coverage": round(lm, 4),
            "prior_only_coverage": round(pm, 4),
            "uniform_coverage": round(um, 4),
            "affordance_coverage_gain_vs_lp": round(am - lm, 4),
            "affordance_coverage_gain_vs_prior": round(am - pm, 4),
            "affordance_noisy_tv_time_share": round(atv, 4),
            "lp_noisy_tv_time_share": round(ltv, 4),
            "seeds": list(seeds),
            "affordance_helps": bool(beats_lp and beats_prior and tv_low),
            # null: ties LP OR matches prior-only OR re-chases noisy-TV
            "null_supported": bool(not (beats_lp and beats_prior and tv_low)),
        }


# ---------------------------------------------------------------------------------------------------
# inline helpers (the established pattern: mechanisms live in the experiment module)
# ---------------------------------------------------------------------------------------------------
def _mlp_probe(x: torch.Tensor, y: torch.Tensor, hidden: int, epochs: int, seed: int) -> float:
    """A small-MLP probe: reads the nonlinear capacity gap over the linear probe."""
    seed_everything(seed)
    n = x.shape[0]
    perm = torch.randperm(n)
    x, y = x[perm], y[perm]
    cut = int(n * 0.7)
    xtr, xte, ytr, yte = x[:cut], x[cut:], y[:cut], y[cut:]
    nc = int(y.max()) + 1
    net = mlp(x.shape[1], nc, hidden, depth=1)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(net(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((net(xte).argmax(-1) == yte).float().mean())


def _transfer_probe(
    xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor, yte: torch.Tensor, epochs: int, seed: int
) -> float:
    """Train a linear identity probe on one viewpoint, test transfer to a held-out viewpoint."""
    seed_everything(seed)
    nc = int(max(int(ytr.max()), int(yte.max()))) + 1
    head = nn.Linear(xtr.shape[1], nc)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((head(xte).argmax(-1) == yte).float().mean())


def _grid_adjacency(grid: int) -> torch.Tensor:
    n = grid * grid
    adj = torch.zeros(n, n)
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid and 0 <= nc < grid:
                    adj[i, nr * grid + nc] = 1.0
    return adj


def _walk_transition_counts(adj: torch.Tensor, steps: int, seed: int) -> torch.Tensor:
    n = adj.shape[0]
    g = torch.Generator().manual_seed(seed)
    T = torch.zeros(n, n)
    cur = int(torch.randint(0, n, (1,), generator=g))
    for _ in range(steps):
        nbrs = torch.nonzero(adj[cur]).flatten()
        nxt = int(nbrs[int(torch.randint(0, len(nbrs), (1,), generator=g))])
        T[cur, nxt] += 1.0
        cur = nxt
    return T + 1e-3  # smoothing so unseen one-step transitions are nonzero but tiny


def _shortcut_pairs(
    adj: torch.Tensor, n: int, n_pairs: int, seed: int
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """Held-out start-goal pairs NOT one-step adjacent. label=1 if reachable within a few hops."""
    g = torch.Generator().manual_seed(seed + 7)
    reach2 = (adj @ adj + adj > 0).float()  # within 2 hops
    pairs, labels = [], []
    for _ in range(n_pairs):
        i = int(torch.randint(0, n, (1,), generator=g))
        j = int(torch.randint(0, n, (1,), generator=g))
        if i == j or adj[i, j] > 0:
            continue
        pairs.append((i, j))
        labels.append(1.0 if reach2[i, j] > 0 else 0.0)
    return pairs, torch.tensor(labels)


def _threshold_acc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Best-threshold binary accuracy of a score against labels (a tuned baseline gets its best cut)."""
    if labels.numel() == 0 or labels.float().std() == 0:
        return 0.0
    best = 0.0
    for thr in torch.quantile(scores, torch.linspace(0.05, 0.95, 19)):
        acc = float(((scores > thr).float() == labels).float().mean())
        best = max(best, acc, 1.0 - acc)
    return best


def _hidden_for_matched(dim: int, adim: int, hidden: int) -> int:
    """Widen the blind predictor hidden so its FLOPs match the action-conditioned arm (input dim+adim
    vs dim). Solve 2*(dim*h + h*dim) == 2*((dim+adim)*hidden + hidden*dim) for h."""
    target = (dim + adim) * hidden + hidden * dim
    h = round(target / (dim + dim))
    return max(1, int(h))


def _fit_dynamics(
    x: torch.Tensor, xnext: torch.Tensor, cut: int, din: int, hidden: int, dout: int, epochs: int, seed: int
) -> float:
    """Fit a next-latent predictor and return held-out MSE."""
    seed_everything(seed)
    xtr, xte = x[:cut], x[cut:]
    ytr, yte = xnext[:cut], xnext[cut:]
    net = mlp(din, dout, hidden, depth=1)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        F.mse_loss(net(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float(F.mse_loss(net(xte), yte))


def _kmeans_codes(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    """VQ assignment by Lloyd k-means (the codebook in the no-pixel-decoder setting)."""
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


def _code_receiver_acc(codes: torch.Tensor, target: torch.Tensor, k: int, seed: int) -> float:
    """Receiver sees ONLY the one-hot code and decodes the target bin (linear readout)."""
    onehot = torch.zeros(codes.shape[0], k)
    onehot[torch.arange(codes.shape[0]), codes] = 1.0
    return float(linear_probe(onehot, target, seed=seed)["score"])


def _affordance_relevance(dim: int, separation: float, seed: int) -> tuple[float, float]:
    """Action-relevance decodability for the learnable region vs the noise region. The learnable region
    carries a 2-class action-relevance contrast (decodable); the noise region is isotropic (near chance),
    so the affordance probe does NOT fire on noise (the honest noisy-TV guard for the affordance signal)."""
    g = torch.Generator().manual_seed(seed)
    n = 200
    centers = torch.randn(2, dim, generator=g)
    yl = torch.randint(0, 2, (n,), generator=g)
    xl = centers[yl] * separation + 0.5 * torch.randn(n, dim, generator=g)
    aff_l = linear_probe(xl, yl, seed=seed)["score"]
    xn = torch.randn(n, dim, generator=g) * separation
    yn = torch.randint(0, 2, (n,), generator=g)  # label is independent of the isotropic noise
    aff_n = linear_probe(xn, yn, seed=seed)["score"]
    return aff_l, aff_n
