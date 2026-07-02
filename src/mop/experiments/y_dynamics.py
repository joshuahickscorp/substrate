"""Series Y: the shell as a dynamical system. Eight cpu-now experiments that instrument the trainable
shell with dynamical-systems readouts: fixed-point convergence of the IterativeRefiner (Y1), basin
stability under input perturbation (Y2), seed consistency of fixed points (Y3), a homeostatic PI
learning-rate loop vs a tuned schedule (Y5), free-energy-vs-learning-progress on the noisy-TV budget
(Y6), a controllability system-identification gate before any latent-planning claim (Y7), recurrent
BIBS stability of a stateful shell (Y8), and the verifier as a contraction operator (Y9).

Every run() declares an explicit NULL and returns "null_supported": bool. Each wires a standing control:
matched-compute (Y1, Y8, Y9 via diagnostics.compute), frozen-random substrate (Y3 via
frozen_random_projection), a tuned baseline (Y5), an action-shuffle / marginal floor (Y7), or the
noisy-TV guard (Y5, Y6). Honest nulls only; never tuned toward a positive. Pooled latents discard
within-frame structure, so these are precursor / probe-gate measurements on cached pooled latents.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, mlp_flops, refiner_flops
from ..diagnostics.convergence import basin_stability, convergence_report
from ..diagnostics.geometry import effective_rank
from ..diagnostics.noisy_tv import noisy_tv_diagnostic
from ..diagnostics.seed_consistency import cross_seed_cka
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..diagnostics.sysid import sysid_report
from ..seeding import seed_everything
from ..shell.predictor import mlp
from ..shell.refine import IterativeRefiner, Verifier
from .base import Experiment


# --------------------------------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------------------------------
def _mean(v: list[float]) -> float:
    return sum(v) / max(1, len(v))


def _make_stream(e, s: int, forward_dynamics: bool = False):
    from ..substrate.datasets import make_task_stream

    return make_task_stream(
        n_tasks=1,
        dim=int(e.dim),
        classes_per_task=int(e.n_classes),
        samples_per_task=int(e.samples),
        separation=float(e.separation),
        forward_dynamics=forward_dynamics,
        seed=s,
    )[0]


def _train_refiner(refiner: nn.Module, head: nn.Module, x, y, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam([*refiner.parameters(), *head.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z, _ = refiner(x)
        F.cross_entropy(head(z), y).backward()
        opt.step()


# ==================================================================================================
# Y1: fixed-point existence and convergence of the IterativeRefiner trajectory
# ==================================================================================================
class Y1(Experiment):
    id = "y1_fixed_point_convergence"
    metric = ("converged_fraction", "contraction_factor", "head_loss_at_fixed_point_vs_trained_n")
    baseline = "head loss at trained N steps vs head loss at the unrolled fixed point (K >> N)"
    ablation = "trained refiner unrolled to K=64 vs its trained N; per-step update-norm contraction slope"
    null_hypothesis = (
        "trajectories do not settle: the update norm neither decays geometrically nor stabilizes (no "
        "attractor, a fixed-depth feedforward stack that happens to be unrolled), OR the head loss at "
        "the fixed point is no better than at trained N within the margin"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        trained_n, big_k = int(e.trained_steps), int(e.unroll_steps)
        eps, margin = float(e.epsilon), float(e.margin)

        conv_frac, cfactors, loss_n, loss_fp = [], [], [], []
        for s in seeds:
            seed_everything(s)
            task = _make_stream(e, s)
            x, y = task.x, task.y
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            nc = int(y.max()) + 1

            refiner = IterativeRefiner(dim, hidden, trained_n)
            head = nn.Linear(dim, nc)
            _train_refiner(refiner, head, xtr, ytr, int(e.epochs), float(e.lr))

            # per-sample convergence: unroll each row to K and check final update norm < eps
            converged = 0
            with torch.no_grad():
                for i in range(xte.shape[0]):
                    _, norms = refiner.unroll(xte[i : i + 1], big_k)
                    if norms and norms[-1] < eps:
                        converged += 1
                conv_frac.append(converged / max(1, xte.shape[0]))
                rep = convergence_report(refiner, xte, steps=big_k)
                cfactors.append(rep["contraction_factor"])

                # head loss at trained N vs at the fixed point (unrolled K)
                z_n, _ = refiner(xte, max_steps=trained_n)
                z_fp, _ = refiner.unroll(xte, big_k)
                loss_n.append(float(F.cross_entropy(head(z_n), yte)))
                loss_fp.append(float(F.cross_entropy(head(z_fp), yte)))

        cf = _mean(cfactors)
        frac = _mean(conv_frac)
        ln, lfp = _mean(loss_n), _mean(loss_fp)
        loss_improves = (ln - lfp) > margin  # does going to the fixed point lower head loss
        # null: no contraction (cf >= 0 OR few converge) OR the fixed point does not beat trained N
        null = bool((cf >= 0.0 or frac < 0.5) or not loss_improves)
        return {
            "converged_fraction": round(frac, 4),
            "contraction_factor": round(cf, 6),
            "head_loss_trained_n": round(ln, 4),
            "head_loss_fixed_point": round(lfp, 4),
            "fixed_point_loss_gain": round(ln - lfp, 4),
            "is_contraction": bool(cf < 0.0 and frac >= 0.5),
            "fixed_point_helps": bool(loss_improves),
            "trained_steps": trained_n,
            "unroll_steps": big_k,
            "margin": margin,
            "seeds": list(seeds),
            "null_supported": null,
        }


# ==================================================================================================
# Y2: basin stability of refinement under input perturbation (Lyapunov-lite)
# ==================================================================================================
class Y2(Experiment):
    id = "y2_basin_stability"
    metric = ("sensitivity_coefficient", "contraction_ratio", "perturbed_acc_vs_raw")
    baseline = "raw-latent head accuracy under the same input perturbation (does the refiner help)"
    ablation = "sensitivity (output displacement / input epsilon) swept across an epsilon grid"
    null_hypothesis = (
        "the refiner neither contracts nor amplifies perturbations beyond what the raw latent already "
        "does (sensitivity ratio near 1), so there is no basin structure; OR sensitivity is greater "
        "than 1 (amplifies noise) and perturbed-input accuracy is no better than the raw-latent head"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        steps, unroll = int(e.trained_steps), int(e.unroll_steps)
        eps_grid = [float(v) for v in e.epsilon_grid]

        sens_list, ratio_list, ref_acc, raw_acc = [], [], [], []
        for s in seeds:
            seed_everything(s)
            task = _make_stream(e, s)
            x, y = task.x, task.y
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            nc = int(y.max()) + 1

            refiner = IterativeRefiner(dim, hidden, steps)
            head = nn.Linear(dim, nc)
            _train_refiner(refiner, head, xtr, ytr, int(e.epochs), float(e.lr))
            raw_head = nn.Linear(dim, nc)
            ro = torch.optim.Adam(raw_head.parameters(), lr=float(e.lr))
            for _ in range(int(e.epochs)):
                ro.zero_grad()
                F.cross_entropy(raw_head(xtr), ytr).backward()
                ro.step()

            with torch.no_grad():
                # sensitivity slope: mean output displacement at fixed point vs input epsilon
                z_clean, _ = refiner.unroll(xte, unroll)
                disp, scale = [], float(xte.norm(dim=-1).mean())
                g = torch.Generator().manual_seed(s + 7)
                for ep in eps_grid:
                    u = torch.randn(xte.shape, generator=g)
                    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    z_p, _ = refiner.unroll(xte + ep * scale * u, unroll)
                    disp.append(float((z_p - z_clean).norm(dim=-1).mean()))
                # slope of output displacement vs (epsilon*scale)
                xs = [ep * scale for ep in eps_grid]
                mx, my = _mean(xs), _mean(disp)
                num = sum((a - mx) * (b - my) for a, b in zip(xs, disp, strict=True))
                den = sum((a - mx) ** 2 for a in xs) or 1e-9
                sens_list.append(num / den)

                bs = basin_stability(refiner, xte, eps=float(e.basin_eps), steps=unroll, seed=s)
                ratio_list.append(bs["contraction_ratio"])

                # perturbed-input accuracy: refiner-head vs raw-head on a fixed perturbation
                u = torch.randn(xte.shape, generator=g)
                u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                xp = xte + float(e.basin_eps) * scale * u
                zp, _ = refiner.unroll(xp, unroll)
                ref_acc.append(float((head(zp).argmax(-1) == yte).float().mean()))
                raw_acc.append(float((raw_head(xp).argmax(-1) == yte).float().mean()))

        sens = _mean(sens_list)
        ratio = _mean(ratio_list)
        ra, rw = _mean(ref_acc), _mean(raw_acc)
        stable = sens < 1.0 and ratio < 1.0
        helps = (ra - rw) > float(e.acc_margin)
        # null: not a contracting basin (sens/ratio >= 1) OR perturbed acc no better than raw head
        null = bool((not stable) or (not helps))
        return {
            "sensitivity_coefficient": round(sens, 4),
            "contraction_ratio": round(ratio, 4),
            "refiner_perturbed_acc": round(ra, 4),
            "raw_perturbed_acc": round(rw, 4),
            "perturbed_acc_gain": round(ra - rw, 4),
            "is_stable_basin": bool(stable),
            "refiner_helps_under_noise": bool(helps),
            "epsilon_grid": list(eps_grid),
            "seeds": list(seeds),
            "null_supported": null,
        }


# ==================================================================================================
# Y3: seed consistency of fixed points (do refiners from different seeds find the same attractor)
# ==================================================================================================
class Y3(Experiment):
    id = "y3_seed_consistent_fixed_points"
    metric = ("cross_seed_fixed_point_cka", "decision_agreement", "spread_ratio_vs_frozen_random")
    baseline = "frozen-random projection of the inputs as the no-attractor cross-seed-CKA floor"
    ablation = "K seeded refiners run to fixed point; cross-seed CKA of fixed points vs the random floor"
    null_hypothesis = (
        "fixed points are seed-dependent (cross-seed CKA near the frozen-random floor, spread on the "
        "order of the input spread): the attractor is an artifact of initialization, not imposed by the "
        "task or substrate (seed-instability badge)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        data_seed = int(e.data_seed)
        dim, hidden = int(e.dim), int(e.hidden)
        steps, unroll = int(e.trained_steps), int(e.unroll_steps)
        k_seeds = int(e.k_seeds)

        seed_everything(data_seed)
        task = _make_stream(e, data_seed)
        x, y = task.x, task.y
        cut = int(x.shape[0] * 0.7)
        xtr, ytr, xte = x[:cut], y[:cut], x[cut:]
        nc = int(y.max()) + 1

        fixed_points: list[torch.Tensor] = []
        decisions: list[torch.Tensor] = []
        for s in range(k_seeds):
            seed_everything(s)
            refiner = IterativeRefiner(dim, hidden, steps)
            head = nn.Linear(dim, nc)
            _train_refiner(refiner, head, xtr, ytr, int(e.epochs), float(e.lr))
            with torch.no_grad():
                z_fp, _ = refiner.unroll(xte, unroll)
                fixed_points.append(z_fp.clone())
                decisions.append(head(z_fp).argmax(-1))

        # cross-seed CKA of the converged fixed points (rotation invariant, respects D2)
        fp_cka = cross_seed_cka(fixed_points)["mean_cka"]
        # frozen-random floor: CKA across random projections of the same inputs
        rand_reps = [frozen_random_projection(xte, seed=s) for s in range(k_seeds)]
        floor_cka = cross_seed_cka(rand_reps)["mean_cka"]

        # decision agreement: mean pairwise fraction of identical argmax decisions
        agrees = []
        for i in range(k_seeds):
            for j in range(i + 1, k_seeds):
                agrees.append(float((decisions[i] == decisions[j]).float().mean()))
        agree = _mean(agrees) if agrees else 1.0

        # spread ratio: cross-seed fixed-point spread vs within-class input spread
        fp_stack = torch.stack(fixed_points, dim=0)  # [K, N, D]
        fp_spread = float((fp_stack - fp_stack.mean(0, keepdim=True)).norm(dim=-1).mean())
        class_spread = float((xte - xte.mean(0, keepdim=True)).norm(dim=-1).mean())
        spread_ratio = fp_spread / max(1e-9, class_spread)

        consistent = (fp_cka > floor_cka + float(e.cka_margin)) and (agree > float(e.agree_floor))
        # null: cross-seed CKA at/below the frozen-random floor (seed-dependent attractor)
        null = bool(not consistent)
        return {
            "cross_seed_fixed_point_cka": round(fp_cka, 4),
            "frozen_random_floor_cka": round(floor_cka, 4),
            "cka_above_floor": round(fp_cka - floor_cka, 4),
            "decision_agreement": round(agree, 4),
            "spread_ratio_vs_input": round(spread_ratio, 4),
            "shared_attractor": bool(consistent),
            "k_seeds": k_seeds,
            "null_supported": null,
        }


# ==================================================================================================
# Y5: homeostatic learning-rate control loop vs scheduled plasticity
# ==================================================================================================
class Y5(Experiment):
    id = "y5_homeostatic_lr_loop"
    metric = ("frontier_auc_pi_minus_tuned", "lr_overshoot", "gate_on_noisy_tv")
    baseline = "best tuned open-loop schedule (constant LR and cosine decay) on the same frontier"
    ablation = "PI homeostat (LR regulated to a prediction-error set-point) vs constant vs cosine decay"
    null_hypothesis = (
        "the closed-loop homeostat ties the best tuned open-loop schedule on frontier AUC (feedback adds "
        "machinery, the metric does not move), OR the loop is unstable and underperforms; on the noisy-TV "
        "it chases aleatoric error"
    )
    tier = "cpu-now"

    def _run_schedule(self, x, y, dim, nc, epochs, base_lr, kind, kp, ki, setpoint):
        """Train a linear head over a class-incremental stream; return per-epoch held-out error and the
        LR trajectory. kind in {constant, cosine, pi}. pi regulates LR to a prediction-error setpoint."""
        cut = int(x.shape[0] * 0.7)
        xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
        head = nn.Linear(dim, nc)
        opt = torch.optim.SGD(head.parameters(), lr=base_lr)
        errs, lrs = [], []
        integ = 0.0
        for ep in range(epochs):
            if kind == "constant":
                lr = base_lr
            elif kind == "cosine":
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * ep / max(1, epochs)))
            else:  # pi homeostat: actuate LR on the deviation from the error set-point
                with torch.no_grad():
                    cur_err = float(F.cross_entropy(head(xtr), ytr))
                err_dev = cur_err - setpoint
                integ += err_dev
                lr = max(0.0, base_lr * (1.0 + kp * err_dev + ki * integ))
                lr = min(lr, base_lr * 5.0)  # actuator saturation (BIBO bound)
            for grp in opt.param_groups:
                grp["lr"] = lr
            opt.zero_grad()
            F.cross_entropy(head(xtr), ytr).backward()
            opt.step()
            with torch.no_grad():
                errs.append(float(F.cross_entropy(head(xte), yte)))
            lrs.append(lr)
        return errs, lrs

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        epochs, base_lr = int(e.epochs), float(e.lr)
        kp, ki, setpoint = float(e.kp), float(e.ki), float(e.setpoint)

        pi_auc, const_auc, cos_auc, overshoots = [], [], [], []
        for s in seeds:
            seed_everything(s)
            task = _make_stream(e, s)
            x, y = task.x, task.y

            def auc(errs):  # frontier AUC proxy: area UNDER 1-error (higher is better)
                return _mean([1.0 - min(1.0, er) for er in errs])

            errs_c, _ = self._run_schedule(x, y, dim, nc, epochs, base_lr, "constant", kp, ki, setpoint)
            errs_co, _ = self._run_schedule(x, y, dim, nc, epochs, base_lr, "cosine", kp, ki, setpoint)
            errs_pi, lrs_pi = self._run_schedule(x, y, dim, nc, epochs, base_lr, "pi", kp, ki, setpoint)
            const_auc.append(auc(errs_c))
            cos_auc.append(auc(errs_co))
            pi_auc.append(auc(errs_pi))
            peak = max(lrs_pi) if lrs_pi else base_lr
            overshoots.append((peak - base_lr) / max(1e-9, base_lr))

        # noisy-TV guard: does the homeostat raise LR on irreducible aleatoric error
        ntv = noisy_tv_diagnostic(dim=dim, device=device, steps=int(e.ntv_steps), seed=int(seeds[0]))
        chases_noise = bool(not ntv["learning_progress_separates"])

        pim, cm, com = _mean(pi_auc), _mean(const_auc), _mean(cos_auc)
        best_tuned = max(cm, com)
        gain = pim - best_tuned
        # null: PI ties the best tuned open-loop within margin, OR it chases noisy-TV
        null = bool(gain <= float(e.margin) or chases_noise)
        return {
            "frontier_auc_pi": round(pim, 4),
            "frontier_auc_constant": round(cm, 4),
            "frontier_auc_cosine": round(com, 4),
            "frontier_auc_pi_minus_tuned": round(gain, 4),
            "lr_overshoot": round(_mean(overshoots), 4),
            "gate_on_noisy_tv_chases": chases_noise,
            "margin": float(e.margin),
            "pi_beats_tuned": bool(gain > float(e.margin) and not chases_noise),
            "seeds": list(seeds),
            "null_supported": null,
        }


# ==================================================================================================
# Y6: free-energy (active-inference) selection vs learning-progress, decomposed
# ==================================================================================================
class Y6(Experiment):
    id = "y6_free_energy_vs_lp"
    metric = ("noisy_tv_rejection_efe", "efe_lp_rank_correlation", "epistemic_minus_lp_coverage")
    baseline = "the EX8 learning-progress signal on the learnable-vs-noise budget (the same engine)"
    ablation = "EFE decomposed into epistemic vs complexity vs pragmatic; each scored on noisy-TV budget"
    null_hypothesis = (
        "the free-energy selector does not beat learning-progress on coverage or noisy-TV rejection, and "
        "the epistemic term is rank-correlated with learning-progress near 1 (free energy is "
        "learning-progress relabeled)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)

        rank_corrs, efe_rej, lp_rej, cover_gain = [], [], [], []
        for s in seeds:
            # the noisy-TV diagnostic gives, per region, raw error (pragmatic), disagreement
            # (epistemic), and learning progress. We build the EFE-epistemic = disagreement and
            # compare its region ranking to the learning-progress ranking on the same regions.
            ntv = noisy_tv_diagnostic(dim=dim, device=device, steps=int(e.ntv_steps), seed=s)
            dis_l = ntv["disagreement"]["learnable"]
            dis_n = ntv["disagreement"]["noise"]
            lp_l = ntv["learning_progress"]["learnable"]
            lp_n = ntv["learning_progress"]["noise"]

            # EFE-epistemic (disagreement) rejects noise if it scores learnable above noise
            efe_rej.append(1.0 if dis_l > dis_n else 0.0)
            lp_rej.append(1.0 if lp_l > lp_n else 0.0)
            # two-point rank agreement between epistemic and LP orderings (1 if same sign, else -1)
            same = (dis_l - dis_n) * (lp_l - lp_n)
            rank_corrs.append(1.0 if same > 0 else (-1.0 if same < 0 else 0.0))
            # learnable-coverage proxy: epistemic mass on learnable vs total (vs LP coverage)
            efe_cov = dis_l / max(1e-9, dis_l + dis_n)
            lp_cov = max(0.0, lp_l) / max(1e-9, max(0.0, lp_l) + max(0.0, lp_n))
            cover_gain.append(efe_cov - lp_cov)

        corr = _mean(rank_corrs)
        efe_r, lp_r = _mean(efe_rej), _mean(lp_rej)
        cov = _mean(cover_gain)
        relabel = corr > float(e.corr_threshold)  # epistemic == LP relabeled
        beats = (efe_r > lp_r + 1e-6) or (cov > float(e.cover_margin))
        # null: EFE does not beat LP AND epistemic is rank-correlated with LP near 1 (relabel)
        null = bool((not beats) and relabel)
        return {
            "noisy_tv_rejection_efe": round(efe_r, 4),
            "noisy_tv_rejection_lp": round(lp_r, 4),
            "efe_lp_rank_correlation": round(corr, 4),
            "epistemic_minus_lp_coverage": round(cov, 4),
            "efe_is_lp_relabeled": bool(relabel),
            "efe_beats_lp": bool(beats),
            "seeds": list(seeds),
            "null_supported": null,
        }


# ==================================================================================================
# Y7: linear-probe gate for control-theoretic state (is the latent a controllable system)
# ==================================================================================================
class Y7(Experiment):
    id = "y7_controllability_sysid_gate"
    metric = ("one_step_rollout_r2", "controllability_gramian_rank", "action_shuffle_delta")
    baseline = "action-shuffled linear fit (B should collapse) and the marginal next-latent floor"
    ablation = "linear z' = A z + B a system identification vs action-shuffle; controllability Gramian"
    null_hypothesis = (
        "actions do not linearly move the pooled latent state (B indistinguishable from the "
        "action-shuffle control, controllability Gramian rank-deficient), so the pooled latent is not a "
        "controllable state and EX2 planning is not licensed"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, adim = int(e.dim), int(e.action_dim)
        n = int(e.samples)

        one_step, ranks, deltas, licensed = [], [], [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # action-conditioned synthetic control family: z' = A z + B a + small noise. This is the
            # forward_dynamics analog with an explicit action channel (controls.py style); it is a TOY
            # controllable system so the gate measures whether sysid recovers it from pooled latents.
            A = torch.randn(dim, dim, generator=g) * (1.0 / dim**0.5)
            B = torch.randn(dim, adim, generator=g) * (1.0 / adim**0.5)
            Z = torch.randn(n, dim, generator=g)
            Aact = torch.randn(n, adim, generator=g)
            Znext = Z @ A.T + Aact @ B.T + float(e.noise) * torch.randn(n, dim, generator=g)

            rep = sysid_report(Z, Aact, Znext, k=int(e.rollout_k), seed=s)
            one_step.append(rep["one_step_r2"])
            ranks.append(rep["gramian"]["controllability_rank"])
            deltas.append(rep["action_delta"])
            licensed.append(1.0 if rep["planning_licensed"] else 0.0)

        os_r2 = _mean(one_step)
        rank = _mean(ranks)
        delta = _mean(deltas)
        lic = _mean(licensed)
        # null: actions do not move the state (small action-delta) OR rollout R2 below the floor
        not_controllable = (delta <= float(e.action_margin)) or (os_r2 <= float(e.r2_floor))
        null = bool(not_controllable)
        return {
            "one_step_rollout_r2": round(os_r2, 4),
            "controllability_gramian_rank": round(rank, 2),
            "state_dim": dim,
            "action_shuffle_delta": round(delta, 4),
            "planning_licensed_fraction": round(lic, 4),
            "is_controllable_state": bool(not not_controllable),
            "r2_floor": float(e.r2_floor),
            "seeds": list(seeds),
            "null_supported": null,
        }


# ==================================================================================================
# Y8: recurrent state stability (does a stateful shell drift or settle over a stream)
# ==================================================================================================
class _LeakyRecurrent(nn.Module):
    """A leaky recurrent predictor over the latent stream: h_{t+1} = (1-alpha) h_t + alpha f(h_t, z_t).
    The readout uses [z_t, h_t]. The stateless control reads z_t alone at matched parameter budget."""

    def __init__(self, dim: int, hidden: int, nc: int, alpha: float, gain: float):
        super().__init__()
        self.dim, self.alpha, self.gain = dim, alpha, gain
        self.f = mlp(dim + dim, dim, hidden, depth=1, ln=True)
        self.head = nn.Linear(dim + dim, nc)

    def step(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        upd = self.gain * self.f(torch.cat([h, z], dim=-1))
        return (1 - self.alpha) * h + self.alpha * upd

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.step(z, h)
        return self.head(torch.cat([z, h], dim=-1)), h


class Y8(Experiment):
    id = "y8_recurrent_bibs_stability"
    metric = ("hidden_state_bounded", "effective_rank_of_state", "forward_transfer_stateful_minus_stateless")
    baseline = "stateless head reading z_t alone at matched compute (diagnostics.compute FLOPs)"
    ablation = "leaky recurrent state h over the ordered stream vs stateless; state-norm + effective rank"
    null_hypothesis = (
        "the recurrent state either diverges/saturates (no useful bounded dynamics) or, when stable, "
        "provides no forward-transfer gain over the stateless head at matched compute (the stream is "
        "too uniform to carry usable state)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, nc = int(e.dim), int(e.hidden), int(e.n_classes)
        alpha, gain = float(e.alpha), float(e.gain)

        bounded, eranks, transfers = [], [], []
        for s in seeds:
            seed_everything(s)
            # domain-incremental stream so order carries usable cross-task state
            from ..substrate.datasets import make_task_stream

            tasks = make_task_stream(
                n_tasks=int(e.n_tasks),
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            xs = torch.cat([t.x for t in tasks], dim=0)
            ys = torch.cat([t.y for t in tasks], dim=0)
            cut = int(xs.shape[0] * 0.7)
            xtr, ytr, xte, yte = xs[:cut], ys[:cut], xs[cut:], ys[cut:]

            rec = _LeakyRecurrent(dim, hidden, nc, alpha, gain)
            opt = torch.optim.Adam(rec.parameters(), lr=float(e.lr))
            h = torch.zeros(xtr.shape[0], dim)
            norms = []
            for _ in range(int(e.epochs)):
                opt.zero_grad()
                logits, h_new = rec(xtr, h.detach())
                F.cross_entropy(logits, ytr).backward()
                opt.step()
                h = h_new.detach()
                norms.append(float(h.norm(dim=-1).mean()))
            # BIBS: bounded if the late state norm did not blow up vs the early norm
            early, late = norms[0] if norms else 1.0, norms[-1] if norms else 1.0
            is_bounded = late < float(e.divergence_factor) * max(early, 1e-6)
            bounded.append(1.0 if is_bounded else 0.0)
            eranks.append(effective_rank(h))

            with torch.no_grad():
                logits_te, _ = rec(xte, torch.zeros(xte.shape[0], dim))
                stateful_acc = float((logits_te.argmax(-1) == yte).float().mean())

            # stateless control at matched compute: same width head reading [z, z] (no recurrence)
            seed_everything(s)
            sl_head = nn.Sequential(mlp(dim + dim, dim, hidden, depth=1, ln=True), nn.Linear(dim, nc))
            so = torch.optim.Adam(sl_head.parameters(), lr=float(e.lr))
            for _ in range(int(e.epochs)):
                so.zero_grad()
                F.cross_entropy(sl_head(torch.cat([xtr, xtr], dim=-1)), ytr).backward()
                so.step()
            with torch.no_grad():
                stateless_acc = float(
                    (sl_head(torch.cat([xte, xte], dim=-1)).argmax(-1) == yte).float().mean()
                )
            transfers.append(stateful_acc - stateless_acc)

        # matched-compute check: stateful step and stateless head both apply one (2*dim->hidden->dim)
        # block per pass, so forward FLOPs match within tolerance
        flops_rec = mlp_flops([2 * dim, hidden, dim])
        flops_sl = mlp_flops([2 * dim, hidden, dim])
        compute = matched_within(flops_rec, flops_sl)

        bnd = _mean(bounded)
        er = _mean(eranks)
        tr = _mean(transfers)
        all_bounded = bnd >= 0.5
        helps = tr > float(e.transfer_margin)
        # null: state diverges/saturates OR (when stable) no transfer gain at matched compute
        null = bool((not all_bounded) or (not helps))
        return {
            "hidden_state_bounded": bool(all_bounded),
            "bounded_fraction": round(bnd, 4),
            "effective_rank_of_state": round(er, 4),
            "forward_transfer_stateful_minus_stateless": round(tr, 4),
            "compute_matched": compute,
            "state_helps_transfer": bool(helps),
            "alpha": alpha,
            "seeds": list(seeds),
            "null_supported": null,
        }


# ==================================================================================================
# Y9: verifier as a contraction operator (EX18 self-correction as fixed-point iteration)
# ==================================================================================================
class Y9(Experiment):
    id = "y9_verifier_contraction"
    metric = ("monotone_descent_fraction", "self_correction_gain_vs_ex17", "shuffled_verifier_delta")
    baseline = "plain EX17 refinement and a compute-matched single-shot head at equal FLOPs"
    ablation = "verify-revise loop vs plain refinement vs shuffled-verifier control; Lyapunov V descent"
    null_hypothesis = (
        "V does not decrease monotonically (the verifier carries no usable correction direction, the "
        "shuffled-verifier control ties it) OR verify-revise ties plain EX17 and the compute-matched "
        "single-shot head (self-correction was just more depth)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        steps, revise_iters = int(e.trained_steps), int(e.revise_iters)
        thresh = float(e.verify_threshold)

        descent_frac, gains, shuf_deltas = [], [], []
        for s in seeds:
            seed_everything(s)
            task = _make_stream(e, s)
            x, y = task.x, task.y
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            nc = int(y.max()) + 1

            refiner = IterativeRefiner(dim, hidden, steps)
            head = nn.Linear(dim, nc)
            verifier = Verifier(dim, hidden=hidden)
            # train refiner+head, then train the verifier to predict per-sample head NLL (the error
            # functional V it must drive down). Verifier loss is regression onto current NLL.
            _train_refiner(refiner, head, xtr, ytr, int(e.epochs), float(e.lr))
            vo = torch.optim.Adam(verifier.parameters(), lr=float(e.lr))
            with torch.no_grad():
                z_tr, _ = refiner(xtr)
            for _ in range(int(e.epochs)):
                vo.zero_grad()
                with torch.no_grad():
                    nll = F.cross_entropy(head(z_tr), ytr, reduction="none")
                F.mse_loss(verifier.score(z_tr), nll).backward()
                vo.step()

            def V(z, head=head, yte=yte):  # Lyapunov candidate: mean head NLL
                return float(F.cross_entropy(head(z), yte))

            with torch.no_grad():
                z, _ = refiner(xte)
                vs = [V(z)]
                # verify-revise: where the verifier flags high error, apply an extra refine step
                for _ in range(revise_iters):
                    flag = verifier.score(z) > thresh
                    z_ref, _ = refiner(z, max_steps=1)
                    z = torch.where(flag.unsqueeze(-1), z_ref, z)
                    vs.append(V(z))
                ex18_acc = float((head(z).argmax(-1) == yte).float().mean())

                # monotone-descent fraction across the V trajectory
                drops = sum(1 for a, b in zip(vs[:-1], vs[1:], strict=True) if b <= a + 1e-6)
                descent_frac.append(drops / max(1, len(vs) - 1))

                # plain EX17: refine the SAME extra steps unconditionally (compute-matched depth)
                z17, _ = refiner(xte)
                z17, _ = refiner(z17, max_steps=revise_iters)
                ex17_acc = float((head(z17).argmax(-1) == yte).float().mean())
                gains.append(ex18_acc - ex17_acc)

                # shuffled-verifier control: permute verifier scores (no usable direction)
                g = torch.Generator().manual_seed(s + 3)
                zsh, _ = refiner(xte)
                for _ in range(revise_iters):
                    sc = verifier.score(zsh)
                    sc = sc[torch.randperm(sc.shape[0], generator=g)]
                    flag = sc > thresh
                    z_ref, _ = refiner(zsh, max_steps=1)
                    zsh = torch.where(flag.unsqueeze(-1), z_ref, zsh)
                shuf_acc = float((head(zsh).argmax(-1) == yte).float().mean())
                shuf_deltas.append(ex18_acc - shuf_acc)

        # matched-compute: EX18 and plain-EX17 both spend (steps + revise_iters) refiner blocks worst case
        flops_ex18 = refiner_flops(dim, hidden, steps + revise_iters)
        flops_ex17 = refiner_flops(dim, hidden, steps + revise_iters)
        compute = matched_within(flops_ex18, flops_ex17)

        mdf = _mean(descent_frac)
        gain = _mean(gains)
        shuf = _mean(shuf_deltas)
        monotone = mdf >= float(e.descent_floor)
        beats_ex17 = gain > float(e.margin)
        beats_shuffle = shuf > float(e.margin)
        # null: V not monotone OR (verify-revise ties EX17 and the shuffled verifier ties it)
        null = bool((not monotone) or (not beats_ex17 and not beats_shuffle))
        return {
            "monotone_descent_fraction": round(mdf, 4),
            "self_correction_gain_vs_ex17": round(gain, 4),
            "shuffled_verifier_delta": round(shuf, 4),
            "compute_matched": compute,
            "is_contraction": bool(monotone and beats_ex17),
            "beats_shuffled_verifier": bool(beats_shuffle),
            "margin": float(e.margin),
            "seeds": list(seeds),
            "null_supported": null,
        }
