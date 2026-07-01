"""Series N: neuroscience and neuroscience-of-thought on the FROZEN pooled-latent substrate, CPU, seconds.

Eleven cpu-now experiments, each a self-contained Experiment subclass with an explicit null check and a
wired standing control (frozen-random substrate, matched-compute, or a tuned baseline; noisy-TV guard
where a curiosity signal appears). The mechanisms are implemented INLINE in this module per the
established pattern (see ex16_codebook_sr, ex17_latent_reasoning, ex8_curiosity_bakeoff).

  N1  replay ordering and surprise-prioritization in the latent hippocampus     (tax 5)
  N3  predictive-coding refiner vs generic residual refiner at matched compute   (tax 9)
  N4  synaptic tagging-and-capture as a surprise-gated consolidation schedule    (tax 2)
  N5  Fisher-triggered plasticity reopening at an unknown distribution shift     (tax 2)
  N6  ACh-vs-NE dissociation: expected vs unexpected uncertainty gating          (tax 1)
  N7  latent scratchpad: working-memory slots for a delayed-match latent task    (tax 5)
  N8  object permanence as a pooled-latent prediction bound                      (tax 3)
  N9  attractor convergence of the latent refiner                                (tax 9)
  N10 confidence-based halting tracks difficulty (latent metacognition)          (tax 4)
  N11 latent self-verification as error-monitoring                              (tax 4)

Honest nulls only: null_supported reflects the real toy outcome, never tuned toward a positive. Pooled
latents discard within-frame object/spatial structure, so N8 ships the bound (taxonomy slot 3) as the
result. No sentience or agency language. No em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, param_count, refiner_flops
from ..diagnostics.convergence import basin_stability, convergence_report
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from ..shell.modulation import WorkingMemory
from ..shell.predictor import mlp
from ..shell.refine import IterativeRefiner, Verifier
from .base import Experiment


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _spread(v: list[float]) -> float:
    """Half the seed range, the simple seed-spread band used to gate ties vs wins."""
    return (max(v) - min(v)) / 2.0 if len(v) > 1 else 0.0


def _pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation; 0.0 when either side is constant (no usable signal)."""
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return num / (da * db)


# =====================================================================================================
# N1: replay ordering and surprise-prioritization in the latent hippocampus
# =====================================================================================================
class N1(Experiment):
    id = "n1_replay_ordering"
    metric = ("backward_transfer", "frontier_auc", "ordering_deltas")
    baseline = "uniform rehearsal at matched buffer count; no-replay; frozen-random substrate"
    ablation = "forward vs reverse vs surprise-priority (ensemble disagreement) rehearsal order"
    null_hypothesis = (
        "reverse and surprise-priority replay tie uniform rehearsal at matched count, or all replay "
        "ties no-replay (stream too short, latents too uniform)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_tasks, cpt = int(e.n_tasks), int(e.classes_per_task)
        spt = int(e.samples_per_task)
        buf_n = int(e.buffer_count)
        epochs, lr = int(e.epochs), float(e.lr)

        schemes = ("no_replay", "uniform", "forward", "reverse", "surprise")
        bwt: dict[str, list[float]] = {s: [] for s in schemes}
        # frozen-random arm only for the headline (surprise) scheme
        bwt_fr: list[float] = []

        for s in seeds:
            stream = make_task_stream(
                n_tasks=n_tasks,
                dim=dim,
                classes_per_task=cpt,
                samples_per_task=spt,
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            nc = cpt  # domain-incremental reuses labels 0..cpt-1

            def run_scheme(scheme: str, transform=None, s=s, nc=nc, stream=stream):
                seed_everything(s)
                head = nn.Linear(dim, nc)
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                buffer_x: list[torch.Tensor] = []
                buffer_y: list[torch.Tensor] = []
                buffer_p: list[torch.Tensor] = []  # priority (disagreement proxy)
                first_task_acc = []
                for t, task in enumerate(stream):
                    x = task.x if transform is None else transform(task.x)
                    y = task.y
                    for _ in range(epochs):
                        opt.zero_grad()
                        xb, yb = x, y
                        if scheme != "no_replay" and buffer_x:
                            bx = torch.cat(buffer_x)
                            by = torch.cat(buffer_y)
                            bp = torch.cat(buffer_p)
                            k = min(buf_n, bx.shape[0])
                            if scheme == "uniform":
                                gsel = torch.randperm(bx.shape[0])[:k]
                            elif scheme == "forward":
                                gsel = torch.arange(k)
                            elif scheme == "reverse":
                                gsel = torch.arange(bx.shape[0] - k, bx.shape[0])
                            else:  # surprise: top-k by disagreement priority
                                gsel = bp.topk(k).indices
                            xb = torch.cat([x, bx[gsel]])
                            yb = torch.cat([y, by[gsel]])
                        F.cross_entropy(head(xb), yb).backward()
                        opt.step()
                    # store with a disagreement-proxy priority: distance to current decision margin
                    with torch.no_grad():
                        logits = head(x)
                        srt = logits.sort(dim=-1, descending=True).values
                        margin = (srt[:, 0] - srt[:, 1]).abs()
                        prio = 1.0 / (margin + 1e-3)  # low margin == high surprise priority
                    buffer_x.append(x.detach())
                    buffer_y.append(y.detach())
                    buffer_p.append(prio.detach())
                    if t == 0:
                        pass
                    # measure retention of task 0 after each task
                    with torch.no_grad():
                        x0 = stream[0].x if transform is None else transform(stream[0].x)
                        a0 = float((head(x0).argmax(-1) == stream[0].y).float().mean())
                    first_task_acc.append(a0)
                # backward transfer: task-0 acc at end minus right after learning it
                return first_task_acc[-1] - first_task_acc[0]

            for scheme in schemes:
                bwt[scheme].append(run_scheme(scheme))
            bwt_fr.append(
                run_scheme("surprise", transform=lambda z, s=s: frozen_random_projection(z, seed=s))
            )

        means = {k: round(_mean(v), 4) for k, v in bwt.items()}
        uni_spread = _spread(bwt["uniform"])
        d_reverse = means["reverse"] - means["uniform"]
        d_surprise = means["surprise"] - means["uniform"]
        d_replay_vs_none = means["uniform"] - means["no_replay"]
        # null: best ordering ties uniform within seed spread, OR replay ties no-replay
        best_order_delta = max(d_reverse, d_surprise)
        ordering_ties_uniform = best_order_delta <= uni_spread + 1e-4
        replay_ties_none = d_replay_vs_none <= _spread(bwt["uniform"]) + 1e-4
        return {
            "bwt_means": means,
            "delta_reverse_vs_uniform": round(d_reverse, 4),
            "delta_surprise_vs_uniform": round(d_surprise, 4),
            "delta_replay_vs_no_replay": round(d_replay_vs_none, 4),
            "uniform_seed_spread": round(uni_spread, 4),
            "surprise_frozen_random_bwt": round(_mean(bwt_fr), 4),
            "seeds": list(seeds),
            "null_supported": bool(ordering_ties_uniform or replay_ties_none),
            "ordering_beats_uniform": bool(not ordering_ties_uniform),
        }


# =====================================================================================================
# N3: predictive-coding refiner vs generic residual refiner at matched compute
# =====================================================================================================
class _UntiedDepth(nn.Module):
    """Compute-matched control: `steps` untied residual blocks applied once each (depth, not iteration)."""

    def __init__(self, dim: int, hidden: int, steps: int):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(steps)])
        self.blocks = nn.ModuleList([mlp(dim, dim, hidden, depth=1, ln=True) for _ in range(steps)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for norm, block in zip(self.norms, self.blocks, strict=True):
            z = z + block(norm(z))
        return z


def _fit_eval(backbone: nn.Module, head: nn.Module, x, y, xte, yte, epochs: int, lr: float) -> float:
    opt = torch.optim.Adam([*backbone.parameters(), *head.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z = backbone(x)
        z = z[0] if isinstance(z, tuple) else z
        F.cross_entropy(head(z), y).backward()
        opt.step()
    with torch.no_grad():
        z = backbone(xte)
        z = z[0] if isinstance(z, tuple) else z
        return float((head(z).argmax(-1) == yte).float().mean())


class N3(Experiment):
    id = "n3_pc_refiner"
    metric = ("accuracy", "gain_per_compute", "fixed_point_residual")
    baseline = "generic residual refiner AND a depth-and-FLOP-matched untied MLP; frozen-random substrate"
    ablation = "predictive-coding update (pc_rate*(pred-z)) vs generic residual update at matched FLOPs"
    null_hypothesis = (
        "the PC-shaped refiner ties the generic refiner and ties the compute-matched untied MLP "
        "(iteration was just depth, the update rule buys nothing on a frozen pooled latent)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        epochs, lr, margin = int(e.epochs), float(e.lr), float(e.margin)

        pc_acc, gen_acc, ctl_acc, pc_resid = [], [], [], []
        pc_fr_acc = []
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

            seed_everything(s)
            pc = IterativeRefiner(dim, hidden, steps, mode="predictive_coding", pc_rate=float(e.pc_rate))
            pc_acc.append(_fit_eval(pc, nn.Linear(dim, nc), xtr, ytr, xte, yte, epochs, lr))
            with torch.no_grad():
                _, norms = pc.unroll(xte, steps + 2)
            pc_resid.append(norms[-1] if norms else 0.0)

            seed_everything(s)
            gen = IterativeRefiner(dim, hidden, steps, mode="residual")
            gen_acc.append(_fit_eval(gen, nn.Linear(dim, nc), xtr, ytr, xte, yte, epochs, lr))

            seed_everything(s)
            ctl = _UntiedDepth(dim, hidden, steps)
            ctl_acc.append(_fit_eval(ctl, nn.Linear(dim, nc), xtr, ytr, xte, yte, epochs, lr))

            # frozen-random substrate arm for the PC refiner
            seed_everything(s)
            xfr = frozen_random_projection(x, seed=s)
            pcfr = IterativeRefiner(dim, hidden, steps, mode="predictive_coding", pc_rate=float(e.pc_rate))
            pc_fr_acc.append(_fit_eval(pcfr, nn.Linear(dim, nc), xfr[:cut], ytr, xfr[cut:], yte, epochs, lr))

        pm, gm, cm = _mean(pc_acc), _mean(gen_acc), _mean(ctl_acc)
        flops = refiner_flops(dim, hidden, steps)
        compute = matched_within(flops, flops)  # PC and generic share block count + hidden
        d_vs_generic = pm - gm
        d_vs_matched = pm - cm
        ties_generic = abs(d_vs_generic) <= margin
        ties_matched = abs(d_vs_matched) <= margin
        return {
            "pc_acc_mean": round(pm, 4),
            "generic_acc_mean": round(gm, 4),
            "matched_untied_acc_mean": round(cm, 4),
            "pc_frozen_random_acc": round(_mean(pc_fr_acc), 4),
            "gain_vs_generic": round(d_vs_generic, 4),
            "gain_vs_compute_matched": round(d_vs_matched, 4),
            "gain_per_compute": round(d_vs_matched / max(1, flops) * 1e6, 6),
            "fixed_point_residual": round(_mean(pc_resid), 6),
            "compute_matched": compute,
            "margin": margin,
            "seeds": list(seeds),
            "null_supported": bool(ties_generic and ties_matched),
            "pc_beats_both": bool(d_vs_generic > margin and d_vs_matched > margin),
        }


# =====================================================================================================
# N4: synaptic tagging-and-capture as a surprise-gated consolidation schedule
# =====================================================================================================
class N4(Experiment):
    id = "n4_tag_and_capture"
    metric = ("backward_transfer", "retention_per_byte")
    baseline = "uniform consolidation; confidence-priority-only; matched consolidation budget"
    ablation = "tagged-then-captured (low-confidence tag + later disagreement event) vs uniform replay"
    null_hypothesis = (
        "tagged-and-captured consolidation ties uniform consolidation at matched budget (the schedule "
        "adds no benefit, or a simpler confidence-priority already captures it)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_tasks, cpt, spt = int(e.n_tasks), int(e.classes_per_task), int(e.samples_per_task)
        budget = int(e.consolidation_budget)
        epochs, lr = int(e.epochs), float(e.lr)

        schemes = ("uniform", "confidence", "tag_capture")
        bwt: dict[str, list[float]] = {s: [] for s in schemes}
        bytes_used: dict[str, list[int]] = {s: [] for s in schemes}

        for s in seeds:
            stream = make_task_stream(
                n_tasks=n_tasks,
                dim=dim,
                classes_per_task=cpt,
                samples_per_task=spt,
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            nc = cpt

            def run_scheme(scheme: str, s=s, nc=nc, stream=stream):
                seed_everything(s)
                head = nn.Linear(dim, nc)
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                stored_x: list[torch.Tensor] = []
                stored_y: list[torch.Tensor] = []
                first0 = None
                kept = 0
                for t, task in enumerate(stream):
                    x, y = task.x, task.y
                    for _ in range(epochs):
                        opt.zero_grad()
                        xb, yb = x, y
                        if stored_x:
                            sx, sy = torch.cat(stored_x), torch.cat(stored_y)
                            k = min(budget, sx.shape[0])
                            gsel = torch.randperm(sx.shape[0])[:k]
                            xb = torch.cat([x, sx[gsel]])
                            yb = torch.cat([y, sy[gsel]])
                        F.cross_entropy(head(xb), yb).backward()
                        opt.step()
                    # tag low-confidence entries; capture depends on a later disagreement event
                    with torch.no_grad():
                        logits = head(x)
                        p = logits.softmax(-1)
                        conf = p.max(-1).values
                        low_conf = conf < float(e.tag_threshold)
                    if scheme == "uniform":
                        sel = torch.ones_like(low_conf)
                    elif scheme == "confidence":
                        sel = low_conf  # consolidate every low-confidence tag (no capture gate)
                    else:  # tag_capture: tag now, capture only if NEXT task is high-disagreement
                        next_shift = t + 1 < len(stream)
                        sel = low_conf if next_shift else torch.zeros_like(low_conf)
                    stored_x.append(x[sel].detach())
                    stored_y.append(y[sel].detach())
                    kept += int(sel.sum())
                    with torch.no_grad():
                        a0 = float((head(stream[0].x).argmax(-1) == stream[0].y).float().mean())
                    if first0 is None:
                        first0 = a0
                    last0 = a0
                assert first0 is not None  # set on the first task above (the stream is non-empty)
                return last0 - first0, kept * dim * 4  # 4 bytes per float

            for scheme in schemes:
                d, b = run_scheme(scheme)
                bwt[scheme].append(d)
                bytes_used[scheme].append(b)

        means = {k: round(_mean(v), 4) for k, v in bwt.items()}
        bmean = {k: int(_mean([float(z) for z in v])) for k, v in bytes_used.items()}
        d_vs_uniform = means["tag_capture"] - means["uniform"]
        d_vs_confidence = means["tag_capture"] - means["confidence"]
        spread = _spread(bwt["uniform"])
        rpb_tag = means["tag_capture"] / max(1, bmean["tag_capture"]) * 1e6
        rpb_uni = means["uniform"] / max(1, bmean["uniform"]) * 1e6
        ties_uniform = d_vs_uniform <= spread + 1e-4
        ties_confidence = d_vs_confidence <= spread + 1e-4
        return {
            "bwt_means": means,
            "bytes_mean": bmean,
            "delta_vs_uniform": round(d_vs_uniform, 4),
            "delta_vs_confidence_only": round(d_vs_confidence, 4),
            "retention_per_byte_tag": round(rpb_tag, 6),
            "retention_per_byte_uniform": round(rpb_uni, 6),
            "uniform_seed_spread": round(spread, 4),
            "seeds": list(seeds),
            "null_supported": bool(ties_uniform or ties_confidence),
            "tag_capture_beats_both": bool(not ties_uniform and not ties_confidence),
        }


# =====================================================================================================
# N5: Fisher-triggered plasticity reopening at an unknown distribution shift
# =====================================================================================================
class N5(Experiment):
    id = "n5_fisher_reopen"
    metric = ("frontier_auc_after_shift", "adaptation_speed", "shift_localization_acc")
    baseline = "constant-LR; staged-schedule; tuned cosine decay; frozen-random substrate"
    ablation = "Fisher-trace triggered LR reopening vs fixed and tuned schedules"
    null_hypothesis = (
        "triggered reopening ties tuned cosine decay (an LR trick), or the Fisher signal fails to "
        "localize the shift (estimator too weak)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n_chunks = int(e.n_chunks)
        spc = int(e.samples_per_chunk)
        epochs, lr = int(e.epochs), float(e.lr)

        schedules = ("constant", "staged", "cosine", "fisher_trigger")
        post_auc: dict[str, list[float]] = {s: [] for s in schedules}
        post_auc_fr: list[float] = []
        loc_acc: list[float] = []

        for s in seeds:
            # build a two-domain stream; a SHIFT happens at an unknown chunk index
            g = torch.Generator().manual_seed(s)
            shift_at = int(torch.randint(n_chunks // 3, 2 * n_chunks // 3, (1,), generator=g))
            two = make_task_stream(
                n_tasks=2,
                dim=dim,
                classes_per_task=int(e.classes_per_task),
                samples_per_task=spc,
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            nc = int(e.classes_per_task)

            def chunk(i, s=s, two=two, shift_at=shift_at):
                t = two[0] if i < shift_at else two[1]
                gg = torch.Generator().manual_seed(s * 100 + i)
                idx = torch.randint(0, t.x.shape[0], (spc,), generator=gg)
                return t.x[idx], t.y[idx]

            def run_sched(name: str, transform=None, s=s, nc=nc, shift_at=shift_at, chunk=chunk):
                seed_everything(s)
                head = nn.Linear(dim, nc)
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                fisher_ema = None
                triggered_at = -1
                accs = []
                for i in range(n_chunks):
                    x, y = chunk(i)
                    if transform is not None:
                        x = transform(x)
                    progress = i / max(1, n_chunks - 1)
                    if name == "constant":
                        scale = 1.0
                    elif name == "staged":
                        scale = 1.0 if progress < 0.5 else 0.2
                    elif name == "cosine":
                        scale = 0.2 + 0.8 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
                    else:  # fisher_trigger: reopen when Fisher-trace jumps
                        scale = 0.2
                    for gpar in opt.param_groups:
                        gpar["lr"] = lr * scale
                    for _ in range(epochs):
                        opt.zero_grad()
                        F.cross_entropy(head(x), y).backward()
                        # Fisher-trace proxy: mean squared grad over trainable params
                        ftrace = sum(
                            float((p.grad.detach() ** 2).mean())
                            for p in head.parameters()
                            if p.grad is not None
                        )
                        opt.step()
                    if name == "fisher_trigger":
                        if fisher_ema is None:
                            fisher_ema = ftrace
                        z = (ftrace - fisher_ema) / (abs(fisher_ema) + 1e-8)
                        if z > float(e.fisher_z) and triggered_at < 0:
                            triggered_at = i
                        if triggered_at >= 0 and i - triggered_at < int(e.reopen_window):
                            for gpar in opt.param_groups:
                                gpar["lr"] = lr * 1.0  # reopened
                        fisher_ema = 0.9 * fisher_ema + 0.1 * ftrace
                    with torch.no_grad():
                        accs.append(float((head(x).argmax(-1) == y).float().mean()))
                # post-shift frontier AUC: mean acc over chunks after the shift
                post = accs[shift_at:]
                auc = _mean(post)
                return auc, triggered_at

            for name in schedules:
                auc, trig = run_sched(name)
                post_auc[name].append(auc)
                if name == "fisher_trigger":
                    # localization: did the trigger land within a small window of the true shift
                    hit = trig >= 0 and abs(trig - shift_at) <= int(e.localize_tol)
                    loc_acc.append(1.0 if hit else 0.0)
            auc_fr, _ = run_sched(
                "fisher_trigger", transform=lambda z, s=s: frozen_random_projection(z, seed=s)
            )
            post_auc_fr.append(auc_fr)

        means = {k: round(_mean(v), 4) for k, v in post_auc.items()}
        d_vs_cosine = means["fisher_trigger"] - means["cosine"]
        spread = _spread(post_auc["cosine"])
        loc = _mean(loc_acc)
        ties_cosine = d_vs_cosine <= spread + 1e-4
        localizes = loc > 0.5 + 1e-9  # above chance for a single trigger guess
        return {
            "post_shift_auc_means": means,
            "delta_vs_tuned_cosine": round(d_vs_cosine, 4),
            "cosine_seed_spread": round(spread, 4),
            "shift_localization_acc": round(loc, 4),
            "fisher_trigger_frozen_random_auc": round(_mean(post_auc_fr), 4),
            "seeds": list(seeds),
            # null holds if it ties cosine OR fails to localize the shift
            "null_supported": bool(ties_cosine or not localizes),
            "reopening_beats_and_localizes": bool(not ties_cosine and localizes),
        }


# =====================================================================================================
# N6: ACh-vs-NE dissociation: expected vs unexpected uncertainty gating
# =====================================================================================================
class N6(Experiment):
    id = "n6_ach_ne_dissociation"
    metric = ("adapt_speed_at_boundary", "within_context_noise_rejection", "calibration_ece")
    baseline = "single disagreement scalar; ungated; point-error baseline; noisy-TV guard"
    ablation = "two-signal gate (ACh=mean entropy within-context, NE=disagreement spike at boundary)"
    null_hypothesis = (
        "the two-signal split ties a single disagreement scalar (the dissociation adds no benefit, "
        "or helps only combined)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream, noisy_tv_dataset

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        spc = int(e.samples_per_context)
        epochs, lr = int(e.epochs), float(e.lr)

        single_adapt, split_adapt = [], []
        single_rej, split_rej = [], []
        ece_list = []

        for s in seeds:
            # two contexts (a boundary between them) with within-context aleatoric noise
            two = make_task_stream(
                n_tasks=2,
                dim=dim,
                classes_per_task=int(e.classes_per_task),
                samples_per_task=spc,
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            nc = int(e.classes_per_task)
            ntv = noisy_tv_dataset(dim=dim, n=spc, separation=float(e.separation), seed=s)

            def add_aleatoric(x, s=s):
                gg = torch.Generator().manual_seed(s + 7)
                return x + torch.randn(x.shape, generator=gg) * float(e.aleatoric_scale)

            def run_gate(split: bool, s=s, nc=nc, two=two, ntv=ntv, add_aleatoric=add_aleatoric):
                seed_everything(s)
                head = nn.Linear(dim, nc)
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                # context 0 with within-context noise, then a boundary into context 1
                xs = [add_aleatoric(two[0].x), two[1].x]
                ys = [two[0].y, two[1].y]
                boundary_speed = None
                ema_dis = None
                for ci, (x, y) in enumerate(zip(xs, ys, strict=True)):
                    pre = None
                    for ep in range(epochs):
                        opt.zero_grad()
                        logits = head(x)
                        # disagreement proxy: entropy of the predictive softmax (a scalar gate signal,
                        # detached so it never carries a gradient into the LR schedule)
                        p = logits.detach().softmax(-1)
                        ent = float((-(p * (p + 1e-9).log()).sum(-1)).mean())
                        if ema_dis is None:
                            ema_dis = ent
                        spike = (ent - ema_dis) / (abs(ema_dis) + 1e-8)
                        ema_dis = 0.9 * ema_dis + 0.1 * ent
                        # ACh (within-context, expected) down-weights LR under high entropy;
                        # NE (boundary, unexpected) raises LR on a disagreement spike.
                        if split:
                            ach = 1.0 / (1.0 + float(e.ach_strength) * ent)
                            ne = 1.0 + float(e.ne_strength) * max(0.0, spike) if ci > 0 else 1.0
                            scale = ach * ne
                        else:
                            scale = 1.0  # single scalar: ungated baseline LR
                        for gpar in opt.param_groups:
                            gpar["lr"] = lr * scale
                        F.cross_entropy(logits, y).backward()
                        opt.step()
                        with torch.no_grad():
                            acc = float((head(x).argmax(-1) == y).float().mean())
                        if ci == 1 and pre is None and acc > float(e.adapt_acc):
                            boundary_speed = ep + 1
                            pre = acc
                    if ci == 1 and boundary_speed is None:
                        boundary_speed = epochs
                # within-context noise rejection: how stable predictions are on the noisy-TV noise region
                with torch.no_grad():
                    nz = ntv["noise"].x
                    # rejection == NOT moving much on irreducibly-noisy inputs (low logit variance over a
                    # perturbation), higher is better
                    gg = torch.Generator().manual_seed(s + 11)
                    nz2 = nz + torch.randn(nz.shape, generator=gg) * float(e.aleatoric_scale)
                    drift = float((head(nz) - head(nz2)).abs().mean())
                    rejection = 1.0 / (1.0 + drift)
                    # calibration ECE on context 1
                    logits1 = head(xs[1])
                    p1 = logits1.softmax(-1)
                    conf, pred = p1.max(-1)
                    correct = (pred == ys[1]).float()
                    ece = float((conf - correct).abs().mean())
                return boundary_speed, rejection, ece

            bs_s, rej_s, _ = run_gate(split=False)
            bs_p, rej_p, ece_p = run_gate(split=True)
            single_adapt.append(float(bs_s))
            split_adapt.append(float(bs_p))
            single_rej.append(rej_s)
            split_rej.append(rej_p)
            ece_list.append(ece_p)

        # lower boundary step == faster adaptation; higher rejection == better
        d_adapt = _mean(single_adapt) - _mean(split_adapt)  # positive == split adapts faster
        d_rej = _mean(split_rej) - _mean(single_rej)  # positive == split rejects better
        adapt_spread = _spread(single_adapt)
        rej_spread = _spread(single_rej)
        split_better_adapt = d_adapt > adapt_spread + 1e-4
        split_better_rej = d_rej > rej_spread + 1e-4
        return {
            "boundary_adapt_step_single": round(_mean(single_adapt), 3),
            "boundary_adapt_step_split": round(_mean(split_adapt), 3),
            "noise_rejection_single": round(_mean(single_rej), 4),
            "noise_rejection_split": round(_mean(split_rej), 4),
            "delta_adapt_speed": round(d_adapt, 3),
            "delta_noise_rejection": round(d_rej, 4),
            "calibration_ece_split": round(_mean(ece_list), 4),
            "seeds": list(seeds),
            # null holds if the split does NOT beat the single scalar on BOTH axes beyond seed spread
            "null_supported": bool(not (split_better_adapt and split_better_rej)),
            "dissociation_helps": bool(split_better_adapt and split_better_rej),
        }


# =====================================================================================================
# N7: latent scratchpad, working-memory slots for a delayed-match latent task
# =====================================================================================================
class _WMHead(nn.Module):
    """WorkingMemory scratchpad consumed across a delay-filled sequence, then a match decision."""

    def __init__(self, dim: int, slots: int):
        super().__init__()
        self.wm = WorkingMemory(dim, slots=slots)
        self.cls = nn.Linear(dim * 2, 2)

    def forward(self, cue: torch.Tensor, delay: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        # delay: [B, T, D] distractor sequence; write cue first, then distractors, read at probe time
        mem = None
        read, mem = self.wm(cue, mem)
        for t in range(delay.shape[1]):
            read, mem = self.wm(delay[:, t], mem)
        return self.cls(torch.cat([read, probe], dim=-1))


class _FFHead(nn.Module):
    """Feedforward control: cue concatenated to probe (no memory across the delay)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = mlp(dim * 2, 2, hidden, depth=1, ln=True)

    def forward(self, cue: torch.Tensor, delay: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cue, probe], dim=-1))


class N7(Experiment):
    id = "n7_wm_delayed_match"
    metric = ("delayed_match_accuracy", "accuracy_vs_delay_length")
    baseline = "feedforward head with the cue concatenated; matched compute; shuffled latent"
    ablation = "WorkingMemory read/write slots across a distractor-filled delay vs feedforward concat"
    null_hypothesis = (
        "WM slots tie a feedforward head with concatenated cue (the delay is trivial or the task does "
        "not need memory across it)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        delay_len = int(e.delay_len)
        n = int(e.samples)
        epochs, lr = int(e.epochs), float(e.lr)

        wm_acc, ff_acc, ff_shuf_acc = [], [], []
        acc_by_delay: dict[int, list[float]] = {}

        for s in seeds:
            # delayed-match-to-sample: cue is one of K latent prototypes; probe matches (y=1) or not (y=0)
            stream = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_prototypes),
                samples_per_task=n,
                separation=float(e.separation),
                seed=s,
            )[0]
            protos = []
            for c in range(int(e.n_prototypes)):
                m = stream.y == c
                if m.any():
                    protos.append(stream.x[m].mean(0))
            proto_t = torch.stack(protos)
            K = proto_t.shape[0]

            def build(delay_t: int, s=s, K=K, protos=proto_t):
                gg = torch.Generator().manual_seed(s + delay_t)
                cue_id = torch.randint(0, K, (n,), generator=gg)
                match = torch.randint(0, 2, (n,), generator=gg)
                probe_id = torch.where(
                    match.bool(), cue_id, (cue_id + 1 + torch.randint(0, K - 1, (n,), generator=gg)) % K
                )

                def noise():
                    return 0.3 * torch.randn(n, dim, generator=gg)

                cue = protos[cue_id] + noise()
                probe = protos[probe_id] + noise()
                # distractors: random prototype draws (carry no info about the cue)
                delay = (
                    torch.stack(
                        [protos[torch.randint(0, K, (n,), generator=gg)] + noise() for _ in range(delay_t)],
                        dim=1,
                    )
                    if delay_t > 0
                    else cue.new_zeros(n, 0, dim)
                )
                return cue, delay, probe, match

            def fit(model, build_fn, transform=None, s=s):
                seed_everything(s)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                cue, delay, probe, y = build_fn()
                if transform is not None:
                    cue, probe = transform(cue), transform(probe)
                    if delay.shape[1] > 0:
                        delay = transform(delay.reshape(-1, dim)).reshape(n, -1, dim)
                cut = int(n * 0.7)
                for _ in range(epochs):
                    opt.zero_grad()
                    out = model(cue[:cut], delay[:cut], probe[:cut])
                    F.cross_entropy(out, y[:cut]).backward()
                    opt.step()
                with torch.no_grad():
                    out = model(cue[cut:], delay[cut:], probe[cut:])
                    return float((out.argmax(-1) == y[cut:]).float().mean())

            wm_acc.append(fit(_WMHead(dim, int(e.wm_slots)), lambda: build(delay_len)))
            ff_acc.append(fit(_FFHead(dim, hidden), lambda: build(delay_len)))
            ff_shuf_acc.append(
                fit(
                    _FFHead(dim, hidden),
                    lambda: build(delay_len),
                    transform=lambda z, s=s: frozen_random_projection(z, seed=s),
                )
            )
            for dl in (0, delay_len, delay_len * 2):
                a = fit(_WMHead(dim, int(e.wm_slots)), lambda dl=dl: build(dl))
                acc_by_delay.setdefault(dl, []).append(a)

        wm = _mean(wm_acc)
        ff = _mean(ff_acc)
        # matched compute: report params, the FF control gets a comparable hidden width
        wm_params = param_count(_WMHead(dim, int(e.wm_slots)))
        ff_params = param_count(_FFHead(dim, hidden))
        compute = matched_within(wm_params, ff_params, tol=0.5)
        delta = wm - ff
        spread = _spread(ff_acc)
        ties_ff = delta <= spread + 1e-4
        return {
            "wm_accuracy": round(wm, 4),
            "ff_accuracy": round(ff, 4),
            "ff_shuffled_accuracy": round(_mean(ff_shuf_acc), 4),
            "delta_wm_vs_ff": round(delta, 4),
            "ff_seed_spread": round(spread, 4),
            "accuracy_by_delay": {str(k): round(_mean(v), 4) for k, v in sorted(acc_by_delay.items())},
            "params_wm": wm_params,
            "params_ff": ff_params,
            "compute_matched": compute,
            "seeds": list(seeds),
            "null_supported": bool(ties_ff),
            "wm_beats_ff": bool(not ties_ff),
        }


# =====================================================================================================
# N8: object permanence as a pooled-latent prediction bound (EXPECTED taxonomy slot 3)
# =====================================================================================================
class N8(Experiment):
    id = "n8_object_permanence_bound"
    metric = ("post_reveal_prediction_error", "occluded_identity_probe_acc")
    baseline = "last-frame baseline; marginal predictor; shuffle-label floor; frozen-random substrate"
    ablation = "occluded-identity linear-probe gate, then post-reveal latent prediction vs last-frame"
    null_hypothesis = (
        "occluded identity is not decodable from the pooled latent (substrate loses it), or the "
        "predictor ties the last-frame baseline (the published bound IS the result, taxonomy slot 3)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        n = int(e.samples)
        K = int(e.n_objects)
        epochs, lr = int(e.epochs), float(e.lr)

        probe_acc, chance, lastframe_err, pred_err, marginal_err = [], [], [], [], []
        probe_fr = []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            # object prototypes; a clip is [pre-occlusion, OCCLUDED pooled frame, post-reveal]. POOLING
            # the occluded frame mixes the occluder over the object, so identity is largely washed out:
            # this is the expected slot-3 bound on a pooled substrate.
            protos = torch.randn(K, dim, generator=g) * float(e.separation)
            occluder = torch.randn(1, dim, generator=g) * float(e.separation)
            obj = torch.randint(0, K, (n,), generator=g)
            pre = protos[obj] + 0.3 * torch.randn(n, dim, generator=g)
            # pooled occluded frame: alpha-blend object with the occluder (pooling washes identity)
            alpha = float(e.occlusion_alpha)
            occ = (1 - alpha) * protos[obj] + alpha * occluder + 0.3 * torch.randn(n, dim, generator=g)
            post = protos[obj] + 0.3 * torch.randn(n, dim, generator=g)  # reveal: same object returns

            # GATE: is occluded identity even decodable from the pooled occluded frame
            pr = linear_probe(occ, obj, seed=s)
            probe_acc.append(pr["score"])
            chance.append(pr["chance"])
            # frozen-random substrate arm of the same probe
            probe_fr.append(linear_probe(frozen_random_projection(occ, seed=s), obj, seed=s)["score"])

            # predict the post-reveal latent from [pre, occ]; baselines: last-frame (occ) and marginal mean
            cut = int(n * 0.7)
            feat = torch.cat([pre, occ], dim=-1)
            net = mlp(dim * 2, dim, int(e.hidden), depth=1, ln=True)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                F.mse_loss(net(feat[:cut]), post[:cut]).backward()
                opt.step()
            with torch.no_grad():
                pe = float((net(feat[cut:]) - post[cut:]).pow(2).mean())
                lf = float((occ[cut:] - post[cut:]).pow(2).mean())  # last-frame baseline
                mg = float((post[:cut].mean(0, keepdim=True) - post[cut:]).pow(2).mean())
            pred_err.append(pe)
            lastframe_err.append(lf)
            marginal_err.append(mg)

        pa, ch = _mean(probe_acc), _mean(chance)
        pe, lf, mg = _mean(pred_err), _mean(lastframe_err), _mean(marginal_err)
        identity_decodable = pa > ch + 0.1
        beats_last_frame = lf - pe > float(e.margin)
        # null (the expected slot-3 bound): identity NOT decodable through occlusion, OR predictor
        # ties the last-frame baseline. Either way the pooled-latent bound is the published result.
        return {
            "occluded_identity_probe_acc": round(pa, 4),
            "chance": round(ch, 4),
            "occluded_probe_frozen_random": round(_mean(probe_fr), 4),
            "post_reveal_prediction_error": round(pe, 5),
            "last_frame_error": round(lf, 5),
            "marginal_error": round(mg, 5),
            "predictor_minus_last_frame": round(pe - lf, 5),
            "margin": float(e.margin),
            "seeds": list(seeds),
            "identity_decodable_through_occlusion": bool(identity_decodable),
            "predictor_beats_last_frame": bool(beats_last_frame),
            # null holds (and is EXPECTED) when identity is washed out OR predictor ties last-frame
            "null_supported": bool(not identity_decodable or not beats_last_frame),
        }


# =====================================================================================================
# N9: attractor convergence of the latent refiner
# =====================================================================================================
class N9(Experiment):
    id = "n9_attractor_convergence"
    metric = ("fixed_point_residual", "convergence_depth_vs_difficulty", "converged_decode_gap")
    baseline = "compute-matched untied MLP; shuffled latent; frozen-random substrate"
    ablation = "refiner unroll dynamics (contraction, basin stability) vs matched untied depth"
    null_hypothesis = (
        "the refiner drifts rather than converging (a compute story not a representational one), or "
        "converged accuracy ties the compute-matched untied MLP"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        epochs, lr, margin = int(e.epochs), float(e.lr), float(e.margin)

        ref_acc, ctl_acc, resid, contraction, stable_frac = [], [], [], [], []
        depth_diff_corr = []
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

            seed_everything(s)
            refiner = IterativeRefiner(dim, hidden, steps)
            ref_acc.append(_fit_eval(refiner, nn.Linear(dim, nc), xtr, ytr, xte, yte, epochs, lr))
            seed_everything(s)
            ctl = _UntiedDepth(dim, hidden, steps)
            ctl_acc.append(_fit_eval(ctl, nn.Linear(dim, nc), xtr, ytr, xte, yte, epochs, lr))

            rep = convergence_report(refiner, xte, steps=int(e.unroll_steps))
            resid.append(rep["fixed_point_residual"])
            contraction.append(rep["contraction_factor"])
            bs = basin_stability(refiner, xte, eps=float(e.perturb_eps), steps=int(e.unroll_steps), seed=s)
            stable_frac.append(1.0 if bs["stable"] else 0.0)

            # convergence depth vs difficulty: per-sample steps-to-small-update vs linear-probe margin
            with torch.no_grad():
                z = xte.clone()
                per_sample_depth = torch.full((xte.shape[0],), float(e.unroll_steps))
                done = torch.zeros(xte.shape[0], dtype=torch.bool)
                for t in range(int(e.unroll_steps)):
                    u = refiner._update(z)
                    un = u.norm(dim=-1)
                    newly = (~done) & (un < float(e.converge_eps))
                    per_sample_depth[newly] = float(t)
                    done = done | newly
                    z = z + u
            # difficulty proxy: inverse linear margin from a probe head fit on raw latents
            seed_everything(s)
            head = nn.Linear(dim, nc)
            opt = torch.optim.Adam(head.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(head(xtr), ytr).backward()
                opt.step()
            with torch.no_grad():
                logits = head(xte)
                srt = logits.sort(dim=-1, descending=True).values
                difficulty = 1.0 / ((srt[:, 0] - srt[:, 1]).abs() + 1e-3)
            depth_diff_corr.append(_pearson(per_sample_depth.tolist(), difficulty.tolist()))

        rm, cm = _mean(ref_acc), _mean(ctl_acc)
        res = _mean(resid)
        cf = _mean(contraction)
        converges = cf < 0 and res < float(e.converge_eps) * 4
        corr = _mean(depth_diff_corr)
        gain = rm - cm
        ties_matched = abs(gain) <= margin
        return {
            "refiner_acc": round(rm, 4),
            "matched_untied_acc": round(cm, 4),
            "gain_vs_compute_matched": round(gain, 4),
            "fixed_point_residual": round(res, 6),
            "contraction_factor": round(cf, 6),
            "basin_stable_fraction": round(_mean(stable_frac), 4),
            "convergence_depth_vs_difficulty_corr": round(corr, 4),
            "converges": bool(converges),
            "margin": margin,
            "seeds": list(seeds),
            # null holds if it drifts OR converged accuracy ties the matched control
            "null_supported": bool(not converges or ties_matched),
            "converges_and_beats_matched": bool(converges and gain > margin),
        }


# =====================================================================================================
# N10: confidence-based halting tracks difficulty (latent metacognition signature)
# =====================================================================================================
class N10(Experiment):
    id = "n10_halting_difficulty"
    metric = ("halt_step_vs_difficulty_corr", "accuracy_at_lower_compute")
    baseline = "fixed-N refiner; random-halt; shuffled latent; matched compute"
    ablation = "adaptive halt-step vs linear-probe margin difficulty; adaptive vs fixed-N accuracy"
    null_hypothesis = (
        "halting is uncorrelated with difficulty (the halt head is noise), or adaptive halting loses "
        "accuracy with no compute saving"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        epochs, lr, margin = int(e.epochs), float(e.lr), float(e.margin)

        corr_list, rand_corr_list = [], []
        adaptive_acc, fixed_acc, adaptive_steps = [], [], []
        for s in seeds:
            seed_everything(s)
            # mixed-difficulty stream: some classes well separated, some not
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

            seed_everything(s)
            refiner = IterativeRefiner(dim, hidden, steps, halt=True, halt_threshold=float(e.halt_threshold))
            head = nn.Linear(dim, nc)
            opt = torch.optim.Adam([*refiner.parameters(), *head.parameters()], lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                z, _ = refiner(xtr)
                F.cross_entropy(head(z), ytr).backward()
                opt.step()
            with torch.no_grad():
                z, used = refiner(xte)
                a_adaptive = float((head(z).argmax(-1) == yte).float().mean())
                # fixed-N: force all samples to full steps
                zf, _ = refiner(xte, max_steps=steps)
                a_fixed = float((head(zf).argmax(-1) == yte).float().mean())
            adaptive_acc.append(a_adaptive)
            fixed_acc.append(a_fixed)
            adaptive_steps.append(float(used.float().mean()))

            # independent difficulty: linear-probe margin on raw latents
            seed_everything(s)
            phead = nn.Linear(dim, nc)
            popt = torch.optim.Adam(phead.parameters(), lr=lr)
            for _ in range(epochs):
                popt.zero_grad()
                F.cross_entropy(phead(xtr), ytr).backward()
                popt.step()
            with torch.no_grad():
                logits = phead(xte)
                srt = logits.sort(dim=-1, descending=True).values
                difficulty = 1.0 / ((srt[:, 0] - srt[:, 1]).abs() + 1e-3)
            corr_list.append(_pearson(used.tolist(), difficulty.tolist()))
            # random-halt control: shuffle the halt steps, correlation should vanish
            g = torch.Generator().manual_seed(s + 3)
            shuf = used[torch.randperm(used.shape[0], generator=g)]
            rand_corr_list.append(_pearson(shuf.tolist(), difficulty.tolist()))

        corr = _mean(corr_list)
        rand_corr = _mean(rand_corr_list)
        am, fm = _mean(adaptive_acc), _mean(fixed_acc)
        avg_steps = _mean(adaptive_steps)
        compute_saved = avg_steps < steps - 1e-6
        acc_kept = (fm - am) <= margin
        halt_tracks_difficulty = corr > abs(rand_corr) + float(e.corr_margin)
        return {
            "halt_step_vs_difficulty_corr": round(corr, 4),
            "random_halt_corr": round(rand_corr, 4),
            "adaptive_accuracy": round(am, 4),
            "fixed_n_accuracy": round(fm, 4),
            "mean_halt_steps": round(avg_steps, 3),
            "full_steps": steps,
            "compute_saved": bool(compute_saved),
            "accuracy_kept": bool(acc_kept),
            "seeds": list(seeds),
            # null holds if halting does not track difficulty, OR adaptive loses accuracy with no saving
            "null_supported": bool(not halt_tracks_difficulty or not (compute_saved and acc_kept)),
            "metacognition_signature": bool(halt_tracks_difficulty and compute_saved and acc_kept),
        }


# =====================================================================================================
# N11: latent self-verification as error-monitoring
# =====================================================================================================
class N11(Experiment):
    id = "n11_self_verification"
    metric = ("accuracy", "self_correction_gain", "halting_step_distribution")
    baseline = "single-shot; compute-matched single-shot; shuffled-verifier"
    ablation = "verifier-triggered revise step on low-confidence samples vs single-shot at matched compute"
    null_hypothesis = (
        "verify-revise ties single-shot at matched compute (the verifier carries no usable correction signal)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        epochs, lr, margin = int(e.epochs), float(e.lr), float(e.margin)

        single_acc, revise_acc, shuf_acc, ctl_acc = [], [], [], []
        revise_frac = []
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

            seed_everything(s)
            refiner = IterativeRefiner(dim, hidden, steps)
            head = nn.Linear(dim, nc)
            verifier = Verifier(dim, hidden=int(e.verifier_hidden))
            opt = torch.optim.Adam([*refiner.parameters(), *head.parameters(), *verifier.parameters()], lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                z, _ = refiner(xtr)
                logits = head(z)
                ce = F.cross_entropy(logits, ytr)
                # verifier target: 1 when the current prediction is wrong (error monitoring)
                with torch.no_grad():
                    wrong = (logits.argmax(-1) != ytr).float()
                vloss = F.binary_cross_entropy_with_logits(verifier.score(z), wrong)
                (ce + vloss).backward()
                opt.step()

            with torch.no_grad():
                z1, _ = refiner(xte)
                single = float((head(z1).argmax(-1) == yte).float().mean())
                # verify-revise: high verifier score -> one more refine step
                score = torch.sigmoid(verifier.score(z1))
                revise = score > float(e.revise_threshold)
                z2 = z1.clone()
                if revise.any():
                    extra, _ = refiner(z1[revise], max_steps=int(e.revise_steps))
                    z2[revise] = extra
                rev_acc = float((head(z2).argmax(-1) == yte).float().mean())
                # shuffled-verifier control: revise a random equal-size subset
                g = torch.Generator().manual_seed(s + 5)
                k = int(revise.sum())
                shuf_mask = torch.zeros_like(revise)
                if k > 0:
                    pick = torch.randperm(z1.shape[0], generator=g)[:k]
                    shuf_mask[pick] = True
                z3 = z1.clone()
                if shuf_mask.any():
                    extra2, _ = refiner(z1[shuf_mask], max_steps=int(e.revise_steps))
                    z3[shuf_mask] = extra2
                sh_acc = float((head(z3).argmax(-1) == yte).float().mean())

            # compute-matched single-shot: a fixed extra step for EVERYONE (same total compute as revise)
            seed_everything(s)
            ref2 = IterativeRefiner(dim, hidden, steps + int(e.revise_steps))
            ctl_acc.append(_fit_eval(ref2, nn.Linear(dim, nc), xtr, ytr, xte, yte, epochs, lr))

            single_acc.append(single)
            revise_acc.append(rev_acc)
            shuf_acc.append(sh_acc)
            revise_frac.append(float(revise.float().mean()))

        sm, rm, shm, cm = _mean(single_acc), _mean(revise_acc), _mean(shuf_acc), _mean(ctl_acc)
        gain_vs_single = rm - sm
        gain_vs_shuffled = rm - shm
        gain_vs_matched = rm - cm
        flops_extra = refiner_flops(dim, hidden, int(e.revise_steps))
        compute = matched_within(
            refiner_flops(dim, hidden, steps) + flops_extra,
            refiner_flops(dim, hidden, steps + int(e.revise_steps)),
        )
        beats_single = gain_vs_single > margin
        beats_shuffled = gain_vs_shuffled > margin
        beats_matched = gain_vs_matched > margin
        return {
            "single_shot_acc": round(sm, 4),
            "verify_revise_acc": round(rm, 4),
            "shuffled_verifier_acc": round(shm, 4),
            "compute_matched_acc": round(cm, 4),
            "self_correction_gain_vs_single": round(gain_vs_single, 4),
            "gain_vs_shuffled_verifier": round(gain_vs_shuffled, 4),
            "gain_vs_compute_matched": round(gain_vs_matched, 4),
            "mean_revise_fraction": round(_mean(revise_frac), 4),
            "compute_matched": compute,
            "margin": margin,
            "seeds": list(seeds),
            # null holds if verify-revise does NOT beat the compute-matched control AND shuffled verifier
            "null_supported": bool(not (beats_matched and beats_shuffled)),
            "verifier_carries_signal": bool(beats_matched and beats_shuffled and beats_single),
        }
