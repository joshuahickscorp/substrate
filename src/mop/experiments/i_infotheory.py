"""Series I: information theory on the frozen pooled latent. Nine cpu-now toy experiments, each with an
explicit NULL and a wired standing control (frozen-random arm, matched compute, or a tuned baseline).
All run on cached pooled latents now; no dense latents, no environment.

I1  information-bottleneck capability-per-bit (width sweep, real vs frozen-random).
I2  two-part MDL model selection vs held-out next-latent loss across a capacity sweep.
I3  compression damages iterative latent reasoning more than shallow classification.
I4i redundancy-reduction (whitening) recode improves continual retention per byte.
I5  rate-distortion frontier of the replay buffer, importance-weighted vs uniform bit allocation.
I6  mutual-information predictivity audit (information available vs information exploited).
I7  predictive-information past-future bottleneck on forward-dynamics streams.
I8  quantization-robustness as a representation-quality signal (real vs frozen-random).
I9  VQ codebook rate-distortion-usability (learned VQ vs k-means vs random codebook).

Doctrine: every experiment declares an explicit NULL; the linear-probe gate precedes any mechanism claim;
pooled latents discard within-frame structure (the published bound IS the result); honest nulls only, no
tuning toward a positive. No em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.bottleneck import capability_per_bit, quantization_robustness
from ..diagnostics.compute import matched_within, param_count
from ..diagnostics.geometry import effective_rank
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.substrate_ablation import frozen_random_projection, quantize_dequantize
from ..seeding import seed_everything
from ..shell.predictor import mlp
from ..shell.refine import IterativeRefiner
from .base import Experiment


# ----------------------------------------------------------------------------------------------------
# shared toy helpers
# ----------------------------------------------------------------------------------------------------
def _mean(v: list[float]) -> float:
    return sum(v) / max(1, len(v))


def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means; returns (hard codes, centroids). The VQ assignment in the no-decoder setting."""
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
    return codes, cent


def _purity(codes: torch.Tensor, y: torch.Tensor, k: int) -> float:
    correct = 0
    for c in range(k):
        m = codes == c
        if m.any():
            correct += int(torch.bincount(y[m]).max())
    return correct / int(y.shape[0])


def _split(x: torch.Tensor, y: torch.Tensor, frac: float = 0.7):
    n = x.shape[0]
    cut = int(n * frac)
    return x[:cut], y[:cut], x[cut:], y[cut:]


# ----------------------------------------------------------------------------------------------------
# I1: information-bottleneck capability-per-bit
# ----------------------------------------------------------------------------------------------------
class I1(Experiment):
    """Squeeze the frozen latent through a width-W linear bottleneck before a probe; sweep W. Does the
    REAL encoder pack the label into fewer bits than a frozen-random projection (a sharp knee where real
    stays high while frozen-random has collapsed), or do the curves overlap (graceful degradation is a
    generic high-dimensional property, taxonomy 3)."""

    id = "i1_info_bottleneck"
    metric = ("probe_accuracy_vs_retained_bits", "knee_width_95pct", "max_small_width_separation")
    baseline = "frozen-random projection of equal width at every bottleneck width"
    ablation = "linear bottleneck width sweep on real vs frozen-random substrate"
    null_hypothesis = (
        "accuracy vs retained bits is flat down to a tiny width (redundant) OR the real curve is "
        "indistinguishable from a frozen-random projection of equal width (the bottleneck is generic)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        widths = tuple(int(w) for w in e.widths)
        seps, knees = [], []
        acc_real_agg: dict[int, list] = {w: [] for w in widths}
        acc_fr_agg: dict[int, list] = {w: [] for w in widths}
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=int(e.dim),
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            rep = capability_per_bit(task.x, task.y, widths=widths, seed=s)
            seps.append(rep["max_small_width_separation"])
            knees.append(rep["knee_width_95pct"])
            for w in widths:
                acc_real_agg[w].append(rep["acc_real"][w])
                acc_fr_agg[w].append(rep["acc_frozen_random"][w])

        acc_real = {w: round(_mean(acc_real_agg[w]), 4) for w in widths}
        acc_fr = {w: round(_mean(acc_fr_agg[w]), 4) for w in widths}
        sep = _mean(seps)
        full = acc_real[max(widths)]
        small = acc_real[min(widths)]
        flat = bool(small >= 0.95 * full)  # the factor needs only a handful of bits
        out = {
            "widths": list(widths),
            "acc_real": acc_real,
            "acc_frozen_random": acc_fr,
            "knee_width_95pct": int(round(_mean([float(k) for k in knees]))),
            "max_small_width_separation": round(sep, 4),
            "curve_is_flat_to_tiny_width": flat,
            "real_packs_tighter": bool(sep > float(e.sep_margin)),
            "seeds": list(seeds),
            # null: flat down to tiny width OR real curve ties frozen-random at small widths
            "null_supported": bool(flat or sep <= float(e.sep_margin)),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# I2: two-part MDL model selection
# ----------------------------------------------------------------------------------------------------
class I2(Experiment):
    """For shells of varying capacity, compute a two-part MDL score = bits to encode quantized weights
    under a simple prior plus residual code length (the Gaussian negative log-likelihood of next-latent
    residuals on the cache). Does MDL rank capacities the same as held-out next-latent loss, giving an
    overfitting-proof criterion that needs no validation split."""

    id = "i2_mdl_selection"
    metric = ("mdl_vs_heldout_corr", "capacity_by_mdl", "capacity_by_val")
    baseline = "validation-split selection (held-out next-latent loss across the capacity sweep)"
    ablation = "two-part MDL (weight code + residual code) vs held-out loss ranking; vs param-count only"
    null_hypothesis = (
        "MDL ranks shells differently from held-out loss (rank correlation low) OR MDL is monotone in "
        "parameter count, so it adds nothing over counting weights"
    )
    tier = "cpu-now"

    def _fit(self, hidden: int, depth: int, xtr, ttr, epochs, lr, seed):
        seed_everything(seed)
        dim = xtr.shape[1]
        net = mlp(dim, dim, hidden, depth=depth)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            F.mse_loss(net(xtr), ttr).backward()
            opt.step()
        return net

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        rungs = [(int(h), int(d)) for h, d in e.rungs]
        wbits = float(e.weight_bits)
        mdl_curves, val_curves, pcounts = [], [], []
        corrs = []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=int(e.dim),
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                forward_dynamics=True,
                seed=s,
            )[0]
            x, t = task.x, task.xnext
            assert t is not None  # forward_dynamics=True guarantees the next-latent target
            xtr, ttr, xte, tte = _split(x, t, 0.7)
            mdls, vals, ps = [], [], []
            for h, d in rungs:
                net = self._fit(h, d, xtr, ttr, int(e.epochs), float(e.lr), s)
                with torch.no_grad():
                    res = net(xte) - tte
                    # residual code length in nats under a Gaussian head (per residual variance)
                    var = res.var().clamp_min(1e-6)
                    nll = 0.5 * (math.log(2 * math.pi * float(var)) + 1.0) * res.numel()
                    val = float((res**2).mean())
                p = param_count(net)
                weight_bits = p * wbits  # bits to encode quantized weights under a flat prior
                mdl = nll / math.log(2) + weight_bits  # total description length in bits
                mdls.append(mdl)
                vals.append(val)
                ps.append(p)
            mdl_curves.append(mdls)
            val_curves.append(vals)
            pcounts.append(ps)
            # rank correlation (Spearman) between MDL and held-out loss across rungs
            corrs.append(_spearman(mdls, vals))

        mdl_mean = [round(_mean([c[i] for c in mdl_curves]), 2) for i in range(len(rungs))]
        val_mean = [round(_mean([c[i] for c in val_curves]), 6) for i in range(len(rungs))]
        p_mean = [int(_mean([c[i] for c in pcounts])) for i in range(len(rungs))]
        corr = _mean(corrs)
        cap_mdl = int(min(range(len(rungs)), key=lambda i: mdl_mean[i]))
        cap_val = int(min(range(len(rungs)), key=lambda i: val_mean[i]))
        # MDL is "just param count" if argmin MDL equals argmin params AND MDL ranks monotone in weights
        cap_pcount = int(min(range(len(rungs)), key=lambda i: p_mean[i]))
        corr_param = _spearman(mdl_mean, [float(p) for p in p_mean])
        mdl_is_param_count = bool(cap_mdl == cap_pcount and corr_param > 0.99)
        out = {
            "rungs": [list(r) for r in rungs],
            "mdl_bits_mean": mdl_mean,
            "heldout_loss_mean": val_mean,
            "params_mean": p_mean,
            "mdl_vs_heldout_spearman": round(corr, 4),
            "capacity_by_mdl": cap_mdl,
            "capacity_by_val": cap_val,
            "mdl_param_count_spearman": round(corr_param, 4),
            "mdl_matches_val": bool(cap_mdl == cap_val),
            "seeds": list(seeds),
            # null: MDL disagrees with held-out loss (low corr) OR MDL is just param count
            "null_supported": bool(corr < float(e.corr_threshold) or mdl_is_param_count),
        }
        return out


def _spearman(a: list[float], b: list[float]) -> float:
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    ra = ta.argsort().argsort().to(torch.float64)
    rb = tb.argsort().argsort().to(torch.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = torch.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-12
    return float((ra * rb).sum() / denom)


# ----------------------------------------------------------------------------------------------------
# I3: compression damages reasoning more than shallow classification
# ----------------------------------------------------------------------------------------------------
class _UntiedDepth(nn.Module):
    """Compute-matched control for the refiner: `steps` untied residual blocks applied once each."""

    def __init__(self, dim: int, hidden: int, steps: int):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(steps)])
        self.blocks = nn.ModuleList([mlp(dim, dim, hidden, depth=1, ln=True) for _ in range(steps)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for norm, block in zip(self.norms, self.blocks, strict=True):
            z = z + block(norm(z))
        return z


class I3(Experiment):
    """Run two readouts off the same compressed latent at matched bit budgets, a shallow linear
    classification head and an IterativeRefiner (EX17) followed by a head. Sweep uniform k-bit
    quantization. Does compression degrade iterative latent reasoning more steeply than shallow
    classification (the sharp EX17 hypothesis), holding the bit budget identical."""

    id = "i3_compression_reasoning"
    metric = ("reasoning_minus_classification_slope", "bits_to_lose_refiner_advantage", "refiner_gain")
    baseline = "shallow linear classification head at the same bit budget; refiner vs untied-depth"
    ablation = "uniform k-bit quantization sweep applied to both readouts at matched bits"
    null_hypothesis = (
        "the two accuracy-vs-bits slopes are equal within seed spread (reasoning is not specially "
        "fragile) OR the refiner has no advantage to lose even uncompressed (EX17 null already holds)"
    )
    tier = "cpu-now"

    def _fit_eval(self, backbone, head, xtr, ytr, xte, yte, epochs, lr) -> float:
        params = [*head.parameters()] + ([*backbone.parameters()] if backbone is not None else [])
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            z = xtr if backbone is None else backbone(xtr)
            z = z[0] if isinstance(z, tuple) else z
            F.cross_entropy(head(z), ytr).backward()
            opt.step()
        with torch.no_grad():
            z = xte if backbone is None else backbone(xte)
            z = z[0] if isinstance(z, tuple) else z
            return float((head(z).argmax(-1) == yte).float().mean())

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        bits = [int(b) for b in e.bits]
        epochs, lr = int(e.epochs), float(e.lr)
        cls_curves, ref_curves, gap_curves = [], [], []
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
            nc = int(y.max()) + 1
            cls_row, ref_row, gap_row = [], [], []
            for b in bits:
                xq = quantize_dequantize(x, b)
                xtr, ytr, xte, yte = _split(xq, y, 0.7)
                seed_everything(s)
                clf = nn.Linear(dim, nc)
                a_cls = self._fit_eval(None, clf, xtr, ytr, xte, yte, epochs, lr)
                seed_everything(s)
                refiner = IterativeRefiner(dim, hidden, steps)
                rhead = nn.Linear(dim, nc)
                a_ref = self._fit_eval(refiner, rhead, xtr, ytr, xte, yte, epochs, lr)
                seed_everything(s)
                ctl = _UntiedDepth(dim, hidden, steps)
                chead = nn.Linear(dim, nc)
                a_ctl = self._fit_eval(ctl, chead, xtr, ytr, xte, yte, epochs, lr)
                cls_row.append(a_cls)
                ref_row.append(a_ref)
                gap_row.append(a_ref - a_ctl)  # refiner advantage over matched compute at this bit budget
            cls_curves.append(cls_row)
            ref_curves.append(ref_row)
            gap_curves.append(gap_row)

        cls_m = [round(_mean([c[i] for c in cls_curves]), 4) for i in range(len(bits))]
        ref_m = [round(_mean([c[i] for c in ref_curves]), 4) for i in range(len(bits))]
        gap_m = [round(_mean([c[i] for c in gap_curves]), 4) for i in range(len(bits))]
        # slope = accuracy gained per added bit (bits ascending)
        order = sorted(range(len(bits)), key=lambda i: bits[i])
        bs = [float(bits[i]) for i in order]
        cls_slope = _slope(bs, [cls_m[i] for i in order])
        ref_slope = _slope(bs, [ref_m[i] for i in order])
        diff = ref_slope - cls_slope  # positive == reasoning recovers faster with bits == more fragile
        # bit budget at which the refiner advantage drops below margin (scanning low to high bits)
        margin = float(e.margin)
        lose = next((bits[i] for i in order if gap_m[i] <= margin), max(bits))
        full_gap = gap_m[order[-1]]  # advantage at the highest bit budget (least compressed)
        no_advantage = bool(full_gap <= margin)
        out = {
            "bits": bits,
            "classification_acc": cls_m,
            "reasoning_acc": ref_m,
            "refiner_minus_control_gap": gap_m,
            "classification_slope_per_bit": round(cls_slope, 5),
            "reasoning_slope_per_bit": round(ref_slope, 5),
            "reasoning_minus_classification_slope": round(diff, 5),
            "bits_to_lose_refiner_advantage": int(lose),
            "refiner_gain_uncompressed": round(full_gap, 4),
            "compute_matched": matched_within(1, 1),
            "seeds": list(seeds),
            # null: equal slopes (reasoning not specially fragile) OR no refiner advantage to lose
            "null_supported": bool(abs(diff) <= float(e.slope_margin) or no_advantage),
        }
        return out


def _slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = _mean(xs), _mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n)) + 1e-12
    return num / den


# ----------------------------------------------------------------------------------------------------
# I4i: redundancy-reduction recode improves continual retention per byte
# ----------------------------------------------------------------------------------------------------
def _zca_whiten(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """ZCA whitening: decorrelate and unit-scale the latent (Barlow-style redundancy reduction)."""
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig, vec = torch.linalg.eigh(cov)
    w = vec @ torch.diag((eig.clamp_min(eps)) ** -0.5) @ vec.T
    return xc @ w


class I4i(Experiment):
    """Learn a redundancy-reduction recode of cached latents (ZCA whitening) before the E2 replay shell;
    optionally truncate to a byte budget. Does decorrelating or whitening the frozen latent improve
    backward transfer per stored byte, or is the redundancy in the pooled latent already benign. A
    frozen-random arm tests whether any gain is a generic whitening effect, not a V-JEPA property."""

    id = "i4i_redundancy_reduction"
    metric = ("backward_transfer", "retention_per_byte", "offdiag_correlation_reduction")
    baseline = "raw-latent replay at matched stored-byte budget"
    ablation = "whitened vs raw vs frozen-random replay; effective_rank and off-diagonal correlation"
    null_hypothesis = (
        "redundancy reduction ties raw-latent replay at matched bytes (the pooled latent redundancy "
        "is not costing retention) OR the same gain appears on a frozen-random projection (generic)"
    )
    tier = "cpu-now"

    def _continual_bwt(self, recode, tasks, dim, nc, epochs, lr, seed, buf_cap):
        from ..shell.buffer import ReplayBuffer

        seed_everything(seed)
        head = nn.Linear(dim, nc)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        buf = ReplayBuffer(buf_cap, dim, prioritized=False, seed=seed)
        acc_after_first = None
        for ti, task in enumerate(tasks):
            xt = recode(task.x)
            yt = task.y
            for _ in range(epochs):
                opt.zero_grad()
                xb, yb = xt, yt
                if len(buf) > 0:
                    s = buf.sample(min(len(buf), xt.shape[0]))
                    xb = torch.cat([xt, s["x"]])
                    yb = torch.cat([yt, s["y"]])
                F.cross_entropy(head(xb), yb).backward()
                opt.step()
            buf.add(xt, yt)
            if ti == 0:
                with torch.no_grad():
                    acc_after_first = float((head(xt).argmax(-1) == yt).float().mean())
        # backward transfer: task0 accuracy at the end minus right after task0
        x0 = recode(tasks[0].x)
        with torch.no_grad():
            acc_end = float((head(x0).argmax(-1) == tasks[0].y).float().mean())
        return acc_end - (acc_after_first or 0.0)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        raw_bwt, wht_bwt, fr_bwt = [], [], []
        raw_rank, wht_rank = [], []
        offdiag_raw, offdiag_wht = [], []
        for s in seeds:
            tasks = make_task_stream(
                n_tasks=int(e.n_tasks),
                dim=dim,
                classes_per_task=int(e.classes_per_task),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            nc = tasks[0].n_classes
            allx = torch.cat([t.x for t in tasks])

            # recodes applied per task (frozen, a one-time bought transform; ZCA centers internally)
            def white(z):
                return _zca_whiten(z)

            def fr(z, s=s):
                return frozen_random_projection(z, s)

            ep, lr, cap = int(e.epochs), float(e.lr), int(e.buffer_capacity)
            raw_bwt.append(self._continual_bwt(lambda z: z, tasks, dim, nc, ep, lr, s, cap))
            wht_bwt.append(self._continual_bwt(white, tasks, dim, nc, ep, lr, s, cap))
            fr_bwt.append(self._continual_bwt(fr, tasks, dim, nc, ep, lr, s, cap))
            raw_rank.append(effective_rank(allx))
            wht_rank.append(effective_rank(white(allx)))
            offdiag_raw.append(_offdiag_corr(allx))
            offdiag_wht.append(_offdiag_corr(white(allx)))

        rb, wb, fb = _mean(raw_bwt), _mean(wht_bwt), _mean(fr_bwt)
        gain = wb - rb
        fr_gain = fb - rb
        margin = float(e.bwt_margin)
        out = {
            "bwt_raw": round(rb, 4),
            "bwt_whitened": round(wb, 4),
            "bwt_frozen_random": round(fb, 4),
            "whitened_gain_vs_raw": round(gain, 4),
            "frozen_random_gain_vs_raw": round(fr_gain, 4),
            "effective_rank_raw": round(_mean(raw_rank), 4),
            "effective_rank_whitened": round(_mean(wht_rank), 4),
            "offdiag_corr_raw": round(_mean(offdiag_raw), 4),
            "offdiag_corr_whitened": round(_mean(offdiag_wht), 4),
            "offdiag_correlation_reduction": round(_mean(offdiag_raw) - _mean(offdiag_wht), 4),
            "seeds": list(seeds),
            # null: whitening ties raw (gain small) OR the gain is generic (frozen-random shows it too)
            "null_supported": bool(gain <= margin or abs(gain - fr_gain) <= margin),
        }
        return out


def _offdiag_corr(x: torch.Tensor) -> float:
    """Mean absolute off-diagonal feature correlation (the redundancy a whitening recode removes)."""
    xc = x - x.mean(0, keepdim=True)
    std = xc.std(0, keepdim=True).clamp_min(1e-6)
    c = (xc / std).T @ (xc / std) / max(1, x.shape[0] - 1)
    d = c.shape[0]
    off = c - torch.diag(torch.diag(c))
    return float(off.abs().sum() / max(1, d * d - d))


# ----------------------------------------------------------------------------------------------------
# I5: rate-distortion frontier of the replay buffer
# ----------------------------------------------------------------------------------------------------
class I5(Experiment):
    """Vary bits-per-stored-latent via uniform quantization, and add an importance-weighted allocation
    arm that gives more bits to higher-Fisher exemplars. Plot forgetting (distortion) against total
    stored bits (rate). Does non-uniform bit allocation beat uniform on the rate-distortion frontier,
    or do they lie on the same curve (the importance signal does not predict which exemplars need bits)."""

    id = "i5_rate_distortion_replay"
    metric = ("rate_distortion_curve", "bits_saved_at_fixed_bwt", "area_under_rd_curve")
    baseline = "uniform quantization at matched total stored bits"
    ablation = "importance-weighted (Fisher) bit allocation vs uniform across a bit-budget sweep"
    null_hypothesis = (
        "importance-weighted allocation lies on the same RD curve as uniform within seed spread (the "
        "importance signal does not predict which exemplars need precision) OR the curve is a flat cliff"
    )
    tier = "cpu-now"

    def _fisher_per_exemplar(self, head, x, y) -> torch.Tensor:
        """Per-exemplar Fisher proxy: squared gradient norm of the loss on each stored latent."""
        fish = torch.zeros(x.shape[0])
        for i in range(x.shape[0]):
            head.zero_grad(set_to_none=True)
            F.cross_entropy(head(x[i : i + 1]), y[i : i + 1]).backward()
            g2 = sum(float((p.grad**2).sum()) for p in head.parameters() if p.grad is not None)
            fish[i] = g2
        head.zero_grad(set_to_none=True)
        return fish

    def _bwt_at_alloc(self, tasks, bits_per_exemplar, dim, nc, epochs, lr, seed):
        """Train continually, storing each exemplar quantized to its allocated bit depth."""
        from ..shell.buffer import ReplayBuffer

        seed_everything(seed)
        head = nn.Linear(dim, nc)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        buf = ReplayBuffer(int(1e6), dim, prioritized=False, seed=seed)
        acc_first = None
        for ti, task in enumerate(tasks):
            xt, yt = task.x, task.y
            for _ in range(epochs):
                opt.zero_grad()
                xb, yb = xt, yt
                if len(buf) > 0:
                    s = buf.sample(min(len(buf), xt.shape[0]))
                    xb = torch.cat([xt, s["x"]])
                    yb = torch.cat([yt, s["y"]])
                F.cross_entropy(head(xb), yb).backward()
                opt.step()
            # quantize each exemplar to its allocated bit depth before storing
            if callable(bits_per_exemplar):
                bpe = bits_per_exemplar(head, xt, yt)
            else:
                bpe = torch.full((xt.shape[0],), float(bits_per_exemplar))
            xq = torch.stack([quantize_dequantize(xt[i : i + 1], int(bpe[i]))[0] for i in range(xt.shape[0])])
            buf.add(xq, yt)
            if ti == 0:
                with torch.no_grad():
                    acc_first = float((head(xt).argmax(-1) == yt).float().mean())
        with torch.no_grad():
            acc_end = float((head(tasks[0].x).argmax(-1) == tasks[0].y).float().mean())
        bwt = acc_end - (acc_first or 0.0)
        return bwt

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        budgets = [int(b) for b in e.uniform_bits]  # uniform bit depths (rate knob)
        lo, hi = int(e.alloc_low_bits), int(e.alloc_high_bits)
        uni_curves, imp_points = [], []
        saved = []
        for s in seeds:
            tasks = make_task_stream(
                n_tasks=int(e.n_tasks),
                dim=dim,
                classes_per_task=int(e.classes_per_task),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            nc = tasks[0].n_classes
            uni = []
            for b in budgets:
                bwt = self._bwt_at_alloc(tasks, b, dim, nc, int(e.epochs), float(e.lr), s)
                rate = b * dim  # total bits per stored exemplar
                uni.append((rate, bwt))
            uni_curves.append(uni)
            # importance-weighted allocation: high-Fisher exemplars get hi bits, rest get lo bits
            mean_bits = (lo + hi) / 2.0

            def alloc(head, x, y, lo=lo, hi=hi):
                f = self._fisher_per_exemplar(head, x, y)
                med = f.median()
                return torch.where(f >= med, torch.full_like(f, float(hi)), torch.full_like(f, float(lo)))

            imp_bwt = self._bwt_at_alloc(tasks, alloc, dim, nc, int(e.epochs), float(e.lr), s)
            imp_rate = mean_bits * dim
            imp_points.append((imp_rate, imp_bwt))
            # bits saved at fixed BWT: at the importance BWT, what uniform rate matches it
            uni_match = _rate_at_bwt(uni, imp_bwt)
            saved.append(uni_match - imp_rate)

        uni_mean = []
        for i in range(len(budgets)):
            rate_m = _mean([c[i][0] for c in uni_curves])
            bwt_m = _mean([c[i][1] for c in uni_curves])
            uni_mean.append((round(rate_m, 1), round(bwt_m, 4)))
        imp_rate = round(_mean([p[0] for p in imp_points]), 1)
        imp_bwt = round(_mean([p[1] for p in imp_points]), 4)
        bits_saved = _mean(saved)
        bwts = [p[1] for p in uni_mean]
        is_cliff = bool(max(bwts) - min(bwts) <= float(e.flat_margin))  # all-or-nothing, no graded frontier
        margin = float(e.saved_margin)
        out = {
            "uniform_rate_distortion": uni_mean,
            "importance_rate": imp_rate,
            "importance_bwt": imp_bwt,
            "bits_saved_at_fixed_bwt": round(bits_saved, 1),
            "area_under_rd_curve": round(_auc(uni_mean), 4),
            "rd_curve_is_flat_cliff": is_cliff,
            "seeds": list(seeds),
            # null: importance ties uniform (small bits saved) OR the RD curve is a flat cliff
            "null_supported": bool(bits_saved <= margin or is_cliff),
        }
        return out


def _rate_at_bwt(curve, target_bwt: float) -> float:
    """The uniform rate whose BWT is closest to target (a crude inverse of the RD curve)."""
    return min(curve, key=lambda rb: abs(rb[1] - target_bwt))[0]


def _auc(curve) -> float:
    pts = sorted(curve, key=lambda rb: rb[0])
    area = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        area += 0.5 * (pts[i][1] + pts[i - 1][1]) * dx
    span = (pts[-1][0] - pts[0][0]) or 1.0
    return area / span


# ----------------------------------------------------------------------------------------------------
# I6: mutual-information predictivity audit
# ----------------------------------------------------------------------------------------------------
class I6(Experiment):
    """Estimate the MI the probe says is decodable (the EX12 ceiling) and the MI the trained shell
    actually uses (from its own outputs), via a plug-in estimator on discretized factors and a
    variational InfoNCE lower bound as a cross-check. Report the exploitation ratio = MI used / MI
    decodable per factor across capacity rungs, separating substrate-bound from shell-bound."""

    id = "i6_mi_audit"
    metric = ("exploitation_ratio", "mi_decodable", "mi_used")
    baseline = "EX12 probe ceiling (MI decodable) as the upper bound on usable information"
    ablation = "plug-in vs InfoNCE MI estimator agreement; exploitation ratio across capacity rungs"
    null_hypothesis = (
        "the exploitation ratio is near 1 for every decodable factor (the shell already uses essentially "
        "all available information, no headroom) OR the two MI estimators disagree so the ratio is moot"
    )
    tier = "cpu-now"

    def _plugin_mi(self, codes: torch.Tensor, y: torch.Tensor, k: int) -> float:
        """Plug-in MI between a discrete code and the label, in nats (empirical joint vs marginals)."""
        nc = int(y.max()) + 1
        joint = torch.zeros(k, nc)
        for c, yy in zip(codes.tolist(), y.tolist(), strict=True):
            joint[c, yy] += 1
        joint = joint / joint.sum().clamp_min(1)
        pc = joint.sum(1, keepdim=True)
        py = joint.sum(0, keepdim=True)
        denom = (pc * py).clamp_min(1e-12)
        nz = joint > 0
        return float((joint[nz] * (joint[nz] / denom[nz]).log()).sum())

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        hiddens = [int(h) for h in e.capacity_rungs]
        ratios_by_rung: dict[int, list] = {h: [] for h in hiddens}
        dec_list, used_list = [], []
        nce_list, plug_list = [], []
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
            xtr, ytr, xte, yte = _split(x, y, 0.7)
            nc = int(y.max()) + 1
            # MI decodable ceiling: discretize the probe-predicted label, plug-in MI with truth
            probe_acc = linear_probe(x, y, seed=s)["score"]
            # cap label entropy by the empirical class distribution
            h_y = _entropy(y, nc)
            mi_dec = max(0.0, h_y * (probe_acc - 1.0 / nc) / (1.0 - 1.0 / nc + 1e-9))  # acc-scaled ceiling
            for h in hiddens:
                seed_everything(s)
                net = mlp(dim, nc, h, depth=1)
                opt = torch.optim.Adam(net.parameters(), lr=float(e.lr))
                for _ in range(int(e.epochs)):
                    opt.zero_grad()
                    F.cross_entropy(net(xtr), ytr).backward()
                    opt.step()
                with torch.no_grad():
                    pred = net(xte).argmax(-1)
                mi_used = self._plugin_mi(pred, yte, nc)
                # InfoNCE cross-check on the shell logits (a variational MI lower bound proxy)
                with torch.no_grad():
                    logits = net(xte)
                mi_nce = _infonce_mi(logits, yte)
                ratio = mi_used / (mi_dec + 1e-9)
                ratios_by_rung[h].append(min(ratio, 2.0))
                if h == hiddens[-1]:
                    dec_list.append(mi_dec)
                    used_list.append(mi_used)
                    plug_list.append(mi_used)
                    nce_list.append(mi_nce)

        ratio_curve = {h: round(_mean(ratios_by_rung[h]), 4) for h in hiddens}
        mi_dec = _mean(dec_list)
        mi_used = _mean(used_list)
        full_ratio = mi_used / (mi_dec + 1e-9)
        # estimator agreement: plug-in vs InfoNCE rank/level agreement at the top rung
        est_corr = _spearman(plug_list, nce_list) if len(plug_list) > 1 else 1.0
        estimators_disagree = bool(abs(_mean(plug_list) - _mean(nce_list)) > float(e.estimator_tol))
        near_one = bool(full_ratio >= float(e.ratio_threshold))
        out = {
            "mi_decodable": round(mi_dec, 4),
            "mi_used": round(mi_used, 4),
            "exploitation_ratio": round(full_ratio, 4),
            "ratio_by_capacity_rung": ratio_curve,
            "mi_plugin_mean": round(_mean(plug_list), 4),
            "mi_infonce_mean": round(_mean(nce_list), 4),
            "estimator_spearman": round(est_corr, 4),
            "seeds": list(seeds),
            # null: ratio near 1 (no headroom) OR the estimators disagree (ratio uninterpretable)
            "null_supported": bool(near_one or estimators_disagree),
        }
        return out


def _entropy(y: torch.Tensor, nc: int) -> float:
    p = torch.bincount(y, minlength=nc).float()
    p = p / p.sum().clamp_min(1)
    nz = p > 0
    return float(-(p[nz] * p[nz].log()).sum())


def _infonce_mi(logits: torch.Tensor, y: torch.Tensor) -> float:
    """A variational MI lower bound proxy from a trained classifier: H(Y) minus the cross-entropy gap."""
    nc = logits.shape[1]
    h_y = _entropy(y, nc)
    ce = float(F.cross_entropy(logits, y))
    return max(0.0, h_y - ce)


# ----------------------------------------------------------------------------------------------------
# I7: predictive information past-future bottleneck
# ----------------------------------------------------------------------------------------------------
class I7(Experiment):
    """On forward-dynamics streams with next-latent targets, train a recode that keeps the predictive
    component (a past-future bottleneck) by regressing the next latent through a width-W code. Compare
    the predictive-only code against the raw latent and a frozen-random projection on next-latent
    prediction (rollout R2). How much of the pooled latent is predictive vs instantaneous."""

    id = "i7_predictive_information"
    metric = ("predictive_information_fraction", "rollout_r2_code_vs_raw", "next_latent_prediction_error")
    baseline = "raw latent and a frozen-random projection at matched code dimension"
    ablation = "predictive-only code (past-future bottleneck) vs raw vs frozen-random on rollout R2"
    null_hypothesis = (
        "the predictive-only code does no better than the raw latent on rollout R2 at matched dimension "
        "(predictive and instantaneous information are entangled) OR predictive information is near zero"
    )
    tier = "cpu-now"

    def _r2(self, pred, target) -> float:
        ss_res = ((pred - target) ** 2).sum()
        ss_tot = ((target - target.mean(0)) ** 2).sum() + 1e-8
        return float(1 - ss_res / ss_tot)

    def _fit_predict(self, feat_tr, t_tr, feat_te, t_te, epochs, lr, seed):
        seed_everything(seed)
        net = nn.Linear(feat_tr.shape[1], t_tr.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            F.mse_loss(net(feat_tr), t_tr).backward()
            opt.step()
        with torch.no_grad():
            return net(feat_te)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, w = int(e.dim), int(e.code_width)
        raw_r2, code_r2, fr_r2, pfrac = [], [], [], []
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
            x, xn = task.x, task.xnext
            assert xn is not None  # forward_dynamics=True guarantees the next-latent target
            xtr, ntr, xte, nte = _split(x, xn, 0.7)
            # predictive-only code: width-W bottleneck trained to predict the FUTURE latent
            seed_everything(s)
            enc = nn.Linear(dim, w)
            dec = nn.Linear(w, dim)
            opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=float(e.lr))
            for _ in range(int(e.epochs)):
                opt.zero_grad()
                F.mse_loss(dec(enc(xtr)), ntr).backward()
                opt.step()
            with torch.no_grad():
                code_pred = dec(enc(xte))
            code_r2.append(self._r2(code_pred, nte))
            # raw latent at full dim, and frozen-random projection, predicting the future
            raw_pred = self._fit_predict(xtr, ntr, xte, nte, int(e.epochs), float(e.lr), s)
            raw_r2.append(self._r2(raw_pred, nte))
            xfr = frozen_random_projection(x, s)
            fxtr, fxte = xfr[: xtr.shape[0]], xfr[xtr.shape[0] :]
            fr_pred = self._fit_predict(fxtr, ntr, fxte, nte, int(e.epochs), float(e.lr), s)
            fr_r2.append(self._r2(fr_pred, nte))
            # predictive information fraction: I(code; future) / I(latent; future), proxied by R2 ratio
            pfrac.append(max(0.0, self._r2(code_pred, nte)) / (max(1e-6, self._r2(raw_pred, nte))))

        rr, cr, fr = _mean(raw_r2), _mean(code_r2), _mean(fr_r2)
        gain = cr - rr
        frac = _mean(pfrac)
        margin = float(e.r2_margin)
        out = {
            "rollout_r2_raw": round(rr, 4),
            "rollout_r2_code": round(cr, 4),
            "rollout_r2_frozen_random": round(fr, 4),
            "predictive_information_fraction": round(min(frac, 1.0), 4),
            "code_gain_vs_raw": round(gain, 4),
            "code_gain_vs_frozen_random": round(cr - fr, 4),
            "next_latent_prediction_error": round(1.0 - cr, 4),
            "code_width": w,
            "seeds": list(seeds),
            # null: code ties raw at matched dim (gain small) OR predictive information near zero
            "null_supported": bool(gain <= margin or rr <= float(e.zero_r2)),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# I8: quantization-robustness as a representation-quality signal
# ----------------------------------------------------------------------------------------------------
class I8(Experiment):
    """A degradation battery: per-feature uniform quantization at decreasing bit depths, applied to real
    and frozen-random latents, with probe-accuracy curves per substrate. Is graceful degradation a
    V-JEPA property specifically, or does any projection degrade identically (the D2 honest fact that
    linear decodability is projection-invariant, taxonomy 3)."""

    id = "i8_quant_robustness"
    metric = ("bits_to_half_accuracy", "degradation_curve_area", "real_minus_frozen_random")
    baseline = "frozen-random projection degradation curve at matched bit depths"
    ablation = "uniform k-bit quantization sweep on real vs frozen-random; per-substrate degradation"
    null_hypothesis = (
        "the real encoder degradation curve overlaps the frozen-random curve (robustness is a generic "
        "high-dimension property, not a V-JEPA property, consistent with D2 projection-invariance)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        bits = tuple(int(b) for b in e.bits)
        gaps, real_areas, fr_areas = [], [], []
        real_agg: dict[int, list] = {b: [] for b in bits}
        fr_agg: dict[int, list] = {b: [] for b in bits}
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=int(e.dim),
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            rep = quantization_robustness(task.x, task.y, bits=bits, seed=s)
            gaps.append(rep["mean_gap"])
            for b in bits:
                real_agg[b].append(rep["acc_real"][b])
                fr_agg[b].append(rep["acc_frozen_random"][b])
            bs = sorted(bits)
            real_areas.append(_auc([(float(b), _mean(real_agg[b][-1:])) for b in bs]))
            fr_areas.append(_auc([(float(b), _mean(fr_agg[b][-1:])) for b in bs]))

        acc_real = {b: round(_mean(real_agg[b]), 4) for b in bits}
        acc_fr = {b: round(_mean(fr_agg[b]), 4) for b in bits}
        gap = _mean(gaps)
        # bits to half accuracy on the real substrate (lowest bit depth keeping >= 0.5)
        order = sorted(bits)
        half = next((b for b in order if acc_real[b] >= 0.5), max(bits))
        margin = float(e.gap_margin)
        out = {
            "bits": list(bits),
            "acc_real": acc_real,
            "acc_frozen_random": acc_fr,
            "mean_gap_real_minus_frozen_random": round(gap, 4),
            "bits_to_half_accuracy_real": int(half),
            "degradation_curve_area_real": round(_mean(real_areas), 4),
            "degradation_curve_area_frozen_random": round(_mean(fr_areas), 4),
            "real_more_robust": bool(gap > margin),
            "seeds": list(seeds),
            # null: real and frozen-random degradation curves overlap (gap small)
            "null_supported": bool(gap <= margin),
        }
        return out


# ----------------------------------------------------------------------------------------------------
# I9: VQ codebook rate-distortion-usability
# ----------------------------------------------------------------------------------------------------
class I9(Experiment):
    """Sweep codebook size (rate in bits per latent = log2 codes) for a learned VQ codebook, k-means,
    and a random codebook. For each rate measure distortion (reconstruction error) and usability
    (cluster purity). Does discrete abstraction trade bits for reusable structure better than k-means
    and a random codebook, or do they tie on usability-per-bit (the EX16 null in RD terms)."""

    id = "i9_vq_rate_distortion"
    metric = ("rate_distortion_curve", "usability_per_bit", "vq_minus_kmeans_margin")
    baseline = "k-means codebook and a random codebook at matched rate"
    ablation = "learned VQ vs k-means vs random codebook across codebook-size (rate) sweep"
    null_hypothesis = (
        "VQ ties k-means and random on usability-per-bit at matched rate (no decodable structure over "
        "raw latents, the EX16 null in rate-distortion terms) OR usability is flat in rate"
    )
    tier = "cpu-now"

    def _vq_fit(self, x, k, iters, lr, seed):
        """A learned VQ codebook: gradient-trained centroids with a straight-through commitment loss."""
        seed_everything(seed)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[:k]
        cb = nn.Parameter(x[idx].clone())
        opt = torch.optim.Adam([cb], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            d = torch.cdist(x, cb)
            codes = d.argmin(1)
            q = cb[codes]
            # commitment + codebook loss (the VQ-VAE objective without a pixel decoder)
            loss = F.mse_loss(q, x.detach()) + 0.25 * F.mse_loss(q.detach(), x)
            loss.backward()
            opt.step()
        with torch.no_grad():
            codes = torch.cdist(x, cb).argmin(1)
        return codes, cb.detach()

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        sizes = [int(k) for k in e.codebook_sizes]
        vq_curve: dict[int, dict] = {k: {"recon": [], "purity": []} for k in sizes}
        km_curve: dict[int, dict] = {k: {"recon": [], "purity": []} for k in sizes}
        rnd_curve: dict[int, dict] = {k: {"recon": [], "purity": []} for k in sizes}
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
            for k in sizes:
                vc, vcb = self._vq_fit(x, k, int(e.vq_iters), float(e.lr), s)
                kc, kcb = _kmeans(x, k, int(e.kmeans_iters), s)
                g = torch.Generator().manual_seed(s + 7)
                ridx = torch.randperm(x.shape[0], generator=g)[:k]
                rcb = x[ridx].clone()
                rc = torch.cdist(x, rcb).argmin(1)
                for codes, cb, cur in ((vc, vcb, vq_curve), (kc, kcb, km_curve), (rc, rcb, rnd_curve)):
                    recon = float(((cb[codes] - x) ** 2).mean())
                    cur[k]["recon"].append(recon)
                    cur[k]["purity"].append(_purity(codes, y, k))

        def summarize(cur):
            return {
                k: {
                    "rate_bits": round(math.log2(k), 4),
                    "recon": round(_mean(cur[k]["recon"]), 4),
                    "purity": round(_mean(cur[k]["purity"]), 4),
                    "usability_per_bit": round(_mean(cur[k]["purity"]) / math.log2(k), 4),
                }
                for k in sizes
            }

        vq, km, rnd = summarize(vq_curve), summarize(km_curve), summarize(rnd_curve)
        # VQ usability-per-bit advantage over k-means, averaged across rates
        vq_upb = [vq[k]["usability_per_bit"] for k in sizes]
        km_upb = [km[k]["usability_per_bit"] for k in sizes]
        rnd_upb = [rnd[k]["usability_per_bit"] for k in sizes]
        vq_minus_km = _mean(vq_upb) - _mean(km_upb)
        purities = [vq[k]["purity"] for k in sizes]
        flat_in_rate = bool(max(purities) - min(purities) <= float(e.flat_margin))
        margin = float(e.upb_margin)
        out = {
            "vq_curve": vq,
            "kmeans_curve": km,
            "random_curve": rnd,
            "vq_usability_per_bit_mean": round(_mean(vq_upb), 4),
            "kmeans_usability_per_bit_mean": round(_mean(km_upb), 4),
            "random_usability_per_bit_mean": round(_mean(rnd_upb), 4),
            "vq_minus_kmeans_margin": round(vq_minus_km, 4),
            "usability_flat_in_rate": flat_in_rate,
            "seeds": list(seeds),
            # null: VQ ties k-means on usability-per-bit (small margin) OR usability flat in rate
            "null_supported": bool(vq_minus_km <= margin or flat_in_rate),
        }
        return out
