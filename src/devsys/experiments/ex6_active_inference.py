"""EX6: active-inference / free-energy shell objective. Does adding a KL-complexity term to the
Gaussian head's prediction loss, and using the resulting expected free energy (EFE) as a SELECTION
signal, buy anything over (a) the plain Gaussian NLL used as both loss and selection signal, or (b) the
learning-progress selector the noisy-TV diagnostic already certifies as immune to irreducible noise. The
mechanism: prediction error (NLL) plus a KL-complexity term over the predicted Gaussian; expected free
energy trades those off to pick which region to sample next.

Complexity term, stated honestly: KL( N(mean, exp(logvar)) || N(0, prior_var) ), a closed-form Gaussian
KL against a FIXED, zero-mean prior of fixed variance. This is a variance-penalty proxy for the
"complexity" half of free energy (it does not marginalize over a hierarchical generative model); the
docstring says so up front rather than dressing it up as full active inference.

Same bake-off structure and same mandatory guard as ex8_curiosity_bakeoff: one pool has a genuinely
learnable rotating-cluster region (reducible error, real forward dynamics) and one has an irreducibly
random noisy-TV region (fresh noise targets each step, nothing to learn). Three selectors compete for
where to spend the next sampling budget: (1) standard-predictor-loss (plain NLL as the selection score,
the naive prediction-error chaser), (2) learning-progress-selection (delta-NLL over a window, the signal
the noisy-TV diagnostic already certifies as immune), (3) the free-energy objective under test (NLL +
KL-complexity as both the training loss AND the selection score).

NULL (verbatim from the registry): the free-energy objective does not improve calibration or selection;
the complexity term is a regularizer, or it chases the noisy-TV. Concretely: null_supported is True
unless EFE selection rejects the noisy-TV region (spends a minority of its budget there) AND beats plain
NLL selection on that rejection AND the KL-trained head is not worse-calibrated than the plain head.
Taxonomy slot 10 (a selection signal that chases irreducible noise is mis-specified). cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..seeding import seed_everything
from ..shell.heads import GaussianHead
from .base import Experiment


def _make_regions(dim: int, noise_scale: float, seed: int):
    """Two fresh-sample generators sharing the noisy-TV diagnostic's spirit: a LEARNABLE region with
    fixed rotating-cluster dynamics (x -> x @ rot, reducible), and a NOISE region whose target is
    irreducibly random each draw (unlearnable). Both stream fresh samples, never a finite replay set,
    so neither region can be memorized into fake learnability."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(2, dim, generator=g)
    rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]

    def learnable(bs: int):
        y = torch.randint(0, 2, (bs,), generator=g)
        x = centers[y] * 2.0 + 0.5 * torch.randn(bs, dim, generator=g)
        return x, x @ rot

    def noise(bs: int):
        x = torch.randn(bs, dim, generator=g)
        return x, torch.randn(bs, dim, generator=g) * noise_scale

    return learnable, noise


def _gaussian_kl_to_prior(mean: torch.Tensor, logvar: torch.Tensor, prior_var: float) -> torch.Tensor:
    """KL( N(mean, exp(logvar)) || N(0, prior_var) ), closed form, mean-reduced. The complexity half of
    the free-energy objective: a fixed, zero-mean, fixed-variance prior, honestly just a variance/mean
    penalty and not a learned hierarchical prior."""
    import math

    var = logvar.exp()
    kl = 0.5 * (-math.log(prior_var) + logvar + (var + mean**2) / prior_var - 1.0)
    return kl.mean()


def _region_error(head: GaussianHead, region, batch: int) -> float:
    """Plain prediction error (NLL) on a fresh held-out draw from a region: the naive selection score
    and also the calibration/coverage input."""
    x, t = region(batch)
    with torch.no_grad():
        return float(head.nll(x, t))


def _region_efe(head: GaussianHead, region, batch: int, prior_var: float, beta: float) -> float:
    """Expected free energy for a region: predicted NLL (risk) plus the KL-complexity term (beta-
    weighted), evaluated on a fresh draw. Lower EFE = preferred; the region MINIMIZING free energy is
    where the active-inference selector samples next."""
    x, t = region(batch)
    with torch.no_grad():
        mean, logvar = head(x)
        risk = 0.5 * (logvar + (t - mean) ** 2 / logvar.exp() + 1.8378770664093453).mean()
        complexity = _gaussian_kl_to_prior(mean, logvar, prior_var)
        return float(risk + beta * complexity)


def _ece_gaussian(head: GaussianHead, region, batch: int, levels: tuple[float, ...]) -> float:
    """Regression calibration error, the Gaussian-head equivalent of classification ECE: for each
    nominal central-interval coverage level (e.g. 50%, 80%, 90%), compare the NOMINAL coverage to the
    EMPIRICAL fraction of targets actually falling inside the predicted mean +/- z*std interval, mean
    absolute gap across levels and output dims. 0 = perfectly calibrated uncertainty."""
    import math

    x, t = region(batch)
    with torch.no_grad():
        mean, logvar = head(x)
        std = (0.5 * logvar).exp()
    err = 0.0
    for level in levels:
        # two-sided z for a central interval of mass `level` under a standard normal, via the inverse
        # error function (no scipy dependency, matches torch's erfinv).
        z = math.sqrt(2.0) * torch.erfinv(torch.tensor(level)).item()
        inside = ((t - mean).abs() <= z * std.clamp_min(1e-6)).float().mean()
        err += abs(float(inside) - level)
    return err / len(levels)


def _train_step(
    head: GaussianHead, opt: torch.optim.Optimizer, x, t, complexity_beta: float, prior_var: float
) -> None:
    opt.zero_grad()
    mean, logvar = head(x)
    nll = 0.5 * (logvar + (t - mean) ** 2 / logvar.exp() + 1.8378770664093453).mean()
    loss = nll
    if complexity_beta > 0.0:
        loss = loss + complexity_beta * _gaussian_kl_to_prior(mean, logvar, prior_var)
    loss.backward()
    opt.step()


class EX6(Experiment):
    id = "ex6_active_inference"
    metric = ("calibration_ece", "fraction_time_learnable", "noise_rejection")
    baseline = "standard-predictor-loss selection (plain Gaussian NLL as both loss and selection score)"
    ablation = "free-energy objective (NLL + KL-complexity) as loss AND expected-free-energy selection"
    null_hypothesis = (
        "the free-energy objective does not improve calibration or selection; the complexity term is a "
        "regularizer, or it chases the noisy-TV"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        hidden = int(e.hidden)
        noise_scale = float(e.noise_scale)
        prior_var = float(e.prior_var)
        beta = float(e.complexity_beta)
        budget = int(e.budget_steps)
        batch = int(e.batch)
        warmup = int(e.warmup_steps)
        lr = float(e.lr)
        levels = tuple(float(v) for v in e.calibration_levels)

        selectors = ("standard_predictor_loss", "learning_progress", "free_energy")
        frac_learnable: dict[str, list[float]] = {s: [] for s in selectors}
        ece_out: dict[str, list[float]] = {s: [] for s in selectors}

        for s in seeds:
            for name in selectors:
                seed_everything(s)
                learnable, noise = _make_regions(dim, noise_scale, s)
                head = GaussianHead(dim, dim, hidden=hidden, depth=1)
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                use_fe_loss = name == "free_energy"

                # brief warmup on both regions so every selector starts from a head that has SOME
                # signal to select with (mirrors ex8's ensemble warmup before the noisy-TV split).
                for _ in range(warmup):
                    x, t = learnable(batch)
                    _train_step(head, opt, x, t, beta if use_fe_loss else 0.0, prior_var)
                    x, t = noise(batch)
                    _train_step(head, opt, x, t, beta if use_fe_loss else 0.0, prior_var)

                prev_err = {
                    "learnable": _region_error(head, learnable, batch),
                    "noise": _region_error(head, noise, batch),
                }
                picks_learnable = 0
                for _t in range(budget):
                    if name == "standard_predictor_loss":
                        err_l = _region_error(head, learnable, batch)
                        err_n = _region_error(head, noise, batch)
                        pick = "learnable" if err_l >= err_n else "noise"  # chase the higher raw error
                    elif name == "learning_progress":
                        err_l = _region_error(head, learnable, batch)
                        err_n = _region_error(head, noise, batch)
                        lp_l = prev_err["learnable"] - err_l
                        lp_n = prev_err["noise"] - err_n
                        pick = "learnable" if lp_l >= lp_n else "noise"  # chase whichever is improving
                        prev_err = {"learnable": err_l, "noise": err_n}
                    else:  # free_energy: minimize expected free energy (risk + complexity)
                        efe_l = _region_efe(head, learnable, batch, prior_var, beta)
                        efe_n = _region_efe(head, noise, batch, prior_var, beta)
                        pick = "learnable" if efe_l <= efe_n else "noise"  # prefer LOWER free energy

                    picks_learnable += int(pick == "learnable")
                    x, t = learnable(batch) if pick == "learnable" else noise(batch)
                    _train_step(head, opt, x, t, beta if use_fe_loss else 0.0, prior_var)

                frac_learnable[name].append(picks_learnable / budget)
                ece_out[name].append(_ece_gaussian(head, learnable, 512, levels))

        def mean(v: list[float]) -> float:
            return sum(v) / len(v)

        table = {
            name: {
                "fraction_time_learnable": round(mean(frac_learnable[name]), 4),
                "calibration_ece": round(mean(ece_out[name]), 4),
                # a selector "rejects" the noisy-TV if it spends a MINORITY of its budget there,
                # i.e. spends a majority on the genuinely learnable region.
                "rejects_noise": bool(mean(frac_learnable[name]) > 0.5),
            }
            for name in selectors
        }

        fe = table["free_energy"]
        base = table["standard_predictor_loss"]
        lp = table["learning_progress"]

        fe_rejects_noise = fe["rejects_noise"]
        fe_beats_baseline_selection = fe["fraction_time_learnable"] > base["fraction_time_learnable"] + 1e-3
        fe_calibration_not_worse = fe["calibration_ece"] <= base["calibration_ece"] + float(e.ece_margin)

        # the positive claim under test: free-energy selection rejects the noisy-TV, beats plain-error
        # selection at doing so, and its complexity term does not degrade calibration versus the plain
        # head. Any one of these failing means the null (regularizer / noise-chaser) stands.
        fe_wins = bool(fe_rejects_noise and fe_beats_baseline_selection and fe_calibration_not_worse)

        return {
            "table": table,
            "seeds": list(seeds),
            "noise_rejection": {
                "standard_predictor_loss": base["rejects_noise"],
                "learning_progress": lp["rejects_noise"],
                "free_energy": fe["rejects_noise"],
            },
            "fraction_time_learnable": {name: table[name]["fraction_time_learnable"] for name in selectors},
            "calibration_ece": {name: table[name]["calibration_ece"] for name in selectors},
            "free_energy_beats_baseline_selection": fe_beats_baseline_selection,
            "free_energy_calibration_not_worse": fe_calibration_not_worse,
            "free_energy_wins": fe_wins,
            # explicit, honest null: the free-energy objective fails to clear ALL three bars above, so
            # it is either a plain regularizer or a noise-chaser (or both) at this toy scale.
            "null_supported": bool(not fe_wins),
        }
