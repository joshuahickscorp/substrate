"""EX11: causal / interventional probing (do-operations on a control family). Does the shell learn
a genuine interventional map do(v) -> post-intervention latent, or does it merely memorize the
observational co-occurrence of v with its natural effect. A controllable synthetic factor v drives a
cluster center; do(v=value) SETS v directly (an intervention), as opposed to sampling v from its
natural distribution (observation). The intervention predictor sees (pre-intervention latent, v) pairs
drawn from a bounded TRAINING range of v and must predict the post-intervention latent; it is tested
both on held-out v inside that range (interpolation) and on v strictly outside it (extrapolation, never
seen at any value nearby). An observational-predictor control never sees do() at all, it only sees
latents paired with the v that naturally co-occurred with them (a narrower slice of the v range, so
seen-v and effect are confounded exactly as in pure observation), then is evaluated the same way.

NULL: the shell cannot learn an interventional map beyond the observational one. It fits seen
intervention values (small post_intervention_error) but the error on unseen, out-of-range values is
much larger than on held-out in-range values (large extrapolation_gap): what looks learned is curve-
fitting over the observed v support, not a mechanism that generalizes to a genuinely new intervention.
Negative-result taxonomy slot 3 (no causal structure beyond correlation in the pooled latent) or 4
(predictor too weak to extrapolate). cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..seeding import seed_everything
from ..shell.predictor import mlp
from .base import Experiment


def _direction(g: torch.Generator, dim: int) -> torch.Tensor:
    """A fixed unit direction in latent space: the axis the scalar factor v moves the cluster along."""
    d = torch.randn(dim, generator=g)
    return d / d.norm().clamp_min(1e-8)


def _make_intervention_data(
    g: torch.Generator,
    n: int,
    dim: int,
    base: torch.Tensor,
    axis: torch.Tensor,
    v_lo: float,
    v_hi: float,
    noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample n rows: a pre-intervention latent (a noisy draw near `base`), a do(v) value drawn
    uniformly from [v_lo, v_hi], and the resulting post-intervention latent y = base + v*axis + noise.
    This is do(v=value): v is SET, independent of the pre-intervention latent, which is the defining
    feature of an intervention versus an observation."""
    x = base + torch.randn(n, base.shape[0], generator=g) * noise
    v = torch.rand(n, generator=g) * (v_hi - v_lo) + v_lo
    y = base + v.unsqueeze(1) * axis.unsqueeze(0) + torch.randn(n, base.shape[0], generator=g) * noise
    return x, v, y


def _make_observational_data(
    g: torch.Generator,
    n: int,
    dim: int,
    base: torch.Tensor,
    axis: torch.Tensor,
    v_lo: float,
    v_hi: float,
    noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample n rows the OBSERVATIONAL way: v is drawn from the natural distribution (here the same
    marginal support as training, narrower than the full do() range) and the pre-intervention latent is
    correlated with v itself (v leaks into the "before" view), the confound an observational-only
    learner cannot break. Used only to fit the observational-predictor control, never the intervention
    predictor."""
    v = torch.rand(n, generator=g) * (v_hi - v_lo) + v_lo
    x = base + 0.5 * v.unsqueeze(1) * axis.unsqueeze(0) + torch.randn(n, base.shape[0], generator=g) * noise
    y = base + v.unsqueeze(1) * axis.unsqueeze(0) + torch.randn(n, base.shape[0], generator=g) * noise
    return x, v, y


class _InterventionPredictor(nn.Module):
    """(pre-intervention latent, do(v)) -> post-intervention latent. v is concatenated as a scalar
    channel, the minimal interventional interface: the model is handed the SET value directly."""

    def __init__(self, dim: int, hidden: int, depth: int):
        super().__init__()
        self.net = mlp(dim + 1, dim, hidden, depth, ln=True)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, v.unsqueeze(1)], dim=1))


def _fit(model: nn.Module, x: torch.Tensor, v: torch.Tensor, y: torch.Tensor, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x, v)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()


@torch.no_grad()
def _mse(model: nn.Module, x: torch.Tensor, v: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape[0] == 0:
        return float("nan")
    return float(((model(x, v) - y) ** 2).mean())


class EX11(Experiment):
    id = "ex11_causal_probing"
    metric = ("post_intervention_error", "extrapolation_gap", "observational_control_error")
    baseline = "observational-predictor: fit on natural (v, latent) co-occurrence, never sees do()"
    ablation = "interventional predictor trained on do(v) in a bounded range vs held-out unseen v range"
    null_hypothesis = (
        "the shell cannot learn an interventional map beyond the observational one; it predicts seen "
        "interventions but fails unseen values"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, depth = int(e.dim), int(e.hidden), int(e.depth)
        n_train, n_test = int(e.n_train), int(e.n_test)
        v_lo, v_hi = float(e.v_train_lo), float(e.v_train_hi)
        extrap_lo, extrap_hi = float(e.v_extrap_lo), float(e.v_extrap_hi)
        noise = float(e.noise)
        epochs, lr = int(e.epochs), float(e.lr)
        margin = float(e.margin)

        seen_errs, unseen_errs, obs_errs = [], [], []
        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)
            base = torch.randn(dim, generator=g)
            axis = _direction(g, dim)

            # interventional predictor: trained ONLY on do(v) in [v_lo, v_hi]
            xtr, vtr, ytr = _make_intervention_data(g, n_train, dim, base, axis, v_lo, v_hi, noise)
            model = _InterventionPredictor(dim, hidden, depth)
            _fit(model, xtr, vtr, ytr, epochs, lr)

            # held-out SEEN-range interventions (interpolation: same support as training)
            xte_seen, vte_seen, yte_seen = _make_intervention_data(
                g, n_test, dim, base, axis, v_lo, v_hi, noise
            )
            seen_err = _mse(model, xte_seen, vte_seen, yte_seen)

            # UNSEEN-range interventions (extrapolation: v strictly outside [v_lo, v_hi])
            xte_un, vte_un, yte_un = _make_intervention_data(
                g, n_test, dim, base, axis, extrap_lo, extrap_hi, noise
            )
            unseen_err = _mse(model, xte_un, vte_un, yte_un)

            # observational-predictor control: never sees do(), fit on confounded natural co-occurrence
            # over the same v support as training, then evaluated under true intervention (seen range)
            xobs, vobs, yobs = _make_observational_data(g, n_train, dim, base, axis, v_lo, v_hi, noise)
            obs_model = _InterventionPredictor(dim, hidden, depth)
            _fit(obs_model, xobs, vobs, yobs, epochs, lr)
            obs_err = _mse(obs_model, xte_seen, vte_seen, yte_seen)

            seen_errs.append(seen_err)
            unseen_errs.append(unseen_err)
            obs_errs.append(obs_err)

        post_intervention_error = sum(seen_errs) / len(seen_errs)
        extrapolation_error = sum(unseen_errs) / len(unseen_errs)
        observational_control_error = sum(obs_errs) / len(obs_errs)
        extrapolation_gap = extrapolation_error - post_intervention_error

        # honest null: the interventional map fits seen v well but the gap to unseen v is large,
        # i.e. what was learned does not extrapolate past the observed intervention support.
        null_supported = bool(extrapolation_gap > margin * max(post_intervention_error, 1e-8))
        # sanity check: an intervention-aware learner should beat the observational-only control
        # when both are evaluated under a TRUE intervention (do() breaks the observational confound).
        beats_observational_control = bool(post_intervention_error < observational_control_error)

        out = {
            "post_intervention_error": round(post_intervention_error, 6),
            "extrapolation_error": round(extrapolation_error, 6),
            "extrapolation_gap": round(extrapolation_gap, 6),
            "observational_control_error": round(observational_control_error, 6),
            "beats_observational_control": beats_observational_control,
            "margin": margin,
            "v_train_range": [v_lo, v_hi],
            "v_extrap_range": [extrap_lo, extrap_hi],
            "seeds": list(seeds),
            "null_supported": null_supported,
        }
        return out
