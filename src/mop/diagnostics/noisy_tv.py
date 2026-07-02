"""Noisy-TV diagnostic: the epistemic-vs-aleatoric guard. A correct curiosity/uncertainty
mechanism allocates near-zero attention to a high-entropy UNLEARNABLE region. We stream
FRESH samples each step (so the predictor cannot memorize a finite noise set, which would
fake learnability) from two regions: a LEARNABLE region with a fixed latent dynamics and a
NOISE region whose target is irreducibly random. An ensemble of forward predictors then
shows the corpus's mandatory contract:
  - raw prediction error stays HIGH on noise (point-error curiosity would chase it)
  - ensemble disagreement (epistemic / Bayesian surprise) is LOW on noise (members agree it
    is unpredictable -> nothing to learn -> do not explore)
  - learning progress over the late window is positive on learnable, ~0 on noise (LP is the
    signal intrinsically immune to noisy-TV)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..devices import DeviceInfo
from ..seeding import seed_everything
from ..shell.ensemble import Ensemble
from ..shell.predictor import Predictor


def noisy_tv_diagnostic(
    dim: int = 64,
    device: DeviceInfo | None = None,
    ensemble_size: int = 5,
    steps: int = 400,
    batch: int = 128,
    noise_scale: float = 5.0,
    seed: int = 0,
) -> dict:
    seed_everything(seed)
    dev = device.device if device else torch.device("cpu")
    g = torch.Generator().manual_seed(seed)
    # fixed learnable dynamics + clusters (so the learnable region is genuinely learnable)
    centers = torch.randn(2, dim, generator=g)
    rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]

    def sample_learnable(bs):
        y = torch.randint(0, 2, (bs,), generator=g)
        x = centers[y] * 2.0 + 0.5 * torch.randn(bs, dim, generator=g)
        return x.to(dev), (x @ rot).to(dev)

    def sample_noise(bs):
        x = torch.randn(bs, dim, generator=g).to(dev)
        return x, (torch.randn(bs, dim, generator=g) * noise_scale).to(dev)

    ens = Ensemble(lambda: Predictor(dim, hidden=128, depth=2), size=ensemble_size).to(dev)
    opt = torch.optim.Adam(ens.parameters(), lr=1e-3)

    # held-out eval batches (fresh, never trained on)
    evL = sample_learnable(256)
    evN = sample_noise(256)

    @torch.no_grad()
    def region_stats(ev):
        x, t = ev
        mean, dis = ens.mean_and_disagreement(x)
        return float((mean - t).abs().mean()), float(dis.mean())

    def train(n):
        for _ in range(n):
            opt.zero_grad()
            loss = torch.zeros((), device=dev)
            for x, t in (sample_learnable(batch), sample_noise(batch)):
                out = ens(x)
                loss = loss + F.mse_loss(out, t.unsqueeze(0).expand_as(out))
            loss.backward()
            opt.step()

    train(steps // 2)
    mid_l, _ = region_stats(evL)
    mid_n, _ = region_stats(evN)
    train(steps - steps // 2)
    raw_l, dis_l = region_stats(evL)
    raw_n, dis_n = region_stats(evN)
    lp_l, lp_n = mid_l - raw_l, mid_n - raw_n
    return {
        "raw_error": {"learnable": raw_l, "noise": raw_n},
        "disagreement": {"learnable": dis_l, "noise": dis_n},
        "learning_progress": {"learnable": lp_l, "noise": lp_n},
        "noise_error_stays_high": raw_n > raw_l * 1.5,
        "epistemic_collapses_on_noise": dis_n < raw_n,
        "learning_progress_separates": lp_l > lp_n + 1e-3,
    }
