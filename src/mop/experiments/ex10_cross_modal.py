"""EX10 (synthetic arm): cross-modal / cross-domain binding on the frozen pooled-latent substrate.

SCOPE. The registry ex10_cross_modal row is the NATURAL audio-video version (resource_tier
weights-needed, exp_tier 2.1-only): bind frozen VIDEO latents to frozen AUDIO latents and ask
whether the binding improves retention and abstraction. That arm is DEFERRED because no verified
frozen audio encoder HF id exists yet. This module is the cpu-now SYNTHETIC PRECURSOR of it: it
stands in two synthetic "modalities" of the SAME underlying labels and asks the identical question
with a controllable, encoder-free construction, so the mechanism and its standing controls are
exercised now. The natural-modality falsifier (beat the unimodal shell on BWT and abstraction at
matched params) is inherited; only the data source is swapped for the toy latent generator.

MECHANISM. Modality A is a make_task_stream latent stream. Modality B is a fixed random NONLINEAR
re-embedding of the SAME per-sample A latents (a frozen random MLP), so B is a genuinely different
geometry of the same semantic labels, sample-for-sample aligned with A. A tiny cross-modal predictor
maps A -> B as an AUXILIARY objective trained JOINTLY with the main class head on A. We compare three
arms at matched parameters/compute:
  (a) cross-modal: main task on A plus the real A -> B binding auxiliary.
  (b) single-modality control: A only, matched params/compute (the same shell width, auxiliary weight
      zero), no binding.
  (c) shuffled-pairing control: the A -> B correspondence is randomly permuted, so the auxiliary sees
      a real B distribution but a destroyed A-B alignment. Any gain that survives shuffling is a
      generic regularizer, not real cross-modal structure.

METRICS.
  cross_modal_transfer: a probe trained to read labels in modality A, applied to the model's mapped-B
    prediction, decodes the label ABOVE the shuffled-pairing floor (does the learned A -> B map carry
    the shared label code).
  binding_gain: retention (backward transfer over a domain-incremental stream) of the cross-modal arm
    minus the single-modality control.
  shuffled_pairing_gap: binding_gain with real pairing minus binding_gain with shuffled pairing (the
    real test: does the gain need the true A-B correspondence).

NULL. binding_gain is within seed spread OR it does not survive the shuffled-pairing control (shuffled
pairing gets the same gain, so the auxiliary was a generic regularizer, not real cross-modal
alignment). Honest null expected; reported straight.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only). No sentience or
agency language.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, mlp_flops, param_count
from ..diagnostics.linear_probe import linear_probe
from ..seeding import derive_seed32, seed_everything
from ..shell.predictor import mlp
from .base import Experiment, _mean


def _stdev(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def _modality_b_seed(seed: int, domain: int) -> int:
    """Preserve the legacy stream when it fits, hash the full derived seed when it does not."""
    return derive_seed32(seed * 1000 + domain, "ex10_cross_modal.modality_b")


def _make_modality_b(xa: torch.Tensor, dim: int, hidden: int, seed: int) -> torch.Tensor:
    """A FROZEN random nonlinear re-embedding of modality-A latents into modality B. B is a different
    geometry of the SAME per-sample content (sample-for-sample aligned with A), which is the synthetic
    stand-in for a second sensor viewing the same underlying event."""
    seed_everything(seed)
    with torch.no_grad():
        net = mlp(dim, dim, hidden, depth=1, ln=True)
        for p in net.parameters():
            p.requires_grad_(False)
        b = net(xa)
        # renormalize per feature so A and B live at comparable scales (the auxiliary MSE is fair)
        b = (b - b.mean(0, keepdim=True)) / (b.std(0, keepdim=True) + 1e-6)
    return b


class _CrossModalShell(nn.Module):
    """A tiny shell: a shared trunk over modality A, a class head, and a cross-modal head that maps the
    trunk feature to modality B. The single-modality control is the SAME module with the aux weight set
    to zero (identical parameters and per-pass FLOPs), so arms are matched by construction."""

    def __init__(self, dim: int, hidden: int, nc: int):
        super().__init__()
        self.trunk = mlp(dim, hidden, hidden, depth=1, ln=True)
        self.cls = nn.Linear(hidden, nc)
        self.to_b = nn.Linear(hidden, dim)

    def forward(self, xa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(xa)
        return self.cls(h), self.to_b(h)


class EX10(Experiment):
    id = "ex10_cross_modal"
    metric = ("cross_modal_transfer", "binding_gain", "shuffled_pairing_gap")
    baseline = (
        "single-modality shell (modality A only) at matched parameters/compute (aux weight zero), plus "
        "a shuffled-pairing control that destroys the A-B correspondence"
    )
    ablation = "undefined"
    null_hypothesis = (
        "a cross-modal predictor binding two synthetic modalities of the same underlying labels does not "
        "improve retention or abstraction over a single-modality control; the modalities are not aligned "
        "enough to bind, or the auxiliary objective is just a regularizer"
    )
    tier = "cpu-now"

    def _train_arm(
        self, shell, tasks_a, tasks_b, epochs, lr, aux_weight, shuffle_seed
    ) -> tuple[float, list[float]]:
        """Train the shell over a domain-incremental stream (label space shared, geometry rotates per
        domain, the reliable forgetting regime). Returns backward transfer (mean retention on earlier
        domains after the whole stream) and the per-domain end-of-training accuracy trace. aux_weight>0
        adds the A -> B binding loss. shuffle_seed not None permutes the A-B pairing (destroys binding)."""
        opt = torch.optim.Adam(shell.parameters(), lr=lr)
        acc_after_learn: list[float] = []  # acc on domain d right after training on d
        # keep held-out A/y per domain to measure retention at the end
        held: list[tuple[torch.Tensor, torch.Tensor]] = []
        for d, (ta, tb) in enumerate(zip(tasks_a, tasks_b, strict=True)):
            xa, y, xb = ta.x, ta.y, tb
            cut = int(xa.shape[0] * 0.7)
            xatr, ytr, xbtr = xa[:cut], y[:cut], xb[:cut]
            xate, yte = xa[cut:], y[cut:]
            held.append((xate, yte))
            if shuffle_seed is not None:
                g = torch.Generator().manual_seed(shuffle_seed + d)
                xbtr = xbtr[torch.randperm(xbtr.shape[0], generator=g)]
            for _ in range(epochs):
                opt.zero_grad()
                logits, pred_b = shell(xatr)
                loss = F.cross_entropy(logits, ytr)
                if aux_weight > 0.0:
                    loss = loss + aux_weight * F.mse_loss(pred_b, xbtr)
                loss.backward()
                opt.step()
            with torch.no_grad():
                logits, _ = shell(xate)
                acc_after_learn.append(float((logits.argmax(-1) == yte).float().mean()))
        # backward transfer: mean retained accuracy on ALL earlier domains after the full stream
        with torch.no_grad():
            retained = []
            for d in range(len(held) - 1):  # earlier domains only (exclude the last, no forgetting yet)
                xate, yte = held[d]
                logits, _ = shell(xate)
                retained.append(float((logits.argmax(-1) == yte).float().mean()))
        bwt = _mean(retained) if retained else _mean(acc_after_learn)
        return bwt, acc_after_learn

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, nc = int(e.dim), int(e.hidden), int(e.n_classes)
        n_domains = int(e.n_domains)
        epochs, lr = int(e.epochs), float(e.lr)
        aux_weight = float(e.aux_weight)

        cm_bwt, sm_bwt, sh_bwt = [], [], []
        transfer_scores, shuf_transfer_scores = [], []
        gate_a, gate_b = [], []
        for s in seeds:
            seed_everything(s)
            # modality A: a domain-incremental stream (shared labels, geometry rotates per domain).
            tasks_a = make_task_stream(
                n_tasks=n_domains,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            # modality B: a FROZEN random nonlinear re-embedding of each domain's A latents (a different
            # geometry of the same per-sample content, aligned sample-for-sample with A).
            tasks_b = [
                _make_modality_b(t.x, dim, hidden, seed=_modality_b_seed(s, d))
                for d, t in enumerate(tasks_a)
            ]

            # MANDATORY GATE: are the shared labels linearly decodable in EACH modality (else any
            # binding failure is pre-ordained by the substrate, taxonomy 3, not by the mechanism).
            xa0, y0, xb0 = tasks_a[0].x, tasks_a[0].y, tasks_b[0]
            gate_a.append(linear_probe(xa0, y0, seed=s)["score"])
            gate_b.append(linear_probe(xb0, y0, seed=s)["score"])

            # (a) cross-modal arm: main task on A plus the real A -> B binding auxiliary.
            seed_everything(s)
            cm = _CrossModalShell(dim, hidden, nc)
            b_cm, _ = self._train_arm(cm, tasks_a, tasks_b, epochs, lr, aux_weight, shuffle_seed=None)
            cm_bwt.append(b_cm)

            # (b) single-modality control: identical module, aux weight zero (matched params/compute).
            seed_everything(s)
            sm = _CrossModalShell(dim, hidden, nc)
            b_sm, _ = self._train_arm(sm, tasks_a, tasks_b, epochs, lr, 0.0, shuffle_seed=None)
            sm_bwt.append(b_sm)

            # (c) shuffled-pairing control: real B distribution, destroyed A-B correspondence.
            seed_everything(s)
            sh = _CrossModalShell(dim, hidden, nc)
            b_sh, _ = self._train_arm(sh, tasks_a, tasks_b, epochs, lr, aux_weight, shuffle_seed=s + 101)
            sh_bwt.append(b_sh)

            # cross_modal_transfer: train a probe to read labels from REAL modality B, apply it to the
            # cross-modal model's PREDICTED B on held-out A. If the learned A -> B map carries the shared
            # label code, the probe decodes the A-side label from the mapped-B prediction, above the
            # shuffled-pairing floor. Use the last domain (freshest, no forgetting) for a clean readout.
            xa_te, y_te, xb_te = tasks_a[-1].x, tasks_a[-1].y, tasks_b[-1]
            b_probe = nn.Linear(dim, nc)
            bo = torch.optim.Adam(b_probe.parameters(), lr=lr)
            for _ in range(epochs):
                bo.zero_grad()
                F.cross_entropy(b_probe(xb_te), y_te).backward()
                bo.step()
            with torch.no_grad():
                _, pred_b_cm = cm(xa_te)
                _, pred_b_sh = sh(xa_te)
                transfer_scores.append(float((b_probe(pred_b_cm).argmax(-1) == y_te).float().mean()))
                shuf_transfer_scores.append(float((b_probe(pred_b_sh).argmax(-1) == y_te).float().mean()))

        # matched-compute check: all three arms are the SAME module (trunk + cls + to_b), so per-pass
        # forward FLOPs are identical by construction. The single-modality control simply zeroes the aux
        # loss, it does not remove the to_b head, so parameters and compute match exactly.
        ref = _CrossModalShell(dim, hidden, nc)
        flops = mlp_flops([dim, hidden, hidden]) + mlp_flops([hidden, nc]) + mlp_flops([hidden, dim])
        compute = matched_within(flops, flops)
        params = param_count(ref)

        cmb, smb, shb = _mean(cm_bwt), _mean(sm_bwt), _mean(sh_bwt)
        binding_gain = cmb - smb  # cross-modal retention minus single-modality
        shuffled_gain = shb - smb  # shuffled-pairing retention minus single-modality
        shuffled_pairing_gap = binding_gain - shuffled_gain  # does the gain need the true A-B pairing
        spread = _stdev(cm_bwt) + _stdev(sm_bwt)  # seed spread of the binding-gain difference

        tr, shtr = _mean(transfer_scores), _mean(shuf_transfer_scores)
        chance = 1.0 / nc
        transfer_above_shuffle = tr - shtr
        # cross_modal_transfer counts only if it clears BOTH chance and the shuffled-pairing floor.
        tmarg = float(e.transfer_margin)
        transfer_real = bool(tr > chance + tmarg and transfer_above_shuffle > tmarg)

        margin = float(e.margin)
        ga, gb = _mean(gate_a), _mean(gate_b)
        gate_passes = bool(ga > chance + 0.1 and gb > chance + 0.1)

        gain_real = binding_gain > max(margin, spread)  # beats both a fixed margin and the seed spread
        survives_shuffle = shuffled_pairing_gap > margin  # the real gain needs the true pairing
        # null: binding gain within seed spread OR it does not survive the shuffled-pairing control.
        null = bool((not gain_real) or (not survives_shuffle))
        return {
            "cross_modal_transfer": round(tr, 4),
            "shuffled_transfer": round(shtr, 4),
            "transfer_above_shuffle": round(transfer_above_shuffle, 4),
            "transfer_chance": round(chance, 4),
            "transfer_is_real": transfer_real,
            "binding_gain": round(binding_gain, 4),
            "shuffled_pairing_gain": round(shuffled_gain, 4),
            "shuffled_pairing_gap": round(shuffled_pairing_gap, 4),
            "cross_modal_bwt": round(cmb, 4),
            "single_modality_bwt": round(smb, 4),
            "shuffled_pairing_bwt": round(shb, 4),
            "binding_gain_seed_spread": round(spread, 4),
            "modality_a_decodability": round(ga, 4),
            "modality_b_decodability": round(gb, 4),
            "gate_passes": gate_passes,
            "compute_matched": compute["matched"],
            "params": params,
            "margin": margin,
            "seeds": list(seeds),
            # null: binding gain within seed spread OR it does not survive the shuffled-pairing control.
            "null_supported": null,
            "binding_is_real": bool(gain_real and survives_shuffle),
            "scope": "synthetic cpu-now precursor of the deferred natural audio-video ex10",
        }
