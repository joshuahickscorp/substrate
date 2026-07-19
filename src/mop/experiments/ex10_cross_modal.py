
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
    return derive_seed32(seed * 1000 + domain, "ex10_cross_modal.modality_b")


def _make_modality_b(xa: torch.Tensor, dim: int, hidden: int, seed: int) -> torch.Tensor:
    seed_everything(seed)
    with torch.no_grad():
        net = mlp(dim, dim, hidden, depth=1, ln=True)
        for p in net.parameters():
            p.requires_grad_(False)
        b = net(xa)
        b = (b - b.mean(0, keepdim=True)) / (b.std(0, keepdim=True) + 1e-6)
    return b


class _CrossModalShell(nn.Module):

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
        opt = torch.optim.Adam(shell.parameters(), lr=lr)
        acc_after_learn: list[float] = []  # acc on domain d right after training on d
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
            tasks_a = make_task_stream(
                n_tasks=n_domains,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            tasks_b = [
                _make_modality_b(t.x, dim, hidden, seed=_modality_b_seed(s, d))
                for d, t in enumerate(tasks_a)
            ]

            xa0, y0, xb0 = tasks_a[0].x, tasks_a[0].y, tasks_b[0]
            gate_a.append(linear_probe(xa0, y0, seed=s)["score"])
            gate_b.append(linear_probe(xb0, y0, seed=s)["score"])

            seed_everything(s)
            cm = _CrossModalShell(dim, hidden, nc)
            b_cm, _ = self._train_arm(cm, tasks_a, tasks_b, epochs, lr, aux_weight, shuffle_seed=None)
            cm_bwt.append(b_cm)

            seed_everything(s)
            sm = _CrossModalShell(dim, hidden, nc)
            b_sm, _ = self._train_arm(sm, tasks_a, tasks_b, epochs, lr, 0.0, shuffle_seed=None)
            sm_bwt.append(b_sm)

            seed_everything(s)
            sh = _CrossModalShell(dim, hidden, nc)
            b_sh, _ = self._train_arm(sh, tasks_a, tasks_b, epochs, lr, aux_weight, shuffle_seed=s + 101)
            sh_bwt.append(b_sh)

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
        tmarg = float(e.transfer_margin)
        transfer_real = bool(tr > chance + tmarg and transfer_above_shuffle > tmarg)

        margin = float(e.margin)
        ga, gb = _mean(gate_a), _mean(gate_b)
        gate_passes = bool(ga > chance + 0.1 and gb > chance + 0.1)

        gain_real = binding_gain > max(margin, spread)  # beats both a fixed margin and the seed spread
        survives_shuffle = shuffled_pairing_gap > margin  # the real gain needs the true pairing
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
            "null_supported": null,
            "binding_is_real": bool(gain_real and survives_shuffle),
            "scope": "synthetic cpu-now precursor of the deferred natural audio-video ex10",
        }
