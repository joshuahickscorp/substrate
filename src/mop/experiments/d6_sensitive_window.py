
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from ..substrate.datasets import make_task_stream
from .base import Experiment, _mean, _std


class D6(Experiment):
    id = "d6_sensitive_window"
    metric = ("early_vs_late_advantage", "window_effect_size", "order_matters")
    baseline = "the same content in the reverse (late) position at matched total data and matched compute"
    ablation = "position of a target concept block (early vs late), plus a matched-total-data check"
    null_hypothesis = (
        "there is no sensitive window: the SAME evidence presented early vs late in training produces the "
        "same final competence at matched total data, so exposure order does not shape the shell the way a "
        "developmental critical period shapes a child"
    )
    tier = "cpu-now"

    def _build_blocks(self, e, s: int) -> list:
        n_blocks = int(e.n_blocks)
        blocks = []
        for bi in range(n_blocks):
            t = make_task_stream(
                n_tasks=1,
                dim=int(e.dim),
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples_per_block),
                separation=float(e.separation),
                incremental="task",
                seed=s * 1000 + bi,
            )[0]
            blocks.append(t)
        return blocks

    def _train_order(self, blocks: list, order: list[int], target_idx: int, e, s: int, project):
        seed_everything(s + 17)
        dim, nc = int(e.dim), int(e.n_classes)
        epochs, lr = int(e.epochs), float(e.lr)

        splits = {}
        for bi, t in enumerate(blocks):
            xb = project(t.x)
            g = torch.Generator().manual_seed(s * 7 + bi)
            perm = torch.randperm(xb.shape[0], generator=g)
            xb, yb = xb[perm], t.y[perm]
            cut = int(xb.shape[0] * float(e.train_frac))
            splits[bi] = (xb[:cut], yb[:cut], xb[cut:], yb[cut:])

        head = nn.Linear(dim, nc)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for bi in order:
            xtr, ytr, _, _ = splits[bi]
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(head(xtr), ytr).backward()
                opt.step()

        xtr_t, ytr_t, xte_t, yte_t = splits[target_idx]
        with torch.no_grad():
            final_acc = float((head(xte_t).argmax(-1) == yte_t).float().mean())
        xall_t = torch.cat([xtr_t, xte_t], 0)
        yall_t = torch.cat([ytr_t, yte_t], 0)
        probe = linear_probe(xall_t, yall_t, seed=s)["score"]
        return final_acc, probe, xte_t.shape[0]

    def _arm(self, blocks: list, target_idx: int, e, s: int, project):
        n_blocks = int(e.n_blocks)
        others = [bi for bi in range(n_blocks) if bi != target_idx]

        early_pos = int(e.early_slot)
        late_pos = n_blocks - 1 - int(e.late_from_end)
        early_pos = max(0, min(n_blocks - 1, early_pos))
        late_pos = max(0, min(n_blocks - 1, late_pos))

        early_order = others[:early_pos] + [target_idx] + others[early_pos:]
        late_order = others[:late_pos] + [target_idx] + others[late_pos:]

        early_acc, early_probe, n_eval_e = self._train_order(blocks, early_order, target_idx, e, s, project)
        late_acc, late_probe, n_eval_l = self._train_order(blocks, late_order, target_idx, e, s, project)
        return {
            "early_acc": early_acc,
            "late_acc": late_acc,
            "early_probe": early_probe,
            "late_probe": late_probe,
            "n_eval_early": n_eval_e,
            "n_eval_late": n_eval_l,
            "early_order": early_order,
            "late_order": late_order,
        }

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        target_idx = int(e.target_block)

        real_adv, fr_adv = [], []
        real_early, real_late = [], []
        target_counts_match = True
        identity = None  # real substrate: latents as-is

        for s in seeds:
            seed_everything(s)
            blocks = self._build_blocks(e, s)

            def project_real(x):
                return x

            def project_fr(x, s=s):
                return frozen_random_projection(x, seed=s)

            real = self._arm(blocks, target_idx, e, s, project_real)
            fr = self._arm(blocks, target_idx, e, s, project_fr)
            _ = identity

            real_adv.append(real["early_acc"] - real["late_acc"])
            fr_adv.append(fr["early_acc"] - fr["late_acc"])
            real_early.append(real["early_acc"])
            real_late.append(real["late_acc"])

            n_target_train = int(blocks[target_idx].x.shape[0] * float(e.train_frac))
            if not (
                real["n_eval_early"] == real["n_eval_late"] == fr["n_eval_early"] == fr["n_eval_late"]
                and n_target_train > 0
            ):
                target_counts_match = False

        adv = _mean(real_adv)  # early_vs_late_advantage on the real substrate
        adv_spread = _std(real_adv)
        fr_adv_mean = _mean(fr_adv)
        window_effect_size = abs(adv)

        margin = float(e.window_margin)
        real_window = window_effect_size > max(margin, adv_spread)
        fr_window = abs(fr_adv_mean) > margin
        substrate_specific = real_window and (abs(adv) - abs(fr_adv_mean) > margin)
        order_matters = bool(real_window)

        null_supported = bool((not real_window) or (not substrate_specific))

        return {
            "early_vs_late_advantage": round(adv, 4),
            "window_effect_size": round(window_effect_size, 4),
            "order_matters": order_matters,
            "final_acc_early": round(_mean(real_early), 4),
            "final_acc_late": round(_mean(real_late), 4),
            "advantage_seed_spread": round(adv_spread, 4),
            "frozen_random_advantage": round(fr_adv_mean, 4),
            "window_also_under_frozen_random": bool(fr_window),
            "substrate_specific_window": bool(substrate_specific),
            "matched_total_data": bool(target_counts_match),
            "window_margin": margin,
            "seeds": list(seeds),
            "genuine_sensitive_window": bool(substrate_specific),
            "null_supported": null_supported,
        }
