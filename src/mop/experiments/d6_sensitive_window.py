"""D6: sensitive / critical developmental window. A standalone developmental-psychology precursor
on the FROZEN pooled-latent substrate, THE most direct moldability test in the catalog. Question:
is there a WINDOW during training when the shell is more plastic to new evidence, so that a concept
introduced EARLY is learned better (or retained/generalized better) than the identical concept
introduced LATE, beyond what matched total data explains.

Construction: a sequence of concept blocks (each block is one class-incremental task family from
make_task_stream) trained sequentially into ONE shared head. A designated TARGET block appears either
near the START (early arm) or near the END (late arm) of an otherwise-identical stream. Total data and
total forward compute are matched between the two arms by construction: the SAME set of blocks is
trained for the SAME number of epochs, only the POSITION of the target block in the order differs. We
then read the target concept's final competence (its own held-out accuracy under the final shared head,
plus a linear-probe decodability check) in the early-position arm vs the late-position arm.

  early_vs_late_advantage = final target accuracy(early) minus final target accuracy(late).

A genuine sensitive window shows early exposure yields DIFFERENT (typically better-retained) final
competence than late exposure at matched data. A matched-total-data control confirms the two arms saw
identical amounts of the target concept (only its position differs). The STANDING control runs the whole
comparison ALSO on a frozen_random_projection of the latents: a window effect that also appears under
frozen-random is a generic optimizer-order / catastrophic-interference artifact of SGD on a moving head,
NOT a substrate/plasticity property of the frozen V-JEPA geometry.

null_supported = early and late positions produce the same final target competence within seed spread
(no window), OR any apparent window effect also appears under frozen-random (a generic order artifact,
not a substrate plasticity property). The corpus expectation (e3/d7 already failing) is that this holds
as a null on a frozen-latent shell, which is itself a clean, doctrine-relevant moldability finding.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only). No sentience or
agency language. Honest nulls only (null_supported reflects the real toy outcome).
"""

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
        """Build a stream of concept blocks (class-incremental task families sharing one label space).
        Each block is one make_task_stream task with its own cluster geometry; all blocks reuse labels
        0..n_classes-1 (task-incremental label reuse) so a single shared head must remap outputs to each
        block's geometry, which is the moving-target regime a window effect would live in."""
        n_blocks = int(e.n_blocks)
        blocks = []
        for bi in range(n_blocks):
            # task-incremental: shared label space, distinct geometry per block. Distinct seeds per
            # block so the target block is a genuinely separate concept, not a repeat.
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
        """Train ONE shared linear head sequentially over the blocks in `order`, for e.epochs epochs per
        block. `project` optionally maps each block's latents through a fixed substrate transform (identity
        for the real arm, frozen_random_projection for the standing control). Returns the target block's
        final held-out accuracy under the final head, plus its final linear-probe decodability."""
        seed_everything(s + 17)
        dim, nc = int(e.dim), int(e.n_classes)
        epochs, lr = int(e.epochs), float(e.lr)

        # a fixed per-block train/test split so the target block's eval set is identical across arms.
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

        # final competence on the TARGET block held-out set, read under the FINAL shared head.
        xtr_t, ytr_t, xte_t, yte_t = splits[target_idx]
        with torch.no_grad():
            final_acc = float((head(xte_t).argmax(-1) == yte_t).float().mean())
        # linear-probe decodability of the target concept from the (projected) latents: a substrate-level
        # readout that does not depend on the trained head order (a sanity floor on target learnability).
        xall_t = torch.cat([xtr_t, xte_t], 0)
        yall_t = torch.cat([ytr_t, yte_t], 0)
        probe = linear_probe(xall_t, yall_t, seed=s)["score"]
        return final_acc, probe, xte_t.shape[0]

    def _arm(self, blocks: list, target_idx: int, e, s: int, project):
        """Run the early-vs-late comparison for ONE substrate (real or frozen-random). The stream is the
        SAME set of blocks in both positions; only the target block's slot in the order changes, so total
        data and total compute (n_blocks * epochs of head updates) are matched by construction."""
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

            # MATCHED-TOTAL-DATA control: the target block's train and eval counts are identical between
            # the early and late orders (only its POSITION differs), and identical across the real and
            # frozen-random arms. Confirm the two positions saw the same amount of the target concept.
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

        # a window is real only if (a) the early-vs-late advantage on the REAL substrate exceeds the seed
        # spread AND a fixed margin, AND (b) it does NOT also appear under frozen-random (else it is a
        # generic optimizer-order artifact, not a substrate plasticity property).
        margin = float(e.window_margin)
        real_window = window_effect_size > max(margin, adv_spread)
        fr_window = abs(fr_adv_mean) > margin
        substrate_specific = real_window and (abs(adv) - abs(fr_adv_mean) > margin)
        order_matters = bool(real_window)

        # null: no window on the real substrate (advantage within seed spread), OR any window effect also
        # appears under frozen-random (it is not a substrate/plasticity property).
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
