"""E9: local-learning head in the STREAMING / online-cost regime. Distinct from I4 (which
runs the full rule grid full-batch and tabulates the accuracy gap). Here the angle is the
online cost: a single multiclass latent task is consumed as a stream (one online pass, small
batches, no multi-epoch full-batch fit), and only the genuinely-local rules compete against
backprop. The question is not "can a local rule match backprop accuracy" (it cannot, per I4)
but "does locality buy anything that matters online": lower activation memory (no stored
forward activations for a backward pass) and online stability (low seed-to-seed spread under
one pass). Backprop is the accuracy ceiling. NULL: no local rule reaches within `margin` of
backprop AND none offers a memory or stability win; we return any_local_within_margin and
best_local_memory_win to answer it explicitly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ..devices import DeviceInfo  # noqa: E402
from ..learning.alternatives import RULES  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..substrate.datasets import make_task_stream  # noqa: E402
from .base import Experiment  # noqa: E402

# the subset that is genuinely local (local credit assignment, no global backward pass)
LOCAL_RULES = ("forward_forward", "equilibrium_prop", "predictive_coding", "target_prop")


class E9(Experiment):
    id = "e9_local"
    metric = ("accuracy_vs_backprop", "activation_memory", "online_stability")
    baseline = "backprop head (the accuracy ceiling) on the same streamed task"
    ablation = "local rules only: forward_forward, equilibrium_prop, predictive_coding, target_prop"
    null_hypothesis = (
        "in the streaming regime no local rule reaches within `margin` of backprop accuracy "
        "and none offers a memory or online-stability win: locality buys nothing online"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        margin = float(e.margin)
        passes = int(e.passes)  # online passes over the stream (1 == strict online)
        # one fixed multiclass latent task, consumed as a stream by every rule
        task = make_task_stream(
            n_tasks=1,
            dim=int(e.dim),
            classes_per_task=int(e.n_classes),
            samples_per_task=int(e.samples),
            separation=float(e.separation),
            seed=int(cfg.seed),
        )[0]
        x, y = task.x, task.y

        rules = ("backprop", *LOCAL_RULES)
        table: dict[str, dict] = {}
        for name in rules:
            fn = RULES[name]
            accs, secs, mems = [], [], []
            last = None
            for s in seeds:
                seed_everything(s)
                # streaming: few passes, small lr; activation_memory is the rule's online footprint
                r = fn(x, y, hidden=int(e.hidden), epochs=passes, lr=float(e.lr), seed=s)
                accs.append(r.test_acc)
                secs.append(r.seconds)
                mems.append(r.activation_memory)
                last = r
            assert last is not None  # seeds is non-empty
            mean = sum(accs) / len(accs)
            std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
            table[name] = {
                "acc_mean": mean,
                "online_stability_std": std,  # lower std == steadier under one streamed pass
                "seconds": sum(secs) / len(secs),
                "activation_memory": sum(mems) / len(mems),
                "local": last.local,
                "weight_transport": last.weight_transport,
                "separate_backward": last.separate_backward,
                "chance": last.chance,
            }

        bp = table["backprop"]
        ceiling = bp["acc_mean"]
        bp_mem = bp["activation_memory"]
        bp_std = bp["online_stability_std"]
        for name in LOCAL_RULES:
            row = table[name]
            row["gap_to_backprop"] = ceiling - row["acc_mean"]
            row["within_margin"] = row["gap_to_backprop"] <= margin
            row["memory_win"] = row["activation_memory"] < bp_mem
            row["stability_win"] = row["online_stability_std"] < bp_std + 1e-9

        any_within = any(table[n]["within_margin"] for n in LOCAL_RULES)
        best_mem_win = min(table[n]["activation_memory"] for n in LOCAL_RULES) < bp_mem
        any_stability_win = any(table[n]["stability_win"] for n in LOCAL_RULES)
        out = {
            "ceiling_backprop_acc": ceiling,
            "backprop_activation_memory": bp_mem,
            "backprop_online_stability_std": bp_std,
            "margin": margin,
            "passes": passes,
            "table": table,
            "any_local_within_margin": any_within,
            "best_local_memory_win": best_mem_win,
            "any_local_stability_win": any_stability_win,
            # the explicit null check: locality buys nothing online (no accuracy match, no memory win)
            "null_supported": (not any_within) and (not best_mem_win),
        }
        self._plot(table, ceiling, bp_mem, run_dir)
        return out

    def _plot(self, table, ceiling, bp_mem, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        names = list(table)
        accs = [table[n]["acc_mean"] for n in names]
        errs = [table[n]["online_stability_std"] for n in names]
        mems = [table[n]["activation_memory"] for n in names]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(range(len(names)), accs, yerr=errs, capsize=3, color="tab:blue")
        ax1.axhline(ceiling, color="k", linestyle="--", alpha=0.6, label="backprop ceiling")
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax1.set_ylabel("streamed test accuracy (err == seed std)")
        ax1.set_title("E9: accuracy vs backprop (streaming)")
        ax1.legend(fontsize=8)
        ax2.bar(range(len(names)), mems, color="tab:orange")
        ax2.axhline(bp_mem, color="k", linestyle="--", alpha=0.6, label="backprop memory")
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax2.set_ylabel("relative activation memory (lower == online win)")
        ax2.set_title("E9: online activation memory")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "e9_streaming.png", dpi=110)
        plt.close(fig)
