"""I4: backprop-alternatives comparison (fully implemented). The SAME small head is trained
on the SAME latents with head, data, seed, and budget fixed, by each rule: backprop (the
ceiling), feedback alignment, direct feedback alignment, forward-forward, target propagation,
equilibrium propagation, and a predictive-coding approximation. We report the accuracy gap
vs backprop, locality (weight transport? separate backward pass? local-only updates?),
compute cost, and stability (std across seeds). The expected, honest result: alternatives
trail backprop in accuracy while some win on locality/activation-memory.
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


class I4(Experiment):
    id = "i4_backprop_alts"
    metric = ("accuracy_gap", "locality", "compute_cost", "stability")
    baseline = "backprop head (the accuracy ceiling)"
    ablation = "rule grid: FA, DFA, forward-forward, target-prop, equilibrium-prop, predictive-coding"
    null_hypothesis = (
        "no alternative rule comes within the stated accuracy margin of backprop at matched "
        "head/data/seed/budget; locality/memory wins, if any, do not offset the accuracy gap"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        margin = float(e.margin)
        # one fixed multiclass latent task (the shared substrate for every rule)
        task = make_task_stream(
            n_tasks=1,
            dim=int(e.dim),
            classes_per_task=int(e.n_classes),
            samples_per_task=int(e.samples),
            separation=float(e.separation),
            seed=int(cfg.seed),
        )[0]
        x, y = task.x, task.y

        table: dict[str, dict] = {}
        for name, fn in RULES.items():
            accs, secs, last = [], [], None
            for s in seeds:
                seed_everything(s)
                r = fn(x, y, hidden=int(e.hidden), epochs=int(e.epochs), lr=float(e.lr), seed=s)
                accs.append(r.test_acc)
                secs.append(r.seconds)
                last = r
            assert last is not None  # seeds is non-empty
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / len(accs)
            table[name] = {
                "acc_mean": mean,
                "acc_std": var**0.5,
                "seconds": sum(secs) / len(secs),
                "weight_transport": last.weight_transport,
                "separate_backward": last.separate_backward,
                "local": last.local,
                "activation_memory": last.activation_memory,
                "chance": last.chance,
            }

        ceiling = table["backprop"]["acc_mean"]
        for row in table.values():
            row["gap_to_backprop"] = ceiling - row["acc_mean"]
        within = {n: r for n, r in table.items() if n != "backprop" and r["gap_to_backprop"] <= margin}
        out = {
            "ceiling_backprop_acc": ceiling,
            "margin": margin,
            "table": table,
            "rules_within_margin": sorted(within),
            "null_supported": len(within) == 0,  # null: nothing matches backprop
            "best_local_rule": _best_local(table),
        }
        self._plot(table, ceiling, run_dir)
        self._write_table(table, ceiling, run_dir)
        return out

    def _plot(self, table, ceiling, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        names = list(table)
        means = [table[n]["acc_mean"] for n in names]
        errs = [table[n]["acc_std"] for n in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(names)), means, yerr=errs, capsize=3)
        ax.axhline(ceiling, color="k", linestyle="--", alpha=0.6, label="backprop ceiling")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("test accuracy")
        ax.set_title("I4: backprop alternatives")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "i4_comparison.png", dpi=110)
        plt.close(fig)

    def _write_table(self, table, ceiling, run_dir: Path) -> None:
        lines = [
            "| rule | acc | gap | local | transport | sep-bwd | act-mem | sec |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for n, r in table.items():
            lines.append(
                f"| {n} | {r['acc_mean']:.3f}±{r['acc_std']:.3f} | {r['gap_to_backprop']:+.3f} | "
                f"{r['local']} | {r['weight_transport']} | {r['separate_backward']} | "
                f"{r['activation_memory']:.1f} | {r['seconds']:.2f} |"
            )
        (run_dir / "i4_table.md").write_text("\n".join(lines) + "\n")


def _best_local(table) -> str | None:
    local = {n: r for n, r in table.items() if r["local"]}
    return max(local, key=lambda n: local[n]["acc_mean"]) if local else None
