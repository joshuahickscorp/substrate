from __future__ import annotations

import torch


def reliability(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> dict:
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    ece = 0.0
    n = probs.shape[0]
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if c == 0:
            bin_acc.append(0.0)
            bin_conf.append(float((lo + hi) / 2))
            bin_count.append(0)
            continue
        a = float(correct[m].mean())
        cf = float(conf[m].mean())
        bin_acc.append(a)
        bin_conf.append(cf)
        bin_count.append(c)
        ece += (c / n) * abs(a - cf)
    return {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count, "ece": float(ece)}


def calibration_plot(rel: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(rel["bin_conf"], rel["bin_acc"], marker="o", label=f"ECE={rel['ece']:.3f}")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
