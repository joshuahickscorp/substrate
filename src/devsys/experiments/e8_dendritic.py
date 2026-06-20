"""E8: dendritic predictor. A parameter-matched plain MLP head vs a DENDRITIC head whose
units are gated nonlinear subunits (active-dendrites style): several branch sub-linears whose
context-gated max forms the unit. Both heads see the SAME domain-incremental latent stream in
an ONLINE regime (one pass, no epochs, one SGD step per minibatch). We report capacity-per-param
(final accuracy at matched parameter count), online adaptation speed (steps to a accuracy floor
on each new domain), and forgetting (mean drop on earlier domains). NULL (Experiment 8): the
dendritic analogy adds complexity without benefit, so it merely ties the matched MLP. We return
`dendritic_beats_mlp` as the explicit null check.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ..devices import DeviceInfo, safe_to  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..substrate.datasets import make_task_stream  # noqa: E402
from .base import Experiment  # noqa: E402


class MLPHead(nn.Module):
    """Plain two-layer MLP head: dim -> hidden -> classes."""

    def __init__(self, dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class DendriticHead(nn.Module):
    """Active-dendrites style head. Each of `hidden` units owns `branches` sub-linears; the
    unit value is the context-gated MAX over its branches (a gated nonlinear subunit). A
    per-input context vector (here the input itself) selects which branch dominates via a
    softmax gate, the interference-reducing mechanism the Iyer et al. precedent targets.
    Branch count is chosen so total params match the MLP head it is compared against.
    """

    def __init__(self, dim: int, hidden: int, branches: int, n_classes: int):
        super().__init__()
        self.hidden = hidden
        self.branches = branches
        self.branch = nn.Linear(dim, hidden * branches)  # all branch pre-activations at once
        self.gate = nn.Linear(dim, hidden * branches)  # context-gated branch selection
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        pre = self.branch(x).view(b, self.hidden, self.branches)
        g = self.gate(x).view(b, self.hidden, self.branches)
        sel = torch.softmax(g, dim=-1)
        gated = pre * sel  # context-gated subunits
        unit = F.relu(gated.max(dim=-1).values)  # max over branches forms the unit
        return self.fc2(unit)


def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _matched_mlp_hidden(dim: int, branches: int, n_classes: int, target: int) -> int:
    """Pick the plain-MLP hidden width whose param count is closest to `target` (the
    dendritic head's param count), so the comparison is at MATCHED parameters not added ones.
    """
    best, best_gap = 1, None
    for h in range(1, 4 * target + 4):
        p = _n_params(MLPHead(dim, h, n_classes))
        gap = abs(p - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
        if p > target * 2:
            break
    return best


def _online_pass(model: nn.Module, stream, device, lr: float, batch: int, floor: float, seed: int):
    """One online pass over the domain stream: no epochs, one SGD step per minibatch. After
    each domain finishes, snapshot accuracy on every domain's held-out split. Returns
    (R matrix [domain_seen][domain_eval], adapt_steps per domain)."""
    seed_everything(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    R: list[list[float]] = []
    adapt: list[int] = []
    for task in stream:
        xtr, ytr, xte, yte = task["xtr"], task["ytr"], task["xte"], task["yte"]
        n = xtr.shape[0]
        hit_at, step = None, 0
        for i in range(0, n, batch):
            xb = safe_to(xtr[i : i + batch], device.device)
            yb = safe_to(ytr[i : i + batch], device.device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            step += 1
            if hit_at is None:
                with torch.no_grad():
                    acc = (model(safe_to(xte, device.device)).argmax(1).cpu() == yte).float().mean().item()
                if acc >= floor:
                    hit_at = step
        adapt.append(hit_at if hit_at is not None else step)
        R.append(_eval_all(model, stream, device))
    return R, adapt


def _eval_all(model: nn.Module, stream, device) -> list[float]:
    model.eval()
    accs = []
    with torch.no_grad():
        for task in stream:
            logit = model(safe_to(task["xte"], device.device))
            accs.append((logit.argmax(1).cpu() == task["yte"]).float().mean().item())
    model.train()
    return accs


def _forgetting(R: list[list[float]]) -> float:
    """Mean (peak minus final) over domains seen before the last one. Higher = more forgetting."""
    T = len(R)
    if T < 2:
        return 0.0
    drops = []
    for j in range(T - 1):
        peak = max(R[i][j] for i in range(j, T))
        drops.append(peak - R[-1][j])
    return float(sum(drops) / len(drops))


class E8(Experiment):
    id = "e8_dendritic"
    metric = ("capacity_per_param", "adaptation_speed", "forgetting")
    baseline = "parameter-matched plain MLP head, same online stream/seed/budget"
    ablation = "dendritic head: context-gated max over branch sub-linears (active-dendrites style)"
    null_hypothesis = (
        "the dendritic head only ties the parameter-matched MLP on a domain-incremental online "
        "stream (the analogy adds complexity, no benefit); dendritic_beats_mlp is False"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, branches = int(e.dim), int(e.hidden), int(e.branches)
        n_tasks, cpt, samples = int(e.n_tasks), int(e.classes_per_task), int(e.samples_per_task)
        sep, lr, batch, floor = float(e.separation), float(e.lr), int(e.batch), float(e.adapt_floor)
        margin = float(e.margin)

        dend_params = _n_params(DendriticHead(dim, hidden, branches, cpt))
        mlp_hidden = _matched_mlp_hidden(dim, branches, cpt, dend_params)
        mlp_params = _n_params(MLPHead(dim, mlp_hidden, cpt))

        per_arm: dict[str, dict] = {}
        for arm in ("mlp", "dendritic"):
            finals, adapts, forgets, R_last = [], [], [], None
            for s in seeds:
                stream = self._stream(dim, n_tasks, cpt, samples, sep, int(s))
                model = (
                    MLPHead(dim, mlp_hidden, cpt)
                    if arm == "mlp"
                    else DendriticHead(dim, hidden, branches, cpt)
                )
                model = model.to(device.device)
                R, adapt = _online_pass(model, stream, device, lr, batch, floor, int(s))
                finals.append(float(sum(R[-1]) / len(R[-1])))  # final mean acc over all domains
                adapts.append(float(sum(adapt) / len(adapt)))  # mean steps-to-floor
                forgets.append(_forgetting(R))
                R_last = R
            params = mlp_params if arm == "mlp" else dend_params
            acc_mean = sum(finals) / len(finals)
            per_arm[arm] = {
                "params": params,
                "final_acc_mean": acc_mean,
                "final_acc_std": _std(finals),
                "capacity_per_param": acc_mean / (params / 1000.0),  # acc per 1k params
                "adaptation_speed": sum(adapts) / len(adapts),  # mean steps to floor (lower=faster)
                "forgetting": sum(forgets) / len(forgets),
                "R_last_seed": R_last,
            }

        mlp_cpp = per_arm["mlp"]["capacity_per_param"]
        dend_cpp = per_arm["dendritic"]["capacity_per_param"]
        mlp_speed = per_arm["mlp"]["adaptation_speed"]
        dend_speed = per_arm["dendritic"]["adaptation_speed"]
        capacity_win = dend_cpp - mlp_cpp > margin
        adapt_win = mlp_speed - dend_speed > 0.0  # fewer steps to floor = faster adaptation
        dendritic_beats_mlp = bool(capacity_win or adapt_win)

        out = {
            "matched_params": {"mlp": mlp_params, "dendritic": dend_params},
            "mlp_hidden_matched": mlp_hidden,
            "arms": {a: {k: v for k, v in d.items() if k != "R_last_seed"} for a, d in per_arm.items()},
            "capacity_per_param_gap": dend_cpp - mlp_cpp,
            "adaptation_speed_gap": mlp_speed - dend_speed,
            "margin": margin,
            "n_seeds": len(seeds),
            "null_supported": not dendritic_beats_mlp,
            "dendritic_beats_mlp": dendritic_beats_mlp,  # the explicit null check
        }
        out["plot"] = str(self._plot(per_arm, out, run_dir))
        return out

    def _stream(self, dim, n_tasks, cpt, samples, sep, seed) -> list[dict]:
        raw = make_task_stream(
            n_tasks=n_tasks,
            dim=dim,
            classes_per_task=cpt,
            samples_per_task=samples,
            separation=sep,
            incremental="domain",
            seed=seed,
        )
        stream = []
        for t in raw:
            n = t.x.shape[0]
            cut = int(n * 0.8)
            stream.append({"xtr": t.x[:cut], "ytr": t.y[:cut], "xte": t.x[cut:], "yte": t.y[cut:]})
        return stream

    def _plot(self, per_arm, out, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        arms = ["mlp", "dendritic"]
        cpp = [per_arm[a]["capacity_per_param"] for a in arms]
        ax[0].bar(range(2), cpp)
        ax[0].set_xticks(range(2))
        ax[0].set_xticklabels([f"{a}\n({per_arm[a]['params']}p)" for a in arms])
        ax[0].set_ylabel("acc per 1k params")
        ax[0].set_title(f"capacity-per-param (beats_mlp={out['dendritic_beats_mlp']})")
        for a in arms:
            R = per_arm[a]["R_last_seed"]
            mean_seen = [sum(R[i][: i + 1]) / (i + 1) for i in range(len(R))]
            ax[1].plot(range(len(R)), mean_seen, marker="o", label=a)
        ax[1].set_xlabel("domains seen")
        ax[1].set_ylabel("mean acc over seen domains")
        ax[1].set_title("online retention")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        path = run_dir / "e8_dendritic.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path


def _std(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
