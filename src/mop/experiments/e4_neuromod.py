"""E4: neuromodulation (uncertainty-gated learning), with the mandatory noisy-TV test.

An Ensemble of forward Predictors learns a LEARNABLE region (fixed latent dynamics, low
aleatoric) while a NOISE region (irreducibly random target) sits beside it. The question is
where a gate sends learning/attention. We compare three gates:
  (a) ungated:               constant attention everywhere (the do-nothing reference)
  (b) point-error gated:     attention ~ raw prediction-error magnitude
  (c) disagreement gated:    attention ~ ensemble variance (epistemic / Bayesian surprise)

For each gate we report adaptation speed on the learnable region and, critically, the gate
VALUE on the noise region (the resource it allocates to noisy-TV). The correct mechanism
spends near-zero on noise. The named null: point-error gating allocates as much to noisy-TV
as ungated (it cannot tell aleatoric from epistemic), while disagreement does not. We return
point_error_chases_noise and disagreement_ignores_noise as the explicit null check, reusing
noisy_tv_diagnostic for the core epistemic-vs-error contrast. A GaussianHead on the learnable
region reports calibration (reliability + ECE). A plot is saved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ..devices import DeviceInfo, safe_to  # noqa: E402
from ..diagnostics import noisy_tv_diagnostic, reliability  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..shell.ensemble import Ensemble  # noqa: E402
from ..shell.heads import GaussianHead  # noqa: E402
from ..shell.predictor import Predictor  # noqa: E402
from ..substrate.datasets import noisy_tv_dataset  # noqa: E402
from .base import Experiment  # noqa: E402


class E4(Experiment):
    id = "e4_neuromod"
    metric = ("adapt_speed_learnable", "gate_on_noise", "calibration_ece")
    baseline = "ungated learning (constant attention on learnable and noise alike)"
    ablation = "gate in {ungated, point-error, ensemble-disagreement} on noisy-TV (learnable + noise regions)"
    null_hypothesis = (
        "point prediction-error gating allocates as much attention to the noisy-TV region as "
        "ungated (conflates epistemic with aleatoric) while ensemble-disagreement gating does not"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        dev = device.device
        seed_everything(int(cfg.seed))
        dim = int(e.dim)
        data = noisy_tv_dataset(
            dim=dim,
            n=int(e.n),
            separation=float(e.separation),
            noise_scale=float(e.noise_scale),
            seed=int(cfg.seed),
        )
        xl = safe_to(data["learnable"].x, dev)
        tl = safe_to(data["learnable"].xnext, dev)  # type: ignore[arg-type]
        xn = safe_to(data["noise"].x, dev)
        tn = safe_to(data["noise"].xnext, dev)  # type: ignore[arg-type]

        def make():
            return Predictor(dim, hidden=int(e.hidden), depth=int(e.depth))

        ens = Ensemble(make, size=int(e.ensemble_size)).to(dev)
        opt = torch.optim.Adam(ens.parameters(), lr=float(e.lr))
        steps = int(e.steps)
        bs = int(e.batch)

        # per-region gate trajectories for the three mechanisms, and learnable-error trajectory
        traj_err_l: list[float] = []
        gate_pt: dict[str, list[float]] = {"learnable": [], "noise": []}
        gate_dis: dict[str, list[float]] = {"learnable": [], "noise": []}

        def region_signals(x, t):
            mean, dis = ens.mean_and_disagreement(x)
            err = float((mean - t).abs().mean())
            return err, float(dis.mean())

        for _ in range(steps):
            with torch.no_grad():
                el, dl = region_signals(xl, tl)
                en, dn = region_signals(xn, tn)
            traj_err_l.append(el)
            gate_pt["learnable"].append(el)
            gate_pt["noise"].append(en)
            gate_dis["learnable"].append(dl)
            gate_dis["noise"].append(dn)
            opt.zero_grad()
            il = torch.randint(0, xl.shape[0], (bs,))
            ino = torch.randint(0, xn.shape[0], (bs,))
            loss = torch.zeros((), device=dev)
            for x, t in ((xl[il], tl[il]), (xn[ino], tn[ino])):
                out = ens(x)
                loss = loss + F.mse_loss(out, t.unsqueeze(0).expand_as(out))
            loss.backward()
            opt.step()

        # gate-on-noise = late-window mean attention each mechanism would send to noisy-TV,
        # normalized to the ungated reference (a constant 1.0 attention everywhere).
        w = max(1, steps // 4)
        ung_l = 1.0  # ungated: attention is the same constant on both regions
        ung_n = 1.0
        pt_l = _late(gate_pt["learnable"], w)
        pt_n = _late(gate_pt["noise"], w)
        dis_l = _late(gate_dis["learnable"], w)
        dis_n = _late(gate_dis["noise"], w)
        # normalize each mechanism by its own learnable-region value so "1.0" means
        # "spends on noise like it spends on learnable", and the ungated bar sits at 1.0.
        pt_noise_rel = _safe_div(pt_n, pt_l)
        dis_noise_rel = _safe_div(dis_n, dis_l)
        ung_noise_rel = _safe_div(ung_n, ung_l)  # exactly 1.0

        chase_ratio = float(e.chase_ratio)
        ignore_ratio = float(e.ignore_ratio)
        point_error_chases_noise = pt_noise_rel >= chase_ratio * ung_noise_rel
        disagreement_ignores_noise = dis_noise_rel <= ignore_ratio * ung_noise_rel
        null_supported = point_error_chases_noise and disagreement_ignores_noise

        adapt_speed_learnable = traj_err_l[0] - traj_err_l[-1]  # error reduction = how fast it adapted
        cal = self._calibrate(data, dim, dev, e, int(cfg.seed))
        diag = noisy_tv_diagnostic(
            dim=dim,
            device=device,
            ensemble_size=int(e.ensemble_size),
            steps=int(e.steps),
            batch=int(e.batch),
            noise_scale=float(e.noise_scale),
            seed=int(cfg.seed),
        )

        out = {
            "adapt_speed_learnable": adapt_speed_learnable,
            "gate_on_noise": {
                "ungated": ung_noise_rel,
                "point_error": pt_noise_rel,
                "disagreement": dis_noise_rel,
            },
            "gate_on_learnable_raw": {"point_error": pt_l, "disagreement": dis_l},
            "calibration_ece": cal["ece"],
            "point_error_chases_noise": bool(point_error_chases_noise),
            "disagreement_ignores_noise": bool(disagreement_ignores_noise),
            "null_supported": bool(null_supported),
            "diagnostic": diag,
        }
        self._plot(traj_err_l, out["gate_on_noise"], cal, run_dir)
        return out

    def _calibrate(self, data, dim: int, dev, e, seed: int) -> dict:
        """Train a GaussianHead next-latent regressor on the learnable region, then read
        reliability/ECE. Confidence is per-sample (low predicted variance = high confidence),
        correctness is per-sample (predicted within a tolerance of the target)."""
        seed_everything(seed)
        x = safe_to(data["learnable"].x, dev)
        t = safe_to(data["learnable"].xnext, dev)
        n = x.shape[0]
        ntr = int(n * 0.8)
        head = GaussianHead(dim, dim, hidden=int(e.cal_hidden)).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        for _ in range(int(e.cal_epochs)):
            opt.zero_grad()
            head.nll(x[:ntr], t[:ntr]).backward()
            opt.step()
        with torch.no_grad():
            mean, var = head.predict(x[ntr:])
            tt = t[ntr:]
            err = (mean - tt).abs().mean(dim=-1)  # [M] per-sample mean abs error
            conf = 1.0 / (1.0 + var.mean(dim=-1))  # [M] high when predicted variance is low
            tol = float(err.median())
            correct = (err <= tol).long()  # [M] in {0,1}
            probs = torch.stack([1.0 - conf, conf], dim=-1).cpu()  # 2-class [P(wrong), P(right)]
            rel = reliability(probs, correct.cpu())
        return rel

    def _plot(self, traj_err_l, gate_on_noise, cal, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].plot(traj_err_l)
        ax[0].set_title("adaptation on learnable region")
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("prediction error")
        names = ["ungated", "point_error", "disagreement"]
        vals = [gate_on_noise[k] for k in names]
        colors = ["gray", "crimson", "seagreen"]
        ax[1].bar(range(3), vals, color=colors)
        ax[1].axhline(1.0, color="k", linestyle="--", alpha=0.6, label="ungated reference")
        ax[1].set_xticks(range(3))
        ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax[1].set_ylabel("attention on noise / on learnable")
        ax[1].set_title("resource sent to noisy-TV")
        ax[1].legend(fontsize=8)
        ax[2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
        ax[2].plot(cal["bin_conf"], cal["bin_acc"], marker="o", label=f"ECE={cal['ece']:.3f}")
        ax[2].set_xlabel("confidence")
        ax[2].set_ylabel("accuracy")
        ax[2].set_title("calibration (learnable)")
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "e4_neuromod.png", dpi=110)
        plt.close(fig)


def _late(seq: list[float], w: int) -> float:
    tail = seq[-w:] if seq else [0.0]
    return sum(tail) / len(tail)


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-9 else 0.0
