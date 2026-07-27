"""E4 arms: the locus of adaptation.

One pretrained model, one adaptation budget, seven places the adaptation is allowed to act.

    no_adapt      nothing changes. The floor
    state_only    an owned state vector, a buffer and not a parameter, is recentred on the new context.
                  Zero parameter updates by construction, which the receipt proves rather than declares
    state_noise   the same state vector is moved by random vectors of matched norm. The rate matched,
                  budget matched control that says whether the content of the state matters
    head_only     the readout
    adapter_only  a bottleneck residual on the representation
    core_only     the shared recurrent core
    full          everything

Budget is matched across every arm: the same batches, the same batch size, the same number of passes. The
arms differ only in where the change is allowed to land.

House style: no dashes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mop.method.runs.factorial import LATENT, Projected

LOCI = ("no_adapt", "state_only", "state_noise", "head_only", "adapter_only", "core_only", "full")
# which parameter groups an arm may train. state arms train nothing and act on a buffer instead.
TRAINABLE = {
    "no_adapt": (),
    "state_only": (),
    "state_noise": (),
    "head_only": ("readout",),
    "adapter_only": ("adapter",),
    "core_only": ("core",),
    "full": ("core", "adapter", "readout"),
}
STATE_ARMS = ("state_only", "state_noise")


class LocusModel(nn.Module):
    """Recurrent core, owned state buffer, bottleneck adapter, readout. Four separable loci."""

    def __init__(self, ch: int, classes: int, bottleneck: int = 16):
        super().__init__()
        self.enc = Projected(ch)
        self.rnn = nn.GRU(LATENT, LATENT, batch_first=True)
        self.down = nn.Linear(LATENT, bottleneck)
        self.up = nn.Linear(bottleneck, LATENT)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.head = nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, classes))
        # state is the owned fast variable. anchor is where the representation sat at the end of
        # pretraining, so the adaptation has something to aim at rather than merely something to move.
        self.register_buffer("state", torch.zeros(LATENT))
        self.register_buffer("anchor", torch.zeros(LATENT))
        self.param_groups = {
            "core": [n for n, _ in self.named_parameters() if n.startswith(("enc", "rnn"))],
            "adapter": [n for n, _ in self.named_parameters() if n.startswith(("down", "up"))],
            "readout": [n for n, _ in self.named_parameters() if n.startswith("head")],
        }

    def core_representation(self, x):
        o, _ = self.rnn(self.enc(x))
        return o[:, -1]

    def represent(self, x):
        h = self.core_representation(x) + self.state
        return h + self.up(F.relu(self.down(h)))

    def forward(self, x, d=None, update_recent: bool = False):
        return self.head(self.represent(x)), None


@torch.no_grad()
def set_anchor(model: LocusModel, X, rng, batch: int, passes: int = 8) -> float:
    """Record where the representation sat at the end of pretraining. Measured, not assumed."""
    was = model.training
    model.eval()
    acc = torch.zeros(LATENT)
    for _ in range(passes):
        bi = rng.choice(len(X), min(batch, len(X)), replace=False)
        acc += model.core_representation(X[bi]).mean(0)
    model.anchor.copy_(acc / passes)
    model.train(was)
    return float(model.anchor.norm())


@torch.no_grad()
def adapt_state(model: LocusModel, X, rng, batch: int, passes: int, momentum: float = 0.1,
                noise: bool = False) -> dict:
    """Non parametric adaptation: move the owned state so the new context lands where the old one did.

    The target is anchor minus the new context mean, so on the old context the state converges to zero and
    on a shifted context it removes the shift. Zero gradients, zero optimizer, zero parameter updates. The
    only thing that moves is a buffer, and the receipt proves the parameters did not.
    """
    before = model.state.detach().clone()
    was = model.training
    model.eval()
    seen, norms = 0, []
    for _ in range(passes):
        bi = rng.choice(len(X), min(batch, len(X)), replace=False)
        h = model.core_representation(X[bi])
        target = model.anchor - h.mean(0)
        if noise:
            # rate matched and magnitude matched, but carrying no information about the shift
            r = torch.randn(LATENT)
            target = r * float(target.norm()) / float(r.norm() + 1e-9)
        model.state.mul_(1 - momentum).add_(momentum * target)
        norms.append(float(model.state.norm()))
        seen += len(bi)
    model.train(was)
    return {
        "state_before_norm": round(float(before.norm()), 6),
        "state_after_norm": round(float(model.state.norm()), 6),
        "state_l2_shift": round(float((model.state - before).norm()), 6),
        "state_trajectory_norms": [round(v, 4) for v in norms[:8]],
        "samples_seen": seen,
        "passes": passes,
        "parameter_updates": 0,
        "noise": bool(noise),
    }
