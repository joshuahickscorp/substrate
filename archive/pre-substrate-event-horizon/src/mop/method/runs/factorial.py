"""E1 arms: the core by readout factorial, parameter matched.

Two factors, crossed, plus two controls that sit inside the same design rather than beside it.

    core     pooled   per timestep encoder then order free pooling. No recurrence, no position, no state
             fast     GRU over the projected sequence, final hidden state
             reset3   the same GRU with its hidden state reset every third of the sequence. On both beds a
                      stream is exactly three concatenated sequences, so a period of three lands on the real
                      segment boundaries. It is therefore an oracle segmented control, and it is declared as
                      one rather than read as a neutral ablation
             reset5   the same ablation at a period of five, which lands on no boundary on either bed. This
                      is the control that measures long range state without the segmentation gift
    readout  linear   one linear map to the classes
             mlp      one hidden layer, then the classes

The pooled cells are the order free control and a factorial cell at once, which is what makes the design
discriminating: if the bed truly needs order, pooled must lose, and if it does not, the whole premise is
about the bed rather than the substrate.

Capacity is matched across cores by construction and the residual mismatch is measured and reported, because
an unmatched capacity comparison answers a question nobody asked.

House style: no dashes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64
CORES = ("pooled", "fast", "reset3", "reset5")
RESET_PERIOD = {"reset3": 3, "reset5": 5}
ORACLE_SEGMENTED = ("reset3",)
READOUTS = ("linear", "mlp")


class PerStepEncoder(nn.Module):
    """No temporal operator at all. Applied independently to every timestep."""

    def __init__(self, ch: int, hidden: int, latent: int = LATENT):
        super().__init__()
        self.a = nn.Linear(ch, latent)
        self.b = nn.Linear(latent, hidden)
        self.c = nn.Linear(hidden, latent)

    def forward(self, x):  # (B,T,ch) -> (B,T,latent)
        return F.relu(self.c(F.relu(self.b(F.relu(self.a(x))))))


class Projected(nn.Module):
    def __init__(self, ch: int, latent: int = LATENT):
        super().__init__()
        self.a = nn.Linear(ch, latent)

    def forward(self, x):
        return F.relu(self.a(x))


class Cell(nn.Module):
    """One factorial cell. core and readout are the only things that vary."""

    def __init__(self, ch: int, classes: int, core: str, readout: str, hidden: int = 192):
        super().__init__()
        assert core in CORES and readout in READOUTS
        self.core_kind, self.readout_kind = core, readout
        self.resets = RESET_PERIOD.get(core, 0)
        # both cores emit a LATENT wide representation, so the readout is identical in shape and in
        # parameter count across every cell. Without that, a core comparison is also a readout comparison.
        if core == "pooled":
            self.enc = PerStepEncoder(ch, hidden)
            self.mix = nn.Linear(LATENT * 3, LATENT)  # mean, std, max are all permutation invariant
        else:
            self.enc = Projected(ch)
            self.rnn = nn.GRU(LATENT, LATENT, batch_first=True)
        feat = LATENT
        if readout == "linear":
            self.head = nn.Linear(feat, classes)
        else:
            self.head = nn.Sequential(nn.Linear(feat, LATENT), nn.ReLU(), nn.Linear(LATENT, classes))
        self.param_groups = {
            "core": [n for n, _ in self.named_parameters() if not n.startswith("head")],
            "readout": [n for n, _ in self.named_parameters() if n.startswith("head")],
        }

    def represent(self, x):
        z = self.enc(x)
        if self.core_kind == "pooled":
            return F.relu(self.mix(torch.cat([z.mean(1), z.std(1), z.amax(1)], 1)))
        if self.core_kind == "fast":
            o, _ = self.rnn(z)
            return o[:, -1]
        # reset: the hidden state is cleared at every block boundary, so nothing crosses one. The final
        # representation can therefore depend only on the last block. That is confounded with effective
        # context length by construction, and that confound is the definition of long range state.
        T = z.shape[1]
        step = max(1, T // self.resets)
        last = None
        for s in range(0, T, step):
            o, _ = self.rnn(z[:, s : s + step])
            last = o[:, -1]
        return last

    def forward(self, x, d=None, update_recent: bool = False):
        return self.head(self.represent(x)), None


def build(ch: int, classes: int, core: str, readout: str, hidden: int = 192) -> Cell:
    return Cell(ch, classes, core, readout, hidden=hidden)


_MATCH_CACHE: dict = {}


def match_hidden(ch: int, classes: int, readout: str, target: str = "fast", lo: int = 32, hi: int = 512) -> int:
    """Choose the per step encoder width so the pooled core matches the recurrent core in parameter count."""
    key = (ch, classes, readout, target, lo, hi)
    if key in _MATCH_CACHE:
        return _MATCH_CACHE[key]
    want = count_core(build(ch, classes, target, readout))
    best, bestd = lo, 10**9
    for h in range(lo, hi + 1, 2):
        d = abs(count_core(build(ch, classes, "pooled", readout, hidden=h)) - want)
        if d < bestd:
            best, bestd = h, d
    _MATCH_CACHE[key] = best
    return best


def count_core(m: Cell) -> int:
    names = set(m.param_groups["core"])
    return int(sum(p.numel() for n, p in m.named_parameters() if n in names))


def count_readout(m: Cell) -> int:
    names = set(m.param_groups["readout"])
    return int(sum(p.numel() for n, p in m.named_parameters() if n in names))
