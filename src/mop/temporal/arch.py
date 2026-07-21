"""Core architectures for the E2 causal factorial.

Every core emits a representation of the same fixed width, so the readout is identical in shape and in
parameter count across families. E1 established that this matters: a core comparison that also changes the
readout is two experiments pretending to be one.

Families

    pooled    order free pooling over per timestep features. No recurrence, no position, no state
    histmlp   the last k timesteps flattened into one vector. Explicit causal history, still no recurrence
    tcn       causal dilated convolution. A receptive field without a carried state
    gru       torch nn.GRU
    lstm      torch nn.LSTM
    mgu       a minimal gated recurrent cell written here, stepped in an explicit python loop. Materially
              independent of torch's fused recurrent kernels, which is what makes it a real replication
              rather than the same implementation under another name
    ff_gru    the inherited fastforge shared fast core path

State horizon is implemented as a reset schedule over the recurrent state, so horizon and reset are the same
mechanism seen from two directions and neither can be changed without the other being recorded.

House style: no dashes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64
FAMILIES = ("pooled", "histmlp", "tcn", "gru", "lstm", "mgu", "ff_gru")
RECURRENT = ("gru", "lstm", "mgu", "ff_gru")
STATELESS = ("pooled", "histmlp", "tcn")
READOUTS = ("linear", "mlp1", "mlp_strong")
CAPACITY_TIERS = ("micro", "small", "medium", "large")
TIER_RANGE = {"micro": (10_000, 25_000), "small": (40_000, 80_000),
              "medium": (150_000, 300_000), "large": (600_000, 1_200_000)}


# ---------------------------------------------------------------- cores


class Pooled(nn.Module):
    """Order free. mean, standard deviation and max over time are all permutation invariant."""

    def __init__(self, ch: int, width: int):
        super().__init__()
        self.a = nn.Linear(ch, width)
        self.b = nn.Linear(width, width)
        self.mix = nn.Linear(width * 3, LATENT)

    def forward(self, x, reset=None):
        z = F.relu(self.b(F.relu(self.a(x))))
        return F.relu(self.mix(torch.cat([z.mean(1), z.std(1), z.amax(1)], 1)))


class HistMLP(nn.Module):
    """The last k timesteps, flattened. Explicit causal history and nothing else."""

    def __init__(self, ch: int, width: int, k: int):
        super().__init__()
        self.k = k
        self.a = nn.Linear(ch * k, width)
        self.b = nn.Linear(width, LATENT)

    def forward(self, x, reset=None):
        w = x[:, -self.k :]
        if w.shape[1] < self.k:
            w = F.pad(w, (0, 0, self.k - w.shape[1], 0))
        return F.relu(self.b(F.relu(self.a(w.flatten(1)))))


class CausalTCN(nn.Module):
    """Dilated causal convolution. A receptive field, not a carried state."""

    def __init__(self, ch: int, width: int, levels: int = 4, kernel: int = 3):
        super().__init__()
        self.kernel, self.levels = kernel, levels
        chans = [ch] + [width] * levels
        self.convs = nn.ModuleList(
            [nn.Conv1d(chans[i], chans[i + 1], kernel, dilation=2**i) for i in range(levels)]
        )
        self.out = nn.Linear(width, LATENT)

    @property
    def receptive_field(self) -> int:
        return 1 + sum((self.kernel - 1) * 2**i for i in range(self.levels))

    def forward(self, x, reset=None):
        h = x.transpose(1, 2)
        for i, c in enumerate(self.convs):
            h = F.relu(c(F.pad(h, ((self.kernel - 1) * 2**i, 0))))
        return F.relu(self.out(h[:, :, -1]))


class MGUCell(nn.Module):
    """A minimal gated recurrent cell, written here and stepped explicitly.

    One forget gate and one candidate, both plain linear maps over the concatenation of input and state.
    Nothing is delegated to a fused kernel, so a result that reproduces here and in nn.GRU has reproduced in
    two implementations that share no recurrent code.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.f = nn.Linear(in_dim + hidden, hidden)
        self.c = nn.Linear(in_dim + hidden, hidden)

    def step(self, x, h):
        z = torch.cat([x, h], 1)
        f = torch.sigmoid(self.f(z))
        cand = torch.tanh(self.c(torch.cat([x, f * h], 1)))
        return (1 - f) * h + f * cand


class Recurrent(nn.Module):
    """One recurrent family plus its reset schedule. reset is a sorted list of timestep indices."""

    def __init__(self, ch: int, width: int, kind: str):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(ch, width)
        if kind == "gru":
            self.rnn = nn.GRU(width, width, batch_first=True)
        elif kind == "lstm":
            self.rnn = nn.LSTM(width, width, batch_first=True)
        elif kind == "mgu":
            self.cell = MGUCell(width, width)
        elif kind == "ff_gru":
            # the inherited shared fast core path: a projection with a temporal convolution then a GRU
            self.ffconv = nn.Conv1d(width, width, 5, padding=2)
            self.rnn = nn.GRU(width, width, batch_first=True)
        else:
            raise ValueError(kind)
        self.out = nn.Linear(width, LATENT)

    def _segments(self, T: int, reset):
        if not reset:
            return [(0, T)]
        cuts = [0] + [int(r) for r in reset if 0 < int(r) < T] + [T]
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]

    def forward(self, x, reset=None):
        z = F.relu(self.proj(x))
        if self.kind == "ff_gru":
            z = F.relu(self.ffconv(z.transpose(1, 2)).transpose(1, 2))
        T = z.shape[1]
        last = None
        for s, e in self._segments(T, reset):
            chunk = z[:, s:e]
            if self.kind == "mgu":
                h = torch.zeros(chunk.shape[0], self.cell.hidden, device=chunk.device, dtype=chunk.dtype)
                for t in range(chunk.shape[1]):
                    h = self.cell.step(chunk[:, t], h)
                last = h
            else:
                o, _ = self.rnn(chunk)
                last = o[:, -1]
        return F.relu(self.out(last))


# ---------------------------------------------------------------- readouts


def build_readout(readout: str, classes: int) -> nn.Module:
    if readout == "linear":
        return nn.Linear(LATENT, classes)
    if readout == "mlp1":
        return nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(), nn.Linear(64, classes))
    if readout == "mlp_strong":
        return nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                             nn.Linear(256, classes))
    raise ValueError(readout)


# ---------------------------------------------------------------- the cell


class Cell(nn.Module):
    def __init__(self, ch: int, classes: int, family: str, width: int, readout: str, history_k: int = 1,
                 reset=None, tcn_levels: int = 4):
        super().__init__()
        assert family in FAMILIES and readout in READOUTS
        self.family, self.readout_kind, self.history_k = family, readout, history_k
        self.reset = list(reset or [])
        if family == "pooled":
            self.core = Pooled(ch, width)
        elif family == "histmlp":
            self.core = HistMLP(ch, width, history_k)
        elif family == "tcn":
            self.core = CausalTCN(ch, width, levels=tcn_levels)
        else:
            self.core = Recurrent(ch, width, family)
        self.head = build_readout(readout, classes)
        self.param_groups = {
            "core": [n for n, _ in self.named_parameters() if n.startswith("core")],
            "readout": [n for n, _ in self.named_parameters() if n.startswith("head")],
        }

    def represent(self, x):
        return self.core(x, self.reset)

    def forward(self, x, d=None, update_recent: bool = False):
        return self.head(self.represent(x)), None


def count(model: Cell) -> dict:
    core = sum(p.numel() for n, p in model.named_parameters() if n.startswith("core"))
    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head"))
    return {"core": int(core), "readout": int(head), "total": int(core + head)}


def width_for(family: str, ch: int, classes: int, tier: str, history_k: int = 1, lo: int = 4,
              hi: int = 5000) -> int:
    """Smallest width whose core parameter count lands inside the tier band, else the closest available."""
    want_lo, want_hi = TIER_RANGE[tier]
    target = (want_lo + want_hi) / 2
    best, bestd = lo, float("inf")
    w = lo
    while w <= hi:
        try:
            n = count(Cell(ch, classes, family, w, "linear", history_k=history_k))["core"]
        except Exception:
            break
        if want_lo <= n <= want_hi:
            return w
        d = abs(n - target)
        if d < bestd:
            best, bestd = w, d
        if n > want_hi * 1.5:
            break
        w += max(1, w // 12)
    return best


def build(*, family: str, ch: int, classes: int, tier: str = "small", readout: str = "linear",
          history_k: int = 1, reset=None, width: int | None = None) -> Cell:
    w = width if width is not None else width_for(family, ch, classes, tier, history_k)
    return Cell(ch, classes, family, w, readout, history_k=history_k, reset=reset)


def history_profile(family: str, *, history_k: int, reset, sequence_length: int) -> dict:
    """What past information this arm can see. Declared once, checked by the history witness."""
    if family == "pooled":
        return {"kinds": ["pooled_history"], "k": None, "effective_horizon": "full_unordered"}
    if family == "histmlp":
        return {"kinds": ["last_k_observations"], "k": history_k, "effective_horizon": history_k}
    if family == "tcn":
        return {"kinds": ["last_k_observations"], "k": history_k, "effective_horizon": history_k}
    horizon = sequence_length
    if reset:
        gaps = []
        cuts = [0] + sorted(int(r) for r in reset) + [sequence_length]
        for i in range(len(cuts) - 1):
            gaps.append(cuts[i + 1] - cuts[i])
        horizon = gaps[-1] if gaps else sequence_length
    return {
        "kinds": ["state_carried_from_previous_observations"],
        "k": None,
        "effective_horizon": int(horizon),
    }
