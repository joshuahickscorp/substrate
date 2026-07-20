"""Owned architectures G and H plus the conventional comparison family.

G  Fast Shared Core with Domain-Local Plasticity. One shared causal fast recurrent core carries dynamics.
   Every slow degree of freedom (adapter, normalization, head) is domain local. No shared medium state and
   no shared trainable slow workspace, both closed by prior evidence.

H  Anchored Fast Dynamics with Interference-Gated Adaptation. The shared core is a frozen anchor plus a
   bounded trainable delta, so shared plasticity is explicitly limited and reversible toward the anchor.
   Domain-local capacity is low rank.

Every module declares param_groups as group name to parameter name list. Group membership is a partition of
the trainable parameters: engine receipts and the interference map are computed from it, so an undeclared or
double counted parameter is a test failure, not a silent bug.

House style: no dashes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64
TAU = 0.5  # bound on the H fast delta, in weight units


class Projection(nn.Module):
    """Per domain input projection. conv is channel specific, lin has a domain independent shape so it can
    be carried by a projection only transfer arm."""

    def __init__(self, ch: int, latent: int = LATENT):
        super().__init__()
        self.conv = nn.Conv1d(ch, 32, 5, padding=2)
        self.lin = nn.Linear(32, latent)

    def forward(self, x):  # (B,T,ch) -> (B,T,latent)
        return F.relu(self.lin(F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)))


class Adapter(nn.Module):
    """Domain local slow adapter: bottleneck residual on the shared core output."""

    def __init__(self, latent: int = LATENT, bottleneck: int = 16):
        super().__init__()
        self.down = nn.Linear(latent, bottleneck)
        self.up = nn.Linear(bottleneck, latent)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.relu(self.down(h)))


class LowRankAdapter(nn.Module):
    """Domain local low rank adapter used by H. Rank 8 keeps domain capacity far below the shared core."""

    def __init__(self, latent: int = LATENT, rank: int = 8):
        super().__init__()
        self.a = nn.Parameter(torch.randn(latent, rank) * 0.02)
        self.b = nn.Parameter(torch.zeros(rank, latent))

    def forward(self, h):
        return h + (h @ self.a) @ self.b


def _names(module: nn.Module, prefix: str) -> list[str]:
    return [f"{prefix}.{n}" if n else prefix for n, _ in module.named_parameters()]


class ArchG(nn.Module):
    """Fast Shared Core with Domain-Local Plasticity."""

    name = "G"

    def __init__(self, domains: dict[str, tuple[int, int]]):
        super().__init__()
        self.domains = list(domains)
        self.projs = nn.ModuleDict({d: Projection(ch) for d, (ch, _) in domains.items()})
        self.core = nn.GRU(LATENT, LATENT, batch_first=True)  # shared causal fast core
        self.adapters = nn.ModuleDict({d: Adapter() for d in domains})
        self.norms = nn.ModuleDict({d: nn.LayerNorm(LATENT) for d in domains})
        self.heads = nn.ModuleDict({d: nn.Linear(LATENT, n) for d, (_, n) in domains.items()})
        self.register_buffer("recent", torch.zeros(LATENT))  # bounded recent state, not trainable
        self.param_groups = {"fast_core": _names(self.core, "core")}
        for d in domains:
            self.param_groups[f"proj_conv.{d}"] = [f"projs.{d}.conv.weight", f"projs.{d}.conv.bias"]
            self.param_groups[f"proj_lin.{d}"] = [f"projs.{d}.lin.weight", f"projs.{d}.lin.bias"]
            self.param_groups[f"adapter.{d}"] = _names(self.adapters[d], f"adapters.{d}")
            self.param_groups[f"norm.{d}"] = _names(self.norms[d], f"norms.{d}")
            self.param_groups[f"head.{d}"] = _names(self.heads[d], f"heads.{d}")

    def features(self, x, d):
        o, _ = self.core(self.projs[d](x))
        return o

    def forward(self, x, d, update_recent: bool = False):
        o = self.features(x, d)
        last = o[:, -1]
        if update_recent and self.training:
            with torch.no_grad():
                self.recent.mul_(0.9).add_(0.1 * last.mean(0))
        h = self.adapters[d](self.norms[d](last))
        return self.heads[d](h), o


class ArchH(nn.Module):
    """Anchored Fast Dynamics with Interference-Gated Adaptation."""

    name = "H"

    def __init__(self, domains: dict[str, tuple[int, int]]):
        super().__init__()
        self.domains = list(domains)
        self.projs = nn.ModuleDict({d: Projection(ch) for d, (ch, _) in domains.items()})
        self._template = nn.GRU(LATENT, LATENT, batch_first=True)
        self.anchor_keys = [n for n, _ in self._template.named_parameters()]
        for n, p in list(self._template.named_parameters()):
            self.register_buffer(f"anchor_{n}", p.detach().clone())
        for p in self._template.parameters():
            p.requires_grad_(False)
        self.delta = nn.ParameterDict(
            {
                n.replace(".", "_"): nn.Parameter(torch.zeros_like(p))
                for n, p in self._template.named_parameters()
            }
        )
        self.adapters = nn.ModuleDict({d: LowRankAdapter() for d in domains})
        self.heads = nn.ModuleDict({d: nn.Linear(LATENT, n) for d, (_, n) in domains.items()})
        self.param_groups = {"fast_delta": [f"delta.{n.replace('.', '_')}" for n in self.anchor_keys]}
        for d in domains:
            self.param_groups[f"proj_conv.{d}"] = [f"projs.{d}.conv.weight", f"projs.{d}.conv.bias"]
            self.param_groups[f"proj_lin.{d}"] = [f"projs.{d}.lin.weight", f"projs.{d}.lin.bias"]
            self.param_groups[f"adapter.{d}"] = [f"adapters.{d}.a", f"adapters.{d}.b"]
            self.param_groups[f"head.{d}"] = _names(self.heads[d], f"heads.{d}")

    def bounded_delta(self, n: str) -> torch.Tensor:
        return TAU * torch.tanh(self.delta[n.replace(".", "_")] / TAU)

    def effective_core(self) -> dict[str, torch.Tensor]:
        return {n: getattr(self, f"anchor_{n}") + self.bounded_delta(n) for n in self.anchor_keys}

    def drift(self) -> float:
        """L2 distance of the effective core from the anchor, in weight units."""
        with torch.no_grad():
            return float(torch.sqrt(sum((self.bounded_delta(n) ** 2).sum() for n in self.anchor_keys)).item())

    def restore_anchor(self, frac: float = 1.0):
        with torch.no_grad():
            for k in self.delta:
                self.delta[k].mul_(1.0 - frac)

    def features(self, x, d):
        o, _ = torch.func.functional_call(self._template, self.effective_core(), (self.projs[d](x),))
        return o

    def forward(self, x, d, update_recent: bool = False):
        o = self.features(x, d)
        h = self.adapters[d](o[:, -1])
        return self.heads[d](h), o


class ArchGR(nn.Module):
    """Architecture G improvement rounds. Same premise, three declared structural changes.

    input_norm   per domain normalization of the projection output, so the shared core sees comparable
                 statistics from both domains while the dynamics stay shared (round G-R1)
    core_delta   the shared core becomes anchor plus bounded delta, so shared change has a ceiling and is
                 reversible, without adopting H's gating (round G-R2)
    split_core   the core is declared as two groups, input to hidden and hidden to hidden, so a return can
                 reopen only one of them (round G-R3)
    """

    name = "GR"

    def __init__(
        self, domains: dict[str, tuple[int, int]], input_norm=False, core_delta=False, split_core=False
    ):
        super().__init__()
        self.domains = list(domains)
        self.input_norm, self.core_delta, self.split_core = input_norm, core_delta, split_core
        self.projs = nn.ModuleDict({d: Projection(ch) for d, (ch, _) in domains.items()})
        self.core = nn.GRU(LATENT, LATENT, batch_first=True)
        self.in_norms = nn.ModuleDict({d: nn.LayerNorm(LATENT) for d in domains}) if input_norm else None
        self.adapters = nn.ModuleDict({d: Adapter() for d in domains})
        self.norms = nn.ModuleDict({d: nn.LayerNorm(LATENT) for d in domains})
        self.heads = nn.ModuleDict({d: nn.Linear(LATENT, n) for d, (_, n) in domains.items()})
        if core_delta:
            for n, p in list(self.core.named_parameters()):
                self.register_buffer(f"anchor_{n}", p.detach().clone())
                p.requires_grad_(False)
            self.anchor_keys = [n for n, _ in self.core.named_parameters()]
            self.delta = nn.ParameterDict(
                {
                    n.replace(".", "_"): nn.Parameter(torch.zeros_like(p))
                    for n, p in self.core.named_parameters()
                }
            )
        core_names = _names(self.core, "core")
        if core_delta:
            self.param_groups = {"fast_core": [f"delta.{n.replace('.', '_')}" for n in self.anchor_keys]}
        elif split_core:
            self.param_groups = {
                "fast_core_ih": [n for n in core_names if "weight_ih" in n or "bias_ih" in n],
                "fast_core_hh": [n for n in core_names if "weight_hh" in n or "bias_hh" in n],
            }
        else:
            self.param_groups = {"fast_core": core_names}
        for d in domains:
            self.param_groups[f"proj_conv.{d}"] = [f"projs.{d}.conv.weight", f"projs.{d}.conv.bias"]
            self.param_groups[f"proj_lin.{d}"] = [f"projs.{d}.lin.weight", f"projs.{d}.lin.bias"]
            if input_norm:
                self.param_groups[f"input_norm.{d}"] = _names(self.in_norms[d], f"in_norms.{d}")
            self.param_groups[f"adapter.{d}"] = _names(self.adapters[d], f"adapters.{d}")
            self.param_groups[f"norm.{d}"] = _names(self.norms[d], f"norms.{d}")
            self.param_groups[f"head.{d}"] = _names(self.heads[d], f"heads.{d}")

    def bounded_delta(self, n):
        return TAU * torch.tanh(self.delta[n.replace(".", "_")] / TAU)

    def drift(self):
        if not self.core_delta:
            return 0.0
        with torch.no_grad():
            return float(torch.sqrt(sum((self.bounded_delta(n) ** 2).sum() for n in self.anchor_keys)))

    def restore_anchor(self, frac=1.0):
        if self.core_delta:
            with torch.no_grad():
                for k in self.delta:
                    self.delta[k].mul_(1.0 - frac)

    def features(self, x, d):
        z = self.projs[d](x)
        if self.input_norm:
            z = self.in_norms[d](z)
        if self.core_delta:
            eff = {n: getattr(self, f"anchor_{n}") + self.bounded_delta(n) for n in self.anchor_keys}
            o, _ = torch.func.functional_call(self.core, eff, (z,))
        else:
            o, _ = self.core(z)
        return o

    def forward(self, x, d, update_recent: bool = False):
        last = self.features(x, d)[:, -1]
        return self.heads[d](self.adapters[d](self.norms[d](last))), None


class ArchXD(nn.Module):
    """Cross domain arm fixture. It deliberately contains every structure the eleven repair arms need to be
    distinguishable: a shared fast recurrent core, a shared slow workspace, a domain local adapter, and a
    domain local slow state. The shared slow workspace is present as a control, not as a proposal: prior
    evidence already found shared slow state harmful, and this architecture is what lets an arm isolate it.
    """

    name = "XD"

    def __init__(self, domains: dict[str, tuple[int, int]]):
        super().__init__()
        self.domains = list(domains)
        self.projs = nn.ModuleDict({d: Projection(ch) for d, (ch, _) in domains.items()})
        self.core = nn.GRU(LATENT, LATENT, batch_first=True)  # shared fast
        self.slow = nn.Sequential(
            nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT)
        )  # shared slow
        self.adapters = nn.ModuleDict({d: Adapter() for d in domains})  # domain local adapter
        self.local_slow = nn.ModuleDict(  # domain local slow state
            {
                d: nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT))
                for d in domains
            }
        )
        self.heads = nn.ModuleDict({d: nn.Linear(LATENT * 2, n) for d, (_, n) in domains.items()})
        self.param_groups = {
            "fast_core": _names(self.core, "core"),
            "slow_core": _names(self.slow, "slow"),
        }
        for d in domains:
            self.param_groups[f"proj_conv.{d}"] = [f"projs.{d}.conv.weight", f"projs.{d}.conv.bias"]
            self.param_groups[f"proj_lin.{d}"] = [f"projs.{d}.lin.weight", f"projs.{d}.lin.bias"]
            self.param_groups[f"adapter.{d}"] = _names(self.adapters[d], f"adapters.{d}")
            self.param_groups[f"local_slow.{d}"] = _names(self.local_slow[d], f"local_slow.{d}")
            self.param_groups[f"head.{d}"] = _names(self.heads[d], f"heads.{d}")

    def features(self, x, d):
        o, _ = self.core(self.projs[d](x))
        return o

    def forward(self, x, d, update_recent: bool = False):
        last = self.features(x, d)[:, -1]
        shared = self.slow(last)
        local = self.local_slow[d](self.adapters[d](last))
        return self.heads[d](torch.cat([shared, local], 1)), None


class Conventional(nn.Module):
    """The conventional comparison family. One class, distinct configurations, distinct trainable groups.

    core   gru | lstm            recurrent core type
    share  True                  one core shared by all domains (matched capacity multi domain learner)
           False                 an independent core per domain (separate per domain models)
    adapt  True                  add a domain local adapter (shared core plus adapters)
    """

    def __init__(
        self,
        domains: dict[str, tuple[int, int]],
        core: str = "lstm",
        share: bool = True,
        adapt: bool = False,
        hidden: int = LATENT,
    ):
        super().__init__()
        self.name = f"{core}{'_shared' if share else '_separate'}{'_adapters' if adapt else ''}"
        self.domains = list(domains)
        self.share, self.adapt = share, adapt
        make = (
            (lambda: nn.GRU(LATENT, hidden, batch_first=True))
            if core == "gru"
            else (lambda: nn.LSTM(LATENT, hidden, batch_first=True))
        )
        self.projs = nn.ModuleDict({d: Projection(ch) for d, (ch, _) in domains.items()})
        self.cores = nn.ModuleDict({"shared": make()} if share else {d: make() for d in domains})
        self.adapters = nn.ModuleDict({d: Adapter(hidden) for d in domains}) if adapt else None
        self.heads = nn.ModuleDict({d: nn.Linear(hidden, n) for d, (_, n) in domains.items()})
        self.param_groups = {}
        if share:
            self.param_groups["fast_core"] = _names(self.cores["shared"], "cores.shared")
        for d in domains:
            if not share:
                self.param_groups[f"fast_core.{d}"] = _names(self.cores[d], f"cores.{d}")
            self.param_groups[f"proj_conv.{d}"] = [f"projs.{d}.conv.weight", f"projs.{d}.conv.bias"]
            self.param_groups[f"proj_lin.{d}"] = [f"projs.{d}.lin.weight", f"projs.{d}.lin.bias"]
            if adapt:
                self.param_groups[f"adapter.{d}"] = _names(self.adapters[d], f"adapters.{d}")
            self.param_groups[f"head.{d}"] = _names(self.heads[d], f"heads.{d}")

    def features(self, x, d):
        o, _ = self.cores["shared" if self.share else d](self.projs[d](x))
        return o

    def forward(self, x, d, update_recent: bool = False):
        h = self.features(x, d)[:, -1]
        if self.adapt:
            h = self.adapters[d](h)
        return self.heads[d](h), None


class BagProjection(nn.Module):
    """Per timestep projection with no temporal operator at all.

    Bounded repair. The first version of the order free control reused Projection, whose Conv1d has a kernel
    width of five and therefore reads local temporal order. That control then beat every recurrent model on
    both principal domains, which would have been read as those domains being order insensitive. The defect
    was in the control, not in the domains. This projection is a per timestep linear map, so pooling its
    output over time is genuinely invariant to the order of the timesteps.
    """

    def __init__(self, ch: int, latent: int = LATENT):
        super().__init__()
        self.a = nn.Linear(ch, 32)
        self.b = nn.Linear(32, latent)

    def forward(self, x):
        return F.relu(self.b(F.relu(self.a(x))))


class BagControl(nn.Module):
    """Order free control. Required by the temporal validity gate, never a principal arm."""

    def __init__(self, domains: dict[str, tuple[int, int]]):
        super().__init__()
        self.domains = list(domains)
        self.projs = nn.ModuleDict({d: BagProjection(ch) for d, (ch, _) in domains.items()})
        self.trunk = nn.Sequential(nn.Linear(LATENT * 3, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT))
        self.heads = nn.ModuleDict({d: nn.Linear(LATENT, n) for d, (_, n) in domains.items()})
        self.param_groups = {"fast_core": _names(self.trunk, "trunk")}
        for d in domains:
            self.param_groups[f"proj_conv.{d}"] = [f"projs.{d}.a.weight", f"projs.{d}.a.bias"]
            self.param_groups[f"proj_lin.{d}"] = [f"projs.{d}.b.weight", f"projs.{d}.b.bias"]
            self.param_groups[f"head.{d}"] = _names(self.heads[d], f"heads.{d}")

    def features(self, x, d):
        return self.projs[d](x)

    def forward(self, x, d, update_recent: bool = False):
        z = self.projs[d](x)
        return self.heads[d](self.trunk(torch.cat([z.mean(1), z.std(1), z.amax(1)], 1))), None


BUILDERS = {
    "G": lambda dom: ArchG(dom),
    "H": lambda dom: ArchH(dom),
    "XD": lambda dom: ArchXD(dom),
    "G_R1_domain_input_norm": lambda dom: ArchGR(dom, input_norm=True),
    "G_R2_bounded_core_delta": lambda dom: ArchGR(dom, core_delta=True),
    "G_R3_split_core": lambda dom: ArchGR(dom, split_core=True),
    "gru": lambda dom: Conventional(dom, core="gru", share=True),
    "lstm": lambda dom: Conventional(dom, core="lstm", share=True),
    "separate": lambda dom: Conventional(dom, core="lstm", share=False),
    "shared_heads": lambda dom: Conventional(dom, core="gru", share=True),
    "shared_adapters": lambda dom: Conventional(dom, core="lstm", share=True, adapt=True),
    "matched_capacity": lambda dom: Conventional(dom, core="lstm", share=True, hidden=96),
    "bag": lambda dom: BagControl(dom),
}


def build(kind: str, domains: dict[str, tuple[int, int]]) -> nn.Module:
    return BUILDERS[kind](domains)


def check_partition(model: nn.Module) -> tuple[bool, list[str], list[str]]:
    """Every trainable parameter belongs to exactly one declared group."""
    declared: list[str] = []
    for names in model.param_groups.values():
        declared.extend(names)
    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    undeclared = sorted(actual - set(declared))
    duplicated = sorted({n for n in declared if declared.count(n) > 1})
    return (not undeclared and not duplicated), undeclared, duplicated


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
