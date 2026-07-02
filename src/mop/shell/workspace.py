"""WorkspaceShell (WP-02): PURE COMPOSITION of the existing shell pieces (predictor, heads, ensemble,
modulation) behind cfg.shell flags. No new science lives here: every module it instantiates already
exists, this class only wires them into one forward pass so the CM4/WS experiments toggle components by
config instead of duplicating training scripts. Building parallel shell classes is explicitly rejected
(13_code_scaffolding.md); extend this composition or the underlying module, never both.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import torch
from torch import nn

from .ensemble import Ensemble
from .heads import build_head
from .modulation import build_modulation
from .predictor import Predictor


class WorkspaceShell(nn.Module):
    """Frozen-latent in, dict out. Order of operations: context gating (if enabled and a context is
    given), working memory (if enabled), then predictor or ensemble-of-predictors, plus the head on the
    modulated latent. Any component may be None (identity for modulation, absent key in the output)."""

    def __init__(
        self,
        dim: int,
        *,
        predictor: Predictor | None = None,
        head: nn.Module | None = None,
        ensemble: Ensemble | None = None,
        modulation: dict[str, nn.Module] | None = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.predictor = predictor
        self.head = head
        self.ensemble = ensemble
        self.modulation = nn.ModuleDict(modulation or {})

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """x [B, D] -> dict with keys: latent (post-modulation), and when present: mem, prediction,
        disagreement (ensemble only), head (logits or (mean, logvar) per the head type)."""
        h = x
        out: dict[str, torch.Tensor] = {}
        if "context_gating" in self.modulation and context is not None:
            h = self.modulation["context_gating"](h, context)
        if "working_memory" in self.modulation:
            h, mem = self.modulation["working_memory"](h, mem)
            out["mem"] = mem
        if self.ensemble is not None:
            pred, dis = self.ensemble.mean_and_disagreement(h)
            out["prediction"], out["disagreement"] = pred, dis
        elif self.predictor is not None:
            out["prediction"] = self.predictor(h)
        if self.head is not None:
            out["head"] = self.head(h)
        out["latent"] = h
        return out

    @classmethod
    def from_cfg(cls, shell_cfg, dim: int) -> WorkspaceShell:
        """Compose from one shell bundle (configs/shell/*.yaml): predictor block always builds; the
        head builds from shell_cfg.heads; ensemble.size > 1 wraps the predictor in an Ensemble;
        modulation blocks build iff their flags are set. Exactly the existing builders, no new knobs."""
        predictor = Predictor.from_cfg(shell_cfg.predictor, dim)
        head = build_head(shell_cfg.heads, dim, int(shell_cfg.heads.n_classes), out_dim=dim)
        ensemble = None
        if int(shell_cfg.ensemble.size) > 1:
            ensemble = Ensemble(
                lambda: Predictor.from_cfg(shell_cfg.predictor, dim),
                size=int(shell_cfg.ensemble.size),
                bootstrap=bool(shell_cfg.ensemble.bootstrap),
            )
        mods = build_modulation(shell_cfg.modulation, dim)
        return cls(dim, predictor=predictor, head=head, ensemble=ensemble, modulation=mods)
