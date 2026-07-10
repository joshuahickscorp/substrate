"""E6: relational map over JEPA latents, run as the 2-vs-2.1 (pooled-vs-dense) contrast.

The corpus claim (Experiment 6): explicit relational structure improves generalization to
novel recombinations, and the help is much larger on V-JEPA 2.1's dense per-token features
than on pooled V-JEPA 2. The recurring null is that a structured head only "adds capacity,
not structure", which a parameter-matched flat baseline catches, and that the pooled latent
does not factor into objects/relations at all (the 2-only failure).

We make the comparison STRUCTURE the deliverable, not the absolute number. Two encoders are
loaded (pooled vjepa2, dense vjepa21). Synthetic clips carry a composite class (attr_a,
attr_b); a relational/recombination task trains on a subset of (a,b) pairs and tests on
HELD-OUT pairs (a flat model is expected to fail recombination). On each encoder's latents we
fit a STRUCTURED head (MLP over pairwise/relational features of the two latents) and a
parameter-matched FLAT head (MLP over the plain concatenation), and report structured-minus-
flat on pooled vs dense plus the headline 2-vs-2.1 delta of those deltas.

The registered path is now cache-first and token-aware. The old direct synthetic comparison remains
only behind an explicit ``allow_legacy_fixture=true`` override for regression tests. It can never
be entered as a fallback when the official learned/random cache pair is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch import nn  # noqa: E402

from ..devices import DeviceInfo, safe_to  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..substrate.encoder import load_encoder  # noqa: E402
from .base import Experiment  # noqa: E402

POOLED_CFG = "configs/encoder/vjepa2_vitl_fpc64_256.yaml"
DENSE_CFG = "configs/encoder/vjepa21_vitl.yaml"


class _Flat(nn.Module):
    """Parameter-matched flat baseline: MLP over the plain concatenation [a||b]."""

    def __init__(self, dim: int, n_classes: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b], dim=-1))


class _Structured(nn.Module):
    """Relational head: MLP over pairwise/relational features (a, b, a*b, a-b). The product
    and difference expose the relation between the two latents, which is what recombination
    generalization needs and what a flat concatenation does not surface."""

    def __init__(self, dim: int, n_classes: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        rel = torch.cat([a, b, a * b, a - b], dim=-1)
        return self.net(rel)


def _matched_hidden(dim: int, flat_hidden: int) -> int:
    """Pick the structured head's hidden width so its parameter count matches the flat head.
    Flat first layer: 2*dim*H_flat; structured: 4*dim*H_struct. Halving the width matches the
    first layer; the output layer differs only by the small head bias/weights, kept comparable."""
    return max(4, flat_hidden // 2)


class E6(Experiment):
    id = "e6_relational"
    metric = ("dense_primary_delta", "position_specificity_delta", "scientific_promotion")
    baseline = "strongest parameter-matched learned-flat, token-shuffle, or random-init cache control"
    ablation = "fixed token-position bins versus token-blind pooling on exact matched ViT-B caches"
    null_hypothesis = (
        "the learned token-aware readout ties the strongest parameter-matched learned-flat, "
        "token-shuffle, or exact-architecture random-init control on held-out combinations"
    )
    tier = "env-later"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        execution_path = str(e.get("execution_path", "cache-first"))
        if execution_path == "cache-first":
            return self._run_cache_first(e, run_dir)
        if execution_path != "legacy-fixture" or not bool(e.get("allow_legacy_fixture", False)):
            raise RuntimeError(
                "E6 refuses legacy encoder fallback; use cache-first or explicitly authorize "
                "the non-promotable legacy fixture"
            )
        seed = int(cfg.seed)
        seed_everything(seed)

        pooled_enc = load_encoder(OmegaConf.load(POOLED_CFG))
        dense_enc = load_encoder(OmegaConf.load(DENSE_CFG))

        pooled = self._eval_encoder("pooled", pooled_enc, e, device, seed)
        dense = self._eval_encoder("dense", dense_enc, e, device, seed)

        pooled_delta = pooled["structured_acc"] - pooled["flat_acc"]
        dense_delta = dense["structured_acc"] - dense["flat_acc"]
        headline = dense_delta - pooled_delta

        tie_eps = float(e.tie_eps)
        structured_beats_flat = bool(dense_delta > tie_eps)
        dense_gain_larger = bool(headline > tie_eps)

        out = {
            "pooled": pooled,
            "dense": dense,
            "pooled_delta": pooled_delta,
            "dense_delta": dense_delta,
            "headline_2_vs_21_delta": headline,
            "tie_eps": tie_eps,
            # the explicit null check: both must hold to REJECT the null
            "structured_beats_flat": structured_beats_flat,
            "dense_gain_larger": dense_gain_larger,
            "null_rejected": bool(structured_beats_flat and dense_gain_larger),
            "backends": {"pooled": pooled_enc.spec.backend, "dense": dense_enc.spec.backend},
        }
        self._plot(out, run_dir)
        out["scientific_promotion"] = False
        out["claim_boundary"] = {
            "legacy_fixture": True,
            "official_vitb_result": False,
            "registered_e6_result": False,
        }
        return out

    def _run_cache_first(self, e: DictConfig, run_dir: Path) -> dict:
        from .e6_dense_relational import TokenReadoutSpec, build_pair_gate, run_dense_relational

        learned = Path(str(e.learned_cache))
        random = Path(str(e.random_cache))
        gate = build_pair_gate(learned, random)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not gate["mechanics_ok"]:
            gate.pop("_learned_private", None)
            gate.pop("_random_private", None)
            output = {
                "execution_path": "cache-first",
                "execution_status": "blocked-missing-or-invalid-cache-pair",
                "dense_primary_delta": None,
                "position_specificity_delta": None,
                "scientific_promotion": False,
                "pair_gate": gate,
                "claim_boundary": {
                    "legacy_fallback_used": False,
                    "model_loaded": False,
                    "forward_executed": False,
                    "registered_e6_result": False,
                },
            }
            (run_dir / "e6_relational.json").write_text(json.dumps(output, indent=2) + "\n")
            return output
        readout = e.cache_readout
        result = run_dense_relational(
            learned,
            random,
            spec=TokenReadoutSpec(
                bins=int(readout.bins),
                feature_rank=int(readout.feature_rank),
                summary_dim=int(readout.summary_dim),
                ridge=float(readout.ridge),
                seeds=tuple(int(value) for value in readout.seeds),
                min_margin=float(readout.min_margin),
                ceiling=float(readout.ceiling),
            ),
        )
        output = {
            **result,
            "execution_path": "cache-first",
            "execution_status": "complete",
            "dense_primary_delta": result["confidence"]["primary_delta"]["mean"],
            "position_specificity_delta": result["confidence"]["position_specificity_delta"]["mean"],
        }
        (run_dir / "e6_relational.json").write_text(json.dumps(output, indent=2) + "\n")
        return output

    def _eval_encoder(self, tag, enc, e, device: DeviceInfo, seed: int) -> dict:
        latents, comp_y, attr_a, attr_b = self._encode_latents(tag, enc, e, device, seed)
        train_idx, test_idx = self._recombination_split(attr_a, attr_b, int(e.n_attr), seed)
        dim = latents.shape[1] // 2  # per-latent dim; latents is [B, 2D] stacked (a then b)
        n_classes = int(e.n_attr) * int(e.n_attr)

        flat_acc = self._fit_head(
            _Flat(dim, n_classes, int(e.hidden)), latents, comp_y, train_idx, test_idx, e, device, seed
        )
        struct_h = _matched_hidden(dim, int(e.hidden))
        structured_acc = self._fit_head(
            _Structured(dim, n_classes, struct_h), latents, comp_y, train_idx, test_idx, e, device, seed
        )
        return {
            "structured_acc": structured_acc,
            "flat_acc": flat_acc,
            "chance": 1.0 / n_classes,
            "n_train": int(train_idx.numel()),
            "n_test_heldout_pairs": int(test_idx.numel()),
            "latent_dim": int(dim),
            "flat_params": _nparams(_Flat(dim, n_classes, int(e.hidden))),
            "structured_params": _nparams(_Structured(dim, n_classes, struct_h)),
        }

    def _encode_latents(self, tag, enc, e, device: DeviceInfo, seed):
        """Synthetic clips with a composite class. Each example is a PAIR of clips: clip_a
        carries attribute a (a per-class mean shift), clip_b carries attribute b. The pair's
        composite label is a*n_attr+b. We encode both clips and keep the two latents; the head
        must combine them to read off the composite class, which is the relational structure."""
        seed_everything(seed + (0 if tag == "pooled" else 1))
        B = int(e.samples)
        n_attr = int(e.n_attr)
        C, T, H, W = 3, 4, 16, 16

        g = torch.Generator().manual_seed(seed)
        attr_a = torch.randint(0, n_attr, (B,), generator=g)
        attr_b = torch.randint(0, n_attr, (B,), generator=g)
        comp_y = (attr_a * n_attr + attr_b).long()

        # attribute -> a fixed spatial bias pattern injected into the clip so the (random)
        # encoder still carries attribute information into the latent.
        shift = float(e.attr_shift)
        bias = torch.randn(n_attr, C, T, H, W, generator=g) * shift
        clips_a = torch.randn(B, C, T, H, W, generator=g) + bias[attr_a]
        clips_b = torch.randn(B, C, T, H, W, generator=g) + bias[attr_b]

        la = _flatten_latent(enc.encode(safe_to(clips_a, device.device)))
        lb = _flatten_latent(enc.encode(safe_to(clips_b, device.device)))
        latents = torch.cat([la, lb], dim=1).cpu()  # [B, 2D] stacked, split back in head call
        return latents, comp_y, attr_a, attr_b

    def _recombination_split(self, attr_a, attr_b, n_attr, seed):
        """Hold out a set of (a,b) attribute COMBINATIONS entirely from training. The test set
        is only those held-out combinations: a flat model that memorizes seen pairs is expected
        to fail; a relational model that generalizes over the factored attributes can transfer."""
        g = torch.Generator().manual_seed(seed + 7)
        all_pairs = [(a, b) for a in range(n_attr) for b in range(n_attr)]
        perm = torch.randperm(len(all_pairs), generator=g).tolist()
        n_held = max(1, len(all_pairs) // 3)
        held = {all_pairs[i] for i in perm[:n_held]}
        is_held = torch.tensor([(int(a), int(b)) in held for a, b in zip(attr_a, attr_b, strict=True)])
        return (~is_held).nonzero(as_tuple=True)[0], is_held.nonzero(as_tuple=True)[0]

    def _fit_head(self, head, latents, y, train_idx, test_idx, e, device: DeviceInfo, seed) -> float:
        seed_everything(seed)
        head = head.to(device.device)
        d = latents.shape[1] // 2
        X = safe_to(latents, device.device)
        Y = safe_to(y, device.device)
        opt = torch.optim.Adam(head.parameters(), lr=float(e.lr))
        loss_fn = nn.CrossEntropyLoss()
        xa, xb = X[:, :d], X[:, d:]
        for _ in range(int(e.epochs)):
            opt.zero_grad()
            logits = head(xa[train_idx], xb[train_idx])
            loss = loss_fn(logits, Y[train_idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(xa[test_idx], xb[test_idx]).argmax(dim=-1)
            acc = float((pred == Y[test_idx]).float().mean().item())
        return acc

    def _plot(self, out: dict, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        subs = ["pooled (2)", "dense (2.1)"]
        flat = [out["pooled"]["flat_acc"], out["dense"]["flat_acc"]]
        struct = [out["pooled"]["structured_acc"], out["dense"]["structured_acc"]]
        x = range(len(subs))
        ax[0].bar([i - 0.2 for i in x], flat, width=0.4, label="flat")
        ax[0].bar([i + 0.2 for i in x], struct, width=0.4, label="structured")
        ax[0].set_xticks(list(x))
        ax[0].set_xticklabels(subs)
        ax[0].set_ylabel("recombination accuracy")
        ax[0].set_title("structured vs flat per substrate")
        ax[0].legend(fontsize=8)
        deltas = [out["pooled_delta"], out["dense_delta"]]
        ax[1].bar(subs, deltas, color=["#888", "#4c78a8"])
        ax[1].axhline(0.0, color="k", linewidth=0.8)
        ax[1].set_ylabel("structured - flat delta")
        ax[1].set_title(f"headline 2-vs-2.1 delta = {out['headline_2_vs_21_delta']:.3f}")
        fig.tight_layout()
        fig.savefig(run_dir / "e6_relational.png", dpi=110)
        plt.close(fig)


def _flatten_latent(feats: torch.Tensor) -> torch.Tensor:
    """Pooled encoders return [B, D]; dense return [B, N, D]. Flatten dense tokens to [B, N*D]
    so a single head consumes either. This is the latent the relational head binds over."""
    return feats if feats.dim() == 2 else feats.reshape(feats.shape[0], -1)


def _nparams(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))
