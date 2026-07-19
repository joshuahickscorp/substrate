
from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo, safe_to
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from .base import Experiment


def _nparams(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


class _SlotAttention(nn.Module):

    def __init__(self, dim: int, n_slots: int, hidden: int, n_classes: int):
        super().__init__()
        self.n_slots = n_slots
        self.hidden = hidden
        self.in_proj = nn.Linear(dim, hidden)
        self.slots = nn.Parameter(torch.randn(n_slots, hidden) * (hidden**-0.5))
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, n_classes)

    def hidden_repr(self, window: torch.Tensor) -> torch.Tensor:
        b = window.shape[0]
        kv = self.in_proj(window)  # [B, T, H]
        q = self.q_proj(self.slots).unsqueeze(0).expand(b, -1, -1)  # [B, K, H]
        k = self.k_proj(kv)  # [B, T, H]
        v = self.v_proj(kv)  # [B, T, H]
        attn = torch.softmax(q @ k.transpose(-1, -2) / (self.hidden**0.5), dim=-1)  # [B, K, T]
        slots_out = attn @ v  # [B, K, H]
        return slots_out.mean(dim=1)  # [B, H]

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden_repr(window))


class _FlatPooled(nn.Module):

    def __init__(self, dim: int, window_len: int, hidden: int, n_classes: int):
        super().__init__()
        self.net_in = nn.Linear(dim * window_len, hidden)
        self.act = nn.GELU()
        self.head = nn.Linear(hidden, n_classes)

    def hidden_repr(self, window: torch.Tensor) -> torch.Tensor:
        b = window.shape[0]
        return self.act(self.net_in(window.reshape(b, -1)))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden_repr(window))


def _matched_flat_hidden(dim: int, window_len: int, n_slots: int, slot_hidden: int, n_classes: int) -> int:
    target = _nparams(_SlotAttention(dim, n_slots, slot_hidden, n_classes))
    best_h, best_gap = 8, None
    for h in range(4, 4 * slot_hidden + 1, 4):
        flat = _FlatPooled(dim, window_len, h, n_classes)
        gap = abs(_nparams(flat) - target)
        if best_gap is None or gap < best_gap:
            best_gap, best_h = gap, h
    return best_h


def _relation_label(delta: torch.Tensor, n_relations: int, contain_radius: float) -> torch.Tensor:
    dist = delta.norm(dim=-1)
    band_width = contain_radius * 2.0
    banded = torch.clamp(((dist - contain_radius) / band_width).floor() + 1, min=0)
    return torch.clamp(banded, max=n_relations - 1).long()


def _make_relation_windows(
    n_windows: int,
    window_len: int,
    dim: int,
    n_relations: int,
    contain_radius: float,
    entity_speed: float,
    pool_noise: float,
    change_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    n_changed = int(round(n_windows * change_frac))
    changed_flags = torch.zeros(n_windows, dtype=torch.bool)
    changed_flags[:n_changed] = True
    changed_flags = changed_flags[torch.randperm(n_windows, generator=g)]

    windows = torch.zeros(n_windows, window_len, dim)
    labels = torch.zeros(n_windows, dtype=torch.long)

    for i in range(n_windows):
        pos_a = torch.randn(dim, generator=g) * 2.0
        pos_b = pos_a + torch.randn(dim, generator=g) * (contain_radius * 3.0)
        vel_a = torch.randn(dim, generator=g) * entity_speed
        vel_b = torch.randn(dim, generator=g) * entity_speed
        changepoint = int(torch.randint(1, max(2, window_len - 1), (1,), generator=g).item())
        for t in range(window_len):
            pos_a = pos_a + vel_a
            pos_b = pos_b + vel_b
            if bool(changed_flags[i]) and t == changepoint:
                jump = torch.randn(dim, generator=g)
                jump = jump / jump.norm().clamp(min=1e-8)
                pos_b = pos_a + jump * (contain_radius * 6.0)
            clip = pos_a + pos_b + torch.randn(dim, generator=g) * pool_noise
            windows[i, t] = clip
        labels[i] = _relation_label((pos_b - pos_a).unsqueeze(0), n_relations, contain_radius)[0]

    return windows, labels, changed_flags


class EX9(Experiment):
    id = "ex9_slot_attention"
    metric = ("relation_decoding", "pooled_ceiling_gap")
    baseline = "parameter-matched-flat: flat-pooled MLP over the concatenated window, same total params"
    ablation = "K learned slot queries cross-attend over the window vs the flat-pooled baseline"
    null_hypothesis = (
        "slot attention over pooled latents ties the parameter-matched flat baseline; pooled "
        "features carry no per-slot structure to factor"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seed = int(cfg.seed)
        seed_everything(seed)

        dim = int(e.dim)
        window_len = int(e.window_len)
        n_relations = int(e.n_relations)
        n_windows = int(e.n_windows)
        n_slots = int(e.n_slots)
        slot_hidden = int(e.slot_hidden)
        tie_eps = float(e.tie_eps)

        t0 = time.perf_counter()
        windows, labels, changed = _make_relation_windows(
            n_windows=n_windows,
            window_len=window_len,
            dim=dim,
            n_relations=n_relations,
            contain_radius=float(e.contain_radius),
            entity_speed=float(e.entity_speed),
            pool_noise=float(e.pool_noise),
            change_frac=float(e.change_frac),
            seed=seed,
        )
        gen_wall = time.perf_counter() - t0

        flat_hidden = _matched_flat_hidden(dim, window_len, n_slots, slot_hidden, n_relations)

        slot_arm = self._fit_and_probe(
            _SlotAttention(dim, n_slots, slot_hidden, n_relations), windows, labels, e, device, seed
        )
        flat_arm = self._fit_and_probe(
            _FlatPooled(dim, window_len, flat_hidden, n_relations), windows, labels, e, device, seed
        )

        flat_windows = windows.reshape(n_windows, -1)
        fr_proj = frozen_random_projection(flat_windows, seed=seed)
        fr_probe = linear_probe(fr_proj, labels, classification=True, seed=seed)
        raw_probe = linear_probe(flat_windows, labels, classification=True, seed=seed)

        relation_decoding = {
            "slot": slot_arm["probe"]["score"],
            "flat": flat_arm["probe"]["score"],
            "frozen_random": fr_probe["score"],
            "raw_concat_ceiling": raw_probe["score"],
            "chance": fr_probe["chance"],
        }
        pooled_ceiling_gap = slot_arm["probe"]["score"] - flat_arm["probe"]["score"]

        slot_beats_flat = bool(pooled_ceiling_gap > tie_eps)
        slot_beats_frozen_random = bool(slot_arm["probe"]["score"] - fr_probe["score"] > tie_eps)
        flat_beats_frozen_random = bool(flat_arm["probe"]["score"] - fr_probe["score"] > tie_eps)

        null_supported = not (slot_beats_flat and slot_beats_frozen_random)

        out = {
            "relation_decoding": relation_decoding,
            "pooled_ceiling_gap": round(pooled_ceiling_gap, 4),
            "slot_beats_flat": slot_beats_flat,
            "slot_beats_frozen_random": slot_beats_frozen_random,
            "flat_beats_frozen_random": flat_beats_frozen_random,
            "tie_eps": tie_eps,
            "null_supported": bool(null_supported),
            "arms": {
                "slot": {"acc": slot_arm["probe"]["score"], "params": slot_arm["params"]},
                "flat": {"acc": flat_arm["probe"]["score"], "params": flat_arm["params"]},
            },
            "params_match_ratio": round(flat_arm["params"] / max(1, slot_arm["params"]), 4),
            "config": {
                "dim": dim,
                "window_len": window_len,
                "n_relations": n_relations,
                "n_windows": n_windows,
                "n_slots": n_slots,
                "slot_hidden": slot_hidden,
                "flat_hidden": flat_hidden,
                "change_frac": float(e.change_frac),
                "n_changed_windows": int(changed.sum().item()),
            },
            "wall_seconds": {
                "window_generation": round(gen_wall, 3),
                "slot_fit": slot_arm["fit_wall"],
                "flat_fit": flat_arm["fit_wall"],
            },
            "seed": seed,
        }
        return out

    def _fit_and_probe(
        self,
        model: _SlotAttention | _FlatPooled,
        windows: torch.Tensor,
        labels: torch.Tensor,
        e,
        device: DeviceInfo,
        seed: int,
    ) -> dict:
        seed_everything(seed)
        model.to(device.device)
        n = windows.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        cut = int(n * 0.7)
        train_idx, test_idx = perm[:cut], perm[cut:]

        x = safe_to(windows, device.device)
        y = safe_to(labels, device.device)
        opt = torch.optim.Adam(model.parameters(), lr=float(e.lr))
        loss_fn = nn.CrossEntropyLoss()

        t0 = time.perf_counter()
        for _ in range(int(e.epochs)):
            opt.zero_grad()
            logits = model(x[train_idx])
            loss = loss_fn(logits, y[train_idx])
            loss.backward()
            opt.step()
        fit_wall = round(time.perf_counter() - t0, 3)

        model.eval()
        with torch.no_grad():
            hid_train = model.hidden_repr(x[train_idx]).cpu()
            hid_test = model.hidden_repr(x[test_idx]).cpu()
        hid = torch.cat([hid_train, hid_test], dim=0)
        y_reordered = torch.cat([y[train_idx].cpu(), y[test_idx].cpu()], dim=0)
        probe = linear_probe(hid, y_reordered, classification=True, seed=seed)
        return {"probe": probe, "params": _nparams(model), "fit_wall": fit_wall}
