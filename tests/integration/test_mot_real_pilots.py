
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from mop.devices import resolve
from mop.substrate.latent_store import LatentStore

torch.set_num_threads(2)  # live-state rule: an encode may own the CPU

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
DIM, N_CLASSES, PER_CLASS = 16, 4, 6

TINY = dict(
    seeds=[0, 1],
    cache="tiny_real",
    n_tasks=2,
    train_frac=0.75,
    epochs_per_task=2,
    lr=1e-2,
)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dev():
    return resolve("cpu")


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cache")
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(N_CLASSES, DIM, generator=g) * 3.0
    y = torch.arange(N_CLASSES).repeat_interleave(PER_CLASS)
    x = centers[y] + 0.5 * torch.randn(len(y), DIM, generator=g)
    store = LatentStore.create(
        root, "tiny_real", feat_shape=(DIM,), capacity=len(y), key_dim=DIM, has_labels=True
    )
    store.write_batch(0, x.numpy(), x.numpy(), y.numpy())
    store.finalize()
    return root


def test_dr2_sparse_real_pilot_runs(dev, cache_dir, tmp_path):
    mod = load_script("mop_dr2_sparse_real_pilot")
    cfg = mod.default_cfg(
        **TINY, data_dir=str(cache_dir), hidden=8, kwta_k=2, n_experts=2, l1_grid=[1e-3, 1e-2]
    )
    out = mod.MotDR2SparseRealPilot().run(cfg, dev, tmp_path)
    assert out["experiment"] == "mop_dr2_sparse_real" and out["pilot"] is True
    assert out["contract"]["null_hypothesis"]
    assert isinstance(out["null_supported"], bool)
    assert out["param_counts"]["kwta"] == out["param_counts"]["dense"]
    assert out["param_match_ok"] is True
    for arm in ("dense", "dense_sparse", "kwta", "moe"):
        assert len(out["bwt_per_seed"][arm]) == 2
    for arm in ("kwta", "moe"):
        v = out["delta_bwt_vs_best_dense_control"][arm]
        assert set(v) >= {"per_seed", "ci", "sign_flips", "win"}
    assert all(l1 in (1e-3, 1e-2) for l1 in out["chosen_l1_per_seed"])


def test_ws5_slot_ablation_pilot_runs(dev, cache_dir, tmp_path):
    mod = load_script("mop_ws5_slot_ablation_pilot")
    cfg = mod.default_cfg(**TINY, data_dir=str(cache_dir), hidden=8, n_experts=2)
    out = mod.MotWS5SlotAblationPilot().run(cfg, dev, tmp_path)
    assert out["experiment"] == "mop_ws5_router_slot" and out["pilot"] is True
    assert isinstance(out["null_supported"], bool)
    assert out["param_match_ok"] is True
    assert len(out["bwt_per_seed"]["full"]) == len(out["bwt_per_seed"]["ablated"]) == 2
    assert set(out["delta_bwt_full_vs_ablated"]) >= {"per_seed", "ci", "sign_flips", "win"}
    assert out["arms"]["full"]["routing_entropy"]["n"] == 2


def test_ws5_slot_read_actually_ablated(cache_dir):
    mod = load_script("mop_ws5_slot_ablation_pilot")
    torch.manual_seed(0)
    net = mod.SlotRoutedNet(DIM, N_CLASSES, n_experts=2, expert_hidden=3, use_slot=False)
    x = torch.randn(5, DIM)
    before = net(x)
    with torch.no_grad():
        net.slot.add_(100.0)
    assert torch.allclose(before, net(x))
    torch.manual_seed(0)
    full = mod.SlotRoutedNet(DIM, N_CLASSES, n_experts=2, expert_hidden=3, use_slot=True)
    ref = full(x)
    with torch.no_grad():
        full.slot.add_(100.0)
    assert not torch.allclose(ref, full(x))  # the full arm genuinely reads the slot


def test_cm4_workspace_pilot_runs(dev, cache_dir, tmp_path):
    mod = load_script("mop_cm4_workspace_pilot")
    cfg = mod.default_cfg(
        **TINY, data_dir=str(cache_dir), proj_dim=8, ws_head_hidden=8, wm_slots=2, capmatch_tol=0.06
    )
    out = mod.MotCM4WorkspacePilot().run(cfg, dev, tmp_path)
    assert out["experiment"] == "mop_cm4_workspace_shell" and out["pilot"] is True
    assert isinstance(out["null_supported"], bool)
    rec = out["capacity_and_compute_matching"]
    assert rec["flops_matched_within"]["matched"] is True  # the depth arm is genuinely FLOP-matched
    assert rec["control_input_dim"] == 8 + TINY["n_tasks"]
    assert abs(rec["dense_params"] - rec["workspace_params"]) <= 0.06 * rec["workspace_params"]
    assert rec["unrolled_steps"] >= 2  # genuinely iterative, not a renamed single pass
    for arm in ("workspace", "dense", "unrolled"):
        assert len(out["bwt_per_seed"][arm]) == 2
    assert out["planning_gain"] is None and "DR1" in out["planning_gain_blocked_reason"]
    assert set(out["delta_bwt_ws_vs_best_control"]) >= {"per_seed", "ci", "sign_flips", "win"}


def test_pilots_fail_loudly_on_missing_cache(dev, tmp_path):
    mod = load_script("mop_dr2_sparse_real_pilot")
    cfg = mod.default_cfg(**{**TINY, "cache": "no_such_cache"}, data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        mod.MotDR2SparseRealPilot().run(cfg, dev, tmp_path)
