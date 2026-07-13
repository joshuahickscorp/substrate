from __future__ import annotations

import torch
from omegaconf import OmegaConf

from mop.devices import resolve
from mop.experiments.ex10_cross_modal import EX10, _make_modality_b, _modality_b_seed
from mop.experiments.f_form_substrate import F20, _f20_arm, _f20_probe_seed, _f20_score
from mop.seeding import UINT32_MAX, derive_seed32, seed_everything

HIGH_GENERATION1_SEED = 1_955_868_442


def test_seed32_preserves_legacy_streams_and_hashes_full_oversized_seed() -> None:
    assert derive_seed32(4_001, "legacy") == 4_001

    oversized = HIGH_GENERATION1_SEED * 1000 + 7
    child = derive_seed32(oversized, "generation1-regression")
    assert 0 <= child <= UINT32_MAX
    assert child == derive_seed32(oversized, "generation1-regression")
    assert child != derive_seed32(oversized + 1, "generation1-regression")
    assert child != derive_seed32(oversized, "other-role")
    seed_everything(child)


def test_ex10_high_generation1_seed_has_distinct_deterministic_domain_streams() -> None:
    assert HIGH_GENERATION1_SEED * 1000 > UINT32_MAX
    assert _modality_b_seed(3, 2) == 3_002
    seeds = [_modality_b_seed(HIGH_GENERATION1_SEED, domain) for domain in range(4)]
    assert all(0 <= seed <= UINT32_MAX for seed in seeds)
    assert seeds == [_modality_b_seed(HIGH_GENERATION1_SEED, domain) for domain in range(4)]
    assert len(set(seeds)) == len(seeds)

    x = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    first = _make_modality_b(x, dim=8, hidden=6, seed=seeds[0])
    repeat = _make_modality_b(x, dim=8, hidden=6, seed=seeds[0])
    other_domain = _make_modality_b(x, dim=8, hidden=6, seed=seeds[1])
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other_domain)


def test_ex10_run_accepts_high_generation1_seed_end_to_end(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {
                "seeds": [HIGH_GENERATION1_SEED],
                "dim": 8,
                "hidden": 8,
                "n_classes": 2,
                "n_domains": 2,
                "samples": 32,
                "separation": 2.0,
                "epochs": 2,
                "lr": 0.01,
                "aux_weight": 0.5,
                "margin": 0.02,
                "transfer_margin": 0.05,
            }
        }
    )

    result = EX10().run(cfg, resolve("cpu"), tmp_path)

    assert result["seeds"] == [HIGH_GENERATION1_SEED]
    assert isinstance(result["null_supported"], bool)


def test_f20_high_generation1_seed_has_distinct_deterministic_probe_streams() -> None:
    assert _f20_probe_seed(4, 34) == 4_034
    seeds = [_f20_probe_seed(HIGH_GENERATION1_SEED, index) for index in range(1, 35)]
    assert all(0 <= seed <= UINT32_MAX for seed in seeds)
    assert seeds == [_f20_probe_seed(HIGH_GENERATION1_SEED, index) for index in range(1, 35)]
    assert len(set(seeds)) == len(seeds)

    kwargs = {
        "n": 40,
        "dim": 8,
        "classes": 2,
        "nuis_classes": 2,
        "signal": 1.0,
        "nuisance": 0.5,
        "rho": 0.8,
        "noise": 0.2,
    }
    first = _f20_arm(**kwargs, seed=seeds[0])
    repeat = _f20_arm(**kwargs, seed=seeds[0])
    other_probe = _f20_arm(**kwargs, seed=seeds[1])
    assert all(torch.equal(left, right) for left, right in zip(first, repeat, strict=True))
    assert not torch.equal(first[0], other_probe[0])

    score = _f20_score(
        *first,
        classes=2,
        epochs=2,
        lr=0.01,
        train_frac=0.6,
        seed=seeds[0],
    )
    assert set(score) == {"crisis_score", "raw_error", "confidence", "decorr_acc"}


def test_f20_run_accepts_high_generation1_seed_end_to_end(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {
                "seeds": [HIGH_GENERATION1_SEED],
                "classes": 2,
                "nuis_classes": 2,
                "samples": 40,
                "dim": 8,
                "epochs": 2,
                "lr": 0.01,
                "train_frac": 0.6,
                "rho": 0.8,
                "world_noise": 0.2,
                "nuisance_crisis": 1.0,
                "signal_wall": 0.4,
                "nuisance_wall": 0.8,
                "wall_noise": 0.5,
                "signal_healthy": 1.0,
                "nuisance_side": 0.5,
                "n_crisis": 2,
                "n_wall": 2,
                "n_healthy": 2,
                "n_noise": 2,
                "fail_threshold": 0.42,
                "crisis_trigger_threshold": 0.2,
                "error_threshold": 0.5,
                "shell_scaleup_flops": 1000.0,
                "margin": 0.1,
                "noise_margin": 0.12,
            }
        }
    )

    result = F20().run(cfg, resolve("cpu"), tmp_path)

    assert result["seeds"] == [HIGH_GENERATION1_SEED]
    assert isinstance(result["null_supported"], bool)
