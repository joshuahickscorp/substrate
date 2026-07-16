from __future__ import annotations

import copy

import pytest
import torch

from mop.studies import generation1_c3_router_redesign as redesign


def test_geometry_features_are_deterministic_and_finite() -> None:
    generator = torch.Generator().manual_seed(17)
    xtr = torch.randn(30, 6, generator=generator)
    ytr = torch.arange(30) % 3
    xte = torch.randn(7, 6, generator=generator)

    for name in redesign.FEATURE_SETS:
        first = redesign.router_features(
            name, xtr, ytr, xte, difficulty_index=2, difficulty_count=5, n_classes=3
        )
        second = redesign.router_features(
            name, xtr, ytr, xte, difficulty_index=2, difficulty_count=5, n_classes=3
        )
        assert torch.equal(first, second)
        assert first.shape[0] == 7
        assert torch.isfinite(first).all()


def test_tiny_redesign_is_sealed_deterministic_and_nonpromotable() -> None:
    variants = redesign.variant_grid()[:2]
    config = redesign.redesign_config(
        train_seed_count=1,
        heldout_seed_count=1,
        difficulty_indices=(0,),
        n_train=36,
        n_test=18,
        n_classes=3,
        dim=4,
        actor_epochs=1,
        variants=variants,
    )
    first = redesign.run_redesign(config)
    second = redesign.run_redesign(config)
    redesign.validate_result(first, config)

    assert first == second
    assert first["activation_allowed"] is False
    assert first["scientific_promotion"] is False
    assert first["decision"]["ready_for_confirmatory_claim"] is False
    assert first["grid"]["shared_actor_evaluation"] is True
    forbidden = set(redesign.FORBIDDEN_HELDOUT_INPUTS)
    assert all(not (set(cell) & forbidden) for cell in first["cells"])


def test_config_and_result_reject_leakage_or_mutation() -> None:
    _, c2_config = redesign.d1.load_c2_authority()
    leaking = redesign.redesign_config()
    leaking["visible_inputs"].append("context_id")
    with pytest.raises(ValueError, match="visible-input contract"):
        redesign.validate_config(leaking, c2_config)

    config = redesign.redesign_config(
        train_seed_count=1,
        heldout_seed_count=1,
        difficulty_indices=(0,),
        n_train=30,
        n_test=15,
        n_classes=3,
        dim=4,
        actor_epochs=1,
        variants=redesign.variant_grid()[:1],
    )
    result = redesign.run_redesign(config)
    mutant = copy.deepcopy(result)
    mutant["activation_allowed"] = True
    with pytest.raises(ValueError, match="self-seal"):
        redesign.validate_result(mutant, config)
