from __future__ import annotations

import copy

import pytest

from mop.studies import generation1_c3_dispatch as dispatch


def test_exact_c2_authority_is_clean_and_non_authorizing() -> None:
    binding, config = dispatch.load_c2_authority()

    assert binding["result_file_sha256"] == dispatch.C2_RESULT_FILE_SHA256
    assert binding["verification_file_sha256"] == dispatch.C2_VERIFICATION_FILE_SHA256
    assert binding["c2_ready_to_preregister"] is True
    assert binding["c2_ready_to_train_confirmatory_dispatcher"] is False
    assert config["seed_count"] == 8192


def test_pilot_is_deterministic_sealed_and_holds_leakage_boundary() -> None:
    config = dispatch.pilot_config(n_train=36, n_test=18, dim=4, router_epochs=2)

    first = dispatch.run_pilot(config)
    second = dispatch.run_pilot(config)
    dispatch.validate_result(first, config)

    assert first == second
    assert first["result_sha256"] == second["result_sha256"]
    assert first["activation_allowed"] is False
    assert first["scientific_promotion"] is False
    assert first["decision"]["ready_for_confirmatory_claim"] is False
    assert first["heldout_contract"]["visible_inputs"] == ["latent_vector", "difficulty_index"]
    forbidden = set(dispatch.FORBIDDEN_HELDOUT_INPUTS)
    assert all(not (set(cell) & forbidden) for cell in first["cells"])


def test_config_rejects_seed_overlap_and_router_leakage() -> None:
    _, c2_config = dispatch.load_c2_authority()
    overlapping = dispatch.pilot_config(heldout_seed_start=20270001)
    with pytest.raises(ValueError, match="fresh and disjoint"):
        dispatch.validate_config(overlapping, c2_config)

    leaking = dispatch.pilot_config()
    leaking["router"]["visible_inputs"].append("context_id")
    with pytest.raises(ValueError, match="visible-input contract"):
        dispatch.validate_config(leaking, c2_config)


def test_result_mutation_is_rejected() -> None:
    config = dispatch.pilot_config(n_train=36, n_test=18, dim=4, router_epochs=2)
    result = dispatch.run_pilot(config)
    mutant = copy.deepcopy(result)
    mutant["activation_allowed"] = True

    with pytest.raises(ValueError, match="self-seal"):
        dispatch.validate_result(mutant, config)
