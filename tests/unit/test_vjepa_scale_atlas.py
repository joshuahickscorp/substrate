import pytest
import torch

from mop.studies.vjepa_scale_atlas import _control_matches, frozen_split_probe, validate_shared_referents


def _row(referents=("a", "b")):
    return {
        "referents": list(referents),
        "factor_a": torch.tensor([0, 1]),
        "factor_b": torch.tensor([1, 0]),
        "splits": {"train": [0], "test": [1]},
    }


def test_shared_referent_contract_accepts_exact_match():
    validate_shared_referents({"small": _row(), "large": _row()})


def test_shared_referent_contract_rejects_reordered_rows():
    with pytest.raises(ValueError, match="referent order mismatch"):
        validate_shared_referents({"small": _row(), "large": _row(("b", "a"))})


def test_shared_referent_contract_rejects_factor_drift():
    wrong = _row()
    wrong["factor_a"] = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="factor_a mismatch"):
        validate_shared_referents({"small": _row(), "large": wrong})


def test_shared_referent_contract_rejects_unassigned_rows():
    wrong = _row()
    wrong["splits"] = {"train": [0], "test": []}
    with pytest.raises(ValueError, match="non-empty frozen val or test"):
        validate_shared_referents({"small": wrong, "large": wrong})


def test_frozen_split_probe_never_resplits_rows():
    x = torch.tensor([[-2.0], [2.0], [-1.0], [1.0]])
    labels = torch.tensor([0, 1, 0, 1])
    splits = {"train": [0, 1], "val": [2], "test": [3]}
    report = frozen_split_probe(x, labels, splits, seed=3, epochs=100)
    assert report["train_n"] == 2
    assert report["heldout_n"] == 2
    assert report["score"] == 1.0
    assert report["train_label_shuffled"] is False


def test_frozen_split_probe_refuses_missing_training_class():
    x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    splits = {"train": [0, 1], "test": [2, 3]}
    with pytest.raises(ValueError, match="lacks factor classes"):
        frozen_split_probe(x, labels, splits, seed=0, epochs=1)


def _control_row(*, objective, seed=None, stimulus=None, resolution=256):
    identity = {
        "model_id": "fixture/model",
        "revision": "pinned",
        "seed": seed,
        "state_dict_sha256": "a" * 64 if seed is not None else None,
    }
    return {
        "objective": objective,
        "identity": identity,
        "manifest": {
            "encoder_config": {
                "arch": "vit_large",
                "embed_dim": 4,
                "patch_size": 16,
                "tubelet": 2,
                "frames_per_clip": 4,
                "resolution": resolution,
                "dense": False,
                "pool": "mean",
            }
        },
        "run_receipt": ({"stimulus": {"set_sha256": stimulus}} if stimulus else {}),
    }


def test_matched_control_requires_architecture_and_stimulus_hash_equality():
    rows = {
        "learned": _control_row(objective="inherited-frozen", stimulus="b" * 64),
        "random_init": _control_row(objective="random-control", seed=7, stimulus="b" * 64),
        "wrong_resolution": _control_row(
            objective="random-control", seed=8, stimulus="b" * 64, resolution=384
        ),
    }
    matches = _control_matches(rows)
    assert [record["tag"] for record in matches["learned"]] == ["random_init"]
    assert matches["learned"][0]["stimulus_hash_exact"] is True


def test_old_cache_without_input_hash_cannot_claim_byte_matched_stimuli():
    rows = {
        "learned": _control_row(objective="inherited-frozen"),
        "random_init": _control_row(objective="random-control", seed=7, stimulus="b" * 64),
    }
    match = _control_matches(rows)["learned"][0]
    assert match["architecture_exact"] is True
    assert match["stimulus_hash_exact"] is False
    assert "predate" in match["stimulus_hash_limitation"]
