import copy
from pathlib import Path

import pytest
import torch

from mop.studies.vjepa_scale_atlas import (
    _control_matches,
    frozen_split_probe,
    validate_factorized_stimulus_identity,
    validate_shared_referents,
)


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


def _identity_fixture():
    clip_hashes = ["b" * 64, "c" * 64]
    learned = _control_row(objective="inherited-frozen")
    learned.update(
        {
            "path": Path("/tmp/learned_cache"),
            "latents": torch.zeros(2, 4),
            "referents": ["r0", "r1"],
            "factors_metadata": {"seed": 0},
        }
    )
    learned["manifest"]["encoder_config"]["name"] = "fixture_encoder"
    random = _control_row(objective="random-control", seed=7)
    random.update(
        {
            "path": Path("/tmp/random_cache"),
            "latents": torch.zeros(2, 4),
            "referents": ["r0", "r1"],
            "factors_metadata": {"seed": 0},
            "run_receipt": {
                "stimulus": {
                    "records": [
                        {"referent": referent, "sha256": digest}
                        for referent, digest in zip(["r0", "r1"], clip_hashes, strict=True)
                    ]
                }
            },
        }
    )
    random["manifest"]["encoder_config"]["name"] = "fixture_encoder"
    source_sha = "d" * 64
    receipt = {
        "schema": "mop-factorized-stimulus-identity/v1",
        "all_ok": True,
        "problems": [],
        "generator_evidence": {
            name: {
                "identical": True,
                "head_sha256": source_sha,
                "current_sha256": source_sha,
                "head_commit": "e" * 40,
            }
            for name in ("make_factorized_clip", "_hue_tint")
        },
        "regenerated_stimulus_hashes": {
            "256": [{"index": index, "sha256": digest} for index, digest in enumerate(clip_hashes)]
        },
        "learned_latent_rebinding": [
            {
                "tag": "learned",
                "cache": "learned_cache",
                "encoder": "fixture_encoder",
                "resolution": 256,
                "clip_index": 0,
                "clip_sha256": clip_hashes[0],
                "latent_dim": 4,
                "bitwise_equal": True,
                "max_abs_diff": 0.0,
            }
        ],
    }
    return {"learned": learned, "random": random}, receipt


def test_separate_identity_receipt_fills_old_learned_hash_only_after_strict_binding():
    rows, receipt = _identity_fixture()
    validation = validate_factorized_stimulus_identity(rows, receipt)
    assert validation["accepted"] is True
    match = _control_matches(rows, validation)["learned"][0]
    assert match["stimulus_hash_exact"] is True
    assert match["stimulus_hash_source"] == "validated-factorized-stimulus-identity"


def test_separate_identity_receipt_rejects_a_tampered_control_hash():
    rows, receipt = _identity_fixture()
    tampered = copy.deepcopy(rows)
    tampered["random"]["run_receipt"]["stimulus"]["records"][1]["sha256"] = "f" * 64
    validation = validate_factorized_stimulus_identity(tampered, receipt)
    assert validation["accepted"] is False
    assert any("random control" in problem for problem in validation["problems"])
