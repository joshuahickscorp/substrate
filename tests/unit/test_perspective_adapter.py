import pytest
import torch

from mop.perspectives import (
    PerspectiveMeta,
    PerspectiveRegistry,
    TensorPerspectiveAdapter,
    build_perspective_matrix,
    perspective_audit,
)
from mop.perspectives.adapter import SubstratePerspectiveAdapter
from mop.substrate.adapter import RandomPixelAdapter


def _meta(tag: str, dim: int, *, control_for: str | None = None, modality: str = "vision") -> PerspectiveMeta:
    return PerspectiveMeta(
        tag=tag,
        modality=modality,
        feature_dim=dim,
        source="unit-test",
        control_for=control_for,
        license="synthetic",
    )


def test_tensor_perspective_validates_referents_and_dim():
    TensorPerspectiveAdapter(_meta("vision", 3), torch.randn(2, 3), ["a", "b"])
    with pytest.raises(ValueError, match="feature_dim"):
        TensorPerspectiveAdapter(_meta("bad", 4), torch.randn(2, 3), ["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        TensorPerspectiveAdapter(_meta("dup", 3), torch.randn(2, 3), ["a", "a"])


def test_perspective_matrix_aligns_by_referent_id_and_factors():
    vision = TensorPerspectiveAdapter(
        _meta("vision", 2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ["clip-a", "clip-b"],
        factors={"shape": torch.tensor([0, 1])},
    )
    language = TensorPerspectiveAdapter(
        _meta("language", 2, modality="language"),
        torch.tensor([[10.0, 0.0], [20.0, 0.0]]),
        ["clip-b", "clip-a"],
        factors={"caption_len": [5, 3]},
    )

    matrix = build_perspective_matrix([vision, language])
    assert matrix.referents == ("clip-b", "clip-a")
    assert torch.equal(matrix.features["vision"], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert torch.equal(matrix.features["language"], torch.tensor([[10.0, 0.0], [20.0, 0.0]]))
    assert torch.equal(matrix.factors["language"]["caption_len"], torch.tensor([5, 3]))


def test_perspective_matrix_rejects_mismatched_referent_sets():
    a = TensorPerspectiveAdapter(_meta("a", 2), torch.randn(2, 2), ["x", "y"])
    b = TensorPerspectiveAdapter(_meta("b", 2), torch.randn(2, 2), ["x", "z"])
    with pytest.raises(ValueError, match="referent mismatch"):
        build_perspective_matrix([a, b])


def test_perspective_audit_names_missing_controls():
    reg = PerspectiveRegistry()
    reg.register(TensorPerspectiveAdapter(_meta("vision", 2), torch.randn(3, 2), [0, 1, 2]))
    reg.register(
        TensorPerspectiveAdapter(
            _meta("vision_rand", 2, control_for="vision"),
            torch.randn(3, 2),
            [0, 1, 2],
        )
    )
    reg.register(
        TensorPerspectiveAdapter(
            _meta("audio", 4, modality="audio"),
            torch.randn(3, 4),
            [0, 1, 2],
        )
    )

    audit = perspective_audit(build_perspective_matrix(reg))
    assert audit["controls"] == {"vision": ["vision_rand"]}
    assert audit["missing_controls"] == ["audio"]
    assert audit["modalities"]["audio"] == "audio"


def test_substrate_perspective_wraps_existing_adapter_without_registry_duplication():
    substrate = RandomPixelAdapter(embed_dim=6, ds=4, tsub=2, seed=0)
    clips = torch.rand(2, 4, 3, 8, 8)
    adapter = SubstratePerspectiveAdapter(substrate, clips, ["c0", "c1"], tag="pixel_control")
    batch = adapter.extract()
    assert batch.features.shape == (2, 6)
    assert batch.referents == ("c0", "c1")
    assert batch.meta.source == "clips"
