import torch

from mop.substrate.form import (
    FormMeta,
    TensorFormAdapter,
    apply_affine_alignment,
    build_form_matrix,
    fit_affine_alignment,
    form_audit,
)


def _meta(tag: str, kind: str = "vision", *, control_for: str | None = None) -> FormMeta:
    objective = "random-control" if control_for else "self-supervised"
    return FormMeta(
        tag=tag,
        kind=kind,
        feature_dim=3,
        source="unit-test",
        objective=objective,
        control_for=control_for,
    )


def test_form_matrix_aligns_referents_and_controls():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    y = torch.tensor([[0.0, 0.0, 30.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    vision = TensorFormAdapter(_meta("vision"), x, ["a", "b", "c"])
    audio = TensorFormAdapter(_meta("audio", "audio", control_for="vision"), y, ["c", "a", "b"])

    matrix = build_form_matrix([vision, audio])

    assert matrix.referents == ("a", "b", "c")
    assert torch.equal(matrix.features["audio"], x * 10.0)
    assert matrix.controls() == {"vision": ["audio"]}
    audit = form_audit(matrix)
    assert audit["all_ok"]
    assert audit["missing_controls"] == []


def test_form_audit_names_missing_controls():
    x = torch.randn(4, 3)
    a = TensorFormAdapter(_meta("vision"), x, ["0", "1", "2", "3"])
    b = TensorFormAdapter(_meta("text", "text"), x + 1.0, ["0", "1", "2", "3"])

    audit = form_audit(build_form_matrix([a, b]))

    assert not audit["all_ok"]
    assert audit["missing_controls"] == ["text", "vision"]
    assert "one or more substantive form arms lack a matched control" in audit["warnings"]


def test_affine_alignment_recovers_paired_forms():
    g = torch.Generator().manual_seed(0)
    source = torch.randn(30, 5, generator=g)
    transform = torch.randn(5, 4, generator=g)
    target = source @ transform + 0.25

    weight = fit_affine_alignment(source, target, ridge=1.0e-5)
    recovered = apply_affine_alignment(source, weight)

    assert torch.mean((recovered - target) ** 2) < 1.0e-6
