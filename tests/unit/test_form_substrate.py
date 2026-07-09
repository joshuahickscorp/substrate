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


def _pooled_store(tmp_path, name="pooled", dim=4, n=6):
    import numpy as np

    from mop.substrate import LatentStore

    store = LatentStore.create(tmp_path, name, feat_shape=(dim,), capacity=n, key_dim=dim)
    x = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    store.write_batch(0, x, x)
    store.finalize()
    return store


def test_latent_store_form_adapter_pooled_aligns_with_tensor_arm(tmp_path):
    from mop.substrate.form import LatentStoreFormAdapter

    store = _pooled_store(tmp_path)
    vision = LatentStoreFormAdapter(store, tag="vision", kind="vision", objective="inherited-frozen")
    batch = vision.extract()
    assert batch.features.shape == (6, 4)
    assert batch.referents == tuple(f"pooled:{i}" for i in range(6))
    assert vision.meta.feature_dim == 4
    assert vision.meta.token_shape == ()

    # a control tensor arm over the same referents builds a valid, control-complete matrix
    control = TensorFormAdapter(
        _meta("vision_rand", "vision", control_for="vision"),
        torch.randn(6, 3),
        list(batch.referents),
    )
    matrix = build_form_matrix([vision, control])
    assert matrix.controls() == {"vision": ["vision_rand"]}


def test_latent_store_form_adapter_dense_records_token_shape(tmp_path):
    import numpy as np

    from mop.substrate import LatentStore
    from mop.substrate.form import LatentStoreFormAdapter

    store = LatentStore.create(tmp_path, "dense", feat_shape=(2, 3), capacity=5, key_dim=6)
    x = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    store.write_batch(0, x, x.reshape(5, 6))
    store.finalize()

    arm = LatentStoreFormAdapter(store, tag="dense", kind="latent")
    batch = arm.extract()
    assert batch.features.shape == (5, 6)  # flattened
    assert arm.meta.token_shape == (2, 3)  # geometry preserved
    assert arm.meta.feature_dim == 6


def test_latent_store_form_adapter_reads_factor_sidecar(tmp_path):
    import json

    from mop.substrate.form import LatentStoreFormAdapter

    store = _pooled_store(tmp_path, name="withfactors")
    (store.root / "factors.json").write_text(json.dumps({"shape": [0, 1, 0, 1, 0, 1]}))
    arm = LatentStoreFormAdapter(store, tag="v", kind="vision")
    batch = arm.extract()
    assert "shape" in batch.factors
    assert torch.equal(batch.factors["shape"], torch.tensor([0, 1, 0, 1, 0, 1]))

    # explicit factors override the sidecar
    arm2 = LatentStoreFormAdapter(store, tag="v", kind="vision", factors={"color": [1, 1, 1, 0, 0, 0]})
    assert "shape" not in arm2.extract().factors
    assert "color" in arm2.extract().factors
