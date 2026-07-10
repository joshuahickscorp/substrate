import torch

from mop.substrate.form import (
    FormMeta,
    FormRegistry,
    SubstrateFormAdapter,
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
    assert batch.features.shape == (5, 2, 3)  # native geometry survives adapter extraction
    assert arm.meta.token_shape == (2, 3)
    assert arm.meta.feature_dim == 6
    matrix = build_form_matrix([arm])
    assert matrix.features["dense"].shape == (5, 6)  # current probes flatten explicitly


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


def test_latent_store_form_adapter_separates_legacy_factor_metadata(tmp_path):
    import json

    from mop.substrate.form import LatentStoreFormAdapter

    store = _pooled_store(tmp_path, name="mixedfactors")
    (store.root / "factors.json").write_text(
        json.dumps(
            {
                "clipset": "bound_nuisance_v1",
                "seed": 0,
                "shape": [0, 1, 0, 1, 0, 1],
                "color": [1, 1, 1, 0, 0, 0],
            }
        )
    )
    batch = LatentStoreFormAdapter(store, tag="v", kind="vision").extract()
    assert set(batch.factors) == {"shape", "color"}


def test_latent_store_form_adapter_reads_v2_factor_columns(tmp_path):
    import json

    from mop.substrate.form import LatentStoreFormAdapter

    store = _pooled_store(tmp_path, name="v2factors")
    (store.root / "factors.json").write_text(
        json.dumps(
            {
                "schema": "mop-factor-sidecar/v2",
                "metadata": {"clipset": "fixture", "seed": 0},
                "columns": {"shape": [0, 1, 0, 1, 0, 1]},
            }
        )
    )
    batch = LatentStoreFormAdapter(store, tag="v", kind="vision").extract()
    assert set(batch.factors) == {"shape"}


def test_latent_store_form_adapter_discovers_explicit_referents(tmp_path):
    import json

    from mop.substrate.form import LatentStoreFormAdapter

    store = _pooled_store(tmp_path, name="withrefs")
    refs = [f"clip-{i}" for i in range(len(store))]
    (store.root / "clip_stems.json").write_text(json.dumps(refs))
    arm = LatentStoreFormAdapter(store, tag="v", kind="vision")
    assert arm.extract().referents == tuple(refs)
    assert arm.meta.referents_explicit
    assert arm.meta.referent_scheme == "clip-stem"


def test_latent_store_form_adapter_can_require_citable_inputs(tmp_path):
    import json

    import pytest

    from mop.substrate.cache_manifest import write_cache_manifest
    from mop.substrate.form import LatentStoreFormAdapter

    store = _pooled_store(tmp_path, name="citable")
    with pytest.raises(ValueError, match="no explicit referent sidecar"):
        LatentStoreFormAdapter(store, require_explicit_referents=True)

    refs = [f"r{i}" for i in range(len(store))]
    (store.root / "referents.json").write_text(json.dumps(refs))
    write_cache_manifest(
        store.root,
        encoder_config={"name": "fixture", "revision": "test"},
        form_kind="vision",
        form_objective="inherited-frozen",
        referent_scheme="fixture-id",
    )
    arm = LatentStoreFormAdapter(store, require_explicit_referents=True, require_manifest=True)
    assert arm.meta.manifest_verified
    assert arm.meta.referent_scheme == "fixture-id"


def test_form_registry_sorts_tags_before_choosing_canonical_referents():
    reg = FormRegistry()
    reg.register(
        TensorFormAdapter(
            _meta("z_form"),
            torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]),
            ["y", "x"],
        )
    )
    reg.register(
        TensorFormAdapter(
            _meta("a_form", "audio"),
            torch.tensor([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]]),
            ["x", "y"],
        )
    )

    matrix = build_form_matrix(reg)

    assert reg.tags() == ["a_form", "z_form"]
    assert matrix.referents == ("x", "y")
    assert torch.equal(
        matrix.features["z_form"],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
    )


def test_form_audit_is_the_combined_form_and_perspective_surface():
    text_meta = FormMeta(
        tag="caption",
        kind="text",
        feature_dim=3,
        source="captions.json",
        objective="programmatic",
        supervised=True,
        derived=True,
        license="fixture",
        referent_scheme="clip-id",
    )
    matrix = build_form_matrix(
        [
            TensorFormAdapter(text_meta, torch.randn(3, 3), ["a", "b", "c"]),
            TensorFormAdapter(
                _meta("audio_control", "audio", control_for="caption"),
                torch.randn(3, 3),
                ["a", "b", "c"],
            ),
        ]
    )

    audit = form_audit(matrix)

    assert audit["n"] == audit["n_referents"] == 3
    assert audit["modalities"] == {"audio_control": "audio", "caption": "text"}
    assert audit["feature_dims"] == {"audio_control": 3, "caption": 3}
    assert audit["supervised"] == ["caption"]
    assert audit["derived"] == ["caption"]
    assert audit["licenses"]["caption"] == "fixture"
    assert audit["objectives"]["caption"] == "programmatic"
    assert audit["referent_schemes"]["caption"] == "clip-id"


def test_substrate_form_adapter_encodes_once_and_reuses_detached_features(monkeypatch):
    from mop.substrate.adapter import RandomPixelAdapter

    substrate = RandomPixelAdapter(embed_dim=6, ds=4, tsub=2, seed=0)
    clips = torch.rand(2, 4, 3, 8, 8)
    original = substrate.extract_batched
    calls = 0

    def counted_extract(x, batch=8):
        nonlocal calls
        calls += 1
        return original(x, batch=batch)

    monkeypatch.setattr(substrate, "extract_batched", counted_extract)
    adapter = SubstrateFormAdapter(substrate, clips, ["c0", "c1"], tag="pixel_control")

    first = adapter.extract()
    second = adapter.extract()

    assert calls == 1
    assert torch.equal(first.features, second.features)
    assert not first.features.requires_grad
    assert first.meta.objective == "random-control"
    assert first.meta.referents_explicit
