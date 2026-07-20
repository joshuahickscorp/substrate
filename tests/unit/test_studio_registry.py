from mop.studio import registry


def test_shipped_datasets_validate_clean():
    assert registry.validate_datasets() == []


def test_shipped_models_validate_clean():
    assert registry.validate_models() == []


def test_full_ego4d_is_deferred_by_default():
    ego = registry.get_dataset("ego4d_full")
    assert ego["status"] == "deferred"
    assert "ego4d_full" in registry.ALWAYS_DEFERRED


def test_seeded_broad_categories_present():
    slugs = {d["slug"] for d in registry.load_datasets()}
    for must in (
        "ssv2",
        "kinetics700_subset",
        "epic_kitchens_subset",
        "ego4d_subset",
        "howto100m_meta",
        "audioset_meta",
        "synthetic_controls",
        "local_import",
    ):
        assert must in slugs


def test_schema_rejects_missing_field():
    bad = {"slug": "x", "name": "X"}  # missing almost everything
    problems = registry.validate_dataset(bad)
    assert any("missing required field" in p for p in problems)


def test_schema_rejects_signed_terms_marked_available():
    good = registry.get_dataset("epic_kitchens_subset")
    dishonest = dict(good)
    dishonest["auth_steps"] = "sign the license agreement and accept the terms"
    dishonest["status"] = "available"
    problems = registry.validate_dataset(dishonest)
    assert any("signed terms" in p and "available" in p for p in problems)


def test_schema_rejects_metadata_only_with_cache_size():
    meta = registry.get_dataset("howto100m_meta")
    bad = dict(meta)
    bad["cache_gb_pooled"] = 5.0  # metadata-only must not claim a video cache
    problems = registry.validate_dataset(bad)
    assert any("metadata-only" in p for p in problems)


def test_schema_rejects_bad_status_and_kind():
    d = dict(registry.get_dataset("ssv2"))
    d["status"] = "totally-fine"
    d["kind"] = "magic"
    problems = registry.validate_dataset(d)
    assert any("status" in p for p in problems)
    assert any("kind" in p for p in problems)


def test_model_registry_merges_canonical_and_auxiliary():
    models = registry.load_models()
    roles = {m["role"] for m in models}
    assert "canonical" in roles
    assert {"auxiliary", "distilled", "quantized"} & roles


def test_canonical_vjepa_marked_real_only_when_verified():
    models = registry.load_models()
    canon = [m for m in models if m["role"] == "canonical"]
    real = [m for m in canon if m["result_tag"] == "real-encoder"]
    assert real, "at least one canonical V-JEPA must be real-encoder"
    for m in real:
        assert m["hf_id"] in registry.verified_real_ids()  # real only if hand-verified on HF
        assert m["available"] is True


def test_unavailable_encoder_does_not_pretend_real():
    models = registry.load_models()
    for m in models:
        if not m["available"]:
            assert m["result_tag"] != "real-encoder", f"{m['slug']} unavailable but tagged real-encoder"


def test_distilled_and_quantized_never_replace_canonical():
    models = registry.load_models()
    for m in models:
        if m["role"] in ("distilled", "quantized"):
            assert m["replaces_canonical"] is False


def test_validate_model_blocks_noncanonical_real_tag():
    bad = {
        "slug": "fake",
        "name": "Fake",
        "role": "auxiliary",
        "hf_id": "x/y",
        "embed_dim": 768,
        "dense": False,
        "available": True,
        "result_tag": "real-encoder",  # only canonical may claim this
        "replaces_canonical": False,
    }
    problems = registry.validate_model(bad)
    assert any("real-encoder" in p for p in problems)


def test_canonical_real_encoder_requires_verified_hf_id():
    bad = {
        "slug": "fake_canon",
        "name": "Fake canonical",
        "role": "canonical",
        "hf_id": "someone/not-verified",
        "embed_dim": 1024,
        "dense": False,
        "available": True,
        "result_tag": "real-encoder",
        "replaces_canonical": False,
    }
    problems = registry.validate_model(bad)
    assert any("VERIFIED_REAL_IDS" in p for p in problems)


def test_auxiliary_cannot_replace_canonical():
    bad = {
        "slug": "aux_imposter",
        "name": "Aux imposter",
        "role": "auxiliary",
        "hf_id": "x/y",
        "embed_dim": 768,
        "dense": False,
        "available": False,
        "result_tag": "provisional",
        "replaces_canonical": True,  # forbidden
    }
    problems = registry.validate_model(bad)
    assert any("replaces_canonical" in p for p in problems)


def test_signed_terms_heuristic_catches_dua_and_request_access():
    base = registry.get_dataset("epic_kitchens_subset")
    for phrase in (
        "complete the data use agreement (DUA)",
        "request access via the portal",
        "accept the EULA",
    ):
        d = dict(base)
        d["auth_steps"] = phrase
        d["status"] = "available"
        problems = registry.validate_dataset(d)
        assert any("signed terms" in p for p in problems), phrase


def test_validate_registry_aggregates_both():
    assert registry.validate_registry() == []
