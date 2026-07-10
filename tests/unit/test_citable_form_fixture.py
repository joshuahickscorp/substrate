from scripts.build_citable_form_fixture import build_fixture

from mop.substrate.cache_tools import validate_cache
from mop.substrate.form import LatentStoreFormAdapter, build_form_matrix, form_audit
from mop.substrate.latent_store import LatentStore


def test_programmatic_fixture_is_dense_citable_and_form_ready(tmp_path):
    receipt = build_fixture(tmp_path, name="fixture", count=20, seed=3)
    assert receipt["all_ok"]
    store = LatentStore.open(tmp_path / "fixture")
    arm = LatentStoreFormAdapter(
        store,
        require_explicit_referents=True,
        require_manifest=True,
    )
    batch = arm.extract()
    assert batch.features.shape == (20, 4, 8)
    assert arm.meta.token_shape == (4, 8)
    assert arm.meta.manifest_verified
    audit = form_audit(build_form_matrix([arm]), require_controls=False, require_citable=True)
    assert audit["uncitable_tags"] == []
    assert audit["unverified_manifest_tags"] == []
    assert validate_cache(store.root, citable=True) == []
