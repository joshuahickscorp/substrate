
from __future__ import annotations

import importlib
import json

from mop.beds.starss23 import BED_ID
from mop.beds.starss23.bed import Starss23EscsBed, build_bed
from mop.beds.starss23.verifier import verify_sealed_file
from mop.ladder.ladder_contracts import Bed
from mop.ladder.stage_ladder import MatchedBudget
from mop.substrate.events import write_canonical_json


def test_bed_satisfies_the_ladder_bed_protocol() -> None:
    bed = build_bed()
    assert isinstance(bed, Bed)
    assert isinstance(bed, Starss23EscsBed)
    assert bed.mechanism_id == BED_ID


def test_bed_declares_the_three_controls_and_the_noisy_tv_guard() -> None:
    controls = build_bed().controls()
    assert controls == ("rate_matched_random", "always_on", "best_single", "noisy_tv")


def test_bed_matched_cost_is_a_nonvacuous_budget_under_the_ceiling() -> None:
    cost = build_bed().matched_cost()
    assert isinstance(cost, MatchedBudget)
    assert cost.params == 3193
    assert 0 < cost.flops <= 60_000_000_000
    assert cost.seeds == 5


def test_bed_regimes_are_deterministic_and_distinct() -> None:
    bed = build_bed()
    favorable = bed.favorable_regime(0)
    null = bed.null_regime(0)
    assert bed.favorable_regime(0).digest() == favorable.digest()
    assert bed.null_regime(0).digest() == null.digest()
    assert favorable.digest() != null.digest()
    assert favorable.regime == "favorable"
    assert null.regime == "null"
    assert bed.favorable_regime(1).digest() != favorable.digest()


def test_producer_seals_a_file_the_independent_verifier_reproduces(
    starss23_bed_artifact, tmp_path
) -> None:
    out_path = tmp_path / "STARSS23_ESCS_BED.json"
    written = write_canonical_json(starss23_bed_artifact.artifact, out_path)
    assert written.exists()

    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk["seal"] == starss23_bed_artifact.artifact["seal"]

    payload = verify_sealed_file(str(written))
    assert payload["independent_referee_reproduction"] is True
    assert payload["independent_scientific_confirmation"] is False
    assert payload["source_kind"] == "synthetic"
    assert payload["mismatches"] == []


def test_verifier_rejects_a_tampered_score(starss23_bed_artifact, tmp_path) -> None:
    tampered = json.loads(json.dumps(starss23_bed_artifact.artifact))
    tampered["per_seed"][0]["clips"][0]["fires"]["candidate"] = []
    out_path = tmp_path / "tampered.json"
    out_path.write_text(json.dumps(tampered), encoding="utf-8")
    payload = verify_sealed_file(str(out_path))
    assert payload["independent_referee_reproduction"] is False
    assert payload["mismatches"]


def test_synthetic_run_can_never_be_scientifically_confirmed(starss23_bed_artifact) -> None:
    assert starss23_bed_artifact.artifact["source_kind"] == "synthetic"
    assert starss23_bed_artifact.artifact["reproductions"] == 0
    assert starss23_bed_artifact.verdict in ("mechanics-ok", "null")
    assert starss23_bed_artifact.verdict != "cleared"


def test_producer_and_verifier_scripts_expose_a_main() -> None:
    for name in ("run_starss23_escs_bed", "verify_starss23_escs_bed"):
        module = importlib.import_module(f"scripts.{name}")
        assert hasattr(module, "main")
        assert callable(module.main)
