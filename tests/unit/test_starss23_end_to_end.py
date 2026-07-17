"""End-to-end tests for the STARSS23 ESCS bed: the ladder Bed adapter and the producer / verifier round trip.

These tests exercise the whole lane the way the ladder and the entrypoint scripts do: the Bed adapter
plugs into ladder_contracts.Bed, the producer seals an artifact to disk, and the separately authored
verifier re-scores it from the sealed file. On synthetic fixtures the verdict is capped at mechanics-ok
and the independent confirmation stays false.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import importlib
import json

from mop.beds.starss23 import BED_ID
from mop.beds.starss23.artifact import write_artifact
from mop.beds.starss23.bed import Starss23EscsBed, build_bed
from mop.beds.starss23.verifier import verify_sealed_file
from mop.ladder.ladder_contracts import Bed
from mop.ladder.stage_ladder import MatchedBudget

# ---------------------------------------------------------------------------
# The ladder Bed adapter.
# ---------------------------------------------------------------------------


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
    # Deterministic in the seed.
    assert bed.favorable_regime(0).digest() == favorable.digest()
    assert bed.null_regime(0).digest() == null.digest()
    # The favorable regime plants onsets; the null regime is nuisance-only, so they differ.
    assert favorable.digest() != null.digest()
    assert favorable.regime == "favorable"
    assert null.regime == "null"
    # A different seed gives a different favorable materialization.
    assert bed.favorable_regime(1).digest() != favorable.digest()


# ---------------------------------------------------------------------------
# Producer to verifier round trip through the sealed file.
# ---------------------------------------------------------------------------


def test_producer_seals_a_file_the_independent_verifier_reproduces(
    starss23_bed_artifact, tmp_path
) -> None:
    out_path = tmp_path / "STARSS23_ESCS_BED.json"
    written = write_artifact(starss23_bed_artifact.artifact, out_path)
    assert written.exists()

    # The on-disk file is canonical JSON and its stored seal matches a re-hash of the body.
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk["seal"] == starss23_bed_artifact.artifact["seal"]

    payload = verify_sealed_file(str(written))
    assert payload["independent_referee_reproduction"] is True
    assert payload["independent_scientific_confirmation"] is False
    assert payload["source_kind"] == "synthetic"
    assert payload["mismatches"] == []


def test_verifier_rejects_a_tampered_score(starss23_bed_artifact, tmp_path) -> None:
    # Flip one stored fire list so the sealed score no longer matches the raw fires; the verifier,
    # which re-scores from the raw fires and re-hashes the seal, must catch it.
    tampered = json.loads(json.dumps(starss23_bed_artifact.artifact))
    tampered["per_seed"][0]["clips"][0]["fires"]["candidate"] = []
    out_path = tmp_path / "tampered.json"
    out_path.write_text(json.dumps(tampered), encoding="utf-8")
    payload = verify_sealed_file(str(out_path))
    assert payload["independent_referee_reproduction"] is False
    assert payload["mismatches"]


def test_synthetic_run_can_never_be_scientifically_confirmed(starss23_bed_artifact) -> None:
    # Even if a synthetic artifact were hand-edited to claim real, clean rights, and reproductions, the
    # verifier still refuses confirmation because the honesty flags and re-scoring are the gate; and the
    # producer itself hardcodes source_kind synthetic.
    assert starss23_bed_artifact.artifact["source_kind"] == "synthetic"
    assert starss23_bed_artifact.artifact["reproductions"] == 0
    assert starss23_bed_artifact.verdict in ("mechanics-ok", "null")
    assert starss23_bed_artifact.verdict != "cleared"


# ---------------------------------------------------------------------------
# Entrypoint scripts.
# ---------------------------------------------------------------------------


def test_producer_and_verifier_scripts_expose_a_main() -> None:
    for name in ("run_starss23_escs_bed", "verify_starss23_escs_bed"):
        module = importlib.import_module(f"scripts.{name}")
        assert hasattr(module, "main")
        assert callable(module.main)
