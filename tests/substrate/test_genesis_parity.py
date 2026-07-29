"""Measured parity audit and capability-density frontier must be able to fail."""

from __future__ import annotations

import resource
import sys
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_parity as parity
from substrate.genesis_material import (
    MaterialBase,
    Observation,
    Opportunity,
    ResourceLedger,
    equal_opportunity,
)


def _ledger(
    *,
    envelope: str = "512MB",
    operations: int = 0,
    durable_writes: int = 0,
    peak_resident_bytes: int = 0,
    checkpoints: int = 0,
    restores: int = 0,
    operation_budget: int = 10_000,
    durable_write_budget: int = 1_000,
) -> ResourceLedger:
    ledger = ResourceLedger(
        envelope=envelope,
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
        byte_budget=C.ENVELOPE_BYTES[envelope],
        operations=operations,
        durable_writes=durable_writes,
        resident_bytes=peak_resident_bytes,
        peak_resident_bytes=peak_resident_bytes,
        checkpoints=checkpoints,
        restores=restores,
    )
    return ledger


def _opportunity(
    *,
    name_for_deprivation: str | None = None,
    observation_digest: str = "obs-aaa",
    sensor_channels: tuple[str, ...] = ("vision", "touch"),
    teaching_digest: str = "teach-aaa",
    ledger: ResourceLedger | None = None,
    deprived: tuple[str, ...] = (),
) -> Opportunity:
    if name_for_deprivation is not None and not deprived:
        deprived = tuple(C.BASELINE_DEPRIVATION.get(name_for_deprivation, ()))
    return Opportunity(
        observation_digest=observation_digest,
        sensor_channels=sensor_channels,
        teaching_digest=teaching_digest,
        ledger=ledger if ledger is not None else _ledger(),
        plasticity_enabled="plasticity" not in deprived,
        persistence_enabled="persistence" not in deprived,
        deprived=deprived,
    )


class _StubMaterial(MaterialBase):
    """Minimal material so the audit can read a live opportunity and ledger."""

    def __init__(self, name: str, opportunity: Opportunity, mechanism: str = "stub") -> None:
        super().__init__(name=name, mechanism=mechanism, _opportunity=opportunity)

    def _transition(self, observation: Observation) -> None:
        return None

    def _answer(self, probe: Any) -> Any:
        raise NotImplementedError

    def _propose(self) -> list[Any]:
        return []

    def _commit(self, proposal: Any) -> None:
        return None

    def _rollback(self, receipt: Any) -> None:
        return None

    def _durable_state(self) -> Any:
        return {"name": self.name}

    def _active_state(self) -> Any:
        return {"seen": self.observations_seen}

    def _restore_durable(self, state: Any) -> None:
        return None

    def _restore_active(self, state: Any) -> None:
        return None


def _arm_record(
    name: str,
    *,
    information: str = "obs-aaa",
    sensors: tuple[str, ...] = ("vision", "touch"),
    teaching: str = "teach-aaa",
    compute: int = 100,
    plasticity: int = 10,
    persistence: int = 2,
    memory: int = 1_000,
    deprived: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "information": information,
        "sensors": sensors,
        "teaching": teaching,
        "compute": compute,
        "plasticity": plasticity,
        "persistence": persistence,
        "memory": memory,
    }
    if deprived is not None:
        row["deprived"] = deprived
    return row


def test_parity_fails_when_measured_budgets_diverge() -> None:
    """One arm spending more operations than another must fail the audit."""
    low = _arm_record("arm_low", compute=100, plasticity=5, persistence=1, memory=500)
    high = _arm_record("arm_high", compute=10_000, plasticity=5, persistence=1, memory=500)
    report = parity.parity_audit([low, high], unit={"unit_id": "divergence-ops"})
    assert report["activation"] is False
    assert report["channel_pass"]["compute"] is False
    assert report["all_pass"] is False
    assert report["channels"]["compute"]["detail"] == "relative_tolerance_exceeded"
    assert report["measured"]["arm_low"]["compute"] == 100
    assert report["measured"]["arm_high"]["compute"] == 10_000


def test_parity_passes_when_measured_channels_match() -> None:
    a = _arm_record("left", compute=100, plasticity=10, persistence=2, memory=1_000)
    b = _arm_record("right", compute=101, plasticity=10, persistence=2, memory=1_000)
    # 1% relative gap on compute is inside 2% tolerance
    report = parity.parity_audit([a, b])
    assert report["channel_pass"]["compute"] is True
    assert report["all_pass"] is True


def test_exempt_arm_is_reported_exempt_not_equal() -> None:
    full = _arm_record("candidate", plasticity=25)
    frozen = _arm_record(
        "static_frozen_field",
        plasticity=0,
        deprived=tuple(C.BASELINE_DEPRIVATION["static_frozen_field"]),
    )
    report = parity.parity_audit([full, frozen], unit={"unit_id": "exempt-plasticity"})
    channel = report["channels"]["plasticity"]
    assert "static_frozen_field" in channel["exempt_arms"]
    assert channel["status"]["static_frozen_field"] == "exempt"
    assert channel["status"]["static_frozen_field"] != "equal"
    assert channel["pass"] is True
    assert report["channel_pass"]["plasticity"] is True
    # Other channels still compared.
    assert "static_frozen_field" in report["channels"]["compute"]["compared_arms"]


def test_baseline_deprivation_table_is_honoured_by_name() -> None:
    full = _arm_record("retrieval_only", plasticity=0)
    # No explicit deprived tuple: name alone maps through BASELINE_DEPRIVATION.
    peer = _arm_record("plastic_peer", plasticity=40)
    report = parity.parity_audit([full, peer])
    assert report["channels"]["plasticity"]["status"]["retrieval_only"] == "exempt"
    assert report["channels"]["plasticity"]["pass"] is True


def test_byte_exact_channels_fail_on_single_flipped_observation() -> None:
    base = _arm_record("arm_a", information="digest-0000")
    flipped = _arm_record("arm_b", information="digest-0001")
    report = parity.parity_audit([base, flipped], unit={"unit_id": "flipped-obs"})
    assert report["channel_pass"]["information"] is False
    assert report["channels"]["information"]["detail"] == "byte_mismatch"
    assert report["channels"]["information"]["status"]["arm_b"] == "unequal"
    assert report["all_pass"] is False


def test_byte_exact_sensors_and_teaching() -> None:
    a = _arm_record("a", sensors=("vision",), teaching="t0")
    b = _arm_record("b", sensors=("vision", "audio"), teaching="t0")
    c = _arm_record("c", sensors=("vision",), teaching="t1")
    sensors = parity.parity_audit([a, b])
    assert sensors["channel_pass"]["sensors"] is False
    teaching = parity.parity_audit([a, c])
    assert teaching["channel_pass"]["teaching"] is False


def test_parity_audit_reads_live_material_ledgers() -> None:
    shared_digest = "live-obs"
    left = _StubMaterial(
        "live_left",
        _opportunity(
            observation_digest=shared_digest,
            ledger=_ledger(operations=50, durable_writes=3, peak_resident_bytes=2_000, checkpoints=1, restores=1),
        ),
    )
    right = _StubMaterial(
        "live_right",
        _opportunity(
            observation_digest=shared_digest,
            ledger=_ledger(operations=50, durable_writes=3, peak_resident_bytes=2_000, checkpoints=1, restores=1),
        ),
    )
    report = parity.parity_audit([left, right], unit={"envelope": "512MB"})
    assert report["all_pass"] is True
    assert report["measured"]["live_left"]["compute"] == 50
    assert report["measured"]["live_left"]["persistence"] == 2


def test_parity_audit_fails_live_material_compute_divergence() -> None:
    left = _StubMaterial(
        "spend_low",
        _opportunity(ledger=_ledger(operations=10, durable_writes=1, peak_resident_bytes=100)),
    )
    right = _StubMaterial(
        "spend_high",
        _opportunity(ledger=_ledger(operations=500, durable_writes=1, peak_resident_bytes=100)),
    )
    report = parity.parity_audit([left, right])
    assert report["channel_pass"]["compute"] is False
    assert report["all_pass"] is False


def test_equal_opportunity_builder_still_usable_for_unit_setup() -> None:
    observations = (
        Observation(index=0, channel="vision", payload=(1, 2, 3), teaching=True),
        Observation(index=1, channel="touch", payload=(4, 5), teaching=False),
    )
    opportunity = equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=("vision", "touch"),
        operation_budget=100,
        durable_write_budget=10,
    )
    material = _StubMaterial("built", opportunity)
    material.observe(observations[0])
    material.observe(observations[1])
    peer = _arm_record(
        "peer",
        information=opportunity.observation_digest,
        sensors=opportunity.sensor_channels,
        teaching=opportunity.teaching_digest,
        compute=material.cost()["compute"],
        plasticity=0,
        persistence=0,
        memory=0,
    )
    report = parity.parity_audit([material, peer])
    assert report["channel_pass"]["information"] is True
    assert report["channel_pass"]["sensors"] is True
    assert report["channel_pass"]["teaching"] is True


def test_ru_maxrss_is_actually_read_and_converted() -> None:
    measured = parity.measure_peak_resident_bytes()
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert measured["ru_maxrss_raw"] == raw
    assert measured["source"] == "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
    assert measured["activation"] is False
    if sys.platform == "darwin":
        assert measured["ru_maxrss_platform_unit"] == "bytes"
        assert measured["peak_resident_bytes"] == raw
        assert measured["conversion"] == "macos_ru_maxrss_already_bytes"
        assert "bytes" in measured["conversion_note"].lower()
    else:
        assert measured["ru_maxrss_platform_unit"] == "kilobytes"
        assert measured["peak_resident_bytes"] == raw * 1024
        assert measured["conversion"] == "linux_ru_maxrss_kilobytes_times_1024"
    # Not a stubbed constant: value is a positive process peak.
    assert measured["peak_resident_bytes"] > 0


def test_capability_density_frontier_covers_six_envelopes_and_known_pareto() -> None:
    """Hand-built points with a known non-dominated set over (bytes, capability)."""
    # Known answer on every envelope (identical geometry):
    #   tiny_weak   ( 50, 0.20)  — on frontier
    #   small_mid   (100, 0.50)  — on frontier
    #   medium_high (200, 0.90)  — on frontier
    #   wasteful    (300, 0.60)  — dominated by medium_high
    #   heavy_same  (400, 0.90)  — dominated by medium_high (same capability, more bytes)
    #   tiny_dup    (100, 0.40)  — dominated by small_mid
    specs = {
        "tiny_weak": (50, 0.20),
        "small_mid": (100, 0.50),
        "medium_high": (200, 0.90),
        "wasteful": (300, 0.60),
        "heavy_same": (400, 0.90),
        "tiny_dup": (100, 0.40),
    }
    arms = [
        {
            "name": name,
            "capability": {envelope: cap for envelope in C.MEMORY_ENVELOPES},
            "resident_bytes": {envelope: size for envelope in C.MEMORY_ENVELOPES},
            "checkpoint_bytes": size // 2,
            "disk_bytes": size,
            "operations": size,
            "wall_clock_seconds_per_episode": 0.001,
            "learning": cap,
            "resident_bytes_source": "hand_built_known_answer",
        }
        for name, (size, cap) in specs.items()
    ]
    report = parity.capability_density_frontier(arms)
    assert report["activation"] is False
    assert report["envelopes"] == list(C.MEMORY_ENVELOPES)
    assert len(report["envelopes"]) == 6
    assert len(report["rows"]) == 6 * 6
    assert report["pareto_objectives"]["size_is_not_the_goal"] is True
    assert report["energy_measurement_policy"]["hardware_energy_measured"] is False
    assert report["resident_measurement_policy"]["serialized_payload_length_refused_as_resident_proxy"] is True

    expected = {"tiny_weak", "small_mid", "medium_high"}
    for envelope in C.MEMORY_ENVELOPES:
        frontier_arms = {point["arm"] for point in report["pareto_frontier_by_envelope"][envelope]}
        assert frontier_arms == expected, envelope
        assert "wasteful" not in frontier_arms
        assert "heavy_same" not in frontier_arms
        assert "tiny_dup" not in frontier_arms

    assert set(report["arms_on_pareto_frontier"]) == expected

    # Process ru_maxrss was actually consulted and unit-converted (not a stub constant).
    process = report["process_peak_resident_bytes"]
    assert process["source"] == "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
    assert process["peak_resident_bytes"] > 0
    assert process["ru_maxrss_raw"] > 0
    if sys.platform == "darwin":
        assert process["conversion"] == "macos_ru_maxrss_already_bytes"
        assert process["peak_resident_bytes"] == process["ru_maxrss_raw"]
    else:
        assert process["conversion"] == "linux_ru_maxrss_kilobytes_times_1024"
        assert process["peak_resident_bytes"] == process["ru_maxrss_raw"] * 1024
    # Peak is non-decreasing for the process; the report's sample is at most current.
    assert process["ru_maxrss_raw"] <= int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # Learning per added byte is defined for non-baseline sizes.
    sample = next(row for row in report["rows"] if row["arm"] == "medium_high" and row["envelope"] == "512MB")
    assert sample["learning_per_added_byte"] == sample["learning"] / (200 - 50)
    assert sample["operation_count_energy_proxy"] == 200
    assert sample["energy_is_proxy"] is True


def test_capability_density_frontier_live_runner_reads_ru_maxrss() -> None:
    def runner(envelope: str) -> dict[str, Any]:
        # Allocate a little so the process peak is real; size for the frontier is still explicit.
        blob = bytearray(64 * 1024)
        blob[0] = 1
        return {
            "capability": 0.1 if envelope == "512MB" else 0.2,
            "resident_bytes": 10_000 if envelope == "512MB" else 20_000,
            "resident_bytes_source": "runner_provided_for_frontier_geometry",
            "checkpoint_bytes": 100,
            "disk_bytes": 200,
            "operations": 7,
            "learning": 0.1,
            "episodes": 1,
        }

    report = parity.capability_density_frontier(
        [{"name": "live_arm", "runner": runner}],
        envelopes=("512MB", "1GB"),
        run_live=True,
    )
    assert len(report["rows"]) == 2
    for row in report["rows"]:
        assert row["process_peak_resident_bytes"] > 0
        assert row["operation_count_energy_proxy"] == 7
        assert row["energy_is_proxy"] is True
        assert row["wall_clock_seconds_per_episode"] is not None
        assert row["wall_clock_seconds_per_episode"] >= 0.0
    assert report["process_peak_resident_bytes"]["source"].startswith("resource.getrusage")
