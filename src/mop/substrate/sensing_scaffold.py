"""Sensing and evidence scaffold contracts over the Wave E0 event graph.

This module raises the scaffolding axis for four potential-atlas facets: native computer audio
(SR3), semantic coverage and perspective plurality (SR5), current substrate controls (SR6), and
data rights and privacy authority (EV4). It supplies machine-checkable contracts, deterministic
synthetic fixtures, declared controls and nulls, and refusal rules that fail closed. It performs
no audio decode, loads no model weights, and touches no network or natural data.

Claim scope: deterministic programmatic mechanics only; no capability claim. A green contract
here means the declarations are complete and self-consistent, never that native audio, extra
perspectives, or rights-clean cohorts carry any demonstrated value.

Composition policy (FORM_SUBSTRATE_CODEMAP.md section 2): this module composes the existing
Wave E0 primitives (events.EventGraph, scenario_factory.make_scenario, the JOIN_CONTROLS probe
pattern) and adds only the sensing-specific contract layer. It duplicates no referent, store,
alignment, or transfer logic.

House style: no em dashes or en dashes.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..environments.scenario_factory import ScenarioBundle
from ..substrate.events import (
    EntityRef,
    EventGraph,
    EventRef,
    FrozenJSON,
    ObservationRef,
    SensorClock,
    SensorClockRef,
    canonical_bytes,
    canonical_sha256,
)

SENSING_SCAFFOLD_SCHEMA = "mop-sensing-scaffold/v1"
# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so the substrate layer never imports the experiments layer; a unit test pins equality.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")

WAVEFORM_ENCODING = "pcm_s16le"
WAVEFORM_KINDS = ("primary", "random-spectral", "handcrafted-spectral", "muted", "shuffled-time")
SPECTRAL_CONTROL_ARMS = ("random-spectral", "handcrafted-spectral")
AUDIO_PROBE_CONTROLS = ("shuffled-time", "muted", "audio-only")

INFORMATION_SOURCES = (
    "vision-cache",
    "audio-waveform",
    "language-text",
    "action-log",
    "code-text",
    "numerosity-count",
    "telemetry",
)
REQUIRED_PERSPECTIVE_BASELINES = ("all-other-perspectives", "shuffled-perspective")
HELD_OUT_SPLIT = "held-out"

RIGHTS_FIELDS = (
    "copyright",
    "attribution",
    "privacy",
    "publicity",
    "trademark",
    "redistribution",
    "derivatives",
)
RIGHTS_STATUSES = ("cleared", "restricted", "unknown")
REUSE_SCOPES = ("research-local", "redistribution", "derivative-works")

CONTRADICTION_BASELINES = ("majority-vote", "mean-fusion")
TEMPORAL_BASELINE_ARM = "fixed-window"
CAUSAL_BASELINE_ARM = "temporal-correlation"


class ScaffoldRefusal(ValueError):
    """Raised whenever a declaration is missing, malformed, or outside its declared scope."""


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ScaffoldRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise ScaffoldRefusal(f"{label} must use stable lowercase characters")


# ---------------------------------------------------------------------------------------------
# (a) Native audio form contract: SR3. Contracts and synthetic fixtures only; no decode.
# ---------------------------------------------------------------------------------------------


def _seeded_bytes(seed: int, label: str, count: int) -> bytes:
    """Deterministic byte stream from repeated hashing; no OS entropy, no time dependence."""

    if seed < 0 or count < 0:
        raise ScaffoldRefusal("seeded byte stream requires nonnegative seed and count")
    blocks: list[bytes] = []
    block_index = 0
    while sum(len(b) for b in blocks) < count:
        blocks.append(
            hashlib.sha256(canonical_bytes({"seed": seed, "label": label, "block": block_index})).digest()
        )
        block_index += 1
    return b"".join(blocks)[:count]


@dataclass(frozen=True, slots=True)
class WaveformSpec:
    """Declaration of raw waveform bytes: shape, clock rate, and content digest.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    waveform_sha256: str
    sample_rate_hz: int
    num_samples: int
    num_channels: int = 1
    bit_depth: int = 16
    encoding: str = WAVEFORM_ENCODING

    def __post_init__(self) -> None:
        _require_sha256(self.waveform_sha256, "WaveformSpec.waveform_sha256")
        if self.sample_rate_hz <= 0 or self.num_samples <= 0:
            raise ScaffoldRefusal("waveform sample rate and sample count must be positive")
        if self.num_channels not in (1, 2):
            raise ScaffoldRefusal("waveform channel count must be 1 or 2")
        if self.bit_depth != 16 or self.encoding != WAVEFORM_ENCODING:
            raise ScaffoldRefusal("waveform contract v1 admits pcm_s16le 16-bit only")

    @property
    def byte_count(self) -> int:
        return self.num_samples * self.num_channels * (self.bit_depth // 8)

    def payload(self) -> dict[str, Any]:
        return {
            "waveform_sha256": self.waveform_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "num_samples": self.num_samples,
            "num_channels": self.num_channels,
            "bit_depth": self.bit_depth,
            "encoding": self.encoding,
        }


def make_synthetic_waveform(
    *,
    seed: int,
    kind: str,
    sample_rate_hz: int = 16_000,
    num_samples: int = 32,
) -> tuple[bytes, WaveformSpec]:
    """Deterministic synthetic PCM fixture. Not audio evidence; a byte-exact test vector.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    if kind not in WAVEFORM_KINDS:
        raise ScaffoldRefusal(f"unsupported waveform kind {kind!r}")
    if seed < 0:
        raise ScaffoldRefusal("waveform seed must be nonnegative")
    if num_samples < 2:
        raise ScaffoldRefusal("synthetic waveform needs at least two samples")

    def encode(samples: list[int]) -> bytes:
        return b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples)

    if kind == "muted":
        raw = encode([0] * num_samples)
    elif kind == "random-spectral":
        noise = _seeded_bytes(seed, "random-spectral", num_samples * 2)
        raw = encode(
            [
                int.from_bytes(noise[2 * i : 2 * i + 2], "little", signed=False) % 2048 - 1024
                for i in range(num_samples)
            ]
        )
    elif kind == "handcrafted-spectral":
        period = 8
        raw = encode(
            [
                (i % period) * 256 - 1024 if (i // period) % 2 == 0 else 1024 - (i % period) * 256
                for i in range(num_samples)
            ]
        )
    else:
        primary = [((i * 991 + seed * 7919) % 2048) - 1024 for i in range(num_samples)]
        if kind == "shuffled-time":
            primary = list(reversed(primary))
        raw = encode(primary)

    spec = WaveformSpec(
        waveform_sha256=hashlib.sha256(raw).hexdigest(),
        sample_rate_hz=sample_rate_hz,
        num_samples=num_samples,
    )
    if kind == "shuffled-time":
        original_spec = make_synthetic_waveform(
            seed=seed, kind="primary", sample_rate_hz=sample_rate_hz, num_samples=num_samples
        )[1]
        if original_spec.waveform_sha256 == spec.waveform_sha256:
            raise ScaffoldRefusal("shuffled-time fixture collapsed onto the primary waveform")
    return raw, spec


@dataclass(frozen=True, slots=True)
class SampleClockBinding:
    """Binds a waveform to one Wave E0 sensor clock with exact integer duration agreement.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    waveform_sha256: str
    clock_ref: SensorClockRef
    sample_rate_hz: int
    num_samples: int

    def __post_init__(self) -> None:
        _require_sha256(self.waveform_sha256, "SampleClockBinding.waveform_sha256")
        if self.sample_rate_hz <= 0 or self.num_samples <= 0:
            raise ScaffoldRefusal("clock binding sample rate and sample count must be positive")

    def assert_consistent(self, clock: SensorClock) -> None:
        if clock.ref != self.clock_ref:
            raise ScaffoldRefusal("clock binding references a different sensor clock")
        span_ticks = clock.capture_end_tick - clock.capture_start_tick
        if span_ticks <= 0:
            raise ScaffoldRefusal("waveform capture interval must span at least one tick")
        # Exact integer cross-multiplication: num_samples / rate == span_ticks * period_ns / 1e9.
        if self.num_samples * 1_000_000_000 != self.sample_rate_hz * span_ticks * clock.tick_period_ns:
            raise ScaffoldRefusal("waveform duration does not equal the declared capture interval")

    def payload(self) -> dict[str, Any]:
        return {
            "waveform_sha256": self.waveform_sha256,
            "clock_ref": str(self.clock_ref),
            "sample_rate_hz": self.sample_rate_hz,
            "num_samples": self.num_samples,
        }


@dataclass(frozen=True, slots=True)
class AudioEventAlignment:
    """Declares which event-graph observation a waveform is evidence about.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    graph_sha256: str
    event_ref: EventRef
    observation_ref: ObservationRef
    entity_refs: tuple[EntityRef, ...]
    waveform_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.graph_sha256, "AudioEventAlignment.graph_sha256")
        _require_sha256(self.waveform_sha256, "AudioEventAlignment.waveform_sha256")
        if not self.entity_refs or len(set(self.entity_refs)) != len(self.entity_refs):
            raise ScaffoldRefusal("alignment entity references must be nonempty and unique")

    def payload(self) -> dict[str, Any]:
        return {
            "graph_sha256": self.graph_sha256,
            "event_ref": str(self.event_ref),
            "observation_ref": str(self.observation_ref),
            "entity_refs": [str(ref) for ref in self.entity_refs],
            "waveform_sha256": self.waveform_sha256,
        }


def validate_audio_alignment(
    *,
    graph: EventGraph,
    alignment: AudioEventAlignment,
    spec: WaveformSpec,
    binding: SampleClockBinding,
) -> None:
    """Fail closed unless waveform, clock, and event graph all agree byte-exactly."""

    if graph.sha256 != alignment.graph_sha256:
        raise ScaffoldRefusal("alignment names a different event graph")
    if spec.waveform_sha256 != alignment.waveform_sha256:
        raise ScaffoldRefusal("alignment and waveform spec digests disagree")
    if binding.waveform_sha256 != spec.waveform_sha256:
        raise ScaffoldRefusal("clock binding and waveform spec digests disagree")
    if binding.sample_rate_hz != spec.sample_rate_hz or binding.num_samples != spec.num_samples:
        raise ScaffoldRefusal("clock binding shape disagrees with the waveform spec")
    observations = {str(row.ref): row for row in graph.observations}
    observation = observations.get(str(alignment.observation_ref))
    if observation is None:
        raise ScaffoldRefusal("alignment observation is absent from the event graph")
    if observation.event_ref != alignment.event_ref:
        raise ScaffoldRefusal("alignment event does not own the aligned observation")
    if set(alignment.entity_refs) != set(observation.entity_refs):
        raise ScaffoldRefusal("alignment entities disagree with the observation entities")
    binding.assert_consistent(observation.clock)


@dataclass(frozen=True, slots=True)
class SpectralControlDeclaration:
    """One declared audio control arm on identical shape and clock: the SR6 matched-input pattern.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    arm: str
    seed: int
    generator_id: str
    spec: WaveformSpec

    def __post_init__(self) -> None:
        if self.arm not in SPECTRAL_CONTROL_ARMS:
            raise ScaffoldRefusal(f"unsupported spectral control arm {self.arm!r}")
        if self.seed < 0:
            raise ScaffoldRefusal("spectral control seed must be nonnegative")
        _require_id(self.generator_id, "SpectralControlDeclaration.generator_id")

    def assert_matched_input(self, primary: WaveformSpec) -> None:
        if (
            self.spec.sample_rate_hz != primary.sample_rate_hz
            or self.spec.num_samples != primary.num_samples
            or self.spec.num_channels != primary.num_channels
        ):
            raise ScaffoldRefusal(f"control arm {self.arm} is not input-matched to the primary waveform")
        if self.spec.waveform_sha256 == primary.waveform_sha256:
            raise ScaffoldRefusal(f"control arm {self.arm} duplicates the primary waveform bytes")

    def payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "generator_id": self.generator_id,
            "spec": self.spec.payload(),
        }


@dataclass(frozen=True, slots=True)
class AudioProbeContract:
    """One refusal-ruled probe declaration; every probe is a preregistered negative control.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    control: str
    probe_waveform_sha256: str
    is_silent: bool
    vision_withheld: bool
    expected_control_accept: bool
    ground_truth_relation: str

    def __post_init__(self) -> None:
        if self.control not in AUDIO_PROBE_CONTROLS:
            raise ScaffoldRefusal(f"unsupported audio probe control {self.control!r}")
        _require_sha256(self.probe_waveform_sha256, "AudioProbeContract.probe_waveform_sha256")
        if self.expected_control_accept:
            raise ScaffoldRefusal(f"negative control {self.control} cannot be marked accepted")
        if not self.ground_truth_relation.strip():
            raise ScaffoldRefusal("audio probe ground truth must not be empty")
        if self.control == "muted" and not self.is_silent:
            raise ScaffoldRefusal("muted probe must declare a silent waveform")
        if self.control != "muted" and self.is_silent:
            raise ScaffoldRefusal("only the muted probe may declare silence")
        if self.control == "audio-only" and not self.vision_withheld:
            raise ScaffoldRefusal("audio-only probe must withhold vision")
        if self.control != "audio-only" and self.vision_withheld:
            raise ScaffoldRefusal("only the audio-only probe may withhold vision")

    def payload(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "probe_waveform_sha256": self.probe_waveform_sha256,
            "is_silent": self.is_silent,
            "vision_withheld": self.vision_withheld,
            "expected_control_accept": self.expected_control_accept,
            "ground_truth_relation": self.ground_truth_relation,
        }


@dataclass(frozen=True, slots=True)
class NativeAudioFormContract:
    """Complete native audio declaration: waveform, clock, alignment, controls, refusal probes.

    A valid instance means the declaration set is complete and self-consistent. It carries no
    decode result and no probe outcome. Claim scope: deterministic programmatic mechanics only;
    no capability claim.
    """

    spec: WaveformSpec
    binding: SampleClockBinding
    alignment: AudioEventAlignment
    controls: tuple[SpectralControlDeclaration, ...]
    probes: tuple[AudioProbeContract, ...]
    decode_performed: bool = False
    claim_scope: str = CLAIM_SCOPE
    schema: str = SENSING_SCAFFOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SENSING_SCAFFOLD_SCHEMA:
            raise ScaffoldRefusal(f"unsupported sensing scaffold schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ScaffoldRefusal("native audio claim scope cannot be widened")
        if self.decode_performed:
            raise ScaffoldRefusal("native audio contract v1 is declaration-only; decode is out of scope")
        if self.binding.waveform_sha256 != self.spec.waveform_sha256:
            raise ScaffoldRefusal("contract binding does not reference the primary waveform")
        if self.alignment.waveform_sha256 != self.spec.waveform_sha256:
            raise ScaffoldRefusal("contract alignment does not reference the primary waveform")
        if tuple(row.arm for row in self.controls) != SPECTRAL_CONTROL_ARMS:
            raise ScaffoldRefusal("spectral control coverage is incomplete or out of canonical order")
        for control in self.controls:
            control.assert_matched_input(self.spec)
        if tuple(row.control for row in self.probes) != AUDIO_PROBE_CONTROLS:
            raise ScaffoldRefusal("audio probe coverage is incomplete or out of canonical order")
        for probe in self.probes:
            if probe.control == "shuffled-time" and (
                probe.probe_waveform_sha256 == self.spec.waveform_sha256
            ):
                raise ScaffoldRefusal("shuffled-time probe bytes must differ from the primary waveform")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "decode_performed": self.decode_performed,
            "spec": self.spec.payload(),
            "binding": self.binding.payload(),
            "alignment": self.alignment.payload(),
            "controls": [row.payload() for row in self.controls],
            "probes": [row.payload() for row in self.probes],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------------------------
# (b) Perspective plurality contract: SR5. Source, computation, metric, and slot-removal rules.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UniqueInformationMetricContract:
    """Preregisters how a perspective's unique information will be measured, before any run.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    metric_name: str
    evaluation_split: str
    baselines: tuple[str, ...]
    conditional_value_definition: str

    def __post_init__(self) -> None:
        _require_id(self.metric_name, "UniqueInformationMetricContract.metric_name")
        if self.evaluation_split != HELD_OUT_SPLIT:
            raise ScaffoldRefusal("unique information must be preregistered on the held-out split")
        if tuple(self.baselines) != REQUIRED_PERSPECTIVE_BASELINES:
            raise ScaffoldRefusal("perspective baselines are incomplete or out of canonical order")
        if not self.conditional_value_definition.strip():
            raise ScaffoldRefusal("conditional value definition must not be empty")

    def payload(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "evaluation_split": self.evaluation_split,
            "baselines": list(self.baselines),
            "conditional_value_definition": self.conditional_value_definition,
        }


@dataclass(frozen=True, slots=True)
class PerspectiveDeclaration:
    """A perspective must name a distinct information source or a distinct computation.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    perspective_id: str
    information_source: str
    computation: str
    metric: UniqueInformationMetricContract

    def __post_init__(self) -> None:
        _require_id(self.perspective_id, "PerspectiveDeclaration.perspective_id")
        if self.information_source not in INFORMATION_SOURCES:
            raise ScaffoldRefusal(f"unsupported information source {self.information_source!r}")
        if not self.computation.strip():
            raise ScaffoldRefusal("perspective computation must not be empty")

    def payload(self) -> dict[str, Any]:
        return {
            "perspective_id": self.perspective_id,
            "information_source": self.information_source,
            "computation": self.computation,
            "metric": self.metric.payload(),
        }


@dataclass(frozen=True, slots=True)
class PerspectivePluralityContract:
    """A plurality is admissible only when no two slots share source and computation.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    declarations: tuple[PerspectiveDeclaration, ...]
    claim_scope: str = CLAIM_SCOPE
    schema: str = SENSING_SCAFFOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SENSING_SCAFFOLD_SCHEMA:
            raise ScaffoldRefusal(f"unsupported sensing scaffold schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ScaffoldRefusal("perspective plurality claim scope cannot be widened")
        if len(self.declarations) < 2:
            raise ScaffoldRefusal("a plurality contract needs at least two perspectives")
        ids = [row.perspective_id for row in self.declarations]
        if len(set(ids)) != len(ids):
            raise ScaffoldRefusal("perspective ids must be unique")
        signatures = [(row.information_source, row.computation.strip()) for row in self.declarations]
        if len(set(signatures)) != len(signatures):
            raise ScaffoldRefusal("two perspectives share one information source and computation")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "declarations": [row.payload() for row in self.declarations],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def apply_slot_removal(
    contract: PerspectivePluralityContract,
    conditional_values: Mapping[str, float],
    *,
    min_conditional_value: float,
    evaluation_split: str = HELD_OUT_SPLIT,
) -> dict[str, Any]:
    """The slot-removal rule as code: remove every slot that adds no held-out conditional value.

    Fails closed on a missing, non-finite, or wrongly split measurement. Claim scope:
    deterministic programmatic mechanics only; no capability claim.
    """

    if evaluation_split != HELD_OUT_SPLIT:
        raise ScaffoldRefusal("slot removal is licensed only by held-out conditional values")
    declared = {row.perspective_id for row in contract.declarations}
    if set(conditional_values) != declared:
        raise ScaffoldRefusal("conditional values must cover exactly the declared perspectives")
    for perspective_id, value in conditional_values.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            raise ScaffoldRefusal(f"conditional value for {perspective_id} is not a finite number")
    retained = sorted(
        perspective_id
        for perspective_id, value in conditional_values.items()
        if value >= min_conditional_value
    )
    removed = sorted(declared - set(retained))
    return {
        "retained": retained,
        "removed": removed,
        "min_conditional_value": min_conditional_value,
        "evaluation_split": evaluation_split,
        "claim_scope": CLAIM_SCOPE,
    }


# ---------------------------------------------------------------------------------------------
# (c) Rights and authority source card: EV4. Refusal of unknown authority and split reuse as code.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RightsFieldDeclaration:
    """One rights field with its named authority; never inferred from a dataset-level license.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    status: str
    authority: str
    inferred_from_dataset_license: bool = False

    def __post_init__(self) -> None:
        if self.status not in RIGHTS_STATUSES:
            raise ScaffoldRefusal(f"unsupported rights status {self.status!r}")
        if self.status != "unknown" and not self.authority.strip():
            raise ScaffoldRefusal("a cleared or restricted rights field must name its authority")

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "authority": self.authority,
            "inferred_from_dataset_license": self.inferred_from_dataset_license,
        }


@dataclass(frozen=True, slots=True)
class SourceCard:
    """Per-source rights card covering all seven fields with a closed reuse scope list.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    source_id: str
    steward: str
    fields: Mapping[str, RightsFieldDeclaration]
    allowed_uses: tuple[str, ...]
    claim_scope: str = CLAIM_SCOPE
    schema: str = SENSING_SCAFFOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SENSING_SCAFFOLD_SCHEMA:
            raise ScaffoldRefusal(f"unsupported sensing scaffold schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ScaffoldRefusal("source card claim scope cannot be widened")
        _require_id(self.source_id, "SourceCard.source_id")
        if not self.steward.strip():
            raise ScaffoldRefusal("source card steward must not be empty")
        if tuple(sorted(self.fields)) != tuple(sorted(RIGHTS_FIELDS)):
            raise ScaffoldRefusal("source card must cover exactly the seven rights fields")
        if not self.allowed_uses or len(set(self.allowed_uses)) != len(self.allowed_uses):
            raise ScaffoldRefusal("allowed uses must be nonempty and unique")
        if any(use not in REUSE_SCOPES for use in self.allowed_uses):
            raise ScaffoldRefusal("allowed uses must come from the closed reuse scope vocabulary")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "source_id": self.source_id,
            "steward": self.steward,
            "fields": {name: self.fields[name].payload() for name in RIGHTS_FIELDS},
            "allowed_uses": list(self.allowed_uses),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


_USE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "research-local": ("copyright", "privacy", "publicity"),
    "redistribution": ("copyright", "attribution", "privacy", "publicity", "trademark", "redistribution"),
    "derivative-works": RIGHTS_FIELDS,
}


def assert_natural_experiment_admissible(card: SourceCard, requested_use: str) -> None:
    """Refuse unknown authority, inferred fields, and split reuse. The rule is code, not prose.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    if requested_use not in REUSE_SCOPES:
        raise ScaffoldRefusal(f"unsupported requested use {requested_use!r}")
    if requested_use not in card.allowed_uses:
        raise ScaffoldRefusal(
            f"split reuse refused: {card.source_id} does not declare {requested_use!r} as an allowed use"
        )
    for name in RIGHTS_FIELDS:
        declaration = card.fields[name]
        if declaration.status == "unknown":
            raise ScaffoldRefusal(f"unknown authority refused: rights field {name!r} on {card.source_id}")
        if declaration.inferred_from_dataset_license:
            raise ScaffoldRefusal(
                f"inferred authority refused: rights field {name!r} on {card.source_id} "
                "cites only a dataset-level license"
            )
    for name in _USE_REQUIRED_FIELDS[requested_use]:
        if card.fields[name].status != "cleared":
            raise ScaffoldRefusal(
                f"restricted authority refused: {requested_use!r} needs cleared {name!r} on {card.source_id}"
            )


# ---------------------------------------------------------------------------------------------
# (d) Experiment contracts over the Wave E0 event graph: f21, f26, f27.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalBindingContract:
    """f21 asynchronous temporal binding declaration over one delayed-sensor event.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    event_ref: EventRef
    early_observation_ref: ObservationRef
    late_observation_ref: ObservationRef
    arrival_skew_ticks: int
    binding_window_ticks: int
    fixed_window_baseline_ticks: int
    shuffled_timing_left_ref: ObservationRef
    shuffled_timing_right_ref: ObservationRef
    shuffled_timing_gap_ticks: int
    baseline_arm: str = TEMPORAL_BASELINE_ARM
    claim_scope: str = CLAIM_SCOPE
    schema: str = SENSING_SCAFFOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SENSING_SCAFFOLD_SCHEMA:
            raise ScaffoldRefusal(f"unsupported sensing scaffold schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ScaffoldRefusal("temporal binding claim scope cannot be widened")
        if self.baseline_arm != TEMPORAL_BASELINE_ARM:
            raise ScaffoldRefusal("temporal binding requires the fixed-window baseline arm")
        if self.early_observation_ref == self.late_observation_ref:
            raise ScaffoldRefusal("temporal binding needs two distinct observations")
        if self.arrival_skew_ticks <= 0:
            raise ScaffoldRefusal("temporal binding requires genuinely asynchronous arrivals")
        if self.binding_window_ticks < self.arrival_skew_ticks:
            raise ScaffoldRefusal("binding window cannot exclude the declared primary pair")
        if self.fixed_window_baseline_ticks <= 0:
            raise ScaffoldRefusal("fixed window baseline must be positive")
        if self.shuffled_timing_gap_ticks <= self.binding_window_ticks:
            raise ScaffoldRefusal("shuffled timing control must fall outside the binding window")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "event_ref": str(self.event_ref),
            "early_observation_ref": str(self.early_observation_ref),
            "late_observation_ref": str(self.late_observation_ref),
            "arrival_skew_ticks": self.arrival_skew_ticks,
            "binding_window_ticks": self.binding_window_ticks,
            "fixed_window_baseline_ticks": self.fixed_window_baseline_ticks,
            "shuffled_timing_left_ref": str(self.shuffled_timing_left_ref),
            "shuffled_timing_right_ref": str(self.shuffled_timing_right_ref),
            "shuffled_timing_gap_ticks": self.shuffled_timing_gap_ticks,
            "baseline_arm": self.baseline_arm,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def build_temporal_binding_contract(
    bundle: ScenarioBundle, *, binding_window_ticks: int | None = None
) -> TemporalBindingContract:
    """Build the f21 contract from the shared Wave E0 fixture; refuse an unusable fixture."""

    delay_event_refs = {event.ref for event in bundle.graph.events if event.kind == "delay"}
    delay_rows = [
        row
        for row in bundle.graph.observations
        if row.role == "primary" and row.event_ref in delay_event_refs
    ]
    if len(delay_rows) < 2:
        raise ScaffoldRefusal("fixture lacks two primary observations of the delayed-sensor event")
    delay_rows.sort(key=lambda row: (row.clock.arrival_tick, str(row.ref)))
    early, late = delay_rows[0], delay_rows[-1]
    skew = late.clock.arrival_tick - early.clock.arrival_tick
    if skew <= 0:
        raise ScaffoldRefusal("fixture delayed-sensor arrivals are not asynchronous")
    if not early.clock.overlaps(late.clock):
        raise ScaffoldRefusal("fixture delayed-sensor capture intervals must overlap")
    window = skew if binding_window_ticks is None else binding_window_ticks
    wrong_time = next((probe for probe in bundle.join_probes if probe.control == "wrong-time"), None)
    if wrong_time is None:
        raise ScaffoldRefusal("fixture lacks the wrong-time join probe")
    observations = {str(row.ref): row for row in bundle.graph.observations}
    control_left = observations[str(wrong_time.left_observation_ref)]
    control_right = observations[str(wrong_time.right_observation_ref)]
    gap = abs(control_right.clock.arrival_tick - control_left.clock.arrival_tick)
    return TemporalBindingContract(
        event_ref=early.event_ref,
        early_observation_ref=early.ref,
        late_observation_ref=late.ref,
        arrival_skew_ticks=skew,
        binding_window_ticks=window,
        fixed_window_baseline_ticks=max(1, skew // 2),
        shuffled_timing_left_ref=wrong_time.left_observation_ref,
        shuffled_timing_right_ref=wrong_time.right_observation_ref,
        shuffled_timing_gap_ticks=gap,
    )


@dataclass(frozen=True, slots=True)
class SourceReport:
    """One source's claim about a shared event, carried with its content digest.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    source_id: str
    event_ref: EventRef
    claim: int
    content: FrozenJSON

    def __post_init__(self) -> None:
        _require_id(self.source_id, "SourceReport.source_id")

    def payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "event_ref": str(self.event_ref),
            "claim": self.claim,
            "content": self.content.payload(),
        }


@dataclass(frozen=True, slots=True)
class ContradictionTriangulationContract:
    """f26 cross-form contradiction declaration: exactly one dissenting source by construction.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    event_ref: EventRef
    reports: tuple[SourceReport, ...]
    contradicted_source_id: str
    baselines: tuple[str, ...] = CONTRADICTION_BASELINES
    metric_name: str = "source_localization_auroc"
    claim_scope: str = CLAIM_SCOPE
    schema: str = SENSING_SCAFFOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SENSING_SCAFFOLD_SCHEMA:
            raise ScaffoldRefusal(f"unsupported sensing scaffold schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ScaffoldRefusal("contradiction triangulation claim scope cannot be widened")
        if tuple(self.baselines) != CONTRADICTION_BASELINES:
            raise ScaffoldRefusal("contradiction baselines are incomplete or out of canonical order")
        _require_id(self.metric_name, "ContradictionTriangulationContract.metric_name")
        if len(self.reports) < 3:
            raise ScaffoldRefusal("triangulation needs at least three independent source reports")
        source_ids = [row.source_id for row in self.reports]
        if len(set(source_ids)) != len(source_ids):
            raise ScaffoldRefusal("triangulation source ids must be unique")
        if any(row.event_ref != self.event_ref for row in self.reports):
            raise ScaffoldRefusal("every report must describe the shared event")
        if self.contradicted_source_id not in source_ids:
            raise ScaffoldRefusal("the contradicted source must be one of the reports")
        consensus = [row.claim for row in self.reports if row.source_id != self.contradicted_source_id]
        dissent = next(row for row in self.reports if row.source_id == self.contradicted_source_id)
        if len(set(consensus)) != 1:
            raise ScaffoldRefusal("consensus sources must agree exactly on the shared claim")
        if dissent.claim == consensus[0]:
            raise ScaffoldRefusal("the designated contradicted source does not actually dissent")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "event_ref": str(self.event_ref),
            "reports": [row.payload() for row in self.reports],
            "contradicted_source_id": self.contradicted_source_id,
            "baselines": list(self.baselines),
            "metric_name": self.metric_name,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def build_contradiction_contract(bundle: ScenarioBundle) -> ContradictionTriangulationContract:
    """Build the f26 contract from the shared Wave E0 fixture; refuse an unusable fixture."""

    delay_event = next((row for row in bundle.graph.events if row.kind == "delay"), None)
    if delay_event is None:
        raise ScaffoldRefusal("fixture lacks a delayed-sensor event for triangulation")
    delay_data = delay_event.data.value()
    shared_claim = 3 if not isinstance(delay_data, dict) else int(delay_data.get("arrival_delay_ticks", 3))
    source_ids = ("vision", "audio", "telemetry")
    contradicted = source_ids[bundle.seed % len(source_ids)]
    reports = tuple(
        SourceReport(
            source_id=source_id,
            event_ref=delay_event.ref,
            claim=shared_claim + 6 if source_id == contradicted else shared_claim,
            content=FrozenJSON.from_value(
                {
                    "source_id": source_id,
                    "event_ref": str(delay_event.ref),
                    "claim": shared_claim + 6 if source_id == contradicted else shared_claim,
                    "seed": bundle.seed,
                }
            ),
        )
        for source_id in source_ids
    )
    return ContradictionTriangulationContract(
        event_ref=delay_event.ref,
        reports=reports,
        contradicted_source_id=contradicted,
    )


@dataclass(frozen=True, slots=True)
class CausalCrossmodalBindingContract:
    """f27 causal crossmodal binding declaration over same-parent intervention branches.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    fork_event_ref: EventRef
    parent_state_sha256: str
    branch_refs: tuple[str, ...]
    chosen_branch_ref: str
    synchronous_unrelated_left_ref: ObservationRef
    synchronous_unrelated_right_ref: ObservationRef
    baseline_arm: str = CAUSAL_BASELINE_ARM
    metric_name: str = "unseen_intervention_prediction_error"
    claim_scope: str = CLAIM_SCOPE
    schema: str = SENSING_SCAFFOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SENSING_SCAFFOLD_SCHEMA:
            raise ScaffoldRefusal(f"unsupported sensing scaffold schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ScaffoldRefusal("causal binding claim scope cannot be widened")
        if self.baseline_arm != CAUSAL_BASELINE_ARM:
            raise ScaffoldRefusal("causal binding requires the temporal-correlation baseline arm")
        _require_sha256(self.parent_state_sha256, "CausalCrossmodalBindingContract.parent_state_sha256")
        _require_id(self.metric_name, "CausalCrossmodalBindingContract.metric_name")
        if len(self.branch_refs) < 2 or len(set(self.branch_refs)) != len(self.branch_refs):
            raise ScaffoldRefusal("causal binding needs at least two unique same-parent branches")
        if self.chosen_branch_ref not in self.branch_refs:
            raise ScaffoldRefusal("the chosen branch must be one of the declared branches")
        if self.synchronous_unrelated_left_ref == self.synchronous_unrelated_right_ref:
            raise ScaffoldRefusal("the synchronous-unrelated control needs two distinct observations")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "fork_event_ref": str(self.fork_event_ref),
            "parent_state_sha256": self.parent_state_sha256,
            "branch_refs": list(self.branch_refs),
            "chosen_branch_ref": self.chosen_branch_ref,
            "synchronous_unrelated_left_ref": str(self.synchronous_unrelated_left_ref),
            "synchronous_unrelated_right_ref": str(self.synchronous_unrelated_right_ref),
            "baseline_arm": self.baseline_arm,
            "metric_name": self.metric_name,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def build_causal_binding_contract(bundle: ScenarioBundle) -> CausalCrossmodalBindingContract:
    """Build the f27 contract from the shared Wave E0 fixture; refuse an unusable fixture."""

    fork_event = next((row for row in bundle.graph.events if row.kind == "intervention"), None)
    if fork_event is None:
        raise ScaffoldRefusal("fixture lacks an intervention fork event")
    branches = [row for row in bundle.graph.branches if row.fork_event_ref == fork_event.ref]
    if len(branches) < 2:
        raise ScaffoldRefusal("fixture intervention fork needs at least two branches")
    parent_digests = {row.parent_state.sha256 for row in branches}
    if len(parent_digests) != 1:
        raise ScaffoldRefusal("fixture branches do not share one parent state")
    chosen = [row for row in branches if row.chosen]
    if len(chosen) != 1:
        raise ScaffoldRefusal("fixture fork must have exactly one chosen branch")
    wrong_event = next((probe for probe in bundle.join_probes if probe.control == "wrong-event"), None)
    if wrong_event is None:
        raise ScaffoldRefusal("fixture lacks the wrong-event join probe for the synchrony control")
    observations = {str(row.ref): row for row in bundle.graph.observations}
    left = observations[str(wrong_event.left_observation_ref)]
    right = observations[str(wrong_event.right_observation_ref)]
    if not left.clock.overlaps(right.clock):
        raise ScaffoldRefusal("the synchronous-unrelated control observations must overlap in time")
    return CausalCrossmodalBindingContract(
        fork_event_ref=fork_event.ref,
        parent_state_sha256=next(iter(parent_digests)),
        branch_refs=tuple(sorted(str(row.ref) for row in branches)),
        chosen_branch_ref=str(chosen[0].ref),
        synchronous_unrelated_left_ref=wrong_event.left_observation_ref,
        synchronous_unrelated_right_ref=wrong_event.right_observation_ref,
    )


# ---------------------------------------------------------------------------------------------
# Fixture bundle: everything above from one seed, digest-bound for exact replay.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensingScaffoldFixture:
    """One seed's complete deterministic sensing scaffold with a stable digest.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    seed: int
    scenario: ScenarioBundle
    audio: NativeAudioFormContract
    perspectives: PerspectivePluralityContract
    source_card: SourceCard
    temporal: TemporalBindingContract
    contradiction: ContradictionTriangulationContract
    causal: CausalCrossmodalBindingContract
    schema: str = field(default=SENSING_SCAFFOLD_SCHEMA)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "scenario_sha256": self.scenario.sha256,
            "audio": self.audio.payload(),
            "perspectives": self.perspectives.payload(),
            "source_card": self.source_card.payload(),
            "temporal": self.temporal.payload(),
            "contradiction": self.contradiction.payload(),
            "causal": self.causal.payload(),
            "claim_scope": CLAIM_SCOPE,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def _fixture_audio_contract(bundle: ScenarioBundle) -> NativeAudioFormContract:
    graph = bundle.graph
    audio_rows = [row for row in graph.observations if row.clock.sensor_id == "audio"]
    if not audio_rows:
        raise ScaffoldRefusal("fixture lacks an audio-clocked observation")
    audio_observation = audio_rows[0]
    span_ticks = audio_observation.clock.capture_end_tick - audio_observation.clock.capture_start_tick
    sample_rate_hz = 16_000
    num_samples = sample_rate_hz * span_ticks * audio_observation.clock.tick_period_ns // 1_000_000_000
    _, spec = make_synthetic_waveform(
        seed=bundle.seed, kind="primary", sample_rate_hz=sample_rate_hz, num_samples=num_samples
    )
    binding = SampleClockBinding(
        waveform_sha256=spec.waveform_sha256,
        clock_ref=audio_observation.clock.ref,
        sample_rate_hz=sample_rate_hz,
        num_samples=num_samples,
    )
    alignment = AudioEventAlignment(
        graph_sha256=graph.sha256,
        event_ref=audio_observation.event_ref,
        observation_ref=audio_observation.ref,
        entity_refs=audio_observation.entity_refs,
        waveform_sha256=spec.waveform_sha256,
    )
    controls = tuple(
        SpectralControlDeclaration(
            arm=arm,
            seed=bundle.seed,
            generator_id=f"sensing-scaffold.{arm}",
            spec=make_synthetic_waveform(
                seed=bundle.seed, kind=arm, sample_rate_hz=sample_rate_hz, num_samples=num_samples
            )[1],
        )
        for arm in SPECTRAL_CONTROL_ARMS
    )
    _, shuffled_spec = make_synthetic_waveform(
        seed=bundle.seed, kind="shuffled-time", sample_rate_hz=sample_rate_hz, num_samples=num_samples
    )
    _, muted_spec = make_synthetic_waveform(
        seed=bundle.seed, kind="muted", sample_rate_hz=sample_rate_hz, num_samples=num_samples
    )
    probes = (
        AudioProbeContract(
            control="shuffled-time",
            probe_waveform_sha256=shuffled_spec.waveform_sha256,
            is_silent=False,
            vision_withheld=False,
            expected_control_accept=False,
            ground_truth_relation="same-bytes-permuted-order",
        ),
        AudioProbeContract(
            control="muted",
            probe_waveform_sha256=muted_spec.waveform_sha256,
            is_silent=True,
            vision_withheld=False,
            expected_control_accept=False,
            ground_truth_relation="same-clock-zero-signal",
        ),
        AudioProbeContract(
            control="audio-only",
            probe_waveform_sha256=spec.waveform_sha256,
            is_silent=False,
            vision_withheld=True,
            expected_control_accept=False,
            ground_truth_relation="same-waveform-vision-withheld",
        ),
    )
    return NativeAudioFormContract(
        spec=spec, binding=binding, alignment=alignment, controls=controls, probes=probes
    )


def _fixture_perspectives() -> PerspectivePluralityContract:
    def metric(name: str) -> UniqueInformationMetricContract:
        return UniqueInformationMetricContract(
            metric_name=name,
            evaluation_split=HELD_OUT_SPLIT,
            baselines=REQUIRED_PERSPECTIVE_BASELINES,
            conditional_value_definition=(
                "held-out score with the slot present minus held-out score with the slot removed, "
                "at matched capacity"
            ),
        )

    return PerspectivePluralityContract(
        declarations=(
            PerspectiveDeclaration(
                perspective_id="vision-linear-readout",
                information_source="vision-cache",
                computation="linear probe over the frozen vision cache",
                metric=metric("vision_conditional_value"),
            ),
            PerspectiveDeclaration(
                perspective_id="audio-waveform-readout",
                information_source="audio-waveform",
                computation="linear probe over declared waveform statistics",
                metric=metric("audio_conditional_value"),
            ),
            PerspectiveDeclaration(
                perspective_id="numerosity-counter",
                information_source="numerosity-count",
                computation="exact count over declared event-graph entities",
                metric=metric("numerosity_conditional_value"),
            ),
        )
    )


def _fixture_source_card(seed: int) -> SourceCard:
    cleared = RightsFieldDeclaration(status="cleared", authority="named steward statement, per item")
    return SourceCard(
        source_id=f"synthetic-fixture-{seed}",
        steward="local deterministic generator",
        fields=dict.fromkeys(RIGHTS_FIELDS, cleared),
        allowed_uses=("research-local",),
    )


def make_sensing_fixture(*, seed: int, make_scenario_fn: Any = None) -> SensingScaffoldFixture:
    """Build the complete deterministic sensing scaffold fixture for one seed.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    if make_scenario_fn is None:
        from ..environments.scenario_factory import make_scenario as make_scenario_fn
    bundle: ScenarioBundle = make_scenario_fn(seed=seed)
    return SensingScaffoldFixture(
        seed=seed,
        scenario=bundle,
        audio=_fixture_audio_contract(bundle),
        perspectives=_fixture_perspectives(),
        source_card=_fixture_source_card(seed),
        temporal=build_temporal_binding_contract(bundle),
        contradiction=build_contradiction_contract(bundle),
        causal=build_causal_binding_contract(bundle),
    )
