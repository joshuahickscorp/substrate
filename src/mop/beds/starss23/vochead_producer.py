"""STARSS23 value-of-computation HEADROOM instrument, component 3: the real-data producer and seal.

Builds per-clip ``ClipTarget`` values for the concurrent-count target and the direction-of-arrival target
from the real STARSS23 subset (by importing the sealed count and DoA label and estimator modules, never
editing them), runs the preregistered budget sweep on the native fold-4 test scope and the full subset,
adds two self-contained synthetic control targets (one known-strong WHAT, one known-harmful WHAT), and
seals ``proof/STARSS23_VOC_HEADROOM.json``.

This is additive-only: it edits no sealed onset, counting, or DoA module and touches no live campaign path.
It writes only its three ``proof/STARSS23_VOC_HEADROOM*.json`` artifacts. It reads nothing under runs/.
``activation_allowed``, ``scientific_promotion``, and ``independent_scientific_confirmation`` are hardcoded
false. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_bytes, canonical_sha256

from . import CLAIM_SCOPE
from .adapter import RealStarssAdapter
from .count_estimator import FrozenCountEstimator
from .count_labels import build_count_clips
from .doa_estimator import FrozenDoaEstimator
from .doa_labels import build_doa_clips, to_arrays
from .doa_referee import DOA_COLD_START
from .vochead_analyzer import (
    METRIC_COUNT_ABS,
    METRIC_DOA_GREATCIRCLE,
    ClipTarget,
    analyze_target,
)
from .vochead_prereg import STAGE, VOCHEAD_INSTRUMENT_ID

VOCHEAD_ARTIFACT_SCHEMA = "mop-starss23-vochead-bed/v1"

DEFAULT_FOA_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/foa_subset/foa_dev")
DEFAULT_METADATA_ROOT = Path(
    "/Users/scammermike/Downloads/mop-data/starss23/metadata_dev_extracted/metadata_dev"
)


class VocHeadProducerRefusal(ValueError):
    """Raised when the instrument cannot build a real target under the additive contract."""


@dataclass(frozen=True, slots=True)
class RealVocHeadConfig:
    """Real-run configuration for the headroom instrument. The corpus is the fixed STARSS23 subset."""

    foa_root: Path = DEFAULT_FOA_ROOT
    metadata_root: Path = DEFAULT_METADATA_ROOT
    max_frames: int | None = None


# ---------------------------------------------------------------------------
# Building the per-clip targets from the real corpus.
# ---------------------------------------------------------------------------


def _count_targets(adapter: RealStarssAdapter, metadata_root: Path) -> dict[str, ClipTarget]:
    estimator = FrozenCountEstimator()
    clips = build_count_clips(adapter, metadata_root)
    out: dict[str, ClipTarget] = {}
    for clip_id, clip in clips.items():
        est = estimator.estimate_track(adapter.audio(clip_id))
        out[clip_id] = ClipTarget(
            clip_id=clip_id,
            room_id=clip.room_id,
            n_frames=clip.n_frames,
            metric_id=METRIC_COUNT_ABS,
            active_mask=tuple([True] * clip.n_frames),  # the count referee scores every frame, silence=0
            gt_values=tuple(int(v) for v in clip.count_track),
            est_values=tuple(int(v) for v in est.tolist()),
            change_frames=clip.change_frames,
            cold_start=0,
        )
    return out


def _doa_targets(adapter: RealStarssAdapter, metadata_root: Path) -> dict[str, ClipTarget]:
    estimator = FrozenDoaEstimator()
    clips = build_doa_clips(adapter, metadata_root)
    out: dict[str, ClipTarget] = {}
    for clip_id, clip in clips.items():
        active_mask, directions = to_arrays(clip)  # bool (n,), float (n, 2)
        if not bool(active_mask.any()):
            # No ground-truth direction anywhere in this clip's window (only possible under a frame cap;
            # the full real run has none): a clip with no active frame is unscoreable for DoA, so skip it.
            continue
        est = estimator.estimate_track(adapter.audio(clip_id))  # (n_frames, 2)
        gt_values = tuple((float(directions[t, 0]), float(directions[t, 1])) for t in range(clip.n_frames))
        est_values = tuple((float(est[t, 0]), float(est[t, 1])) for t in range(clip.n_frames))
        out[clip_id] = ClipTarget(
            clip_id=clip_id,
            room_id=clip.room_id,
            n_frames=clip.n_frames,
            metric_id=METRIC_DOA_GREATCIRCLE,
            active_mask=tuple(bool(flag) for flag in active_mask.tolist()),
            gt_values=gt_values,
            est_values=est_values,
            change_frames=clip.change_frames,
            cold_start=(float(DOA_COLD_START[0]), float(DOA_COLD_START[1])),
        )
    return out


def _synthetic_control_targets() -> dict[str, ClipTarget]:
    """Two self-contained count-metric targets: one strong-WHAT, one harmful-WHAT, for instrument validity."""

    gt = tuple(int(v) for v in ([0, 0, 1, 1, 1, 2, 2, 2, 0, 0] * 6))
    n = len(gt)
    changes = tuple(t for t in range(1, n) if gt[t] != gt[t - 1])
    strong = ClipTarget(
        clip_id="synthetic_strong_what",
        room_id="synthetic",
        n_frames=n,
        metric_id=METRIC_COUNT_ABS,
        active_mask=tuple([True] * n),
        gt_values=gt,
        est_values=gt,  # a perfect fresh estimator: refreshing at a change tracks truth exactly
        change_frames=changes,
        cold_start=0,
    )
    harmful = ClipTarget(
        clip_id="synthetic_harmful_what",
        room_id="synthetic",
        n_frames=n,
        metric_id=METRIC_COUNT_ABS,
        active_mask=tuple([True] * n),
        gt_values=gt,
        est_values=tuple([9] * n),  # a maximally wrong fresh estimator: refreshing is worse than coasting 0
        change_frames=changes,
        cold_start=0,
    )
    return {strong.clip_id: strong, harmful.clip_id: harmful}


# ---------------------------------------------------------------------------
# Assemble and seal.
# ---------------------------------------------------------------------------


def _scope_analysis(
    family: str, metric_id: str, targets: dict[str, ClipTarget], clip_ids: list[str]
) -> dict[str, Any]:
    subset = [targets[cid] for cid in clip_ids if cid in targets]
    if not subset:
        raise VocHeadProducerRefusal(f"scope for {family} has no clips")
    return analyze_target(family, metric_id, subset)


def build_vochead_artifact(config: RealVocHeadConfig | None = None) -> dict[str, Any]:
    """Build the sealed headroom artifact on the real STARSS23 subset plus the synthetic controls."""

    config = config or RealVocHeadConfig()
    adapter = RealStarssAdapter(config.foa_root, config.metadata_root, max_frames=config.max_frames)
    if adapter.source_kind() != "real" or not adapter.rights_clean():
        raise VocHeadProducerRefusal("the headroom instrument requires the real rights-clean corpus")

    dev = adapter.dev_split()
    all_ids = sorted(clip.clip_id for clip in adapter.clips())
    test_ids = sorted(dev.dev_test)

    count_targets = _count_targets(adapter, config.metadata_root)
    doa_targets = _doa_targets(adapter, config.metadata_root)
    control_targets = _synthetic_control_targets()

    corpus_targets = {
        "count": {cid: t.payload() for cid, t in count_targets.items()},
        "doa": {cid: t.payload() for cid, t in doa_targets.items()},
        "synthetic_control": {cid: t.payload() for cid, t in control_targets.items()},
    }

    analysis: dict[str, Any] = {
        "test_fold": {
            "clip_ids": test_ids,
            "count": _scope_analysis("count", METRIC_COUNT_ABS, count_targets, test_ids),
            "doa": _scope_analysis("doa", METRIC_DOA_GREATCIRCLE, doa_targets, test_ids),
        },
        "full_subset": {
            "clip_ids": all_ids,
            "count": _scope_analysis("count", METRIC_COUNT_ABS, count_targets, all_ids),
            "doa": _scope_analysis("doa", METRIC_DOA_GREATCIRCLE, doa_targets, all_ids),
        },
        "synthetic_control": {
            "clip_ids": sorted(control_targets),
            "strong_what": analyze_target(
                "synthetic_strong_what", METRIC_COUNT_ABS, [control_targets["synthetic_strong_what"]]
            ),
            "harmful_what": analyze_target(
                "synthetic_harmful_what", METRIC_COUNT_ABS, [control_targets["synthetic_harmful_what"]]
            ),
        },
    }

    control_ok = (
        analysis["synthetic_control"]["strong_what"]["interpretation"] == "real_headroom"
        and analysis["synthetic_control"]["harmful_what"]["interpretation"] == "what_floor_collapse"
    )

    content: dict[str, Any] = {
        "schema": VOCHEAD_ARTIFACT_SCHEMA,
        "stage": STAGE,
        "instrument_id": VOCHEAD_INSTRUMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "source_kind": adapter.source_kind(),
        "rights_clean": adapter.rights_clean(),
        "real_corpus": {
            "foa_root": str(config.foa_root),
            "metadata_root": str(config.metadata_root),
            "n_clips": len(all_ids),
            "n_test_clips": len(test_ids),
            "test_rooms": sorted({count_targets[cid].room_id for cid in test_ids}),
            "split_rule": "test = native fold-4 dev-test; full_subset = every fixed-subset clip",
        },
        "corpus_targets": corpus_targets,
        "analysis": analysis,
        "synthetic_control_ok": control_ok,
        "flags": {
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_scientific_confirmation": False,
            "is_instrument_not_bed": True,
        },
        "headline": _headline(analysis),
    }
    seal = canonical_sha256(content)
    return {**content, "seal": {"sha256": seal}}


def _headline(analysis: dict[str, Any]) -> dict[str, Any]:
    """A compact, honest summary of the per-scope interpretations. Reads only computed aggregates."""

    return {
        "test_fold_count_interpretation": analysis["test_fold"]["count"]["interpretation"],
        "test_fold_doa_interpretation": analysis["test_fold"]["doa"]["interpretation"],
        "full_subset_count_interpretation": analysis["full_subset"]["count"]["interpretation"],
        "full_subset_doa_interpretation": analysis["full_subset"]["doa"]["interpretation"],
        "test_fold_count_refreshable_range": analysis["test_fold"]["count"]["refreshable_range"],
        "test_fold_doa_refreshable_range": analysis["test_fold"]["doa"]["refreshable_range"],
    }


DEFAULT_VOCHEAD_ARTIFACT_PATH = Path("proof/STARSS23_VOC_HEADROOM.json")


def write_vochead_artifact(
    artifact: dict[str, Any], out_path: str | Path = DEFAULT_VOCHEAD_ARTIFACT_PATH
) -> Path:
    """Write the sealed artifact as canonical JSON bytes so its on-disk digest is stable."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(artifact))
    return path
