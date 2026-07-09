"""DR1 source intake receipt.

The real-video DR1 path should fail before compute if the source tree is not a real, licensed,
caption-covered bound-attribute corpus. This module performs that filesystem-only check and emits a
receipt the Studio spine can treat as a launch gate. It also runs the cheap caption-side
recoverability probe, but still does not decode video or load encoders.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..diagnostics import linear_probe
from ..provenance import RESULT_TAGS
from ..substrate.video import list_class_files, validate_source

SCHEMA = "mop-dr1-source-intake/v1"
SOURCE_CARD_SCHEMA = "mop-dr1-source-card/v1"
SOURCE_CARD_VALIDATION_SCHEMA = "mop-dr1-source-card-validation/v1"
DEFAULT_FACTORS = ("object", "count", "relation", "action")
CAPTIONS_NAME = "captions.json"
ACCEPT_MARGIN = 0.10
ACCEPT_PROBE_SEED = 0
SOURCE_CARD_REQUIRED = ("source_id", "license", "allowed_use", "provenance_tag", "non_overlap_proof")
UNKNOWN_VALUES = {"", "unknown", "todo", "tbd", "none", "unverified"}


def build_dr1_source_intake(
    *,
    source: str | Path,
    factors: tuple[str, ...] = DEFAULT_FACTORS,
    min_per_cell: int = 16,
    source_card: dict[str, Any] | None = None,
    source_card_path: str | Path | None = None,
    require_source_card: bool = True,
    check_caption_recoverability: bool = True,
) -> dict[str, Any]:
    """Validate the DR1 source layout, captions, and source provenance before encode.

    The check deliberately avoids decoding video. It proves the corpus is structurally launchable, that
    captions carry the factors label-free, and that a human-visible source card exists before Studio
    compute can turn into claim evidence.
    """
    src = Path(source)
    factors_tuple = tuple(str(f) for f in factors if str(f))
    problems: list[str] = []
    manifest: dict[str, Any] | None = None
    cells: dict[str, Any] = {}
    captions_summary: dict[str, Any] = {
        "path": str(src / CAPTIONS_NAME),
        "exists": False,
        "total": 0,
        "covered": 0,
        "missing": [],
        "blank": [],
        "extra": [],
    }
    caption_recoverability_summary: dict[str, Any] = {
        "checked": False,
        "passed": False,
        "margin": ACCEPT_MARGIN,
        "report": {},
        "problems": [],
    }
    duplicate_stems: list[str] = []

    if not factors_tuple:
        problems.append("no composable factors supplied")
    if int(min_per_cell) <= 0:
        problems.append("min_per_cell must be positive")

    try:
        manifest = validate_source(src)
    except Exception as e:  # noqa: BLE001
        problems.append(f"source layout invalid: {e}")

    clip_stems: list[str] = []
    if manifest is not None and factors_tuple:
        cells, cell_problems = _cell_summary(manifest, factors_tuple, int(min_per_cell))
        problems.extend(cell_problems)
        try:
            _classes, files = list_class_files(src)
        except Exception as e:  # noqa: BLE001
            problems.append(f"could not list source clips: {e}")
            files = []
        clip_stems = [path.stem for path, _label in files]
        duplicate_stems = sorted(stem for stem, count in Counter(clip_stems).items() if count > 1)
        if duplicate_stems:
            problems.append("duplicate clip stems make captions ambiguous: " + ", ".join(duplicate_stems[:8]))
        captions, caption_problem = _load_captions(src / CAPTIONS_NAME)
        if caption_problem:
            problems.append(caption_problem)
        captions_summary = _caption_summary(src / CAPTIONS_NAME, captions, clip_stems)
        if captions_summary["missing"]:
            problems.append(
                f"{len(captions_summary['missing'])} clip(s) lack captions, first="
                f"{captions_summary['missing'][:8]}"
            )
        if captions_summary["blank"]:
            problems.append(
                f"{len(captions_summary['blank'])} caption(s) are blank, first="
                f"{captions_summary['blank'][:8]}"
            )
        if (
            check_caption_recoverability
            and not duplicate_stems
            and not captions_summary["missing"]
            and not captions_summary["blank"]
            and captions
            and files
        ):
            clip_cells = [path.parent.name for path, _label in files]
            clip_captions = [captions[path.stem] for path, _label in files]
            caption_recoverability_summary = _caption_recoverability_summary(
                clip_captions,
                clip_cells,
                factors_tuple,
            )
            if not caption_recoverability_summary["passed"]:
                problems.extend(str(p) for p in caption_recoverability_summary["problems"])

    card_summary, card_problems = _source_card_summary(
        source_card,
        source_card_path=source_card_path,
        require_source_card=require_source_card,
        expected_clip_count=manifest.get("n_clips") if manifest else None,
    )
    problems.extend(card_problems)

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(src),
        "factors": list(factors_tuple),
        "min_per_cell": int(min_per_cell),
        "manifest": manifest,
        "cells": cells,
        "captions": captions_summary,
        "caption_recoverability": caption_recoverability_summary,
        "duplicate_clip_stems": duplicate_stems,
        "source_card": card_summary,
        "problems": problems,
    }
    receipt["all_ok"] = not problems
    return receipt


def load_source_card(path: str | Path | None) -> dict[str, Any] | None:
    """Load a source-card JSON object if a path is supplied and exists."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"source card {p} must be a JSON object")
    return data


def build_dr1_source_card(
    *,
    source_id: str,
    license_name: str,
    allowed_use: str,
    non_overlap_proof: str | dict[str, Any],
    provenance_tag: str = "natural-video",
    clip_count: int | None = None,
    requires_manual_license: bool = False,
    accepted_terms: bool = False,
    license_url: str | None = None,
    source_url: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build the DR1 source-card JSON object the Studio intake expects."""
    card: dict[str, Any] = {
        "schema": SOURCE_CARD_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "source_id": source_id,
        "license": license_name,
        "allowed_use": allowed_use,
        "provenance_tag": provenance_tag,
        "non_overlap_proof": non_overlap_proof,
        "requires_manual_license": bool(requires_manual_license),
        "accepted_terms": bool(accepted_terms),
    }
    if clip_count is not None:
        card["clip_count"] = int(clip_count)
    if license_url:
        card["license_url"] = license_url
    if source_url:
        card["source_url"] = source_url
    if notes:
        card["notes"] = notes
    return card


def validate_dr1_source_card(
    card: dict[str, Any] | None,
    *,
    source_card_path: str | Path | None = None,
    expected_clip_count: int | None = None,
    require_source_card: bool = True,
) -> dict[str, Any]:
    """Return a durable validation receipt for a DR1 source card."""
    summary, problems = _source_card_summary(
        card,
        source_card_path=source_card_path,
        require_source_card=require_source_card,
        expected_clip_count=expected_clip_count,
    )
    card_schema = card.get("schema") if isinstance(card, dict) else None
    if card_schema is not None and card_schema != SOURCE_CARD_SCHEMA:
        problems.append(f"source card schema {card_schema!r} != {SOURCE_CARD_SCHEMA!r}")
    if isinstance(card, dict):
        summary["schema"] = card_schema
    return {
        "schema": SOURCE_CARD_VALIDATION_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "source_card": summary,
        "expected_clip_count": expected_clip_count,
        "problems": problems,
        "all_ok": not problems,
    }


def write_dr1_source_card(card: dict[str, Any], path: str | Path) -> None:
    """Write a DR1 source-card JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, default=str) + "\n")


def write_dr1_source_card_validation(receipt: dict[str, Any], path: str | Path) -> None:
    """Write a DR1 source-card validation receipt."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")


def write_dr1_source_intake(receipt: dict[str, Any], path: str | Path) -> None:
    """Write a source-intake receipt."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")


def _cell_summary(
    manifest: dict[str, Any], factors: tuple[str, ...], min_per_cell: int
) -> tuple[dict[str, Any], list[str]]:
    cells: dict[str, Any] = {}
    values: dict[str, set[str]] = {factor: set() for factor in factors}
    thin: list[tuple[str, int]] = []
    problems: list[str] = []
    for cell, raw_count in sorted(dict(manifest.get("per_class", {})).items()):
        count = int(raw_count)
        try:
            parsed = _parse_cell(cell, factors)
        except ValueError as e:
            problems.append(str(e))
            continue
        cells[cell] = {**parsed, "count": count}
        for factor in factors:
            values[factor].add(parsed[factor])
        if count < min_per_cell:
            thin.append((cell, count))
    degenerate = [factor for factor, vals in values.items() if len(vals) < 2]
    if degenerate:
        problems.append(f"composable factors with fewer than 2 values: {degenerate}")
    if thin:
        problems.append(f"cells below min_per_cell={min_per_cell}: {thin[:12]}")
    return {
        "total": len(cells),
        "factor_value_counts": {factor: len(vals) for factor, vals in values.items()},
        "per_cell": cells,
    }, problems


def _parse_cell(cell: str, factors: tuple[str, ...]) -> dict[str, str]:
    parts = cell.split("-")
    if len(parts) != len(factors) or not all(parts):
        raise ValueError(f"cell folder {cell!r} does not name all {len(factors)} factors joined by '-'")
    return dict(zip(factors, parts, strict=True))


def _load_captions(path: Path) -> tuple[dict[str, str], str | None]:
    if not path.exists():
        return {}, f"missing paired-caption sidecar {path}"
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return {}, f"invalid captions JSON {path}: {e}"
    if not isinstance(data, dict):
        return {}, f"{path} must be a JSON object mapping clip_stem to caption"
    return {str(k): str(v) for k, v in data.items()}, None


def _caption_recoverability_summary(
    captions: list[str],
    cells: list[str],
    factors: tuple[str, ...],
) -> dict[str, Any]:
    try:
        report = _assert_caption_recoverable(captions, cells, factors)
    except Exception as e:  # noqa: BLE001
        return {
            "checked": True,
            "passed": False,
            "margin": ACCEPT_MARGIN,
            "report": {},
            "problems": [f"caption recoverability failed: {e}"],
        }
    failed = [factor for factor, rec in report.items() if not rec["passed"]]
    problems = (
        [
            "caption recoverability below chance+"
            f"{ACCEPT_MARGIN} for factor(s) {failed}; fix captions or preserve this as a DR1 source null"
        ]
        if failed
        else []
    )
    return {
        "checked": True,
        "passed": not failed,
        "margin": ACCEPT_MARGIN,
        "report": report,
        "problems": problems,
    }


def _assert_caption_recoverable(
    captions: list[str],
    cells: list[str],
    factors: tuple[str, ...],
) -> dict[str, Any]:
    if len(captions) != len(cells):
        raise ValueError(f"captions ({len(captions)}) and cells ({len(cells)}) must be 1:1 per clip")
    parsed = [_parse_cell(cell, factors) for cell in cells]
    report: dict[str, Any] = {}
    for factor in factors:
        vals = sorted({p[factor] for p in parsed})
        idx = {value: i for i, value in enumerate(vals)}
        report[factor] = _caption_recoverability(captions, [idx[p[factor]] for p in parsed])
    return report


def _caption_recoverability(
    captions: list[str],
    labels: list[int],
    seed: int = ACCEPT_PROBE_SEED,
) -> dict[str, Any]:
    import torch

    x = _caption_features(captions)
    y = torch.tensor(labels, dtype=torch.long)
    out = linear_probe(x, y, classification=True, seed=seed)
    margin = float(out["score"]) - float(out["chance"])
    return {
        "score": round(float(out["score"]), 4),
        "chance": round(float(out["chance"]), 4),
        "margin": round(margin, 4),
        "passed": bool(margin >= ACCEPT_MARGIN),
    }


def _caption_features(captions: list[str], dim: int = 256):
    import torch

    feats = torch.zeros(len(captions), dim)
    for i, cap in enumerate(captions):
        s = cap.lower()
        for j in range(len(s) - 2):
            tri = s[j : j + 3]
            feats[i, _stable_hash(tri) % dim] += 1.0
    norms = feats.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return feats / norms


def _stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _caption_summary(path: Path, captions: dict[str, str], clip_stems: list[str]) -> dict[str, Any]:
    wanted = set(clip_stems)
    missing = sorted(stem for stem in wanted if stem not in captions)
    blank = sorted(stem for stem in wanted if not captions.get(stem, "").strip())
    extra = sorted(stem for stem in captions if stem not in wanted)
    return {
        "path": str(path),
        "exists": path.exists(),
        "total": len(captions),
        "covered": len(wanted) - len(missing),
        "missing": missing,
        "blank": blank,
        "extra": extra,
    }


def _source_card_summary(
    card: dict[str, Any] | None,
    *,
    source_card_path: str | Path | None,
    require_source_card: bool,
    expected_clip_count: int | None,
) -> tuple[dict[str, Any], list[str]]:
    path = None if source_card_path is None else str(source_card_path)
    if card is None:
        if require_source_card:
            return {"path": path, "exists": False}, ["missing DR1 source card with license/provenance proof"]
        return {"path": path, "exists": False}, []

    problems: list[str] = []
    missing = [field for field in SOURCE_CARD_REQUIRED if field not in card]
    if missing:
        problems.append(f"source card missing required field(s): {missing}")
    for field in ("source_id", "license", "allowed_use"):
        if _unknown(card.get(field)):
            problems.append(f"source card field {field!r} is empty or unknown")
    tag = str(card.get("provenance_tag", ""))
    if tag not in RESULT_TAGS:
        problems.append(f"source card provenance_tag {tag!r} not in {RESULT_TAGS}")
    elif tag != "natural-video":
        problems.append("DR1 source card must carry provenance_tag 'natural-video'")
    if not _proof_ok(card.get("non_overlap_proof")):
        problems.append("source card non_overlap_proof is missing or not affirmative")
    if bool(card.get("requires_manual_license")) and not bool(card.get("accepted_terms")):
        problems.append("source card requires manual license but accepted_terms is not true")
    if expected_clip_count is not None and card.get("clip_count") is not None:
        try:
            if int(card["clip_count"]) != int(expected_clip_count):
                problems.append(
                    f"source card clip_count {card['clip_count']} != source manifest {expected_clip_count}"
                )
        except Exception:  # noqa: BLE001
            problems.append(f"source card clip_count is not an integer: {card.get('clip_count')!r}")

    summary = {
        "path": path,
        "exists": True,
        "source_id": card.get("source_id"),
        "license": card.get("license"),
        "allowed_use": card.get("allowed_use"),
        "provenance_tag": card.get("provenance_tag"),
        "requires_manual_license": bool(card.get("requires_manual_license")),
        "accepted_terms": bool(card.get("accepted_terms")),
        "clip_count": card.get("clip_count"),
        "non_overlap_proof": card.get("non_overlap_proof"),
    }
    return summary, problems


def _unknown(value: Any) -> bool:
    return str(value or "").strip().lower() in UNKNOWN_VALUES


def _proof_ok(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return not _unknown(value)
    if isinstance(value, dict):
        if value.get("ok") is False:
            return False
        status = str(value.get("status") or value.get("verdict") or "").lower()
        if status in {"passed", "pass", "complete", "ok", "clear"}:
            return True
        return bool(value.get("ok"))
    return False
