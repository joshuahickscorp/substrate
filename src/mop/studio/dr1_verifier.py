
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "mop-dr1-adversarial-verification/v1"


@dataclass(frozen=True)
class DR1VerifierConfig:
    cache_dir: Path | str
    require_a6: bool = True
    require_perspective: bool = True


def build_dr1_verification(config: DR1VerifierConfig | Path | str) -> dict[str, Any]:
    cfg = config if isinstance(config, DR1VerifierConfig) else DR1VerifierConfig(cache_dir=config)
    root = Path(cfg.cache_dir)
    checks: list[dict[str, Any]] = []
    merge = _load_json_check(root / "merge_manifest.json", "merge_manifest", checks)
    perspective = _load_json_check(root / "perspective_matrix_receipt.json", "perspective_receipt", checks)
    a6 = _load_json_check(root / "a6_residual_guard.json", "a6_residual_guard", checks)

    checks.extend(_merge_checks(root, merge))
    checks.extend(_leg_checks(root, merge))
    checks.extend(_perspective_checks(perspective, merge, required=cfg.require_perspective))
    checks.extend(_a6_checks(a6, required=cfg.require_a6))

    integrity_ok = all(c["ok"] for c in checks)
    a6_survives = _a6_survives(a6)
    positive_safe = bool(integrity_ok and a6_survives)
    problems = [f"{c['name']}: {c['detail']}" for c in checks if not c["ok"]]
    if integrity_ok and not a6_survives:
        problems.append("a6_residual_guard: decisive condition did not survive, positive claim refused")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "cache_dir": str(root),
        "independent": True,
        "adversarial": True,
        "integrity_ok": integrity_ok,
        "a6_survives": a6_survives,
        "passed": positive_safe,
        "all_ok": positive_safe,
        "checks": checks,
        "summary": {
            "total": len(checks),
            "passed": sum(1 for c in checks if c["ok"]),
            "failed": sum(1 for c in checks if not c["ok"]),
        },
        "problems": problems,
    }


def write_dr1_verification(report: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")


def _load_json_check(path: Path, name: str, checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not path.exists():
        checks.append(_check(name, False, f"missing {path}"))
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        checks.append(_check(name, False, f"invalid JSON: {e}"))
        return None
    checks.append(_check(name, True, str(path)))
    return data if isinstance(data, dict) else None


def _merge_checks(root: Path, merge: dict[str, Any] | None) -> list[dict[str, Any]]:
    if merge is None:
        return []
    total = _positive_int(merge.get("total_encoded"))
    backends = list(merge.get("backends") or [])
    return [
        _check("merge_contiguous", bool(merge.get("contiguous")), "leg ranges are contiguous"),
        _check("merge_total_encoded", total > 0, f"total_encoded={total}"),
        _check(
            "merge_real_backend",
            bool(backends) and "frozen_random" not in backends,
            f"backends={backends}",
        ),
        _check(
            "merge_clip_order",
            bool(merge.get("clip_order_persisted")),
            f"clip_order_persisted={merge.get('clip_order_persisted')}",
        ),
        _check("cache_dir_exists", root.exists(), str(root)),
    ]


def _leg_checks(root: Path, merge: dict[str, Any] | None) -> list[dict[str, Any]]:
    ranges = merge.get("legs") if isinstance(merge, dict) else []
    if not ranges:
        return [_check("legs_present", False, "merge_manifest has no leg ranges")]
    checks = [_check("legs_present", True, f"{len(ranges)} leg range(s)")]
    for raw_range in ranges:
        if not isinstance(raw_range, list | tuple) or len(raw_range) != 2:
            checks.append(_check("leg_range_shape", False, f"bad range {raw_range!r}"))
            continue
        start, end = int(raw_range[0]), int(raw_range[1])
        leg_dir = root / f"leg_{start}_{end}"
        cells = _read_json(leg_dir / "cells.json")
        if cells is None:
            checks.append(
                _check(f"leg_{start}_{end}_cells", False, f"missing or invalid {leg_dir / 'cells.json'}")
            )
            continue
        n_encoded = _positive_int(cells.get("n_encoded"))
        expected = max(0, end - start)
        checks.append(
            _check(
                f"leg_{start}_{end}_count",
                n_encoded == expected,
                f"n_encoded={n_encoded}, expected={expected}",
            )
        )
        acceptance = cells.get("acceptance_report")
        passed = _acceptance_passed(acceptance)
        checks.append(
            _check(
                f"leg_{start}_{end}_caption_gate",
                passed,
                "all caption factors passed" if passed else "missing or failed acceptance_report",
            )
        )
        hashes = cells.get("clip_hashes")
        hash_count = len(hashes) if isinstance(hashes, list) else "missing"
        checks.append(
            _check(
                f"leg_{start}_{end}_clip_hashes",
                isinstance(hashes, list) and len(hashes) == n_encoded,
                f"clip_hash_count={hash_count}, n_encoded={n_encoded}",
            )
        )
    return checks


def _perspective_checks(
    perspective: dict[str, Any] | None,
    merge: dict[str, Any] | None,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    if perspective is None:
        return [] if not required else [_check("perspective_required", False, "missing perspective receipt")]
    total = _positive_int((merge or {}).get("total_encoded"))
    tags = set(perspective.get("tags") or [])
    ok = bool(perspective.get("ok"))
    n_ref = _positive_int(perspective.get("n_referents"))
    factor_counts = perspective.get("factor_counts")
    return [
        _check("perspective_ok", ok, f"ok={ok}"),
        _check(
            "perspective_tags",
            {"vision_vjepa2", "caption_text"}.issubset(tags),
            f"tags={sorted(tags)}",
        ),
        _check("perspective_referents", total == 0 or n_ref == total, f"n_referents={n_ref}, total={total}"),
        _check(
            "perspective_factors",
            isinstance(factor_counts, dict) and bool(factor_counts),
            "factor counts present",
        ),
    ]


def _a6_checks(a6: dict[str, Any] | None, *, required: bool) -> list[dict[str, Any]]:
    if a6 is None:
        return [] if not required else [_check("a6_required", False, "missing A6 receipt")]
    conditions = a6.get("conditions")
    decisive = str(a6.get("decisive_condition") or "")
    verdict = str(a6.get("verdict") or "")
    has_decisive = isinstance(conditions, dict) and decisive in conditions
    decisive_obj = conditions.get(decisive) if isinstance(conditions, dict) else None
    has_survival_flag = isinstance(decisive_obj, dict) and isinstance(decisive_obj.get("survives"), bool)
    return [
        _check(
            "a6_guard_name", "a6_residual_alignment" in str(a6.get("guard", "")), str(a6.get("guard", ""))
        ),
        _check("a6_decisive_condition", has_decisive, f"decisive_condition={decisive}"),
        _check(
            "a6_decisive_result",
            has_survival_flag,
            f"survives={decisive_obj.get('survives') if isinstance(decisive_obj, dict) else None}",
        ),
        _check("a6_verdict_present", bool(verdict), verdict or "missing verdict"),
    ]


def _a6_survives(a6: dict[str, Any] | None) -> bool:
    if not isinstance(a6, dict):
        return False
    conditions = a6.get("conditions")
    decisive = str(a6.get("decisive_condition") or "")
    if not isinstance(conditions, dict) or decisive not in conditions:
        return False
    return bool((conditions.get(decisive) or {}).get("survives"))


def _acceptance_passed(acceptance: Any) -> bool:
    if not isinstance(acceptance, dict) or not acceptance:
        return False
    return all(isinstance(v, dict) and bool(v.get("passed")) for v in acceptance.values())


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _positive_int(value: Any) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return 0
    return out if out > 0 else 0


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}
