"""Independent verification: recomputation, counterfeits, clone and regeneration.

Recomputation deliberately does not import the analysis that produced the
published numbers. It re-derives the decisive effect from the raw published
rows with its own arithmetic, so that agreement means two independent paths
reached the same answer rather than one path being called twice.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_io as io


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def recompute_decisive_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    """Re-derive the paired effect from raw rows without the analysis module.

    Plain arithmetic on purpose: mean per arm per history, paired difference,
    and a normal-approximation interval. It will not match the published
    bias-corrected interval exactly, and it is not supposed to — it exists to
    catch an effect that is wrong by much more than the two intervals differ.
    """
    per_cell: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        per_cell.setdefault((int(row["history_id"]), str(row["arm"])), []).append(float(row["score"]))
    per_history: dict[int, dict[str, float]] = {}
    for (history_id, arm), values in per_cell.items():
        per_history.setdefault(history_id, {})[arm] = _mean(values)

    differences: list[float] = []
    for history_id in sorted(per_history):
        cell = per_history[history_id]
        if candidate in cell and comparator in cell:
            differences.append(cell[candidate] - cell[comparator])
    if len(differences) < 2:
        raise io.Refused("recomputation needs at least two paired developmental histories")

    effect = _mean(differences)
    variance = sum((value - effect) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    half_width = 1.959963985 * standard_error
    return {
        "method": "independent_plain_arithmetic_paired_mean",
        "candidate": candidate,
        "comparator": comparator,
        "histories": len(differences),
        "effect": effect,
        "normal_lower": effect - half_width,
        "normal_upper": effect + half_width,
        "standard_error": standard_error,
        "activation": False,
    }


def agrees_with(published: Mapping[str, Any], recomputed: Mapping[str, Any], *, tolerance: float = 1e-9) -> dict[str, Any]:
    """The point estimate must match; the intervals need only agree in sign."""
    effect_matches = abs(float(published["effect"]) - float(recomputed["effect"])) <= tolerance
    sign_matches = (float(published["confidence_lower"]) > 0) == (float(recomputed["normal_lower"]) > 0)
    return {
        "published_effect": published["effect"],
        "recomputed_effect": recomputed["effect"],
        "effect_matches": effect_matches,
        "lower_bound_sign_matches": sign_matches,
        "published_lower": published["confidence_lower"],
        "recomputed_lower": recomputed["normal_lower"],
        "all_pass": effect_matches and sign_matches,
        "activation": False,
    }


# --------------------------------------------------------------------------
# Counterfeit systems
# --------------------------------------------------------------------------

COUNTERFEITS = {
    "record_store_null": "stores observed fields and copies them; produces no development",
    "random_growth_plastic": "grows structure without verified value; produces the growth signature without the value",
    "shuffled_history_plastic": "fully plastic but the order of experience destroyed",
    "wrong_history_plastic": "fully plastic on a history that is not its own",
    "static_frozen_field": "cannot write durable state at all",
}


def counterfeit_report(rows: Sequence[Mapping[str, Any]], *, selected: str) -> dict[str, Any]:
    """Every counterfeit must score no better than the selected candidate.

    A counterfeit that matches or beats the selected material means the measure
    is rewarding the counterfeit's shortcut rather than development, and the
    result cannot be used.
    """
    per_arm: dict[str, list[float]] = {}
    for row in rows:
        per_arm.setdefault(str(row["arm"]), []).append(float(row["score"]))
    selected_score = _mean(per_arm.get(selected, []))
    findings = {}
    for arm, description in COUNTERFEITS.items():
        if arm not in per_arm:
            findings[arm] = {"description": description, "ran": False, "rejected": False, "reason": "did not run"}
            continue
        score = _mean(per_arm[arm])
        findings[arm] = {
            "description": description,
            "ran": True,
            "score": score,
            "selected_score": selected_score,
            "rejected": score <= selected_score,
        }
    surviving = sorted(arm for arm, row in findings.items() if not row["rejected"])
    return {
        "selected": selected,
        "selected_score": selected_score,
        "findings": findings,
        "surviving_counterfeits": surviving,
        "all_pass": not surviving,
        "activation": False,
    }


# --------------------------------------------------------------------------
# Clean clone and deterministic regeneration
# --------------------------------------------------------------------------


def clean_clone(*, reference: str = "HEAD", keep: bool = False) -> dict[str, Any]:
    """Clone the repository into a fresh directory, install it, and run the suite."""
    root = Path(tempfile.mkdtemp(prefix="genesis-clean-clone-"))
    clone = root / "substrate"
    steps: list[dict[str, Any]] = []

    def step(name: str, command: Sequence[str], cwd: Path) -> bool:
        result = subprocess.run(list(command), cwd=cwd, capture_output=True, text=True, check=False)
        steps.append(
            {
                "step": name,
                "command": " ".join(command),
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
        return result.returncode == 0

    ok = step("clone", ["git", "clone", "--no-hardlinks", str(io.ROOT), str(clone)], root)
    if ok:
        ok = step("checkout", ["git", "checkout", "--detach", reference], clone)
    if ok:
        ok = step("install", ["python3", "-m", "pip", "install", "--quiet", ".[dev]"], clone)
    if ok:
        ok = step("tests", ["python3", "-m", "pytest", "tests/substrate", "-q", "-x"], clone)

    return {
        "clone_root": str(clone),
        "reference": reference,
        "steps": steps,
        "kept": keep,
        "all_pass": ok,
        "activation": False,
    }


def regeneration(build: Any) -> dict[str, Any]:
    """Run a deterministic build twice and require byte-identical digests."""
    first = build()
    second = build()
    first_digest = io.digest(first)
    second_digest = io.digest(second)
    return {
        "first_digest": first_digest,
        "second_digest": second_digest,
        "identical": first_digest == second_digest,
        "all_pass": first_digest == second_digest,
        "activation": False,
    }


def demo() -> None:
    rows = []
    for history in range(32):
        rows.append({"history_id": history, "arm": "K1", "score": 0.5 + ((history * 7) % 5 - 2) / 100.0})
        rows.append({"history_id": history, "arm": C.CANONICAL_S2_ID, "score": 0.5 + ((history * 7) % 5 - 2) / 100.0})
    recomputed = recompute_decisive_effect(rows, candidate="K1", comparator=C.CANONICAL_S2_ID)
    assert abs(recomputed["effect"]) < 1e-12, recomputed

    published = {"effect": 0.0, "confidence_lower": 0.0}
    assert agrees_with(published, recomputed)["all_pass"]

    # A published effect that does not match the raw rows must be caught.
    assert not agrees_with({"effect": 0.2, "confidence_lower": 0.1}, recomputed)["all_pass"]

    counterfeit_rows = [
        {"arm": "K1", "score": 0.3},
        {"arm": "record_store_null", "score": 0.1},
        {"arm": "random_growth_plastic", "score": 0.9},
    ]
    report = counterfeit_report(counterfeit_rows, selected="K1")
    assert "random_growth_plastic" in report["surviving_counterfeits"], report
    assert not report["all_pass"]

    counter = {"value": 0}

    def build() -> dict[str, Any]:
        counter["value"] += 1
        return {"stable": "yes"}

    assert regeneration(build)["identical"]
    print("genesis verification self-check passed")


if __name__ == "__main__":
    demo()
