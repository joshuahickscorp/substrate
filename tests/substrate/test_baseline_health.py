"""BASELINE HEALTH — does the untouched ancestor still verify?

Steer S001 §2 separates two questions that must not be conflated:

  BASELINE HEALTH        does ODYSSEY_READY_BASELINE_V1 still verify?   must stay green
  ASCENSION BUILD STATUS is the new implementation ready to freeze?     may stay unfrozen

This file answers only the first.  It exists because a one-key rename in a pinned
module silently invalidated machine-gate sealing across twelve tests, and the
failure surfaced as an unrelated-looking `Refused` deep inside a rehearsal test.
A direct check names the cause immediately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from substrate import odyssey_transition

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _implementation_paths() -> dict[str, Path]:
    """Exactly the resolution odyssey_authority._validate_frozen_build uses."""
    return {
        **odyssey_transition.implementation_inputs(ROOT),
        "odyssey_worker": ROOT / "src/substrate/odyssey_worker.py",
        "odyssey_authority": ROOT / "src/substrate/odyssey_authority.py",
    }


def test_the_ancestor_frozen_implementation_has_not_drifted() -> None:
    """Every frozen-build-pinned module still matches its sealed digest.

    A failure here means some pinned module was edited.  During Ascension
    engineering that may be intentional (S001 §2: drift before the final freeze is
    development) -- but it must be a decision, not a surprise, and it means the
    ancestral G01-G15 seals no longer regenerate.
    """
    frozen = json.loads(FROZEN.read_text())
    pins: dict[str, str] = frozen.get("implementation_sha256", {})
    assert pins, "frozen build declares no implementation_sha256 map"

    paths = _implementation_paths()
    drifted = []
    for name, expected in sorted(pins.items()):
        source = paths.get(name)
        if source is None or not source.is_file():
            drifted.append(f"{name}: MISSING")
            continue
        observed = _digest(source)
        if observed != expected:
            drifted.append(f"{name}: expected {expected[:12]} observed {observed[:12]}")

    assert not drifted, (
        "FROZEN_IMPLEMENTATION_DRIFT -- the ancestor no longer verifies and no machine "
        f"gate can seal against frozen build {frozen.get('sha256', '?')[:12]}:\n  "
        + "\n  ".join(drifted)
    )


def _drifted(pins: dict[str, str], digests: dict[str, str]) -> list[str]:
    """Pure comparison, extracted so the detector itself can be tested."""
    out = []
    for name, expected in sorted(pins.items()):
        observed = digests.get(name)
        if observed is None:
            out.append(f"{name}: MISSING")
        elif observed != expected:
            out.append(f"{name}: expected {expected[:12]} observed {observed[:12]}")
    return out


def test_the_drift_detector_itself_detects_drift() -> None:
    """A guard that cannot fail is not a guard."""
    pins = {"odyssey_arms": "a" * 64, "odyssey_worker": "b" * 64}
    assert _drifted(pins, {"odyssey_arms": "a" * 64, "odyssey_worker": "b" * 64}) == []
    assert _drifted(pins, {"odyssey_arms": "c" * 64, "odyssey_worker": "b" * 64}) == [
        "odyssey_arms: expected aaaaaaaaaaaa observed cccccccccccc"
    ]
    assert _drifted(pins, {"odyssey_arms": "a" * 64}) == ["odyssey_worker: MISSING"]


def test_every_pinned_module_resolves_to_a_real_file() -> None:
    """The pinned boundary is not confined to src/substrate.

    telegram_notifier resolves to tools/odyssey7d_telegram_notifier.py.  Any audit
    that scans only src/substrate leaves part of the boundary unclassified.
    """
    pins = json.loads(FROZEN.read_text()).get("implementation_sha256", {})
    paths = _implementation_paths()
    missing = sorted(n for n in pins if n not in paths or not paths[n].is_file())
    assert not missing, f"pinned modules that do not resolve: {missing}"
    outside = sorted(n for n in pins if "src/substrate" not in str(paths[n]))
    # Not a failure -- an assertion that we know about it.
    assert outside == ["telegram_notifier"], f"pinned modules outside src/substrate changed: {outside}"
