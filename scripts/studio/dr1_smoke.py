#!/usr/bin/env python
"""DR1 pre-encode SMOKE (laptop-safe, no encoder, no Studio RAM): build a tiny composable-factor
fixture and prove the caption-recoverability ACCEPTANCE GATE both PASSES on captions that carry every
factor AND REFUSES (a tie is a null) when a factor is not recoverable from the caption text. This
de-risks the spine's binding gate before any Studio encode is spent.

The gate path (validate_source -> assert_bound_and_stocked -> load_captions -> _clip_stems_in_leg ->
assert_caption_recoverable) does no video decode, so empty .mp4 files plus captions.json exercise it
fully. Run anywhere: PYTHONPATH=src:. .venv/bin/python scripts/studio/dr1_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.studio.dr1_curate_bound_video import run_acceptance_gate  # noqa: E402

FACTORS = ("object", "action")
OBJECTS = ("dog", "cat")
ACTIONS = ("running", "sitting")
PER_CELL = 10


def _build(source: Path, carry_object: bool) -> None:
    """Lay out <object>-<action>/<stem>.mp4 (empty) plus captions.json. If carry_object is False the
    caption replaces the object word with a constant 'animal' so the object factor is NOT recoverable
    from the caption text, which the gate must refuse."""
    import json

    caps: dict[str, str] = {}
    for obj in OBJECTS:
        for act in ACTIONS:
            cell = f"{obj}-{act}"
            (source / cell).mkdir(parents=True)
            for i in range(PER_CELL):
                stem = f"{cell}_clip{i}"
                (source / cell / f"{stem}.mp4").write_bytes(b"")  # empty: gate never decodes
                noun = obj if carry_object else "animal"
                caps[stem] = f"a {noun} is {act} in the yard"
    (source / "captions.json").write_text(json.dumps(caps, indent=2))


def main() -> int:
    end = len(OBJECTS) * len(ACTIONS) * PER_CELL
    # (1) POSITIVE: captions carry object and action -> gate must PASS.
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "pos"
        _build(src, carry_object=True)
        report = run_acceptance_gate(str(src), FACTORS, min_per_cell=PER_CELL, start=0, end=end)
        assert all(report[f]["passed"] for f in FACTORS), f"positive fixture should pass: {report}"
        print(f"[PASS] positive fixture accepted: "
              f"{ {f: report[f]['score'] for f in FACTORS} }")

    # (2) NEGATIVE: captions drop the object word -> object not recoverable -> gate must REFUSE.
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "neg"
        _build(src, carry_object=False)
        refused = False
        try:
            run_acceptance_gate(str(src), FACTORS, min_per_cell=PER_CELL, start=0, end=end)
        except SystemExit:
            refused = True
        assert refused, "negative fixture (object not caption-recoverable) should REFUSE the encode"
        print("[PASS] negative fixture refused (object not caption-recoverable, a tie is a null)")

    print("\nDR1 SMOKE PASS: the caption acceptance gate passes on carried factors and refuses on a "
          "non-recoverable factor. The spine's pre-encode gate is de-risked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
