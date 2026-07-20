#!/usr/bin/env python

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
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "pos"
        _build(src, carry_object=True)
        report = run_acceptance_gate(str(src), FACTORS, min_per_cell=PER_CELL, start=0, end=end)
        assert all(report[f]["passed"] for f in FACTORS), f"positive fixture should pass: {report}"
        print(f"[PASS] positive fixture accepted: { {f: report[f]['score'] for f in FACTORS} }")

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

    print(
        "\nDR1 SMOKE PASS: the caption acceptance gate passes on carried factors and refuses on a "
        "non-recoverable factor. The spine's pre-encode gate is de-risked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
