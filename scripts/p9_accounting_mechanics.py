#!/usr/bin/env python
"""P9 accounting mechanics receipt over a real bounded workload.

Wraps a genuine small workload (deterministic clip generation, hashing, a torch compute
block, and a checkpoint write/cleanup) in the P9 WorkloadAccountant and writes
proof/P9_ACCOUNTING_MECHANICS.json. Claim scope: accounting mechanics only. Energy is not
reported; the receipt says so itself.

No em or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.cache_factorized_encoder as cfe  # noqa: E402

from mop.studies.p9_accounting import WorkloadAccountant  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "proof" / "P9_ACCOUNTING_MECHANICS.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    with tempfile.TemporaryDirectory(prefix="p9-mechanics-") as scratch_dir:
        scratch = Path(scratch_dir)
        accountant = WorkloadAccountant(
            workload="p9-mechanics: clip generation, hashing, torch compute, checkpoint",
            watch_paths={"scratch": scratch},
        )

        cfe.RES, cfe.FRAMES = 256, 64
        generator = torch.Generator().manual_seed(0)
        clips: list[torch.Tensor] = []
        with accountant.phase("decode"):
            cells = [(a, b) for a in range(2) for b in range(2)]
            for index in range(8):
                a, b = cells[index % len(cells)]
                clips.append(cfe.make_factorized_clip(a, b, 2, 2, generator))

        digests: list[str] = []
        with accountant.phase("input"):
            for clip in clips:
                digests.append(cfe._tensor_sha256(clip))

        with accountant.phase("model"):
            weight = torch.randn(512, 512, generator=torch.Generator().manual_seed(1))
            activation = torch.randn(512, 512, generator=torch.Generator().manual_seed(2))
            for _ in range(50):
                activation = torch.tanh(activation @ weight)
            checksum = float(activation.abs().sum())

        with accountant.phase("checkpoint"):
            blob = scratch / "checkpoint.pt"
            torch.save({"activation": activation}, blob)
            checkpoint_bytes = blob.stat().st_size

        receipt = accountant.receipt()

    receipt["workload_evidence"] = {
        "clips": len(clips),
        "stimulus_set_sha256": hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
        "model_checksum": checksum,
        "checkpoint_bytes": checkpoint_bytes,
    }
    receipt["claim_boundary"] = {
        "scientific_promotion": False,
        "statement": "accounting mechanics over a bounded real workload; no capability or energy claim",
    }
    out = Path(args.out)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = {
        "phases": [p["name"] for p in receipt["phases"]],
        "checkpoint_storage_delta_positive": receipt["phases"][3]["storage_delta_bytes"]["scratch"] > 0,
        "idle_nonnegative": receipt["totals"]["idle_seconds"] >= 0,
        "energy_measured": receipt["energy"]["measured"],
    }
    print(json.dumps({"out": str(out), **checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
