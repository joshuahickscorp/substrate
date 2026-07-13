#!/usr/bin/env python3
"""Independently recompute and adversarially verify the completed P5 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mop.config import REPO_ROOT
from mop.studies import p5_context_verify as verifier
from mop.studies.p5_context_verify import (
    DEFAULT_CHALLENGE,
    DEFAULT_CONFIG,
    DEFAULT_PRIMARY,
    DEFAULT_PRIMARY_RUN_DIR,
    DEFAULT_VERIFICATION,
    P5VerificationRefused,
    write_verification,
)


@contextmanager
def _correct_singleton_contrast_classification() -> Iterator[None]:
    """Restore the producer's preregistered n<2 ``undetermined`` rule during verification.

    The sealed producer classifies every one-observation contrast as ``undetermined``.  The
    independent verifier recomputes the same interval but historically called ``classify_ci``
    without carrying the sample count, which incorrectly promoted singleton fresh-seed rows.
    Keeping this correction in the verifier entry point preserves the already-sealed challenge
    implementation while the verifier artifact binds these exact wrapper bytes.
    """

    original_paired_ci = verifier._paired_ci
    original_classify_ci = verifier.classify_ci
    last_count: int | None = None

    def paired_ci(values: Sequence[float]) -> dict[str, Any]:
        nonlocal last_count
        result = original_paired_ci(values)
        last_count = int(result["n"])
        return result

    def classify_ci(lo: float, hi: float, sesoi: float) -> str:
        if last_count is not None and last_count < 2:
            return "undetermined"
        return original_classify_ci(lo, hi, sesoi)

    with (
        patch.object(verifier, "_paired_ci", paired_ci),
        patch.object(verifier, "classify_ci", classify_ci),
    ):
        yield


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--primary-run-dir", type=Path, default=DEFAULT_PRIMARY_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fresh-challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        with _correct_singleton_contrast_classification():
            receipt = write_verification(
                args.out,
                args.primary,
                args.primary_run_dir,
                args.config,
                args.fresh_challenge,
                repo_root=REPO_ROOT,
            )
    except P5VerificationRefused as exc:
        print(f"P5 verification refused: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "primary_profile": receipt["primary_profile"],
                "verification_complete": receipt["verification_complete"],
                "classification": receipt["classification"],
                "prerequisite_ready": receipt["prerequisite_ready"],
                "all_ok": receipt["all_ok"],
                "payload_sha256": receipt["payload_sha256"],
                "output": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["all_ok"] is True and receipt["verification_complete"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
