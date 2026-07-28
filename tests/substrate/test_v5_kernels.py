from __future__ import annotations

import copy

import pytest

from substrate import v5kernels


def test_v5_all_bounded_kernel_candidates_execute_shared_fixture() -> None:
    results = [
        v5kernels.evaluate(candidate, iterations=2)
        for candidate in v5kernels.CANDIDATES
    ]
    assert len(results) == 5
    assert all(row["checks"]["identity_persistence"] for row in results)
    assert all(row["checks"]["model_replacement"] for row in results)
    assert all(row["checks"]["checkpoint_restore"] for row in results)
    assert all(row["activation"] is False for row in results)


def test_v5_hybrid_wins_integrated_kernel_comparison() -> None:
    report = v5kernels.benchmark(iterations=8)
    assert report["selected"] == "candidate_d_hybrid_explicit_latent"
    selected = next(
        row for row in report["candidates"] if row["candidate"] == report["selected"]
    )
    assert selected["checks"]["explicit_latent_sync"]
    assert selected["checks"]["explicit_provenance"]
    assert selected["checks"]["object_permanence"]


def test_v5_kernel_refuses_reordered_events_and_corrupt_checkpoint() -> None:
    kernel = v5kernels.HybridKernel()
    kernel.apply(v5kernels.KernelEvent(1, "image", "observation", "object", True))
    with pytest.raises(v5kernels.Refused):
        kernel.apply(
            v5kernels.KernelEvent(1, "image", "observation", "object", False)
        )
    checkpoint = kernel.checkpoint()
    corrupted = copy.deepcopy(checkpoint)
    corrupted["body"]["identity"] = "different"
    with pytest.raises(v5kernels.Refused):
        v5kernels.HybridKernel().restore(corrupted)
