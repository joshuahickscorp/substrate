"""Shared fixtures for the STARSS23 ESCS bed tests.

The producer runs the whole bed (fixtures, featurizer, gate training, three controls, referee, harness,
stats) which is not free, so it is built exactly once per session at a small but still-robust
configuration and shared across the artifact and end-to-end test modules.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def starss23_fast_config():
    """A small, fast, deterministic bed configuration that still clears every bar robustly."""

    from mop.beds.starss23.artifact import BedConfig

    return BedConfig(
        clip_seconds=6.0,
        clips_per_room=3,
        onsets_per_clip=5,
        nuisance_per_clip=6,
        target_rates=(0.10, 0.08),
        noisy_tv_frames=300,
    )


@pytest.fixture(scope="session")
def starss23_bed_artifact(starss23_fast_config):
    """The assembled sealed bed artifact, built once and shared across the heavy test modules."""

    from mop.beds.starss23.artifact import build_bed_artifact

    return build_bed_artifact(starss23_fast_config)
