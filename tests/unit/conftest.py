
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def starss23_fast_config():

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

    from mop.beds.starss23.artifact import build_bed_artifact

    return build_bed_artifact(starss23_fast_config)
