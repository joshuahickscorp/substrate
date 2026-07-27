from __future__ import annotations

import pytest

from mop.beds.starss23.schema import (
    FRAME_MS,
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
    Clip,
    ClipSplit,
    SchemaRefusal,
)

_DIGEST = "a" * 64


def _clip(clip_id: str, room_id: str, n_frames: int = 60) -> Clip:
    return Clip(clip_id, room_id, n_frames, _DIGEST)


def test_frozen_acquisition_grid_and_clip_contract() -> None:
    assert (SAMPLE_RATE_HZ, FRAME_MS, N_CHANNELS, SAMPLES_PER_FRAME, N_CLASSES) == (
        24_000,
        100,
        4,
        2400,
        13,
    )
    assert _clip("fold3_room0_mix000", "room0").n_frames == 60
    for values in (
        ("", "room", 1, _DIGEST),
        ("clip", "", 1, _DIGEST),
        ("clip", "room", 0, _DIGEST),
        ("clip", "room", 1, "xyz"),
    ):
        with pytest.raises(SchemaRefusal):
            Clip(*values)


def test_clip_split_refuses_room_or_clip_overlap() -> None:
    a = _clip("fold3_room0_mix000", "room0")
    b = _clip("fold3_room1_mix000", "room1")
    c = _clip("fold3_room2_mix000", "room2")
    shared_room = _clip("fold3_room0_mix001", "room0")
    with pytest.raises(SchemaRefusal):
        ClipSplit(train=(a,), val=(shared_room,), test=(c,))
    with pytest.raises(SchemaRefusal):
        ClipSplit(train=(a,), val=(b,), test=(a,))
    ClipSplit(train=(a,), val=(b,), test=(c,))
