
from __future__ import annotations

import pytest

from mop.beds.starss23.schema import (
    COLLAR_FRAMES,
    FRAME_MS,
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
    Clip,
    ClipSplit,
    OnsetEvent,
    SchemaRefusal,
    clip_partition,
    room_disjoint_split,
)

_DIGEST = "a" * 64


def _clip(clip_id: str, room_id: str, onsets=(), n_frames: int = 60) -> Clip:
    return Clip(
        clip_id=clip_id,
        room_id=room_id,
        n_frames=n_frames,
        audio_sha256=_DIGEST,
        onsets=tuple(onsets),
    )


def test_frozen_acquisition_grid() -> None:
    assert SAMPLE_RATE_HZ == 24_000
    assert FRAME_MS == 100
    assert N_CHANNELS == 4
    assert SAMPLES_PER_FRAME == 2400  # 24 kHz times 100 ms
    assert COLLAR_FRAMES == 2  # plus or minus 200 ms at the 100 ms grid
    assert N_CLASSES == 13


def test_onset_event_validates_its_fields() -> None:
    ok = OnsetEvent(frame=3, class_id=2, azimuth=45.0, elevation=-10.0, distance=2.0)
    assert ok.frame == 3
    with pytest.raises(SchemaRefusal):
        OnsetEvent(frame=-1, class_id=0, azimuth=0.0, elevation=0.0, distance=1.0)
    with pytest.raises(SchemaRefusal):
        OnsetEvent(frame=0, class_id=N_CLASSES, azimuth=0.0, elevation=0.0, distance=1.0)
    with pytest.raises(SchemaRefusal):
        OnsetEvent(frame=0, class_id=0, azimuth=200.0, elevation=0.0, distance=1.0)
    with pytest.raises(SchemaRefusal):
        OnsetEvent(frame=0, class_id=0, azimuth=0.0, elevation=91.0, distance=1.0)
    with pytest.raises(SchemaRefusal):
        OnsetEvent(frame=0, class_id=0, azimuth=0.0, elevation=0.0, distance=0.0)


def test_clip_requires_sorted_unique_in_range_onsets() -> None:
    good = _clip("fold3_room0_mix000", "room0", onsets=[OnsetEvent(2, 0, 0.0, 0.0, 1.0), OnsetEvent(9, 1, 0.0, 0.0, 1.0)])
    assert good.onset_frames == (2, 9)
    assert len(good.digest()) == 64
    with pytest.raises(SchemaRefusal):
        _clip("fold3_room0_mix001", "room0", onsets=[OnsetEvent(60, 0, 0.0, 0.0, 1.0)], n_frames=60)
    with pytest.raises(SchemaRefusal):
        _clip("fold3_room0_mix002", "room0", onsets=[OnsetEvent(9, 0, 0.0, 0.0, 1.0), OnsetEvent(2, 0, 0.0, 0.0, 1.0)])
    with pytest.raises(SchemaRefusal):
        Clip(clip_id="fold3_room0_mix003", room_id="room0", n_frames=60, audio_sha256="xyz", onsets=())


def test_room_disjoint_split_partitions_by_sorted_room() -> None:
    clips = [_clip(f"fold3_room{r}_mix00{c}", f"room{r}") for r in range(5) for c in range(2)]
    split = room_disjoint_split(clips, n_train_rooms=2, n_val_rooms=1)
    assert split.rooms["train"] == ("room0", "room1")
    assert split.rooms["val"] == ("room2",)
    assert split.rooms["test"] == ("room3", "room4")
    train_rooms = set(split.rooms["train"])
    assert train_rooms.isdisjoint(split.rooms["val"])
    assert train_rooms.isdisjoint(split.rooms["test"])
    part = clip_partition(split)
    all_ids = [*part["train"], *part["val"], *part["test"]]
    assert len(all_ids) == len(set(all_ids)) == 10


def test_room_disjoint_split_refuses_too_few_rooms() -> None:
    clips = [_clip(f"fold3_room{r}_mix000", f"room{r}") for r in range(2)]
    with pytest.raises(SchemaRefusal):
        room_disjoint_split(clips, n_train_rooms=2, n_val_rooms=1)


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
