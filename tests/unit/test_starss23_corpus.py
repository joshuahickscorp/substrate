
from __future__ import annotations

import numpy as np
import pytest

from mop.beds.starss23 import adapter as A
from mop.beds.starss23 import synthetic_corpus as SC
from mop.beds.starss23.schema import N_CHANNELS, SAMPLES_PER_FRAME


def _tiny() -> SC.SyntheticStarssCorpus:
    return SC.generate_corpus(SC.SyntheticCorpusConfig.tiny(), seed=0)




def test_same_seed_gives_byte_identical_audio_and_metadata() -> None:
    config = SC.SyntheticCorpusConfig.tiny()
    first = SC.generate_corpus(config, seed=11)
    second = SC.generate_corpus(config, seed=11)
    assert first.clip_ids == second.clip_ids
    for clip_id in first.clip_ids:
        assert first.audio(clip_id).tobytes() == second.audio(clip_id).tobytes()
        assert first.metadata_text(clip_id) == second.metadata_text(clip_id)
        assert first.planted_by_clip[clip_id] == second.planted_by_clip[clip_id]


def test_different_seed_changes_the_audio() -> None:
    config = SC.SyntheticCorpusConfig.tiny()
    base = SC.generate_corpus(config, seed=11)
    other = SC.generate_corpus(config, seed=12)
    assert any(base.audio(cid).tobytes() != other.audio(cid).tobytes() for cid in base.clip_ids)


def test_audio_is_contiguous_float32_and_a_whole_number_of_frames() -> None:
    corpus = _tiny()
    n_frames = corpus.config.clip_frames
    for clip_id in corpus.clip_ids:
        audio = corpus.audio(clip_id)
        assert audio.dtype == np.float32
        assert audio.shape == (N_CHANNELS, n_frames * SAMPLES_PER_FRAME)
        assert audio.shape[1] % SAMPLES_PER_FRAME == 0




def test_default_split_is_clip_and_room_disjoint_with_no_leakage() -> None:
    corpus = SC.generate_corpus(seed=0)
    split = corpus.default_split()
    partitions = {"train": split.train, "val": split.val, "test": split.test}

    clip_sets = {name: {clip.clip_id for clip in clips} for name, clips in partitions.items()}
    room_sets = {name: {clip.room_id for clip in clips} for name, clips in partitions.items()}
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert clip_sets[left].isdisjoint(clip_sets[right]), f"clip leak {left} vs {right}"
            assert room_sets[left].isdisjoint(room_sets[right]), f"room leak {left} vs {right}"

    all_split_clips = [clip.clip_id for clips in partitions.values() for clip in clips]
    assert sorted(all_split_clips) == sorted(corpus.clip_ids)
    assert len(all_split_clips) == len(set(all_split_clips))


def test_score_partition_is_exactly_the_held_out_dev_test() -> None:
    corpus = SC.generate_corpus(seed=0)
    split = corpus.default_split()
    dev = corpus.adapter().dev_split()
    assert {clip.clip_id for clip in split.test} == set(dev.dev_test)


def test_onset_frames_within_a_clip_are_distinct_and_in_range() -> None:
    corpus = _tiny()
    for clip in corpus.adapter().clips():
        frames = [onset.frame for onset in clip.onsets]
        assert len(frames) == len(set(frames))  # schema forbids two onsets on one frame
        assert all(0 <= frame < clip.n_frames for frame in frames)




def test_planted_labels_round_trip_through_metadata_and_adapter() -> None:
    corpus = _tiny()
    adapter = corpus.adapter()
    for clip_id in corpus.clip_ids:
        planted = corpus.planted_by_clip[clip_id]
        from_metadata = A.onset_events_from_rows(
            A.parse_starss23_metadata(corpus.metadata_text(clip_id))
        )
        assert from_metadata == planted
        assert adapter.clip(clip_id).onsets == planted


def test_metadata_text_is_native_six_column_starss23() -> None:
    corpus = _tiny()
    clip_id = corpus.clip_ids[0]
    lines = [line for line in corpus.metadata_text(clip_id).splitlines() if line]
    assert lines, "expected at least one onset row"
    for line in lines:
        fields = line.split(",")
        assert len(fields) == 6
        assert all(field.lstrip("-").isdigit() for field in fields)




def test_default_corpus_clears_the_validation_and_test_positive_floors() -> None:
    corpus = SC.generate_corpus(seed=0)
    counts = SC.SyntheticStarssCorpus.positive_counts(corpus.default_split())
    assert counts["val"] >= SC.MIN_VAL_ONSETS
    assert counts["test"] >= SC.MIN_TEST_ONSETS




def test_planted_onsets_carry_real_energy_a_spatial_oracle_can_score() -> None:
    corpus = SC.generate_corpus(seed=0)
    onset_energy: list[float] = []
    background_energy: list[float] = []
    for clip_id in corpus.clip_ids:
        energy = SC.frame_energy(corpus.audio(clip_id), channel=0)
        onset_frames = {onset.frame for onset in corpus.planted_by_clip[clip_id]}
        nuisance_frames = set(corpus.nuisance_frames_by_clip[clip_id])
        for frame in range(len(energy)):
            if frame in onset_frames:
                onset_energy.append(float(energy[frame]))
            elif frame not in nuisance_frames:
                background_energy.append(float(energy[frame]))
    assert np.median(onset_energy) > np.median(background_energy)


def test_bare_energy_threshold_cannot_isolate_onsets_because_nuisance_intrudes() -> None:
    corpus = SC.generate_corpus(seed=0)
    admitted_nuisance = 0
    for clip_id in corpus.clip_ids:
        energy = SC.frame_energy(corpus.audio(clip_id), channel=0)
        onset_frames = {onset.frame for onset in corpus.planted_by_clip[clip_id]}
        nuisance_frames = set(corpus.nuisance_frames_by_clip[clip_id])
        recall_one_threshold = min(energy[frame] for frame in onset_frames)
        admitted_nuisance += sum(
            1 for frame in nuisance_frames if energy[frame] >= recall_one_threshold
        )
    assert admitted_nuisance > 0




def test_write_to_and_from_dir_preserve_audio_sha256(tmp_path) -> None:
    corpus = _tiny()
    corpus.write_to(tmp_path)
    disk_adapter = A.SyntheticStarssAdapter.from_dir(tmp_path)
    memory = {clip.clip_id: clip for clip in corpus.adapter().clips()}
    for clip in disk_adapter.clips():
        assert clip.audio_sha256 == memory[clip.clip_id].audio_sha256
        assert clip.onsets == memory[clip.clip_id].onsets




@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_fold3_rooms": 2, "n_val_rooms": 2},  # no training room left
        {"active_frames": 40, "clip_frames": 10},  # active longer than the clip
        {"onsets_per_clip": 30, "nuisances_per_clip": 30, "clip_frames": 12},  # cannot fit events
        {"distance_min_cm": 500, "distance_max_cm": 100},  # inverted distance range
    ],
)
def test_config_refuses_inconsistent_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(SC.CorpusRefusal):
        SC.SyntheticCorpusConfig(**kwargs)


def test_generate_corpus_refuses_a_negative_seed() -> None:
    with pytest.raises(SC.CorpusRefusal):
        SC.generate_corpus(SC.SyntheticCorpusConfig.tiny(), seed=-1)


def test_real_scale_config_is_valid_and_mirrors_the_dev_shape() -> None:
    config = SC.SyntheticCorpusConfig.real_scale()
    assert config.clip_frames == 600  # 60 s clips
    assert config.n_train_rooms == config.n_fold3_rooms - config.n_val_rooms
