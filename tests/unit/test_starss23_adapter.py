
from __future__ import annotations

import wave

import numpy as np
import pytest

from mop.beds.starss23 import adapter as A
from mop.beds.starss23 import synthetic_corpus as SC
from mop.beds.starss23.schema import SAMPLES_PER_FRAME, Clip, OnsetEvent
from mop.escs.accounting import WorkVector


def _zeros_audio(n_frames: int) -> np.ndarray:
    return np.zeros((4, n_frames * SAMPLES_PER_FRAME), dtype=np.float32)




def test_parse_clip_name_extracts_fold_room_mix_and_normalizes_ids() -> None:
    name = A.parse_clip_name("fold3_room4_mix007")
    assert (name.fold, name.room, name.mix) == (3, 4, 7)
    assert name.clip_id == "fold3_room4_mix007"
    assert name.room_id == "room04"


def test_parse_clip_name_strips_extensions() -> None:
    assert A.parse_clip_name("fold4_room21_mix003.wav").room_id == "room21"
    assert A.parse_clip_name("fold4_room21_mix003.csv").mix == 3


@pytest.mark.parametrize("bad", ["", "room4_mix1", "fold3_room4", "clip0", "fold3-room4-mix1"])
def test_parse_clip_name_refuses_non_starss_names(bad: str) -> None:
    with pytest.raises(A.AdapterRefusal):
        A.parse_clip_name(bad)


def test_format_clip_id_round_trips_through_parse() -> None:
    clip_id = A.format_clip_id(3, 4, 12)
    assert clip_id == "fold3_room4_mix012"
    assert A.parse_clip_name(clip_id).clip_id == clip_id




def test_parse_metadata_reads_six_column_starss23_rows() -> None:
    text = "5,0,0,10,-20,150\n5,1,1,30,0,220\n"
    rows = A.parse_starss23_metadata(text)
    assert len(rows) == 2
    assert rows[0] == A.MetadataRow(5, 0, 0, 10, -20, 150)
    assert rows[1].distance == 220


def test_parse_metadata_reads_five_column_rows_without_distance() -> None:
    rows = A.parse_starss23_metadata("7,2,0,90,10\n")
    assert rows[0].distance == A.DISTANCE_ABSENT
    assert not rows[0].has_distance


def test_parse_metadata_ignores_blank_lines() -> None:
    rows = A.parse_starss23_metadata("\n5,0,0,0,0,100\n\n")
    assert len(rows) == 1


@pytest.mark.parametrize("bad", ["5,0,0,10", "5,0,0,10,-20,150,999", "5,0,0,10,x,150"])
def test_parse_metadata_refuses_malformed_lines(bad: str) -> None:
    with pytest.raises(A.AdapterRefusal):
        A.parse_starss23_metadata(bad + "\n")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"azimuth": 181},
        {"azimuth": -181},
        {"elevation": 91},
        {"class_id": 13},
        {"frame": -1},
        {"distance": 0},
    ],
)
def test_metadata_row_validates_ranges(kwargs: dict[str, int]) -> None:
    base = {"frame": 1, "class_id": 0, "source_id": 0, "azimuth": 0, "elevation": 0, "distance": 100}
    base.update(kwargs)
    with pytest.raises(A.AdapterRefusal):
        A.MetadataRow(**base)


def test_format_metadata_round_trips_rows_sorted_canonically() -> None:
    rows = (
        A.MetadataRow(6, 1, 1, 30, 0, 220),
        A.MetadataRow(5, 0, 0, 10, -20, 150),
    )
    text = A.format_starss23_metadata(rows)
    reparsed = A.parse_starss23_metadata(text)
    assert reparsed == tuple(sorted(rows, key=A.MetadataRow.sort_key))


def test_format_metadata_refuses_partial_distance_column() -> None:
    rows = (
        A.MetadataRow(5, 0, 0, 10, -20, 150),
        A.MetadataRow(6, 1, 1, 30, 0, A.DISTANCE_ABSENT),
    )
    with pytest.raises(A.AdapterRefusal):
        A.format_starss23_metadata(rows)




def test_onset_derivation_takes_first_frame_of_each_activity_run() -> None:
    rows = tuple(A.MetadataRow(frame, 0, 0, 0, 0, 200) for frame in (5, 6, 7, 10, 11))
    onsets = A.onset_events_from_rows(rows)
    assert tuple(onset.frame for onset in onsets) == (5, 10)


def test_onset_derivation_collapses_simultaneous_onsets_keeping_nearest() -> None:
    rows = (
        A.MetadataRow(5, 0, 0, 10, 0, 300),  # farther source
        A.MetadataRow(5, 1, 1, 20, 0, 100),  # nearer source
    )
    onsets = A.onset_events_from_rows(rows)
    assert len(onsets) == 1
    assert onsets[0].distance == 100.0
    assert onsets[0].class_id == 1


def test_onset_derivation_refuses_rows_without_distance() -> None:
    rows = (A.MetadataRow(5, 0, 0, 10, 0, A.DISTANCE_ABSENT),)
    with pytest.raises(A.AdapterRefusal):
        A.onset_events_from_rows(rows)


def test_metadata_text_from_onsets_round_trips_integer_onsets() -> None:
    onsets = (
        OnsetEvent(frame=2, class_id=0, azimuth=10.0, elevation=-5.0, distance=200.0),
        OnsetEvent(frame=6, class_id=3, azimuth=-30.0, elevation=15.0, distance=150.0),
    )
    text = A.metadata_text_from_onsets(onsets, active_frames=2)
    recovered = A.onset_events_from_rows(A.parse_starss23_metadata(text))
    assert recovered == onsets


def test_metadata_serialization_refuses_non_integer_onset_fields() -> None:
    onsets = (OnsetEvent(frame=2, class_id=0, azimuth=10.5, elevation=0.0, distance=200.0),)
    with pytest.raises(A.AdapterRefusal):
        A.metadata_rows_from_onsets(onsets)


def test_domain_seed_preserves_the_four_noisy_tv_streams() -> None:
    cases = (
        ("mop.beds.starss23.real.noisy_tv", b"mop-starss23-real-noisy-tv-v1", 481492669),
        ("mop.beds.starss23.count.noisy_tv", b"mop-starss23-count-noisy-tv-v1", 2165001989),
        (
            "mop.beds.starss23.count_repro_featurizer_estimator.noisy_tv",
            b"mop-starss23-count-repro-featurizer-estimator-noisy-tv-v1",
            1391724852,
        ),
        ("mop.beds.starss23.doa.noisy_tv", b"mop-starss23-doa-noisy-tv-v1", 3650713416),
    )
    assert tuple(A.domain_seed(7, key, domain) for key, domain, _ in cases) == tuple(
        expected for _, _, expected in cases
    )




def test_synthetic_adapter_builds_clips_through_the_parse_path() -> None:
    onsets = (
        OnsetEvent(frame=2, class_id=0, azimuth=10.0, elevation=-5.0, distance=200.0),
        OnsetEvent(frame=6, class_id=1, azimuth=40.0, elevation=0.0, distance=120.0),
    )
    clip_id = "fold3_room0_mix000"
    adapter = A.SyntheticStarssAdapter(
        {clip_id: _zeros_audio(10)},
        {clip_id: A.metadata_text_from_onsets(onsets, active_frames=2)},
    )
    assert isinstance(adapter, A.StarssAdapter)
    assert adapter.source_kind() == A.SOURCE_KIND_SYNTHETIC
    assert adapter.rights_clean() is True
    clip = adapter.clip(clip_id)
    assert isinstance(clip, Clip)
    assert clip.room_id == "room00"
    assert clip.n_frames == 10
    assert clip.onsets == onsets
    assert adapter.onsets(clip_id) == onsets
    assert adapter.audio(clip_id).shape == (4, 10 * SAMPLES_PER_FRAME)


def test_synthetic_adapter_refuses_mismatched_audio_and_metadata_keys() -> None:
    with pytest.raises(A.AdapterRefusal):
        A.SyntheticStarssAdapter(
            {"fold3_room0_mix000": _zeros_audio(4)},
            {"fold3_room1_mix000": "0,0,0,0,0,100\n"},
        )


def test_synthetic_adapter_refuses_unknown_clip_lookups() -> None:
    clip_id = "fold3_room0_mix000"
    adapter = A.SyntheticStarssAdapter({clip_id: _zeros_audio(4)}, {clip_id: "0,0,0,0,0,100\n"})
    with pytest.raises(A.AdapterRefusal):
        adapter.audio("fold3_room9_mix000")
    with pytest.raises(A.AdapterRefusal):
        adapter.clip("fold3_room9_mix000")


def test_synthetic_adapter_refuses_bad_audio_shape() -> None:
    clip_id = "fold3_room0_mix000"
    with pytest.raises(A.AdapterRefusal):
        A.SyntheticStarssAdapter(
            {clip_id: np.zeros((4, SAMPLES_PER_FRAME + 1), dtype=np.float32)},
            {clip_id: "0,0,0,0,0,100\n"},
        )


def test_transport_charge_reports_audio_bytes_as_raw_transport() -> None:
    clip_id = "fold3_room0_mix000"
    audio = _zeros_audio(4)
    adapter = A.SyntheticStarssAdapter({clip_id: audio}, {clip_id: "0,0,0,0,0,100\n"})
    charge = adapter.transport_charge()
    assert isinstance(charge, WorkVector)
    assert charge.raw_transport_and_adapters == adapter.audio(clip_id).nbytes
    assert charge.event_formation == 0




def test_dev_split_is_room_disjoint_and_matches_the_fold_shape() -> None:
    corpus = SC.generate_corpus(
        SC.SyntheticCorpusConfig(
            n_fold3_rooms=4,
            n_val_rooms=1,
            n_fold4_rooms=2,
            clips_per_room=2,
            clip_frames=20,
            onsets_per_clip=3,
            nuisances_per_clip=3,
        ),
        seed=1,
    )
    adapter = corpus.adapter()
    dev = adapter.dev_split()
    assert dev.sizes == {"dev_train": 4 * 2, "dev_test": 2 * 2}
    train_rooms = {A.parse_clip_name(c).room_id for c in dev.dev_train}
    test_rooms = {A.parse_clip_name(c).room_id for c in dev.dev_test}
    assert train_rooms.isdisjoint(test_rooms)


def test_harness_split_nests_inside_the_dev_split() -> None:
    corpus = SC.generate_corpus(SC.SyntheticCorpusConfig.tiny(), seed=2)
    adapter = corpus.adapter()
    dev = adapter.dev_split()
    split = adapter.harness_split(n_train_rooms=2, n_val_rooms=1)
    test_ids = {clip.clip_id for clip in split.test}
    assert test_ids == set(dev.dev_test)


def test_native_fold_split_and_corpus_mapping_have_one_shared_authority() -> None:
    adapter = SC.generate_corpus(SC.SyntheticCorpusConfig.tiny(), seed=3).adapter()
    split = A.native_fold_split(adapter, n_val_rooms=1)
    assert {clip.clip_id for clip in split.test} == set(adapter.dev_split().dev_test)
    assert split.detail["split_rule"] == (
        "test = native fold-4 dev-test; val = last N fold-3 rooms; train = rest of fold-3"
    )
    mapped = A.map_clip_audio(adapter, lambda audio: np.asarray([audio.shape[1]]))
    assert mapped == {
        clip.clip_id: np.asarray([clip.n_frames * SAMPLES_PER_FRAME]) for clip in adapter.clips()
    }
    with pytest.raises(A.AdapterRefusal, match="leave at least one train room"):
        A.native_fold_split(adapter, n_val_rooms=0)


def test_dev_split_refuses_a_non_dev_fold() -> None:
    clip_id = "fold9_room0_mix000"
    adapter = A.SyntheticStarssAdapter({clip_id: _zeros_audio(4)}, {clip_id: "0,0,0,0,0,100\n"})
    with pytest.raises(A.AdapterRefusal):
        adapter.dev_split()


def test_native_dev_split_refuses_overlapping_clips() -> None:
    with pytest.raises(A.AdapterRefusal):
        A.NativeDevSplit(dev_train=("fold3_room0_mix000",), dev_test=("fold3_room0_mix000",))




def _write_real_tree(root, clips: dict[str, tuple[int, tuple[OnsetEvent, ...]]]):

    foa_root = root / "foa_dev"
    meta_root = root / "metadata_dev"
    for clip_id, (n_frames, onsets) in clips.items():
        fold_dir = "dev-train-sony" if clip_id.startswith("fold3") else "dev-test-sony"
        (foa_root / fold_dir).mkdir(parents=True, exist_ok=True)
        (meta_root / fold_dir).mkdir(parents=True, exist_ok=True)
        n_samples = n_frames * SAMPLES_PER_FRAME
        pcm = np.zeros((n_samples, 4), dtype="<i2")
        with wave.open(str(foa_root / fold_dir / f"{clip_id}.wav"), "wb") as handle:
            handle.setnchannels(4)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(pcm.tobytes())
        (meta_root / fold_dir / f"{clip_id}.csv").write_text(
            A.metadata_text_from_onsets(onsets, active_frames=2), encoding="utf-8"
        )
    return foa_root, meta_root


def test_real_adapter_decodes_a_real_foa_tree_through_the_shared_parse_path(tmp_path) -> None:
    onsets = (
        OnsetEvent(frame=3, class_id=0, azimuth=10.0, elevation=-5.0, distance=200.0),
        OnsetEvent(frame=9, class_id=2, azimuth=-40.0, elevation=12.0, distance=150.0),
    )
    foa_root, meta_root = _write_real_tree(tmp_path, {"fold3_room4_mix001": (20, onsets)})
    real = A.RealStarssAdapter(foa_root, meta_root, rights_clean=True)
    assert isinstance(real, A.StarssAdapter)
    assert real.source_kind() == A.SOURCE_KIND_REAL
    assert real.rights_clean() is True
    clip = real.clip("fold3_room4_mix001")
    assert clip.room_id == "room04"
    assert clip.n_frames == 20
    assert clip.onsets == onsets
    assert real.audio("fold3_room4_mix001").shape == (4, 20 * SAMPLES_PER_FRAME)


def test_real_adapter_dev_split_is_room_disjoint_across_folds(tmp_path) -> None:
    one = (OnsetEvent(frame=2, class_id=0, azimuth=0.0, elevation=0.0, distance=100.0),)
    foa_root, meta_root = _write_real_tree(
        tmp_path,
        {
            "fold3_room4_mix001": (12, one),
            "fold3_room6_mix002": (12, one),
            "fold4_room8_mix003": (12, one),
        },
    )
    dev = A.RealStarssAdapter(foa_root, meta_root).dev_split()
    assert dev.sizes == {"dev_train": 2, "dev_test": 1}
    train_rooms = {A.parse_clip_name(c).room_id for c in dev.dev_train}
    test_rooms = {A.parse_clip_name(c).room_id for c in dev.dev_test}
    assert train_rooms.isdisjoint(test_rooms)


def test_real_adapter_defaults_to_rights_clean_for_the_mit_corpus(tmp_path) -> None:
    one = (OnsetEvent(frame=2, class_id=0, azimuth=0.0, elevation=0.0, distance=100.0),)
    foa_root, meta_root = _write_real_tree(tmp_path, {"fold3_room4_mix001": (12, one)})
    assert A.RealStarssAdapter(foa_root, meta_root).rights_clean() is True


def test_real_adapter_refuses_a_missing_tree() -> None:
    with pytest.raises(A.AdapterRefusal):
        A.RealStarssAdapter("/does/not/exist", "/also/missing")


def test_real_adapter_truncates_onsets_past_the_kept_length(tmp_path) -> None:
    onsets = (
        OnsetEvent(frame=3, class_id=0, azimuth=0.0, elevation=0.0, distance=100.0),
        OnsetEvent(frame=18, class_id=1, azimuth=0.0, elevation=0.0, distance=100.0),
    )
    foa_root, meta_root = _write_real_tree(tmp_path, {"fold3_room4_mix001": (10, onsets)})
    real = A.RealStarssAdapter(foa_root, meta_root)
    clip = real.clip("fold3_room4_mix001")
    assert clip.onset_frames == (3,)
    trunc = {t.clip_id: t for t in real.truncations()}["fold3_room4_mix001"]
    assert trunc.dropped_onsets_past_end == 1




def test_from_dir_round_trips_a_written_corpus(tmp_path) -> None:
    corpus = SC.generate_corpus(SC.SyntheticCorpusConfig.tiny(), seed=3)
    corpus.write_to(tmp_path)
    disk_adapter = A.SyntheticStarssAdapter.from_dir(tmp_path)
    memory_adapter = corpus.adapter()
    disk_clips = {clip.clip_id: clip for clip in disk_adapter.clips()}
    for clip in memory_adapter.clips():
        assert disk_clips[clip.clip_id].onsets == clip.onsets
        assert disk_clips[clip.clip_id].audio_sha256 == clip.audio_sha256


def test_from_dir_refuses_a_non_starss_tree(tmp_path) -> None:
    with pytest.raises(A.AdapterRefusal):
        A.SyntheticStarssAdapter.from_dir(tmp_path)
