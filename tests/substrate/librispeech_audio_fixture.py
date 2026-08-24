"""Minimal synthetic LibriSpeech fixture for corpus-free Odyssey tests.

The production clip index is committed, but evaluator transcripts live under the
gitignored 90 GB corpus tree. Clean-clone gates therefore cannot exercise the
real ``require_evaluator_ground_truth=True`` path without a tiny stand-in.

This package points the documented task-bank override seams at a few synthetic
clips under ``tests/fixtures/librispeech_minimal/``. Transcript lines are
obviously fake (``FIXTURE ALPHA …``) and must never be real LibriSpeech text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from substrate import odyssey_task_bank as task_bank

# Relative to the repository root (pytest and G13 both run with cwd=root).
FIXTURE_DIR = Path("tests/fixtures/librispeech_minimal")
FIXTURE_INDEX = FIXTURE_DIR / "LIBRISPEECH_CLIP_INDEX.json"
FIXTURE_CORPUS = FIXTURE_DIR / "corpus"
FIXTURE_CLIP_PATH_PREFIX = FIXTURE_CORPUS.as_posix().rstrip("/") + "/"


def install_librispeech_audio_fixture(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route audio seams at the committed minimal fixture corpus.

    ``_audio_corpus_root`` and ``_audio_clip_index_path`` are the documented
    test override seams. Candidate-visible paths are assembled from the module
    constant ``_AUDIO_CLIP_PATH_PREFIX`` (kept separate so index-only tests can
    still emit production-shaped paths without a corpus); that constant is
    also redirected here so ``Path(clip_path).is_file()`` and structural-safety
    prefix checks resolve against the same fixture tree.
    """
    if not FIXTURE_INDEX.is_file():
        raise FileNotFoundError(f"missing librispeech fixture index: {FIXTURE_INDEX}")
    if not FIXTURE_CORPUS.is_dir():
        raise FileNotFoundError(f"missing librispeech fixture corpus: {FIXTURE_CORPUS}")

    monkeypatch.setattr(task_bank, "_audio_corpus_root", lambda: FIXTURE_CORPUS)
    monkeypatch.setattr(task_bank, "_audio_clip_index_path", lambda: FIXTURE_INDEX)
    monkeypatch.setattr(task_bank, "_AUDIO_CLIP_PATH_PREFIX", FIXTURE_CLIP_PATH_PREFIX)
    task_bank._AUDIO_CLIP_INDEX_CACHE.clear()
    return FIXTURE_CORPUS
