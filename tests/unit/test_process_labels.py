from __future__ import annotations

import os
import subprocess
import sys

import pytest

from mop import process_labels


class _Recorder:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def setproctitle(self, label: str) -> None:
        self.labels.append(label)


def test_process_label_is_forwarded_to_os_title_provider(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(process_labels, "_setproctitle", recorder)

    assert process_labels.set_process_label("mop-c2-s03-worker") is True
    assert recorder.labels == ["mop-c2-s03-worker"]


@pytest.mark.parametrize("label", ("", "   ", "mop\0worker"))
def test_process_label_rejects_ambiguous_values(label):
    with pytest.raises(ValueError, match="process label"):
        process_labels.set_process_label(label)


def test_process_label_is_best_effort_when_provider_is_missing(monkeypatch):
    monkeypatch.setattr(process_labels, "_setproctitle", None)
    assert process_labels.set_process_label("mop-supervisor") is False


def test_mop_import_applies_inherited_process_label() -> None:
    environment = {**os.environ, "MOP_PROCESS_LABEL": "mop-import-smoke"}
    observed = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import mop, setproctitle; print(setproctitle.getproctitle())",
        ],
        env=environment,
        text=True,
    ).strip()

    assert observed == "mop-import-smoke"
