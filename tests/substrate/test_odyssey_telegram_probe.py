"""Focused tests for the frozen Telegram acknowledgement receipt."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from substrate import odyssey_authority as authority
from substrate import odyssey_telegram_probe as probe


def test_loader_registers_the_dynamic_module_for_dataclass_initialization(tmp_path: Path) -> None:
    notifier = tmp_path / "notifier.py"
    notifier.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Receipt:\n"
        "    value: int\n"
        "def probe(_digest):\n"
        "    return {'delivered': True, 'message_id': Receipt(73).value}\n",
        encoding="utf-8",
    )

    loaded = probe._load_notifier(notifier)

    assert loaded.probe("x")["message_id"] == 73
    sys.modules.pop("substrate_odyssey_telegram_notifier", None)


def test_probe_receipt_binds_an_acknowledgement_to_the_frozen_notifier(tmp_path: Path, monkeypatch) -> None:
    frozen_sha256 = "a" * 64
    notifier_sha256 = "b" * 64
    frozen = {
        "sha256": frozen_sha256,
        "implementation_sha256": {"telegram_notifier": notifier_sha256},
    }
    notifier_path = tmp_path / probe.NOTIFIER
    notifier_path.parent.mkdir(parents=True, exist_ok=True)
    notifier_path.write_text("# pinned notifier\n", encoding="utf-8")
    monkeypatch.setattr(authority, "_read_json", lambda *_args, **_kwargs: frozen)
    monkeypatch.setattr(authority, "_validate_frozen_build", lambda *_args, **_kwargs: frozen)
    monkeypatch.setattr(authority, "file_digest", lambda _path: notifier_sha256)
    monkeypatch.setattr(authority, "_git_head", lambda _root: "c" * 40)
    notifier = SimpleNamespace(
        probe=lambda digest: {
            "frozen_build_sha256": digest,
            "delivered": True,
            "message_id": 73,
        }
    )
    monkeypatch.setattr(probe, "_load_notifier", lambda _path: notifier)

    receipt = probe.run(tmp_path, Path("receipt.json"))

    assert receipt["all_pass"] is True
    assert receipt["delivery"] == {"message_id": 73, "acknowledged": True}
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))["sha256"] == receipt["sha256"]
