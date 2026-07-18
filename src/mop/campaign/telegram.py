"""Unified campaign engine: campaign-level Telegram events.

Reuses the already-configured Hawking/MOP Telegram bot (token and chat id from macOS Keychain via
``mop.studio.telegram_rung_notifier.send_message``). Campaign events are deduplicated on disk and notifier
failures are isolated from scientific execution: a delivery error is recorded and swallowed, never raised
into the scheduler.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _dedup_path(root: Path) -> Path:
    return Path(root) / "telegram_sent.json"


def _already_sent(root: Path, key: str) -> bool:
    path = _dedup_path(root)
    if not path.exists():
        return False
    try:
        return key in set(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return False


def _mark_sent(root: Path, key: str) -> None:
    path = _dedup_path(root)
    try:
        current = set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()
    except Exception:
        current = set()
    current.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(current)), encoding="utf-8")


def send_campaign_event(
    kind: str, text: str, root: str | Path = ".", dedup_key: str | None = None
) -> dict[str, Any]:
    """Send one deduplicated campaign event. Returns a delivery receipt; never raises on notifier failure.

    Returns {delivered: bool, kind, key, message_id?, error?}.
    """

    root = Path(root)
    key = dedup_key or hashlib.sha256(f"{kind}|{text}".encode()).hexdigest()[:16]
    if _already_sent(root, key):
        return {"delivered": False, "kind": kind, "key": key, "reason": "deduplicated"}
    body = f"[MOP campaign] {kind}\n{text}"
    try:
        from mop.studio.telegram_rung_notifier import send_message

        result = send_message(body)
        _mark_sent(root, key)
        message_id = None
        if isinstance(result, dict):
            message_id = (
                (result.get("result") or {}).get("message_id")
                if "result" in result
                else result.get("message_id")
            )
        return {"delivered": True, "kind": kind, "key": key, "message_id": message_id}
    except Exception as exc:  # noqa: BLE001 (notifier failure must not affect science)
        return {"delivered": False, "kind": kind, "key": key, "error": f"{type(exc).__name__}: {exc}"}


def record_delivery(root: str | Path, receipt: dict[str, Any]) -> Path:
    """Persist a delivery receipt so the compliance verifier can confirm real Telegram delivery."""

    path = Path(root) / "telegram_delivery.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(receipt)
    path.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    return path
