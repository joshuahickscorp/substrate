from __future__ import annotations

from typing import Any

_setproctitle: Any
try:
    import setproctitle as _setproctitle
except ImportError:  # pragma: no cover - the runtime dependency is declared
    _setproctitle = None


def set_process_label(label: str) -> bool:
    if not isinstance(label, str) or not label.strip() or "\0" in label:
        raise ValueError("process label must be a nonempty string without NUL bytes")
    if _setproctitle is None:
        return False
    try:
        _setproctitle.setproctitle(label)
    except (OSError, RuntimeError):
        return False
    return True


__all__ = ["set_process_label"]
