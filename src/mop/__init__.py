"""mop: a developmental continual-learning system built AROUND a frozen V-JEPA
perceptual substrate. The substrate is inherited perception, loaded once, never
trained. Everything else here (the shell, learning rules, metrics, experiments,
campaign) is the trainable system and the research surface.

This is NOT a JEPA and is not trained with a JEPA objective. V-JEPA is one frozen
module inside it. The package name is deliberately neutral.
"""

from __future__ import annotations

import os as _os

__version__ = "0.1.0"

_runtime_process_label = _os.environ.get("MOP_PROCESS_LABEL")
if _runtime_process_label:
    from mop.process_labels import set_process_label as _set_process_label

    _set_process_label(_runtime_process_label)

del _os, _runtime_process_label
if "_set_process_label" in globals():
    del _set_process_label
