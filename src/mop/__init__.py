"""mop: a developmental continual-learning system built AROUND a frozen V-JEPA
perceptual substrate. The substrate is inherited perception, loaded once, never
trained. Everything else here (the shell, learning rules, metrics, experiments,
campaign) is the trainable system and the research surface.

This is NOT a JEPA and is not trained with a JEPA objective. V-JEPA is one frozen
module inside it. The package name is deliberately neutral.
"""

from __future__ import annotations

__version__ = "0.1.0"
