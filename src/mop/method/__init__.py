"""Experiment validity kernel.

One compact system that every new substrate experiment must pass through. It is not a second campaign
engine: execution, scheduling, evidence indexing and configuration stay where they are. This package owns
one question only, asked before compute is spent rather than after: is this experiment capable of producing
the finding it claims it will produce.

House style: no dashes.
"""

from mop.method import (  # noqa: F401
    arms,
    baseline,
    bed,
    calibration,
    contracts,
    controls,
    defects,
    gate,
    graph,
    hypothesis,
    io,
    mechanism,
    power,
    report,
    voi,
)

__all__ = [
    "arms",
    "baseline",
    "bed",
    "calibration",
    "contracts",
    "controls",
    "defects",
    "gate",
    "graph",
    "hypothesis",
    "io",
    "mechanism",
    "power",
    "report",
    "voi",
]
