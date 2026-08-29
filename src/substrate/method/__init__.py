"""Experiment validity kernel.

One compact system that every new substrate experiment must pass through. It is not a second campaign
engine: execution, scheduling, evidence indexing and configuration stay where they are. This package owns
one question only, asked before compute is spent rather than after: is this experiment capable of producing
the finding it claims it will produce.

"""

from substrate.method import contracts, gate, graph, voi  # noqa: F401

__all__ = ["contracts", "gate", "graph", "voi"]
