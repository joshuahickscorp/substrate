"""Temporal core integration, against a core that was never licensed.

Section 9 says: if the current Temporal Core authority has not selected a valid minimal core, use a
declared control implementation and preserve the scientific limitation. That is exactly the situation. The
factorial is terminal, role B found the load bearing baselines unconverged on all three beds, and
`MOP_OWNED_TEMPORAL_CORE_V1.json` records `selected: false`.

So this module wires a control, and the control is labelled a control everywhere it appears. The interface
is versioned so that a licensed core can replace it without the runtime changing, and a test asserts the
runtime cannot tell the difference at the call site while the receipt always can.

The other half of section 9 matters more than the first. The runtime must not silently collapse five
information sources into one state vector: current observation, explicit history, temporal core state,
memory retrieval, and world model prediction. Collapsing them is how a system claims a temporal effect that
was really a lookup. Every read here is tagged with which of the five it came from, and a reader that asks
for a merged view gets the tags with it.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from substrate import evidence as io
from substrate import program as P

INTERFACE_VERSION = "substrate-temporal-core-interface/v1"

# section 9, what the core must provide
PROVIDES = (
    "current_temporal_state",
    "state_confidence",
    "transition_signal",
    "history_summary",
    "reset_authority",
    "checkpoint",
    "restore",
)

# section 9, the five sources that must stay distinguishable
SOURCES = (
    "current_observation",
    "explicit_history",
    "temporal_core_state",
    "memory_retrieval",
    "world_model_prediction",
)

TEMPORAL_AUTHORITY = "temporal:MOP_OWNED_TEMPORAL_CORE_V1.json"


class Refused(RuntimeError):
    """A temporal read the interface will not serve."""


@dataclass
class Reading:
    """A value and which of the five sources it came from. The tag is not optional."""

    value: object
    source: str
    confidence: float

    def __post_init__(self):
        if self.source not in SOURCES:
            raise Refused(f"undeclared information source {self.source!r}")


class TemporalCore:
    """The versioned interface. A licensed core implements this; the control below also does."""

    version = INTERFACE_VERSION
    is_control = True
    limitation = ""

    def __init__(self, decay: float = 0.6, horizon: int = 8):
        self.decay = decay
        self.horizon = horizon
        self.state: list[float] = []
        self.history: list[object] = []
        self.resets = 0

    # ------------------------------------------------------------ the seven declared capabilities
    def observe(self, value: float) -> TemporalCore:
        self.state.append(float(value))
        self.history.append(value)
        if len(self.state) > self.horizon:
            self.state = self.state[-self.horizon :]
        return self

    def current_temporal_state(self) -> Reading:
        if not self.state:
            return Reading(0.0, "temporal_core_state", 0.0)
        weights = [self.decay**i for i in range(len(self.state))][::-1]
        total = sum(w for w in weights) or 1.0
        value = sum(v * w for v, w in zip(self.state, weights, strict=False)) / total
        return Reading(round(value, 6), "temporal_core_state", self.state_confidence())

    def state_confidence(self) -> float:
        return round(min(len(self.state) / max(self.horizon, 1), 1.0), 6)

    def transition_signal(self) -> Reading:
        if len(self.state) < 2:
            return Reading(0.0, "temporal_core_state", 0.0)
        return Reading(round(abs(self.state[-1] - self.state[-2]), 6), "temporal_core_state", self.state_confidence())

    def history_summary(self, k: int = 4) -> Reading:
        """Explicit history is a different source from core state, and is tagged as one."""
        return Reading(list(self.history[-k:]), "explicit_history", 1.0 if self.history else 0.0)

    def reset_authority(self) -> dict:
        return {
            "who_may_reset": ["goal_authority", "episode_boundary"],
            "resets_so_far": self.resets,
            "rule": "a reset is destructive and is recorded, never silent",
        }

    def reset(self, by: str) -> dict:
        if by not in self.reset_authority()["who_may_reset"]:
            raise Refused(f"{by} is not authorized to reset the temporal core")
        self.state, self.resets = [], self.resets + 1
        return {"reset_by": by, "resets": self.resets}

    def checkpoint(self) -> dict:
        return {
            "version": self.version,
            "is_control": self.is_control,
            "state": list(self.state),
            "history": list(self.history),
            "resets": self.resets,
            "horizon": self.horizon,
            "decay": self.decay,
            "identity": io.sha_obj({"s": self.state, "h": self.history, "r": self.resets}),
        }

    def restore(self, snapshot: dict) -> TemporalCore:
        if snapshot.get("version") != self.version:
            raise Refused("a checkpoint from another interface version is not restorable here")
        self.state = list(snapshot["state"])
        self.history = list(snapshot["history"])
        self.resets = snapshot["resets"]
        if io.sha_obj({"s": self.state, "h": self.history, "r": self.resets}) != snapshot["identity"]:
            raise Refused("restored temporal state does not reproduce the checkpoint identity")
        return self


class DeclaredControl(TemporalCore):
    """The control. An exponentially weighted trace, chosen because it is the simplest thing that keeps
    state at all, and labelled a control in every receipt it touches."""

    is_control = True
    limitation = (
        "no temporal core was scientifically licensed. The factorial is terminal and its "
        "independent verification did not pass, so this is a declared control and not a "
        "selected minimal core. Any result that depends on it inherits that limitation"
    )


class LicensedCore(TemporalCore):
    """The slot a licensed core would occupy. It refuses to instantiate without a licensing receipt."""

    is_control = False

    def __init__(self, *a, **kw):
        state = P.evidence_state(TEMPORAL_AUTHORITY)
        raise Refused(
            "no licensed temporal core exists. "
            f"{TEMPORAL_AUTHORITY} counts as evidence: {state['counts']}, reason: "
            f"{state.get('reason') or 'selected is false'}"
        )


def resolve_core() -> TemporalCore:
    """Use a licensed core when one exists, and say plainly when one does not."""
    try:
        return LicensedCore()
    except Refused:
        return DeclaredControl()


def merged_view(core: TemporalCore, *, observation, retrieved=None, predicted=None) -> dict:
    """Every source stays tagged. A merged view that lost its tags would be the collapse section 9 forbids."""
    readings = {
        "current_observation": Reading(observation, "current_observation", 1.0),
        "explicit_history": core.history_summary(),
        "temporal_core_state": core.current_temporal_state(),
        "memory_retrieval": Reading(retrieved, "memory_retrieval", 0.0 if retrieved is None else 0.7),
        "world_model_prediction": Reading(predicted, "world_model_prediction", 0.0 if predicted is None else 0.6),
    }
    return {
        "readings": {k: {"value": r.value, "source": r.source, "confidence": r.confidence} for k, r in readings.items()},
        "sources_present": sorted(k for k, r in readings.items() if r.confidence > 0),
        "sources_declared": list(SOURCES),
        "collapsed": False,
        "rule": "the five sources are never merged into one untagged value",
    }


def declaration() -> dict:
    core = resolve_core()
    for v in (0.1, 0.4, 0.35, 0.9):
        core.observe(v)
    view = merged_view(core, observation={"label": "a"}, retrieved=None, predicted=None)
    snapshot = core.checkpoint()
    revived = DeclaredControl().restore(snapshot)
    return {
        "schema": "substrate-temporal-core/v1",
        "interface_version": INTERFACE_VERSION,
        "provides": list(PROVIDES),
        "information_sources": list(SOURCES),
        "implementation": type(core).__name__,
        "is_control": core.is_control,
        "scientific_limitation": core.limitation,
        "licensing": {
            "authority": TEMPORAL_AUTHORITY,
            "counts_as_evidence": P.evidence_state(TEMPORAL_AUTHORITY)["counts"],
            "licensed_core_available": False,
            "rule": "a licensed core drops into the same interface without the runtime changing",
        },
        "merged_view_example": view,
        "checkpoint_roundtrip": {
            "identity": snapshot["identity"][:16],
            "restored": revived.checkpoint()["identity"] == snapshot["identity"],
        },
        "reset_authority": core.reset_authority(),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_TEMPORAL_CORE.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "implementation": doc["implementation"],
                "is_control": doc["is_control"],
                "sources": len(doc["information_sources"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
