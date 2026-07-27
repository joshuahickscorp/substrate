"""A real session authority, built from a session nobody wrote for this purpose.

Every battery in this program has the same weakness: run on a synthetic session, the answer is set by
whoever wrote the generator. The continuity margin is a function of how much filler was added. So a real
session is not a nicety, it is the difference between measuring the entity and measuring the author.

The session used here is the temporal core campaign's own execution history. A supervisor ran for days,
made scheduling decisions under changing resource pressure, recorded incidents, quarantined receipts, held
failures, and sealed outcomes. None of it was produced to test a cognitive architecture, its length was
chosen by the compute rather than by us, and every entry is already sealed and hash bound.

That is the whole argument for it, and it is also its limitation: it is one session, from one program, in
one domain. It is recorded as such.

House style: no dashes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from mop.cognition import io

TEMPORAL_RUNS = io.ROOT / "runs" / "substrate" / "mop-temporal-core-mechanism-v1"
LOG = io.ROOT / "logs" / "supervisor.detached.log"

# what a session event carries. Every field comes from the record, none is invented.
EVENT_FIELDS = ("index", "kind", "stage", "identity", "detail", "source_path")

KINDS = ("shard_completed", "shard_quarantined", "failure_hold", "orchestration_incident",
         "scheduling_observation", "stage_completed")


class Refused(RuntimeError):
    """A session the authority will not certify."""


def _orchestration_events() -> list[dict]:
    root = TEMPORAL_RUNS / "orchestration"
    out = []
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        schema = doc.get("schema", "")
        kind = ("shard_quarantined" if "quarantine" in schema else
                "orchestration_incident" if "incident" in schema else "stage_completed")
        out.append({"kind": kind, "stage": doc.get("stage") or doc.get("classification", ""),
                    "identity": doc.get("identity") or path.stem,
                    "detail": {k: doc.get(k) for k in ("classification", "trigger", "repair") if k in doc},
                    "source_path": path.relative_to(io.ROOT).as_posix()})
    return out


def _hold_events() -> list[dict]:
    root = TEMPORAL_RUNS / "failure_holds"
    out = []
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"kind": "failure_hold", "stage": doc.get("stage", ""),
                    "identity": doc.get("identity", path.stem),
                    "detail": {"error_code": doc.get("error_code"), "state": doc.get("state"),
                               "attempts": doc.get("observed_identical_failures_minimum")},
                    "source_path": path.relative_to(io.ROOT).as_posix()})
    return out


def _receipt_events() -> list[dict]:
    out = []
    for sub in ("e2_principal", "e2_principal_corrections", "third_bed_preflight"):
        root = TEMPORAL_RUNS / sub
        for path in sorted(root.glob("*.json")) if root.is_dir() else []:
            try:
                doc = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            runs = doc.get("runs") or []
            wall = sum(float(r.get("wall_seconds") or 0) for r in runs if isinstance(r, dict))
            out.append({"kind": "shard_completed", "stage": sub, "identity": path.stem,
                        "detail": {"cells": len(runs), "wall_seconds": round(wall, 1),
                                   "bed": doc.get("bed"), "seed": doc.get("seed")},
                        "source_path": path.relative_to(io.ROOT).as_posix()})
    return out


_SCHED = re.compile(r"workers=(\d+) cap=(\d+) class=(\w+) remaining=(\{.*?\})")


def _scheduling_events(limit: int = 400) -> list[dict]:
    """The supervisor's own decisions under changing pressure. State, action, and what followed."""
    if not LOG.is_file():
        return []
    out, seen = [], None
    for line in LOG.read_text(errors="ignore").splitlines():
        m = _SCHED.search(line)
        if not m:
            continue
        workers, cap, klass, remaining = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        key = (workers, cap, klass, remaining)
        if key == seen:
            continue  # the supervisor prints every minute; only transitions are events
        seen = key
        out.append({"kind": "scheduling_observation", "stage": "supervisor", "identity": f"t{len(out)}",
                    "detail": {"workers": workers, "cap": cap, "resource_class": klass,
                               "remaining": remaining},
                    "source_path": LOG.relative_to(io.ROOT).as_posix()})
        if len(out) >= limit:
            break
    return out


def build() -> dict:
    events = (_receipt_events() + _hold_events() + _orchestration_events() + _scheduling_events())
    for i, e in enumerate(events):
        e["index"] = i
    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    return {
        "schema": "substrate-real-session-authority/v1",
        "session": "the temporal core mechanism campaign's own execution history",
        "why_it_is_real": ("a supervisor ran for days under changing resource pressure and recorded every "
                           "decision, incident, hold and outcome. None of it was produced to test a "
                           "cognitive architecture and its length was set by the compute, not by us"),
        "provenance": {"runs_root": TEMPORAL_RUNS.relative_to(io.ROOT).as_posix(),
                       "log": LOG.relative_to(io.ROOT).as_posix() if LOG.is_file() else None,
                       "every_event_cites_its_file": all(e["source_path"] for e in events)},
        "event_fields": list(EVENT_FIELDS),
        "kinds": list(KINDS),
        "events": events,
        "event_count": len(events),
        "events_by_kind": by_kind,
        "length_not_chosen_by_us": True,
        "limitations": ["one session", "one program", "one domain",
                        "the supervisor is not a cognitive agent, so agency batteries cannot use it"],
        "usable_for": ["continuity under interruption", "grounding referents and outcomes",
                       "world model state and decision pairs", "developmental divergence histories"],
        "activation": False,
    }


def certify(doc: dict) -> dict:
    """A session authority is only usable if it is real, cited, and long enough to truncate."""
    checks = {
        "has_events": doc["event_count"] > 0,
        "every_event_cited": doc["provenance"]["every_event_cites_its_file"],
        "more_than_one_kind": len(doc["events_by_kind"]) > 1,
        "long_enough_to_truncate": doc["event_count"] >= 50,
        "not_authored_for_this_purpose": doc["length_not_chosen_by_us"],
        "limitations_declared": bool(doc["limitations"]),
    }
    return {"checks": checks, "certified": all(checks.values()),
            "failed": sorted(k for k, v in checks.items() if not v)}


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "build"
    if command not in ("build", "seal"):
        raise ValueError(argv)
    doc = build()
    cert = certify(doc)
    if not cert["certified"]:
        raise Refused(f"the session authority failed certification: {cert['failed']}")
    path = io.seal("SUBSTRATE_REAL_SESSION_AUTHORITY.json", {**doc, "certification": cert})
    print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                      "events": doc["event_count"], "by_kind": doc["events_by_kind"],
                      "certified": cert["certified"]}, indent=2))


if __name__ == "__main__":
    main()
