"""Uniform process-label scheme for MoP chain parents.

The going-forward standard for labelling the detached chain PARENT processes we
control is::

    mop:<program>:<role>[:<detail>]

For example ``mop:fullgen:extension`` labels the full-generations extension chain
parent, and ``mop:fullgen:extension:resume`` would label a resume variant of the
same parent.

Sealed programs keep their legacy labels for now. The generic supervisor labels
(``mop-supervisor:<id>``, ``mop-capsule:<id>``) and the sealed mechanics worker
labels are pinned by live sealed campaign manifests and must not be re-labelled
piecemeal. They stay on their legacy strings until a future supervisor seal
adopts this scheme wholesale. This module only defines the standard and is used
by the net-new, not-yet-launched chain parents.
"""

from __future__ import annotations

# Named role constants for the scheme.
ROLE_EXTENSION = "extension"
ROLE_SUPERVISOR = "supervisor"
ROLE_CAPSULE = "capsule"
ROLE_WORKER = "worker"
ROLE_ADOPTER = "adopter"
ROLE_LAUNCHER = "launcher"


def mop_label(program: str, role: str, detail: str = "") -> str:
    """Return a uniform ``mop:<program>:<role>[:<detail>]`` process label.

    ``program`` and ``role`` are required and must be non-empty. ``detail`` is
    optional; when empty the label has three colon-separated fields, otherwise
    four. None of the fields may contain a colon, since colon is the reserved
    field separator.
    """

    program = program.strip()
    role = role.strip()
    detail = detail.strip()
    if not program:
        raise ValueError("program must be a non-empty label field")
    if not role:
        raise ValueError("role must be a non-empty label field")
    for name, value in (("program", program), ("role", role), ("detail", detail)):
        if ":" in value:
            raise ValueError(f"{name} label field must not contain ':'")
    parts = ["mop", program, role]
    if detail:
        parts.append(detail)
    return ":".join(parts)


__all__ = [
    "ROLE_ADOPTER",
    "ROLE_CAPSULE",
    "ROLE_EXTENSION",
    "ROLE_LAUNCHER",
    "ROLE_SUPERVISOR",
    "ROLE_WORKER",
    "mop_label",
]
