
from __future__ import annotations

ROLE_EXTENSION = "extension"
ROLE_SUPERVISOR = "supervisor"
ROLE_CAPSULE = "capsule"
ROLE_WORKER = "worker"
ROLE_ADOPTER = "adopter"
ROLE_LAUNCHER = "launcher"


def mop_label(program: str, role: str, detail: str = "") -> str:

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
