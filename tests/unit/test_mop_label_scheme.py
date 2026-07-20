from __future__ import annotations

import pytest

from mop.studio import mop_label_scheme as scheme


def test_three_field_label() -> None:
    assert scheme.mop_label("fullgen", scheme.ROLE_EXTENSION) == "mop:fullgen:extension"


def test_four_field_label_with_detail() -> None:
    assert scheme.mop_label("fullgen", scheme.ROLE_EXTENSION, "resume") == "mop:fullgen:extension:resume"


def test_detail_defaults_to_empty() -> None:
    assert scheme.mop_label("fullgen", scheme.ROLE_LAUNCHER) == "mop:fullgen:launcher"


def test_fields_are_stripped() -> None:
    assert scheme.mop_label("  fullgen ", " extension ", " resume ") == "mop:fullgen:extension:resume"


@pytest.mark.parametrize("program", ["", "   "])
def test_empty_program_rejected(program: str) -> None:
    with pytest.raises(ValueError, match="program"):
        scheme.mop_label(program, scheme.ROLE_EXTENSION)


@pytest.mark.parametrize("role", ["", "   "])
def test_empty_role_rejected(role: str) -> None:
    with pytest.raises(ValueError, match="role"):
        scheme.mop_label("fullgen", role)


@pytest.mark.parametrize(
    ("program", "role", "detail"),
    [
        ("full:gen", "extension", ""),
        ("fullgen", "ext:ension", ""),
        ("fullgen", "extension", "re:sume"),
    ],
)
def test_colon_in_field_rejected(program: str, role: str, detail: str) -> None:
    with pytest.raises(ValueError, match="must not contain ':'"):
        scheme.mop_label(program, role, detail)


def test_role_constants() -> None:
    assert scheme.ROLE_EXTENSION == "extension"
    assert scheme.ROLE_SUPERVISOR == "supervisor"
    assert scheme.ROLE_CAPSULE == "capsule"
    assert scheme.ROLE_WORKER == "worker"
    assert scheme.ROLE_ADOPTER == "adopter"
    assert scheme.ROLE_LAUNCHER == "launcher"
