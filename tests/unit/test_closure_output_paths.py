from pathlib import Path

import pytest
from scripts import close_e7_sparse, close_ex2_planning, close_ex5_local_rules


@pytest.mark.parametrize(
    ("module", "filename"),
    (
        (close_e7_sparse, "close_e7_sparse.json"),
        (close_ex5_local_rules, "close_ex5_local_rules.json"),
        (close_ex2_planning, "close_ex2_planning.json"),
    ),
)
def test_closure_driver_default_output_is_repo_anchored(module, filename):
    expected = module.REPO / "runs" / "pre_studio" / filename
    assert module._parse_args([]).out == expected
    assert expected == module.DEFAULT_OUT


@pytest.mark.parametrize(
    "module",
    (close_e7_sparse, close_ex5_local_rules, close_ex2_planning),
)
def test_closure_driver_accepts_explicit_output(module, tmp_path):
    requested = tmp_path / f"{module.__name__.split('.')[-1]}.json"
    parsed = module._parse_args(["--out", str(requested)])
    assert parsed.out == Path(requested)
