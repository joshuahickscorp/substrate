#!/usr/bin/env python

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
results: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def _run(cmd: list[str]) -> tuple[bool, str]:
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (result.stdout + result.stderr).strip().splitlines()[-1:] or [""]
    return result.returncode == 0, tail[0]


def main() -> int:
    ok, tail = _run([PY, "-m", "pytest", "-q"])
    step("full test suite", ok, tail)
    ok, _ = _run([str(ROOT / ".venv/bin/ruff"), "check", "src", "tests", "scripts"])
    step("ruff lint", ok)
    ok, _ = _run([str(ROOT / ".venv/bin/ruff"), "format", "--check", "src", "tests", "scripts"])
    step("ruff format", ok)
    ok, tail = _run([str(ROOT / ".venv/bin/mypy")])
    step("mypy types", ok, tail)

    from mop import config, devices
    from mop.diagnostics import linear_probe, noisy_tv_diagnostic
    from mop.experiments import REGISTRY, get_experiment
    from mop.harness import validate
    from mop.studio.profiles import PROFILES
    from mop.substrate.datasets import make_task_stream

    step(
        "experiment registry minimal",
        set(REGISTRY) == {"mop_cm7_min_objective_probe", "mop_cm8_custom_jepa_pilot"},
    )
    cfg = config.compose(["experiment=mop_cm7_min_objective_probe", "device=cpu"])
    step("CM7 config composes", cfg.experiment.id == "mop_cm7_min_objective_probe")
    step("CM7 experiment resolves", get_experiment("mop_cm7_min_objective_probe").id == cfg.experiment.id)

    dev = devices.resolve("cpu")
    task = make_task_stream(
        n_tasks=1, dim=32, classes_per_task=4, samples_per_task=300, separation=3.0
    )[0]
    probe_ok = linear_probe(task.x, task.y)["decodable"]
    noisy = noisy_tv_diagnostic(dim=40, device=dev, steps=250)
    step(
        "diagnostics",
        bool(probe_ok and noisy["noise_error_stays_high"] and noisy["epistemic_collapses_on_noise"]),
    )
    step("configuration validation", validate.check_all() == [])
    step("Studio profiles", {"m3pro-local-max", "studio-m1ultra"} <= set(PROFILES))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== ACCEPTANCE: {passed}/{len(results)} checks passed ====")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
