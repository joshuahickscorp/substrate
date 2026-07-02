#!/usr/bin/env python
"""Definition-of-done acceptance. Runs the full suite, lint, types, the E1 gate, every
diagnostic, the I4 comparison, the campaign queue dry-run, and one toy Tier C leg, then
prints a PASS/FAIL summary and exits nonzero on any failure. The human reads this after the
unattended run.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
results: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def _run(cmd: list[str]) -> tuple[bool, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
    return p.returncode == 0, tail[0]


def main() -> int:
    ok, tail = _run([PY, "-m", "pytest", "-q"])
    step("full test suite", ok, tail)
    ok, _ = _run([str(ROOT / ".venv/bin/ruff"), "check", "src", "tests", "scripts"])
    step("ruff lint", ok)
    ok, _ = _run([str(ROOT / ".venv/bin/ruff"), "format", "--check", "src", "tests", "scripts"])
    step("ruff format", ok)
    ok, tail = _run([str(ROOT / ".venv/bin/mypy")])
    step("mypy types", ok, tail)

    # in-process functional checks
    from mop import config, devices
    from mop.experiments import REGISTRY, get_experiment
    from mop.harness.queue import run_queue

    dev = devices.resolve("cpu")
    tmp = Path(tempfile.mkdtemp())

    try:
        cfg = config.compose(
            [
                "experiment=e1_baseline",
                "device=cpu",
                "experiment.stream.dim=48",
                "experiment.stream.samples_per_task=200",
                "shell.buffer.index=brute",
                "shell.consolidation.method=ewc",
                "shell.consolidation.ewc_lambda=1000.0",
            ]
        )
        out = get_experiment("e1_baseline").run(cfg, dev, tmp / "e1")
        step("E1 gate (forget then retain)", bool(out["gate"]["passed"]), str(out["gate"]))
    except Exception as e:
        step("E1 gate (forget then retain)", False, repr(e))

    try:
        from mop.diagnostics import linear_probe, noisy_tv_diagnostic
        from mop.substrate.datasets import make_task_stream

        t = make_task_stream(n_tasks=1, dim=32, classes_per_task=4, samples_per_task=300, separation=3.0)[0]
        lp = linear_probe(t.x, t.y)["decodable"]
        nt = noisy_tv_diagnostic(dim=40, device=dev, steps=250)
        ok = lp and nt["noise_error_stays_high"] and nt["epistemic_collapses_on_noise"]
        step("diagnostics (linear-probe + noisy-TV)", bool(ok))
    except Exception as e:
        step("diagnostics (linear-probe + noisy-TV)", False, repr(e))

    try:
        cfg = config.compose(
            [
                "experiment=i4_backprop_alts",
                "device=cpu",
                "experiment.samples=240",
                "experiment.epochs=120",
                "experiment.seeds=[0,1]",
            ]
        )
        out = get_experiment("i4_backprop_alts").run(cfg, dev, tmp / "i4")
        step("I4 comparison table", len(out["table"]) == 7 and (tmp / "i4" / "i4_table.md").exists())
    except Exception as e:
        step("I4 comparison table", False, repr(e))

    try:
        plan = run_queue(dry_run=True, enabled_tiers={"C"})
        step("campaign queue dry-run", len(plan["planned"]) >= 1, f"{len(plan['planned'])} legs planned")
    except Exception as e:
        step("campaign queue dry-run", False, repr(e))

    try:
        ran = run_queue(dry_run=False, enabled_tiers={"C"}, toy=True, max_runs_per_leg=1, max_legs=1)
        step("one toy Tier C leg", len(ran.get("results", {})) >= 1, str(ran.get("results")))
    except Exception as e:
        step("one toy Tier C leg", False, repr(e))

    step("experiment registry populated", len(REGISTRY) >= 11, f"{len(REGISTRY)} experiments")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== ACCEPTANCE: {passed}/{len(results)} checks passed ====")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
