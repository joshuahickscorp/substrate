#!/usr/bin/env python

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_BIN = Path(sys.executable).parent
PY = str(ENV_BIN / "python")
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
    ok, _ = _run([str(ENV_BIN / "ruff"), "check", "src", "tests", "scripts"])
    step("ruff lint", ok)
    ok, _ = _run([str(ENV_BIN / "ruff"), "format", "--check", "src", "tests", "scripts"])
    step("ruff format", ok)
    ok, tail = _run([str(ENV_BIN / "mypy")])
    step("mypy types", ok, tail)

    from mop import config
    from mop.evidence import canonical_sha256
    from mop.experiments import REGISTRY
    from mop.harness import validate
    from mop.studio.profiles import PROFILES

    expected = {"mop_cm7_min_objective_probe", "mop_cm8_custom_jepa_pilot"}
    step("historical experiment registry", set(REGISTRY) == expected)
    cm7 = json.loads((ROOT / "proof/CUSTOM_SUBSTRATE_PILOT.json").read_text())
    cm7_promotion = cm7["authoritative_promotion"]
    chain_ok = all(
        hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
        for item in cm7["receipt_chain"].values()
    )
    step(
        "CM7 sealed null",
        cm7["complete"]
        and chain_ok
        and cm7_promotion["verdict"] == "not-promoted"
        and not cm7_promotion["cm7_local_objective_lever_promotable"],
    )
    cm8 = json.loads((ROOT / "proof/CUSTOM_SUBSTRATE_CM8_PREFLIGHT.json").read_text())
    step("CM8 closed descendant", not cm8["scientific_execution_ready"] and not cm8["scientific_promotion"])
    sanpo = json.loads((ROOT / "proof/SANPO_DR1_CM1_ATTRIBUTE_MAP.json").read_text())
    sessions, pairs = sanpo["development_sessions"], sanpo["candidate_factor_pairs"]
    gate_column = pairs["fields"].index("dr1_gate_met_at_smoke_scale")
    legacy = sanpo["legacy_aggregate"]
    legacy_bytes = subprocess.check_output(
        ["git", "show", f"{legacy['tag']}:proof/SANPO_DR1_CM1_ATTRIBUTE_MAP.json"], cwd=ROOT
    )
    legacy_blob = subprocess.check_output(
        ["git", "rev-parse", f"{legacy['tag']}:proof/SANPO_DR1_CM1_ATTRIBUTE_MAP.json"],
        cwd=ROOT,
        text=True,
    ).strip()
    legacy_body = json.loads(legacy_bytes)
    reconstructed = {k: v for k, v in sanpo.items() if k not in {"complete", "legacy_aggregate"}}
    reconstructed.update(schema=legacy_body["schema"], all_ok=sanpo["complete"])
    reconstructed["dr1_cm1_verdict"] = reconstructed.pop("authoritative_verdict")
    for name, table in (("development_sessions", sessions), ("candidate_factor_pairs", pairs)):
        reconstructed[name] = [dict(zip(table["fields"], row, strict=True)) for row in table["rows"]]
    current_bindings = reconstructed["bindings"]
    current_binding_paths = {
        "intake_receipt_sha256": ROOT / "proof/SANPO_REAL_SMOKE_INTAKE.json",
        "bridge_preflight_sha256": ROOT / "proof/SANPO_CUSTOM_SUBSTRATE_BRIDGE_PREFLIGHT.json",
    }
    current_bindings_ok = all(
        hashlib.sha256(path.read_bytes()).hexdigest() == current_bindings[name]
        for name, path in current_binding_paths.items()
    )
    reconstructed["bindings"] = legacy_body["bindings"]
    step(
        "SANPO compact attribute authority",
        sanpo["complete"]
        and len(sessions["rows"]) == 8
        and all(len(row) == len(sessions["fields"]) for row in sessions["rows"])
        and len(pairs["rows"]) == 36
        and all(len(row) == len(pairs["fields"]) and not row[gate_column] for row in pairs["rows"])
        and not sanpo["authoritative_verdict"]["any_pair_meets_dr1_min_per_cell"]
        and current_bindings_ok
        and reconstructed == legacy_body
        and (len(legacy_bytes), len(legacy_bytes.decode().splitlines())) == (legacy["bytes"], legacy["lines"])
        and hashlib.sha256(legacy_bytes).hexdigest() == legacy["sha256"]
        and legacy_blob == legacy["git_blob"],
    )
    cfg = config.compose(["device=cpu"])

    identity = canonical_sha256({"experiments": sorted(REGISTRY), "seed": int(cfg.seed)})
    step("evidence identity", len(identity) == 64 and identity == identity.lower())
    step("configuration validation", validate.check_all() == [])
    step("Studio profiles", {"m3pro-local-max", "studio-m1ultra"} <= set(PROFILES))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== ACCEPTANCE: {passed}/{len(results)} checks passed ====")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
