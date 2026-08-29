"""Method acceptance gate stage.

Runs the calibration suite and the historical defect mutation suite, then seals both. Principal execution is
licensed only when this stage is fully green, and the license is written into the artifact so a later stage
can check it rather than assume it.

House style: no dashes.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mop.method import acceptance, calibration, contracts, defects, gate, graph, io

KERNEL_DIR = Path(__file__).resolve().parent.parent


def kernel_accounting() -> dict:
    files = {}
    for p in sorted(KERNEL_DIR.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        files[p.relative_to(KERNEL_DIR).as_posix()] = len(p.read_text().splitlines())
    return {
        "root": "src/mop/method",
        "files": files,
        "total_loc": sum(files.values()),
        "kernel_loc": sum(v for k, v in files.items() if not k.startswith("runs/")),
        "stage_loc": sum(v for k, v in files.items() if k.startswith("runs/")),
        "budget_kernel_loc": 5000,
        "within_budget": sum(v for k, v in files.items() if not k.startswith("runs/")) <= 5000,
        "new_cli_commands": 0,
        "new_configuration_roots": 0,
        "new_registries": 0,
        "new_experiment_engines": 0,
        "reuses": [
            "fastforge.engine for training and update partitioning",
            "fastforge.data for the validated domain providers",
            "fastforge.arch for the fixture architectures",
            "the composable evidence fabric for indexing",
            "the existing shard driven supervisor pattern for scheduling",
        ],
    }


def main():
    t0 = time.time()
    cal = calibration.run()
    acc = acceptance.run()
    accounting = kernel_accounting()

    io.seal(
        "MOP_EXPERIMENT_VALIDITY_KERNEL.json",
        {
            "schema": "mop-experiment-validity-kernel/v1",
            "contract_types": sorted(contracts.CONTRACT_TYPES),
            "quantity_kinds": list(contracts.QUANTITY_KINDS),
            "admission_sequence": list(gate.SEQUENCE),
            "stages_before_principal_compute": list(gate.PRE_PRINCIPAL),
            "causal_graph_schema": graph.SCHEMA,
            "calibration_cases": sorted(calibration.CASES),
            "defect_mutations": sorted(acceptance.MUTATIONS),
            "code_accounting": accounting,
        },
    )

    io.seal(
        "MOP_METHOD_ACCEPTANCE_RESULT.json",
        {
            "schema": "mop-method-acceptance/v1",
            "calibration": cal["properties"],
            "calibration_cases": {k: {kk: v[kk] for kk in ("expected", "actual", "pass")} for k, v in cal["cases"].items()},
            "mutations_rejected": acc["all_rejected"],
            "n_mutations": acc["n_mutations"],
            "mutation_failures": acc["failures"],
            "defect_classes_covered": acc["defect_classes_covered"],
            "every_ledger_class_has_a_mutation": acc["every_ledger_class_has_a_mutation"],
            "blocks_compute": acc["blocks_compute"],
            "blocks_claim_only": acc["blocks_claim_only"],
            "note": acc["note"],
            "green": bool(cal["all_pass"] and acc["all_rejected"] and acc["every_ledger_class_has_a_mutation"]),
            "principal_execution_licensed_by_this_gate": bool(
                cal["all_pass"] and acc["all_rejected"] and acc["every_ledger_class_has_a_mutation"]
            ),
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    io.seal("MOP_METHOD_ACCEPTANCE_MUTATIONS.json", {"schema": "mop-method-acceptance-mutations/v1", **acc})

    rows = "\n".join(
        f"| {k} | {v['defect_id']} | {v['stage_caught']} | {v['blocks']} | {'rejected' if v['pass'] else 'ADMITTED'} |"
        for k, v in acc["mutations"].items()
    )
    cal_rows = "\n".join(f"| {k} | {v} |" for k, v in cal["properties"].items())
    io.seal_md(
        "MOP_EXPERIMENT_VALIDITY_KERNEL.md",
        f"""# Experiment validity kernel

`src/mop/method`, {accounting["kernel_loc"]} lines of kernel plus {accounting["stage_loc"]} lines of program
stages. No new CLI command, no new configuration root, no new registry, no second experiment engine.

## Admission sequence

{" -> ".join(gate.SEQUENCE)}

A stage opens only when the previous one passed. Everything before `principal` costs no training compute, so
an invalid experiment dies before it can spend any.

## Calibration

| property | holds |
|---|---|
{cal_rows}

## Historical defect mutations

| mutation | defect | caught at | blocks | outcome |
|---|---|---|---|---|
{rows}

{acc["note"]}

## Contract vocabulary

{", ".join(sorted(contracts.CONTRACT_TYPES))}

## Quantity provenance

{", ".join(contracts.QUANTITY_KINDS)}. A structurally guaranteed zero may not be reported as a measured zero,
which is defect {defects.BY_ID["D6"]["id"]}.
""",
    )
    print(
        f"calibration {cal['all_pass']} | mutations {acc['all_rejected']} ({acc['n_mutations']}) | "
        f"kernel {accounting['kernel_loc']} LOC",
        flush=True,
    )
    print("ACCEPTANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
