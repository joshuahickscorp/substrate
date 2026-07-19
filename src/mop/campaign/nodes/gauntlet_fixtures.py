"""Disposable node runners for the prelaunch chaos gauntlet.

These exist only to exercise the orchestrator's failure and invariance handling on disposable run roots.
``ok_runner`` seals a deterministic artifact; ``failing_runner`` always raises so worker-failure handling
can be proven; ``counting_runner`` records how many times it executed so double-execution can be detected.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

from ..runners import NodeContext, RunResult, register_runner


@register_runner("gauntlet.ok")
def ok_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """A deterministic node: identical seed yields an identical seal (scheduling/width invariance)."""

    content = {
        "schema": "mop-gauntlet-ok/v1",
        "node_id": ctx.node_id,
        "seed": ctx.seed,
        "value": (ctx.seed % 997) * 3 + 1,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(str(path), seal, "ok", is_null=False, detail={})


@register_runner("gauntlet.failing")
def failing_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Always raises, so the fleet must fail the node per policy without accepting a partial artifact."""

    raise RuntimeError("gauntlet induced worker failure")


@register_runner("gauntlet.counting")
def counting_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Append one line to a shared counter file each execution, so double-execution is detectable."""

    counter = ctx.proof_root.parent / "exec_count.txt"
    counter.parent.mkdir(parents=True, exist_ok=True)
    with open(counter, "a", encoding="utf-8") as handle:
        handle.write(f"{ctx.node_id}\n")
    content = {
        "schema": "mop-gauntlet-counting/v1",
        "node_id": ctx.node_id,
        "seed": ctx.seed,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(str(path), seal, "ok", is_null=False, detail={})
