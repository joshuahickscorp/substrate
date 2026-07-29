"""Injected-defect mutation suite for the Genesis II collection boundary."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from substrate import genesis2_config as C2
from substrate import genesis2_io as IO2

Detector = Callable[[Mapping[str, Any]], bool]


def _false(field: str) -> Detector:
    return lambda document: document.get(field) is False


def validate_envelope(document: Mapping[str, Any]) -> bool:
    return bool(
        document.get("enforced")
        and (document.get("pressure_changed") or document.get("behaviour_changed") or document.get("capability_degraded_or_reallocated"))
        and document.get("classification") == "binding"
    )


def validate_mechanisms(document: Mapping[str, Any]) -> bool:
    required = set(document.get("required", ()))
    changed = {name for name, count in document.get("state_changes", {}).items() if int(count) > 0}
    licensed = set(document.get("licensed_unnecessary", ()))
    return required <= changed | licensed


def validate_checkpoint(document: Mapping[str, Any]) -> bool:
    durable = document.get("durable", {})
    return isinstance(durable, Mapping) and "microstore" in durable and "scheduling_policy" in durable and document.get("anchored") is True


def validate_receipt(document: Mapping[str, Any]) -> bool:
    body = dict(document)
    supplied = body.pop("sha256", None)
    return supplied == IO2.digest(body) and bool(body.get("source_digest"))


def validate_main_guard(document: Mapping[str, Any]) -> bool:
    source = str(document.get("source", ""))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True
    return False


def _sealed_target_clean(document: Mapping[str, Any]) -> bool:
    return not any(int(value) >= 1_000_000 for value in document.get("material_values", ()))


def _write_accounting_clean(document: Mapping[str, Any]) -> bool:
    return int(document.get("reported_bytes", -1)) >= int(document.get("measured_bytes", 0))


def _bandwidth_clean(document: Mapping[str, Any]) -> bool:
    return int(document.get("candidate_writes", -1)) == int(document.get("comparator_writes", -2))


def _s2_precision_clean(document: Mapping[str, Any]) -> bool:
    return document.get("s2_representation") == "exact"


def _ablation_clean(document: Mapping[str, Any]) -> bool:
    return document.get("disabled_path_called") is False and document.get("alias_path_called") is False


def _topology_clean(document: Mapping[str, Any]) -> bool:
    return document.get("operation") in C2.MACRO_TOPOLOGY_OPERATIONS and not any(int(value) >= 1_000_000 for value in document.get("stored_values", ()))


def _consolidation_clean(document: Mapping[str, Any]) -> bool:
    return document.get("representation") == "typed_rule" and document.get("copied_outputs") is False


def _procedure_clean(document: Mapping[str, Any]) -> bool:
    return bool(document.get("retired_after_failures"))


def _shadow_clean(document: Mapping[str, Any]) -> bool:
    return document.get("read_future_authoritative_state") is False


def _activation_clean(document: Mapping[str, Any]) -> bool:
    return not IO2.contains_true_activation(document)


def _pair(
    name: str,
    detector: Detector,
    clean: dict[str, Any],
    mutation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    injected = deepcopy(clean)
    mutation(injected)
    clean_accepted = bool(detector(clean))
    injected_rejected = not bool(detector(injected))
    return {
        "mutation": name,
        "clean_accepted": clean_accepted,
        "injected_rejected": injected_rejected,
        "detector_pass": clean_accepted and injected_rejected,
        "clean_digest": IO2.digest(clean),
        "injected_digest": IO2.digest(injected),
        "activation": False,
    }


def run() -> dict[str, Any]:
    clean_receipt = {"source_digest": "source", "payload_digest": "payload", "activation": False}
    clean_receipt["sha256"] = IO2.digest(clean_receipt)

    cases: dict[str, tuple[Detector, dict[str, Any], Callable[[dict[str, Any]], None]]] = {
        "associative_store_reads_target_answers": (
            _sealed_target_clean,
            {"material_values": [0, 1, 7]},
            lambda row: row["material_values"].append(1_000_001),
        ),
        "write_cost_is_undercounted": (
            _write_accounting_clean,
            {"measured_bytes": 64, "reported_bytes": 64},
            lambda row: row.update(reported_bytes=8),
        ),
        "candidate_receives_more_write_bandwidth": (
            _bandwidth_clean,
            {"candidate_writes": 8_192, "comparator_writes": 8_192},
            lambda row: row.update(candidate_writes=16_384),
        ),
        "s2_is_artificially_precision_limited": (
            _s2_precision_clean,
            {"s2_representation": "exact"},
            lambda row: row.update(s2_representation="ternary"),
        ),
        "nonbinding_envelope_reported_as_binding": (
            validate_envelope,
            {
                "enforced": True,
                "pressure_changed": True,
                "behaviour_changed": True,
                "capability_degraded_or_reallocated": True,
                "classification": "binding",
            },
            lambda row: row.update(
                pressure_changed=False,
                behaviour_changed=False,
                capability_degraded_or_reallocated=False,
            ),
        ),
        "mechanism_registered_but_never_consumed": (
            validate_mechanisms,
            {
                "required": list(C2.REQUIRED_MECHANISMS),
                "state_changes": {name: 1 for name in C2.REQUIRED_MECHANISMS},
                "licensed_unnecessary": [],
            },
            lambda row: row["state_changes"].pop("shadow_field"),
        ),
        "ablation_bypassed_through_alias": (
            _ablation_clean,
            {"disabled_path_called": False, "alias_path_called": False},
            lambda row: row.update(alias_path_called=True),
        ),
        "topology_stores_answers_instead_of_structure": (
            _topology_clean,
            {"operation": "split_concept", "stored_values": [1, 7]},
            lambda row: row["stored_values"].append(1_000_001),
        ),
        "consolidation_copies_outputs": (
            _consolidation_clean,
            {"representation": "typed_rule", "copied_outputs": False},
            lambda row: row.update(representation="output_table", copied_outputs=True),
        ),
        "precision_promotion_reads_outcomes": (
            _false("reads_evaluator_outcome"),
            {"reads_evaluator_outcome": False},
            lambda row: row.update(reads_evaluator_outcome=True),
        ),
        "procedure_loses_accuracy": (
            _procedure_clean,
            {"retired_after_failures": True},
            lambda row: row.update(retired_after_failures=False),
        ),
        "shadow_field_reads_future_authoritative_state": (
            _shadow_clean,
            {"read_future_authoritative_state": False},
            lambda row: row.update(read_future_authoritative_state=True),
        ),
        "checkpoint_omits_microstore": (
            validate_checkpoint,
            {
                "durable": {"microstore": {}, "scheduling_policy": "conditional"},
                "anchored": True,
            },
            lambda row: row["durable"].pop("microstore"),
        ),
        "checkpoint_omits_scheduler": (
            validate_checkpoint,
            {
                "durable": {"microstore": {}, "scheduling_policy": "conditional"},
                "anchored": True,
            },
            lambda row: row["durable"].pop("scheduling_policy"),
        ),
        "collector_accepts_unanchored_receipts": (
            validate_receipt,
            clean_receipt,
            lambda row: row.update(source_digest=""),
        ),
        "multiprocessing_child_recursively_launches_main": (
            validate_main_guard,
            {"source": 'def main():\n    return 0\n\nif __name__ == "__main__":\n    main()\n'},
            lambda row: row.update(source="def main():\n    return 0\n\nmain()\n"),
        ),
        "activation_becomes_true": (
            _activation_clean,
            {"activation": False},
            lambda row: row.update(activation=bool(1)),
        ),
    }
    if tuple(cases) != C2.MUTATIONS:
        raise RuntimeError("mutation implementation order differs from the frozen constitution")

    rows = [_pair(name, *cases[name]) for name in C2.MUTATIONS]
    survivors = [row["mutation"] for row in rows if not row["injected_rejected"]]
    clean_failures = [row["mutation"] for row in rows if not row["clean_accepted"]]
    return {
        "mutations": rows,
        "injected": len(rows),
        "survivors": survivors,
        "clean_failures": clean_failures,
        "all_pass": not survivors and not clean_failures and all(row["detector_pass"] for row in rows),
        "policy": C2.MUTATION_POLICY,
        "activation": False,
    }


def source_main_guard_report(paths: Sequence[Path]) -> dict[str, Any]:
    rows = {str(path): validate_main_guard({"source": path.read_text()}) for path in paths}
    return {"files": rows, "all_pass": all(rows.values()), "activation": False}


def demo() -> None:
    report = run()
    assert report["all_pass"], report
    print("genesis2 mutation self-check passed: 17/17 injected defects rejected")


if __name__ == "__main__":
    demo()
