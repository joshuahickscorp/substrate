"""Instrument calibration suite.

Before an instrument may measure something unknown, it must classify eleven known things correctly. Each
case is executed, not asserted: a synthetic world with a known answer is built, the real kernel code is run
against it, and the classification is compared to the truth.

The two cases that matter most are the ones the prior program got wrong in both directions: a bed with no
headroom must classify as invalid rather than as a mechanism null, and an underpowered estimator must be
distinguished from a mechanism that does nothing.

House style: no dashes.
"""

from __future__ import annotations

import numpy as np

from mop.method import arms, baseline, bed, controls, mechanism, power

PREREG = power.preregistration(
    name="calibration",
    independent_unit="synthetic unit",
    expected_sd=0.03,
    sesoi=0.05,
    seeds=8,
    units=8,
    max_seeds=8,
    futility=0.01,
    harm=0.05,
)


def _units(effect: float, sd: float, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(effect, sd, n))


def _full_bed(**over) -> dict:
    m = {
        "construct_valid": True,
        "units": {"group_disjoint": True, "test_touched": False, "n_units": 12},
        "leakage": {"clean": True},
        "oracle_headroom": 0.20,
        "residual_headroom_lcb": 0.08,
        "baseline_converged": True,
        "order_necessity": 0.18,
        "intervention_possible": True,
        "seed_stability": 0.01,
    }
    m.update(over)
    return m


# ---------------------------------------------------------------- the eleven cases


def case_known_positive() -> dict:
    d = power.decide(_units(0.12, 0.02, 8, 1), PREREG)
    return {"expected": "positive", "actual": d["verdict"], "detail": d}


def case_known_null() -> dict:
    d = power.decide(_units(0.0, 0.02, 8, 2), PREREG)
    return {"expected": "null", "actual": d["verdict"].replace("null_futile", "null"), "detail": d}


def case_known_harm() -> dict:
    d = power.decide(_units(-0.15, 0.02, 8, 3), PREREG)
    return {"expected": "harm", "actual": d["verdict"], "detail": d}


def case_invalid_bed() -> dict:
    r = bed.classify(_full_bed(construct_valid=False))
    return {"expected": "invalid_no_construct", "actual": r["classification"], "detail": r["checks"]}


def case_no_headroom_bed() -> dict:
    """The bed that must not become a mechanism null. Oracle equals the strongest control."""
    r = bed.classify(_full_bed(oracle_headroom=0.0, residual_headroom_lcb=0.0))
    return {"expected": "invalid_no_headroom", "actual": r["classification"], "detail": r["checks"]}


def case_leakage() -> dict:
    la = bed.leakage_audit([1, 2, 3], [3, 4], "train_and_test")
    r = bed.classify(_full_bed(leakage=la))
    return {
        "expected": "invalid_instrumentation",
        "actual": r["classification"],
        "detail": {"leakage_audit": la},
    }


def _arm(name: str, tag: str) -> dict:
    return arms.record(
        name,
        source=f"impl_{tag}",
        config={"policy": tag},
        call_graph=[f"branch_{tag}"],
        state_transitions=[tag, tag + "1"],
        param_delta={"g": tag},
        memory={"policy": tag},
        resources={"updates": 100, "params": 10 if tag == "a" else 20},
        outputs=[0.1, 0.2] if tag == "a" else [0.3, 0.4],
    )


def case_aliased_arm() -> dict:
    a = _arm("arm_one", "a")
    clone = dict(a)
    clone["name"] = "arm_two"
    r = arms.distinctness([a, clone])
    return {
        "expected": "rejected",
        "actual": "rejected" if not r["all_distinct"] else "accepted",
        "detail": {"aliased_pairs": r["aliased_pairs"]},
    }


def case_inactive_mechanism() -> dict:
    off = {k: 0 for k in mechanism.REQUIRED_MEASUREMENTS}
    on = dict(off)
    on.update({"intervention_count": 0, "affected_samples": 0, "counterfactual_difference": 0.0, "cost": 0.0})
    r = mechanism.activity({"enabled": on, "disabled": off})
    return {
        "expected": "inactive_instrumentation",
        "actual": r["classification"],
        "detail": {"failed": r["failed"]},
    }


def case_active_mechanism() -> dict:
    off = {k: 0 for k in mechanism.REQUIRED_MEASUREMENTS}
    on = {
        "intervention_count": 40,
        "intervention_timing": [1, 5, 9],
        "affected_samples": 256,
        "affected_parameter_groups": ["shared"],
        "affected_state": ["h"],
        "affected_memory": ["buffer"],
        "counterfactual_difference": 0.07,
        "downstream_path": ["shared", "head", "logits"],
        "cost": 1.5,
    }
    r = mechanism.activity({"enabled": on, "disabled": off})
    return {"expected": "active", "actual": r["classification"], "detail": {"failed": r["failed"]}}


def case_underpowered_estimator() -> dict:
    """A real effect of 0.06 buried in noise of 0.25 with three seeds must not read as a mechanism null."""
    d = power.decide(_units(0.06, 0.25, 3, 4), PREREG)
    underpowered = not d["adequately_powered"]
    return {
        "expected": "insufficient_power",
        "actual": "insufficient_power" if underpowered else d["verdict"],
        "detail": d,
    }


def case_unconverged_baseline() -> dict:
    rec = baseline.receipt(
        "still_training",
        identity="lstm_gdumb",
        model="lstm",
        parameters=1000,
        updates=50,
        data_exposure=1000,
        memory=600,
        compute_seconds=1.0,
        validation_curve=[0.30, 0.45, 0.58, 0.69, 0.78],  # monotone rise, never plateaus
        selected_checkpoint="last",
        seed_scores=[0.78, 0.77],
    )
    c = baseline.comparison("treatment_vs_baseline", "treated", rec, "lstm_gdumb")
    return {
        "expected": "provisional",
        "actual": c["status"],
        "detail": {"converged": rec["converged"], "reason": rec["reason"]},
    }


def case_wrong_control() -> dict:
    """An order free control that still reads order. This is defect D1, rebuilt from scratch."""
    import torch
    import torch.nn as nn

    class OrderFreeInName(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(3, 3, kernel_size=5, padding=2)

        def forward(self, x):  # x is (N, T, C)
            return self.conv(x.transpose(1, 2)).mean(-1)

    class ActuallyOrderFree(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)

        def forward(self, x):
            return self.lin(x.mean(1))

    torch.manual_seed(0)
    x = torch.randn(4, 16, 3)
    bad = OrderFreeInName().eval()
    good = ActuallyOrderFree().eval()
    rb = controls.order_free(lambda t: bad(t), x, module=bad)
    rg = controls.order_free(lambda t: good(t), x, module=good)
    ok = (not rb["all_pass"]) and rg["all_pass"]
    return {
        "expected": "wrong_control_rejected_and_real_control_accepted",
        "actual": "wrong_control_rejected_and_real_control_accepted" if ok else "misclassified",
        "detail": {
            "conv_control_pass": rb["all_pass"],
            "conv_control_failures": [k for k, v in rb.items() if v is False],
            "real_control_pass": rg["all_pass"],
        },
    }


CASES = {
    "known_positive": case_known_positive,
    "known_null": case_known_null,
    "known_harm": case_known_harm,
    "invalid_bed": case_invalid_bed,
    "no_headroom_bed": case_no_headroom_bed,
    "leakage_case": case_leakage,
    "aliased_arm_case": case_aliased_arm,
    "inactive_mechanism_case": case_inactive_mechanism,
    "active_mechanism_case": case_active_mechanism,
    "underpowered_estimator_case": case_underpowered_estimator,
    "unconverged_baseline_case": case_unconverged_baseline,
    "wrong_control_case": case_wrong_control,
}


def run() -> dict:
    results, ok = {}, True
    for name, fn in CASES.items():
        r = fn()
        r["pass"] = r["actual"] == r["expected"]
        ok &= r["pass"]
        results[name] = r
    props = {
        "positive_passes": results["known_positive"]["pass"],
        "null_fails_positive_gate": results["known_null"]["detail"]["verdict"] != "positive",
        "harm_triggers_harm_classification": results["known_harm"]["pass"],
        "invalid_bed_does_not_become_mechanism_null": results["invalid_bed"]["actual"].startswith("invalid"),
        "no_headroom_bed_becomes_invalid": results["no_headroom_bed"]["actual"] == "invalid_no_headroom",
        "leakage_is_rejected": results["leakage_case"]["pass"],
        "arm_aliasing_is_rejected": results["aliased_arm_case"]["pass"],
        "inactive_mechanism_is_rejected": results["inactive_mechanism_case"]["pass"],
        "weak_estimator_distinguished_from_mechanism_failure": results["underpowered_estimator_case"]["pass"],
        "unconverged_baseline_blocks_comparison": results["unconverged_baseline_case"]["pass"],
        "wrong_control_is_rejected": results["wrong_control_case"]["pass"],
    }
    props["all_pass"] = all(props.values()) and ok
    return {"cases": results, "properties": props, "all_pass": props["all_pass"]}
