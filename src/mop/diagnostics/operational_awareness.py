"""Operational-awareness diagnostics: the OA1-OA8 suite of OPERATIONAL_AWARENESS.md.

Operational awareness is measurable self-and-world management: missing-form detection, confidence
calibration, memory availability, mode selection, compute-value estimation, crisis detection,
rewrite caution, and report grounding. This is structured self-MONITORING (diagnostics over run
receipts), NOT self-awareness, consciousness, or any mental-state claim: rendered text is gated by
north_star.assert_no_sentience_claims, same as the metacognition report. Pure torch and python,
deterministic, nothing trains. Composes riskcov/calibration primitives instead of duplicating them.

Recorded priors these metrics must respect (OPERATIONAL_AWARENESS.md section 4): the test-time
compute lane closed with matched-compute nulls, the trained router lost on the real cache, and
uncertainty gating chased noise. An OA score is reported next to its baseline, never alone.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .riskcov import auroc, ece_equal_mass

OA_SCHEMA = "mop-operational-awareness/v1"

OA_COMPONENTS = (
    "oa1_missing_form",
    "oa2_calibration",
    "oa3_memory_availability",
    "oa4_mode_selection",
    "oa5_compute_value",
    "oa6_crisis_detection",
    "oa7_rewrite_caution",
    "oa8_report_grounding",
)


def _as1d(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return t.detach().float().flatten()


def missing_form_detection(scores, absent) -> dict:
    """OA1: do detector scores rank truly absent/unreliable forms above present ones (AUROC)."""
    s, a = _as1d(scores), _as1d(absent)
    if s.shape != a.shape:
        raise ValueError(f"scores {tuple(s.shape)} != absence labels {tuple(a.shape)}")
    return {
        "auroc": auroc(s, a),
        "absent_rate": float(a.mean()),
        "chance": 0.5,
    }


def confidence_calibration(confidences, correct) -> dict:
    """OA2: does confidence predict correctness (equal-mass ECE, lower better; AUROC, higher)."""
    c, y = _as1d(confidences), _as1d(correct)
    return {
        "ece": ece_equal_mass(c, y),
        "auroc": auroc(c, y),
        "accuracy": float(y.mean()),
    }


def memory_availability(claimed, payoff, *, threshold: float = 0.0) -> dict:
    """OA3: does the claimed-availability signal rank retrievals that actually paid off higher."""
    c, p = _as1d(claimed), _as1d(payoff)
    useful = (p > threshold).float()
    return {
        "auroc": auroc(c, useful),
        "useful_rate": float(useful.mean()),
        "chance": 0.5,
    }


def mode_selection(chosen, oracle, random_baseline, fixed_baseline=None) -> dict:
    """OA4: per-episode scores of the selected mode vs oracle, random, and fixed routing.

    Regret is oracle minus chosen (0 is perfect). The deltas restate the doctrine baselines: a
    selector that does not beat random and fixed routing has no selection competence, whatever
    its architecture (the EX-ROUTER-DENSITY lesson).
    """
    ch, orc, rnd = _as1d(chosen), _as1d(oracle), _as1d(random_baseline)
    if not (ch.shape == orc.shape == rnd.shape):
        raise ValueError("chosen, oracle, and random score vectors must align per episode")
    if bool((orc < ch - 1.0e-6).any()):
        raise ValueError("oracle scores below chosen scores; the oracle must upper-bound selection")
    out = {
        "regret_vs_oracle": float((orc - ch).mean()),
        "delta_vs_random": float((ch - rnd).mean()),
        "oracle_gap_report": float((orc - rnd).mean()),
    }
    if fixed_baseline is not None:
        out["delta_vs_fixed"] = float((ch - _as1d(fixed_baseline)).mean())
    return out


def compute_value(continue_scores, marginal_gains, *, step_cost: float = 0.0) -> dict:
    """OA5: do continue-computing decisions rank the steps whose marginal gain beat the cost."""
    s, g = _as1d(continue_scores), _as1d(marginal_gains)
    worth_it = (g > step_cost).float()
    return {
        "auroc": auroc(s, worth_it),
        "worth_it_rate": float(worth_it.mean()),
        "chance": 0.5,
    }


def crisis_detection(crisis_scores, failed, raw_error=None) -> dict:
    """OA6: does the crisis score predict realized substrate failure beyond the raw error signal."""
    s, f = _as1d(crisis_scores), _as1d(failed)
    out = {"auroc": auroc(s, f), "failure_rate": float(f.mean()), "chance": 0.5}
    if raw_error is not None:
        raw = auroc(_as1d(raw_error), f)
        out["raw_error_auroc"] = raw
        out["auroc_over_raw_error"] = out["auroc"] - raw
    return out


def rewrite_caution(triggered_on_noise, triggered_on_real) -> dict:
    """OA7: trigger rates on aleatoric-noise streams vs real-failure streams (noisy-TV guard)."""
    noise, real = _as1d(triggered_on_noise), _as1d(triggered_on_real)
    false_rate, true_rate = float(noise.mean()), float(real.mean())
    return {
        "false_trigger_rate": false_rate,
        "true_trigger_rate": true_rate,
        "caution_margin": true_rate - false_rate,
    }


def report_grounding(reported: Mapping[str, object], traced: Mapping[str, object]) -> dict:
    """OA8: does the self-report match the run trace on the fields both sides state."""
    shared = sorted(set(reported) & set(traced))
    if not shared:
        raise ValueError("report_grounding needs at least one shared field between report and trace")
    mismatched = [k for k in shared if reported[k] != traced[k]]
    return {
        "grounded_fraction": 1.0 - len(mismatched) / len(shared),
        "shared_fields": len(shared),
        "mismatched_fields": mismatched,
    }


def oa_suite(**components: Mapping[str, float] | None) -> dict:
    """Assemble provided OA component results into one suite receipt.

    There is deliberately NO composite awareness score: aggregating OA1-OA8 into one number would
    invite exactly the prose escalation the rail exists to stop. The suite names what was measured
    and what is missing, and each component carries its own baseline.
    """
    unknown = sorted(set(components) - set(OA_COMPONENTS))
    if unknown:
        raise ValueError(f"unknown OA components {unknown}; allowed {OA_COMPONENTS}")
    present = {k: dict(v) for k, v in components.items() if v is not None}
    return {
        "schema": OA_SCHEMA,
        "components": present,
        "components_present": sorted(present),
        "components_missing": sorted(set(OA_COMPONENTS) - set(present)),
    }


def render_oa_md(suite: Mapping, *, level_note: str | None = None) -> str:
    """Render the suite as markdown, gated by the sentience rail (a claim cannot ship)."""
    from ..devel.north_star import assert_no_sentience_claims

    lines = ["## Operational awareness suite", ""]
    lines.append(
        "Structured self-monitoring diagnostics. This report makes no claim of sentience, "
        "consciousness, self-awareness, feelings, or agency."
    )
    lines.append("")
    for name in sorted(suite.get("components", {})):
        metrics = suite["components"][name]
        stated = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
        lines.append(f"- {name}: {stated}")
    missing = suite.get("components_missing", [])
    if missing:
        lines.append(f"- not measured: {', '.join(missing)}")
    if level_note:
        lines.append("")
        lines.append(level_note)
    text = "\n".join(lines)
    assert_no_sentience_claims(text, where="operational_awareness report")
    return text
