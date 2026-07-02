"""Leg 11D: negative-result registry. The doctrine demands every experiment declare a null and
that "it just did not work" is never an acceptable answer: a refuted lever must land in a named
taxonomy slot. This leg runs the CHEAPEST predicted-null ablation already baked into each
experiment's run() (no new computation, just compose + run at toy scale) and reads the
null-related booleans straight out of the result dict.

Per experiment we record: the declared null_hypothesis (off the Experiment class attr), the
predicted_null (the corpus expectation that on synthetic toy latents the lever ties its baseline),
the observed boolean(s), a verdict, and the negative-result taxonomy slot (EXPERIMENTS.md, 10
categories). Verdict semantics, uniform across experiments:
  confirmed = the null held: NO lever-beats-baseline boolean fired (the predicted negative result).
  refuted   = ALL the lever-beats-baseline booleans fired (the lever cleanly won).
  mixed     = some fired and some did not (partial / inconsistent signal).
For E1 the GATE inverts this: the gate PASSING (forgetting then retention) refutes the
no-retention-gap null, so the lever-beats-baseline signal is gate.passed.

Synthetic latents from make_task_stream => every entry is tagged provisional. Each experiment is
isolated: a failed run is surfaced as a 'error' verdict, never swallowed.
"""

from __future__ import annotations

from ..config import REPO_ROOT, compose
from ..experiments import get_experiment
from ..harness.runner import run_experiment

# Toy overrides per experiment: tiny dims/streams/budgets so every run is sub-second on cpu while
# still exercising the real null check. Knob paths are the experiment's own cfg.experiment.* keys.
_STREAM = [
    "experiment.stream.dim=32",
    "experiment.stream.n_tasks=3",
    "experiment.stream.samples_per_task=48",
    "experiment.stream.classes_per_task=3",
]
TOY: dict[str, list[str]] = {
    "e1_baseline": _STREAM + ["experiment.train.epochs_per_task=2", "experiment.head.hidden=16"],
    "e2_replay": _STREAM + ["experiment.train.epochs_per_task=2", "experiment.head.hidden=16"],
    "e3_plasticity": _STREAM
    + [
        "experiment.train.epochs_per_task=2",
        "experiment.head.hidden=16",
        "experiment.fisher.hidden=16",
        "experiment.fisher.checkpoints=4",
        "experiment.fisher.steps_per_ckpt=4",
    ],
    "e4_neuromod": [
        "experiment.dim=24",
        "experiment.n=64",
        "experiment.ensemble_size=3",
        "experiment.hidden=24",
        "experiment.steps=40",
        "experiment.batch=32",
        "experiment.cal_epochs=20",
        "experiment.cal_hidden=16",
    ],
    "e7_sparse": _STREAM
    + [
        "experiment.hidden=16",
        "experiment.epochs_per_task=8",
        "experiment.n_experts=2",
        "experiment.kwta_k=4",
    ],
    "e8_dendritic": [
        "experiment.dim=24",
        "experiment.n_tasks=3",
        "experiment.samples_per_task=60",
        "experiment.classes_per_task=3",
        "experiment.hidden=8",
        "experiment.branches=2",
        "experiment.seeds=[0,1]",
    ],
    "e9_local": [
        "experiment.dim=24",
        "experiment.n_classes=4",
        "experiment.samples=80",
        "experiment.hidden=16",
        "experiment.passes=2",
        "experiment.seeds=[0,1]",
    ],
    "i4_backprop_alts": [
        "experiment.dim=24",
        "experiment.n_classes=4",
        "experiment.samples=80",
        "experiment.hidden=16",
        "experiment.epochs=40",
        "experiment.seeds=[0,1]",
    ],
}

# Which result-dict keys are the "lever beats its baseline" booleans for each experiment, plus the
# negative-result taxonomy slot the refuted lever lands in (EXPERIMENTS.md categories 1-10). E1 is
# special-cased: its single signal is gate.passed (gate passing REFUTES the no-retention-gap null).
SIGNALS: dict[str, tuple[tuple[str, ...], int, str]] = {
    "e2_replay": (
        ("replay_beats_noreplay", "prioritized_beats_random"),
        5,
        "task too easy / stream too short for buffer content to matter",
    ),
    "e3_plasticity": (
        ("staged_beats_both", "fisher_rise_then_fall"),
        2,
        "frozen substrate has no sensitivity window to shape (needs unfrozen encoder)",
    ),
    "e4_neuromod": (
        ("point_error_chases_noise_INV", "disagreement_ignores_noise"),
        8,
        "epistemic-vs-aleatoric only separable with an ensemble/distributional head",
    ),
    "e7_sparse": (
        ("sparse_beats_dense",),
        4,
        "any interference reduction was just capacity (parameter-matched dense ties)",
    ),
    "e8_dendritic": (
        ("dendritic_beats_mlp",),
        4,
        "dendritic analogy adds complexity, no benefit at matched params",
    ),
    "e9_local": (
        ("any_local_within_margin", "best_local_memory_win"),
        4,
        "local rules trail backprop accuracy and buy no memory win online",
    ),
    "i4_backprop_alts": (
        ("__rules_within_margin",),
        4,
        "no alternative rule reaches within margin of backprop at matched budget",
    ),
}


def _bool_signals(eid: str, out: dict) -> dict[str, bool]:
    """Pull the lever-beats-baseline booleans named in SIGNALS out of the run() result. A few keys
    need a derived value: E4 point-error CHASING noise means the gate failed to separate (so the
    lever-win is point_error NOT chasing); I4 exposes a list not a bool."""
    keys, _, _ = SIGNALS[eid]
    sig: dict[str, bool] = {}
    for k in keys:
        if k == "point_error_chases_noise_INV":
            sig["point_error_separates_noise"] = bool(not out["point_error_chases_noise"])
        elif k == "__rules_within_margin":
            sig["any_rule_within_margin"] = bool(len(out["rules_within_margin"]) > 0)
        else:
            sig[k] = bool(out[k])
    return sig


def _verdict(beats: dict[str, bool]) -> str:
    vals = list(beats.values())
    if all(vals):
        return "refuted"
    if any(vals):
        return "mixed"
    return "confirmed"


def negative_registry(toy: bool = True) -> dict:
    """Run each predicted-null ablation at toy scale and record its verdict + taxonomy slot."""
    entries: list[dict] = []
    summary = {"confirmed": 0, "refuted": 0, "mixed": 0, "error": 0}

    for eid in TOY:
        exp = get_experiment(eid)
        null_hyp = exp.null_hypothesis
        ov = TOY[eid] if toy else []
        try:
            out = run_experiment(compose([f"experiment={eid}", "device=cpu", "seed=0"] + ov))
        except Exception as ex:  # surface, never swallow: a broken ablation is itself a finding
            summary["error"] += 1
            entries.append(
                {
                    "experiment": eid,
                    "null_hypothesis": null_hyp,
                    "predicted_null": True,
                    "observed": {},
                    "verdict": "error",
                    "error": str(ex),
                    "taxonomy_category": None,
                    "taxonomy_label": None,
                    "provenance": "provisional",
                }
            )
            continue

        if eid == "e1_baseline":
            # The gate PASSING (naive forgets, protected retains, both learn) refutes the
            # no-retention-gap null. A failed gate at toy scale = null confirmed; which leaf
            # (task too easy vs latent not decodable) follows from why it failed.
            g = out["gate"]
            beats = {"gate_passed": bool(g["passed"])}
            verdict = _verdict(beats)
            if g["passed"]:
                cat, label = 5, "retention gap present (lever works): null refuted"
            elif not g["both_learn_last_task"]:
                cat, label = 3, "target not decodable from the frozen latent (no learning)"
            else:
                cat, label = 5, "tasks too easy: no forgetting to protect against"
            observed = {
                **beats,
                "naive_bwt": g["naive_bwt"],
                "protected_bwt": g["protected_bwt"],
                "both_learn_last_task": bool(g["both_learn_last_task"]),
            }
        else:
            beats = _bool_signals(eid, out)
            verdict = _verdict(beats)
            _, cat, label = SIGNALS[eid]
            observed = dict(beats)

        summary[verdict] += 1
        entries.append(
            {
                "experiment": eid,
                "null_hypothesis": null_hyp,
                "predicted_null": True,  # these are the corpus's predicted-null ablations
                "observed": observed,
                "verdict": verdict,
                "taxonomy_category": cat,
                "taxonomy_label": label,
                "provenance": "provisional",
            }
        )

    return {"entries": entries, "summary": summary, "provenance": "provisional", "toy": bool(toy)}


def to_markdown(reg: dict) -> str:
    """Render the registry as a markdown table + summary line."""
    s = reg["summary"]
    rows = [
        "# Leg 11D negative-result registry (provisional, synthetic toy latents)",
        "",
        f"Summary: confirmed={s['confirmed']} refuted={s['refuted']} mixed={s['mixed']} "
        f"error={s.get('error', 0)}",
        "",
        "| Exp | Verdict | Taxonomy | Observed (lever-beats-baseline) | Null hypothesis |",
        "|---|---|---|---|---|",
    ]
    for e in reg["entries"]:
        obs = ", ".join(f"{k}={v}" for k, v in e["observed"].items() if isinstance(v, bool)) or "(error)"
        cat = "" if e["taxonomy_category"] is None else f"{e['taxonomy_category']} {e['taxonomy_label']}"
        null = e["null_hypothesis"].replace("\n", " ")
        rows.append(f"| {e['experiment']} | {e['verdict']} | {cat} | {obs} | {null} |")
    return "\n".join(rows) + "\n"


def main() -> dict:
    reg = negative_registry(toy=True)
    out_dir = REPO_ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "negative_registry.md").write_text(to_markdown(reg))
    return reg
