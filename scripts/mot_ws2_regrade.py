#!/usr/bin/env python
"""WS2 regrade (ZERO compute): re-score the fusion tournament under the PREREGISTERED DUAL contract that
the original run silently weakened into an OR.

The registry contract for WS2 (mot_ws2_fusion_tournament.py docstring) states the metric is the PAIR
(heldout_acc_delta_vs_concat, nll_delta_vs_concat) and the null is "at matched capacity every structured
fusion TIES the concat-MLP on held-out accuracy AND NLL". The original scorer, however, rejected the null
when acc_win OR nll_win cleared (line: any_win = any_win or acc_win or nll_win). Under that OR rule
gwt_broadcast rejected the null on accuracy alone while LOSING on NLL, exactly the over-claim the audit
flagged for al2 and ws2.

This regrade loads the existing runs/mot/ws2_fusion_tournament_seeds10.json (no model is built or trained)
and enforces the honest DUAL AND rule:

  An arm rejects the null ONLY IF it beats concat-MLP on BOTH accuracy AND NLL, where "beats" for each
  metric means the MEAN per-seed paired delta has a 95 percent CI (seed_ci, normal approx over the MEAN
  of the per-seed deltas, NOT a best-of-K max over arms or seeds) that excludes zero from below, with no
  per-seed sign flip on that metric.

The MEAN-baseline guard: the win statistic is the CI on the MEAN of the paired deltas. We never take a
max over arms or a best seed. If a null is rejected here it is because a single named arm cleared both
metrics on the seed-averaged paired contrast, which is what the contract asked for.

Output: runs/mot/ws2_fusion_regrade.json with a per-arm dual-metric table (acc delta CI, nll improvement
CI, per-metric sign-flip, per-metric pass, and dual_pass = acc_pass AND nll_pass) and the corrected
verdict. clears_own_control is true only if some arm's dual_pass is genuinely true.

No em dashes or en dashes (BLACKHOLE.md).

Usage: python scripts/mot_ws2_regrade.py
       python scripts/mot_ws2_regrade.py --in runs/mot/ws2_fusion_tournament_seeds10.json \
           --out runs/mot/ws2_fusion_regrade.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from devsys.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402

DEFAULT_IN = "runs/mot/ws2_fusion_tournament_seeds10.json"
DEFAULT_OUT = "runs/mot/ws2_fusion_regrade.json"
BASELINE = "concat_mlp"


def _paired_deltas(per_seed: list[dict], arm: str) -> tuple[list[float], list[float]]:
    """Per-seed paired contrasts of `arm` against the concat baseline, in the same orientation the
    contract scores as a WIN (larger is better):
      acc_delta = arm.acc - concat.acc           (arm more accurate -> positive)
      nll_improvement = concat.nll - arm.nll      (arm lower NLL -> positive)
    Both are paired within seed so seed variance cancels before the CI is taken."""
    acc_d, nll_d = [], []
    for row in per_seed:
        sc = row["scores"]
        acc_d.append(sc[arm]["acc"] - sc[BASELINE]["acc"])
        nll_d.append(sc[BASELINE]["nll"] - sc[arm]["nll"])
    return acc_d, nll_d


def _metric_block(deltas: list[float]) -> dict:
    """CI on the MEAN of the per-seed paired deltas plus the sign-flip report. A metric passes only if
    the CI excludes zero from below (lo > 0) AND the per-seed signs are consistently positive (no flip).
    This is the mean-baseline guard: the pass hinges on the seed-averaged contrast, never a best seed."""
    ci = seed_ci(deltas)
    flip = sign_flip_report(deltas)
    ci_excludes_zero = ci["lo"] > 0
    consistent_positive = flip["consistent_sign"] == 1
    passed = bool(ci_excludes_zero and consistent_positive and not ci["unstable"])
    return {
        "per_seed_delta": [round(float(d), 4) for d in deltas],
        "ci": ci,
        "sign_flip": flip,
        "ci_excludes_zero_below": ci_excludes_zero,
        "sign_consistent_positive": consistent_positive,
        "pass": passed,
    }


def regrade(data: dict) -> dict:
    per_seed = data["per_seed"]
    seeds = data.get("seeds", [r["seed"] for r in per_seed])
    arms = sorted(k for k in per_seed[0]["scores"] if k != BASELINE)

    table: dict[str, dict] = {}
    any_dual_pass = False
    winners: list[str] = []
    for arm in arms:
        acc_d, nll_d = _paired_deltas(per_seed, arm)
        acc_block = _metric_block(acc_d)
        nll_block = _metric_block(nll_d)
        dual_pass = bool(acc_block["pass"] and nll_block["pass"])
        any_dual_pass = any_dual_pass or dual_pass
        if dual_pass:
            winners.append(arm)
        table[arm] = {
            "acc_delta_vs_concat": acc_block,
            "nll_improvement_vs_concat": nll_block,
            "acc_pass": acc_block["pass"],
            "nll_pass": nll_block["pass"],
            "dual_pass": dual_pass,
            "note": _arm_note(arm, acc_block["pass"], nll_block["pass"]),
        }

    null_rejected = any_dual_pass
    if null_rejected:
        verdict = (
            "NULL REJECTED (DUAL): arm(s) "
            + ", ".join(winners)
            + " beat concat-MLP on BOTH accuracy and NLL at matched capacity with no sign flip; "
            "structure is more than param count"
        )
    else:
        verdict = (
            "NULL (honest): no structured fusion clears the preregistered DUAL contract. Under the "
            "original OR rule gwt_broadcast rejected on accuracy alone while its NLL improvement CI "
            "includes zero, so that rejection does not survive the AND contract. Structure buys nothing "
            "beyond param count on both metrics jointly."
        )

    return {
        "experiment": "ws2_fusion_regrade",
        "source": DEFAULT_IN,
        "regrade_contract": {
            "id": "ws2_fusion_regrade",
            "metric": ["heldout_acc_delta_vs_concat", "nll_delta_vs_concat"],
            "rule": (
                "DUAL AND: an arm rejects the null only if BOTH the mean per-seed acc delta and the mean "
                "per-seed NLL improvement have a 95 percent CI excluding zero from below with no per-seed "
                "sign flip. Mean-baseline guard: the win statistic is the CI on the MEAN of the paired "
                "deltas, never a best-of-K max over arms or seeds."
            ),
            "baseline": data.get("contract", {}).get("baseline", "unstructured concat-MLP at matched params"),
            "fixes": (
                "original scorer used acc_win OR nll_win (any_win); this enforces the preregistered "
                "acc_win AND nll_win"
            ),
        },
        "seeds": seeds,
        "reference_params": data.get("reference_params"),
        "table": table,
        "winners_dual": winners,
        "null_rejected": null_rejected,
        "clears_own_control": bool(any_dual_pass),
        "verdict": verdict,
    }


def _arm_note(arm: str, acc_pass: bool, nll_pass: bool) -> str:
    if acc_pass and nll_pass:
        return "clears BOTH metrics: genuine dual win"
    if acc_pass and not nll_pass:
        return "acc-only win; NLL improvement CI includes zero, fails the dual contract"
    if nll_pass and not acc_pass:
        return "nll-only win; accuracy delta CI includes zero, fails the dual contract"
    return "ties concat on both metrics"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WS2 dual-contract regrade (zero compute)")
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    in_path = Path(a.in_path)
    if not in_path.is_absolute():
        in_path = _ROOT / in_path
    data = json.loads(in_path.read_text())
    result = regrade(data)

    out_path = Path(a.out_path)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    summary = {
        "table": {
            arm: {
                "acc_pass": t["acc_pass"],
                "nll_pass": t["nll_pass"],
                "dual_pass": t["dual_pass"],
                "acc_ci": t["acc_delta_vs_concat"]["ci"],
                "nll_ci": t["nll_improvement_vs_concat"]["ci"],
            }
            for arm, t in result["table"].items()
        },
        "winners_dual": result["winners_dual"],
        "null_rejected": result["null_rejected"],
        "clears_own_control": result["clears_own_control"],
        "verdict": result["verdict"],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
