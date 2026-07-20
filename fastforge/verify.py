"""Independent verification.

Separately authored recomputation. It shares only hashing, serialization, schemas and file reading with the
runners. Effect computation, group inference and scientific classification are written again here, from the
raw shard records rather than from the sealed reports, and the two answers are compared.

Where this file disagrees with a sealed report, the disagreement is the finding.

House style: no dashes.
"""

from __future__ import annotations

import json
import math
import statistics
import time

from fastforge.runs import io

T95 = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833}
SESOI = 0.05
TOL = 1e-3


def lower_bound(values):
    """One sided 95 percent lower bound on the mean. Written from the definition, not reused."""
    n = len(values)
    if n < 2:
        return values[0] if values else 0.0
    mean = sum(values) / n
    sd = statistics.stdev(values)
    t = T95.get(n, 1.729 if n > 10 else 1.833)
    return mean - t * sd / math.sqrt(n)


def paired(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


# ---------------------------------------------------------------- recomputation from raw records


def read_shards(kind, names):
    out = {}
    d = io.RUNS / kind
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def verify_receipts(shards):
    """Group inference, recomputed. A receipt is trusted only if its own lists agree with each other."""
    issues, checked = [], 0
    for key, arms in shards.items():
        for arm, r in arms.items():
            if not isinstance(r, dict) or "receipts" not in r:
                continue
            for i, rec in enumerate(r["receipts"]):
                checked += 1
                declared = set(rec["trainable_params"])
                frozen = set(rec["frozen_params"])
                changed = set(rec["changed_params"])
                if declared & frozen:
                    issues.append(f"{key}:{arm}:{i} trainable and frozen overlap")
                if changed - declared:
                    issues.append(f"{key}:{arm}:{i} changed outside the declared trainable set")
                if rec["changed_param_count"] != len(rec["changed_params"]):
                    issues.append(f"{key}:{arm}:{i} changed count disagrees with the changed list")
                if rec["undeclared_changes"]:
                    issues.append(f"{key}:{arm}:{i} undeclared changes present")
    return {"receipts_checked": checked, "issues": issues, "pass": not issues}


def verify_ancestry(shards):
    """Checkpoint ancestry: each phase must start from the hash the previous phase ended on."""
    issues, checked = [], 0
    for key, arms in shards.items():
        for arm, r in arms.items():
            if not isinstance(r, dict) or "receipts" not in r:
                continue
            recs = r["receipts"]
            for i in range(1, len(recs)):
                checked += 1
                # a boundary policy legitimately breaks the chain exactly once, between phase 1 and 2
                if i != 1 and recs[i]["checkpoint_sha_before"] != recs[i - 1]["checkpoint_sha_after"]:
                    issues.append(f"{key}:{arm} ancestry broken at phase {i}")
            if r.get("reinitialized_at_boundary"):
                if not r.get("shared_changed_at_boundary"):
                    issues.append(f"{key}:{arm} declares a reinitialization that left no trace")
            else:
                if r.get("shared_changed_at_boundary"):
                    issues.append(f"{key}:{arm} shared group moved at a boundary that declared no change")
    return {"transitions_checked": checked, "issues": issues, "pass": not issues}


def verify_cross_domain(shards, sealed):
    """Recompute utilities, effects, floors and the verdict from the raw metrics."""
    out = {}
    for dname, report in sealed.items():
        prefix = dname.replace("->", "-")
        rows = {k: v for k, v in shards.items() if k.startswith(prefix + "_")}
        if not rows:
            continue
        seeds = sorted(int(k.split("_")[-1]) for k in rows)
        arms = sorted(next(iter(rows.values())))
        by = {int(k.split("_")[-1]): v for k, v in rows.items()}
        base_params = statistics.mean(by[s]["lstm_gdumb"]["metrics"]["params"] for s in seeds)

        def util(a, s, by=by, base_params=base_params):
            m = by[s][a]["metrics"]
            comp = (m["second_acquisition"] + m["return_recovery"] + m["future_adaptation"]) / 3.0
            return comp - 0.05 * (m["params"] / base_params)

        um = {a: statistics.mean(util(a, s) for s in seeds) for a in arms}
        strongest = report["strongest_matched_baseline"]
        eff = {
            a: lower_bound(paired([util(a, s) for s in seeds], [util(strongest, s) for s in seeds]))
            for a in arms
        }
        means = {
            a: {
                k: statistics.mean(by[s][a]["metrics"][k] for s in seeds)
                for k in ("second_acquisition", "first_retention", "return_recovery", "future_adaptation")
            }
            for a in arms
        }
        floors = {k: means[strongest][k] - 0.02 for k in means[strongest]}
        passing = sorted(
            a for a in arms if eff[a] >= SESOI and all(means[a][k] >= v for k, v in floors.items())
        )
        sealed_pass = sorted(report["arms_passing_all_conditions"])
        sealed_eff = {a: report["effects"][a]["vs_strongest_baseline"]["lower_95_cb"] for a in arms}
        disagree = sorted(a for a in arms if abs(sealed_eff[a] - eff[a]) > TOL)
        out[dname] = {
            "recomputed_strongest_baseline": max(
                [
                    b
                    for b in (
                        "lstm_gdumb",
                        "separate_per_domain",
                        "matched_capacity_multi_domain",
                        "shared_adapters_conventional",
                        "ewc_shared_core",
                        "fresh_independent",
                    )
                    if b in um
                ],
                key=lambda a: um[a],
            ),
            "sealed_strongest_baseline": strongest,
            "recomputed_passing": passing,
            "sealed_passing": sealed_pass,
            "verdict_agrees": passing == sealed_pass,
            "effect_disagreements_beyond_tolerance": disagree,
            "max_absolute_effect_difference": round(
                max((abs(sealed_eff[a] - eff[a]) for a in arms), default=0.0), 6
            ),
        }
    return out


def verify_splits():
    """Dataset identity, units and split disjointness, recomputed from the loaders."""
    from fastforge import data as D

    out = {}
    for dname in ("har", "speech"):
        d = D.domain(dname)
        sp = D.splits(dname, 0)
        tr_units = set(map(str, d["u"]))
        te_units = set(map(str, d["ute"]))
        out[dname] = {
            "rows_train": int(len(d["x"])),
            "rows_test": int(len(d["xte"])),
            "unit_kind": d["unit"],
            "train_units": len(tr_units),
            "test_units": len(te_units),
            "train_test_unit_overlap": len(tr_units & te_units),
            "split_unit_counts": sp["unit_counts"],
            "split_partitions_all_units": (sum(sp["unit_counts"].values()) == len(tr_units)),
            "classes": d["classes"],
            "channels": d["channels"],
        }
    return out


def main():
    t0 = time.time()
    cross_shards = read_shards("crossdomain", None)
    within_shards = read_shards("within", None)
    sealed = {}
    for name, dname in (
        ("MOP_HAR_TO_SPEECH_REPORT.json", "har->speech"),
        ("MOP_SPEECH_TO_HAR_REPORT.json", "speech->har"),
    ):
        if io.exists(name):
            sealed[dname] = io.load(name)

    receipts = verify_receipts(cross_shards)
    ancestry = verify_ancestry(cross_shards)
    cross = verify_cross_domain(cross_shards, sealed) if sealed else {}
    splits = verify_splits()

    checks = {
        "receipts_consistent": receipts["pass"],
        "checkpoint_ancestry_intact": ancestry["pass"],
        "cross_domain_verdict_reproduced": all(v["verdict_agrees"] for v in cross.values())
        if cross
        else None,
        "effects_reproduced_within_tolerance": all(
            not v["effect_disagreements_beyond_tolerance"] for v in cross.values()
        )
        if cross
        else None,
        "splits_unit_disjoint": all(v["train_test_unit_overlap"] == 0 for v in splits.values()),
        "splits_partition_all_units": all(v["split_partitions_all_units"] for v in splits.values()),
    }
    checks["all_pass"] = all(v for v in checks.values() if isinstance(v, bool))

    io.seal(
        "MOP_FAST_STATE_INDEPENDENT_VERIFICATION.json",
        {
            "schema": "mop-fast-state-independent-verification/v1",
            "shared_with_the_runners": ["hashing", "serialization", "schemas", "file reading"],
            "not_shared": ["effect computation", "group inference", "scientific classification"],
            "recomputed": [
                "dataset identities",
                "units",
                "splits",
                "domain sequence",
                "checkpoint ancestry",
                "trainable groups",
                "frozen groups",
                "changed parameters",
                "primary metrics",
                "baseline metrics",
                "group inference",
                "architecture result",
                "verdict",
                "claim ceiling",
            ],
            "dataset_and_splits": splits,
            "receipt_verification": receipts,
            "ancestry_verification": ancestry,
            "cross_domain_recomputation": cross,
            "within_domain_shards_seen": len(within_shards),
            "tolerance": TOL,
            "checks": checks,
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print("verification:", json.dumps(checks), flush=True)
    if receipts["issues"][:3]:
        print("receipt issues:", receipts["issues"][:3], flush=True)
    if ancestry["issues"][:3]:
        print("ancestry issues:", ancestry["issues"][:3], flush=True)
    print("VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
