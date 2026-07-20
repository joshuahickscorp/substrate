"""Secondary cross domain matrix on the third valid domain.

The principal matrix pairs a sensor domain with an audio domain: two sequence domains that do not obviously
share temporal structure. If shared fast dynamics fail there, one honest reading is that the pair was wrong,
not the premise. This matrix tests the premise on a pair that does share temporal structure: Speech Commands
keyword identity and a continuous stream of keywords from one speaker.

Secondary by declaration. It can support or fail to support the premise. It cannot overturn the principal
verdict, and a positive here would be reported as a scope condition, not as a cross modality result.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from fastforge import engine as E
from fastforge import sequence as S
from fastforge.runs import io

SEEDS = list(range(8))
DIRECTIONS = [("speech", "speech_stream"), ("speech_stream", "speech")]
ARMS = [
    "fresh_independent",
    "separate_per_domain",
    "lstm_gdumb",
    "G0_always_trainable",
    "G1_frozen_after_first",
    "G4_adapters_only",
    "H_always_update",
    "H_cosine_gate",
]
BASELINES = ["fresh_independent", "separate_per_domain", "lstm_gdumb"]


def shard(direction, seed):
    out = {}
    for arm in ARMS:
        t0 = time.time()
        out[arm] = S.run(arm, direction, seed)
        print(
            f"  {direction[0]}->{direction[1]} seed{seed} {arm:26s} "
            f"acq2={out[arm]['metrics']['second_acquisition']:.3f} "
            f"rec={out[arm]['metrics']['return_recovery']:.3f} [{time.time() - t0:.0f}s]",
            flush=True,
        )
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "shard":
        d, s = sys.argv[2].split(":")
        res = shard(tuple(d.split("|")), int(s))
        p = io.RUNS / "secondary"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{d.replace('|', '-')}_{s}.json").write_text(json.dumps(res, default=str))
        print("SHARD_DONE", d, s, flush=True)
        return

    t0 = time.time()
    per_dir = {}
    for a, b in DIRECTIONS:
        dname = f"{a}->{b}"
        rows = {}
        for s in SEEDS:
            f = io.RUNS / "secondary" / f"{a}-{b}_{s}.json"
            if f.is_file():
                rows[s] = json.loads(f.read_text())
        if len(rows) < len(SEEDS):
            print(f"secondary waiting: {dname} {len(rows)} of {len(SEEDS)}", flush=True)
            return
        seeds = sorted(rows)
        base_params = float(np.mean([rows[s]["lstm_gdumb"]["metrics"]["params"] for s in seeds]))

        def util(arm, s, rows=rows, base_params=base_params):
            m = rows[s][arm]["metrics"]
            return float(
                np.mean([m["second_acquisition"], m["return_recovery"], m["future_adaptation"]])
                - 0.05 * (m["params"] / base_params)
            )

        um = {arm: float(np.mean([util(arm, s) for s in seeds])) for arm in ARMS}
        strongest = max(BASELINES, key=lambda x: um[x])
        keys = (
            "first_acquisition",
            "second_acquisition",
            "first_retention",
            "return_recovery",
            "future_adaptation",
            "negative_transfer",
        )
        means = {
            arm: {k: round(float(np.mean([rows[s][arm]["metrics"][k] for s in seeds])), 4) for k in keys}
            for arm in ARMS
        }
        effects = {
            arm: E.effect([util(arm, s) for s in seeds], [util(strongest, s) for s in seeds]) for arm in ARMS
        }
        floors = {
            k: means[strongest][k] - 0.02
            for k in ("second_acquisition", "first_retention", "return_recovery", "future_adaptation")
        }
        passing = [
            arm
            for arm in ARMS
            if effects[arm]["lower_95_cb"] >= io.SESOI and all(means[arm][k] >= v for k, v in floors.items())
        ]
        per_dir[dname] = {
            "seeds": seeds,
            "strongest_matched_baseline": strongest,
            "utility": {k: round(v, 4) for k, v in um.items()},
            "means": means,
            "effects": effects,
            "component_floors": {k: round(v, 4) for k, v in floors.items()},
            "arms_passing_all_conditions": passing,
            "verdict": "secondary_positive" if passing else "secondary_null",
        }
    both = (
        sorted(
            set(per_dir[f"{a}->{b}"]["arms_passing_all_conditions"])
            & set(per_dir[f"{b}->{a}"]["arms_passing_all_conditions"])
            for a, b in [DIRECTIONS[0]]
        )[0]
        if len(per_dir) == 2
        else []
    )
    io.seal(
        "MOP_FAST_STATE_SECONDARY_MATRIX.json",
        {
            "schema": "mop-fast-state-secondary-matrix/v1",
            "pairing_rationale": "two beds that share temporal structure and modality, unlike the principal "
            "sensor to audio pairing",
            "status": "secondary. It cannot overturn the principal verdict and a positive here is a scope "
            "condition, not a cross modality result.",
            "seeds": SEEDS,
            "arms": ARMS,
            "per_direction": per_dir,
            "arms_passing_in_both_directions": both,
            "verdict": "secondary_positive" if both else "secondary_null",
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    for d, v in per_dir.items():
        print(
            f"  {d}: {v['verdict']} strongest {v['strongest_matched_baseline']} "
            f"passing {v['arms_passing_all_conditions']}",
            flush=True,
        )
    print("SECONDARY_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
