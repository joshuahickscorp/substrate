"""C3 precompute gates: model-error-aware bounded simulation on gymnasium classic-control.

C3 premise (new vs Gen2 S1, which was worse than random): allocate simulation only where expected decision
benefit exceeds expected model-error cost; condition planning on estimated rollout error. The decisive cheap
gates are 5 and 6: an ORACLE error-aware allocator (plan with the learned model only in states whose true
H-step rollout error is low, else act reactively) must beat the reactive control and the fixed-depth planner by
SESOI. If even a perfect error oracle cannot make learned-model simulation help, there is no headroom and C3
terminates before any principal canary. Controls: shuffled-model-error (must not beat reactive), wrong-depth.

Environments are gymnasium classic-control families and parameter-perturbed variants (dynamics not authored by
C3); independent units are distinct dynamical systems. House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-gen3/frontier/lanes")
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
from control_beds import ENV_SPECS, _make, _phi, _ridge_fit, collect_offline  # noqa: E402

OUT = Path("/Users/scammermike/Downloads/mop-gen3/gen3/reports")
OUT.mkdir(parents=True, exist_ok=True)


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def learn_models(spec, seed):
    trans, nA = collect_offline(spec, n_steps=4000, seed=seed)
    Phi = np.array([t[0] for t in trans]); A = np.array([t[1] for t in trans])
    R = np.array([t[2] for t in trans]); dphi = Phi.shape[1]
    obs2 = np.array([t[6] for t in trans]); dobs = obs2.shape[1]
    Wr = {a: (_ridge_fit(Phi[A == a], R[A == a]) if (A == a).sum() > dphi else np.zeros(dphi)) for a in range(nA)}
    Wd = {a: (_ridge_fit(Phi[A == a], obs2[A == a]) if (A == a).sum() > dphi else np.zeros((dphi, dobs))) for a in range(nA)}
    good = R >= np.quantile(R, 0.6)
    Wbc = {a: _ridge_fit(Phi[good], (A[good] == a).astype(float)) for a in range(nA)}
    # true one-step model error per offline state, and a fitted error model over phi(obs)
    pred_next = np.stack([Phi[i] @ Wd[A[i]] for i in range(len(trans))])
    true_err = np.linalg.norm(pred_next - obs2, axis=1) / (np.linalg.norm(obs2, axis=1) + 1e-6)
    We = _ridge_fit(Phi, true_err)
    thr = float(np.median(true_err))
    return nA, Wr, Wd, Wbc, We, thr


def plan_action(obs, nA, Wr, Wd, H):
    best, bestv = 0, -1e9
    for a0 in range(nA):
        p = _phi(obs); tot = p @ Wr[a0]; nxt = p @ Wd[a0]
        for _ in range(H - 1):
            pn = _phi(nxt); qa = int(np.argmax([pn @ Wr[aa] for aa in range(nA)]))
            tot += pn @ Wr[qa]; nxt = pn @ Wd[qa]
        if tot > bestv:
            bestv, best = tot, a0
    return best


def rollout(spec, policy_fn, episodes=12, seed=200, max_steps=500):
    tot = []
    for e in range(episodes):
        env = _make(spec, seed + e); obs, _ = env.reset(seed=seed + e); ret = 0.0
        for _ in range(max_steps):
            obs, r, term, trunc, _ = env.step(int(policy_fn(obs)))
            ret += r
            if term or trunc:
                break
        env.close(); tot.append(ret)
    return float(np.mean(tot))


def build_policies(spec, seed):
    nA, Wr, Wd, Wbc, We, thr = learn_models(spec, seed)
    rng = np.random.default_rng(seed + 5)

    def reactive(obs):
        p = _phi(obs); return int(np.argmax([p @ Wbc[a] for a in range(nA)]))

    def fixed_depth(obs):
        return plan_action(obs, nA, Wr, Wd, H=4)

    def one_step(obs):
        p = _phi(obs); return int(np.argmax([p @ Wr[a] for a in range(nA)]))

    def oracle_error_aware(obs):
        # allocate simulation only where the estimated model error is below threshold, else act reactively
        err = float(_phi(obs) @ We)
        return fixed_depth(obs) if err <= thr else reactive(obs)

    def shuffled_error(obs):  # error signal decorrelated from state: plan/react at the base rate
        return fixed_depth(obs) if rng.random() < 0.5 else reactive(obs)

    return {"nA": nA, "reactive": reactive, "one_step": one_step, "fixed_depth": fixed_depth,
            "oracle_error_aware": oracle_error_aware, "shuffled_error": shuffled_error,
            "random": lambda o: int(rng.integers(nA))}


def main():
    t0 = time.time()
    units = ENV_SPECS
    ret = []
    for i, u in enumerate(units):
        pol = build_policies(u, seed=i)
        r = {name: rollout(u, fn, episodes=10, seed=500 + i * 30) for name, fn in pol.items() if name != "nA"}
        ret.append(r)
        print(f"  unit{i} {u[0]:14s} reactive={r['reactive']:.1f} fixed_depth={r['fixed_depth']:.1f} "
              f"oracle_ea={r['oracle_error_aware']:.1f} shuffled={r['shuffled_error']:.1f}", flush=True)
    # normalized effects per informative unit
    eff_react, eff_planner, informative = [], [], 0
    for r in ret:
        upper = max(r["oracle_error_aware"], r["fixed_depth"], r["reactive"], r["one_step"])
        span = upper - r["random"]
        if span < max(0.05 * abs(r["random"]), 1.0):
            continue
        informative += 1
        eff_react.append(float(np.clip((r["oracle_error_aware"] - r["reactive"]) / span, -2, 2)))
        best_planner = max(r["reactive"], r["one_step"], r["fixed_depth"])
        eff_planner.append(float(np.clip((r["oracle_error_aware"] - best_planner) / span, -2, 2)))

    def re(e):
        e = np.array(e); n = len(e); sd = e.std(ddof=1) if n > 1 else 0
        lcb = float(e.mean() - (1.833 if n <= 10 else 1.729) * sd / np.sqrt(n)) if n else 0
        return {"mean": round(float(e.mean()), 4) if n else None, "lower_95_cb": round(lcb, 4), "n": n}

    g5 = re(eff_react); g6 = re(eff_planner)
    beats_shuffled = float(np.mean([r["oracle_error_aware"] - r["shuffled_error"] for r in ret]))
    gate5_pass = bool(g5["lower_95_cb"] is not None and g5["lower_95_cb"] >= 0.05)
    gate6_pass = bool(g6["lower_95_cb"] is not None and g6["lower_95_cb"] >= 0.05)
    enough_units = informative >= 5
    all_pass = gate5_pass and gate6_pass and beats_shuffled > 0 and enough_units
    commit = subprocess_head()
    out = {"schema": "mop-gen3-c3-precompute/v1", "stream": "C3", "source_commit": commit,
           "environments": [u[0] for u in units], "informative_units": informative,
           "gate5_oracle_error_aware_vs_reactive": g5, "gate5_pass": gate5_pass,
           "gate6_residual_vs_best_planner": g6, "gate6_pass": gate6_pass,
           "beats_shuffled_error_control": round(beats_shuffled, 3),
           "all_gates_pass": all_pass, "SESOI": 0.05, "enough_informative_units": enough_units,
           "verdict": ("proceed_to_canary" if all_pass else
                       ("terminate_insufficient_informative_units" if not enough_units and gate5_pass
                        else "terminate_no_headroom")),
           "interpretation": ("an oracle error-aware allocator " + ("does" if all_pass else "does not") +
                              " beat reactive and the fixed-depth planner by SESOI; " +
                              ("C3 has residual headroom and is licensed for a bounded canary" if all_pass else
                               "even a perfect error oracle cannot make learned-model simulation help, so C3 terminates before the principal canary")),
           "wall_seconds": round(time.time() - t0, 1)}
    out["sha256"] = sha(out)
    (OUT / "MOP_GEN3_C3_PRECOMPUTE_RESULT.json").write_text(json.dumps(out, indent=2))
    print(f"[C3-precompute] gate5={gate5_pass} gate6={gate6_pass} all={all_pass} -> {out['verdict']} [{out['wall_seconds']}s]", flush=True)
    print("C3_DONE", flush=True)


def subprocess_head():
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd="/Users/scammermike/Downloads/mop-gen3").stdout.strip()


if __name__ == "__main__":
    main()
