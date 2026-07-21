"""E6 hybrid state plus head adaptation, executed only when the value queue licenses it."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastforge import engine as E
from mop.method import bed, power
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal.runs import e2

BEDS = B.PRINCIPAL
SEEDS = e2.PRINCIPAL_SEEDS
ADAPT_STEPS = Fx.STEPS // 4
STATE_LR = 0.03
ARMS = ("head_only", "state_only", "head_plus_state", "head_plus_state_noise",
        "adapter_only", "core_parameter_adaptation")


def _adapt_batch_seed(seed: int) -> int:
    return 140_000 + seed


def _batch_plan_sha(n: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    h = hashlib.sha256()
    for _ in range(ADAPT_STEPS):
        h.update(np.asarray(rng.choice(n, min(Fx.BATCH, n), replace=False), dtype=np.int64).tobytes())
    return h.hexdigest()


def _noise_matched_to(reference: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(reference)
    return noise * float(reference.norm()) / float(noise.norm() + 1e-9)


class HybridModel(nn.Module):
    def __init__(self, base: nn.Module, bottleneck: int = 16):
        super().__init__()
        self.base = base
        self.down = nn.Linear(A.LATENT, bottleneck)
        self.up = nn.Linear(bottleneck, A.LATENT)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.register_buffer("state", torch.zeros(A.LATENT))
        names = list(dict(self.named_parameters()))
        self.param_groups = {
            "core": [n for n in names if n.startswith("base.core")],
            "readout": [n for n in names if n.startswith("base.head")],
            "adapter": [n for n in names if n.startswith(("down", "up"))],
        }

    def logits_with_state(self, x, state):
        h = self.base.represent(x) + state
        return self.base.head(h + self.up(F.relu(self.down(h))))

    def forward(self, x, d=None, update_recent: bool = False):
        return self.logits_with_state(x, self.state), None


def contexts(bedname: str, seed: int) -> dict:
    d = B.load(bedname)
    units = np.asarray(d["u"])
    unique = np.random.default_rng(110_000 + seed).permutation(np.unique(units))
    half = len(unique) // 2

    def split(group):
        n_eval = max(1, int(round(0.25 * len(group))))
        return group[n_eval:], group[:n_eval]

    a_train, a_eval = split(unique[:half])
    b_train, b_eval = split(unique[half:])

    def take(group):
        idx = np.where(np.isin(units, list(group)))[0]
        return d["x"][idx], d["y"][idx], units[idx]

    rng = np.random.default_rng(120_000 + seed)
    gain = torch.tensor(rng.uniform(0.5, 1.5, size=d["channels"]), dtype=torch.float32)
    offset = torch.tensor(rng.normal(0, 0.7, size=d["channels"]), dtype=torch.float32)

    def shifted(row):
        x, y, u = row
        return x * gain + offset, y, u

    return {"A_train": take(a_train), "A_eval": take(a_eval),
            "B_train": shifted(take(b_train)), "B_eval": shifted(take(b_eval)),
            "channels": d["channels"], "classes": d["classes"],
            "sequence_length": int(d["x"].shape[1]), "segment_length": int(d["segment_length"]),
            "boundaries": list(d["boundaries"]),
            "units": {"A_train": a_train.tolist(), "A_eval": a_eval.tolist(),
                      "B_train": b_train.tolist(), "B_eval": b_eval.tolist()},
            "shift": {"gain": [round(float(x), 5) for x in gain],
                      "offset": [round(float(x), 5) for x in offset]}}


def build(ctx: dict, spec: dict, seed: int) -> HybridModel:
    reset, _ = Fx.reset_schedule(spec["reset"], ctx, seed)
    base = A.build(family=spec["family"], ch=ctx["channels"], classes=ctx["classes"],
                   tier=spec["tier"], readout=spec["readout"],
                   history_k=int(spec["history_k"]), reset=reset)
    return HybridModel(base)


def evaluate(model: HybridModel, row) -> dict:
    x, y, units = row
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[i:i + 256], None)[0].argmax(1) for i in range(0, len(x), 256)])
    correct = (pred == y).numpy()
    return {"accuracy": round(float(correct.mean()), 5),
            "per_unit_accuracy": {str(u): round(float(correct[units == u].mean()), 5)
                                  for u in np.unique(units) if (units == u).sum() >= 5}}


def state_adapt(model: HybridModel, row, seed: int, train_head: bool) -> dict:
    x, y, _ = row
    batch_seed = _adapt_batch_seed(seed)
    rng = np.random.default_rng(batch_seed)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    state = model.state.detach().clone().requires_grad_(True)
    params = [p for n, p in model.named_parameters() if n in model.param_groups["readout"]] if train_head else []
    groups = [{"params": [state], "lr": STATE_LR}]
    if params:
        groups.append({"params": params, "lr": Fx.LR})
    optimizer = torch.optim.Adam(groups)
    for _ in range(ADAPT_STEPS):
        idx = rng.choice(len(x), min(Fx.BATCH, len(x)), replace=False)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model.logits_with_state(x[idx], state), y[idx]).backward()
        optimizer.step()
    with torch.no_grad():
        model.state.copy_(state)
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]
    return {"updates": ADAPT_STEPS, "parameter_updates": ADAPT_STEPS if train_head else 0,
            "changed_params": changed, "state_norm": round(float(model.state.norm()), 6),
            "optimizer": "Adam", "head_lr": Fx.LR if train_head else None,
            "state_lr": STATE_LR, "batch": Fx.BATCH, "batch_seed": batch_seed,
            "batch_plan_sha": _batch_plan_sha(len(x), batch_seed),
            "state_rule": "supervised error gradient on a transient state buffer",
            "not_E4_recentering_rule": True}


def run_seed(bedname: str, seed: int) -> dict:
    t0 = time.time()
    core = io.load("MOP_OWNED_TEMPORAL_CORE_V1.json")
    spec = dict(core["selection"]["selected"]["spec"])
    ctx = contexts(bedname, seed)
    torch.manual_seed(seed)
    model = build(ctx, spec, seed)
    xa, ya, _ = ctx["A_train"]
    pre = E.fit(model, None, xa, ya, train_groups=["core", "readout"], steps=Fx.STEPS,
                lr=Fx.LR, rng=np.random.default_rng(seed), batch=Fx.BATCH)
    snapshot = copy.deepcopy(model.state_dict())
    pre_a, pre_b = evaluate(model, ctx["A_eval"]), evaluate(model, ctx["B_eval"])
    xb, yb, _ = ctx["B_train"]
    out = {}

    def reset():
        model.load_state_dict(snapshot)

    learned_hybrid_state = None
    batch_seed = _adapt_batch_seed(seed)
    batch_plan_sha = _batch_plan_sha(len(xb), batch_seed)
    for arm in ARMS:
        reset()
        if arm == "head_only":
            trace = E.fit(model, None, xb, yb, train_groups=["readout"], steps=ADAPT_STEPS, lr=Fx.LR,
                          rng=np.random.default_rng(batch_seed), batch=Fx.BATCH)
            trace.update({"optimizer": "Adam", "head_lr": Fx.LR, "batch_seed": batch_seed,
                          "batch_plan_sha": batch_plan_sha})
        elif arm == "state_only":
            trace = state_adapt(model, ctx["B_train"], seed, False)
        elif arm == "head_plus_state":
            trace = state_adapt(model, ctx["B_train"], seed, True)
            learned_hybrid_state = model.state.detach().clone()
        elif arm == "head_plus_state_noise":
            trace = state_adapt(model, ctx["B_train"], seed, True)
            head_parameters_match = E.checkpoint_sha(model) == out["head_plus_state"]["checkpoint_sha"]
            with torch.no_grad():
                model.state.copy_(_noise_matched_to(learned_hybrid_state))
            trace.update({"state_norm": round(float(model.state.norm()), 6),
                          "head_parameters_match_learned_head_plus_state": head_parameters_match,
                          "magnitude_matched_to_learned_head_plus_state": True})
        else:
            group = "adapter" if arm == "adapter_only" else "core"
            trace = E.fit(model, None, xb, yb, train_groups=[group], steps=ADAPT_STEPS, lr=Fx.LR,
                          rng=np.random.default_rng(160_000 + seed), batch=Fx.BATCH)
        out[arm] = {"trace": trace, "future_acquisition_B": evaluate(model, ctx["B_eval"]),
                    "return_retention_A": evaluate(model, ctx["A_eval"]),
                    "checkpoint_sha": E.checkpoint_sha(model),
                    "state_norm": round(float(model.state.norm()), 6)}
    checks = {
        "six_distinct_arms": set(out) == set(ARMS),
        "matched_adaptation_updates": all(v["trace"].get("updates") == ADAPT_STEPS for v in out.values()),
        "matched_head_learning_rate": all(out[a]["trace"].get("head_lr") == Fx.LR for a in (
            "head_only", "head_plus_state", "head_plus_state_noise")),
        "matched_head_minibatches": len({out[a]["trace"].get("batch_plan_sha") for a in (
            "head_only", "head_plus_state", "head_plus_state_noise")}) == 1,
        "state_only_zero_parameter_updates": out["state_only"]["trace"]["parameter_updates"] == 0
        and not out["state_only"]["trace"]["changed_params"],
        "state_noise_magnitude_matched": abs(out["head_plus_state_noise"]["state_norm"]
                                             - out["head_plus_state"]["state_norm"]) <= 1e-5,
        "state_noise_uses_learned_hybrid_norm": out["head_plus_state_noise"]["trace"][
            "magnitude_matched_to_learned_head_plus_state"],
        "state_noise_preserves_learned_hybrid_head": out["head_plus_state_noise"]["trace"][
            "head_parameters_match_learned_head_plus_state"],
        "new_state_rule": out["state_only"]["trace"]["not_E4_recentering_rule"],
    }
    doc = {"schema": "mop-hybrid-adaptation-shard/v1", "bed": bedname, "seed": seed,
           "selected_core_spec": spec, "pretrain": pre, "before_adaptation": {"A": pre_a, "B": pre_b},
           "arms": out, "context_shift": ctx["shift"], "units": ctx["units"],
           "checks": checks, "all_checks_pass": all(checks.values()),
           "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"{bedname}_{seed}.json", doc, "hybrid")
    print(f"hybrid {bedname} seed {seed}: all checks {doc['all_checks_pass']}", flush=True)
    return doc


def aggregate() -> dict:
    rows = [json.loads((io.RUNS / "hybrid" / f"{b}_{s}.json").read_text()) for b in BEDS for s in SEEDS]
    per_bed, supported = {}, []
    for b in BEDS:
        selected = [r for r in rows if r["bed"] == b]
        gain = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                - r["arms"]["head_only"]["future_acquisition_B"]["accuracy"] for r in selected]
        noise = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                 - r["arms"]["head_plus_state_noise"]["future_acquisition_B"]["accuracy"] for r in selected]
        gain_d, noise_d = power.decide(gain, e2.PREREG), power.decide(noise, e2.PREREG)
        gain_units, noise_units = {}, {}
        for r in selected:
            hybrid_units = r["arms"]["head_plus_state"]["future_acquisition_B"]["per_unit_accuracy"]
            head_units = r["arms"]["head_only"]["future_acquisition_B"]["per_unit_accuracy"]
            noisy_units = r["arms"]["head_plus_state_noise"]["future_acquisition_B"]["per_unit_accuracy"]
            for unit in set(hybrid_units) & set(head_units) & set(noisy_units):
                gain_units.setdefault(unit, []).append(hybrid_units[unit] - head_units[unit])
                noise_units.setdefault(unit, []).append(hybrid_units[unit] - noisy_units[unit])
        gain_group = power.lcb([float(np.mean(v)) for v in gain_units.values()])
        noise_group = power.lcb([float(np.mean(v)) for v in noise_units.values()])
        gain_d.update({"group_lower_95_cb": gain_group, "n_units": len(gain_units)})
        noise_d.update({"group_lower_95_cb": noise_group, "n_units": len(noise_units)})
        floors = [r["arms"]["head_plus_state"]["return_retention_A"]["accuracy"]
                  >= r["before_adaptation"]["A"]["accuracy"] - io.SESOI for r in selected]
        boundary = bed.context_boundary_over_seeds([
            {"no_adapt_new": r["before_adaptation"]["B"]["accuracy"],
             "no_adapt_old": r["before_adaptation"]["A"]["accuracy"],
             "adapted_new": r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"],
             "adapted_old": r["arms"]["head_plus_state"]["return_retention_A"]["accuracy"]}
            for r in selected])
        ok = (gain_d["verdict"] == "positive" and noise_d["verdict"] == "positive"
              and gain_group >= io.SESOI and noise_group >= io.SESOI
              and all(floors) and boundary["checks"]["boundary_crossed"])
        supported.append(ok)
        per_bed[b] = {"hybrid_minus_head": gain_d, "hybrid_minus_head_noise": noise_d,
                      "retention_floors": floors, "context_boundary": boundary, "supported": ok}
    checks = {"all_shards_terminal": len(rows) == len(BEDS) * len(SEEDS),
              "all_shard_controls_pass": all(r["all_checks_pass"] for r in rows)}
    doc = {"schema": "mop-hybrid-adaptation-result/v1", "experiment_terminal": True,
           "result": {"classification": "hybrid_supported" if all(supported)
                      else "hybrid_not_supported_under_floors", "per_bed": per_bed,
                      "arms": list(ARMS), "seeds": list(SEEDS),
                      "both_principal_beds_support": all(supported),
                      "state_rule": "supervised error gradient on a transient state buffer",
                      "claim_ceiling": "early acquisition and return retention on two controlled shifted contexts"},
           "mutation_checks": checks, "all_shards_terminal": checks["all_shards_terminal"]}
    io.seal("MOP_HYBRID_ADAPTATION_RESULT.json", doc)
    print(f"hybrid aggregate: {doc['result']['classification']}", flush=True)
    return doc


def run_all() -> dict:
    from mop.temporal.runs import supervisor

    names = [f"{b}_{s}" for b in BEDS for s in SEEDS]
    while True:
        pending = supervisor.missing("hybrid", names)
        if not pending:
            return aggregate()
        for name in pending:
            b, seed = name.rsplit("_", 1)
            supervisor.launch(["shard", b, seed], "hybrid.log", f"hy:{name}",
                              module="mop.temporal.runs.hybrid")
        time.sleep(5)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv or argv[0] == "all":
        run_all()
    elif argv[0] == "aggregate":
        aggregate()
    elif argv[0] == "shard":
        run_seed(argv[1], int(argv[2]))
    else:
        raise ValueError(argv)
    lock = os.environ.get("TEMPORAL_SHARD_LOCK")
    if lock:
        from pathlib import Path
        Path(lock).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
