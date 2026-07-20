"""Fast core representation analysis.

Explanatory only. Nothing here can turn a null into a positive, and nothing here is allowed to be quoted as
evidence for the premise. Its job is to say what the shared fast dynamics actually encode after a domain
sequence, so that a null has a mechanism attached to it rather than a shrug.

House style: no dashes.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from fastforge import arch as A
from fastforge import data as D
from fastforge import engine as E
from fastforge import sequence as S
from fastforge.runs import io

SEEDS = [0, 1, 2]
DIRECTIONS = [("har", "speech"), ("speech", "har")]
ARMS = ["G0_always_trainable", "G1_frozen_after_first", "H_cosine_gate"]


@torch.no_grad()
def feats(model, d, X, chunk=256):
    model.eval()
    return torch.cat([model.features(X[k : k + chunk], d)[:, -1] for k in range(0, len(X), chunk)])


def linear_probe(F, Y, seed, steps=250):
    """A linear probe on frozen features. Decodability, not learning capacity."""
    n = len(F)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    head = torch.nn.Linear(F.shape[1], int(Y.max()) + 1)
    opt = torch.optim.Adam(head.parameters(), 3e-3)
    for _ in range(steps):
        bi = rng.choice(len(tr), min(256, len(tr)), replace=False)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(head(F[tr][bi]), Y[tr][bi]).backward()
        opt.step()
    with torch.no_grad():
        return float((head(F[te]).argmax(1) == Y[te]).float().mean())


def timestep_probe(model, d, X, seed):
    """Temporal transition decodability: can the position in the sequence be read from the state."""
    with torch.no_grad():
        model.eval()
        o = model.features(X[:512], d)
    T = o.shape[1]
    marks = [T // 4, T // 2, (3 * T) // 4, T - 1]
    F = torch.cat([o[:, m] for m in marks])
    Y = torch.cat([torch.full((o.shape[0],), i, dtype=torch.long) for i in range(len(marks))])
    return linear_probe(F, Y, seed)


def analyse(arm, direction, seed):
    a, b = direction
    r_model = _train(arm, direction, seed)
    model, sa, sb, pre_state = r_model
    out = {}
    fa, fb = feats(model, a, sa["test"][0][:1500]), feats(model, b, sb["test"][0][:1500])
    out["linear_decodability_first_domain"] = linear_probe(fa, sa["test"][1][:1500], seed)
    out["linear_decodability_second_domain"] = linear_probe(fb, sb["test"][1][:1500], seed)
    out["class_separability_first_domain"] = out["linear_decodability_first_domain"]
    out["class_separability_second_domain"] = out["linear_decodability_second_domain"]
    n = min(len(fa), len(fb))
    F = torch.cat([fa[:n], fb[:n]])
    Y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    out["domain_separability"] = linear_probe(F, Y, seed)
    out["temporal_transition_decodability_first"] = timestep_probe(model, a, sa["test"][0], seed)
    out["temporal_transition_decodability_second"] = timestep_probe(model, b, sb["test"][0], seed)

    # representation drift between the first domain checkpoint and the final one
    cur = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        f_now = feats(model, a, sa["test"][0][:800])
        model.load_state_dict(pre_state)
        f_before = feats(model, a, sa["test"][0][:800])
        model.load_state_dict(cur)
    out["representation_drift_l2"] = float((f_now - f_before).norm(dim=1).mean())
    out["representation_drift_cosine"] = float(
        torch.nn.functional.cosine_similarity(f_now, f_before, dim=1).mean())
    # return to domain geometry: do class centroids come back to where they were
    def cents(F, Y):
        return torch.stack([F[Y == c].mean(0) for c in sorted(set(Y.tolist())) if (Y == c).any()])
    ya = sa["test"][1][:800]
    ca, cb = cents(f_before, ya), cents(f_now, ya)
    out["return_to_domain_centroid_cosine"] = float(
        torch.nn.functional.cosine_similarity(ca, cb, dim=1).mean())

    # fast state reset sensitivity: only the second half of the sequence is shown
    half = sa["test"][0][:1500][:, sa["test"][0].shape[1] // 2 :]
    out["fast_state_reset_accuracy"] = linear_probe(feats(model, a, half), sa["test"][1][:1500], seed)
    out["fast_state_reset_sensitivity"] = (out["linear_decodability_first_domain"]
                                           - out["fast_state_reset_accuracy"])
    # dependence ablations
    out["accuracy_first_domain"] = E.evaluate(model, a, *sa["test"])
    if hasattr(model, "adapters"):
        saved = copy.deepcopy(model.adapters[a].state_dict())
        with torch.no_grad():
            for p in model.adapters[a].parameters():
                p.zero_()
        out["accuracy_without_adapter"] = E.evaluate(model, a, *sa["test"])
        model.adapters[a].load_state_dict(saved)
        out["adapter_dependence"] = out["accuracy_first_domain"] - out["accuracy_without_adapter"]
    fresh = A.build(model.name if hasattr(model, "name") else "G",
                    {a: (sa["channels"], sa["classes"]), b: (sb["channels"], sb["classes"])})
    shared = S.shared_group_name(model)
    if shared:
        named, fnamed = dict(model.named_parameters()), dict(fresh.named_parameters())
        saved = {n: named[n].detach().clone() for n in model.param_groups[shared]}
        with torch.no_grad():
            for n in model.param_groups[shared]:
                named[n].copy_(fnamed[n])
        out["accuracy_with_random_shared_core"] = E.evaluate(model, a, *sa["test"])
        with torch.no_grad():
            for n, v in saved.items():
                named[n].copy_(v)
        out["shared_core_dependence"] = out["accuracy_first_domain"] - out["accuracy_with_random_shared_core"]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def _train(arm, direction, seed):
    """A compact replay of the sequence that keeps the model and the first domain checkpoint."""
    a, b = direction
    p = S.ARMS[arm]
    sa, sb = D.splits(a, seed), D.splits(b, seed)
    torch.manual_seed(1000 + seed)
    model = A.build(p["arch"], {a: (sa["channels"], sa["classes"]), b: (sb["channels"], sb["classes"])})
    shared = S.shared_group_name(model)
    rng = np.random.default_rng(seed)
    mem_a, mem_b = E.Memory("gdumb", 600), E.Memory("gdumb", 600)
    E.fit(model, a, *sa["main"], train_groups=S.local_groups(model, a) + [shared],
          steps=S.budget(a), lr=S.lr_for(a), rng=rng, memory=mem_a)
    mem_a.add(*sa["main"], rng)
    pre = copy.deepcopy(model.state_dict())
    g2 = S.local_groups(model, b) + ([shared] if p.get("shared_p2", True) else [])
    E.fit(model, b, *sb["main"], train_groups=g2, steps=S.budget(b), lr=S.lr_for(b), rng=rng, memory=mem_b)
    g3 = S.local_groups(model, a) + ([shared] if p.get("shared_p3", True) else [])
    E.fit(model, a, *sa["main"], train_groups=g3, steps=S.budget(a, "r"), lr=S.lr_for(a), rng=rng,
          memory=mem_a)
    return model, sa, sb, pre


def main():
    t0 = time.time()
    rows = {}
    for direction in DIRECTIONS:
        dname = f"{direction[0]}->{direction[1]}"
        for arm in ARMS:
            per = [analyse(arm, direction, s) for s in SEEDS]
            rows[f"{dname}|{arm}"] = {
                k: round(float(np.mean([p[k] for p in per])), 4) for k in per[0]
            }
            print(f"  {dname} {arm} done", flush=True)
    keys = sorted({k for v in rows.values() for k in v})
    io.seal("MOP_FAST_CORE_REPRESENTATION_REPORT.json", {
        "schema": "mop-fast-core-representation/v1",
        "seeds": SEEDS, "arms": ARMS, "directions": [f"{a}->{b}" for a, b in DIRECTIONS],
        "measurements": rows,
        "status": "explanatory only, cannot determine a verdict",
    })
    lines = ["# Fast core representation report", "",
             "Explanatory only. These numbers describe what the shared fast dynamics encode after a domain",
             "sequence. They cannot make a null into a positive.", "",
             "| measurement | " + " | ".join(rows) + " |",
             "| --- | " + " | ".join("---" for _ in rows) + " |"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(rows[r].get(k, "")) for r in rows) + " |")
    io.seal_md("MOP_FAST_CORE_REPRESENTATION_REPORT.md", "\n".join(lines) + "\n")
    print("represent done", round(time.time() - t0, 1), "s", flush=True)
    print("REPRESENT_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
