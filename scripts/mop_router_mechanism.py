
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
from mop.diagnostics import compute as C  # noqa: E402
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402

BASE = _ROOT / "data" / "cache"
SEEDS = list(range(10))
N_CLASS = 5  # shape, chance 0.20
FLOP_TOL = 0.10  # matched-FLOP tolerance (same as repo convention)
PARAM_TOL = 0.10  # matched-param tolerance
DEVICE = "cpu"


def zscore_fit(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    return mu, sd


X_vj = np.load(BASE / "vjepa2_vitl_nuisance/features.npy").astype(np.float32)  # (200,1024)
X_sf = np.load(BASE / "vjepa2_vitl_singleframe/features.npy").astype(np.float32)  # (200,1024)
X_dn = np.load(BASE / "dinov2s_nuisance_real/latents.npy").astype(np.float32)  # (200,384)
Y = np.load(BASE / "vjepa2_vitl_nuisance/labels_shape.npy").astype(np.int64)  # (200,)
assert X_vj.shape[0] == X_sf.shape[0] == X_dn.shape[0] == Y.shape[0] == 200

READERS = [("vjepa", X_vj), ("dino", X_dn), ("single", X_sf)]
READER_DIMS = [x.shape[1] for x in (X_vj, X_dn, X_sf)]  # [1024, 384, 1024]


def stratified_split(y, rng, frac_train=0.5, frac_router=0.25):
    idx_tr, idx_ro, idx_ev = [], [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0].copy()
        rng.shuffle(ci)
        n = len(ci)
        n_tr = int(round(n * frac_train))
        n_ro = int(round(n * frac_router))
        idx_tr += list(ci[:n_tr])
        idx_ro += list(ci[n_tr : n_tr + n_ro])
        idx_ev += list(ci[n_tr + n_ro :])
    return np.array(idx_tr), np.array(idx_ro), np.array(idx_ev)


class LinearHead(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


def train_head(X, y, idx_tr, d_out, seed, epochs=300, lr=0.05, wd=1e-3):
    torch.manual_seed(seed)
    mu = X[idx_tr].mean(0, keepdims=True)
    sd = X[idx_tr].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    Xt = torch.tensor(Xn)
    yt = torch.tensor(y)
    head = LinearHead(X.shape[1], d_out)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    tr = torch.tensor(idx_tr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(Xt[tr])
        loss = F.cross_entropy(logits, yt[tr])
        loss.backward()
        opt.step()
    return head, mu, sd


def head_logits(head, X, mu, sd):
    Xn = (X - mu) / sd
    with torch.no_grad():
        return head(torch.tensor(Xn))  # (N, d_out)


class Router(nn.Module):

    def __init__(self, d_desc, k, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_desc, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, desc):
        return self.net(desc)  # gate logits (N,k)


def build_descriptor(idx_tr):
    parts = []
    for _, X in READERS:
        mu = X[idx_tr].mean(0, keepdims=True)
        sd = X[idx_tr].std(0, keepdims=True) + 1e-6
        parts.append((X - mu) / sd)
    return np.concatenate(parts, axis=1).astype(np.float32)  # (200, 1024+384+1024)


def train_router(desc, expert_logits, y, idx_ro, k, seed, hidden, epochs=400, lr=0.03, wd=1e-3):
    torch.manual_seed(seed + 1000)
    d_desc = desc.shape[1]
    router = Router(d_desc, k, hidden)
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=wd)
    D = torch.tensor(desc)
    yt = torch.tensor(y)
    EL = torch.stack([torch.as_tensor(e) for e in expert_logits], dim=0)  # (K,N,C)
    probs = F.softmax(EL, dim=-1)  # (K,N,C)
    ro = torch.tensor(idx_ro)
    for _ in range(epochs):
        opt.zero_grad()
        gate = F.softmax(router(D[ro]), dim=-1)  # (n_ro, K)
        mix = torch.einsum("nk,knc->nc", gate, probs[:, ro, :])  # (n_ro, C)
        loss = F.nll_loss(torch.log(mix + 1e-9), yt[ro])
        loss.backward()
        opt.step()
    return router


def routed_acc(router, desc, expert_logits, y, idx_ev):
    D = torch.tensor(desc)
    EL = torch.stack([torch.as_tensor(e) for e in expert_logits], dim=0)
    probs = F.softmax(EL, dim=-1)
    ev = torch.tensor(idx_ev)
    with torch.no_grad():
        gate = F.softmax(router(D[ev]), dim=-1)
        mix = torch.einsum("nk,knc->nc", gate, probs[:, ev, :])
        pred = mix.argmax(-1)
    return (pred.numpy() == y[idx_ev]).mean()


def head_flops(d_in, d_out):
    return C.linear_flops(d_in, d_out, batch=1)


def router_flops(d_desc, hidden, k):
    return C.mlp_flops([d_desc, hidden, k], batch=1)


def head_params(d_in, d_out):
    return d_in * d_out + d_out


def router_params(d_desc, hidden, k):
    return d_desc * hidden + hidden + hidden * k + k


def run():
    ROUTER_HIDDEN = 32
    desc_dim = sum(READER_DIMS)  # 2432

    reader_flops = [head_flops(d, N_CLASS) for d in READER_DIMS]
    f_router_arm = sum(reader_flops) + router_flops(desc_dim, ROUTER_HIDDEN, 3)

    per_seed = []
    d_router_vs_best = []
    d_router_vs_homo = []
    match_ok_best = []
    match_ok_homo = []

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        idx_tr, idx_ro, idx_ev = stratified_split(Y, rng)

        heads = []
        expert_logits = []
        reader_eval_acc = {}
        for (name, X), d in zip(READERS, READER_DIMS, strict=False):
            head, mu, sd = train_head(X, Y, idx_tr, N_CLASS, seed)
            lg = head_logits(head, X, mu, sd).numpy()  # (200,C)
            heads.append((head, mu, sd, X, d, name))
            expert_logits.append(lg)
            acc = (lg[idx_ev].argmax(1) == Y[idx_ev]).mean()
            acc_ro = (lg[idx_ro].argmax(1) == Y[idx_ro]).mean()
            reader_eval_acc[name] = dict(eval=float(acc), router_train=float(acc_ro))

        desc = build_descriptor(idx_tr)

        router = train_router(desc, expert_logits, Y, idx_ro, k=3, seed=seed, hidden=ROUTER_HIDDEN)
        acc_router = routed_acc(router, desc, expert_logits, Y, idx_ev)

        best_name = max(reader_eval_acc, key=lambda n: reader_eval_acc[n]["router_train"])
        best_i = [r[0] for r in READERS].index(best_name)
        acc_best = reader_eval_acc[best_name]["eval"]
        best_dim = READER_DIMS[best_i]

        best_reader_flop = head_flops(best_dim, N_CLASS)
        k = max(
            2,
            round(
                (f_router_arm - router_flops(best_dim, ROUTER_HIDDEN, None if False else 2))
                / best_reader_flop
            ),
        )
        for _ in range(3):
            rf = router_flops(best_dim, ROUTER_HIDDEN, k)
            k = max(2, round((f_router_arm - rf) / best_reader_flop))
        Xbest = READERS[best_i][1]
        copy_logits = []
        for ci in range(k):
            crng = np.random.RandomState(seed * 100 + ci)
            sub = idx_tr.copy()
            crng.shuffle(sub)
            keep = sub[: int(round(len(sub) * 0.95))]
            ch, cmu, csd = train_head(Xbest, Y, keep, N_CLASS, seed * 100 + ci)
            copy_logits.append(head_logits(ch, Xbest, cmu, csd).numpy())
        mu = Xbest[idx_tr].mean(0, keepdims=True)
        sd = Xbest[idx_tr].std(0, keepdims=True) + 1e-6
        homo_desc = ((Xbest - mu) / sd).astype(np.float32)
        homo_router = train_router(
            homo_desc, copy_logits, Y, idx_ro, k=k, seed=seed + 7, hidden=ROUTER_HIDDEN
        )
        acc_homo = routed_acc(homo_router, homo_desc, copy_logits, Y, idx_ev)

        homo_arm_flops = k * best_reader_flop + router_flops(best_dim, ROUTER_HIDDEN, k)
        m_rb = C.matched_within(f_router_arm, best_reader_flop, tol=FLOP_TOL)  # router vs best-single FLOPs
        m_rh = C.matched_within(f_router_arm, homo_arm_flops, tol=FLOP_TOL)  # router vs homo bank FLOPs

        p_router_arm = sum(head_params(d, N_CLASS) for d in READER_DIMS) + router_params(
            desc_dim, ROUTER_HIDDEN, 3
        )
        p_best = head_params(best_dim, N_CLASS)
        p_homo = k * head_params(best_dim, N_CLASS) + router_params(best_dim, ROUTER_HIDDEN, k)
        mp_rh = C.matched_within(p_router_arm, p_homo, tol=PARAM_TOL)

        d_rb = float(acc_router - acc_best)
        d_rh = float(acc_router - acc_homo)
        d_router_vs_best.append(d_rb)
        d_router_vs_homo.append(d_rh)
        match_ok_best.append(bool(m_rb["matched"]))
        match_ok_homo.append(bool(m_rh["matched"] and mp_rh["matched"]))

        per_seed.append(
            dict(
                seed=seed,
                reader_eval_acc=reader_eval_acc,
                best_reader=best_name,
                acc_router=float(acc_router),
                acc_best_single=float(acc_best),
                acc_homo_bank=float(acc_homo),
                k_copies=int(k),
                delta_router_vs_best=d_rb,
                delta_router_vs_homo=d_rh,
                flops={
                    "router_arm": int(f_router_arm),
                    "best_single": int(best_reader_flop),
                    "homo_bank": int(homo_arm_flops),
                },
                params={
                    "router_arm": int(p_router_arm),
                    "best_single": int(p_best),
                    "homo_bank": int(p_homo),
                },
                match_router_vs_best_flops=m_rb,
                match_router_vs_homo_flops=m_rh,
                match_router_vs_homo_params=mp_rh,
            )
        )

    ci_rb = seed_ci(d_router_vs_best)
    ci_rh = seed_ci(d_router_vs_homo)
    sf_rb = sign_flip_report(d_router_vs_best)
    sf_rh = sign_flip_report(d_router_vs_homo)

    R1 = (ci_rb["lo"] > 0) and (not sf_rb["any_flip"]) and (ci_rb["mean"] > 0)
    R2 = (ci_rh["lo"] > 0) and (not sf_rh["any_flip"]) and (ci_rh["mean"] > 0) and all(match_ok_homo)
    R3 = all(match_ok_homo)

    if R1 and R2 and R3:
        verdict = (
            "WIN: trained router beats BOTH the tuned best-single reader and the matched homogeneous "
            "k-copy bank with ci lo > 0 and no sign flip. First trained-mechanism win on the real cache."
        )
        null_supported = False
    elif R3:
        verdict = (
            "NULL (density null CONFIRMED on the real cache): the trained router ties or loses to at "
            "least one baseline at matched FLOPs. Heterogeneous routing does not beat a tuned baseline. "
            "KILL-SWITCH honored, no tuning toward a win. This is an asset."
        )
        null_supported = True
    else:
        verdict = (
            "NOT-EVALUABLE: homogeneous compute match failed on at least one seed; the matched-FLOP "
            "heterogeneity test could not be run at the preregistered tolerance."
        )
        null_supported = None

    out = dict(
        experiment="router_mechanism_real_cache",
        lane="router (capability density mechanism)",
        target="shape under nuisance, 5-way, chance 0.20",
        substrate=(
            "real caches: vjepa2_vitl_nuisance (1024d), dinov2s_nuisance_real (384d), "
            "vjepa2_vitl_singleframe (1024d); identical 200 clips"
        ),
        preregistered_null=(
            "H0: trained router ties or loses to best single reader at matched FLOPs (a tie is a null). "
            "Reject only if router beats BOTH best-single (R1) and matched homogeneous k-copy bank (R2) "
            "with ci lo>0, no sign flip, compute matched (R3)."
        ),
        seeds=SEEDS,
        config=dict(
            n_class=N_CLASS,
            router_hidden=ROUTER_HIDDEN,
            desc_dim=desc_dim,
            flop_tol=FLOP_TOL,
            param_tol=PARAM_TOL,
            split="per-class 4 reader-train / 2 router-train / 2 eval (8 per class)",
        ),
        router_vs_best_single=dict(
            metric="accuracy delta (router spends ~3x FLOPs; NOT FLOP-matched by construction)",
            per_seed=d_router_vs_best,
            seed_ci=ci_rb,
            sign_flip=sf_rb,
            flop_ratio_note="router arm runs 3 readers vs best-single 1 reader; a tie here is a density LOSS",
        ),
        router_vs_homogeneous_bank=dict(
            metric="accuracy delta at MATCHED FLOPs and MATCHED params (clean heterogeneity test)",
            per_seed=d_router_vs_homo,
            seed_ci=ci_rh,
            sign_flip=sf_rh,
            all_seeds_flop_matched=all(match_ok_homo),
        ),
        gates=dict(
            R1_beats_best_single=bool(R1), R2_beats_homo_bank_matched=bool(R2), R3_compute_matched=bool(R3)
        ),
        verdict=verdict,
        null_supported=null_supported,
        per_seed=per_seed,
    )
    return out


if __name__ == "__main__":
    res = run()
    path = str(_ROOT / "runs" / "mot" / "router_mechanism.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", path)
    print("verdict:", res["verdict"])
    print("router_vs_best ci:", res["router_vs_best_single"]["seed_ci"])
    print("router_vs_homo ci:", res["router_vs_homogeneous_bank"]["seed_ci"])
    print("gates:", res["gates"])
    accs = {
        n: np.mean([ps["reader_eval_acc"][n]["eval"] for ps in res["per_seed"]])
        for n in ["vjepa", "dino", "single"]
    }
    print("mean reader eval acc:", {k: round(v, 3) for k, v in accs.items()})
    print(
        "mean acc_router",
        round(np.mean([ps["acc_router"] for ps in res["per_seed"]]), 3),
        "mean acc_best",
        round(np.mean([ps["acc_best_single"] for ps in res["per_seed"]]), 3),
        "mean acc_homo",
        round(np.mean([ps["acc_homo_bank"] for ps in res["per_seed"]]), 3),
    )
    print("best reader per seed:", [ps["best_reader"] for ps in res["per_seed"]])
    print("k copies per seed:", [ps["k_copies"] for ps in res["per_seed"]])
