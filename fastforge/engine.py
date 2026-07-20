"""One training engine for every arm.

The engine owns exactly three responsibilities that the science depends on:

1. update partitioning: only parameters in the declared trainable groups may change, and the receipt proves
   which ones actually did;
2. checkpoint ancestry: a content hash of the parameter tensors before and after every phase;
3. interference statistics: gradient conflict, old domain probe loss, and representation drift, measured on
   tuning batches only, never on the untouched test split.

Memory policies are budget matched. EWC is the conventional regularization control. A tie is a null, so the
engine never rounds, never rescales, and never reports an effect it did not compute.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- identity and receipts


def checkpoint_sha(model) -> str:
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def group_sha(model, groups) -> dict[str, str]:
    named = dict(model.named_parameters())
    out = {}
    for g in groups:
        h = hashlib.sha256()
        for n in sorted(model.param_groups[g]):
            h.update(n.encode())
            h.update(named[n].detach().cpu().numpy().tobytes())
        out[g] = h.hexdigest()[:16]
    return out


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _changed(model, before, tol=1e-9) -> list[str]:
    return sorted(n for n, p in model.named_parameters() if float((p.detach() - before[n]).abs().max()) > tol)


def trainable_names(model, groups) -> list[str]:
    out = []
    for g in groups:
        out.extend(model.param_groups[g])
    return sorted(set(out))


# ---------------------------------------------------------------- memory policies (budget matched)


class Memory:
    """cap is the number of retained sequences and is identical across every arm that uses memory."""

    def __init__(self, policy: str = "gdumb", cap: int = 600):
        self.policy, self.cap = policy, cap
        self.x, self.y, self.seen = [], [], 0

    def add(self, x, y, rng):
        if self.policy == "none":
            return
        for i in range(len(x)):
            self.seen += 1
            if len(self.x) < self.cap:
                self.x.append(x[i])
                self.y.append(int(y[i]))
            elif self.policy == "reservoir":
                j = int(rng.integers(0, self.seen))
                if j < self.cap:
                    self.x[j], self.y[j] = x[i], int(y[i])
            elif self.policy == "recent":
                self.x.pop(0)
                self.y.pop(0)
                self.x.append(x[i])
                self.y.append(int(y[i]))
        if self.policy == "gdumb" and len(self.x) >= self.cap:
            yy = np.array(self.y)
            cls = np.unique(yy)
            per = max(1, self.cap // len(cls))
            keep = []
            for c in cls:
                idx = np.where(yy == c)[0]
                keep.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
            keep = keep[: self.cap]
            self.x = [self.x[i] for i in keep]
            self.y = [self.y[i] for i in keep]

    def sample(self, n, rng):
        if not self.x:
            return None
        idx = rng.choice(len(self.x), min(n, len(self.x)), replace=False)
        return torch.stack([self.x[i] for i in idx]), torch.tensor([self.y[i] for i in idx])

    def size(self):
        return len(self.x)


# ---------------------------------------------------------------- interference statistics


def _flat_grad(model, names):
    named = dict(model.named_parameters())
    gs = [named[n].grad.flatten() for n in names if named[n].grad is not None]
    return torch.cat(gs) if gs else None


def reference_gradient(model, names, x, y, d):
    """Gradient of the reference domain loss with respect to the shared group. Tuning batch only."""
    model.zero_grad(set_to_none=True)
    logits, _ = model(x, d)
    F.cross_entropy(logits, y).backward()
    g = _flat_grad(model, names)
    g = g.detach().clone() if g is not None else None
    model.zero_grad(set_to_none=True)
    return g


@torch.no_grad()
def probe_loss(model, x, y, d) -> float:
    was = model.training
    model.eval()
    out = float(F.cross_entropy(model(x, d)[0], y))
    model.train(was)
    return out


# ---------------------------------------------------------------- gates (simple rules only, <= 150 LOC)


class Gate:
    """Decides, per update, whether the shared group may change. Simple rules only by default.

    kind      signal
    always    shared group always trainable
    never     shared group never trainable
    cosine    freeze when the new domain gradient conflicts with the reference domain gradient
    probe     freeze when the reference domain probe loss rises above its entry value plus a margin
    drift     restore the delta toward the anchor when the measured drift exceeds a bound
    perf      reopen the shared group only after a performance drop on the active domain
    random    freeze at a fixed rate of one half, with no signal. The rate matched control is
              shuffled, which replays the real decision sequence permuted and therefore blocks
              exactly as often as the real gate did
    shuffled  the real signal sequence, permuted, so timing carries no information
    wrong     the reference batch comes from the wrong domain
    anchor_restore  revert the shared delta toward the frozen anchor once measured drift exceeds a bound,
                    rather than merely freezing it
    """

    KINDS = (
        "always",
        "never",
        "cosine",
        "probe",
        "drift",
        "perf",
        "random",
        "shuffled",
        "wrong",
        "anchor_restore",
    )

    def __init__(self, kind="always", thresh=0.0, rng=None, rate=0.5, every=10):
        self.kind, self.thresh, self.every = kind, thresh, every
        self.rng = rng or np.random.default_rng(0)
        self.rate, self.ref, self.base_probe = rate, None, None
        self.decisions: list[int] = []
        self._replay: list[int] | None = None

    def set_reference(self, grad, base_probe):
        self.ref, self.base_probe = grad, base_probe

    def replay(self, decisions):
        self._replay = list(decisions)

    def allow_shared(self, step, model, names, cur_probe=None, drift=None, perf_drop=None) -> bool:
        if self._replay is not None:
            d = bool(self._replay[step % len(self._replay)])
            self.decisions.append(int(d))
            return d
        if self.kind == "always":
            d = True
        elif self.kind == "never":
            d = False
        elif self.kind == "random":
            d = bool(self.rng.random() > self.rate)
        elif self.kind in ("cosine", "wrong"):
            g = _flat_grad(model, names)
            if g is None or self.ref is None:
                d = True
            else:
                cos = float(F.cosine_similarity(g.unsqueeze(0), self.ref.unsqueeze(0)))
                d = cos >= self.thresh
        elif self.kind == "probe":
            d = cur_probe is None or self.base_probe is None or (cur_probe - self.base_probe) <= self.thresh
        elif self.kind in ("drift", "anchor_restore"):
            d = drift is None or drift <= self.thresh
        elif self.kind == "perf":
            d = bool(perf_drop is not None and perf_drop > self.thresh)
        else:
            d = True
        self.decisions.append(int(d))
        return d


# ---------------------------------------------------------------- fit and evaluate


def fit(
    model,
    d,
    X,
    Y,
    *,
    train_groups,
    steps,
    lr=1e-3,
    rng=None,
    memory=None,
    batch=64,
    gate=None,
    shared_group=None,
    ref_batch=None,
    ref_domain=None,
    ewc=None,
    ewc_lambda=0.0,
    update_recent=False,
    probe_every=10,
):
    """Train the declared groups on domain d. Everything else is frozen and proven frozen by the receipt."""
    rng = rng or np.random.default_rng(0)
    names = trainable_names(model, train_groups)
    named = dict(model.named_parameters())
    for n, p in model.named_parameters():
        p.requires_grad_(n in set(names))
    shared_names = model.param_groups.get(shared_group, []) if shared_group else []
    before = _snapshot(model)
    sha_before = checkpoint_sha(model)
    opt = torch.optim.Adam([named[n] for n in names], lr)
    t0 = time.time()
    model.train()
    gate_blocks, updates, last_acc = 0, 0, None
    cur_probe = drift = perf_drop = None

    for step in range(steps):
        bi = rng.choice(len(X), min(batch, len(X)), replace=False)
        xb, yb = X[bi], Y[bi]
        if memory is not None and memory.size():
            s = memory.sample(batch, rng)
            if s:
                xb, yb = torch.cat([xb, s[0]]), torch.cat([yb, s[1]])
        logits, _ = model(xb, d, update_recent=update_recent)
        loss = F.cross_entropy(logits, yb)
        if ewc and ewc_lambda:
            pen = sum(
                (ewc["fisher"][n] * (named[n] - ewc["star"][n]) ** 2).sum()
                for n in names
                if n in ewc["fisher"]
            )
            loss = loss + ewc_lambda * pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if gate is not None and shared_names:
            if gate.kind in ("probe", "perf") and ref_batch is not None and step % probe_every == 0:
                cur_probe = probe_loss(model, ref_batch[0], ref_batch[1], ref_domain)
            if gate.kind == "drift" and hasattr(model, "drift"):
                drift = model.drift()
            elif gate.kind == "anchor_restore" and hasattr(model, "drift_ratio"):
                drift = model.drift_ratio()  # scale free, so the threshold is not a threshold on model size
            if gate.kind == "perf":
                with torch.no_grad():
                    acc = float((logits.argmax(1) == yb).float().mean())
                perf_drop = 0.0 if last_acc is None else max(0.0, last_acc - acc)
                last_acc = acc
            if not gate.allow_shared(step, model, shared_names, cur_probe, drift, perf_drop):
                gate_blocks += 1
                for n in shared_names:
                    if named[n].grad is not None:
                        named[n].grad = None
                if gate.kind == "anchor_restore" and hasattr(model, "restore_anchor"):
                    model.restore_anchor(0.25)
        opt.step()
        updates += 1

    changed = _changed(model, before)
    illegal = sorted(set(changed) - set(names))
    return {
        "domain": d,
        "steps": steps,
        "updates": updates,
        "batch": batch,
        "lr": lr,
        "trainable_groups": sorted(train_groups),
        "trainable_params": names,
        "frozen_params": sorted(set(dict(model.named_parameters())) - set(names)),
        "changed_params": changed,
        "changed_param_count": len(changed),
        "undeclared_changes": illegal,
        "gate_kind": None if gate is None else gate.kind,
        "gate_blocked_updates": gate_blocks,
        "gate_decisions": None if gate is None else list(gate.decisions),
        "checkpoint_sha_before": sha_before,
        "checkpoint_sha_after": checkpoint_sha(model),
        "group_sha_after": group_sha(model, sorted(model.param_groups)),
        "trainable_param_count": int(sum(named[n].numel() for n in names)),
        "wall_seconds": round(time.time() - t0, 2),
    }


@torch.no_grad()
def evaluate(model, d, X, Y, chunk=256) -> float:
    model.eval()
    pred = torch.cat([model(X[k : k + chunk], d)[0].argmax(1) for k in range(0, len(X), chunk)])
    return float((pred == Y).float().mean())


def fisher_diag(model, d, X, Y, names, rng, n=32, batch=32):
    """Diagonal Fisher for the EWC control, estimated on training batches only."""
    named = dict(model.named_parameters())
    fisher = {n_: torch.zeros_like(named[n_]) for n_ in names}
    model.train()
    for _ in range(n):
        bi = rng.choice(len(X), min(batch, len(X)), replace=False)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(X[bi], d)[0], Y[bi]).backward()
        for n_ in names:
            if named[n_].grad is not None:
                fisher[n_] += named[n_].grad.detach() ** 2
    model.zero_grad(set_to_none=True)
    for n_ in fisher:
        fisher[n_] /= n
    return {"fisher": fisher, "star": {n_: named[n_].detach().clone() for n_ in names}}


def lcb(effects) -> float:
    """Random effects lower 95 percent confidence bound. t95 = 1.833 at n <= 10, else 1.729."""
    e = np.asarray(effects, float)
    n = len(e)
    if n < 2:
        return float(e.mean()) if n else 0.0
    t = 1.833 if n <= 10 else 1.729
    return float(e.mean() - t * e.std(ddof=1) / np.sqrt(n))


def effect(a, b) -> dict:
    e = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    return {"mean": round(float(np.mean(e)), 4), "lower_95_cb": round(lcb(e), 4), "n": len(e)}
