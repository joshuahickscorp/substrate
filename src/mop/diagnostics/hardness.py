from __future__ import annotations

from dataclasses import dataclass

import torch

SLOT_NAMES = ("shape", "color", "motion", "count")
SLOT_CARD = {"shape": 4, "color": 4, "motion": 3, "count": 5}  # count in {0,1,2,3,4}
SLOT_ORDER = tuple(SLOT_NAMES)


@dataclass
class GradedTask:
    x: torch.Tensor
    slots: torch.Tensor  # long (n, 4)
    y: torch.Tensor  # long (n,)
    hardness: torch.Tensor  # long (n,) number of corrupted slots
    hard_mask: torch.Tensor  # bool (n,)
    slot_dim: int
    dim: int


def slot_block_dim(slot_dim: int) -> dict:
    spans, cur = {}, 0
    for name in SLOT_ORDER:
        spans[name] = (cur, cur + slot_dim)
        cur += slot_dim
    return spans


def make_graded_slot_task(
    n: int,
    slot_dim: int = 16,
    noise: float = 1.6,
    hard_frac: float = 0.45,
    hard_threshold: int = 2,
    seed: int = 0,
) -> GradedTask:
    g = torch.Generator().manual_seed(seed)
    dim = slot_dim * len(SLOT_ORDER)
    spans = slot_block_dim(slot_dim)

    codebook = {name: torch.randn(SLOT_CARD[name], slot_dim, generator=g) * 1.5 for name in SLOT_ORDER}

    slots = torch.stack([torch.randint(0, SLOT_CARD[name], (n,), generator=g) for name in SLOT_ORDER], dim=1)

    x = torch.zeros(n, dim)
    for i, name in enumerate(SLOT_ORDER):
        lo, hi = spans[name]
        x[:, lo:hi] = codebook[name][slots[:, i]]
    x = x + torch.randn(n, dim, generator=g) * 0.35  # light global jitter, present on every sample

    hardness = _draw_corrupt_counts(n, hard_frac, hard_threshold, g)
    for s in range(n):
        k = int(hardness[s].item())
        if k == 0:
            continue
        which = torch.randperm(len(SLOT_ORDER), generator=g)[:k]
        for j in which.tolist():
            lo, hi = spans[SLOT_ORDER[j]]
            x[s, lo:hi] = x[s, lo:hi] + torch.randn(slot_dim, generator=g) * noise

    from mop.shell.verifier_exec import target_program

    y = target_program().execute_on_slots(slots)
    hard_mask = hardness >= hard_threshold
    return GradedTask(
        x=x, slots=slots, y=y, hardness=hardness, hard_mask=hard_mask, slot_dim=slot_dim, dim=dim
    )


def _draw_corrupt_counts(n: int, hard_frac: float, hard_threshold: int, g: torch.Generator) -> torch.Tensor:
    max_slots = len(SLOT_ORDER)
    is_hard = torch.rand(n, generator=g) < hard_frac
    easy_counts = torch.randint(0, hard_threshold, (n,), generator=g)  # 0..threshold-1
    hard_counts = torch.randint(hard_threshold, max_slots + 1, (n,), generator=g)  # threshold..4
    return torch.where(is_hard, hard_counts, easy_counts).long()


def hardness_gradient_certificate(task: GradedTask, seed: int = 0, margin: float = 0.05) -> dict:
    from mop.diagnostics.difficulty_calibration import reference_separation
    from mop.diagnostics.linear_probe import linear_probe

    cal = reference_separation(task.x, task.y, seed=seed, margin=margin)
    probe = linear_probe(task.x, task.y, seed=seed, epochs=400)
    easy_acc, hard_acc = _per_bin_probe_acc(task, seed)
    gap = easy_acc - hard_acc
    return {
        "reference_score": cal["reference_score"],
        "chance": cal["chance"],
        "gap_over_chance": cal["gap"],
        "regime_calibrated": cal["regime_calibrated"],
        "probe_overall": round(float(probe["score"]), 4),
        "probe_acc_easy": round(easy_acc, 4),
        "probe_acc_hard": round(hard_acc, 4),
        "easy_hard_gap": round(gap, 4),
        "gradient_present": bool(cal["regime_calibrated"] and gap > margin),
    }


def _per_bin_probe_acc(task: GradedTask, seed: int) -> tuple[float, float]:
    import torch.nn.functional as F
    from torch import nn

    from mop.seeding import seed_everything

    n = task.x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]
    seed_everything(seed)
    nc = int(task.y.max().item()) + 1
    probe = nn.Linear(task.dim, nc)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(probe(task.x[tr]), task.y[tr]).backward()
        opt.step()
    with torch.no_grad():
        correct = (probe(task.x[te]).argmax(-1) == task.y[te]).float()
    hard_te = task.hard_mask[te]
    easy_acc = float(correct[~hard_te].mean()) if (~hard_te).any() else 0.0
    hard_acc = float(correct[hard_te].mean()) if hard_te.any() else 0.0
    return easy_acc, hard_acc
