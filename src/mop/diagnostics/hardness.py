"""D3 hardness gradient: a slot-structured, difficulty-calibrated task with a PER-SAMPLE hardness
label, built so a deterministic DSL program over its slots decides the label (verifier_exec executes
that DSL). This is the graded test bed the A3 verifier-reasoning runner grades on the HARD bin.

Every sample is four discrete slots (shape, color, motion, count), each drawn from a small vocabulary
and embedded into a continuous slot-latent (a block per slot, so the tensor looks like a frozen-encoder
latent the shell refines). The LABEL is a fixed target DSL program executed on the true slots (see
verifier_exec.target_program): a conjunction over the slots that a plain linear read of clean slots can
satisfy, so the regime is separable when clean. Difficulty is injected per sample by CORRUPTING a
graded number of slot blocks with Gaussian noise: a HARD sample has more corrupted slots, so its slot
identities (and therefore the DSL decision) are ambiguous from the latent alone. Hardness is thus a
real, per-sample, decision-relevant quantity, not a label-independent nuisance.

The regime is tuned toward non-trivial-but-not-ceiling best-mode accuracy (~0.55 to 0.85): the noise
scale and the fraction of hard samples are chosen so a strong linear read clears chance (D3 calibrated,
reference_separation) yet misses a substantial hard tail (room for test-time compute to matter).

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# slot vocabulary sizes, fixed. The DSL (verifier_exec) is defined over exactly these four slots.
SLOT_NAMES = ("shape", "color", "motion", "count")
SLOT_CARD = {"shape": 4, "color": 4, "motion": 3, "count": 5}  # count in {0,1,2,3,4}
SLOT_ORDER = tuple(SLOT_NAMES)


@dataclass
class GradedTask:
    """A slot-structured graded regime. x is the per-sample slot-latent (n, dim); slots is the integer
    slot table (n, 4) in SLOT_ORDER; y is the DSL-program label (n,); hardness is a per-sample integer
    (number of corrupted slot blocks, 0..4); hard_mask marks the preregistered HARD bin."""

    x: torch.Tensor
    slots: torch.Tensor  # long (n, 4)
    y: torch.Tensor  # long (n,)
    hardness: torch.Tensor  # long (n,) number of corrupted slots
    hard_mask: torch.Tensor  # bool (n,)
    slot_dim: int
    dim: int


def slot_block_dim(slot_dim: int) -> dict:
    """Contiguous latent-block layout: each slot owns `slot_dim` dimensions, laid out in SLOT_ORDER.
    Returned as {slot: (start, stop)} so a decoder head can read a slot from its own block if wanted
    and so the corruption injector hits whole slots."""
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
    """Build the graded slot task. Each slot value gets a fixed random codeword in its block; the
    sample latent is the concatenation of its four slot codewords plus per-sample Gaussian jitter. A
    graded number of slots (0..4, drawn so `hard_frac` of samples corrupt at least `hard_threshold`)
    are then heavily corrupted with `noise`-scaled Gaussian, blurring those slots' identity. The label
    is the target DSL program executed on the TRUE (uncorrupted) slots, so recovering it under
    corruption requires reading through the noise. hard_mask marks corrupted-slot-count >= threshold."""
    g = torch.Generator().manual_seed(seed)
    dim = slot_dim * len(SLOT_ORDER)
    spans = slot_block_dim(slot_dim)

    # fixed codebook: one random codeword per (slot, value), shared across samples (seed-stable)
    codebook = {name: torch.randn(SLOT_CARD[name], slot_dim, generator=g) * 1.5 for name in SLOT_ORDER}

    slots = torch.stack([torch.randint(0, SLOT_CARD[name], (n,), generator=g) for name in SLOT_ORDER], dim=1)

    x = torch.zeros(n, dim)
    for i, name in enumerate(SLOT_ORDER):
        lo, hi = spans[name]
        x[:, lo:hi] = codebook[name][slots[:, i]]
    x = x + torch.randn(n, dim, generator=g) * 0.35  # light global jitter, present on every sample

    # graded corruption: draw a per-sample corrupted-slot count with the target hard fraction. The
    # count distribution is chosen so P(count >= hard_threshold) ~= hard_frac.
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
    """Per-sample corrupted-slot count in {0..4}. A two-mode mixture: an EASY mode (0 or 1 corrupted
    slots) and a HARD mode (hard_threshold..4). hard_frac of samples fall in the hard mode, so the hard
    bin has the intended mass and hardness correlates with corruption by construction."""
    max_slots = len(SLOT_ORDER)
    is_hard = torch.rand(n, generator=g) < hard_frac
    easy_counts = torch.randint(0, hard_threshold, (n,), generator=g)  # 0..threshold-1
    hard_counts = torch.randint(hard_threshold, max_slots + 1, (n,), generator=g)  # threshold..4
    return torch.where(is_hard, hard_counts, easy_counts).long()


def hardness_gradient_certificate(task: GradedTask, seed: int = 0, margin: float = 0.05) -> dict:
    """D3 certificate that the graded regime carries a REAL, decision-relevant hardness gradient: a
    strong linear read (reference_separation) must clear chance overall AND separate the easy bin from
    the hard bin by `margin`. If the easy-hard accuracy gap vanishes, corruption did not move the DSL
    decision and grading on the hard bin is meaningless (unreadable), not a win."""
    from mop.diagnostics.difficulty_calibration import reference_separation
    from mop.diagnostics.linear_probe import linear_probe

    cal = reference_separation(task.x, task.y, seed=seed, margin=margin)
    probe = linear_probe(task.x, task.y, seed=seed, epochs=400)
    # per-bin accuracy from the same fitted probe, read on the held-out split it scored on. linear_probe
    # returns only a scalar score, so re-fit lightweight and read per-bin on a fresh split here.
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
    """Fit a linear probe on a 70/30 split and read accuracy separately on the easy and hard test
    samples, so the certificate can show the hard bin is genuinely harder for a strong linear read."""
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
