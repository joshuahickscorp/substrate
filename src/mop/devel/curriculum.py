from __future__ import annotations

from pathlib import Path

import torch

from ..diagnostics import linear_probe
from ..logging_utils import get_logger

log = get_logger("curriculum")


def clamp_clips_for(profile, n: int) -> int:
    if profile is not None:
        return profile.clamp_clips(n)
    return max(1, int(n))


SIGNAL_MARGIN = 0.15  # real must beat shuffled-label accuracy by this to count as decodable
PROGRESS_EPS = 0.05  # learning progress below this is "flat" (used in the stopping condition)
MASTERY_THRESH = 0.92  # decodability above this is "already learned" (low remaining value)
DEFAULT_EVAL_CLIPS = 48


def _encode_source(src: Path, encoder_name: str, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    from ..config import compose
    from ..devices import resolve
    from ..substrate import iter_video_clips, load_encoder

    dev = resolve("cpu")
    enc = load_encoder(compose([f"encoder={encoder_name}", "device=cpu"]).encoder).to(dev.device)
    xs, ys = [], []
    for clip_b, lab_b in iter_video_clips(src, frames_per_clip=8, res=16, batch=4, limit=n, stratified=True):
        with torch.no_grad():
            lat = enc.encode(clip_b.to(dev.device))
        if lat.dim() == 3:  # [B, tokens, dim] -> pool tokens to [B, dim]
            lat = lat.mean(1)
        xs.append(lat.cpu().float())
        ys.append(lab_b)
    if not xs:
        raise ValueError(f"no clips encoded from {src}")
    return torch.cat(xs), torch.cat(ys)


def _mean_probe(x: torch.Tensor, y: torch.Tensor, seeds: range) -> float:
    return float(sum(linear_probe(x, y, seed=s)["score"] for s in seeds) / len(seeds))


def learning_progress(x: torch.Tensor, y: torch.Tensor, seed: int = 0, folds: int = 5) -> dict:
    n = int(x.shape[0])
    small = max(8, n // 3)
    acc_full = _mean_probe(x, y, range(seed, seed + folds))
    acc_small = _mean_probe(x[:small], y[:small], range(seed, seed + folds))
    shuffled = 0.0
    for k in range(folds):
        g = torch.Generator().manual_seed(seed + 1000 + k)
        shuffled += linear_probe(x, y[torch.randperm(n, generator=g)], seed=seed + k)["score"]
    shuffled /= folds
    chance = linear_probe(x, y, seed=seed)["chance"]
    signal = acc_full - shuffled
    progress = acc_full - acc_small
    decodable = bool(signal > SIGNAL_MARGIN)
    is_noise = not decodable  # real labels no more decodable than shuffled => aleatoric noisy-TV
    return {
        "acc_small": round(acc_small, 4),
        "acc_full": round(acc_full, 4),
        "shuffled": round(shuffled, 4),
        "signal": round(signal, 4),
        "progress": round(progress, 4),
        "chance": round(chance, 4),
        "mastery": round(acc_full, 4),
        "decodable": decodable,
        "is_noise": is_noise,
    }


def score_candidate(slug: str, src: Path, encoder_name: str, n: int, seed: int) -> dict:
    try:
        x, y = _encode_source(src, encoder_name, n, seed)
    except Exception as e:
        return {"slug": slug, "error": f"{type(e).__name__}: {e}", "score": -1.0, "is_noise": False}
    lp = learning_progress(x, y, seed=seed)
    if lp["is_noise"]:
        score = -1.0  # never chase the noisy-TV (aleatoric)
    elif lp["mastery"] >= MASTERY_THRESH:
        score = 0.05 + 0.1 * max(0.0, lp["progress"])  # already learned: low remaining value
    else:
        score = (1.0 - lp["mastery"]) + 0.5 * max(0.0, lp["progress"])  # learnable room + momentum
    return {"slug": slug, "n_eval": int(x.shape[0]), "score": round(float(score), 4), **lp}


def next_lesson(
    sources: list[dict],
    encoder_name: str = "vjepa2_vitl_fpc64_256",
    profile=None,
    eval_clips: int = DEFAULT_EVAL_CLIPS,
    next_clips: int = 64,
    seed: int = 0,
) -> dict:
    n_eval = clamp_clips_for(profile, eval_clips)
    scored = [score_candidate(s["slug"], Path(s["root"]), encoder_name, n_eval, seed) for s in sources]
    by_slug = {s["slug"]: s for s in sources}
    rejected_noise = [s["slug"] for s in scored if s.get("is_noise")]
    viable = [s for s in scored if not s.get("is_noise") and s.get("score", -1) >= 0]
    viable.sort(key=lambda s: s["score"], reverse=True)

    if not viable:
        return {
            "ranked": scored,
            "chosen": None,
            "reason": "no learnable candidate (all noise or already mastered or errored)",
            "rejected_noise": rejected_noise,
            "north_star_step": "choose what to study -> (nothing learnable now)",
        }
    top = viable[0]
    fallback = viable[1]["slug"] if len(viable) > 1 else None
    cap = by_slug.get(top["slug"], {}).get("capacity", "abstraction")
    return {
        "ranked": viable + [s for s in scored if s.get("is_noise") or s.get("score", -1) < 0],
        "chosen": top["slug"],
        "reason": (
            f"highest learning progress ({top['progress']:+.3f}: acc {top['acc_small']:.2f} -> "
            f"{top['acc_full']:.2f}), learnable but not yet mastered, not noisy-TV"
        ),
        "expected_capacity": cap,
        "expected_cost_clips": clamp_clips_for(profile, next_clips),
        "stopping_condition": f"mastery >= {MASTERY_THRESH} or progress < {PROGRESS_EPS}",
        "fallback": fallback,
        "rejected_noise": rejected_noise,
        "north_star_step": "notice surprise -> choose what to study (curiosity = learning progress)",
    }


def render_md(manifest: dict) -> str:
    L = ["# Next-lesson manifest (curriculum engine)", ""]
    if manifest.get("chosen") is None:
        L += [
            f"No lesson chosen: {manifest['reason']}",
            "",
            f"Rejected as noisy-TV: {manifest['rejected_noise']}",
        ]
        return "\n".join(L) + "\n"
    L += [
        f"**Study next**: `{manifest['chosen']}` -> targets capacity **{manifest['expected_capacity']}**",
        f"- why: {manifest['reason']}",
        f"- expected cost: {manifest['expected_cost_clips']} clips",
        f"- stop when: {manifest['stopping_condition']}",
        f"- fallback: {manifest['fallback']}",
        f"- rejected as noisy-TV (aleatoric, not chased): {manifest['rejected_noise']}",
        "",
        "| candidate | progress | acc_small | acc_full | mastery | noise | score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in manifest["ranked"]:
        if "error" in s:
            L.append(f"| {s['slug']} | error | - | - | - | - | {s['score']} |")
        else:
            L.append(
                f"| {s['slug']} | {s['progress']:+.3f} | {s['acc_small']:.2f} | {s['acc_full']:.2f} | "
                f"{s['mastery']:.2f} | {'yes' if s['is_noise'] else 'no'} | {s['score']:.3f} |"
            )
    return "\n".join(L) + "\n"
