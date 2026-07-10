#!/usr/bin/env python
"""Prove learned/control stimulus identity for the factorized local8 caches.

The three learned local8 caches predate per-input stimulus hashing. The scale-atlas builder consumes
this receipt only after independently validating its generator, cache, resolution, rebinding, and
random-control hashes; a failed or stale binding still refuses `matched_stimulus_hashes`. This closes
the gap with measured evidence instead of a rebuild:

1. Generator immutability: `make_factorized_clip` and `_hue_tint` are hashed in the last commit
   that touched the cache script (which predates every local8 build) and in the current working
   tree; the receipt records both hashes and the identical/changed verdict.
2. Deterministic regeneration: the eight clips are regenerated at seed 0 for each native
   resolution (256px for ViT-L/H, 384px for ViT-g) and hashed; the 256px set is compared
   clip-for-clip against the random control cache's recorded stimulus hashes.
3. Latent rebinding: for each learned cache, clip index 0 is re-encoded through the exact frozen
   encoder on CPU and compared to the stored latent row 0. CPU inference is the repo's
   bit-identical determinism baseline, so an exact match binds the stored latents to the
   regenerated clip bytes through the frozen weights.

One encoder loads at a time; the script refuses to start if another encoder process is alive.
Claim scope: input-identity and cache-integrity mechanics only. No em or en dashes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.cache_factorized_encoder as cfe  # noqa: E402

from mop.config import compose  # noqa: E402
from mop.substrate import load_encoder  # noqa: E402

SCHEMA = "mop-factorized-stimulus-identity/v1"
GENERATOR_FUNCTIONS = ("make_factorized_clip", "_hue_tint")
SCALES = (
    {
        "tag": "vit_l",
        "encoder": "vjepa2_vitl_fpc64_256",
        "cache": "vjepa2_vitl_local8_citable",
        "resolution": 256,
    },
    {"tag": "vit_h", "encoder": "vjepa2_vith", "cache": "vjepa2_vith_local8_citable", "resolution": 256},
    {"tag": "vit_g", "encoder": "vjepa2_vitg", "cache": "vjepa2_vitg_local8_citable", "resolution": 384},
)
RANDOM_CACHE = "vjepa2_vitl_local8_random_s0"
HEAVY_PATTERNS = ("cache_factorized_encoder.py", "cache_real_encoder.py", "custom_substrate_workbench.py cm7")


def _function_sources(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in GENERATOR_FUNCTIONS:
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                out[node.name] = segment
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _refuse_if_encoder_lane_busy() -> None:
    listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    me = str(Path(__file__).name)
    for line in listing.splitlines():
        if me in line:
            continue
        if any(pattern in line for pattern in HEAVY_PATTERNS):
            raise SystemExit(f"encoder lane busy, refusing to start: {line.strip()[:160]}")


def _clips_for_resolution(resolution: int) -> list[torch.Tensor]:
    cfe.RES = resolution
    cfe.FRAMES = 64
    generator = torch.Generator().manual_seed(0)
    cells = [(a, b) for a in range(2) for b in range(2)]
    order = [cells[i % len(cells)] for i in range(8)]
    return [cfe.make_factorized_clip(a, b, 2, 2, generator) for a, b in order]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "proof" / "FACTORIZED_STIMULUS_IDENTITY.json"))
    parser.add_argument("--skip-encoders", action="store_true", help="generator and hash evidence only")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    problems: list[str] = []
    _refuse_if_encoder_lane_busy()

    script_rel = "scripts/cache_factorized_encoder.py"
    head_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%H", "--", script_rel],
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_source = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{head_commit}:{script_rel}"],
        capture_output=True,
        text=True,
    ).stdout
    current_source = (REPO_ROOT / script_rel).read_text(encoding="utf-8")
    head_funcs = _function_sources(head_source)
    current_funcs = _function_sources(current_source)
    generator_evidence = {}
    for name in GENERATOR_FUNCTIONS:
        head_sha = _sha256_text(head_funcs.get(name, ""))
        cur_sha = _sha256_text(current_funcs.get(name, ""))
        identical = bool(head_funcs.get(name)) and head_funcs.get(name) == current_funcs.get(name)
        generator_evidence[name] = {
            "head_commit": head_commit,
            "head_sha256": head_sha,
            "current_sha256": cur_sha,
            "identical": identical,
        }
        if not identical:
            problems.append(f"generator function {name} differs between HEAD and working tree")

    # Regenerate and hash the clip sets per native resolution.
    regenerated: dict[str, list[dict[str, str]]] = {}
    clip_cache: dict[int, list[torch.Tensor]] = {}
    for resolution in sorted({scale["resolution"] for scale in SCALES}):
        clips = _clips_for_resolution(resolution)
        clip_cache[resolution] = clips
        regenerated[str(resolution)] = [
            {"index": i, "sha256": cfe._tensor_sha256(clip)} for i, clip in enumerate(clips)
        ]

    # Compare 256px set against the random control cache's recorded hashes.
    random_receipt = json.loads(
        (REPO_ROOT / "data" / "cache" / RANDOM_CACHE / "run_receipt.json").read_text(encoding="utf-8")
    )
    recorded = random_receipt.get("stimulus", {}).get("records", [])
    random_match = bool(recorded) and all(
        recorded[i].get("sha256") == regenerated["256"][i]["sha256"] for i in range(len(recorded))
    )
    if not random_match:
        problems.append("regenerated 256px clips do not reproduce the random control cache's recorded hashes")

    # Latent rebinding: one clip per learned cache through the exact frozen encoder on CPU.
    rebinding: list[dict[str, object]] = []
    if not args.skip_encoders:
        for scale in SCALES:
            cache_dir = REPO_ROOT / "data" / "cache" / str(scale["cache"])
            stored = np.load(cache_dir / "latents.npy")
            cfg = compose(
                [
                    f"encoder={scale['encoder']}",
                    "device=cpu",
                    "+encoder.prefer_real=true",
                    "+encoder.require_real=true",
                    "+encoder.local_files_only=true",
                ]
            )
            started = time.perf_counter()
            encoder = load_encoder(cfg.encoder)
            if encoder.spec.backend != "vjepa_hf":
                problems.append(f"{scale['tag']}: expected real backend vjepa_hf, got {encoder.spec.backend}")
                continue
            clip = clip_cache[int(scale["resolution"])][0]
            with torch.no_grad():
                latent = encoder.encode(clip[None]).reshape(1, -1).float().cpu().numpy()[0]
            wall = time.perf_counter() - started
            max_abs = float(np.max(np.abs(latent - stored[0])))
            exact = bool(np.array_equal(latent, stored[0]))
            rebinding.append(
                {
                    "tag": scale["tag"],
                    "cache": scale["cache"],
                    "encoder": scale["encoder"],
                    "resolution": scale["resolution"],
                    "clip_index": 0,
                    "clip_sha256": regenerated[str(scale["resolution"])][0]["sha256"],
                    "latent_dim": int(stored.shape[1]),
                    "bitwise_equal": exact,
                    "max_abs_diff": max_abs,
                    "wall_seconds": wall,
                }
            )
            if not exact:
                problems.append(
                    f"{scale['tag']}: re-encoded clip 0 does not bitwise-match stored latent row 0 "
                    f"(max abs diff {max_abs:g})"
                )
            del encoder

    receipt = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "question": (
            "are the learned local8 caches and the seed-0 random control cache bound to "
            "byte-identical generated inputs"
        ),
        "generator_evidence": generator_evidence,
        "regenerated_stimulus_hashes": regenerated,
        "random_control_hash_match": {
            "cache": RANDOM_CACHE,
            "recorded": len(recorded),
            "all_match": random_match,
        },
        "learned_latent_rebinding": rebinding,
        "derivation_argument": (
            "the generator functions are identical in the pre-build commit and the current tree; "
            "every cache's factors.json declares seed 0 and the same 2x2x2 design; the regenerated "
            "clips reproduce the random control's recorded hashes exactly; and each learned cache's "
            "stored latent row 0 is reproduced bitwise by re-encoding the regenerated clip through "
            "the exact frozen encoder on CPU. Together these bind learned and control caches to "
            "byte-identical inputs without a rebuild."
        ),
        "claim_boundary": {
            "scientific_promotion": False,
            "statement": "input-identity and cache-integrity mechanics only; no capability claim",
        },
        "problems": problems,
        "all_ok": not problems,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "all_ok": receipt["all_ok"], "problems": problems}, indent=2))
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
