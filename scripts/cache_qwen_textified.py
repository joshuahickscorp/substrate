#!/usr/bin/env python

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cache_randominit_vitl_features import assert_encoder_lane_free  # noqa: E402
from scripts.featurize_programmatic import (  # noqa: E402
    CACHE_ROOT,
    CLIPSET,
    clip_checksum,
    guard_post_encode,
    iter_bound_nuisance_clips,
    read_encode_params,
    to_uint8,
    write_factors_sidecar,
)

from mop.config import REPO_ROOT  # noqa: E402
from mop.diagnostics import linear_probe  # noqa: E402
from mop.substrate import LatentStore  # noqa: E402

STAGING_MANIFEST = REPO_ROOT / "runs" / "mot" / "staging_manifest.json"
STORE_PREFIX = "qwen05b_textified"
TSUB, GRID = 8, 4
PALETTE = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.45, 0.45, 0.45),
    "red": (0.9, 0.15, 0.15),
    "green": (0.15, 0.8, 0.2),
    "blue": (0.15, 0.25, 0.9),
    "yellow": (0.9, 0.85, 0.15),
    "magenta": (0.85, 0.2, 0.8),
    "cyan": (0.2, 0.8, 0.85),
}


def textify_clip(frames_u8: torch.Tensor, tsub: int = TSUB, grid: int = GRID) -> str:
    t = frames_u8.shape[0]
    frames = frames_u8[:: max(1, t // tsub)][:tsub].float() / 255.0  # [t,3,H,W]
    names = list(PALETTE)
    pal = torch.tensor([PALETTE[k] for k in names])  # [P,3]
    lines = []
    for f in range(frames.shape[0]):
        h, w = frames.shape[2], frames.shape[3]
        cells = frames[f, :, : (h // grid) * grid, : (w // grid) * grid]
        cells = cells.reshape(3, grid, h // grid, grid, w // grid).mean(dim=(2, 4))  # [3,grid,grid]
        flat = cells.permute(1, 2, 0).reshape(-1, 3)  # [grid*grid, 3]
        idx = ((flat[:, None, :] - pal[None]) ** 2).sum(-1).argmin(dim=1)  # nearest palette color
        bright = flat.mean(dim=1).argmax()
        rows = [" ".join(names[int(idx[r * grid + c])] for c in range(grid)) for r in range(grid)]
        lines.append(
            f"frame {f}: " + " | ".join(rows) + f" ; brightest cell row {int(bright) // grid} "
            f"col {int(bright) % grid}"
        )
    return "scene report.\n" + "\n".join(lines)


def resolve_model_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    if STAGING_MANIFEST.exists():
        return json.loads(STAGING_MANIFEST.read_text())["models"]["text_encoder"]["repo_id"]
    raise SystemExit(f"{STAGING_MANIFEST} missing: run scripts/stage_small_substrates.py first")


def load_arm(model_id: str, arm: str, seed: int):
    from transformers import AutoConfig, AutoModel

    if arm == "real":
        model = AutoModel.from_pretrained(model_id, local_files_only=True, dtype=torch.float32)
    else:
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
        torch.manual_seed(seed)
        model = AutoModel.from_config(cfg)
    return model.eval()


def mid_layer_mean(model, tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    return hs[len(hs) // 2][0].mean(dim=0).float()


def encode_arm(model_id: str, arm: str, params: dict, cache_root: Path, seed: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = load_arm(model_id, arm, seed)
    dim = int(model.config.hidden_size)
    n = params["n_shape"] * params["n_color"] * params["per"]
    store = LatentStore.create(
        cache_root, f"{STORE_PREFIX}_{arm}", feat_shape=(dim,), capacity=n, key_dim=dim, has_labels=True
    )
    shapes, colors, checks = [], [], {}
    t0 = time.perf_counter()
    for i, s, c, clip, _ in iter_bound_nuisance_clips(
        params["n_shape"], params["n_color"], params["per"], params["seed"]
    ):
        if i == 0:
            checks["first"] = clip_checksum(clip)
        if i == n - 1:
            checks["last"] = clip_checksum(clip)
        frames = to_uint8(clip)
        del clip  # stream as uint8, one clip in memory at a time
        z = mid_layer_mean(model, tokenizer, textify_clip(frames))
        store.write_batch(i, z[None].numpy(), z[None].numpy(), torch.tensor([s]).numpy())
        shapes.append(s)
        colors.append(c)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"[{arm}] encoded {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    store.finalize()
    write_factors_sidecar(
        store.root,
        params,
        shapes,
        colors,
        checks,
        source="cache_qwen_textified",
        model_id=model_id,
        arm=arm,
        tsub=TSUB,
        grid=GRID,
    )
    probe = linear_probe(store.latents(), torch.tensor(shapes), classification=True, epochs=300, seed=seed)
    del model, tokenizer
    gc.collect()
    return {
        "store": str(store.root),
        "embed_dim": dim,
        "shape_probe_acc": round(probe["score"], 4),
        "chance": round(probe["chance"], 4),
        "clip_checksums": checks,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="small-LLM textified nuisance cache (encoder lane)")
    ap.add_argument("--model-id", default=None, help="override; default reads the staging manifest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/cache_qwen.json")
    ap.add_argument("--force", action="store_true", help="skip the post-encode guard (encoder lane free)")
    a = ap.parse_args(argv)
    guard_post_encode(a.force)
    assert_encoder_lane_free()  # live-process mutex: one encoder at a time, never skipped by --force
    model_id = resolve_model_id(a.model_id)
    params = read_encode_params(seed=a.seed)
    arms = {
        arm: encode_arm(model_id, arm, params, Path(a.cache_root), a.seed) for arm in ("real", "randominit")
    }
    out = {
        "experiment": "cache_qwen_textified",
        "clipset": CLIPSET,
        "model_id": model_id,
        "params": params,
        "tsub": TSUB,
        "grid": GRID,
        "arms": arms,
        "note": "label-free textification (pixels only, labels never touch the text); the claim scope "
        "is pretraining-over-random-init within this rendering, nothing more",
    }
    text = json.dumps(out, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
