"""Speech Commands audio domain preflight: speaker-disjoint temporal-order gate.

STARSS23 cached features were audited invalid (8 clips, aggregated per-clip latents, no recording/room units,
no temporal sequence). Substituted with Speech Commands v0.02: lawful public, speaker-disjoint via the speaker
hash in each filename, read with stdlib wave + numpy FFT (torchaudio/librosa absent).

Gate: does temporal order matter (GRU beats order-free bag-of-frames, and shuffling frames hurts).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/Users/scammermike/Downloads/mop-substrate-forge")
DATA = ROOT / "integrated/data/speech_commands"
OUT = ROOT / "integrated"
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
PER_WORD = 320
FRAME, HOP, BINS = 400, 160, 40  # 25ms frame, 10ms hop at 16k -> ~98 frames
SEEDS = [0, 1, 2, 3, 4]
STEPS, HIDDEN = 800, 128
_C = {}


def _melbank(n_fft_bins=FRAME // 2 + 1, sr=16000, n_mel=BINS):
    """Triangular mel filterbank. Linear frequency pooling is too crude for speech."""

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(20), hz2mel(sr / 2), n_mel + 2))
    bins = np.floor((n_fft_bins - 1) * 2 * edges / sr).astype(int)
    fb = np.zeros((n_mel, n_fft_bins), dtype=np.float32)
    for i in range(n_mel):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid == lo:
            mid = lo + 1
        if hi <= mid:
            hi = mid + 1
        fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


_FB = _melbank()


def sha_obj(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) < 16000:
        x = np.pad(x, (0, 16000 - len(x)))
    return x[:16000]


def features(x: np.ndarray) -> np.ndarray:
    """Frame the waveform and bin FFT magnitude: (frames, BINS). Temporal order preserved."""
    win = np.hanning(FRAME).astype(np.float32)
    out = []
    for s in range(0, 16000 - FRAME + 1, HOP):
        mag = np.abs(np.fft.rfft(x[s : s + FRAME] * win))
        out.append(np.log1p(_FB @ mag))
    return np.array(out, dtype=np.float32)


def load():
    if "d" in _C:
        return _C["d"]
    X, Y, S = [], [], []
    for wi, word in enumerate(WORDS):
        d = DATA / word
        paths = sorted(d.glob("*.wav"))[:PER_WORD]
        for p in paths:
            spk = p.name.split("_")[0]
            try:
                X.append(features(read_wav(p)))
            except Exception:
                continue
            Y.append(wi)
            S.append(spk)
    X = np.array(X, dtype=np.float32)
    mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6
    X = (X - mu) / sd
    _C["d"] = (torch.tensor(X), torch.tensor(np.array(Y)), np.array(S))
    return _C["d"]


class GRU(nn.Module):
    def __init__(s, ch, no):
        super().__init__()
        s.g = nn.GRU(ch, HIDDEN, batch_first=True)
        s.h = nn.Linear(HIDDEN, no)

    def forward(s, x):
        o, _ = s.g(x)
        return s.h(o[:, -1])


class Bag(nn.Module):
    def __init__(s, ch, no):
        super().__init__()
        s.n = nn.Sequential(
            nn.Linear(ch * 3, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, no)
        )

    def forward(s, x):
        return s.n(torch.cat([x.mean(1), x.std(1), x.amax(1)], 1))


def train_eval(Model, tr, te, X, Y, seed, transform=None):
    torch.manual_seed(seed)
    m = Model(X.shape[2], len(WORDS))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    rng = np.random.default_rng(seed)
    xt = X[tr] if transform is None else transform(X[tr], rng)
    yt = Y[tr]
    for _ in range(STEPS):
        bi = rng.choice(len(xt), min(128, len(xt)), replace=False)
        opt.zero_grad()
        F.cross_entropy(m(xt[bi]), yt[bi]).backward()
        opt.step()
    m.eval()
    xe = X[te] if transform is None else transform(X[te], rng)
    with torch.no_grad():
        pred = torch.cat([m(xe[k : k + 256]).argmax(1) for k in range(0, len(xe), 256)])
    return float((pred == Y[te]).float().mean())


def shuffle_t(X, rng):
    return X[:, torch.tensor(rng.permutation(X.shape[1])), :]


def main():
    t0 = time.time()
    X, Y, S = load()
    spk = np.unique(S)
    rng0 = np.random.default_rng(0)
    tr_spk = set(rng0.choice(spk, int(0.7 * len(spk)), replace=False).tolist())
    tr = np.where(np.isin(S, list(tr_spk)))[0]
    te = np.where(~np.isin(S, list(tr_spk)))[0]
    gc = np.array([train_eval(GRU, tr, te, X, Y, s) for s in SEEDS])
    sh = np.array([train_eval(GRU, tr, te, X, Y, s, shuffle_t) for s in SEEDS])
    bg = np.array([train_eval(Bag, tr, te, X, Y, s) for s in SEEDS])
    th = float((gc - bg).mean())
    th_lcb = float((gc - bg).mean() - 1.833 * (gc - bg).std(ddof=1) / np.sqrt(len(SEEDS)))
    om = float((gc - sh).mean())
    verdict = (
        "temporal_headroom_present" if (th_lcb >= 0.03 and om > 0.02) else "invalid_no_temporal_headroom"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    out = {
        "schema": "mop-substrate-audio-preflight/v2-melbank",
        "domain": "SpeechCommands_v0.02",
        "substitution_reason": "STARSS23 cached features invalid: 8 clips, aggregated per-clip latents, "
        "no recording/room/site-disjoint units, no temporal sequence",
        "rights": "Speech Commands v0.02, public (CC BY 4.0), Google",
        "source_commit": commit,
        "units": "speaker-disjoint (speaker hash in filename)",
        "n_clips": int(len(X)),
        "frames": int(X.shape[1]),
        "bins": int(X.shape[2]),
        "classes": len(WORDS),
        "n_speakers": int(len(spk)),
        "seeds": SEEDS,
        "gru_correct": round(float(gc.mean()), 4),
        "gru_shuffled": round(float(sh.mean()), 4),
        "bag_order_free": round(float(bg.mean()), 4),
        "temporal_headroom_gru_vs_bag": round(th, 4),
        "temporal_headroom_lcb": round(th_lcb, 4),
        "order_matters_gru_vs_shuffled": round(om, 4),
        "verdict": verdict,
        "wall_seconds": round(time.time() - t0, 1),
        "bounded_repair": "v1 used 20 linear-pooled FFT bins, 300 steps, 64 hidden -> 0.50 accuracy on 10 keywords (standard MFCC+GRU reaches ~0.93). Insufficient estimator capacity, not a domain property. v2 uses a 40-band mel filterbank, 10ms hop, 800 steps, 128 hidden.",
    }
    out["sha256"] = sha_obj(out)
    (OUT / "MOP_SUBSTRATE_AUDIO_REPORT.json").write_text(json.dumps(out, indent=2))
    print(
        f"[audio] {verdict} | gru {gc.mean():.3f} bag {bg.mean():.3f} shuffled {sh.mean():.3f} "
        f"th_lcb {th_lcb:.3f} order {om:.3f} | {len(X)} clips {len(spk)} speakers [{out['wall_seconds']}s]"
    )
    print("AUDIO_DONE")


if __name__ == "__main__":
    main()
