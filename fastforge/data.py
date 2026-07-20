"""Domain providers for the fast-state plasticity forge.

Two validated temporal domains (HAR raw inertial, Speech Commands mel frames) plus a third-domain preflight
hook. Every provider returns sequences of shape (N, T, channels) with natural independent units (HAR subject,
Speech speaker) so splits are unit-disjoint, not row-disjoint.

House style: no dashes.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

HAR_ROOT = Path(
    "/Users/scammermike/Downloads/mop-gen3/runs/generation3/mop-generation3-discovery-v1/data/UCI HAR Dataset"
)
SPEECH_ROOT = Path("/Users/scammermike/Downloads/mop-substrate-forge/integrated/data/speech_commands")
SPEECH_CACHE = Path("/Users/scammermike/Downloads/mop-substrate-forge/integrated/data/speech_feats.npz")
HAR_CH = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
FRAME, HOP, MEL = 400, 160, 40
_CACHE: dict = {}


# ---------------------------------------------------------------- speech features (stdlib wave + numpy FFT)


def _melbank(n_fft_bins=FRAME // 2 + 1, sr=16000, n_mel=MEL):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(20), hz2mel(sr / 2), n_mel + 2))
    bins = np.floor((n_fft_bins - 1) * 2 * edges / sr).astype(int)
    fb = np.zeros((n_mel, n_fft_bins), dtype=np.float32)
    for i in range(n_mel):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        mid = max(mid, lo + 1)
        hi = max(hi, mid + 1)
        fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


_FB = _melbank()


def speech_features(x: np.ndarray) -> np.ndarray:
    win = np.hanning(FRAME).astype(np.float32)
    return np.array(
        [
            np.log1p(_FB @ np.abs(np.fft.rfft(x[s : s + FRAME] * win)))
            for s in range(0, 16000 - FRAME + 1, HOP)
        ],
        dtype=np.float32,
    )


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return np.pad(x, (0, max(0, 16000 - len(x))))[:16000]


# ---------------------------------------------------------------- domain loaders


def load_har():
    """HAR raw inertial windows. Units are subjects; the official train/test split is subject disjoint."""
    if "har" in _CACHE:
        return _CACHE["har"]

    def ld(sp):
        X = np.stack(
            [np.loadtxt(HAR_ROOT / sp / "Inertial Signals" / f"{c}_{sp}.txt") for c in HAR_CH], -1
        ).astype(np.float32)
        y = np.loadtxt(HAR_ROOT / sp / f"y_{sp}.txt").astype(int) - 1
        subj = np.loadtxt(HAR_ROOT / sp / f"subject_{sp}.txt").astype(int)
        return X, y, subj

    Xtr, ytr, str_ = ld("train")
    Xte, yte, ste = ld("test")
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    d = {
        "x": torch.tensor((Xtr - mu) / sd),
        "y": torch.tensor(ytr),
        "u": str_,
        "xte": torch.tensor((Xte - mu) / sd),
        "yte": torch.tensor(yte),
        "ute": ste,
        "channels": 9,
        "classes": 6,
        "unit": "subject",
    }
    _CACHE["har"] = d
    return d


def load_speech():
    """Speech Commands mel frames. Units are speakers (hash prefix of the filename)."""
    if "speech" in _CACHE:
        return _CACHE["speech"]
    if SPEECH_CACHE.exists():
        z = np.load(SPEECH_CACHE, allow_pickle=True)
        X, Y, S = z["X"], z["Y"], z["S"]
    else:
        X, Y, S = [], [], []
        for wi, word in enumerate(WORDS):
            for p in sorted((SPEECH_ROOT / word).glob("*.wav"))[:320]:
                X.append(speech_features(_read_wav(p)))
                Y.append(wi)
                S.append(p.name.split("_")[0])
        X = np.array(X, dtype=np.float32)
        mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6
        X, Y, S = (X - mu) / sd, np.array(Y), np.array(S)
        np.savez(SPEECH_CACHE, X=X, Y=Y, S=S)
    X, Y, S = torch.tensor(np.asarray(X)), torch.tensor(np.asarray(Y)), np.asarray(S)
    spk = np.unique(S)
    rng = np.random.default_rng(0)
    te_spk = set(rng.choice(spk, max(1, int(0.3 * len(spk))), replace=False).tolist())
    te = np.where(np.isin(S, list(te_spk)))[0]
    tr = np.where(~np.isin(S, list(te_spk)))[0]
    d = {
        "x": X[tr], "y": Y[tr], "u": S[tr],
        "xte": X[te], "yte": Y[te], "ute": S[te],
        "channels": int(X.shape[2]), "classes": len(WORDS), "unit": "speaker",
    }
    _CACHE["speech"] = d
    return d


PAMAP2_PROTO = Path(
    "/Users/scammermike/Downloads/mop-autonomous-substrate-evolution/runs/substrate/"
    "mop-autonomous-substrate-evolution-v1/data/pamap2/PAMAP2_Dataset/Protocol"
)
PAMAP2_CACHE = Path(__file__).resolve().parents[1] / "runs/substrate/data/pamap2_transition.npz"
PAMAP2_COLS = [4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46]
PAM_WIN, PAM_STRIDE, PAM_HORIZON, PAM_DECIM = 256, 128, 200, 2


def load_pamap2_transition():
    """Third temporal domain, redesigned. Not window classification.

    The task is next activity prediction across continuous streams: given a window of raw IMU signal, name
    the activity two seconds after the window ends. Windows that straddle a transition are kept, not dropped,
    because the transition is the signal. That makes the task depend on sequence state and on the direction
    of change inside the window, which a bag of timesteps cannot see.
    """
    if "pamap2" in _CACHE:
        return _CACHE["pamap2"]
    if PAMAP2_CACHE.exists():
        z = np.load(PAMAP2_CACHE)
        X, Y, S, Tr = z["X"], z["Y"], z["S"], z["Tr"]
    else:
        X, Y, S, Tr = [], [], [], []
        for f in sorted(PAMAP2_PROTO.glob("subject*.dat")):
            sid = int(f.name.split("subject")[1][:3])
            d = np.loadtxt(f)
            act = d[:, 1].astype(int)
            sig = d[:, PAMAP2_COLS]
            for c in range(sig.shape[1]):
                col = sig[:, c]
                mask = np.isnan(col)
                if mask.any():
                    idx = np.where(~mask)[0]
                    if len(idx):
                        col[mask] = np.interp(np.where(mask)[0], idx, col[idx])
                    sig[:, c] = col
            sig = np.nan_to_num(sig)
            end = len(sig) - PAM_WIN - PAM_HORIZON
            for st in range(0, max(0, end), PAM_STRIDE):
                label = act[st + PAM_WIN + PAM_HORIZON]
                now = act[st + PAM_WIN - 1]
                if label == 0 or now == 0:
                    continue
                X.append(sig[st : st + PAM_WIN : PAM_DECIM])
                Y.append(int(label))
                S.append(sid)
                Tr.append(int(label != now))
        X, Y, S, Tr = np.array(X, np.float32), np.array(Y), np.array(S), np.array(Tr)
        acts, counts = np.unique(Y, return_counts=True)
        keep = acts[np.argsort(-counts)[:6]]
        m = np.isin(Y, keep)
        X, Y, S, Tr = X[m], Y[m], S[m], Tr[m]
        remap = {a: i for i, a in enumerate(sorted(keep))}
        Y = np.array([remap[y] for y in Y])
        mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6
        X = (X - mu) / sd
        PAMAP2_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(PAMAP2_CACHE, X=X, Y=Y, S=S, Tr=Tr)
    subj = np.unique(S)
    te_s = set(subj[::3].tolist())  # subject disjoint test split
    te = np.where(np.isin(S, list(te_s)))[0]
    tr = np.where(~np.isin(S, list(te_s)))[0]
    d = {
        "x": torch.tensor(X[tr]), "y": torch.tensor(Y[tr]), "u": S[tr],
        "xte": torch.tensor(X[te]), "yte": torch.tensor(Y[te]), "ute": S[te],
        "channels": int(X.shape[2]), "classes": int(Y.max()) + 1, "unit": "subject",
        "transition_fraction": float(Tr.mean()),
    }
    _CACHE["pamap2"] = d
    return d


HARTH_ROOT = Path(__file__).resolve().parents[1] / "runs/substrate/data/harth/harth"
HARTH_CACHE = Path(__file__).resolve().parents[1] / "runs/substrate/data/harth_transition.npz"
HARTH_WIN, HARTH_STRIDE, HARTH_HORIZON = 150, 50, 100  # 3 s window, 1 s stride, 2 s prediction horizon


def load_harth_transition():
    """Third temporal domain: HARTH free living continuous accelerometry, next activity prediction.

    HARTH records unscripted daily activity, so activity changes are frequent and are not separated by a
    rest label the way a scripted protocol separates them. Given a three second window of back and thigh
    accelerometer signal, name the activity two seconds after the window ends. Windows that straddle a
    change are kept.
    """
    if "harth" in _CACHE:
        return _CACHE["harth"]
    if HARTH_CACHE.exists():
        z = np.load(HARTH_CACHE)
        X, Y, S, Tr = z["X"], z["Y"], z["S"], z["Tr"]
    else:
        X, Y, S, Tr = [], [], [], []
        for f in sorted(HARTH_ROOT.glob("S*.csv")):
            rows = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7))
            sig, act = rows[:, :6].astype(np.float32), rows[:, 6].astype(int)
            sig = np.nan_to_num(sig)
            sid = f.stem
            for st in range(0, len(sig) - HARTH_WIN - HARTH_HORIZON, HARTH_STRIDE):
                label = act[st + HARTH_WIN + HARTH_HORIZON]
                now = act[st + HARTH_WIN - 1]
                X.append(sig[st : st + HARTH_WIN])
                Y.append(int(label))
                S.append(sid)
                Tr.append(int(label != now))
        X, Y, S, Tr = np.array(X, np.float32), np.array(Y), np.array(S), np.array(Tr)
        acts, counts = np.unique(Y, return_counts=True)
        keep = acts[np.argsort(-counts)[:6]]
        m = np.isin(Y, keep)
        X, Y, S, Tr = X[m], Y[m], S[m], Tr[m]
        remap = {a: i for i, a in enumerate(sorted(keep))}
        Y = np.array([remap[y] for y in Y])
        mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6
        X = (X - mu) / sd
        HARTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(HARTH_CACHE, X=X, Y=Y, S=S, Tr=Tr)
    # Bounded repair. At natural prevalence only 5 percent of windows straddle a change, so the label is
    # almost always the current activity and the task collapses back into window classification, which the
    # order gate correctly rejected. Balancing transition and non transition windows makes the question the
    # transition itself. This changes the sampling, not the labels, the signal or the units.
    keep_tr = np.where(Tr == 1)[0]
    keep_no = np.where(Tr == 0)[0]
    keep_no = np.random.default_rng(0).choice(keep_no, min(len(keep_no), len(keep_tr)), replace=False)
    sel = np.sort(np.concatenate([keep_tr, keep_no]))
    X, Y, S, Tr = X[sel], Y[sel], S[sel], Tr[sel]
    subj = np.unique(S)
    te_s = set(subj[::3].tolist())
    te = np.where(np.isin(S, list(te_s)))[0]
    tr = np.where(~np.isin(S, list(te_s)))[0]
    d = {
        "x": torch.tensor(X[tr]), "y": torch.tensor(Y[tr]), "u": S[tr],
        "xte": torch.tensor(X[te]), "yte": torch.tensor(Y[te]), "ute": S[te],
        "channels": int(X.shape[2]), "classes": int(Y.max()) + 1, "unit": "subject",
        "transition_fraction": float(Tr.mean()),
        "balanced_repair": "transition and non transition windows sampled at equal rate",
    }
    _CACHE["harth"] = d
    return d


def load_speech_stream(per_stream: int = 3, n_streams: int = 6000):
    """Third domain attempt three: speaker disjoint continuous audio.

    Several keyword clips from one speaker are concatenated into a stream and the label is the keyword that
    ends the stream. An order free reader sees the same set of frames whichever keyword came last, so the
    task depends on position in the sequence rather than on content alone. Units stay speaker disjoint
    because every clip in a stream comes from the same speaker.
    """
    if "speech_stream" in _CACHE:
        return _CACHE["speech_stream"]
    base = load_speech()
    parts = []
    for split, xk, yk, uk in (("train", "x", "y", "u"), ("test", "xte", "yte", "ute")):
        X, Y, U = base[xk], base[yk], np.asarray(base[uk])
        by_spk: dict[str, list[int]] = {}
        for i, s in enumerate(U):
            by_spk.setdefault(str(s), []).append(i)
        usable = [s for s, ix in by_spk.items() if len(ix) >= per_stream]
        rng = np.random.default_rng(0 if split == "train" else 1)
        xs, ys, us = [], [], []
        target = n_streams if split == "train" else n_streams // 3
        for _ in range(target):
            if not usable:
                break
            s = usable[int(rng.integers(len(usable)))]
            pick = rng.choice(by_spk[s], per_stream, replace=False)
            xs.append(torch.cat([X[i] for i in pick])[::2])
            ys.append(int(Y[pick[-1]]))
            us.append(s)
        parts.append((torch.stack(xs), torch.tensor(ys), np.array(us)))
    (xtr, ytr, utr), (xte, yte, ute) = parts
    d = {"x": xtr, "y": ytr, "u": utr, "xte": xte, "yte": yte, "ute": ute,
         "channels": int(xtr.shape[2]), "classes": int(base["classes"]), "unit": "speaker",
         "frames_per_stream": int(xtr.shape[1]), "clips_per_stream": per_stream}
    _CACHE["speech_stream"] = d
    return d


DOMAINS = {"har": load_har, "speech": load_speech,
           "pamap2_transition": load_pamap2_transition, "harth_transition": load_harth_transition,
           "speech_stream": load_speech_stream}


def domain(name: str):
    return DOMAINS[name]()


# ---------------------------------------------------------------- continual task construction


def tasks(name: str, seed: int, classes_per_task: int = 2):
    """Within-domain continual sequence: disjoint class subsets, unit-disjoint held-out test per task."""
    d = domain(name)
    rng = np.random.default_rng(seed)
    order = rng.permutation(d["classes"])
    out = []
    ytr, yte = d["y"].numpy(), d["yte"].numpy()
    for t in range(d["classes"] // classes_per_task):
        cs = order[t * classes_per_task : (t + 1) * classes_per_task]
        itr = np.concatenate([np.where(ytr == c)[0] for c in cs])
        ite = np.concatenate([np.where(yte == c)[0] for c in cs])
        out.append({"x": d["x"][itr], "y": d["y"][itr], "test": (d["xte"][ite], d["yte"][ite]), "classes": cs})
    return out, d["channels"], d["classes"]


def held_out_units(name: str, seed: int, frac: float = 0.25):
    """Split the training pool by natural unit into an adaptation pool and the rest. Used for future adaptation."""
    d = domain(name)
    u = np.asarray(d["u"])
    uniq = np.unique(u)
    rng = np.random.default_rng(1000 + seed)
    held = set(rng.choice(uniq, max(1, int(frac * len(uniq))), replace=False).tolist())
    mask = np.isin(u, list(held))
    return np.where(~mask)[0], np.where(mask)[0]


def splits(name: str, seed: int, tune_frac: float = 0.2, future_frac: float = 0.25):
    """Unit disjoint split of the training pool into main, tuning and future units.

    main    principal training data
    tune    every tuning signal: gate references, probe losses, oracle selection, headroom estimation
    future  never seen during acquisition, used only for future unit adaptation
    test    the official held out split, untouched by every policy in this program

    Splitting by natural unit and not by row is what makes the tuning signals honest: a policy tuned on rows
    from a training subject would be tuned on the same person it later scores.
    """
    d = domain(name)
    u = np.asarray(d["u"])
    uniq = np.unique(u)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(uniq)
    n_fut = max(1, int(future_frac * len(uniq)))
    n_tun = max(1, int(tune_frac * len(uniq)))
    future_u, tune_u, main_u = perm[:n_fut], perm[n_fut : n_fut + n_tun], perm[n_fut + n_tun :]
    idx = {k: np.where(np.isin(u, list(v)))[0] for k, v in
           (("main", main_u), ("tune", tune_u), ("future", future_u))}
    fut = idx["future"][np.random.default_rng(7000 + seed).permutation(len(idx["future"]))]
    cut = int(0.7 * len(fut))
    return {
        "main": (d["x"][idx["main"]], d["y"][idx["main"]]),
        "tune": (d["x"][idx["tune"]], d["y"][idx["tune"]]),
        "adapt": (d["x"][fut[:cut]], d["y"][fut[:cut]]),
        "adapt_eval": (d["x"][fut[cut:]], d["y"][fut[cut:]]),
        "test": (d["xte"], d["yte"]),
        "unit_counts": {k: int(len(np.unique(u[v]))) for k, v in idx.items()},
        "channels": d["channels"], "classes": d["classes"], "unit": d["unit"],
    }


def mask_timesteps(x: torch.Tensor, frac: float, rng) -> torch.Tensor:
    """Missing-observation stressor: zero a fraction of timesteps."""
    y = x.clone()
    k = max(1, int(frac * x.shape[1]))
    y[:, torch.tensor(rng.choice(x.shape[1], k, replace=False))] = 0.0
    return y
