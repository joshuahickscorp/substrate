from __future__ import annotations
import numpy as np
import torch
from mop.temporal import io
PRINCIPAL = ("har_stream", "speech_stream")
_CACHE: dict = {}
BED_IDENTITY_FIELDS = (
    "bed",
    "task_identity",
    "unit_kind",
    "sequence_length",
    "segment_length",
    "segments_per_stream",
    "channels",
    "classes",
    "n_train",
    "n_test",
    "n_units_train",
    "n_units_test",
    "data_hash",
)
def load(name: str) -> dict:
    if name in _CACHE:
        return _CACHE[name]
    if name in PRINCIPAL:
        from fastforge import data as D
        d = D.domain(name)
        d = dict(d)
        d["segments_per_stream"] = int(d.get("sequences_per_stream") or d.get("clips_per_stream") or 3)
    elif name == "pamap2_stream":
        d = _pamap2_stream()
    elif name == "harth_stream":
        d = _harth_stream()
    else:
        raise KeyError(name)
    d["segment_length"] = int(d["x"].shape[1] // d["segments_per_stream"])
    d["boundaries"] = [d["segment_length"] * i for i in range(1, d["segments_per_stream"])]
    _CACHE[name] = d
    return d
def identity(name: str) -> dict:
    d = load(name)
    h = io.sha_obj({
        "x": [float(d["x"][:64].sum()), float(d["x"].std())],
        "y": [int(d["y"][:64].sum()), int(d["y"].max())],
        "shape": list(d["x"].shape),
    })
    return {
        "bed": name,
        "task_identity": "stream classification: name the sequence that ends the stream",
        "unit_kind": d["unit"],
        "sequence_length": int(d["x"].shape[1]),
        "segment_length": int(d["segment_length"]),
        "segments_per_stream": int(d["segments_per_stream"]),
        "channels": int(d["channels"]),
        "classes": int(d["classes"]),
        "n_train": int(len(d["x"])),
        "n_test": int(len(d["xte"])),
        "n_units_train": int(len(set(np.asarray(d["u"]).tolist()))),
        "n_units_test": int(len(set(np.asarray(d["ute"]).tolist()))),
        "data_hash": h,
    }
def splits(name: str, seed: int, tune_frac: float = 0.2) -> dict:
    d = load(name)
    u = np.asarray(d["u"])
    uniq = np.unique(u)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(uniq)
    n_tun = max(1, int(tune_frac * len(uniq)))
    tune_u, main_u = perm[:n_tun], perm[n_tun:]
    mi = np.where(np.isin(u, list(main_u)))[0]
    ti = np.where(np.isin(u, list(tune_u)))[0]
    return {
        "bed": name,
        "main": (d["x"][mi], d["y"][mi]),
        "main_units": u[mi],
        "tune": (d["x"][ti], d["y"][ti]),
        "tune_units": u[ti],
        "test": (d["xte"], d["yte"]),
        "test_units": np.asarray(d["ute"]),
        "channels": int(d["channels"]),
        "classes": int(d["classes"]),
        "unit": d["unit"],
        "sequence_length": int(d["x"].shape[1]),
        "segment_length": int(d["segment_length"]),
        "boundaries": list(d["boundaries"]),
        "units": {"main": main_u.tolist(), "tune": tune_u.tolist(),
                  "test": np.unique(np.asarray(d["ute"])).tolist()},
    }
def majority_rate(y) -> float:
    y = torch.as_tensor(y)
    return float(torch.bincount(y).max()) / len(y)
def chance_rate(classes: int) -> float:
    return 1.0 / classes
def _stream_from(X, Y, U, per_stream: int, n_streams: int, decim: int, seed: int):
    by_unit: dict = {}
    for i, u in enumerate(U):
        by_unit.setdefault(str(u), []).append(i)
    usable = [u for u, ix in by_unit.items() if len(ix) >= per_stream]
    rng = np.random.default_rng(seed)
    xs, ys, us = [], [], []
    for _ in range(n_streams):
        if not usable:
            break
        u = usable[int(rng.integers(len(usable)))]
        pick = rng.choice(by_unit[u], per_stream, replace=False)
        xs.append(np.concatenate([X[i] for i in pick])[::decim])
        ys.append(int(Y[pick[-1]]))
        us.append(u)
    return np.stack(xs).astype(np.float32), np.array(ys), np.array(us)
def _windows(sig: np.ndarray, act: np.ndarray, subj, win: int, stride: int, decim: int):
    X, Y, U = [], [], []
    for s in range(0, len(sig) - win, stride):
        seg = act[s : s + win]
        if seg[0] == 0 or not np.all(seg == seg[0]):
            continue
        X.append(sig[s : s + win : decim])
        Y.append(int(seg[0]))
        U.append(subj)
    return X, Y, U
def _pamap2_stream(per_stream: int = 3, n_train: int = 4000):
    cache = io.DATA_ROOT / "pamap2" / "pamap2_stream.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        Xtr, Ytr, Utr, Xte, Yte, Ute = (z[k] for k in ("Xtr", "Ytr", "Utr", "Xte", "Yte", "Ute"))
    else:
        root = io.DATA_ROOT / "pamap2" / "PAMAP2_Dataset" / "Protocol"
        cols = [4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46]
        allX, allY, allU = [], [], []
        for f in sorted(root.glob("subject*.dat")):
            sid = int(f.name.split("subject")[1][:3])
            d = np.loadtxt(f)
            act = d[:, 1].astype(int)
            sig = d[:, cols]
            for c in range(sig.shape[1]):
                col = sig[:, c]
                m = np.isnan(col)
                if m.any():
                    idx = np.where(~m)[0]
                    if len(idx):
                        col[m] = np.interp(np.where(m)[0], idx, col[idx])
                    sig[:, c] = col
            sig = np.nan_to_num(sig).astype(np.float32)
            X, Y, U = _windows(sig, act, sid, win=256, stride=256, decim=4)
            allX += X
            allY += Y
            allU += U
        allY = np.array(allY)
        keep = np.unique(allY)[np.argsort(-np.bincount(allY)[np.unique(allY)])][:6]
        sel = [i for i, y in enumerate(allY) if y in keep]
        remap = {int(c): i for i, c in enumerate(sorted(keep))}
        X = [allX[i] for i in sel]
        Y = np.array([remap[int(allY[i])] for i in sel])
        U = np.array([allU[i] for i in sel])
        mu, sd = np.mean([x.mean(0) for x in X], 0), np.std([x.mean(0) for x in X], 0) + 1e-6
        X = [(x - mu) / sd for x in X]
        units = np.unique(U)
        te_u = set(units[: max(1, len(units) // 3)].tolist())
        tri = [i for i, u in enumerate(U) if u not in te_u]
        tei = [i for i, u in enumerate(U) if u in te_u]
        Xtr, Ytr, Utr = _stream_from([X[i] for i in tri], Y[tri], U[tri], per_stream, n_train, 2, 0)
        Xte, Yte, Ute = _stream_from([X[i] for i in tei], Y[tei], U[tei], per_stream, n_train // 3, 2, 1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, Xtr=Xtr, Ytr=Ytr, Utr=Utr, Xte=Xte, Yte=Yte, Ute=Ute)
    return {
        "x": torch.tensor(Xtr), "y": torch.tensor(Ytr), "u": Utr,
        "xte": torch.tensor(Xte), "yte": torch.tensor(Yte), "ute": Ute,
        "channels": int(Xtr.shape[2]), "classes": int(Ytr.max() + 1), "unit": "subject",
        "segments_per_stream": per_stream,
    }
def _harth_stream(per_stream: int = 3, n_train: int = 4000):
    import csv
    cache = io.DATA_ROOT / "harth" / "harth_stream.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        Xtr, Ytr, Utr, Xte, Yte, Ute = (z[k] for k in ("Xtr", "Ytr", "Utr", "Xte", "Yte", "Ute"))
    else:
        root = io.DATA_ROOT / "harth" / "harth"
        if not root.is_dir():
            root = io.DATA_ROOT / "harth"
        allX, allY, allU = [], [], []
        for f in sorted(root.rglob("*.csv")):
            sid = f.stem
            rows, labs = [], []
            cols = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
            with open(f, newline="") as fh:
                rd = csv.DictReader(fh)
                if not set(cols) <= set(rd.fieldnames or []):
                    continue
                for r in rd:
                    try:
                        rows.append([float(r[c]) for c in cols])
                        labs.append(int(float(r["label"])))
                    except (ValueError, KeyError, TypeError):
                        continue
            if len(rows) < 2000:
                continue
            sig = np.asarray(rows, dtype=np.float32)
            act = np.asarray(labs, dtype=int)
            X, Y, U = _windows(sig, act, sid, win=250, stride=250, decim=5)
            allX += X
            allY += Y
            allU += U
        allY = np.array(allY)
        counts = np.bincount(allY)
        keep = np.argsort(-counts)[:6]
        keep = [int(c) for c in keep if counts[c] > 50]
        remap = {c: i for i, c in enumerate(sorted(keep))}
        sel = [i for i, y in enumerate(allY) if y in remap]
        X = [allX[i] for i in sel]
        Y = np.array([remap[int(allY[i])] for i in sel])
        U = np.array([allU[i] for i in sel])
        mu, sd = np.mean([x.mean(0) for x in X], 0), np.std([x.mean(0) for x in X], 0) + 1e-6
        X = [(x - mu) / sd for x in X]
        units = np.unique(U)
        te_u = set(units[: max(1, len(units) // 3)].tolist())
        tri = [i for i, u in enumerate(U) if u not in te_u]
        tei = [i for i, u in enumerate(U) if u in te_u]
        Xtr, Ytr, Utr = _stream_from([X[i] for i in tri], Y[tri], U[tri], per_stream, n_train, 2, 0)
        Xte, Yte, Ute = _stream_from([X[i] for i in tei], Y[tei], U[tei], per_stream, n_train // 3, 2, 1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, Xtr=Xtr, Ytr=Ytr, Utr=Utr, Xte=Xte, Yte=Yte, Ute=Ute)
    return {
        "x": torch.tensor(Xtr), "y": torch.tensor(Ytr), "u": Utr,
        "xte": torch.tensor(Xte), "yte": torch.tensor(Yte), "ute": Ute,
        "channels": int(Xtr.shape[2]), "classes": int(Ytr.max() + 1), "unit": "subject",
        "segments_per_stream": per_stream,
    }
