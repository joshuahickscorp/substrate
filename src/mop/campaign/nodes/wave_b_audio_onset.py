"""Wave B native-audio onset localization: spectral flux vs a raw-energy threshold.

WAVE B probes native sensing breadth in the audio modality. We synthesize independent 1-D waveforms
(the experimental units), each carrying a handful of known onset times built from short tonal-plus-noise
amplitude bursts laid over colored broadband noise. The candidate detector is a handcrafted spectral-flux
onset detector: a short-time Fourier magnitude, its positive first difference summed across frequency bins,
then adaptive peak-picking. The NAMED CONTROL is an energy-threshold detector that only ever sees the raw
waveform envelope (short-time RMS) and fires on rising threshold crossings, with no spectral resolution.

Per unit we score onset detection F1 against ground truth within a tolerance window and take the paired delta
candidate_F1 minus control_F1 (positive favors spectral flux). The exact one-sided sign-flip over the units,
a small structural SESOI on F1, and the neutral verdict decide the outcome. A tie or a wrong-direction result
is a legitimate null; nothing here is tuned toward a positive.

failure_domain: onsets whose spectral change is masked by high broadband energy, where the energy envelope
never crosses its adaptive threshold even though the spectrum turns over.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner

# Fixed acquisition and analysis geometry. Chosen once, up front, and never tuned against the outcome.
SR = 8000
DUR_SAMPLES = 12000
NFFT = 256
HOP = 128
N_UNITS = 10
TOL_FRAMES = 3
REFRACTORY_FRAMES = 3
LOCALMAX_HALF = 2
FLUX_K = 1.0
ENERGY_K = 1.0
SESOI = 0.05
TONE_BANK_HZ = (400.0, 800.0, 1200.0, 1800.0, 2400.0, 3000.0)


def _frame_matrix(signal: np.ndarray) -> np.ndarray:
    """Stack Hann-windowed frames of length NFFT hopped by HOP into a (n_frames, NFFT) matrix."""

    n_frames = 1 + (len(signal) - NFFT) // HOP
    window = np.hanning(NFFT)
    frames = np.empty((n_frames, NFFT), dtype=np.float64)
    for t in range(n_frames):
        start = t * HOP
        frames[t] = signal[start : start + NFFT] * window
    return frames


def _spectral_flux(signal: np.ndarray) -> np.ndarray:
    """Positive first difference of STFT magnitude summed across bins: the candidate novelty curve."""

    frames = _frame_matrix(signal)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    diff = mag[1:] - mag[:-1]
    flux = np.clip(diff, 0.0, None).sum(axis=1)
    curve = np.empty(mag.shape[0], dtype=np.float64)
    curve[0] = 0.0
    curve[1:] = flux
    return curve


def _energy_envelope(signal: np.ndarray) -> np.ndarray:
    """Short-time RMS of the raw waveform frames: the control sees only this broadband envelope."""

    frames = _frame_matrix(signal)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _peak_pick(curve: np.ndarray, k: float) -> list[int]:
    """Adaptive peak-pick: accept local maxima above mean + k*std, honoring a refractory gap."""

    if curve.size == 0:
        return []
    threshold = float(curve.mean() + k * curve.std())
    picks: list[int] = []
    last = -(REFRACTORY_FRAMES + 1)
    for t in range(curve.size):
        if curve[t] <= threshold:
            continue
        lo = max(0, t - LOCALMAX_HALF)
        hi = min(curve.size, t + LOCALMAX_HALF + 1)
        if curve[t] < curve[lo:hi].max():
            continue
        if t - last < REFRACTORY_FRAMES:
            continue
        picks.append(t)
        last = t
    return picks


def _threshold_crossings(envelope: np.ndarray, k: float) -> list[int]:
    """Energy-threshold detector: rising crossings of an adaptive envelope threshold, with refractoriness."""

    if envelope.size == 0:
        return []
    threshold = float(envelope.mean() + k * envelope.std())
    picks: list[int] = []
    last = -(REFRACTORY_FRAMES + 1)
    for t in range(1, envelope.size):
        if envelope[t] > threshold >= envelope[t - 1] and t - last >= REFRACTORY_FRAMES:
            picks.append(t)
            last = t
    return picks


def _f1(gt_frames: list[int], det_frames: list[int], tol: int) -> float:
    """Onset F1: greedily match each ground-truth onset to the nearest unused detection within tolerance."""

    used = [False] * len(det_frames)
    tp = 0
    for g in sorted(gt_frames):
        best_i = -1
        best_d = tol + 1
        for i, d in enumerate(det_frames):
            if used[i]:
                continue
            dist = abs(d - g)
            if dist <= tol and dist < best_d:
                best_d = dist
                best_i = i
        if best_i >= 0:
            used[best_i] = True
            tp += 1
    if tp == 0:
        return 0.0
    fp = len(det_frames) - tp
    fn = len(gt_frames) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def _synthesize(gen: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """Build one waveform: colored broadband noise plus tonal-noise onset bursts. Return signal and frames."""

    # Colored (one-pole low-passed) broadband background. Broadband energy is what masks quiet onsets.
    white = gen.standard_normal(DUR_SAMPLES)
    background = np.empty(DUR_SAMPLES, dtype=np.float64)
    alpha = 0.85
    acc = 0.0
    for n in range(DUR_SAMPLES):
        acc = alpha * acc + (1.0 - alpha) * white[n]
        background[n] = acc
    background *= 0.6 / (background.std() + 1e-12)

    signal = background.copy()
    n_onsets = int(gen.integers(6, 11))
    min_gap = 1200
    onset_samples: list[int] = []
    attempts = 0
    while len(onset_samples) < n_onsets and attempts < 200:
        attempts += 1
        cand = int(gen.integers(NFFT, DUR_SAMPLES - 800))
        if all(abs(cand - s) >= min_gap for s in onset_samples):
            onset_samples.append(cand)
    onset_samples.sort()

    time_axis = np.arange(DUR_SAMPLES) / SR
    for s in onset_samples:
        freq = float(gen.choice(TONE_BANK_HZ))
        # Deliberately wide amplitude spread: some onsets are loud, some are quiet but spectrally distinct.
        amp = float(gen.uniform(0.15, 0.9))
        tau = float(gen.uniform(0.03, 0.07))
        length = min(int(0.18 * SR), DUR_SAMPLES - s)
        idx = np.arange(length)
        env = np.exp(-idx / (tau * SR))
        tone = np.sin(2.0 * np.pi * freq * time_axis[s : s + length])
        transient = 0.3 * gen.standard_normal(length)
        signal[s : s + length] += amp * env * (tone + transient)

    gt_frames = sorted({int(round(s / HOP)) for s in onset_samples})
    return signal, gt_frames


@register_runner("wave_b.native_audio_onset")
def wave_b_audio_onset_runner(params: dict, ctx: NodeContext) -> RunResult:
    """Spectral-flux onset detector vs an energy-threshold control across synthetic native-audio units."""

    n_units = int(params.get("n_units", N_UNITS))

    per_unit: list[dict[str, Any]] = []
    deltas: list[float] = []
    for u in range(n_units):
        gen = rng(ctx.seed, "wave_b_audio_onset", u)
        signal, gt_frames = _synthesize(gen)

        flux = _spectral_flux(signal)
        envelope = _energy_envelope(signal)
        cand_det = _peak_pick(flux, FLUX_K)
        ctrl_det = _threshold_crossings(envelope, ENERGY_K)

        cand_f1 = _f1(gt_frames, cand_det, TOL_FRAMES)
        ctrl_f1 = _f1(gt_frames, ctrl_det, TOL_FRAMES)
        delta = cand_f1 - ctrl_f1
        deltas.append(delta)
        per_unit.append(
            {
                "unit_id": f"wave_b_audio_onset_u{u:02d}",
                "n_onsets": len(gt_frames),
                "candidate_f1": round(cand_f1, 9),
                "control_f1": round(ctrl_f1, 9),
                "delta": round(delta, 9),
            }
        )

    sign_flip = exact_sign_flip_one_sided(deltas)
    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], SESOI)
    is_null = verdict != "survives"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_b_audio_onset/v1",
        {
            "form_family": "native_audio",
            "phenomenon": "temporal_boundary",
            "mechanism_family": "spectral_flux_onset_detection",
            "unit_class": "synthetic_1d_waveform",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "control": (
                "energy-threshold detector on the raw waveform short-time RMS envelope: rising crossings of "
                "an adaptive mean-plus-std threshold, no spectral flux and no per-bin spectral resolution"
            ),
            "candidate": (
                "handcrafted spectral-flux onset detector: STFT magnitude positive first difference summed "
                "across bins, adaptive local-maximum peak-pick above a mean-plus-std threshold"
            ),
            "geometry": {
                "sr": SR,
                "dur_samples": DUR_SAMPLES,
                "nfft": NFFT,
                "hop": HOP,
                "tol_frames": TOL_FRAMES,
                "refractory_frames": REFRACTORY_FRAMES,
                "flux_k": FLUX_K,
                "energy_k": ENERGY_K,
                "n_units": n_units,
            },
            "sesoi": SESOI,
            "per_unit": per_unit,
            "sign_flip": sign_flip,
            "verdict": verdict,
            "is_null": is_null,
            "alternative_explanation": (
                "any candidate advantage may reflect the specific synthesis choices (tonal bursts over "
                "low-passed colored noise, this amplitude spread and tolerance window) rather than a general "
                "superiority of spectral flux for native-audio onset localization"
            ),
            "failure_domain": (
                "onsets masked by broadband energy: when background energy is high, quiet but spectrally "
                "distinct bursts never lift the RMS envelope over its threshold, so the energy control drops "
                "onsets that spectral flux can still resolve"
            ),
        }
    )

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        str(path),
        seal,
        verdict,
        is_null,
        {
            "mean_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units": sign_flip["n_units"],
            "n_units_favorable": sign_flip["n_units_favorable"],
        },
    )
