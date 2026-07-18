"""Wave B native-audio source counting: a spectral peak-count estimator vs a modal fixed-guess control.

WAVE B probes native sensing breadth in the audio modality. Here the question is concurrent-source counting:
how many simultaneous tonal sources are present in a mixture. We synthesize independent 1-D waveforms (the
experimental units), each a sum of a known number (1 to 4) of well-separated pure tones laid over broadband
noise. The candidate is a peak-count estimator on the short-time spectrum: it averages the STFT magnitude
across frames and counts prominent spectral peaks above the noise floor, clamping the estimate into the known
1..4 range. The NAMED CONTROL is a fixed-guess estimator that ignores the waveform entirely and always emits
the modal true count of the dataset (the best single constant predictor).

Per unit the score is the negative absolute count error, so the paired delta is control_error minus
candidate_error (positive favors the peak-count estimator). The exact one-sided sign-flip over the units, a
small structural SESOI in count-error units, and the neutral verdict decide the outcome. A tie or a
wrong-direction result is a legitimate null; nothing here is tuned toward a positive.

failure_domain: mixtures where tones are close in frequency, unequal in amplitude, or buried in noise so that
weak partials never lift a distinct peak above the floor, collapsing the peak count below the true source
count while a loud dataset mode happens to match.

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
DUR_SAMPLES = 8192
NFFT = 512
HOP = 256
N_UNITS = 10
MIN_SOURCES = 1
MAX_SOURCES = 4
NOISE_STD = 0.15
AMP_LO = 0.6
AMP_HI = 1.0
PEAK_MULT = 6.0
MIN_PEAK_SEP = 10
SESOI = 0.25
TONE_BANK_HZ = (350.0, 750.0, 1150.0, 1550.0, 1950.0, 2350.0, 2750.0, 3150.0)


def _frame_matrix(signal: np.ndarray) -> np.ndarray:
    """Stack Hann-windowed frames of length NFFT hopped by HOP into a (n_frames, NFFT) matrix."""

    n_frames = 1 + (len(signal) - NFFT) // HOP
    window = np.hanning(NFFT)
    frames = np.empty((n_frames, NFFT), dtype=np.float64)
    for t in range(n_frames):
        start = t * HOP
        frames[t] = signal[start : start + NFFT] * window
    return frames


def _avg_magnitude(signal: np.ndarray) -> np.ndarray:
    """Time-averaged STFT magnitude spectrum: the surface the candidate counts sources on."""

    frames = _frame_matrix(signal)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    return mag.mean(axis=0)


def _count_peaks(spectrum: np.ndarray) -> int:
    """Count prominent spectral peaks above a median-referenced floor, clamped to the known source range."""

    if spectrum.size == 0:
        return MIN_SOURCES
    floor = float(np.median(spectrum))
    threshold = floor * PEAK_MULT
    picks: list[int] = []
    last = -(MIN_PEAK_SEP + 1)
    for b in range(1, spectrum.size - 1):
        value = float(spectrum[b])
        if value <= threshold:
            continue
        lo = max(0, b - MIN_PEAK_SEP)
        hi = min(spectrum.size, b + MIN_PEAK_SEP + 1)
        if value < float(spectrum[lo:hi].max()):
            continue
        if b - last < MIN_PEAK_SEP:
            continue
        picks.append(b)
        last = b
    return int(min(MAX_SOURCES, max(MIN_SOURCES, len(picks))))


def _synthesize(gen: np.random.Generator) -> tuple[np.ndarray, int]:
    """Build one mixture: k well-separated pure tones over broadband noise. Return signal and true count."""

    k = int(gen.integers(MIN_SOURCES, MAX_SOURCES + 1))
    freqs = gen.choice(np.asarray(TONE_BANK_HZ), size=k, replace=False)
    time_axis = np.arange(DUR_SAMPLES) / SR
    signal = NOISE_STD * gen.standard_normal(DUR_SAMPLES)
    for freq in freqs:
        amp = float(gen.uniform(AMP_LO, AMP_HI))
        phase = float(gen.uniform(0.0, 2.0 * np.pi))
        signal += amp * np.sin(2.0 * np.pi * float(freq) * time_axis + phase)
    return signal, k


@register_runner("wave_b.source_counting")
def wave_b_audio_count_runner(params: dict, ctx: NodeContext) -> RunResult:
    """Spectral peak-count source estimator vs a modal fixed-guess control across native-audio mixtures."""

    n_units = int(params.get("n_units", N_UNITS))

    signals: list[np.ndarray] = []
    true_counts: list[int] = []
    for u in range(n_units):
        gen = rng(ctx.seed, "wave_b_audio_count", u)
        signal, k = _synthesize(gen)
        signals.append(signal)
        true_counts.append(k)

    # Control: the best single constant predictor, the modal true count. Ties break to the smaller count.
    histogram = np.bincount(np.asarray(true_counts), minlength=MAX_SOURCES + 1)
    modal_count = int(np.argmax(histogram))

    per_unit: list[dict[str, Any]] = []
    deltas: list[float] = []
    for u in range(n_units):
        true_k = true_counts[u]
        cand_count = _count_peaks(_avg_magnitude(signals[u]))
        candidate_error = abs(cand_count - true_k)
        control_error = abs(modal_count - true_k)
        delta = float(control_error - candidate_error)
        deltas.append(delta)
        per_unit.append(
            {
                "unit_id": f"wave_b_audio_count_u{u:02d}",
                "true_count": true_k,
                "candidate_count": cand_count,
                "control_count": modal_count,
                "candidate_error": candidate_error,
                "control_error": control_error,
                "delta": round(delta, 9),
            }
        )

    sign_flip = exact_sign_flip_one_sided(deltas)
    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], SESOI)
    is_null = verdict != "survives"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_b_audio_count/v1",
        {
            "form_family": "native_audio",
            "phenomenon": "source_counting",
            "mechanism_family": "count_estimator",
            "unit_class": "synthetic_1d_waveform",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "control": (
                "fixed-guess modal-count estimator: ignores the waveform and always emits the dataset modal "
                "true source count, the best single constant predictor with no spectral analysis"
            ),
            "candidate": (
                "peak-count estimator on the time-averaged STFT magnitude spectrum: counts local-maximum "
                "peaks above a median-referenced floor, clamped into the known 1..4 source range"
            ),
            "geometry": {
                "sr": SR,
                "dur_samples": DUR_SAMPLES,
                "nfft": NFFT,
                "hop": HOP,
                "min_sources": MIN_SOURCES,
                "max_sources": MAX_SOURCES,
                "noise_std": NOISE_STD,
                "amp_lo": AMP_LO,
                "amp_hi": AMP_HI,
                "peak_mult": PEAK_MULT,
                "min_peak_sep": MIN_PEAK_SEP,
                "n_units": n_units,
            },
            "modal_count": modal_count,
            "sesoi": SESOI,
            "per_unit": per_unit,
            "sign_flip": sign_flip,
            "verdict": verdict,
            "is_null": is_null,
            "alternative_explanation": (
                "any candidate advantage may reflect the synthesis choices (well-separated tone bank, this "
                "amplitude spread and noise level) rather than a general superiority of spectral peak "
                "counting for native-audio source counting; a different count mix could favor the modal guess"
            ),
            "failure_domain": (
                "mixtures with close or unequal-amplitude tones buried in noise: weak partials never lift a "
                "distinct peak above the floor, so the peak count collapses below the true source count "
                "while a loud dataset mode happens to match"
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
