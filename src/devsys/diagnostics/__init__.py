"""Diagnostics that gate the experiments: linear-probe distinctiveness, noisy-TV
epistemic/aleatoric guard, calibration, Fisher trace (critical-period signature), and the
determinism sanity loop."""

from __future__ import annotations

from .calibration import calibration_plot, reliability
from .determinism import assert_reproducible, determinism_loop
from .fisher_trace import critical_period_signature, fisher_trace, fisher_trace_over_training
from .linear_probe import linear_probe
from .noisy_tv import noisy_tv_diagnostic

__all__ = [
    "linear_probe",
    "noisy_tv_diagnostic",
    "reliability",
    "calibration_plot",
    "fisher_trace",
    "fisher_trace_over_training",
    "critical_period_signature",
    "determinism_loop",
    "assert_reproducible",
]
