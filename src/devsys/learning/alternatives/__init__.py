"""Backprop alternatives for I4 / E9. Each rule trains the SAME small head on the SAME
latents and reports accuracy vs backprop, locality (weight transport? separate backward
pass?), compute cost, and stability. The expected result is that all underperform backprop
by a stated margin while some win on locality/memory: that gap IS the finding.

Each trainer has the signature train_*(x, y, hidden, epochs, lr, seed) -> RuleResult.
"""

from __future__ import annotations

from .rules import (
    RuleResult,
    train_backprop,
    train_dfa,
    train_equilibrium_prop,
    train_feedback_alignment,
    train_forward_forward,
    train_predictive_coding,
    train_target_prop,
)

RULES = {
    "backprop": train_backprop,
    "feedback_alignment": train_feedback_alignment,
    "dfa": train_dfa,
    "forward_forward": train_forward_forward,
    "target_prop": train_target_prop,
    "equilibrium_prop": train_equilibrium_prop,
    "predictive_coding": train_predictive_coding,
}

__all__ = ["RULES", "RuleResult"] + [
    f"train_{k}"
    for k in (
        "backprop",
        "feedback_alignment",
        "dfa",
        "forward_forward",
        "target_prop",
        "equilibrium_prop",
        "predictive_coding",
    )
]
