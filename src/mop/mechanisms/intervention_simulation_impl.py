"""Deterministic, seeded implementations of the four intervention-simulation sub-mechanisms.

This module carries the runnable machinery for one ladder mechanism workstream (EPOCH
intervention_simulation). Every function here is a pure, deterministic computation over primitive
inputs: no wall clock, no unseeded randomness, no weights, no network, no natural data. The four
sub-mechanisms and their matched control policies are:

- a do-operator arm that predicts an interventional outcome versus an observational-only control
  that is fooled by confounding when confounding is present;
- a rollout planner that simulates a known model and plays the argmax action versus a random-action
  control at matched compute;
- a calibrated-uncertainty estimator that reports the Bayes probability versus an overconfident
  control that collapses every probability to zero or one;
- a reducible-novelty curiosity signal that spends its budget on learnable states versus a
  random-curiosity control and a count-based control that chases rare but unlearnable novelty.

Each arm is designed so that, on a world with no exploitable structure, it scores exactly what its
control scores (an exact tie), and on a world with genuine structure it strictly beats its control.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from collections.abc import Sequence


class InterventionSimulationImplError(ValueError):
    """Raised when a scoring input is malformed. The scorers fail closed rather than guess."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise InterventionSimulationImplError("cannot take the mean of an empty sequence")
    return sum(values, 0.0) / len(values)


def _mean_absolute_error(predictions: Sequence[float], truths: Sequence[float]) -> float:
    if len(predictions) != len(truths):
        raise InterventionSimulationImplError("prediction and truth sequences must match in length")
    if not predictions:
        raise InterventionSimulationImplError("cannot score an empty comparison")
    return _mean([abs(p - t) for p, t in zip(predictions, truths, strict=True)])


# ---------------------------------------------------------------------------
# Causal intervention. The estimand is the interventional outcome E[Y | do(X = x)]. Under an
# intervention the confounding path into X is severed, so the true outcome depends only on the
# causal effect. The do-operator arm computes exactly that; the observational arm reports the
# confounded association E[Y | X = x] = (causal_effect + confound_bias) * x.
# ---------------------------------------------------------------------------


def interventional_outcome(causal_effect: float, x: float) -> float:
    """The ground-truth outcome under do(X = x): confounding is severed, only the effect remains."""

    return causal_effect * x


def do_operator_prediction(causal_effect: float, x: float) -> float:
    """The do-operator arm predicts the interventional outcome directly."""

    return causal_effect * x


def observational_prediction(causal_effect: float, confound_bias: float, x: float) -> float:
    """The observational-only control reports the confounded association, not the causal effect."""

    return (causal_effect + confound_bias) * x


def do_operator_score(causal_effect: float, test_inputs: Sequence[float]) -> float:
    """Score the do-operator arm as one over one plus its mean absolute prediction error."""

    truths = [interventional_outcome(causal_effect, x) for x in test_inputs]
    predictions = [do_operator_prediction(causal_effect, x) for x in test_inputs]
    return 1.0 / (1.0 + _mean_absolute_error(predictions, truths))


def observational_score(
    causal_effect: float, confound_bias: float, test_inputs: Sequence[float]
) -> float:
    """Score the observational-only control. With no confounding it matches the do-operator arm."""

    truths = [interventional_outcome(causal_effect, x) for x in test_inputs]
    predictions = [observational_prediction(causal_effect, confound_bias, x) for x in test_inputs]
    return 1.0 / (1.0 + _mean_absolute_error(predictions, truths))


# ---------------------------------------------------------------------------
# Simulation for action. The reward table is reward_table[step][action]. The rollout planner
# simulates the known model and takes the argmax action each step, reaching the optimal return.
# The random-action control earns the uniform-policy expected return at matched compute.
# ---------------------------------------------------------------------------


def optimal_return(reward_table: Sequence[Sequence[float]]) -> float:
    """The best achievable return: the sum over steps of the maximum per-step reward."""

    return sum((max(row) for row in reward_table), 0.0)


def rollout_return(reward_table: Sequence[Sequence[float]]) -> float:
    """A planner that simulates the model and plays argmax each step realizes the optimal return."""

    return sum((max(row) for row in reward_table), 0.0)


def uniform_return(reward_table: Sequence[Sequence[float]]) -> float:
    """The random-action control earns the expected return of the uniform policy."""

    return sum((_mean(row) for row in reward_table), 0.0)


def rollout_score(reward_table: Sequence[Sequence[float]]) -> float:
    """Normalize the rollout return by the optimal return. Ties the control when nothing controls."""

    best = optimal_return(reward_table)
    if best <= 0.0:
        return 0.0
    return rollout_return(reward_table) / best


def random_action_score(reward_table: Sequence[Sequence[float]]) -> float:
    """Normalize the uniform-policy return by the optimal return, at matched compute."""

    best = optimal_return(reward_table)
    if best <= 0.0:
        return 0.0
    return uniform_return(reward_table) / best


# ---------------------------------------------------------------------------
# Calibrated uncertainty. Each item carries a Bayes probability p. The calibrated estimator reports
# p; the overconfident control collapses p to zero or one (a temperature toward zero). The score is
# one minus the reducible Brier term, the mean squared gap between the reported confidence and p.
# ---------------------------------------------------------------------------


def calibrated_confidence(bayes_probability: float) -> float:
    """A calibrated estimator reports the Bayes probability itself."""

    return bayes_probability


def overconfident_confidence(bayes_probability: float) -> float:
    """The overconfident control collapses the probability to a hard zero or one."""

    return 1.0 if bayes_probability >= 0.5 else 0.0


def _reducible_brier(confidences: Sequence[float], probs: Sequence[float]) -> float:
    if len(confidences) != len(probs):
        raise InterventionSimulationImplError("confidence and probability sequences must match")
    return _mean([(c - p) ** 2 for c, p in zip(confidences, probs, strict=True)])


def calibrated_score(bayes_probs: Sequence[float]) -> float:
    """Score the calibrated estimator. It reports p exactly, so its reducible Brier term is zero."""

    confidences = [calibrated_confidence(p) for p in bayes_probs]
    return 1.0 - _reducible_brier(confidences, bayes_probs)


def overconfident_score(bayes_probs: Sequence[float]) -> float:
    """Score the overconfident control. On determined items it ties; on uncertain items it loses."""

    confidences = [overconfident_confidence(p) for p in bayes_probs]
    return 1.0 - _reducible_brier(confidences, bayes_probs)


# ---------------------------------------------------------------------------
# Reducible-novelty curiosity. Each state carries a reducible-novelty value (the learning progress
# available there) and a count-novelty value (its rarity). The reducible-novelty signal spends its
# budget on the most learnable states; the count-based control chases the rarest states, which
# includes the unlearnable noisy state; the random-curiosity control spends uniformly.
# ---------------------------------------------------------------------------


def _best_obtainable(reducible_novelty: Sequence[float], budget: int) -> float:
    ranked = sorted(reducible_novelty, reverse=True)
    return sum(ranked[:budget], 0.0)


def reducible_novelty_score(reducible_novelty: Sequence[float], budget: int) -> float:
    """The reducible-novelty arm spends its budget on the most learnable states."""

    best = _best_obtainable(reducible_novelty, budget)
    if best <= 0.0:
        return 0.0
    order = sorted(range(len(reducible_novelty)), key=lambda i: reducible_novelty[i], reverse=True)
    obtained = sum((reducible_novelty[i] for i in order[:budget]), 0.0)
    return obtained / best


def count_based_score(
    reducible_novelty: Sequence[float], count_novelty: Sequence[float], budget: int
) -> float:
    """The count-based control chases the rarest states and can stall on unlearnable novelty."""

    if len(reducible_novelty) != len(count_novelty):
        raise InterventionSimulationImplError("novelty sequences must match in length")
    best = _best_obtainable(reducible_novelty, budget)
    if best <= 0.0:
        return 0.0
    order = sorted(range(len(count_novelty)), key=lambda i: count_novelty[i], reverse=True)
    obtained = sum((reducible_novelty[i] for i in order[:budget]), 0.0)
    return obtained / best


def random_curiosity_score(reducible_novelty: Sequence[float], budget: int) -> float:
    """The random-curiosity control spends its budget uniformly across states in expectation."""

    best = _best_obtainable(reducible_novelty, budget)
    if best <= 0.0:
        return 0.0
    count = len(reducible_novelty)
    if count == 0:
        return 0.0
    fraction = min(budget, count) / count
    expected = fraction * sum(reducible_novelty, 0.0)
    return expected / best
