
from __future__ import annotations

from collections.abc import Sequence


class InterventionSimulationImplError(ValueError):
    pass


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




def interventional_outcome(causal_effect: float, x: float) -> float:

    return causal_effect * x


def do_operator_prediction(causal_effect: float, x: float) -> float:

    return causal_effect * x


def observational_prediction(causal_effect: float, confound_bias: float, x: float) -> float:

    return (causal_effect + confound_bias) * x


def do_operator_score(causal_effect: float, test_inputs: Sequence[float]) -> float:

    truths = [interventional_outcome(causal_effect, x) for x in test_inputs]
    predictions = [do_operator_prediction(causal_effect, x) for x in test_inputs]
    return 1.0 / (1.0 + _mean_absolute_error(predictions, truths))


def observational_score(
    causal_effect: float, confound_bias: float, test_inputs: Sequence[float]
) -> float:

    truths = [interventional_outcome(causal_effect, x) for x in test_inputs]
    predictions = [observational_prediction(causal_effect, confound_bias, x) for x in test_inputs]
    return 1.0 / (1.0 + _mean_absolute_error(predictions, truths))




def optimal_return(reward_table: Sequence[Sequence[float]]) -> float:

    return sum((max(row) for row in reward_table), 0.0)


def rollout_return(reward_table: Sequence[Sequence[float]]) -> float:

    return sum((max(row) for row in reward_table), 0.0)


def uniform_return(reward_table: Sequence[Sequence[float]]) -> float:

    return sum((_mean(row) for row in reward_table), 0.0)


def rollout_score(reward_table: Sequence[Sequence[float]]) -> float:

    best = optimal_return(reward_table)
    if best <= 0.0:
        return 0.0
    return rollout_return(reward_table) / best


def random_action_score(reward_table: Sequence[Sequence[float]]) -> float:

    best = optimal_return(reward_table)
    if best <= 0.0:
        return 0.0
    return uniform_return(reward_table) / best




def calibrated_confidence(bayes_probability: float) -> float:

    return bayes_probability


def overconfident_confidence(bayes_probability: float) -> float:

    return 1.0 if bayes_probability >= 0.5 else 0.0


def _reducible_brier(confidences: Sequence[float], probs: Sequence[float]) -> float:
    if len(confidences) != len(probs):
        raise InterventionSimulationImplError("confidence and probability sequences must match")
    return _mean([(c - p) ** 2 for c, p in zip(confidences, probs, strict=True)])


def calibrated_score(bayes_probs: Sequence[float]) -> float:

    confidences = [calibrated_confidence(p) for p in bayes_probs]
    return 1.0 - _reducible_brier(confidences, bayes_probs)


def overconfident_score(bayes_probs: Sequence[float]) -> float:

    confidences = [overconfident_confidence(p) for p in bayes_probs]
    return 1.0 - _reducible_brier(confidences, bayes_probs)




def _best_obtainable(reducible_novelty: Sequence[float], budget: int) -> float:
    ranked = sorted(reducible_novelty, reverse=True)
    return sum(ranked[:budget], 0.0)


def reducible_novelty_score(reducible_novelty: Sequence[float], budget: int) -> float:

    best = _best_obtainable(reducible_novelty, budget)
    if best <= 0.0:
        return 0.0
    order = sorted(range(len(reducible_novelty)), key=lambda i: reducible_novelty[i], reverse=True)
    obtained = sum((reducible_novelty[i] for i in order[:budget]), 0.0)
    return obtained / best


def count_based_score(
    reducible_novelty: Sequence[float], count_novelty: Sequence[float], budget: int
) -> float:

    if len(reducible_novelty) != len(count_novelty):
        raise InterventionSimulationImplError("novelty sequences must match in length")
    best = _best_obtainable(reducible_novelty, budget)
    if best <= 0.0:
        return 0.0
    order = sorted(range(len(count_novelty)), key=lambda i: count_novelty[i], reverse=True)
    obtained = sum((reducible_novelty[i] for i in order[:budget]), 0.0)
    return obtained / best


def random_curiosity_score(reducible_novelty: Sequence[float], budget: int) -> float:

    best = _best_obtainable(reducible_novelty, budget)
    if best <= 0.0:
        return 0.0
    count = len(reducible_novelty)
    if count == 0:
        return 0.0
    fraction = min(budget, count) / count
    expected = fraction * sum(reducible_novelty, 0.0)
    return expected / best
