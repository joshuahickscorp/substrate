"""Continual-learning metrics computed from the accuracy matrix R, where
R[i][j] = accuracy on task j after finishing training on task i (i,j in 0..T-1).

  avg_accuracy      = mean_j R[T-1][j]                      (final mean accuracy)
  backward_transfer = mean_{j<T-1} (R[T-1][j] - R[j][j])    (negative => forgetting)
  forward_transfer  = mean_{j>0}  (R[j-1][j] - chance)      (zero-shot help from prior tasks)
  adaptation_speed  = steps to reach a threshold on a new task (tracked during training)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ContinualResult:
    R: list[list[float]]  # accuracy matrix
    chance: float = 0.5
    adapt_steps: list[int] = field(default_factory=list)  # steps-to-threshold per task

    @property
    def T(self) -> int:
        return len(self.R)

    def avg_accuracy(self) -> float:
        return float(sum(self.R[-1]) / self.T)

    def backward_transfer(self) -> float:
        if self.T < 2:
            return 0.0
        return float(sum(self.R[-1][j] - self.R[j][j] for j in range(self.T - 1)) / (self.T - 1))

    def forward_transfer(self) -> float:
        if self.T < 2:
            return 0.0
        return float(sum(self.R[j - 1][j] - self.chance for j in range(1, self.T)) / (self.T - 1))

    def adaptation_speed(self) -> float:
        """Mean steps-to-threshold; lower is faster. inf-safe."""
        valid = [s for s in self.adapt_steps if s >= 0]
        return float(sum(valid) / len(valid)) if valid else float("inf")

    def summary(self) -> dict[str, float]:
        return {
            "avg_accuracy": self.avg_accuracy(),
            "backward_transfer": self.backward_transfer(),
            "forward_transfer": self.forward_transfer(),
            "adaptation_speed": self.adaptation_speed(),
            "final_first_task_acc": float(self.R[-1][0]),
            "first_task_peak_acc": float(self.R[0][0]),
        }


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())
