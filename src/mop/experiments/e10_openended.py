"""E10: minimal open-ended JEPA (the program's capstone). SCAFFOLD ONLY, marked env-later.
The bounded persistent action contract now runs locally. The real version still needs a
parameterizable environment generator, population search, transfer, and a sustained horizon;
here we stand up the spine of that
experiment at toy scale so the contract, the metrics, and the null check are exercised in
seconds. The corpus is explicit (Experiment 10, vol1): open-endedness is most likely a
population-and-environment property, and a single agent on a fixed environment plateaus. We
encode that as the null and let the ablation ladder report which components, if any, move it.

The pieces, all minimal stubs:
  StubEnv          a compositional procedural environment returning latent-like observations.
  SkillArchive     a MAP-Elites style dict keyed by a discretized latent behavior signature.
  GoalGenerator    samples latent goals (autotelic curriculum stub).
The ablation ladder is the spine: [curiosity_only, +memory, +plasticity, +goal_generation].
Each arm runs a short toy loop and reports distinct skills, skill reuse, archive diversity,
and a collapse detector. NULL: single_agent_plateaus (diversity stops growing across arms).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ..devices import DeviceInfo  # noqa: E402
from ..environments import bounded_trajectory_contract  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from .base import Experiment  # noqa: E402

ARMS = ("curiosity_only", "+memory", "+plasticity", "+goal_generation")


class StubEnv:
    """Tiny compositional procedural environment. State is a latent vector; an action mixes a
    few latent "primitives" so behaviors compose (stepping stones exist in principle). The
    resulting trajectory latent is what gets characterized into a behavior signature. This is a
    placeholder for the real V-JEPA-encoded environment, hence tier env-later."""

    def __init__(self, dim: int, n_primitives: int, rng: np.random.Generator):
        self.dim = dim
        self.primitives = rng.standard_normal((n_primitives, dim)).astype(np.float32)
        self.rng = rng
        self.state = np.zeros(dim, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.state = np.zeros(self.dim, dtype=np.float32)
        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
        # action is a mixing weight over primitives; latent composition + mild noise
        mix = action @ self.primitives
        self.state = np.tanh(self.state * 0.5 + mix) + 0.01 * self.rng.standard_normal(self.dim).astype(
            np.float32
        )
        return self.state

    def rollout(self, action: np.ndarray, horizon: int) -> np.ndarray:
        self.reset()
        last = self.state
        for _ in range(horizon):
            last = self.step(action)
        return last  # the trajectory-end latent, characterized downstream


class SkillArchive:
    """MAP-Elites style archive: a dict keyed by a discretized latent behavior signature. Each
    cell holds the best (highest novelty/fitness) skill found for that niche. Diversity is the
    number of filled cells; reuse is how often a new behavior lands in an already-filled cell."""

    def __init__(self, bins: int):
        self.bins = bins
        self.cells: dict[tuple, dict] = {}
        self.inserts = 0
        self.reuse_hits = 0

    def signature(self, latent: np.ndarray, proj: np.ndarray) -> tuple:
        z = np.tanh(latent @ proj)  # project to a low-dim behavior descriptor
        return tuple(np.clip(((z + 1.0) * 0.5 * self.bins).astype(int), 0, self.bins - 1).tolist())

    def insert(self, sig: tuple, action: np.ndarray, fitness: float) -> bool:
        self.inserts += 1
        cur = self.cells.get(sig)
        if cur is None:
            self.cells[sig] = {"action": action, "fitness": fitness}
            return True  # new niche discovered
        self.reuse_hits += 1  # behavior reused an existing niche
        if fitness > cur["fitness"]:
            self.cells[sig] = {"action": action, "fitness": fitness}
        return False

    @property
    def distinct(self) -> int:
        return len(self.cells)

    @property
    def reuse_rate(self) -> float:
        return self.reuse_hits / max(self.inserts, 1)


class GoalGenerator:
    """Samples latent goals (autotelic curriculum stub). Without it, actions are exploratory
    only; with it, the agent is biased toward unfilled regions of behavior space."""

    def __init__(self, dim: int, rng: np.random.Generator):
        self.dim = dim
        self.rng = rng

    def sample(self, n: int) -> np.ndarray:
        return self.rng.standard_normal((n, self.dim)).astype(np.float32)


class E10(Experiment):
    id = "e10_openended"
    metric = ("distinct_skills", "skill_reuse", "archive_diversity", "collapse")
    baseline = "fixed-task agent (no goal generation) and a random-goal agent"
    ablation = "ablation ladder: curiosity_only, +memory, +plasticity, +goal_generation"
    null_hypothesis = (
        "the single agent plateaus: archive diversity stops growing across the ablation ladder, "
        "so open-endedness needs population-level search and environment generation, not just a "
        "richer single agent"
    )
    tier = "env-later"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seed_everything(int(cfg.seed))
        rng = np.random.default_rng(int(cfg.seed))
        dim = int(e.dim)
        n_primitives = int(e.n_primitives)
        n_actions = int(e.n_actions)
        horizon = int(e.horizon)
        bins = int(e.bins)
        descriptor_dim = int(e.descriptor_dim)
        plateau_eps = float(e.plateau_eps)
        proj = rng.standard_normal((dim, descriptor_dim)).astype(np.float32)

        arms: dict[str, dict] = {}
        for arm in ARMS:
            arms[arm] = self._run_arm(arm, dim, n_primitives, n_actions, horizon, bins, proj, rng)

        # null check: as components are stacked, does the fullest agent keep expanding behavior
        # diversity, or does it plateau? The corpus expects the single agent to plateau (open-
        # endedness is a population/environment property). We answer it cleanly: plateau if the
        # final, fullest rung does not improve on the best earlier rung by more than plateau_eps
        # (relative). A flat-or-declining tail is exactly the predicted single-agent ceiling.
        diversity = [arms[a]["archive_diversity"] for a in ARMS]
        best_prior = max(diversity[:-1])
        last_gain = diversity[-1] - diversity[-2]  # +goal_generation over +plasticity
        best_gain = max(diversity[i + 1] - diversity[i] for i in range(len(diversity) - 1))
        rel_over_best = (diversity[-1] - best_prior) / max(best_prior, 1)
        single_agent_plateaus = bool(rel_over_best <= plateau_eps)  # fullest arm did not pull ahead
        any_collapsed = any(arms[a]["collapse"] for a in ARMS)

        local_action_contract = bounded_trajectory_contract(
            seed=int(cfg.seed), episodes=4, horizon=max(6, min(horizon * 2, 12))
        )
        out = {
            "arms": arms,
            "per_arm_diversity": {a: arms[a]["archive_diversity"] for a in ARMS},
            "diversity_ladder": diversity,
            "last_rung_gain": last_gain,
            "best_rung_gain": best_gain,
            "best_prior_diversity": best_prior,
            "rel_gain_over_best": rel_over_best,
            "plateau_eps": plateau_eps,
            "single_agent_plateaus": single_agent_plateaus,  # the explicit null answer
            "any_arm_collapsed": any_collapsed,
            "tier": self.tier,
            "scaffold_only": True,  # env-later: not run at scale, stub environment
            "local_action_environment_available": local_action_contract["verified"],
            "local_action_environment_contract": local_action_contract,
            "remaining_scientific_blockers": [
                "population-level search",
                "environment generation",
                "sustained non-plateau horizon",
                "cross-environment transfer",
            ],
        }
        self._plot(diversity, arms, run_dir)
        return out

    def _run_arm(self, arm, dim, n_primitives, n_actions, horizon, bins, proj, rng) -> dict:
        """A short toy loop for one rung of the ladder. Components stack along ARMS order."""
        use_memory = arm in ("+memory", "+plasticity", "+goal_generation")
        use_plasticity = arm in ("+plasticity", "+goal_generation")
        use_goals = arm == "+goal_generation"

        env = StubEnv(dim, n_primitives, rng)
        archive = SkillArchive(bins)
        goals = GoalGenerator(n_primitives, rng) if use_goals else None
        memory: list[np.ndarray] = []  # replayed actions when memory is on
        explore = 1.0  # plasticity controller shrinks exploration over time when on
        behavior_hist: list[tuple] = []

        for _t in range(n_actions):
            if use_goals and goals is not None:
                action = goals.sample(1)[0]  # autotelic: aim at a sampled latent goal direction
            elif use_memory and memory and rng.random() < 0.3:
                action = memory[rng.integers(len(memory))] + 0.2 * rng.standard_normal(n_primitives)
            else:
                action = rng.standard_normal(n_primitives)
            action = (explore * action).astype(np.float32)

            latent = env.rollout(action, horizon)
            sig = archive.signature(latent, proj)
            novelty = self._novelty(sig, behavior_hist)
            fitness = float(novelty - 0.01 * float(np.linalg.norm(action)))
            is_new = archive.insert(sig, action, fitness)
            behavior_hist.append(sig)
            if use_memory and is_new:
                memory.append(action)  # store skills that opened new niches, for reuse
            if use_plasticity:
                explore = max(0.3, explore * 0.985)  # anneal exploration: stabilize discovered skills

        collapse = self._collapse(behavior_hist)
        return {
            "distinct_skills": archive.distinct,
            "skill_reuse": archive.reuse_rate,
            "archive_diversity": archive.distinct,
            "collapse": collapse,
            "components": {"memory": use_memory, "plasticity": use_plasticity, "goals": use_goals},
            "steps": n_actions,
        }

    @staticmethod
    def _novelty(sig: tuple, hist: list[tuple]) -> float:
        if not hist:
            return 1.0
        recent = hist[-50:]
        s = np.array(sig, dtype=np.float32)
        dists = [float(np.linalg.norm(s - np.array(h, dtype=np.float32))) for h in recent]
        return float(np.mean(dists))

    @staticmethod
    def _collapse(hist: list[tuple]) -> bool:
        """Collapse detector: did behavior converge to one strategy? True if the last third of
        the run is dominated by a single behavior signature."""
        if len(hist) < 6:
            return False
        tail = hist[len(hist) * 2 // 3 :]
        top = max(tail.count(h) for h in set(tail))
        return top / len(tail) > 0.6

    def _plot(self, diversity, arms, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(range(len(ARMS)), diversity, marker="o")
        ax1.set_xticks(range(len(ARMS)))
        ax1.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=7)
        ax1.set_ylabel("archive diversity (distinct skills)")
        ax1.set_title("E10 ablation ladder: diversity per arm")
        reuse = [arms[a]["skill_reuse"] for a in ARMS]
        ax2.bar(range(len(ARMS)), reuse)
        ax2.set_xticks(range(len(ARMS)))
        ax2.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=7)
        ax2.set_ylabel("skill reuse rate")
        ax2.set_title("E10: skill reuse per arm")
        fig.tight_layout()
        fig.savefig(run_dir / "e10_ladder.png", dpi=110)
        plt.close(fig)
