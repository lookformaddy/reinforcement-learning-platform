"""Multi-armed bandits, from scratch.

Environment: Bernoulli arms with fixed (synthetic) success probabilities.
Agents: epsilon-greedy, UCB1, Thompson sampling (Beta-Bernoulli posterior).
``run_bandit`` tracks per-step reward and *cumulative regret* against the
optimal arm, the metric that matters for bandit evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rl_platform.rng import SeededRNG


@dataclass
class BernoulliBandit:
    """K Bernoulli arms. ``arm_probs`` are synthetic, never real data."""

    arm_probs: list[float]

    def __post_init__(self) -> None:
        if not self.arm_probs:
            raise ValueError("need at least one arm")
        if any(not 0.0 <= p <= 1.0 for p in self.arm_probs):
            raise ValueError("arm probabilities must be in [0, 1]")

    @property
    def n_arms(self) -> int:
        return len(self.arm_probs)

    @property
    def optimal_mean(self) -> float:
        return max(self.arm_probs)

    def pull(self, arm: int, rng: SeededRNG) -> int:
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"arm {arm} out of range")
        return rng.bernoulli(self.arm_probs[arm])


class _BaseBandit:
    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.counts: list[int] = [0] * n_arms
        self.values: list[float] = [0.0] * n_arms

    def _update_mean(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental mean: numerically stable, O(1) memory.
        self.values[arm] += (reward - self.values[arm]) / n

    def select_action(self, rng: SeededRNG) -> int:  # pragma: no cover
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self._update_mean(arm, reward)


class EpsilonGreedyBandit(_BaseBandit):
    """Explore with prob. ``epsilon``, otherwise exploit the best mean."""

    def __init__(self, n_arms: int, epsilon: float = 0.1) -> None:
        super().__init__(n_arms)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon

    def select_action(self, rng: SeededRNG) -> int:
        if rng.random() < self.epsilon:
            return rng.randint(0, self.n_arms - 1)
        best = max(self.values)
        # Break ties by first index: deterministic given identical values.
        return self.values.index(best)


class UCBBandit(_BaseBandit):
    """UCB1: untried arms first, then mean + c * sqrt(ln t / n_a)."""

    def __init__(self, n_arms: int, c: float = 2.0) -> None:
        super().__init__(n_arms)
        self.c = c
        self.t = 0

    def select_action(self, rng: SeededRNG) -> int:  # noqa: ARG002 - API parity
        for arm, count in enumerate(self.counts):
            if count == 0:
                return arm
        total = max(self.t, 1)
        bonuses = [
            self.values[a] + self.c * math.sqrt(math.log(total) / self.counts[a])
            for a in range(self.n_arms)
        ]
        return bonuses.index(max(bonuses))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.t += 1


class ThompsonSamplingBandit(_BaseBandit):
    """Beta-Bernoulli Thompson sampling: sample a posterior draw per arm."""

    def __init__(self, n_arms: int, alpha0: float = 1.0, beta0: float = 1.0) -> None:
        super().__init__(n_arms)
        self.alpha: list[float] = [alpha0] * n_arms
        self.beta: list[float] = [beta0] * n_arms

    def select_action(self, rng: SeededRNG) -> int:
        draws = [rng.betavariate(self.alpha[a], self.beta[a]) for a in range(self.n_arms)]
        return draws.index(max(draws))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


@dataclass
class BanditRun:
    rewards: list[float] = field(default_factory=list)
    cumulative_regret: list[float] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def final_regret(self) -> float:
        return self.cumulative_regret[-1] if self.cumulative_regret else 0.0


def run_bandit(env: BernoulliBandit, agent: _BaseBandit, steps: int, rng: SeededRNG) -> BanditRun:
    """Run one bandit agent; regret at step t is optimal_mean - reward_t."""
    run = BanditRun()
    regret = 0.0
    for _ in range(steps):
        arm = agent.select_action(rng)
        reward = float(env.pull(arm, rng))
        agent.update(arm, reward)
        run.rewards.append(reward)
        regret += env.optimal_mean - reward
        run.cumulative_regret.append(regret)
    return run
