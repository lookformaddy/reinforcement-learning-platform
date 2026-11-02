"""Tabular agents: Q-learning (off-policy) and SARSA (on-policy).

Both share an epsilon-decay schedule interface: ``epsilon`` starts at
``epsilon_start`` and decays geometrically toward ``epsilon_min`` per
episode, so early episodes explore and late episodes exploit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rl_platform.gridworld import GridWorld
from rl_platform.rng import SeededRNG

State = tuple[int, int]


def epsilon_schedule(episode: int, start: float = 1.0, decay: float = 0.995, minimum: float = 0.05) -> float:
    """Geometric decay, floored at ``minimum``. Episode 0 -> ``start``."""
    return max(minimum, start * (decay**episode))


class _TabularAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q: dict[State, list[float]] = {}

    def _q(self, state: State) -> list[float]:
        return self.q.setdefault(state, [0.0] * 4)

    def greedy_action(self, state: State) -> int:
        values = self._q(state)
        return values.index(max(values))

    def select_action(self, state: State, episode: int, rng: SeededRNG) -> int:
        eps = epsilon_schedule(episode, self.epsilon_start, self.epsilon_decay, self.epsilon_min)
        if rng.random() < eps:
            return rng.randint(0, 3)
        return self.greedy_action(state)

    def to_json(self) -> dict:
        """Q-table serialisation: keys are ``"row,col"`` strings."""
        return {f"{r},{c}": values for (r, c), values in sorted(self.q.items())}

    @classmethod
    def from_json(cls, data: dict, **kwargs) -> "_TabularAgent":
        agent = cls(**kwargs)
        agent.q = {
            tuple(int(x) for x in key.split(",")): list(values)  # type: ignore[arg-type]
            for key, values in data.items()
        }
        return agent


class QLearningAgent(_TabularAgent):
    """Off-policy TD: bootstrap from max Q of the next state."""

    def update(
        self, state: State, action: int, reward: float, next_state: State, done: bool
    ) -> None:
        q = self._q(state)
        target = reward if done else reward + self.gamma * max(self._q(next_state))
        q[action] += self.alpha * (target - q[action])


class SARSAAgent(_TabularAgent):
    """On-policy TD: bootstrap from the Q of the action actually taken next."""

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int | None,
        done: bool,
    ) -> None:
        q = self._q(state)
        if done or next_action is None:
            target = reward
        else:
            target = reward + self.gamma * self._q(next_state)[next_action]
        q[action] += self.alpha * (target - q[action])


@dataclass
class TabularRun:
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0


def train_tabular(env: GridWorld, agent, episodes: int, rng: SeededRNG) -> TabularRun:
    """Train a tabular agent; SARSA and Q-learning differ only in update."""
    run = TabularRun()
    is_sarsa = isinstance(agent, SARSAAgent)
    for episode in range(episodes):
        state = env.reset()
        action = agent.select_action(state, episode, rng)
        total, steps, done = 0.0, 0, False
        while not done and steps < env.max_steps:
            next_state, reward, done = env.step(state, action)
            total += reward
            steps += 1
            if is_sarsa:
                next_action = None if done else agent.select_action(next_state, episode, rng)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action  # type: ignore[assignment]
            else:
                agent.update(state, action, reward, next_state, done)
                if not done:
                    action = agent.select_action(next_state, episode, rng)
            state = next_state
        run.episode_rewards.append(total)
        run.episode_lengths.append(steps)
    return run
