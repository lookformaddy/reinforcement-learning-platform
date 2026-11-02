"""Small deterministic gridworld: the tabular-RL test bed.

States are ``(row, col)`` tuples. Actions: 0=up, 1=down, 2=left, 3=right.
Stepping into a wall or obstacle leaves the agent in place with a wall
penalty; reaching the goal ends the episode with a goal reward; every other
step costs a small step penalty, which makes shortest paths optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ACTIONS = ("up", "down", "left", "right")
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    obstacles: frozenset[tuple[int, int]] = frozenset({(2, 2), (1, 3)})
    step_reward: float = -0.1
    goal_reward: float = 10.0
    wall_reward: float = -0.5
    max_steps: int = 100

    def __post_init__(self) -> None:
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("start/goal must not be obstacles")
        if not (0 <= self.start[0] < self.height and 0 <= self.start[1] < self.width):
            raise ValueError("start out of bounds")
        if not (0 <= self.goal[0] < self.height and 0 <= self.goal[1] < self.width):
            raise ValueError("goal out of bounds")

    def states(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.obstacles
        ]

    def reset(self) -> tuple[int, int]:
        return self.start

    def step(
        self, state: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], float, bool]:
        """Return ``(next_state, reward, done)``. Fully deterministic."""
        if not 0 <= action < 4:
            raise ValueError(f"action {action} out of range")
        if state == self.goal:
            return state, 0.0, True
        dr, dc = _DELTAS[action]
        nxt = (state[0] + dr, state[1] + dc)
        if (
            not (0 <= nxt[0] < self.height and 0 <= nxt[1] < self.width)
            or nxt in self.obstacles
        ):
            return state, self.wall_reward, False
        if nxt == self.goal:
            return nxt, self.goal_reward, True
        return nxt, self.step_reward, False

    def transitions(
        self, state: tuple[int, int], action: int
    ) -> list[tuple[tuple[int, int], float, bool]]:
        """Deterministic dynamics as a one-element distribution.

        Policy evaluation/improvement consume this interface, so the same
        code works unchanged if the environment later becomes stochastic.
        """
        nxt, reward, done = self.step(state, action)
        return [(nxt, reward, done)]
