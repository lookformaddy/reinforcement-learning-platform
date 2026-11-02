"""Deterministic seeded RNG: one wrapper so every experiment is reproducible
from a single integer seed, with no hidden global-state coupling."""

from __future__ import annotations

import random


class SeededRNG:
    """Thin, explicit wrapper over :class:`random.Random`.

    All stochastic decisions in the platform go through an instance of this
    class, so re-running with the same seed reproduces the same trajectory.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, low: int, high: int) -> int:
        """Inclusive ``low``..``high``."""
        return self._rng.randint(low, high)

    def choice(self, seq):
        return self._rng.choice(seq)

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._rng.gauss(mu, sigma)

    def betavariate(self, alpha: float, beta: float) -> float:
        return self._rng.betavariate(alpha, beta)

    def bernoulli(self, p: float) -> int:
        return 1 if self._rng.random() < p else 0
